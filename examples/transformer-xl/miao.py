class FMoE(nn.Module):
    r"""
    A general moe implementation that supports an arbitrary module as the
    expert.
    * `num_expert` stands for the number of experts on **each** worker.
    * `world_size` stands for the total number of workers that contains
    different experts.
    * `slice_group` can be a torch's communication group, indicating that
    specific model parallel is applied across the group, and workers in the
    group hold the same copy of input feature, and requires the same copy of
    the output. For each worker, FMoE only computes the output of a certain
    slice of the input batch, and will all-gather the outputs after
    computation.
    * `top_k` stands for the number of experts each token is going to.
    * `gate` is a gate class which can found in `fmoe.gates`.
    * `expert` can be specified as a module class, it is used to generate
    `num_expert` expert modules.
    """

    """
    通用的MoE实现, 支持任意的模块作为专家
    num_expert: 代表每个worker对应的expert数量
    world_size: 表示worker的总数, 即机器的数量
    slice_group: torch的通信组, 表示特定的模型并行应用于整个组，
    组内的工作者持有相同的输入特征副本，并需要相同的输出副本
    对于每个工作者, FMoE只计算input batch的某一片的输出
    并将计算后的输出全部收集起来
    top_k: 代表每个token要对应的专家的数量
    gate: 门类, 可以在`fmoe.gates`中找到,包含switch, swipe, gshard等gates
    expert: 模块类, 用于生成num_expert模块
    
    """

    def __init__(
        self,
        num_expert=32, 
        d_model=1024,
        world_size=1,
        mp_group=None,  # being deprecated 被弃用
        slice_group=None,
        moe_group=None,
        top_k=2,
        gate=NaiveGate, # 选择对应的gate
        expert=None,
        gate_hook=None,
        mask=None,
        mask_dict=None,
    ):
        super().__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        # world_size进程, 实际就是机器的个数, 例如两台机器一起训练的话, world_size就设置为2
        self.world_size = world_size

        self.slice_group = slice_group
        # mp_group = True，group被弃用，弹出警告
        if mp_group is not None:
            print("[Warning] mp_group is being deprecated")
            self.slice_group = mp_group
        # 没有slice_group
        if self.slice_group is None:
            self.slice_size = 1
            # rank: 区分主节点和从节点的, 主节点为0, 剩余的为了1-(N-1), N = world_size
            # 没有slice_group，只有一个主节点，设slice_rank = 0
            self.slice_rank = 0
        else:
            self.slice_size = self.slice_group.size()
            self.slice_rank = self.slice_group.rank()

        self.top_k = top_k
        # expert初始化
        if type(expert) is list:
            self.experts = nn.ModuleList([e(d_model) for e in expert])        
            self.experts_fused = False
            self.num_expert = num_expert = len(expert)
        elif expert is not None:
            self.experts = nn.ModuleList([expert(d_model) for _ in range(num_expert)])
            self.experts_fused = False
        else:
            self.experts_fused = True

        self.gate = gate(d_model, num_expert, world_size, top_k)
        self.gate_hook = gate_hook
        self.mask = mask
        self.mask_dict = mask_dict
        self.moe_group = moe_group

    def expert_fn(self, inp, fwd_expert_count):
        r"""
        The default expert function which either calls the experts as a whole
        or as separate experts.

        可以将expert当作整体调用或者调用单个专家
        """
        #将expert当作整体调用
        if self.experts_fused:
            return self.experts(inp, fwd_expert_count)
        # 如果fwd_expert_count是CUDA tensor的形式，需要改成numpy的数据格式
        # 
        if isinstance(fwd_expert_count, torch.Tensor):
            fwd_expert_count = fwd_expert_count.cpu().numpy() 
            # CUDA tensor格式的数据改成numpy
            # numpy不能读取CUDA tensor 需要将它转化为 CPU tensor
        outputs = []
        base_idx = 0
        for i in range(self.num_expert):
            # fwd_expert_count存储每个expert对应的batch_size
            batch_size = fwd_expert_count[i]
            # input按照batchsize选取
            inp_slice = inp[base_idx : base_idx + batch_size]
            # 得到output
            outputs.append(self.experts[i](inp_slice))
            # 下一个batch
            base_idx += batch_size
        # 将每个outputs拼接
        return torch.cat(outputs, dim=0)

    def mark_parallel_comm(self, expert_dp_comm="none"):  
        r"""
        Automatically mark the data parallel comms of the parameters within the
        module. This can be typically called at the end of the __init__ function
        in child classes.
        自动标记模块内的参数的数据并行通信。
        这通常可以在子类中的__init__函数的末尾调用。
        """
        if self.experts is not None:
            comm = expert_dp_comm
            # 判断experts是否是list
            if isinstance(self.experts, list):
                for e in self.experts:
                    mark_module_parallel_comm(e, comm)
            else:
                mark_module_parallel_comm(self.experts, comm)
        mark_module_parallel_comm(self.gate, "gate")

    def forward(self, moe_inp):
        r"""
        The FMoE module first computes gate output, and then conduct MoE forward
        according to the gate.  The score of the selected gate given by the
        expert is multiplied to the experts' output tensors as a weight.

        FMoE module首先计算gate output,然后根据门的情况计算MoE forward。 
        专家给出的所选门的分数被乘以专家的输出张量作为权重。
        """

        # MoE input 的 batch_size
        moe_inp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
        )
        assert all(
            [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
        ), "MoE inputs must have the same batch size"

        # 显卡数量大于1
        if self.world_size > 1:
            
            def ensure_comm_func(tensor):
                # 判断多显卡之间是否可以进行交互
                ensure_comm(tensor, self.moe_group)
            # 判断MoE_input是否在多机器间交互
            tree.map_structure(ensure_comm_func, moe_inp)
        
        if self.slice_size > 1:

            def slice_func(tensor):
                # 通过slice_func 得到一个Slice类
                return Slice.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_inp = tree.map_structure(slice_func, moe_inp)

        # 计算gate的输出
        # gate的index和score
        gate_top_k_idx, gate_score = self.gate(moe_inp)
        # gate_hook is True
        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)

        # delete masked tensors
        if self.mask is not None and self.mask_dict is not None:
            # TODO: to fix
            def delete_mask_func(tensor):
                # to: (BxL') x d_model
                tensor = tensor[mask == 0, :]
                return tensor

            mask = self.mask.view(-1)
            moe_inp = tree.map_structure(delete_mask_func, moe_inp)
            gate_top_k_idx = gate_top_k_idx[mask == 0, :]

        # 计算 MoE forward
        # the output features of experts
        fwd = _fmoe_general_global_forward(
            moe_inp, gate_top_k_idx, self.expert_fn,
            self.num_expert, self.world_size,
            experts=self.experts
        )

        # recover deleted tensors
        if self.mask is not None and self.mask_dict is not None:

            def recover_func(tensor):
                # to: (BxL') x top_k x dim
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                # to: (BxL) x top_k x d_model
                x = torch.zeros(
                    mask.shape[0],
                    self.top_k,
                    dim,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                # recover
                x[mask == 0] = tensor
                for k, v in self.mask_dict.items():
                    x[mask == k] = v
                return x

            moe_outp = tree.map_structure(recover_func, fwd)
        else:

            def view_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                # view中一个参数定为-1，代表动态调整这个维度上的元素个数，以保证元素的总数不变
                return tensor

            moe_outp = tree.map_structure(view_func, fwd)

        gate_score = gate_score.view(-1, 1, self.top_k)

        def bmm_func(tensor):
            dim = tensor.shape[-1] # shape[-1]表示列数
            # 按照列数dim, reshape
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor

        moe_outp = tree.map_structure(bmm_func, moe_outp)

        if self.slice_size > 1:

            def all_gather_func(tensor):
                return AllGather.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_outp = tree.map_structure(all_gather_func, moe_outp)

        moe_outp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_outp)
        )
        assert all(
            [batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]
        ), "MoE outputs must have the same batch size"
        return moe_outp
        
class FMoETransformerMLP(FMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    一个完整的MoE MLP模块在transformer中。
    * `activation`是每个专家在MLP中使用的激活函数。
    * `d_hidden`是MLP层的dimension。
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        **kwargs
    ):
        super().__init__(num_expert=num_expert, d_model=d_model, **kwargs)
        # 初始化_Expert类
        self.experts = _Expert(
            num_expert, d_model, d_hidden, activation, rank=expert_rank
        )
        # 标记模块内参数并行
        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        这个模块用reshape、残差和 layer
        normalization 来包装FMoE模块。
        """
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        output = super().forward(inp)
        return output.reshape(original_shape)


class CustomizedMoEPositionwiseFF(FMoETransformerMLP):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False, moe_num_expert=64, moe_top_k=1):
        activation = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # 在子类调用父类的函数用super()
        super().__init__(num_expert=moe_num_expert, d_model=d_model, d_hidden=d_inner, top_k=moe_top_k,
                activation=activation)

        self.pre_lnorm = pre_lnorm
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = super().forward(self.layer_norm(inp))
            core_out = self.dropout(core_out)

            ##### residual connection
            output = core_out 
        else:
            ##### positionwise feed-forward
            core_out = super().forward(inp)
            core_out = self.dropout(core_out)

            ##### residual connection + layer normalization
            output = self.layer_norm(core_out)

        return output

class Miao():
    workers = []

    def _init_(self, d_model, d_inner, dropout, pre_lnorm=False):

        for i in range([1,2,4,8]):
            self.workers.append(CustomizedMoEPositionwiseFF(self, d_model, d_inner, dropout, pre_lnorm,moe_num_expert = i, moe_top_k = 1))
    
    def forward(self, inp):
        output = []
        for worker in self.workers:
            output += worker.forward(inp)
        return output