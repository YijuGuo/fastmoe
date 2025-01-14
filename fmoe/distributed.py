r"""
Supportive modules to conduct distributed training
"""
import torch
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from .utils import get_torch_default_comm, get_rank_0_in_comm


class DistributedGroupedDataParallel(nn.Module):
    r"""
    A customized DDP module to support different all-reduce regions in the
    model.  The all-reduce region is defined as an attribution `dp_comm` in the
    weight object.
    一个定制的DDP模块,支持模型中不同的all-reduce regions。
    all-reduce regions被定义为权重对象中的一个属性`dp_comm'。
    AllReduce(全规约)：对所有服务器上的数据做一个规约操作（如最大值、求和），再将数据写入根服务器
    

    The grads of the weights are identified to be reduced in different groups
    according to the weigths' `dp_comm` attribute.
    If it is set to `dp`, it will only be reduced across the data-parallel
    groups, which means that in the model parallel group, they are not
    synchronized.

    根据权重的 "dp_comm "属性，权重的等级被确定为不同组别中的减数。 
    根据权重的 "dp_comm "属性。 如果它被设置为 "dp"，它将只在数据并行组中被减少

    If it is set to `world`, the gradients is synchronized across all workers,
    regardless their model or data parallel group. This is extremely useful for
    shared layers like the gate.
    如果它被设置为`world'，梯度就会在所有工作者之间同步，
    不管他们的模型或数据并行组。
    这对于像门这样的共享层来说是非常有用的。
    """

    def __init__(
        self,
        module,
        auto_allreduce=False,
        need_sync=True,
        **kwargs
    ):
        assert not auto_allreduce, "Automatic all-reduce is not implemented yet"

        super().__init__()
        self.module = module

        self.comms = dict()
        for k in kwargs:
            if k.endswith('_group'):
                self.comms[k[:-6]] = kwargs[k]
        for k in ['dp', 'gate', 'moe', 'world']:
            if k not in self.comms:
                self.comms[k] = get_torch_default_comm()

        def allreduce_params(no_scale=False,
                reduce_after=False, fp32_allreduce=False):
            groups = dict()
            for p in self.module.parameters():
                if not p.requires_grad or p.grad is None:
                    continue
                if hasattr(p, "dp_comm"):
                    dp_comm = p.dp_comm
                else:
                    dp_comm = "dp"
                group_key = (dp_comm, p.dtype)
                if group_key not in groups:
                    groups[group_key] = [p]
                else:
                    groups[group_key].append(p)
            for (dp_comm, dtype), group in groups.items():
                if dp_comm not in self.comms:
                    continue
                comm = self.comms[dp_comm]
                grads = [p.grad.data for p in group]
                coalesced = _flatten_dense_tensors(grads)
                if fp32_allreduce and dtype != torch.float32:
                    coalesced = coalesced.float()
                if not no_scale and not reduce_after:
                    coalesced /= comm.size()
                torch.distributed.all_reduce(coalesced, group=comm)
                torch.cuda.synchronize()
                if not no_scale and reduce_after:
                    coalesced /= comm.size()
                synced = _unflatten_dense_tensors(coalesced, grads)
                for g, s in zip(grads, synced):
                    g.copy_(s)

        self.allreduce_params = allreduce_params
        if need_sync:
            self._sync_params()

    def _sync_params(self):
        groups = dict()
        for p in self.module.parameters():
            if hasattr(p, "dp_comm"):
                dp_comm = p.dp_comm
            else:
                dp_comm = "dp"
            group_key = (dp_comm, p.dtype)
            if group_key not in groups:
                groups[group_key] = [p]
            else:
                groups[group_key].append(p)
        for (dp_comm, _), group in groups.items():
            if dp_comm not in self.comms:
                continue
            comm = self.comms[dp_comm]
            datas = [p.data for p in group]
            coalesced = _flatten_dense_tensors(datas)
            torch.distributed.broadcast(coalesced,
                    get_rank_0_in_comm(comm), group=comm)
            torch.cuda.synchronize()
            synced = _unflatten_dense_tensors(coalesced, datas)
            for d, s in zip(datas, synced):
                d.copy_(s)

    def forward(self, *args, **kwargs):
        r"""
        Directly call the module's forward function.
        """
        return self.module(*args, **kwargs)
