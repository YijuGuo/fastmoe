import os, sys
import glob

from collections import Counter, OrderedDict
import numpy as np
import torch

from utils.vocabulary import Vocab

# 训练数据整理成batch格式，数据的加载和预处理
class LMOrderedIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """

        # bsz batch size = 60
        self.bsz = bsz
        # bptt target lenth= 150
        # 如果用一个批次处理完所有的数据，以训练数据为例，每个句子长度高达1720450
        # 显然是不科学的，因此需要限定每个批次中的句子长度允许的最大值bptt
        self.bptt = bptt
        # ext_len = 0 
        self.ext_len = ext_len if ext_len is not None else 0
        # 判断机器是cpu还是gpu
        self.device = device

        # data.size(0) = 一亿，
        # n_step = 一亿/60 = 1720450
        # Work out how cleanly we can divide the dataset into bsz parts.
        # 取整数得到一个nbatch代表需要多少次batch后能够遍历完所有数据
        self.n_step = data.size(0) // bsz

        # 修剪掉任何不能整齐排列的多余元素
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        # data = tensor:narrow(dim, index, size) 
        # 表示取出tensor中第dim维上索引从index开始到index+size-1的所有元素存放在data中 
        # 第一个参数是代表横轴删除还是纵轴删除, 0为横轴，1为纵轴
        # 第二个和第三个参数代表保留开始轴到结束轴的数值.类似于切片
        # 使用narrow方法对不规整的剩余数据删除
        data = data.narrow(0, 0, self.n_step * bsz)

        # Evenly divide the data across the bsz batches.
        # 将数据均匀地划分到各批次的bsz中
        # 使用view方法对data进行矩阵变换
        # 因为会做转置操作, 因此矩阵的形状是[None, bsz]
        # view方法没有拷贝新的张量，没有开辟新的内存，与原张量共享内存
        # view方法重新定义访问张量的规则，使取出的张量按照我们所希望的形状展现
        # contiguous()：断开这两个变量之间的依赖
        # 调用contiguous()时，会强制拷贝一份tensor
        # 让它的布局和从头创建的一模一样，但是两个tensor完全没有联系
        self.data = data.view(bsz, -1).t().contiguous().to(device)

        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    def get_batch(self, i, bptt=None):
        if bptt is None: bptt = self.bptt
        # seq_len = bptt
        # 首先我们确定句子长度, 它将是在bptt和len(source) - 1 - i中最小值
        # 实质上, 前面的批次中都会是bptt的值, 只不过最后一个批次中, 句子长度
        # 可能不够bptt的60个, 因此会变为len(source) - 1 - i的值
        # seq_len = bptt = 152
        seq_len = min(bptt, self.data.size(0) - 1 - i)

        # i = 0+152 
        end_idx = i + seq_len
        # begin_idx = 0
        beg_idx = max(0, i - self.ext_len)

        # 对self.data截取，从begin_idx截取到end_idx
        # 仅对行进行截取，列向量不变
        # 截取出来的维度为[152,60]
        # 语言模型训练的源数据的第i批数据将是batchify的结果的切片[i:i+seq_len]
        data = self.data[beg_idx:end_idx]

        # [152,60]
        # target从第1行开始截取，data从第0行开始截取
        # 根据语言模型训练的语料规定, 它的目标数据是源数据向后移动一位
        target = self.data[i+1:i+1+seq_len]

        return data, target, seq_len

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        # max_len - 150+3*5
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            # bptt = 150
            # 如果random小于0.95，bptt等于默认值，反之则bptt = bptt/2
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
            # random(bptt = 152/148/146, std = 5) 与 min_len = 5 取max
            # 再与max_len比较取较小的结果
            # random函数导致结果不一样
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            # 通过get_batch方法获取数据
            data, target, seq_len = self.get_batch(i, bptt)
            # seq_len加到i上，index后移到下一个需要处理的句子的首个单词的位置
            i += seq_len
            # yield函数：跳出循环，获取到第一个批次的数据送入模型训练
            yield data, target, seq_len
            # 判断i是否大于等于data总长度
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        return self.get_fixlen_iter()


class LMShuffledIterator(object):
    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None, shuffle=False):
        """
            data -- list[LongTensor] -- there is no order among the LongTensors
        """
        self.data = data

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self):
        # index iterator
        epoch_indices = np.random.permutation(len(self.data)) if self.shuffle \
            else np.array(range(len(self.data)))

        # sentence iterator
        for idx in epoch_indices:
            yield self.data[idx]

    def stream_iterator(self, sent_stream):
        # streams for each data in the batch
        streams = [None] * self.bsz

        data = torch.LongTensor(self.bptt, self.bsz)
        target = torch.LongTensor(self.bptt, self.bsz)

        n_retain = 0

        while True:
            # data   : [n_retain+bptt x bsz]
            # target : [bptt x bsz]
            data[n_retain:].fill_(-1)
            target.fill_(-1)

            valid_batch = True

            for i in range(self.bsz):
                n_filled = 0
                try:
                    while n_filled < self.bptt:
                        if streams[i] is None or len(streams[i]) <= 1:
                            streams[i] = next(sent_stream)
                        # number of new tokens to fill in
                        n_new = min(len(streams[i]) - 1, self.bptt - n_filled)
                        # first n_retain tokens are retained from last batch
                        data[n_retain+n_filled:n_retain+n_filled+n_new, i] = \
                            streams[i][:n_new]
                        target[n_filled:n_filled+n_new, i] = \
                            streams[i][1:n_new+1]
                        streams[i] = streams[i][n_new:]
                        n_filled += n_new
                except StopIteration:
                    valid_batch = False
                    break

            if not valid_batch:
                return

            data = data.to(self.device)
            target = target.to(self.device)

            yield data, target, self.bptt

            n_retain = min(data.size(0), self.ext_len)
            if n_retain > 0:
                data[:n_retain] = data[-n_retain:]
            data.resize_(n_retain + self.bptt, data.size(1))

    def __iter__(self):
        # sent_stream is an iterator
        sent_stream = self.get_sent_stream()

        for batch in self.stream_iterator(sent_stream):
            yield batch


class LMMultiFileIterator(LMShuffledIterator):
    def __init__(self, paths, vocab, bsz, bptt, device='cpu', ext_len=None,
        shuffle=False):

        self.paths = paths
        self.vocab = vocab

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self, path):
        sents = self.vocab.encode_file(path, add_double_eos=True)
        if self.shuffle:
            np.random.shuffle(sents)
        sent_stream = iter(sents)

        return sent_stream

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.paths)

        for path in self.paths:
            # sent_stream is an iterator
            sent_stream = self.get_sent_stream(path)
            for batch in self.stream_iterator(sent_stream):
                yield batch


class Corpus(object):
    def __init__(self, path, dataset, *args, **kwargs):
        # 赋值到self.dataset 
        self.dataset = dataset
        # 将Vocab类赋值给self.vocab
        self.vocab = Vocab(*args, **kwargs)

        # 判断dataset
        if self.dataset in ['ptb', 'wt2', 'enwik8', 'text8']:
            self.vocab.count_file(os.path.join(path, 'train.txt'))
            self.vocab.count_file(os.path.join(path, 'valid.txt'))
            self.vocab.count_file(os.path.join(path, 'test.txt'))
        elif self.dataset == 'wt103':
            self.vocab.count_file(os.path.join(path, 'train.txt'))
        elif self.dataset == 'lm1b':
            train_path_pattern = os.path.join(
                path, '1-billion-word-language-modeling-benchmark-r13output',
                'training-monolingual.tokenized.shuffled', 'news.en-*')
            train_paths = glob.glob(train_path_pattern)
            # the vocab will load from file when build_vocab() is called
        # 词典的构建
        self.vocab.build_vocab()

        if self.dataset in ['ptb', 'wt2', 'wt103']:
            self.train = self.vocab.encode_file(
                os.path.join(path, 'train.txt'), ordered=True)
            self.valid = self.vocab.encode_file(
                os.path.join(path, 'valid.txt'), ordered=True)
            self.test  = self.vocab.encode_file(
                os.path.join(path, 'test.txt'), ordered=True)
        elif self.dataset in ['enwik8', 'text8']:
            self.train = self.vocab.encode_file(
                os.path.join(path, 'train.txt'), ordered=True, add_eos=False)
            self.valid = self.vocab.encode_file(
                os.path.join(path, 'valid.txt'), ordered=True, add_eos=False)
            self.test  = self.vocab.encode_file(
                os.path.join(path, 'test.txt'), ordered=True, add_eos=False)
        elif self.dataset == 'lm1b':
            self.train = train_paths
            self.valid = self.vocab.encode_file(
                os.path.join(path, 'valid.txt'), ordered=False, add_double_eos=True)
            self.test  = self.vocab.encode_file(
                os.path.join(path, 'test.txt'), ordered=False, add_double_eos=True)
    
    #加载训练数据时调用
    def get_iterator(self, split, *args, **kwargs):
        if split == 'train':
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
                # LMorderedIterator 将数据整理成batch的格式
                data_iter = LMOrderedIterator(self.train, *args, **kwargs)
            elif self.dataset == 'lm1b':
                kwargs['shuffle'] = True
                # LMMultiFileIterator
                data_iter = LMMultiFileIterator(self.train, self.vocab, *args, **kwargs)
        elif split in ['valid', 'test']:
            data = self.valid if split == 'valid' else self.test
            if self.dataset in ['ptb', 'wt2', 'wt103', 'enwik8', 'text8']:
                data_iter = LMOrderedIterator(data, *args, **kwargs)
            elif self.dataset == 'lm1b':
                # LMShuffledIterator
                data_iter = LMShuffledIterator(data, *args, **kwargs)

        return data_iter

# 进入数据预处理的functon
def get_lm_corpus(datadir, dataset):
    # 加载路径，将路径存储为.pt格式的文件
    fn = os.path.join(datadir, 'cache.pt')
    # 若存在路径，加载预训练好的数据
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    # 若不存在路径，开始预处理
    else:
        print('Producing dataset {}...'.format(dataset))
        # 生成词典
        kwargs = {}
        # 判断dataset是否为wt103或wt2两个数据集当中
        if dataset in ['wt103', 'wt2']:
            # 词典添加对应的key和value
            # <eos> 标志符：读完一行需要加一个eos标志符
            kwargs['special'] = ['<eos>']
            # 是否转变大小写
            kwargs['lower_case'] = False
        elif dataset == 'ptb':
            kwargs['special'] = ['<eos>']
            kwargs['lower_case'] = True
        elif dataset == 'lm1b':
            kwargs['special'] = []
            kwargs['lower_case'] = False
            kwargs['vocab_file'] = os.path.join(datadir, '1b_word_vocab.txt')
        elif dataset in ['enwik8', 'text8']:
            pass
        # Corpus预处理
        corpus = Corpus(datadir, dataset, **kwargs)
        # 存储
        torch.save(corpus, fn)

    return corpus

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='unit test')
    parser.add_argument('--datadir', type=str, default='../data/text8',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='text8',
                        choices=['ptb', 'wt2', 'wt103', 'lm1b', 'enwik8', 'text8'],
                        help='dataset name')
    args = parser.parse_args()

    corpus = get_lm_corpus(args.datadir, args.dataset)
    print('Vocab size : {}'.format(len(corpus.vocab.idx2sym)))
