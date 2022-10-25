import os
from collections import Counter, OrderedDict
# Counter: 统计词频，作为词典进行使用
import torch

# 构建词典
class Vocab(object):
    def __init__(self, special=[], min_freq=0, max_size=None, lower_case=True,
                 delimiter=None, vocab_file=None):
        self.counter = Counter() # Counter方法
        self.special = special #eos标志符
        self.min_freq = min_freq # 统计最小词频
        self.max_size = max_size # 统计最大词频
        self.lower_case = lower_case # 大小写
        self.delimiter = delimiter # 分隔符
        self.vocab_file = vocab_file # 保存词典的路径
    
    # 对句子进行分词并添加<eos>
    def tokenize(self, line, add_eos=False, add_double_eos=False):
        # strip函数清洗当前内容
        line = line.strip()
        # 判断是否是大小写，convert to lower case
        if self.lower_case:
            line = line.lower()

        # empty delimiter '' will evaluate False
        if self.delimiter == '':
            symbols = line
        else:
            # 当前文本切割成多个token
            symbols = line.split(self.delimiter)
        # 判断是否传入eos
        if add_double_eos: # lm1b
            return ['<S>'] + symbols + ['<S>']
        elif add_eos:
            return symbols + ['<eos>']
        else:
            return symbols
    # 读取语料
    def count_file(self, path, verbose=False, add_eos=False):
        # 判断是否存在路径
        if verbose: print('counting file {} ...'.format(path))
        # 若存在路径则直接打开路径
        assert os.path.exists(path)

        sents = []
        # 一行一行读取内容
        with open(path, 'r', encoding='utf-8') as f:
            # idx表示读到第几行
            # line表示具体的内容
            for idx, line in enumerate(f):
                # 每500000行 print一次
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                # symbols表示当前文本分词之后得到的list
                symbols = self.tokenize(line, add_eos=add_eos)
                # symbols直接add到counter里面
                self.counter.update(symbols)
                # list 一行一行添加symbols的list
                sents.append(symbols)

        return sents

    def count_sents(self, sents, verbose=False):
        """
            sents : a list of sentences, each a list of tokenized symbols
        """
        if verbose: print('counting {} sents ...'.format(len(sents)))
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('    line {}'.format(idx))
            self.counter.update(symbols)

    def _build_from_file(self, vocab_file):
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                symb = line.strip().split()[0]
                self.add_symbol(symb)
        self.unk_idx = self.sym2idx['<UNK>']
    # 构建词典
    def build_vocab(self):
        # 判断vocab file是否存在
        if self.vocab_file:
            print('building vocab from {}'.format(self.vocab_file))
            self._build_from_file(self.vocab_file)
            print('final vocab size {}'.format(len(self)))
        else:
            print('building vocab with min_freq={}, max_size={}'.format(
                self.min_freq, self.max_size))
            # id对应的词
            self.idx2sym = []
            # 词对应的id
            self.sym2idx = OrderedDict()

            # <eos>添加到这两个词典中
            for sym in self.special:
                self.add_special(sym)

            # counter中已经有训练语料，利用most common进行统计
            # 统计词频比较高的词作为词典
            for sym, cnt in self.counter.most_common(self.max_size):
                # 如果词频小于我们设置参数最小值
                if cnt < self.min_freq: break
                self.add_symbol(sym)

            print('final vocab size {} from {} unique tokens'.format(
                len(self), len(self.counter)))
   
    # 把token转换成相应的index
    def encode_file(self, path, ordered=False, verbose=False, add_eos=True,
            add_double_eos=False):
        if verbose: print('encoding file {} ...'.format(path))
        assert os.path.exists(path)
        encoded = []
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print('    line {}'.format(idx))
                symbols = self.tokenize(line, add_eos=add_eos,
                    add_double_eos=add_double_eos)
                # encoded 当前每一行文本转换成idex的list
                encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            # 把所有的list拼接
            # [1,2,3,4]*[2,3,4,5] = [1,2,3,4,2,3,4,5]
            encoded = torch.cat(encoded)
        # encoded 的维度为一亿个token的list
        return encoded

    def encode_sents(self, sents, ordered=False, verbose=False):
        if verbose: print('encoding {} sents ...'.format(len(sents)))
        encoded = []
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print('    line {}'.format(idx))
            encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, '{}_idx'.format(sym.strip('<>')), self.sym2idx[sym])

    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def get_sym(self, idx):
        assert 0 <= idx < len(self), 'Index {} out of range'.format(idx)
        return self.idx2sym[idx]

    def get_idx(self, sym):
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        else:
            # print('encounter unk {}'.format(sym))
            assert '<eos>' not in sym
            assert hasattr(self, 'unk_idx')
            return self.sym2idx.get(sym, self.unk_idx)

    def get_symbols(self, indices):
        return [self.get_sym(idx) for idx in indices]

    def get_indices(self, symbols):
        return [self.get_idx(sym) for sym in symbols]

    def convert_to_tensor(self, symbols):
        return torch.LongTensor(self.get_indices(symbols))

    def convert_to_sent(self, indices, exclude=None):
        if exclude is None:
            return ' '.join([self.get_sym(idx) for idx in indices])
        else:
            return ' '.join([self.get_sym(idx) for idx in indices if idx not in exclude])

    def __len__(self):
        return len(self.idx2sym)
