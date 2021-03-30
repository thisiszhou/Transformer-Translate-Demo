from utils.dictionary import Dictionary
from typing import Dict, List, Tuple
import torch


class Dataset(object):
    def __init__(self,
                 src_dict: Dict[str, int],
                 tgt_dict: Dict[str, int],
                 corpus: List[List[Tuple]],
                 src_sequence_size: int,
                 tgt_sequence_size: int,):
        self.src_dict = Dictionary(src_dict)
        self.tgt_dict = Dictionary(tgt_dict)
        self.corpus = corpus
        self.src_sequence_size = src_sequence_size
        self.tgt_sequence_size = tgt_sequence_size

        # inner para
        self.srcs = dict()
        self.datas = []
        self.offset = 0
        self.data_length = 0
        self.init()

    def init(self):
        for i, sentence in enumerate(self.corpus):
            src, tgt = sentence
            src = [self.src_dict.w2i[x] for x in src]
            tgt = [self.tgt_dict.w2i[x] for x in tgt]
            self.srcs[i] = src
            for j, word in enumerate(tgt):
                context = tgt[: j]
                if len(context) == 0:
                    continue
                one_case = (i, context, word)
                self.datas.append(one_case)
        self.data_length = len(self.datas)

    def get_batch(self, batch_size=2, padding_str='<pad>', need_padding_mask=False):
        """
        :return: src, tgt_in, tgt_out, src_padding_mask, tgt_padding_mask
        src:              Tensor, shape :[batch_size, src_sequence_size]
        tgt_in:           Tensor, shape: [batch_size, tgt_sequence_size]
        tgt_out:          Tensor, shape: [batch_size, tgt_sequence_size]
        src_padding_mask: Tensor, shape: [batch_size, src_sequence_size]
        tgt_padding_mask: Tensor, shape: [batch_size, tgt_sequence_size]
        """
        src_pad_i = self.src_dict.w2i[padding_str]
        tgt_pad_i = self.tgt_dict.w2i[padding_str]
        if self.offset >= self.data_length:
            self.offset = 0
        data = self.datas[self.offset: self.offset + batch_size]
        self.offset += batch_size
        src = [self.srcs[x[0]] for x in data]
        tgt_in = [x[1] for x in data]
        tgt_out = [x[2] for x in data]
        [pad_(x, self.src_sequence_size, src_pad_i) for x in src]
        [pad_(x, self.tgt_sequence_size, tgt_pad_i) for x in tgt_in]

        src = torch.tensor(src, dtype=torch.int64)
        tgt_in = torch.tensor(tgt_in, dtype=torch.int64)
        tgt_out = torch.tensor(tgt_out, dtype=torch.int64)
        if not need_padding_mask:
            return src, tgt_in, tgt_out, None, None
        else:
            src_padding_mask = src == src_pad_i
            tgt_padding_mask = tgt_in == tgt_pad_i
            return src, tgt_in, tgt_out, src_padding_mask, tgt_padding_mask


def pad_(seq: List, align_length: int, pad):
    seq.extend([pad for _ in range(align_length - len(seq))])





