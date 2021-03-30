from model.transformer import Transformer
from torch import Tensor
from typing import List
from utils.dictionary import Dictionary
import torch


def transform_words_to_tensor(sequence: List[str],
                              word_dict: Dictionary,
                              sentence_length=8,
                              pad_str='<pad>') -> Tensor:
    sequence = [word_dict.w2i[x] for x in sequence]
    sequence.extend([word_dict.w2i[pad_str] for _ in range(sentence_length - len(sequence))])
    tensor = torch.tensor(sequence, dtype=torch.int64).view(1, -1)
    return tensor


def get_padding_mask(input_tensor: Tensor, word_dict: Dictionary, pad_str='<pad>') -> Tensor:
    output = input_tensor == word_dict.w2i[pad_str]
    return output


def infer_with_transformer(model: Transformer,
                           src: Tensor,
                           src_padding_mask: Tensor,
                           tgt_dict: Dictionary,
                           max_length: int,
                           need_padding_mask: bool,
                           tgt_mask: Tensor) -> List[str]:
    out_seq = ['<bos>']
    predict_word = ''
    while len(out_seq) < max_length and predict_word != '<eos>':
        tgt_in = transform_words_to_tensor(out_seq, tgt_dict)
        if need_padding_mask:
            src_pad_mask = src_padding_mask
            tgt_pad_mask = get_padding_mask(tgt_in, tgt_dict)
            tgt_mask = tgt_mask
        else:
            src_pad_mask = None
            tgt_pad_mask = None
            tgt_mask = None
        output = model(src, tgt_in,
                       src_key_padding_mask=src_pad_mask,
                       tgt_key_padding_mask=tgt_pad_mask,
                       tgt_mask=tgt_mask)
        word_i = torch.argmax(output, -1).item()
        predict_word = tgt_dict.i2w[word_i]
        out_seq.append(predict_word)
    return out_seq


def translate(model,
              in_sentence,
              src_dict: Dictionary,
              tgt_dict: Dictionary,
              max_length=8,
              need_padding_mask=False,
              tgt_mask=None):
    src_tensor = transform_words_to_tensor(in_sentence, src_dict)
    if need_padding_mask:
        src_padding_mask = get_padding_mask(src_tensor, src_dict)
    else:
        src_padding_mask = None
    ret = infer_with_transformer(model, src_tensor, src_padding_mask, tgt_dict,
                                 max_length=max_length,
                                 need_padding_mask=need_padding_mask,
                                 tgt_mask=tgt_mask)
    print("Input sentence:", in_sentence, "\n", "After translate:", ret)


def get_upper_triangular(size):
    return torch.triu(torch.ones(size, size), diagonal=1) == 1

