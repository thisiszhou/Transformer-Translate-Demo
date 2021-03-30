# coding: utf-8
import data.base_data as bd
from utils.dictionary import Dictionary
from utils.infer import translate, get_upper_triangular
import torch


if __name__ == "__main__":
    save_file = 'weights/triu_mask_model.pkl'
    transformer = torch.load(save_file)
    print(f"load transformer from : {save_file}")
    cn_dict = Dictionary(bd.cn_dict)
    en_dict = Dictionary(bd.en_dict)
    upper_tri = get_upper_triangular(8)

    input_sentence = ['<bos>', 'i', 'love', 'you', 'three', 'thousand', 'times', '<eos>']
    translate(transformer, input_sentence, en_dict, cn_dict, tgt_mask=upper_tri)

    input_sentence = ['<bos>', 'i', 'am', 'iron', 'man', '<eos>']
    translate(transformer, input_sentence, en_dict, cn_dict, tgt_mask=upper_tri)






