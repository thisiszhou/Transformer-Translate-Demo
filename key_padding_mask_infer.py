# coding: utf-8
import data.base_data as bd
from utils.dictionary import Dictionary
from utils.infer import translate
import torch


if __name__ == "__main__":
    save_file = 'weights/padding_mask_model.pkl'
    transformer = torch.load(save_file)
    print(f"load transformer from : {save_file}")
    transformer.eval()
    cn_dict = Dictionary(bd.cn_dict)
    en_dict = Dictionary(bd.en_dict)

    input_sentence = ['<bos>', 'i', 'love', 'you', 'three', 'thousand', 'times', '<eos>']
    translate(transformer, input_sentence, en_dict, cn_dict, need_padding_mask=True)

    input_sentence = ['<bos>', 'i', 'am', 'iron', 'man', '<eos>']
    translate(transformer, input_sentence, en_dict, cn_dict, need_padding_mask=True)


