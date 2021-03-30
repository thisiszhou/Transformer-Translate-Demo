# Transformer Translation Demo

## Introduction
This demo is just for learning Transformer, which is basing on only one pair of sentences:
Input sentence: ['\<bos\>', 'i', 'am', 'iron', 'man', '\<eos\>']
After translate: ['\<bos\>', '我', '是', '钢铁', '侠', '\<eos\>']

> Entire tutorial of Transformer Translation model, please see: https://zhuanlan.zhihu.com/p/360343417

## Requirement
```bashrc
$ pip install -r requirements.txt
```

## Quick Start

```bashrc
$ python no_mask_train.py
$ python no_mask_infer.py
```

## train and infer with key_padding_mask of Attention
```bashrc
$ python key_padding_mask_train.py
$ python key_padding_mask_infer.py
```

## train and infer with tgt upper triangular matrix attn_mask of Attention
```bashrc
$ python triu_mask_train.py
$ python triu_mask_infer.py
```