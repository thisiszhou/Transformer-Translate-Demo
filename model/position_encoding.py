from torch import nn
from torch import Tensor
import torch
import math


class PositionalEncoding(nn.Module):

    def __init__(self, word_emb_dim: int, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position_emb = torch.zeros(max_len, word_emb_dim)

        # position 编码
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        dim_div_term = torch.exp(torch.arange(0, word_emb_dim, 2).float() * (-math.log(10000.0) / word_emb_dim))

        # word_emb_dim 编码
        position_emb[:, 0::2] = torch.sin(position * dim_div_term)
        position_emb[:, 1::2] = torch.cos(position * dim_div_term)
        pe = position_emb.unsqueeze(0).transpose(0, 1)  # shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor):
        """
        :param x: Tensor, shape: [batch_size, sequence_length, word_emb_dim]
        :return: Tensor, shape: [batch_size, sequence_length, word_emb_dim]
        """

        # 编码信息与原始信息加和后输出
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)




