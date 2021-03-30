from torch.nn import Module, Linear
import torch
from torch import Tensor
from typing import Optional


class MultiheadAttention(Module):
    def __init__(self,
                 word_emb_dim,
                 nheads,
                 dropout_prob=0.
                 ):
        super(MultiheadAttention, self).__init__()
        self.word_emb_dim = word_emb_dim
        self.num_heads = nheads
        self.dropout_prob = dropout_prob
        self.head_dim = word_emb_dim // nheads
        assert self.head_dim * nheads == self.word_emb_dim  # embed_dim must be divisible by num_heads

        self.q_in_proj = Linear(word_emb_dim, word_emb_dim)
        self.k_in_proj = Linear(word_emb_dim, word_emb_dim)
        self.v_in_proj = Linear(word_emb_dim, word_emb_dim)

        self.out_proj = Linear(word_emb_dim, word_emb_dim)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None):
        """
        :param query: Tensor, shape: [tgt_sequence_size, batch_size, word_emb_dim]
        :param key:   Tensor, shape: [src_sequence_size, batch_size, word_emb_dim]
        :param value: Tensor, shape: [src_sequence_size, batch_size, word_emb_dim]
        :param key_padding_mask:  Tensor, shape: [batch_size, src_sequence_size]
        :param attn_mask: Tensor, shape: [tgt_sequence_size, src_sequence_size]
        :return: Tensor, shape: [tgt_sequence_size, batch_size, word_emb_dim]
        """

        # 获取query的shape,这里按照torch源码要求，按照tgt_sequence_size, batch_size, word_emb_dim顺序排列
        tgt_len, batch_size, word_emb_dim = query.size()
        num_heads = self.num_heads
        assert word_emb_dim == self.word_emb_dim
        head_dim = word_emb_dim // num_heads

        # 检查word_emb_dim是否可以被num_heads整除
        assert head_dim * num_heads == word_emb_dim
        scaling = float(head_dim) ** -0.5

        # 三个Q、K、V的全连接层
        q = self.q_in_proj(query)
        k = self.k_in_proj(key)
        v = self.v_in_proj(value)

        # 这里对Q进行一个统一常数放缩
        q = q * scaling

        # multihead运算技巧，将word_emb_dim切分为num_heads个head_dim，并且让num_heads与batch_size暂时使用同一维度
        # 切分word_emb_dim后将batch_size * num_heads转换至第0维，为三维矩阵的矩阵乘法（bmm）做准备
        q = q.contiguous().view(tgt_len, batch_size * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, batch_size * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, batch_size * num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        # Q、K进行bmm批次矩阵乘法，得到权重矩阵
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [batch_size * num_heads, tgt_len, src_len]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(batch_size, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(batch_size * num_heads, tgt_len, src_len)

        # 权重矩阵进行softmax，使得单行的权重和为1
        attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
        attn_output_weights = torch.dropout(attn_output_weights, p=self.dropout_prob, train=self.training)

        # 权重矩阵与V矩阵进行bmm操作，得到输出
        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [batch_size * num_heads, tgt_len, head_dim]

        # 转换维度，将num_heads * head_dim reshape回word_emb_dim，并且将batch_size调回至第1维
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, batch_size, word_emb_dim)

        # 最后一层全连接层，得到最终输出
        attn_output = self.out_proj(attn_output)
        return attn_output


