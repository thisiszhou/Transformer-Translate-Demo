from torch.nn import Module, Linear, Dropout, LayerNorm, ModuleList
from model.multi_head_attention import MultiheadAttention
from torch import Tensor
from typing import Optional
import torch
import copy


class TransformerDecoderLayer(Module):

    def __init__(self, word_emb_dim, nhead, dim_feedforward=2048, dropout_prob=0.1):
        super(TransformerDecoderLayer, self).__init__()
        # 初始化基本层
        self.self_attn = MultiheadAttention(word_emb_dim, nhead, dropout_prob=dropout_prob)
        self.multihead_attn = MultiheadAttention(word_emb_dim, nhead, dropout_prob=dropout_prob)
        # Implementation of Feedforward model
        self.linear1 = Linear(word_emb_dim, dim_feedforward)
        self.dropout = Dropout(dropout_prob)
        self.linear2 = Linear(dim_feedforward, word_emb_dim)

        self.norm1 = LayerNorm(word_emb_dim)
        self.norm2 = LayerNorm(word_emb_dim)
        self.norm3 = LayerNorm(word_emb_dim)
        self.dropout1 = Dropout(dropout_prob)
        self.dropout2 = Dropout(dropout_prob)
        self.dropout3 = Dropout(dropout_prob)

        self.activation = torch.relu

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        :param tgt:                     Tensor, shape: [tgt_sequence_size, batch_size, word_emb_dim]
        :param memory:                  Tensor, shape: [src_sequence_size, batch_size, word_emb_dim]
        :param tgt_mask:                Tensor, shape: [tgt_sequence_size, tgt_sequence_size]
        :param memory_mask:             Tensor, shape: [src_sequence_size, src_sequence_size]
        :param tgt_key_padding_mask:    Tensor, shape: [batch_size, tgt_sequence_size]
        :param memory_key_padding_mask: Tensor, shape: [batch_size, src_sequence_size]
        :return:                        Tensor, shape: [tgt_sequence_size, batch_size, word_emb_dim]
        """
        # tgt的self attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # tgt与memory的attention
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # 两层全连接层
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerDecoder(Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        :param tgt:                     Tensor, shape: [tgt_sequence_size, batch_size, word_emb_dim]
        :param memory:                  Tensor, shape: [src_sequence_size, batch_size, word_emb_dim]
        :param tgt_mask:                Tensor, shape: [tgt_sequence_size, tgt_sequence_size]
        :param memory_mask:             Tensor, shape: [src_sequence_size, src_sequence_size]
        :param tgt_key_padding_mask:    Tensor, shape: [batch_size, tgt_sequence_size]
        :param memory_key_padding_mask: Tensor, shape: [batch_size, src_sequence_size]
        :return:                        Tensor, shape: [tgt_sequence_size, batch_size, word_emb_dim]
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        output = self.norm(output)

        return output


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for _ in range(N)])