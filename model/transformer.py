from torch.nn import Module, Embedding, Linear, LayerNorm
from model.position_encoding import PositionalEncoding
from model.transformer_encoder import TransformerEncoder, TransformerEncoderLayer
from model.transformer_decoder import TransformerDecoder, TransformerDecoderLayer
from torch import Tensor
import torch
from typing import Optional


class Transformer(Module):
    def __init__(
            self,
            src_vocab_size: int,
            tgt_vocab_size: int,
            tgt_sequence_size: int,
            word_emb_dim: int = 32,
            nhead: int = 1,
            num_encoder_layers: int = 2,
            num_decoder_layers: int = 2,
            dim_feedforward: int = 128,
            dropout_prob: float = 0.1,
    ) -> None:
        super(Transformer, self).__init__()

        # para
        self.word_emb_dim = word_emb_dim
        self.nhead = nhead
        self.tgt_sequence_size = tgt_sequence_size

        # layers
        self.src_word_emb = Embedding(src_vocab_size, word_emb_dim)
        self.tgt_word_emb = Embedding(tgt_vocab_size, word_emb_dim)
        self.position_encoding = PositionalEncoding(word_emb_dim, dropout_prob)
        encoder_layer = TransformerEncoderLayer(word_emb_dim, nhead, dim_feedforward, dropout_prob)
        encoder_norm = LayerNorm(word_emb_dim)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        decoder_layer = TransformerDecoderLayer(word_emb_dim, nhead, dim_feedforward, dropout_prob)
        decoder_norm = LayerNorm(word_emb_dim)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.generator = Linear(word_emb_dim * tgt_sequence_size, tgt_vocab_size)

    def forward(
            self,
            src: Tensor,
            tgt: Tensor,
            src_mask: Optional[Tensor] = None,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        :input:
        src:      Tensor, shape: [batch_size, src_sequence_size]
        tgt:      Tensor, shape: [batch_size, tgt_sequence_size]
        src_mask: Tensor, shape: [src_sequence_size, src_sequence_size]
        tgt_mask: Tensor, shape: [tgt_sequence_size, tgt_sequence_size]
        memory_mask: Tensor, shape: [tgt_sequence_size, src_sequence_size]
        src_key_padding_mask: Tensor, shape: [batch_size, src_sequence_size]
        tgt_key_padding_mask: Tensor, shape: [batch_size, tgt_sequence_size]
        memory_key_padding_mask: Tensor, shape: [batch_size, src_sequence_size]

        :return:
        Tensor, shape: [batch_size, tgt_vocab_size]
        """

        # check batch size
        assert src.size(0) == tgt.size(0)
        # get word emb and position emb
        src = self.position_encoding(self.src_word_emb(src))
        tgt = self.position_encoding(self.tgt_word_emb(tgt))

        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        decode_output: Tensor = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                             tgt_key_padding_mask=tgt_key_padding_mask,
                                             memory_key_padding_mask=memory_key_padding_mask)

        decode_output = decode_output.transpose(0, 1).contiguous().view(-1, self.word_emb_dim * self.tgt_sequence_size)

        output = self.generator(decode_output)
        return torch.softmax(output, dim=-1)

