import torch
import torch.nn as nn
import math
from .modules import *
from .encoder import *
from .decoder import *
from .transformer import Transfomer


def build_transformer(
    src_vocab_size: int,
    trgt_vocab_size: int,
    src_seq_len: int,
    trgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 0,
    dropout: float = 0.1,
    d_ff=2048,
) -> Transfomer:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    trgt_embed = InputEmbeddings(d_model, trgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    trgt_pos = PositionalEncoding(d_model, trgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model,encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model,decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)
    encoder = Encoder(d_model,nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model,nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, trgt_vocab_size)
    transformer = Transfomer(
        encoder, decoder, src_embed, trgt_embed, src_pos, trgt_pos, projection_layer
    )
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return transformer
