import copy

import torch.nn as nn

from src.model.Decoder import Decoder, DecoderLayer
from src.model.Embeddings import Embeddings
from src.model.Encoder import Encoder, EncoderLayer
from src.model.EncoderDecoder import EncoderDecoder
from src.model.Generator import Generator
from src.model.MultiHeadAttention import MultiHeadAttention
from src.model.PositionalEncoding import PositionalEncoding
from src.model.PositionwiseFeedForward import PositionwiseFeedForward


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    Helper function which constructs a transformer model from the hyperparameters.

    :param src_vocab:
    :param tgt_vocab:
    :param N:
    :param d_model:
    :param d_ff:
    :param h:
    :param dropout:
    :return:
    """
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model