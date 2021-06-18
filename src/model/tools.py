import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    """
    Produce N identical layers.

    :param module:
    :param N:
    :return:
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    """
    Mask out subsequent positions.
    :param size:
    :return:
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention.

    Scaled dot product attention has the advantage that the gradient of the softmax function
    is bounded. With regular dot-product (multiplicative) attention, the magnitudes of the can
    grow quite large, pushing the gradient of the softmax function to zero.

    :param query:
    :param key:
    :param value:
    :param mask:
    :param dropout:
    :return:
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
