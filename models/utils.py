import math

import torch as T
import torch.nn as nn
import torch.nn.functional as F


def glorot_uniform_init(weight, fan_in, fan_out):
    v = 6 if (fan_in != 0 and fan_out != 0) else 3
    bound = float(math.sqrt(v / (fan_in + fan_out)))
    nn.init.uniform_(weight, a=-bound, b=bound)


def generate_absolute_positional_embeddings(max_len, d_model, freeze=True):
    with T.no_grad():
        # Compute the positional encodings once in log space.
        pe = T.zeros(max_len, d_model)
        position = T.arange(0, max_len).unsqueeze(1)
        div_term = T.exp(T.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
        pe[:, 0::2] = T.sin(position * div_term)
        pe[:, 1::2] = T.cos(position * div_term)
        assert pe.size() == (max_len, d_model)
    return pe.unsqueeze(0), nn.Embedding.from_pretrained(pe,
                                                         freeze=freeze)


def generate_relative_positional_embeddings(max_len, d_model):
    with T.no_grad():
        # Compute the positional encodings once in log space.
        pe = T.zeros(2 * max_len + 1, d_model)
        position = T.arange(-max_len, max_len + 1).unsqueeze(1)
        div_term = T.exp(T.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
        pe[:, 0::2] = T.sin(position * div_term)
        pe[:, 1::2] = T.cos(position * div_term)
        assert pe.size() == (2 * max_len + 1, d_model)
        pe = nn.Embedding.from_pretrained(pe,
                                          freeze=True)
    return pe


def generate_temporal_encodings(time, d_model):
    with T.no_grad():
        pe = T.zeros(d_model).float()
        div_term = T.exp(T.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
        pe[0::2] = T.sin(time * div_term)
        pe[1::2] = T.cos(time * div_term)

        pe = pe.view(1, 1, d_model)

    return pe


def masked_softmax(logits, mask, dim):
    if mask is None:
        return F.softmax(logits, dim=dim)

    #logits = logits.masked_fill(~mask, float("-inf"))
    logits = sum_normalize(mask * F.softmax(logits, dim=dim))

    return logits


def sum_normalize(logits, eps=1e-10, dim=-1):
    return logits / T.sum(logits + eps, keepdim=True, dim=dim)


# https://gist.github.com/GongXinyuu/3536da55639bd9bfdd5a905ebf3ab88e
def topk_gumbel_softmax(logits, mask=None, tau=1, eps=1e-10, dim=-1, k=5, training=True):
    # type: (Tensor, float, bool, float, int) -> Tensor
    r"""
    Samples from the `Gumbel-Softmax distribution`_ and optionally discretizes.
    You can use this function to replace "F.gumbel_softmax".

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.
    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.
    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.
    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`
      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)
    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)
    .. _Gumbel-Softmax distribution:
        https://arxiv.org/abs/1611.00712
        https://arxiv.org/abs/1611.01144
    """

    if training:
        u = logits.data.new(*logits.size()).uniform_()
        gumbel_noise = -T.log(-T.log(u + eps) + eps)

        y_soft = masked_softmax((logits + gumbel_noise) / tau, mask=mask, dim=dim)  # ~Gumbel(logits,tau)

        S = y_soft.size(dim)
        index = T.topk(y_soft, dim=dim, k=k)[1]
        y_hard = F.one_hot(index, num_classes=S).float()
        if dim == 0:
            dim2 = 0
        else:
            dim2 = dim - 1
        assert dim2 == -2
        ret = y_hard - y_soft.unsqueeze(dim2).detach() + y_soft.unsqueeze(dim2)
    else:
        y_soft = masked_softmax(logits / tau, mask=mask, dim=dim)
        S = y_soft.size(dim)
        index = T.topk(y_soft, dim=dim, k=k)[1]
        ret = F.one_hot(index, num_classes=S).float()

    return ret, y_soft


def gelu(x):
    return 0.5 * x * (1 + T.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
