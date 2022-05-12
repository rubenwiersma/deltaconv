import numbers
from itertools import repeat

import torch


class RandomTranslateGlobal(object):
    r"""Translates shapes by randomly sampled translation values
    within a given interval. This translation happens for the entire shape,
    retaining local relationships.

    Args:
        translate (sequence or float or int): Maximum translation in each
            dimension, defining the range
            :math:`(-\mathrm{translate}, +\mathrm{translate})` to sample from.
            If :obj:`translate` is a number instead of a sequence, the same
            range is used for each dimension.
    """

    def __init__(self, translate):
        self.translate = translate

    def __call__(self, data):
        (_, dim), t = data.pos.size(), self.translate
        if isinstance(t, numbers.Number):
            t = list(repeat(t, times=dim))
        assert len(t) == dim

        ts = []
        for d in range(dim):
            ts.append(data.pos.new_empty(1).uniform_(-abs(t[d]), abs(t[d])))

        data.pos = data.pos + torch.stack(ts, dim=-1)
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.translate)
