import numbers
from itertools import repeat

import torch
import torch.linalg as LA


class RandomNormals(object):
    r"""Jitters normals by a translation within a given interval.
    This is followed by normalization to ensure unit normals.

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
        (n, dim), t = data.pos.size(), self.translate
        if isinstance(t, numbers.Number):
            t = list(repeat(t, times=dim))
        assert len(t) == dim

        ts = []
        for d in range(dim):
            ts.append(data.pos.new_empty(n).uniform_(-abs(t[d]), abs(t[d])))

        data.norm = data.norm + torch.stack(ts, dim=-1)
        data.norm = data.norm / LA.norm(data.norm, dim=-1, keepdims=True).clamp(1e-5)
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.translate)