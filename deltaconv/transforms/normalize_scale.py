import torch
import torch.linalg as LA


class NormalizeScale(object):
    r"""Centers and normalizes node positions to the interval :math:`(-1, 1)`.
    """
    def __init__(self, norm_ord=2, scaling_factor=None):
        self.norm_ord = norm_ord
        self.scaling_factor = scaling_factor

    def __call__(self, data):
        data.pos = data.pos - (torch.max(data.pos, dim=0)[0] + torch.min(data.pos, dim=0)[0]) / 2

        if self.scaling_factor is None:
            scale = (1 / LA.norm(data.pos, ord=self.norm_ord, dim=1).max()) * 0.999999
        else:
            scale = (1 / self.scaling_factor) * 0.999999
        data.pos = data.pos * scale

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)