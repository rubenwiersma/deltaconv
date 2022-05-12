from tkinter import W
import torch


class NormalizeAxes(object):
    r"""Rotates all points so that the eigenvectors overlie the axes of the
    Cartesian coordinate system.
    If the data additionally holds normals saved in :obj:`data.norm` these will
    be also rotated.

    Args:
        max_points (int, optional): If set to a value greater than :obj:`0`,
            only a random number of :obj:`max_points` points are sampled and
            used to compute eigenvectors. (default: :obj:`-1`)
    """

    def __init__(self, max_points=-1):
        self.max_points = max_points

    def __call__(self, data):
        pos = data.pos

        std = torch.std(pos, dim=0)
        data.pos = data.pos[:, torch.sort(std)[1]]

        scale = 1 / (2 * data.pos.max(0).values[2])
        data.pos = data.pos * scale

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)