import torch
import torch.linalg as LA


class NormalizeArea(object):
    r"""Centers node positions and normalizes area of shape to 1.
    """

    def __init__(self):
        return

    def __call__(self, data):
        data.pos = data.pos - (torch.max(data.pos, dim=0)[0] + torch.min(data.pos, dim=0)[0]) / 2
        v, f = data.pos.numpy(), data.face.numpy().T
        e1 = data.pos[data.face[:, 1]] - data.pos[data.face[:, 0]]
        e2 = data.pos[data.face[:, 2]] - data.pos[data.face[:, 0]]
        area = 1 / torch.sqrt(LA.norm(LA.cross(e1, e2, dim=-1), dim=-1).sum() / 2)
        data.pos = data.pos * area

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)