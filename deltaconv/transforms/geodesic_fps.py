import torch
from ..geometry.fps import geodesic_fps
from math import ceil

class GeodesicFPS(object):
    r"""Sample points using geodesic furthest point samples.
    """

    def __init__(self, n_samples=None, store_original=False):
        self.n_samples = n_samples
        self.store_original = store_original
        return

    def __call__(self, data):

        if self.n_samples is None:
            self.n_samples = data.pos.size(0)
            
        idx = torch.from_numpy(geodesic_fps(data.pos.cpu().numpy(), self.n_samples)).long()
        if data.pos.size(0) < self.n_samples:
            idx = idx[:data.pos.size(0)].repeat(ceil(self.n_samples / data.pos.size(0)))

        idx = idx[:self.n_samples]
        assert idx.max() <= data.pos.size(0)
        assert idx.min() >= 0
        
        data.sample_idx = idx

        if self.store_original:
            data.pos_original = data.pos
            data.y_original = data.y

        data.pos = data.pos[idx]
        if hasattr(data, 'norm') and data.norm is not None:
            data.norm = data.norm[idx]
        if hasattr(data, 'normal') and data.normal is not None:
            data.norm = data.normal[idx]
        if hasattr(data, 'x') and data.x is not None:
            data.x = data.x[idx]
        if hasattr(data, 'y') and data.y is not None and type(data.y) is not int and data.y.size(0) > 1:
            data.y = data.y[idx]

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)