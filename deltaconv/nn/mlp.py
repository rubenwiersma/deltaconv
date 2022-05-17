import torch
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d, LeakyReLU

from .nonlin import BatchNorm1d, VectorNonLin


def MLP(channels, bias=False, nonlin=LeakyReLU(negative_slope=0.2)):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i], bias=bias), BatchNorm1d(channels[i]), nonlin)
        for i in range(1, len(channels))
    ])

def VectorMLP(channels, batchnorm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i], bias=False), VectorNonLin(channels[i], batchnorm=BatchNorm1d(channels[i]) if batchnorm else None))
        for i in range(1, len(channels))
    ])

class ScalarVectorMLP(torch.nn.Module):
    def __init__(self, channels, nonlin=True, vector_stream=True):
        super(ScalarVectorMLP, self).__init__()
        self.scalar_mlp = MLP(channels, nonlin=LeakyReLU(negative_slope=0.2) if nonlin else torch.nn.Identity())
        self.vector_mlp = None
        if vector_stream:
            self.vector_mlp = VectorMLP(channels)

    def forward(self, x):
        assert self.vector_mlp is None or (self.vector_mlp is not None and type(x) is tuple)

        if type(x) is tuple:
            x, v = x

        x = self.scalar_mlp(x)

        if self.vector_mlp is not None:
            v = self.vector_mlp(v)
            x = (x, v)

        return x

class ScalarVectorIdentity(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ScalarVectorIdentity, self).__init__()

    def forward(self, input):
        return input
