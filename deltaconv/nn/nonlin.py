import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn.inits import zeros


EPS = 1e-8


class BatchNorm1d(torch.nn.Module):
    r"""Convenience wrapper around BatchNorm1d that transforms an
    input tensor from [N x C] to [1 x C x N] so that it uses the faster
    batch-wise implementation of PyTorch.
    """
    def __init__(self, in_channels, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm1d, self).__init__()
        self.bn = torch.nn.BatchNorm1d(in_channels, eps, momentum, affine,
                              track_running_stats)
        self.reset_parameters()


    def reset_parameters(self):
        self.bn.reset_parameters()


    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(2).transpose(0, 2)
        x = self.bn(x)
        return x.transpose(0, 2).squeeze(2).contiguous()


    def __repr__(self):
        return f'{self.__class__.__name__}({self.bn.num_features})'


class VectorNonLin(torch.nn.Module):
    r"""Applies a non-linearity to the norm of vector features.

    Args:
        in_channels (int): the number of channels in the input tensor.
        nonlin (Module, optional): non-linearity that will be applied
            to the features (default: ReLU).
        batchnorm (Module, optional): batchnorm operation to call before
            the non-linearity is applied (default: None).
    """
    def __init__(self, in_channels, nonlin=torch.nn.ReLU(), batchnorm=None):
        super(VectorNonLin, self).__init__()
        self.bias = torch.nn.Parameter(torch.Tensor(in_channels))
        self.nonlin = nonlin
        self.batchnorm = batchnorm

        self.reset_parameters()


    def reset_parameters(self):
        zeros(self.bias)
        if self.batchnorm is not None:
            self.batchnorm.reset_parameters()
        

    def forward(self, x: Tensor) -> Tensor:
        N, C = x.size()

        # Compute norm of vector features
        x = x.view(-1, 2, C)
        norm = torch.linalg.norm(x, dim=1, keepdim=False)

        # Add a bias or apply batchnorm
        # This bias and batch-norm 'shift' the non-linearity,
        # which is necessary because norms are non-negative
        # and would be untouched by a ReLU otherwise.
        if self.batchnorm is None:
            norm_shifted = norm + self.bias.view(1, -1)
        else:
            norm_shifted = self.batchnorm(norm)
        norm_nonlin = self.nonlin(norm_shifted)
        x_nonlin = x * (norm_nonlin / norm.clamp(EPS)).unsqueeze(1)

        # Reshape x back to its original shape and return
        return (x_nonlin).view(N, C).contiguous()


    def __repr__(self):
        return f'{self.__class__.__name__}(batchnorm={self.batchnorm.__repr__()})'
