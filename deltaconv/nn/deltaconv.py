import torch
from torch_scatter import scatter

from .mlp import MLP, VectorMLP
from ..geometry.operators import curl, norm, I_J, hodge_laplacian, laplacian


class DeltaConv(torch.nn.Module):
    """ DeltaConv convolution from the paper 
    "DeltaConv: Anisotropic Operators for Geometric Deep Learning on Point Clouds".
    This convolution learns a combination of operators from vector calculus:
        grad, co-grad, div, curl; and their compositions Laplacian and Hodge-Laplacian
    and separates features into a scalar and vector stream.

    DeltaConv can be applied to any discretization. Simply provide the discretized gradient and divergence
    Depending on the discretization, the implementation of the rotation matrix (J) and norm should be updated.

    Args:
        in_channels (int): the number of input channels of the features.
        out_channels (int): the number of output channels after the convolution.
        depth (int, optional): the depth of the MLPs (default: 1).
        centralized (bool, optional): centralizes the input features
            before maximum aggregation if set to True (default: False):
            p_j = p_j - p_i.
        vector (bool, optional): determines whether the vector stream is propagated 
            set this to false in the last layer of a network that only outputs scalars (default: True).
        aggr (string, optional): the type of aggregation used in the scalar stream (default: 'max').
    """
    def __init__(self, in_channels, out_channels, depth=1, centralized=False, vector=True, aggr='max'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.centralized = centralized
        self.aggr = aggr

        self.s_mlp_max = MLP([in_channels] + [out_channels] * depth)
        self.s_mlp = MLP([in_channels * 4] + [out_channels] * depth)
        if vector:
            self.v_mlp = VectorMLP([in_channels * 4 + out_channels * 2] + [out_channels] * depth)
        else:
            self.v_mlp = None

    def forward(self, x, v, grad, div, edge_index):

        # Scalar -> Scalar, Vector -> Scalar
        # ----------------------------------

        # Aggregation in scalar stream, defaults to maximum aggregation.
        if self.centralized:
            x_edge = x[edge_index[1]] - x[edge_index[0]]
            x_max = scatter(self.s_mlp_max(x_edge), edge_index[0], dim=0, reduce=self.aggr)
        else:
            x_max = scatter(self.s_mlp_max(x)[edge_index[1]], edge_index[0], dim=0, reduce=self.aggr)

        # Apply operators and concatenate.
        x_cat = torch.cat([x, div @ v, curl(v, div), norm(v)], dim=1)
        # Combine the operators with an MLP.
        x = x_max + self.s_mlp(x_cat)

        # Vector -> Vector, Scalar -> Vector
        # ----------------------------------

        if self.v_mlp is not None:
            # Apply operators and concatenate.
            v_cat = torch.cat([v, hodge_laplacian(v, grad, div), grad @ x], dim=1)
            # Combine the operators and their 90-degree rotated variants (I_J) with an MLP.
            v = self.v_mlp(I_J(v_cat))

        return x, v

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels})'