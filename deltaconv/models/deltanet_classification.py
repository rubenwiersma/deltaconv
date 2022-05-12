import torch
from torch.nn import Sequential as Seq, Dropout, Linear
from torch_geometric.nn import global_max_pool, global_mean_pool

from . import DeltaNetBase 
from ..nn import MLP


class DeltaNetClassification(torch.nn.Module):
    def __init__(self, in_channels, num_classes, conv_channels=[64, 64, 128, 256], num_neighbors=20, grad_regularizer=1e-3, grad_kernel_width=1):
        """Classification of Point Clouds with DeltaConv.
        The architecture is based on the architecture used by DGCNN (https://dl.acm.org/doi/10.1145/3326362.

        Args:
            in_channels (int): the number of channels provided as input.
            num_classes (int): the number of classes to classify.
            conv_channels (list[int]): the number of output channels of each convolution.
            num_neighbors (int): the number of neighbors to use in estimating the gradient.
            grad_regularizer (float): the regularizer value used in the least-squares fitting procedure.
                In the paper, this value is referred to as \lambda.
                Larger grad_regularizer gives a smoother, but less accurate gradient.
                Lower grad_regularizer gives a more accurate, but more variable gradient.
                The grad_regularizer value should be >0 (e.g., 1e-4) to prevent exploding values.
            grad_kernel_width (float): the width of the gaussian kernel used to weight the
                least-squares problem to approximate the gradient.
                Larger kernel width means that more points are included, which is a 'smoother' gradient.
                Lower kernel width gives a more accurate, but possibly noisier gradient.
        """
        super().__init__()
        
        self.deltanet_base = DeltaNetBase(in_channels, conv_channels, 1, num_neighbors, grad_regularizer, grad_kernel_width)

        self.lin_embedding = MLP([sum(conv_channels), 1024])
        self.classification_head = Seq(
            MLP([1024 * 2, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Linear(256, num_classes))


    def forward(self, data):
        conv_out = self.deltanet_base(data)

        x = torch.cat(conv_out, dim=1)
        x = self.lin_embedding(x)

        batch = data.batch
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)

        x = torch.cat([x_max, x_mean], dim=1)

        return self.classification_head(x)