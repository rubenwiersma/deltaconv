import torch
from torch.nn import Sequential as Seq, Dropout, LeakyReLU, Linear
from torch_geometric.nn import global_max_pool 

from . import DeltaNetBase
from ..nn import MLP


class DeltaNetSegmentation(torch.nn.Module):
    def __init__(self, in_channels, num_classes, conv_channels=[64, 128, 256], mlp_depth=2, embedding_size=1024, categorical_vector=False, num_neighbors=20, grad_regularizer=0.001, grad_kernel_width=1):
        """Segmentation of Point Clouds with DeltaConv.
        The architecture is based on the architecture used by DGCNN (https://dl.acm.org/doi/10.1145/3326362.

        Args:
            in_channels (int): the number of channels provided as input.
            num_classes (int): the number of classes to segment.
            conv_channels (list[int]): the number of output channels of each convolution.
            mlp_depth (int): the depth of the MLPs of each convolution.
            embedding_size (int): the embedding size before the segmentation head is applied.
            categorical_vector(bool, optional): whether to use the categorical encoding in the model.
                Many authors use this in their models for ShapeNet.
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

        self.categorical_vector = categorical_vector

        self.deltanet_base = DeltaNetBase(in_channels, conv_channels, mlp_depth, num_neighbors, grad_regularizer, grad_kernel_width)

        # Global embedding
        self.lin_global = MLP([sum(conv_channels), embedding_size])

        # For ShapeNet segmentation, most authors add an embedding of the category to aid with segmentation.
        if categorical_vector:
            self.lin_categorical = MLP([16, 64])
            self.segmentation_head = Seq(
                MLP([embedding_size + sum(conv_channels) + 64, 256]), Dropout(0.5), MLP([256, 256]), Dropout(0.5),
                Linear(256, 128), LeakyReLU(negative_slope=0.2), Linear(128, num_classes))
        else:
            self.segmentation_head = Seq(
                MLP([embedding_size + sum(conv_channels), 256]), Dropout(0.5), MLP([256, 256]), Dropout(0.5),
                Linear(256, 128), LeakyReLU(negative_slope=0.2), Linear(128, num_classes))


    def forward(self, data):
        conv_out = self.deltanet_base(data)

        x = torch.cat(conv_out, dim=1)
        x = self.lin_global(x)

        batch = data.batch
        x_max = global_max_pool(x, batch)[batch]

        if self.categorical_vector:
            cat = self.lin_categorical(data.category)[batch]
            x_max = torch.cat([x_max, cat], dim=1)

        x = torch.cat([x_max] + conv_out, dim=1)

        return self.segmentation_head(x)