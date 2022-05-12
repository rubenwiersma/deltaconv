import torch
from torch_geometric.nn import knn_graph

from ..nn import DeltaConv, MLP
from ..geometry.grad_div import build_grad_div, build_tangent_basis, estimate_basis
    

class DeltaNetBase(torch.nn.Module):
    def __init__(self, in_channels, conv_channels, mlp_depth, num_neighbors, grad_regularizer, grad_kernel_width, centralize_first=True):
        """Classification of Point Clouds with DeltaConv.
        The architecture is based on the architecture used by DGCNN (https://dl.acm.org/doi/10.1145/3326362.

        Args:
            in_channels (int): the number of channels provided as input.
            conv_channels (list[int]): the number of output channels of each convolution.
            mlp_depth (int): the depth of the MLPs of each convolution.
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
            centralize_first (bool, optional): whether to centralize the input features (default: True).
        """
        super().__init__()
        self.k = num_neighbors
        self.grad_regularizer = grad_regularizer
        self.grad_kernel_width = grad_kernel_width

        # Create convolution layers
        conv_channels = [in_channels] + conv_channels
        self.convs = torch.nn.ModuleList()
        for i in range(len(conv_channels) - 1):
            last_layer = i == (len(conv_channels) - 2)
            self.convs.append(DeltaConv(conv_channels[i], conv_channels[i + 1], depth=mlp_depth, centralized=(centralize_first and i == 0), vector=not(last_layer))) 


    def forward(self, data):
        pos = data.pos
        batch = data.batch

        # Operator construction
        # ---------------------

        # Create a kNN graph, which is used to:
        # 1) Perform maximum aggregation in the scalar stream.
        # 2) Approximate the gradient and divergence oeprators
        edge_index = knn_graph(pos, self.k, batch, loop=True, flow='target_to_source')

        # Use the normals provided by the data or estimate a normal from the data.
        #   It is advised to estimate normals as a pre-transform.

        # Note: the x_basis and y_basis are referred to in the DeltaConv paper as e_u, and e_v, respectively.
        # Wherever x and y are used to denote tangential coordinates, they can be interchanged with u and v. 
        if hasattr(data, 'norm') and data.norm is not None:
            normal = data.norm
            x_basis, y_basis = build_tangent_basis(normal)
        else:
            edge_index_normal = knn_graph(pos, 10, batch, loop=True, flow='target_to_source')
            # When normal orientation is unknown, we opt for a locally consistent orientation.
            normal, x_basis, y_basis = estimate_basis(pos, edge_index_normal, orientation=pos)

        # Build the gradient and divergence operators.
        # grad and div are two sparse matrices in the form of SparseTensor.
        grad, div = build_grad_div(pos, normal, x_basis, y_basis, edge_index, batch, kernel_width=self.grad_kernel_width, regularizer=self.grad_regularizer)

        
        # Forward pass convolutions
        # ---------------------------------

        # The scalar features are stored in x
        x = pos
        # Vector features in v
        v = grad @ x
        
        # Store each of the interim outputs in a list
        out = []
        for conv in self.convs:
            x, v = conv(x, v, grad, div, edge_index)
            out.append(x)

        # Return the interim outputs
        return out