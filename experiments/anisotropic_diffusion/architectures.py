import torch
import torch.linalg as LA
import kornia as K

"""
Reimplementation for images of seven graph/surface/point cloud convolution operators.
The architecture is a ResNet with `num_layers` depth, and `out_channels` width.
"""

class ConvNet(torch.nn.Module):
    """2D image convolutions."""
    def __init__(self, num_layers = 8, out_channels = 32, in_channels = 1):
        super(ConvNet, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.residual_lins = torch.nn.ModuleList()

        in_c = in_channels
        for i in range(num_layers):
            if i == num_layers - 1: out_channels = in_channels
            self.convs.append(torch.nn.Conv2d(in_c, out_channels, kernel_size=3, bias=False, padding=1))

            if in_c != out_channels:
                self.residual_lins.append(torch.nn.Conv2d(in_c, out_channels, kernel_size=1, bias=False))
            else:
                self.residual_lins.append(torch.nn.Identity())
            in_c = out_channels

    @staticmethod
    def name():
        return 'CNN'
    
    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x = torch.nn.functional.leaky_relu(self.residual_lins[i](x) + conv(x), negative_slope=0.2)
        return x


class DeltaNet(torch.nn.Module):
    """DeltaConv operator that is simplified: only the gradient and divergence connections are used."""
    def __init__(self, num_layers = 8, out_channels = 32, in_channels = 1):
        super(DeltaNet, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.v_convs = torch.nn.ModuleList()
        self.v_bias = torch.nn.ParameterList()
        self.residual_lins = torch.nn.ModuleList()

        in_c = in_channels
        for i in range(num_layers):
            if i == num_layers - 1: out_channels = in_channels
            self.convs.append(torch.nn.Conv2d(in_c + out_channels, out_channels, kernel_size=1, bias=False))
            self.v_convs.append(torch.nn.Conv2d(in_c, out_channels, kernel_size=1, bias=False))
            self.v_bias.append(torch.nn.Parameter(torch.Tensor(out_channels)))
            torch.nn.init.uniform_(self.v_bias[-1], -1e-4, 1e-4)
            if in_c != out_channels:
                self.residual_lins.append(torch.nn.Conv2d(in_c, out_channels, kernel_size=1, bias=False))
            else:
                self.residual_lins.append(torch.nn.Identity())
            in_c = out_channels
    
    @staticmethod
    def name():
        return 'DeltaConv'
    
    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x_residual = self.residual_lins[i](x)

            # Connect from scalar to vector stream
            v = K.filters.spatial_gradient(x)

            # Apply MLP to vector features
            v_sh = v.size()
            v = self.v_convs[i](v.reshape(v_sh[0], v_sh[1], v_sh[2] * v_sh[3], v_sh[4]))
            v = v.reshape(v_sh[0], v.size(1), v_sh[2], v_sh[3], v_sh[4])
            v = self.v_nonlin(v, i)

            # Compute divergence
            x_div = K.filters.spatial_gradient(v[:, :, 0])[:, :, 0] + K.filters.spatial_gradient(v[:, :, 1])[:, :, 1]
            x = torch.nn.functional.leaky_relu(x_residual + conv(torch.cat([x, x_div], dim=1)), negative_slope=0.2)

        return x

    def v_nonlin(self, v, i):
        v_mag = LA.norm(v, dim=2, keepdim=True)
        v_nonlin = torch.nn.functional.relu(v_mag + self.v_bias[i][None, :, None, None, None])
        v = v * (v_nonlin / v_mag.clip(1e-5))
        return v


class DiffusionNet(torch.nn.Module):
    """DiffusionNet operator from https://dl.acm.org/doi/full/10.1145/3507905.
    For this experiment, we are interested in the anisotropic features from DiffusionNet
    and thus implement a simplified isotropic diffusion step by explicit time integration."""
    def __init__(self, num_layers = 8, out_channels = 32, in_channels = 1):
        super(DiffusionNet, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.v_convs = torch.nn.ModuleList()
        self.v_bias = torch.nn.ParameterList()
        self.residual_lins = torch.nn.ModuleList()

        in_c = in_channels
        for i in range(num_layers):
            if i == num_layers - 1: out_channels = in_channels
            self.convs.append(torch.nn.Conv2d(in_c * 3, out_channels, kernel_size=1, bias=False))
            self.v_convs.append(torch.nn.Conv2d(in_c, in_c, kernel_size=1, bias=False))
            if in_c != out_channels:
                self.residual_lins.append(torch.nn.Conv2d(in_c, out_channels, kernel_size=1, bias=False))
            else:
                self.residual_lins.append(torch.nn.Identity())
            in_c = out_channels
    
    @staticmethod
    def name():
        return 'DiffusionNet'
    
    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x_residual = self.residual_lins[i](x)

            # Compute the spatial gradient feature from DiffusionNet
            # First compute the gradient
            v = K.filters.spatial_gradient(x)
            v_sh = v.size()

            # Apply a weight matrix that rescales and combines the vector features
            v_mlp = self.v_convs[i](v.reshape(v_sh[0], v_sh[1], v_sh[2] * v_sh[3], v_sh[4]))
            v_mlp = v_mlp.reshape(v_sh[0], v.size(1), v_sh[2], v_sh[3], v_sh[4])

            # And return the scalar product between the vector feature and the combined vector features
            scalar_v = torch.tanh(torch.sum(v * v_mlp, dim=2))

            # Add diffusion for a set timestep, applied explicitly (similar to Table 7 in DiffusionNet paper)
            x_lapl = K.filters.laplacian(x, 3)

            # Combine the gradient features with the diffusion step
            x = torch.nn.functional.leaky_relu(x_residual + conv(torch.cat([x, -x_lapl, scalar_v], dim=1)), negative_slope=0.2)

        return x


class EdgeNet(torch.nn.Module):
    """EdgeConv operator from https://dl.acm.org/doi/10.1145/3326362."""
    def __init__(self, num_layers = 8, out_channels = 32, in_channels = 1):
        super(EdgeNet, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.max_convs = torch.nn.ModuleList()
        self.residual_lins = torch.nn.ModuleList()

        in_c = in_channels
        for i in range(num_layers):
            if i == num_layers - 1: out_channels = in_channels
            self.convs.append(torch.nn.Conv2d(in_c * 2, out_channels, kernel_size=1, bias=False))
            if in_c != out_channels:
                self.residual_lins.append(torch.nn.Conv2d(in_c, out_channels, kernel_size=1, bias=False))
            else:
                self.residual_lins.append(torch.nn.Identity())
            in_c = out_channels
    
    @staticmethod
    def name():
        return 'EdgeConv'
    
    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x_residual = self.residual_lins[i](x)

            # Create 'edges' from pixel grid
            x_sh = x.size()
            x_unfold = torch.nn.functional.unfold(x, kernel_size=3, padding=1).view(x_sh[0], x_sh[1], -1, x_sh[2], x_sh[3])

            # Compute edge feature by taking the difference between the center pixel and its neighbors
            x_edge = torch.cat([x_unfold[:, :, 4:5].expand(-1, -1, 9, -1, -1), x_unfold - x_unfold[:, :, 4:5]], dim=1)
            x_edge = x_edge.view(x_sh[0], x_sh[1] * 2, -1, x_sh[2] * x_sh[3])

            # Apply the MLP and aggregate with maximum aggregation 
            x_max = torch.max(torch.nn.functional.leaky_relu(conv(x_edge), negative_slope=0.2), dim=2)[0].view(x_sh[0], -1, x_sh[2], x_sh[3])

            # Connect the residual
            x = torch.nn.functional.leaky_relu(x_residual + x_max, negative_slope=0.2)

        return x


class PointNet(torch.nn.Module):
    """PointNet++ operator from https://proceedings.neurips.cc/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf"""
    def __init__(self, num_layers = 8, out_channels = 32, in_channels = 1):
        super(PointNet, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.max_convs = torch.nn.ModuleList()
        self.residual_lins = torch.nn.ModuleList()

        in_c = in_channels
        for i in range(num_layers):
            if i == num_layers - 1: out_channels = in_channels
            self.convs.append(torch.nn.Conv2d(in_c, out_channels, kernel_size=1, bias=False))
            if in_c != out_channels:
                self.residual_lins.append(torch.nn.Conv2d(in_c, out_channels, kernel_size=1, bias=False))
            else:
                self.residual_lins.append(torch.nn.Identity())
            in_c = out_channels
    
    @staticmethod
    def name():
        return 'PointNet++'
    
    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x_residual = self.residual_lins[i](x)
            x_sh = x.size()

            # Create edges from pixel grid
            x_unfold = torch.nn.functional.unfold(x, kernel_size=3, padding=1).view(x_sh[0], x_sh[1], -1, x_sh[2], x_sh[3])
            x_edge = x_unfold.view(x_sh[0], x_sh[1], -1, x_sh[2] * x_sh[3])

            # Apply MLP and aggregate with maximum aggregation
            x_max = torch.max(torch.nn.functional.leaky_relu(conv(x_edge), negative_slope=0.2), dim=2)[0].view(x_sh[0], -1, x_sh[2], x_sh[3])

            # Connect the residual 
            x = torch.nn.functional.leaky_relu(x_residual + x_max, negative_slope=0.2)

        return x


class GCN(torch.nn.Module):
    """GCN operator from https://arxiv.org/abs/1609.02907"""
    def __init__(self, num_layers = 8, out_channels = 32, in_channels = 1):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.max_convs = torch.nn.ModuleList()
        self.residual_lins = torch.nn.ModuleList()

        in_c = in_channels
        for i in range(num_layers):
            if i == num_layers - 1: out_channels = in_channels
            self.convs.append(torch.nn.Conv2d(in_c, out_channels, kernel_size=1, bias=False))
            if in_c != out_channels:
                self.residual_lins.append(torch.nn.Conv2d(in_c, out_channels, kernel_size=1, bias=False))
            else:
                self.residual_lins.append(torch.nn.Identity())
            in_c = out_channels
    
    @staticmethod
    def name():
        return 'GCN'
    
    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x_residual = self.residual_lins[i](x)
            x_sh = x.size()

            # Create edges from pixel grid
            x_unfold = torch.nn.functional.unfold(x, kernel_size=3, padding=1).view(x_sh[0], x_sh[1], -1, x_sh[2], x_sh[3])
            x_edge = x_unfold.view(x_sh[0], x_sh[1], -1, x_sh[2] * x_sh[3])

            # Apply weight matrix and average over neighbors
            # Because we are operating on a pixel grid, the degree matrix is a multiple of the identity,
            # so we exclude this component and simply average.
            x_mean = torch.mean(conv(x_edge), dim=2).view(x_sh[0], -1, x_sh[2], x_sh[3])

            # Connect the residual
            x = torch.nn.functional.leaky_relu(x_residual + x_mean, negative_slope=0.2)

        return x
