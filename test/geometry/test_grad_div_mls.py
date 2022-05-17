import torch
import torch.linalg as LA

from torch_geometric.nn import knn_graph

from deltaconv.geometry.grad_div_mls import *
from deltaconv.geometry.utils import batch_dot
from deltaconv.geometry.operators import curl, hodge_laplacian, laplacian, J
from deltaconv.geometry.connection import rotate_around


def test_build_tangent_basis():
    normal = torch.rand(100, 3)
    normal = normal / LA.norm(normal, dim=1, keepdim=True).clamp(1e-8)
    x_basis, y_basis = build_tangent_basis(normal)

    # 1. The basis must be orthonormal
    basis = torch.stack([normal, x_basis, y_basis], dim=-1)
    basisTbasis = torch.bmm(basis.transpose(1, 2), basis)
    identity = torch.stack([torch.eye(3)]*100, 0)
    assert torch.allclose(basisTbasis, identity, atol=1e-7)

    # 2. The basis must be right-handed
    assert (batch_dot(torch.cross(x_basis, y_basis, dim=1), normal) < 0).sum() == 0


def test_estimate_basis():
    # Generate random points in a plane
    pos = torch.cat([torch.rand(100, 2), torch.zeros(100, 1)], dim=1)

    # Generate a random normal 
    normal = torch.rand(1, 3)
    normal = normal / LA.norm(normal, dim=1, keepdim=True).clamp(1e-8)
    # And compute an orthonormal basis around the normal
    xy_basis = build_tangent_basis(normal)
    
    # Transform points with new basis
    T = torch.stack([*xy_basis, normal], dim=-1).squeeze(0)
    pos = pos @ T.T

    # Estimate bases with SVD
    edge_index = knn_graph(pos, 20, loop=True, flow='target_to_source')
    out_normal, out_x_basis, out_y_basis = estimate_basis(pos, edge_index)

    # 1. The basis must be orthonormal
    basis = torch.stack([out_normal, out_x_basis, out_y_basis], dim=-1)
    basisTbasis = torch.bmm(basis.transpose(1, 2), basis)
    identity = torch.stack([torch.eye(3)]*100, 0)
    assert torch.allclose(basisTbasis, identity, atol=1e-5)
    
    # 2. The basis must be right-handed
    assert (batch_dot(torch.cross(out_x_basis, out_y_basis, dim=1), out_normal) < 0).sum() == 0

    # 3. The normal should align with the ground truth normal
    assert torch.allclose(torch.abs((normal * out_normal).sum(dim=1)), torch.ones(100))


def test_coords_projected():
    # Setup a simple surface f(x, y) = [x, y, x^2 + y^2]
    x, y = torch.rand(2, 100) * 2 - 1
    x[0] = y[0] = 0
    z = x**2 + y**2

    pos = torch.stack([x, y, z], dim=1) + torch.rand(3)

    # And rotate the surface in 3D

    # Generate a random normal 
    normal = torch.rand(1, 3)
    normal = normal / LA.norm(normal, dim=1, keepdim=True).clamp(1e-8)
    # And compute an orthonormal basis around the normal
    x_basis, y_basis = build_tangent_basis(normal)
    
    # Transform points with new basis
    T = torch.stack([x_basis, y_basis, normal], dim=-1).squeeze(0)
    pos = pos @ T.T

    # Compute coordinates by projection
    edge_index = knn_graph(pos, 20, loop=True, flow='target_to_source')
    out_coords = coords_projected(pos, normal.repeat(100, 1), x_basis.repeat(100, 1), y_basis.repeat(100, 1), edge_index)

    # 1. The coordinates should be equal to the original x, y coordinates
    true_coords = torch.stack([x[edge_index[1, :20]], y[edge_index[1, :20]]], dim=1)
    assert torch.allclose(out_coords[:20], true_coords)


def test_gaussian_weights():
    # Random distances
    dist = torch.rand(1000)

    out_weights = gaussian_weights(dist, 20)

    # 1. No NaNs
    assert torch.isnan(out_weights).sum() == 0
    # 2. Sum of weights is 1 for every neighborhood
    assert torch.allclose(out_weights.view(-1, 20).sum(dim=1), torch.ones(50))

    # 3. Points with closer distance have higher weight
    dist = torch.Tensor([0.1, 0.5, 1., 1.5, 2.])
    out_weights = gaussian_weights(dist, 5)
    assert out_weights[0] > out_weights[1]
    assert out_weights[1] > out_weights[2]
    assert out_weights[2] > out_weights[3]
    assert out_weights[3] > out_weights[4]


def test_weighted_least_squares():
    N = 1000
    k = 20
    # Setup a simple surface f(x, y) = [x, y, c0 + c1*x + c2*y + c3*x**2 + c4*xy + c5*y**2]
    coords = torch.rand(N, k, 2) * 2 - 1 
    # Always add the center point
    coords[:, 0] = 0
    coords = coords.view(N * k, 2)

    # Compute XTX, so we can create a quadratic function
    coords_const = torch.cat([coords.new_ones(coords.size(0), 1), coords], dim=1)
    B = torch.bmm(coords_const.unsqueeze(-1), coords_const.unsqueeze(-2))
    triu = torch.triu_indices(3, 3)
    B = B[:, triu[0], triu[1]]
    B = B.view(-1, k, 6) # [1, x, y, x**2, xy, y**2]

    # Set random coefficients
    coefficients = torch.rand(N, 6)
    # And compute dummy function
    f = (B * coefficients.unsqueeze(1)).sum(dim=-1, keepdim=True) # [N, k, 1]

    dist = LA.norm(coords, dim=1)
    weights = gaussian_weights(dist, k)
    out_wls = weighted_least_squares(coords, weights, k, 0)
    
    out_coefficients = (out_wls.view(N, k, 6) * f).sum(dim=1)
    # 1. The recovered coefficients should be equal to the actual coefficients
    assert torch.allclose(out_coefficients, coefficients, atol=1e-3)

    # 2. The coefficients should be close when the regularizer is used
    out_wls = weighted_least_squares(coords, weights, k, 1e-5)
    out_coefficients = (out_wls.view(N, k, 6) * f).sum(dim=1)
    assert torch.allclose(out_coefficients, coefficients, atol=5e-2)

    # 3. If we add noise to the function f, we want the derived coefficients to remain close
    f_noise = f + torch.rand(N, k, 1) * 0.01 - 0.005
    out_coefficients = (out_wls.view(N, k, 6) * f_noise).sum(dim=1)
    assert torch.allclose(out_coefficients, coefficients, atol=1e-1)
    # On average the error is < 0.05
    assert torch.abs(out_coefficients - coefficients).mean() < 5e-2

    # 4. If we add outliers to the function f, we want the derived coefficients to remain close
    # Outliers are added with a 5% chance
    f_noise = f + (torch.rand(N, k, 1) > 0.95) * torch.rand(N, k, 1) * 0.1
    out_coefficients = (out_wls.view(N, k, 6) * f_noise).sum(dim=1)
    # The error is bounded by a value < 0.5
    assert torch.allclose(out_coefficients, coefficients, atol=5e-1)
    # On average the error is close < 0.05
    assert torch.abs(out_coefficients - coefficients).mean() < 5e-2


def test_fit_vector_mapping():
    N = 1000
    k = 20
    # Testing strategy:
    # -----------------
    # 1. Create N separate patches with k points each.
    #    Each patch is a surface randomly sampled from a quadratic polynomial with random coefficients.
    # 2. We setup a basis for each point in the patch and rotate it.
    # 3. Then we check that the fit_vector_mapping function correctly transforms from
    #    a neighboring basis to the center points' basis.
    # 4. If the basis is transformed correctly, that means any vector expressed 
    #    in this basis will also be transformed correctly.

    # Setup a simple surface f(x, y) = [x, y, c0 * x**2 + c1 * xy + c2 * y**2]
    # for all N patches. In the comments, we refer to coords as X.
    # ------------------
    coords = torch.rand(N, k, 2) * 2 - 1 
    # Always add the center point
    coords[:, 0] = 0

    # Set random coefficients
    coefficients = torch.rand(N, 3)
    # And compute dummy function
    x = coords[..., 0]
    y = coords[..., 1]
    f = coefficients[:, None, 0] * x**2 + coefficients[:, None, 1] * x * y + coefficients[:, None, 2] * y ** 2
    coords = coords.view(-1, 2)

    # Assemble positions
    pos = torch.cat([coords, f.view(-1, 1)], dim=1)

    # Compute basis and randomize in-plane rotation
    # ---------------------------------------------

    # Setup basis for each point
    # df/dx = [1, 0, 2*c0*x + c1*y]
    dfdx_z = 2 * coefficients[:, None, 0] * x + coefficients[:, None, 1] * y
    dfdx = torch.stack([
        torch.ones_like(coords[:, 0]), 
        torch.zeros_like(coords[:, 0]), 
        dfdx_z.flatten()
    ], dim=1) # [N * k, 3]
    # df/dy = [1, 0, c1*x + 2*c2*y]
    dfdy_z = coefficients[:, None, 1] * x + 2 * coefficients[:, None, 2] * y
    dfdy = torch.stack([
        torch.zeros_like(coords[:, 0]), 
        torch.ones_like(coords[:, 0]), 
        dfdy_z.flatten()
    ], dim=1) # [N * k, 3]
    # Normal is dfdx x dfdy
    normal = torch.cross(dfdx, dfdy, dim=1) # [N * k, 3]
    normal = normal / LA.norm(normal, dim=1, keepdim=True).clamp(1e-8)

    # Add a random rotation to each basis
    # Mix the x- and y-basis with random, non-zero weights
    weights = torch.rand(N * k, 2) + 1e-2 # [2, N * k]
    # Flip some weights
    weights[:, 0] = torch.where(torch.rand(N * k) > 0.5, weights[:, 0], -weights[:, 0])
    weights[:, 1] = torch.where(torch.rand(N * k) > 0.5, weights[:, 1], -weights[:, 1])
    # Normalize weights
    weights = weights / LA.norm(weights, dim=1, keepdim=True).clamp(1e-8)
    # Don't change center points
    weights = weights.view(N, k, 2)
    weights[:, 0] = weights.new_tensor([1, 0])
    weights = weights.view(N * k, 2)
    # Mix x and y basis
    x_basis = weights[:, 0:1] * dfdx + weights[:, 1:] * dfdy
    x_basis = x_basis / LA.norm(x_basis, dim=1, keepdim=True).clamp(1e-8)
    # Recompute y-basis with cross-product between x-basis and normal
    y_basis = torch.cross(normal, x_basis)

    #       
    #    ((
    #  c|  |
    #   |__|
    #  
    # Hey, nice to see you here! Glad to see someone reads tests :)
    # Come say hi on twitter (@rtwiersma)
    # As a token of my appreciation, here's a way to easily visualize the whole test surface we just made
    # first, install polyscope (pip install polyscope) then uncomment the next few lines
    # and make sure you set N=1 at the start of this test.
    # ----------------------
    # import polyscope as ps
    # ps.init()
    # cloud = ps.register_point_cloud('cloud', pos)
    # cloud.add_vector_quantity('dfdx', dfdx, enabled=True)
    # cloud.add_vector_quantity('dfdy', dfdy)
    # cloud.add_vector_quantity('normal', normal, enabled=True)
    # cloud.add_vector_quantity('x_basis', x_basis, enabled=False)
    # cloud.add_vector_quantity('y_basis', y_basis, enabled=False)
    # ps.set_ground_plane_mode('none')
    # ps.show()

    # Compute the vector mapping
    # --------------------------
    edge_index = torch.stack([
        torch.arange(N).repeat_interleave(k) * k,
        torch.arange(N * k)
    ], dim=0)
    # The WLS and fit_vector_mapping code assumes that we projected
    # the coordinates of each neighbor to the tangent plane of the center point.
    dist = LA.norm(coords, dim=1)
    wls_weights = gaussian_weights(dist, k)
    wls = weighted_least_squares(coords, wls_weights, k, regularizer=0)

    out_vector_mapping = fit_vector_mapping(pos, normal, x_basis, y_basis, edge_index, wls, coords)

    # 0. Check some preliminaries
    assert out_vector_mapping.size() == (N * k, 2, 2)
    assert out_vector_mapping.isnan().sum() == 0

    # The vector fitting procedure solves
    # a_0^x dfdx|j + a_0^y dfdy|j = a_j^x e_j^x + a_j^y e_j^y
    # So, for a_j = [1, 0], we want [dfdx dfdy] a_0 == e_j^x
    # and for a_j = [0, 1], we want [dfdx dfdy] a_0 == e_j^y
    assert torch.allclose(out_vector_mapping[:, None, 0, 0] * dfdx + out_vector_mapping[:, None, 1, 0] * dfdy, x_basis, atol=1e-6)
    assert torch.allclose(out_vector_mapping[:, None, 0, 1] * dfdx + out_vector_mapping[:, None, 1, 1] * dfdy, y_basis, atol=1e-6)
    # That's it!

    
def test_build_grad_div():
    # Testing strategy
    # ----------------
    # 1. Sample points from a parametric surface.
    # 2. Compute analytical derivatives.
    # 3. Compare result of gradient matrix with analytical derivatives.
    N = 1000
    k = 20

    torch.manual_seed(42)

    # Setup a simple surface f(x, y) = [x, y, c0 * x**2 + c1 * xy + c2 * y**2]
    coords = torch.rand(N, 2) * 2 - 1

    # Compute XTX, so we can create a quadratic function
    coords_const = torch.cat([coords.new_ones(coords.size(0), 1), coords], dim=1)
    B = torch.bmm(coords_const.unsqueeze(-1), coords_const.unsqueeze(-2))
    triu = torch.triu_indices(3, 3)
    B = B[:, triu[0], triu[1]]
    B = B.view(-1, 6) # [1, x, y, x**2, xy, y**2]

    # Set random coefficients
    coefficients = torch.rand(6)
    # And compute dummy function
    f = (B * coefficients.unsqueeze(0)).sum(dim=1, keepdim=True) # [N, 1]

    # Simple surface with boundary
    pos = torch.cat([coords, f], dim=1)

    # Compute coordinate frame
    x, y = coords.T
    dfdx_z = coefficients[1] + 2 * coefficients[3] * x + coefficients[4] * y
    dfdx = torch.stack([
        torch.ones_like(coords[:, 0]), 
        torch.zeros_like(coords[:, 0]), 
        dfdx_z.flatten()
    ], dim=1) # [N * k, 3]
    dfdy_z = coefficients[2] + coefficients[4] * x + 2 * coefficients[5] * y
    dfdy = torch.stack([
        torch.zeros_like(coords[:, 0]), 
        torch.ones_like(coords[:, 0]), 
        dfdy_z.flatten()
    ], dim=1) # [N * k, 3]
    # Normal is dfdx x dfdy
    normal = torch.cross(dfdx, dfdy, dim=1) # [N * k, 3]
    normal = normal / LA.norm(normal, dim=1, keepdim=True).clamp(1e-8)
    # Normalize x_basis
    x_basis = dfdx / LA.norm(dfdx, dim=1, keepdim=True).clamp(1e-8)
    y_basis = torch.cross(normal, x_basis)

    edge_index = knn_graph(pos, k, loop=True, flow='target_to_source')
    out_grad, out_div = build_grad_div(pos, normal, x_basis, y_basis, edge_index, regularizer=1e-8, normalized=False)

    # 1. Size checking
    # ----------------
    # 1a. Grad size must be [N*2, N]
    assert out_grad.size(0) == N*2
    assert out_grad.size(1) == N
    # 1b. Div size must be [N, N*2]
    assert out_div.size(0) == N
    assert out_div.size(1) == N*2

    # 2. Checking output for NaNs
    # ---------------------------
    # 2a. We shouldn't get NaNs from applying grad 
    assert (out_grad @ torch.rand(N, 1)).isnan().sum() == 0
    # 2b. No NaNs from applying div
    assert (out_div @ torch.rand(N*2, 1)).isnan().sum() == 0

    # 3. De Rham complex properties
    # -----------------------------
    # 3a. The result of applying grad to a constant function should be 0
    assert torch.allclose(out_grad @ pos.new_ones(N, 1), pos.new_zeros(N*2, 1), atol=1e-2)
    # 3b. The result of applying laplacian (div grad) to a constant function should be 0
    # We check this with the normalized L1 norm (mean of absolute values)
    assert torch.abs(laplacian(pos.new_ones(N, 1), out_grad, out_div)).mean() < 1e-2
    # 3c. The L1 norm of div grad x should be > 0 for a random function
    assert LA.norm(laplacian(torch.rand(N, 1), out_grad, out_div), ord=1) > 0
    # 3d. Applying curl grad x should return all 0 for any function
    # We will have some non-zero outliers, e.g., on bounaries, so we'll check the mean
    # of the squared L2 norm (emphasizes lower values).
    assert torch.pow(curl(out_grad @ pos[:, 0:1], out_div), 2).mean() < 1e-2
    assert torch.pow(curl(out_grad @ pos[:, 0:1], out_div), 2).median() < 1e-2
    # 3e. Applying div co-grad x should return all 0 for any function
    assert torch.pow(out_div @ J(out_grad @ pos[:, 0:1]), 2).mean() < 1e-2
    assert torch.pow(out_div @ J(out_grad @ pos[:, 0:1]), 2).median() < 1e-2

    # 4. Correct gradients/divergences
    # --------------------------------
    # We set up f as a height map over the x, y plane.
    # Therefore, the gradient in the x- and y-direction of f
    # should be the projection of [0, 0, 1] onto the two tangent vectors.
    grad_x_f, grad_y_f = (out_grad @ f).view(N, 2).T
    assert torch.allclose(grad_x_f, x_basis[:, 2], atol=1e-2)
    assert torch.allclose(grad_y_f, y_basis[:, 2], atol=1e-2)

    # Applying div grad to the point positions should
    # result in the mean curvature vector, pointing roughly along the normal.
    mean_curvature = laplacian(pos, out_grad, out_div)
    assert torch.allclose(-batch_dot(mean_curvature, normal), LA.norm(mean_curvature, dim=1, keepdims=True), atol=1e-2)

    # 5. Coordinate equivariance of gradient/divergence
    # -------------------------------------------------
    # Reuse the random point cloud from before
    edge_index = knn_graph(pos, k, loop=True, flow='target_to_source')

    x_basis_rot = rotate_around(x_basis, normal, torch.rand(N) * 2 * torch.pi)
    y_basis_rot = torch.cross(normal, x_basis_rot)

    # Compute gradient and divergence on random choices of basis
    G, D = build_grad_div(pos, normal, x_basis, y_basis, edge_index, regularizer=1e-8)
    G_rot, D_rot = build_grad_div(pos, normal, x_basis_rot, y_basis_rot, edge_index, regularizer=1e-8)

    # 5a. grad x should be grad_rotated x in ambient space for any x
    # G = grad, Gr = grad_rotated
    x = torch.rand(N, 1)
    G_x_x, G_x_y = (G @ x).view(-1, 2).T
    ambient_G_x = G_x_x.unsqueeze(-1) * x_basis + G_x_y.unsqueeze(-1) * y_basis
    G_rot_x_x, G_rot_x_y = (G_rot @ x).view(-1, 2).T
    ambient_G_rot_x = G_rot_x_x.unsqueeze(-1) * x_basis_rot + G_rot_x_y.unsqueeze(-1) * y_basis_rot
    assert torch.allclose(ambient_G_x, ambient_G_rot_x, atol=1e-3)

    # 5b. div grad x should be exactly the same as div_rotated grad_rotated x
    assert torch.allclose(D @ (G @ x), D_rot @ (G_rot @ x), atol=1e-3)


