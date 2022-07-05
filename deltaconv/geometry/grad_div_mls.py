import torch
import torch.linalg as LA
from torch_sparse import SparseTensor
from torch_scatter import scatter_add, scatter_max, scatter_mean
from .utils import batch_dot

EPS = 1e-5


def estimate_basis(pos, edge_index, k=None, orientation=None):
    """Estimates a tangent basis for each point, given a k-nn graph and positions.
    Note: this function is only faster if used in batch mode on the GPU.
    Use pointcloud-ops when applying transforms on the CPU.

    Args:
        pos (Tensor): an [N, 3] tensor with the point positions.
        edge_index (Tensor): indices of the adjacency matrix of the k-nn graph [2, N * k].
        k (int, optional): the number of neighbors per point,
            is derived from edge_index when no k is provided (default: None).
        orientation (Tensor, optional): an [N, 3] tensor with a rough direction of the normal to
            orient the estimated normals.
    """
    row, col = edge_index
    k = (row == 0).sum() if k is None else k
    row, col = row.view(-1, k), col.view(-1, k)
    local_pos = (pos[col] - pos[row]).transpose(-2, -1)
    
    # SVD to estimate bases
    svd = LA.svd(local_pos)
    
    # Normal corresponds to smallest singular vector and normalize
    normal = svd.U[:, :, 2]
    normal = normal / LA.norm(normal, dim=-1, keepdim=True).clamp(EPS)

    # If normals are given, orient using the given normals
    if orientation is not None:
        normal = torch.where(torch.bmm(normal.unsqueeze(1), orientation.unsqueeze(-1)).squeeze(-1) < 0, -normal, normal)

    # X axis to largest singular vector and normalize
    x_basis = svd.U[:, :, 0]
    x_basis = x_basis / LA.norm(x_basis, dim=-1, keepdim=True).clamp(EPS)
    
    # Create orthonormal basis by taking cross product
    y_basis = torch.cross(normal, x_basis)
    y_basis = y_basis / LA.norm(y_basis, dim=-1, keepdim=True).clamp(EPS)
    
    return normal, x_basis, y_basis


def build_tangent_basis(normal):
    """Constructs an orthonormal tangent basis, given a normal vector.

    Args:
        normal (Tensor): an [N, 3] tensor with normals per point.
    """

    # Pick an arbitrary basis vector that does not align too much with the normal
    testvec = normal.new_tensor([[1, 0, 0]]).expand(normal.size(0), 3)
    testvec_alt = normal.new_tensor([[0, 1, 0]]).expand(normal.size(0), 3)
    testvec = torch.where(torch.bmm(normal.unsqueeze(1), testvec.unsqueeze(-1)).squeeze(-1).abs() > 0.9, testvec_alt, testvec)

    # Derive x basis using cross product and normalize
    x_basis = torch.cross(testvec, normal)
    x_basis = x_basis / LA.norm(x_basis, dim=-1, keepdim=True).clamp(EPS)

    # Derive y basis using cross product and normalize
    y_basis = torch.cross(normal, x_basis)
    y_basis = y_basis / LA.norm(y_basis, dim=-1, keepdim=True).clamp(EPS)
    return x_basis, y_basis


def coords_projected(pos, normal, x_basis, y_basis, edge_index, k=None):
    """Projects neighboring points to the tangent basis
    and returns the local coordinates.

    Args:
        pos (Tensor): an [N, 3] tensor with the point positions.
        normal (Tensor): an [N, 3] tensor with normals per point.
        x_basis (Tensor): an [N, 3] tensor with x basis per point.
        y_basis (Tensor): an [N, 3] tensor with y basis per point.
        edge_index (Tensor): indices of the adjacency matrix of the k-nn graph [2, N * k].
        k (int): the number of neighbors per point.
    """
    row, col = edge_index
    k = (row == 0).sum() if k is None else k

    # Compute coords
    normal = normal.unsqueeze(1).expand(-1, k, -1).reshape(-1, 3)
    x_basis = x_basis.unsqueeze(1).expand(-1, k, -1).reshape(-1, 3)
    y_basis = y_basis.unsqueeze(1).expand(-1, k, -1).reshape(-1, 3)
    local_pos = pos[col] - pos[row]
    local_pos = local_pos - normal * torch.bmm(local_pos.unsqueeze(1), normal.unsqueeze(-1)).squeeze(-1)
    x_pos = torch.bmm(local_pos.unsqueeze(1), x_basis.unsqueeze(-1)).flatten()
    y_pos = torch.bmm(local_pos.unsqueeze(1), y_basis.unsqueeze(-1)).flatten()
    coords = torch.stack([x_pos, y_pos], dim=1)

    return coords


def gaussian_weights(dist, k, batch=None, kernel_width=1):
    """Computes gaussian weights per edge and normalizes the sum per neighborhood.

    Args:
        dist (Tensor): an [N * k] tensor with the geodesic distance of each edge.
        k (int): the number of neighbors per point.
        batch (Tensor, optional): an [N] tensor denoting which batch each shape belongs to (default: None).
        kernel_width (float, optional): the size of the kernel,
            relative to the average edge length in each shape (default: 1).
    """
    batch = batch if batch is not None else torch.zeros(dist.size(0) // k).long()
    dist = dist.view(-1, k)
    avg_dist = scatter_mean(dist.mean(dim=1, keepdim=True), batch, dim=0)[batch]
    weights = torch.exp(- dist.pow(2) / (kernel_width * avg_dist).pow(2))
    weights = weights / weights.sum(dim=1, keepdim=True).clamp(EPS)
    
    return weights.flatten()


def weighted_least_squares(coords, weights, k, regularizer, shape_regularizer=None):
    """Solves a weighted least squares equation (see http://www.nealen.net/projects/mls/asapmls.pdf).
    In practice, we compute the inverse of the left-hand side of a weighted-least squares problem:
        B^TB c = B^Tf(x).

    This inverse can be multiplied with the right hand side to find the coefficients
    of a second order polynomial that approximates f(x).
        c = (BTB)^-1 B^T f(x).
    
    The weighted least squares problem is regularized by adding a small value \lambda
    to the diagonals of the matrix on the left hand side of the equation:
        B^TB + \lambda I.
    """
    # Setup polynomial basis
    coords_const = torch.cat([coords.new_ones(coords.size(0), 1), coords], dim=1)
    B = torch.bmm(coords_const.unsqueeze(-1), coords_const.unsqueeze(-2))
    triu = torch.triu_indices(3, 3)
    B = B[:, triu[0], triu[1]]
    B = B.view(-1, k, 6) # [1, x, y, x**2, xy, y**2]

    # Compute weighted least squares
    lI = regularizer * torch.eye(6, 6, device=B.device).unsqueeze(0)
    BT = (weights.view(-1, k, 1) * B).transpose(-2, -1)
    BTB = torch.bmm(BT, B) + lI
    BTB_inv = LA.inv(BTB)
    wls = torch.bmm(BTB_inv, BT).transpose(-2, -1).reshape(-1, 6)

    if shape_regularizer is not None:
        lI = shape_regularizer * torch.eye(6, 6, device=B.device).unsqueeze(0)
        BTB = torch.bmm(BT, B) + lI
        BTB_inv = LA.inv(BTB)
        wls_shape = torch.bmm(BTB_inv, BT).transpose(-2, -1).reshape(-1, 6)
        return wls, wls_shape
    return wls


def fit_vector_mapping(pos, normal, x_basis, y_basis, edge_index, wls, coords):
    """Finds the transformation between a basis at point pj
    and the basis at point pi pushed forward to pj.

    See equation (15) in the supplement of DeltaConv for more details.
    """
    row, col = edge_index

    # Compute the height over the patch by projecting the relative positions onto the normal
    patch_f = batch_dot(normal[row], pos[col] - pos[row])   
    coefficients = scatter_add(wls * patch_f, row, dim=0)

    # Equation (3) and (4) from supplement
    h_x = (coefficients[row, 1] + 2 * coefficients[row, 3] * coords[:, 0] + coefficients[row, 4] * coords[:, 1])
    h_y = (coefficients[row, 2] + coefficients[row, 4] * coords[:, 0] + 2 * coefficients[row, 5] * coords[:, 1])

    # Push forward bases to p_j
    # In equation (15): \partial_u \Gamma(u_j, v_j)
    gamma_x = x_basis[row] + normal[row] * h_x.unsqueeze(-1)
    # In equation (15): \partial_v \Gamma(u_j, v_j)
    gamma_y = y_basis[row] + normal[row] * h_y.unsqueeze(-1)

    # Determine inverse metric for mapping
    # Inverse metric is given in equation (9) of supplement
    det_metric = (1 + h_x.pow(2) + h_y.pow(2))
    E, F, G = 1 + h_x.pow(2), h_x * h_y, 1 + h_y.pow(2)
    inverse_metric = torch.stack([
        G, -F,
        -F, E
    ], dim=-1).view(-1, 2, 2)
    inverse_metric = inverse_metric / det_metric.view(-1, 1, 1)
    basis_transformation = torch.cat([
        batch_dot(gamma_x, x_basis[col]),
        batch_dot(gamma_x, y_basis[col]),
        batch_dot(gamma_y, x_basis[col]),
        batch_dot(gamma_y, y_basis[col])
    ], dim=1).view(-1, 2, 2)
    
    # Compute mapping of vectors
    return torch.bmm(inverse_metric, basis_transformation) # [N, 2, 2]


def build_grad_div(pos, normal, x_basis, y_basis, edge_index, batch=None, kernel_width=1, regularizer=0.001, normalized=True, shape_regularizer=None):
    """Builds a gradient and divergence operators using Weighted Least Squares (WLS).
    Note: this function is only faster if used on the GPU.
    Use pointcloud-ops when applying transforms on the CPU.

    Args:
        pos (Tensor): an [N, 3] tensor with the point positions.
        normal (Tensor): an [N, 3] tensor with normals per point.
        x_basis (Tensor): an [N, 3] tensor with x basis per point.
        y_basis (Tensor): an [N, 3] tensor with y basis per point.
        edge_index (Tensor): indices of the adjacency matrix of the k-nn graph [2, N * k].
        batch (Tensor): an [N] tensor denoting which batch each shape belongs to (default: None).
        kernel_width (float, optional): the size of the kernel,
            relative to the average edge length in each shape (default: 1).
        regularizer (float: optional): the regularizer parameter
            for weighted least squares fitting (default: 0.001).
        normalized (bool: optional): Normalizes the operators by the
            infinity norm if set to True (default: True):
            G = G / |G|_{\inf}
        shape_regularizer (float: optional): sets the regularizer parameter
            for weighted least squares fitting of the surface, rather than the signal on the surface.
            By default, this is set to None and the same value is used for the surface and the signal.
    """

    if batch is None:
        batch = pos.new_zeros(pos.size(0)).long()
    row, col = edge_index
    k = (row == 0).sum()

    edge_mask = col != pos.size(0)
    col[edge_mask.logical_not()] = 0

    # Get coordinates in tangent plane by projecting along the normal of the plane
    coords = coords_projected(pos, normal, x_basis, y_basis, edge_index, k)

    # Compute weights based on distance in euclidean space
    dist = LA.norm(pos[col] - pos[row], dim=1)
    weights = gaussian_weights(dist, k, batch, kernel_width)
    weights[edge_mask.logical_not()] = 0

    # Get weighted least squares result
    # wls multiplied with a function f at k neighbors will give the coefficients c0-c5
    # for the surface f(x, y) = [x, y, c0 + c1*x + c2*y + c3*x**2 + c4*xy + c5*y**2]
    # defined on a neighborhood of each point.
    if shape_regularizer is None:
        wls = weighted_least_squares(coords, weights, k, regularizer)
    else:
        wls, wls_shape = weighted_least_squares(coords, weights, k, regularizer, shape_regularizer)

    # Format as sparse matrix

    # The gradient of f at (0, 0) will be
    # df/dx|(0, 0) = [1, 0, c1 + 2*c3*0 + c4*0] = [1, 0, c1]
    # df/dy|(0, 0) = [0, 1, c2 + c4*0 + 2*c5*0] = [0, 1, c2]
    # Hence, we can use the row in wls that outputs c1 and c2 for the gradient
    # in x direction and y direction, respectively
    grad_row = torch.stack([row[edge_mask] * 2, row[edge_mask] * 2 + 1], dim=1).flatten()
    grad_col = torch.stack([col[edge_mask]]*2, dim=1).flatten()
    grad_values = torch.stack([wls[edge_mask, 1], wls[edge_mask, 2]], dim=1).flatten()

    # Normalize
    if normalized:
        infinity_norm = scatter_max(LA.norm(scatter_add(torch.abs(grad_values), grad_row, dim=0).view(-1, 2), dim=1), batch)[0]
        grad_values = torch.where(torch.repeat_interleave(infinity_norm[batch], 2)[grad_row] > 1e-5, grad_values / torch.repeat_interleave(infinity_norm[batch], 2)[grad_row], grad_values)

    # Create gradient matrix
    grad = SparseTensor(row=grad_row, col=grad_col, value=grad_values, sparse_sizes=(pos.size(0) * 2, pos.size(0)))

    # Divergence
    if shape_regularizer is not None:
        wls = wls_shape
    vector_mapping = fit_vector_mapping(pos, normal, x_basis, y_basis, (row, col), wls, coords)

    # Store as sparse tensor 
    grad_vec = grad_values.view(-1, 1, 2)
    div_vec = torch.bmm(grad_vec, vector_mapping[edge_mask]).flatten()
    div_row = torch.stack([row[edge_mask]] * 2, dim=1).flatten()
    div_col = torch.stack([col[edge_mask] * 2, col[edge_mask] * 2 + 1], dim=1).flatten()
    div = SparseTensor(row=div_row, col=div_col, value=div_vec, sparse_sizes=(pos.size(0), pos.size(0) * 2))

    return grad, div
