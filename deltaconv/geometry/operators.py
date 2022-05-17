import torch
import torch.linalg as LA

def norm(v):
    """Computes the norm of a vector field."""
    _, C = v.size()
    return LA.norm(v.view(-1, 2, C), dim=1)

def J(v):
    """Rotates a vector field by 90-degrees counter-clockwise."""
    N, C = v.size()
    v = v.view(-1, 2, C)
    J_v = torch.zeros_like(v)
    J_v[:, 0] = -v[:, 1]
    J_v[:, 1] = v[:, 0]
    J_v = J_v.view(N, C)
    return J_v

def I_J(v):
    """Concatenates a vector field and its 90-degree rotated counterpart."""
    return torch.cat([v, J(v)], dim=1)

def curl(v, div):
    """Computes the curl of a vector field using divergence:
    curl = - div J V.
    """
    return - (div @ J(v))

def laplacian(x, grad, div):
    """Computes the laplacian of a function using gradient and divergence:
    laplacian = - div grad X.
    """
    return - (div @ (grad @ x))

def hodge_laplacian(v, grad, div):
    """Computes the Hodge-Laplacian of a vector field using gradient and divergence:
    hodge-laplacian = - (grad div + J grad curl) V.
    """
    # Compute - G G.T v (grad div)
    grad_div_v = grad @ (div @ v)

    # Compute J G G.T J v (J grad curl)
    J_grad_curl_v = J(grad @ curl(v, div))

    # Combine
    return - (grad_div_v + J_grad_curl_v)
