import torch

from deltaconv.geometry import norm, J, I_J, curl, laplacian, hodge_laplacian, batch_dot


def random_v(N=1024, C=16, return_components=False):
    v_norm = torch.rand(N, C) * 5
    v_angles = torch.rand(N, C) * 2 * torch.pi
    v_x = v_norm * torch.cos(v_angles)
    v_y = v_norm * torch.sin(v_angles)

    v = torch.stack([v_x, v_y], dim=1).view(-1, C)
    if return_components:
        return v, v_norm, v_angles, v_x, v_y
    return v


def test_norm():
    v, v_norm, _, _, _ = random_v(1024, 16, True)
    out = norm(v)
    assert torch.allclose(out, v_norm)


def test_J():
    N = 1024
    C = 16
    v, _, _, v_x, v_y = random_v(N, C, True)

    J_v = torch.stack([-v_y, v_x], dim=1).view(-1, C)
    out = J(v)
    assert torch.allclose(out, J_v)
    dot_v_J_v = (v.view(-1, 2, C) * out.view(-1, 2, C)).sum(dim=1)
    assert torch.allclose(dot_v_J_v, torch.zeros_like(v_x))


def test_I_J():
    N = 1024
    C = 16
    v = random_v(N, C)
    out = I_J(v)
    assert out.size(1) == v.size(1) * 2
    assert torch.allclose(out[:, :C], v)
    assert torch.allclose(out[:, C:], J(v))

# Curl, Laplacian, and Hodge-Laplacian are tested in test_gradient