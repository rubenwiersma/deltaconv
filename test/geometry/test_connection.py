from tkinter import W
import torch

from deltaconv.geometry.connection import *
from deltaconv.geometry.grad_div import build_tangent_basis


def test_rotate_around():
    N = 1000
    torch.manual_seed(42)

    # Setup random v
    v = torch.rand(N, 3)
    v = v / torch.linalg.norm(v, dim=1, keepdim=True).clamp(1e-8)
    # Compute an orthogonal vector
    axis, _ = build_tangent_basis(v)

    # 1. Rotate 90 degrees should give a vector that is axis x v
    assert torch.allclose(rotate_around(v, axis, torch.pi / 2 * torch.ones(N, 1)), torch.cross(axis, v), 1e-4)

    # 2. Rotate 180 degrees should give a vector that is -v
    assert torch.allclose(rotate_around(v, axis, torch.pi * torch.ones(N, 1)), -v, atol=1e-4)

    # 3. Rotate 360 degrees should give a vector that is v
    assert torch.allclose(rotate_around(v, axis, 2 * torch.pi * torch.ones(N, 1)), v, atol=1e-4)
    #    This should also be the case for any axis
    assert torch.allclose(rotate_around(v, torch.rand(N, 3), 2 * torch.pi * torch.ones(N, 1)), v, atol=1e-4)


def test_angle_in_plane():
    N = 1000

    u = torch.zeros(N, 3)
    u[:, 0] = 1

    # Generate v from a random angle
    angle = torch.rand(N, 1) * torch.pi
    v = torch.concat([
        torch.cos(angle),
        torch.sin(angle),
        torch.zeros_like(angle)
    ], dim=1)

    # Transform to a new plane around a random normal
    normal = torch.rand(N, 3)
    normal = normal / torch.linalg.norm(normal, dim=1, keepdim=True).clamp(1e-8)
    x_basis, y_basis = build_tangent_basis(normal)
    T = torch.stack([x_basis, y_basis, normal], dim=2)

    u = torch.bmm(T, u.unsqueeze(-1)).squeeze(-1)
    v = torch.bmm(T, v.unsqueeze(-1)).squeeze(-1)

    out_angle = angle_in_plane(u, v, normal)

    # 1. No NaNs and correct size
    assert out_angle.isnan().sum() == 0
    assert out_angle.size() == (N, 1)

    # 2. Returned angles are correct
    assert torch.allclose(out_angle, angle)


def test_build_transport():
    N = 1
    target_n = torch.rand(N, 3)
    target_n = target_n / torch.linalg.norm(target_n, dim=1, keepdim=True).clamp(1e-8)
    target_x, target_y = build_tangent_basis(target_n)

    # Create a basis of a neighbor that is rotated around the normal
    rotation_angle = torch.rand(N) * 2 * torch.pi
    source_x = rotate_around(target_x, target_n, rotation_angle)
    
    # And rotate the basis around some axis that is orthogonal to N
    axis = rotate_around(target_x, target_n, torch.rand(N))
    axis = axis / torch.linalg.norm(axis, dim=1, keepdim=True).clamp(1e-8)
    basis_angle = torch.rand(N) * 0.5 * torch.pi
    source_n = rotate_around(target_n, axis, basis_angle)
    source_x = rotate_around(source_x, axis, basis_angle)

    out_connection = build_transport(target_n, target_x, target_y, source_n, source_x, non_oriented=False)

    # 1. Correct size and no NaNs
    assert out_connection.size() == (N, 4)
    assert out_connection.isnan().sum() == 0

    # 2. Connection should be norm-preserving
    out_connection = out_connection.view(-1, 2, 2)
    v = torch.rand(N, 2, 1)
    v_norm = torch.linalg.norm(v, dim=1)
    v_transported = torch.bmm(out_connection, v)
    v_transported_norm = torch.linalg.norm(v_transported, dim=1)
    assert torch.allclose(v_norm, v_transported_norm)

    # 3. Connection should transport [1, 0] to
    # to the rotation_angle that we used to create the basis
    assert torch.allclose(out_connection[:, 0, 0], torch.cos(rotation_angle))
    assert torch.allclose(out_connection[:, 1, 0], torch.sin(rotation_angle))
