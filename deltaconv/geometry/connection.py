import torch
import torch.linalg as LA
from .utils import batch_dot


def build_transport(target_n, target_x, target_y, source_n, source_x, non_oriented=True):
    # This implementation is a PyTorch reimplementation of functionality in Geometry Central.
    # Credits for the original C++ implementation are due to Nicholas Sharp
    # and his collaborators on Geometry Central.

    # This code is able to work with unoriented surfaces.
    # It tries to orient the surface locally by adding a reflection after
    # parallel transport if the the normals of two neighboring points are
    # pointing in opposite directions. 
    inverted = batch_dot(source_n, target_n) < 0
    target_n = torch.where(inverted, -target_n, target_n)
    target_y = torch.where(inverted, -target_y, target_y)

    axis = torch.cross(target_n, source_n)
    axis_norm = LA.norm(axis, dim=-1, keepdim=True)
    axis = torch.where(axis_norm > 1e-6, axis / axis_norm, source_x)

    angle = angle_in_plane(source_n, target_n, axis)

    source_x_in_target_3D = rotate_around(source_x, axis, angle)
    source_x_in_target = torch.cat([
        batch_dot(source_x_in_target_3D, target_x),
        batch_dot(source_x_in_target_3D, target_y)
    ], dim=1)

    source_x_in_target_norm = LA.norm(source_x_in_target, dim=-1, keepdim=True)
    identity = torch.zeros_like(source_x_in_target)
    identity[:, 0] = 1
    source_x_in_target = torch.where(source_x_in_target_norm > 1e-6, source_x_in_target / source_x_in_target_norm, identity)

    conj = torch.ones(source_x_in_target.size(0), device=source_x_in_target.device).float()
    if non_oriented:
        conj = torch.where(inverted.flatten(), -conj, conj)

    connection = torch.stack([
        source_x_in_target[:, 0],
        -source_x_in_target[:, 1],
        source_x_in_target[:, 1] * conj,
        source_x_in_target[:, 0] * conj
    ], dim=1)

    return connection


def angle_in_plane(u, v, normal):
    u_plane = u - batch_dot(u, normal) * normal
    u_plane = u_plane / LA.norm(u_plane, dim=-1, keepdim=True).clamp(1e-8)
    basis_y = torch.cross(normal, u_plane)
    basis_y = basis_y / LA.norm(basis_y, dim=-1, keepdim=True).clamp(1e-8)

    x_comp = batch_dot(v, u_plane)
    y_comp = batch_dot(v, basis_y)

    return torch.atan2(y_comp, x_comp)


def rotate_around(v, axis, angle):
    if len(angle.size()) == 1:
        angle = angle.unsqueeze(-1)

    parallel_comp = axis * batch_dot(v, axis)
    tangent_comp = v - parallel_comp

    tangent_comp_norm = LA.norm(tangent_comp, dim=-1, keepdim=True).clamp(1e-8)
    basis_x = tangent_comp / tangent_comp_norm
    basis_y = torch.cross(axis, basis_x)

    rotated_v = tangent_comp_norm * (torch.cos(angle) * basis_x + torch.sin(angle) * basis_y)
    rotated_v = rotated_v + parallel_comp

    return torch.where(tangent_comp_norm > 0, rotated_v, parallel_comp)
