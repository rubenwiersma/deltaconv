import torch
import torch.nn.functional as F

from torch_geometric.nn import knn_graph

from deltaconv.nn import DeltaConv
from deltaconv.geometry import build_grad_div, estimate_basis, rotate_around


def test_deltaconv():
    N = 1000
    C_in = 3
    C_out = 32

    torch.manual_seed(1)

    # 1. Test input/output sizes
    conv = DeltaConv(C_in, C_out, depth=1, centralized=True, vector=True)
    assert conv.__repr__() == f'DeltaConv({C_in}, {C_out})'

    x = torch.rand(N, C_in)
    edge_index = knn_graph(x, 20, loop=True, flow='target_to_source')
    normal, x_basis, y_basis = estimate_basis(x, edge_index)
    grad, div = build_grad_div(x, normal, x_basis, y_basis, edge_index, regularizer=1e-8)
    assert grad.size(0) == 2 * N
    assert grad.size(1) == N
    assert div.size(0) == N
    assert div.size(1) == 2 * N

    v = grad @ x
    assert v.size(0) == 2 * N

    x_out, v_out = conv(x, v, grad, div, edge_index)
    assert x_out.size(1) == C_out
    assert v_out.size(1) == C_out

    conv1 = DeltaConv(C_in, C_out, depth=1, centralized=True, vector=False)
    x_out, v_out = conv1(x, v, grad, div, edge_index)
    assert x_out.size(1) == C_out
    assert torch.allclose(v, v_out)

    # 2. Test coordinate-independence
    # "a forward pass on a shape with one choice of bases leads
    #  to the same output and weight updates when run with different bases"
    # i.e.: using two different bases shouldn't affect the gradients of the backward pass.
     
    # Rotate bases with a random angle
    x_basis_rot = rotate_around(x_basis, normal, torch.rand(N) * 2 * torch.pi)
    y_basis_rot = torch.cross(normal, x_basis_rot)
    grad_rot, div_rot = build_grad_div(x, normal, x_basis_rot, y_basis_rot, edge_index, regularizer=1e-8)

    # Target x
    target_x = torch.rand(N, 1)

    # Perform forward pass
    conv2 = DeltaConv(C_in, 1, depth=1, centralized=False)

    # Baseline
    conv2.zero_grad()
    v = grad @ x
    out, _ = conv2(x, v, grad, div, edge_index)
    loss = F.l1_loss(out, target_x)
    loss.backward()
    gradients = torch.cat([p.grad.flatten() for p in conv2.parameters() if p.grad is not None])

    # Rotated
    conv2.zero_grad()
    v_rot = grad_rot @ x
    out_rot, _ = conv2(x, v_rot, grad_rot, div_rot, edge_index)
    loss_rot = F.l1_loss(out_rot, target_x)
    loss_rot.backward()
    gradients_rot = torch.cat([p.grad.flatten() for p in conv2.parameters() if p.grad is not None])

    assert torch.allclose(gradients, gradients_rot, atol=1e-5)