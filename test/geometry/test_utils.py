import torch

from deltaconv.geometry import batch_dot


def test_batch_dot():
    a = torch.rand(1024, 10)
    b = torch.rand(1024, 10)

    a_dot_b = (a * b).sum(dim=1, keepdim=True)
    out = batch_dot(a, b)

    assert torch.allclose(out, a_dot_b)
