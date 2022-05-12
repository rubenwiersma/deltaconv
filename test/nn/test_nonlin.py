import torch
from torch import Tensor

from deltaconv.nn import BatchNorm1d, VectorNonLin
from deltaconv.geometry import norm

def test_batchnorm1d():
    bn = BatchNorm1d(10)
    bn.reset_parameters()
    assert bn.__repr__() == 'BatchNorm1d(10)'

    # 1. Simple batchnorm task: batchnorm on this tensor should return all zeros
    # because the variance is 0
    x = torch.stack([torch.rand(10)] * 4, dim=0)
    out = bn(x)
    assert isinstance(out, Tensor)
    assert out.size() == x.size()
    assert torch.allclose(out, torch.zeros_like(x))
    assert torch.isnan(out).sum() == 0

    # New BatchNorm instance for next test
    bn = BatchNorm1d(5)
    bn.reset_parameters()
    assert bn.__repr__() == 'BatchNorm1d(5)'

    # 2. Scale and shift the values in a 0-mean vector with a different value per channel
    zeromean = torch.FloatTensor([2, 1, 0, -1, -2, 1.5, -1.5, 1, 1, -2])
    assert zeromean.mean() == 0
    shifts = torch.FloatTensor([1, 2, 3, 4, 5])
    x = torch.stack([zeromean] * 5, dim=1)
    x_shifted = x * shifts + shifts
    assert torch.allclose(x_shifted.mean(dim=0), shifts)

    # Should return the same as batchnorm applied to the original vector for each channel
    out = bn(x)
    out_shifted = bn(x_shifted)
    assert torch.allclose(out, out_shifted)


def test_vectornonlin():
    vnl = VectorNonLin(4)
    vnl.reset_parameters()
    assert vnl.__repr__() == 'VectorNonLin(batchnorm=None)'

    # Instantiate a dummy vector
    v = torch.rand((10, 4))

    # 1. VectorNonLin without batchnorm and initial bias should return identity
    out = vnl(v)
    assert isinstance(out, Tensor)
    assert torch.allclose(out, v)
    assert torch.isnan(out).sum() == 0

    # VectorNonLin with batchnorm
    vnl_bn = VectorNonLin(1, batchnorm=BatchNorm1d(1))

    # 2. Vector features where the mean vector norm is 0.5
    # And the vectors are either of size 0, 0.25, 0.5, 0.75 or 1
    v_x       = torch.FloatTensor([1, 0, -0.75, 0.25, 0.5,  0, 0,    0,     0,    0])
    v_y       = torch.FloatTensor([0, 0,     0,    0,   0, -1, 0, 0.75, -0.25, -0.5])
    v_norm_gt = torch.FloatTensor([1, 0,  0.75, 0.25, 0.5,  1, 0, 0.75,  0.25,  0.5]).unsqueeze(-1)


    # 2.a. Test vector norm
    v = torch.stack([v_x, v_y], dim=1).view(-1, 1)
    v_norm = norm(v)
    assert torch.allclose(v_norm, v_norm_gt)

    out = vnl_bn(v)
    out_norm = norm(out)

    # The resulting vectors should become 0 when <= 0.5 and scaled if > 0.5
    assert torch.allclose(out_norm > 0, v_norm_gt > 0.5)
    assert torch.allclose(out_norm == 0, v_norm_gt <= 0.5)
    out_x, out_y = out.view(-1, 2).T

    # The directions of vectors should have been left untouched
    # The input vectors all point along either the x- or y-axis
    # All values between -0.5 and 0.5 should have been mapped to 0 
    assert torch.allclose(out_x == 0, (v_x <= 0.5) * (v_x >= -0.5))
    assert torch.allclose(out_y == 0, (v_y <= 0.5) * (v_y >= -0.5))

    # All positive values above 0.5 should remain positive
    assert torch.allclose(out_x > 0, v_x > 0.5)
    assert torch.allclose(out_y > 0, v_y > 0.5)

    # All negative values under 0.5 should remain negative
    assert torch.allclose(out_x < 0, v_x < -0.5)
    assert torch.allclose(out_y < 0, v_y < -0.5)