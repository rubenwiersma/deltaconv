import torch
import numpy as np
import pytest

from deltaconv.geometry import geodesic_fps


def test_geodesic_fps():
    n = 1024
    n_samples = 512

    pos = np.random.randn(n, 3)

    # 1. Make sure we can sample all points if wanted
    samples1 = geodesic_fps(pos, n)
    assert samples1.shape[0] == n
    assert np.unique(samples1).shape[0] == n

    # 2. And that we get the exact amount of points requested
    samples2 = geodesic_fps(pos, n_samples)
    assert samples2.shape[0] == n_samples
    assert np.unique(samples2).shape[0] == n_samples

    # 3. Finally, expect errors when a wrong point cloud is given.
    with pytest.raises(ValueError):
        samples3 = geodesic_fps(torch.rand(n, 3), n)
    with pytest.raises(ValueError):
        samples3 = geodesic_fps(np.random.randn(n, 2, 3), n)