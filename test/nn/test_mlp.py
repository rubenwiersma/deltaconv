import torch

from deltaconv.nn import MLP, VectorMLP, ScalarVectorMLP, ScalarVectorIdentity


def test_mlp():
    x = torch.rand(10, 16)

    # 1. Test a single-layer perceptron
    mlp1 = MLP((16, 32))
    out = mlp1(x)
    assert out.size(1) == 32
    assert out.isnan().sum() == 0

    # 2. Test a three-layer MLP
    mlp2 = MLP((16, 32, 32, 64))
    out = mlp2(x)
    assert out.size(1) == 64
    assert out.isnan().sum() == 0


def test_vectormlp():
    # Number of points to simulate
    N = 1000
    # Input and output features of each mlp
    C_in = 16
    C_out = 32
    v = torch.rand(N, C_in)

    # 1. Test a single-layer perceptron
    v_mlp1 = VectorMLP((C_in, C_out))
    out = v_mlp1(v)
    assert out.size(1) == C_out 
    assert out.isnan().sum() == 0

    # 2. Test a three-layer MLP
    v_mlp2 = VectorMLP((C_in, C_out, C_out, C_out))
    out = v_mlp2(v)
    assert out.size(1) == C_out
    assert out.isnan().sum() == 0

    # 3. Test transformation equivariance of MLP
    # For a random orthonormal 2x2 transformation matrix T,
    # we expect MLP(Tv) == T MLP(v)
    # That means the vector MLPs are equivariant to both rotation and reflection.

    # Compute a random 2D transformation for each point
    angle = torch.rand(N // 2) * 2 * torch.pi
    c, s = torch.cos(angle), torch.sin(angle)
    R = torch.stack([
        torch.stack([c, s], dim=1),
        torch.stack([-s, c], dim=1)
    ], dim=1)

    # Flip some
    ones = torch.ones(N // 2)
    zeros = torch.zeros(N // 2)
    reflect = torch.where(torch.rand(N // 2) > 0.1, ones, -ones)
    F = torch.stack([
        torch.stack([ones, zeros], dim=1),
        torch.stack([zeros, reflect], dim=1)
    ], dim=1)

    # T = F R
    T = torch.bmm(F, R)

    # T MLP(v)
    t_mlp1_v = torch.bmm(T, v_mlp1(v).view(-1, 2, C_out)).view(-1, C_out)
    t_mlp2_v = torch.bmm(T, v_mlp2(v).view(-1, 2, C_out)).view(-1, C_out)

    # MLP(Tv)
    v_transformed = torch.bmm(T, v.view(-1, 2, C_in)).view(-1, C_in)
    mlp1_t_v = v_mlp1(v_transformed)
    mlp2_t_v = v_mlp2(v_transformed)

    # Assert they are equal
    assert torch.allclose(t_mlp1_v, mlp1_t_v, atol=1e-5)
    assert torch.allclose(t_mlp2_v, mlp2_t_v, atol=1e-5)


def test_scalarvectormlp_identity():
    N = 1000
    C_in = 16
    C_out = 32

    # Create dummy features
    x = torch.rand(N, C_in)
    v = torch.rand(N * 2, C_in)

    # 1. Test ScalarVectorMLP with vector stream
    sv_mlp = ScalarVectorMLP((C_in, C_out), vector_stream=True)
    
    sv_out_sv = sv_mlp((x, v))
    assert type(sv_out_sv) is tuple
    assert sv_out_sv[0].size(1) == C_out
    assert sv_out_sv[1].size(1) == C_out

    # 2. Test ScalarVectorMLP without vector stream
    s_mlp = ScalarVectorMLP((C_in, C_out), vector_stream=False)
    s_out_s = s_mlp(x)
    assert type(s_out_s) is torch.Tensor
    assert s_out_s.size(1) == C_out

    # 3. Test ScalarVectorMLP without vector stream
    # should output only scalar when both scalar and vector is input
    s_out_sv = s_mlp((x, v))
    assert type(s_out_sv) is torch.Tensor
    assert s_out_sv.size(1) == C_out

    # 4. ScalarVectorIdentity should always return the same as the input
    identity = ScalarVectorIdentity()
    assert torch.allclose(x, identity(x))
    assert (x, v) == identity((x, v))
    assert torch.allclose(v, identity(v))