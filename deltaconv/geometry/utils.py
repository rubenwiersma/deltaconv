import torch

def batch_dot(a, b):
    return torch.bmm(a.unsqueeze(1), b.unsqueeze(-1)).squeeze(-1)