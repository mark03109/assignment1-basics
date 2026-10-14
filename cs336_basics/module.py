import torch
from torch import nn
import math
from einops import einsum

class LinearModule(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        mean = 0
        std = math.sqrt(2/(in_features + out_features))
        self.weight = nn.Parameter(torch.normal(
            mean = mean, std = std, size = (
                out_features, in_features
            ), device = device, dtype = dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features")
        return x

class EmbeddingModule(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device = None, dtype = None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim,
        device = device, dtype = dtype))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """input: token_ids(...), output: (..., d_model)"""
        return self.weight[token_ids]# 按token_ids内部的每个元素展开

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        self.g = nn.Parameter(torch.randn(d_model, device = device, dtype = dtype))
        self.eps = eps

    def rms_norm(self, x: torch.Tensor, dim = -1):
        mean_square = x.square().mean(dim = dim) + self.eps
        return einsum(x, torch.rsqrt(mean_square), 
                      "batch_size sequence_length d_model, batch_size, sequence_length -> batch_size sequence_length d_model")
     
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        result = self.rms_norm(x)
        return result.to(in_dtype)

