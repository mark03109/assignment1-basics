import torch
from torch import nn
import math
from einops import einsum
from typing import IO, Any, BinaryIO

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

# def run_swiglu(
#     d_model: int,
#     d_ff: int,
#     w1_weight: Float[Tensor, " d_ff d_model"],
#     w2_weight: Float[Tensor, " d_model d_ff"],
#     w3_weight: Float[Tensor, " d_ff d_model"],
#     in_features: Float[Tensor, " ... d_model"],
# ) -> Float[Tensor, " ... d_model"]:

class Swiglu(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            d_ff = 8 * d_model / 3
            d_ff = (d_ff // 64 + 1) * 64
        self.d_ff = d_ff
        self.weight1 = nn.Parameter(torch.randn(d_ff, d_model))
        self.weight2 = nn.Parameter(torch.randn(d_model, d_ff)) 
        self.weight3 = nn.Parameter(torch.randn(d_ff, d_model))

    def silu(self, in_features: torch.Tensor):
        return torch.sigmoid(in_features) * in_features
        
    def forward(self, x: torch.Tensor):
        mul1 = self.silu(einsum(self.weight1, x, 
                                "d_ff d_model, ... d_model -> ... d_ff"))
        mul2 = einsum(self.weight3, x,
                      "d_ff d_model, ... d_model-> ... d_ff")
        mul3 = einsum(mul1, mul2, 
                      "... d_ff, ... d_ff -> ... d_ff")
        return einsum(self.weight2, mul3, 
                      "d_model d_ff, ... d_ff -> ... d_model")

