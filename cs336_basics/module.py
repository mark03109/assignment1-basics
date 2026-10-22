import torch
from torch import nn
import math
from einops import einsum, rearrange
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
        super().__init__()
        self.g = nn.Parameter(torch.randn(d_model, device = device, dtype = dtype))
        self.eps = eps

    def rms_norm(self, x: torch.Tensor, dim = -1):
        mean_square = x.square().mean(dim = dim) + self.eps
        rms = einsum(x, torch.rsqrt(mean_square), 
                      "batch_size sequence_length d_model, batch_size sequence_length -> batch_size sequence_length d_model")
        rms = einsum(rms, self.g,
                     "batch_size sequence_length d_model, d_model -> batch_size sequence_length d_model")
        return rms

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

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super().__init__()
        self.theata = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        positions = torch.arange(max_seq_len, dtype = torch.float32, device = device)
        pair_indicies = torch.arange(d_k // 2, dtype = torch.float32, device = device)
        inv_freq = theta ** (-2 * pair_indicies / d_k)
        angles = positions[:, None] * inv_freq[None, :]

        self.register_buffer(
            "cos_cache", angles.cos(), persistent = False
        ) # [max_seq_len, d_k//2]
        self.register_buffer(
            "sin_cache", angles.sin(), persistent = False
        ) # [max_seq_len, d_k//2]

    def forward(self, x: torch.tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x [..., seq_len, d_k], token_positions [..., seq_len], d_k is dimension of query and key vectors
        
        cos = self.cos_cache[token_positions] # [seq_len, d_k//2]
        sin = self.sin_cache[token_positions] # [seq_len, d_k//2]

        x_even = x[..., 0::2] # [..., seq, d_k//2]
        x_odd = x[..., 1::2]  # [..., seq, d_k//2]

        # y_even = einsum(cos, x_even,
        #                 "seq_len d_k//2, ... seq_len d_k//2 -> ... seq_len d_k//2"
        #                     ) - \
        #          einsum(sin, x_odd, 
        #                 "seq_len d_k//2, ... seq_len d_k//2 -> ... seq_len d_k//2"
        #                     )
        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos
        # 按元素乘对应正余弦值

        y = torch.stack((y_even, y_odd), dim = -1) # (..., seq_len, d_k//2, 2)
        return rearrange(y, "... seq_len pairs r -> ... seq_len (pairs r)")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, device = None, dtype = None, rope: RotaryPositionalEmbedding | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_head = d_model // num_heads
        self.Wq = LinearModule(d_model, d_model, device, dtype)
        self.Wk = LinearModule(d_model, d_model, device, dtype)
        self.Wv = LinearModule(d_model, d_model, device, dtype)
        self.Wo = LinearModule(d_model, d_model, device, dtype)
        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        """x shape (... sequence_length d_model)"""
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        seq_len = x.shape[-2]
        if self.rope is None:
            multi_Q = rearrange(Q, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads = self.num_heads, d_head = self.d_head)
            multi_K = rearrange(K, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads = self.num_heads, d_head = self.d_head)
            multi_V = rearrange(V, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads = self.num_heads, d_head = self.d_head)
        else:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device = x.device)
            multi_Q = rearrange(Q, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads = self.num_heads, d_head = self.d_head)
            multi_K = rearrange(K, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads = self.num_heads, d_head = self.d_head)
            multi_V = rearrange(V, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads = self.num_heads, d_head = self.d_head)
            multi_Q = self.rope(multi_Q, token_positions)
            multi_K = self.rope(multi_K, token_positions)
        mask = torch.ones(seq_len, seq_len, device=x.device, dtype= bool).tril()
        muti_attention =MyFunctions.scaled_dot_product_attention(multi_Q, multi_K, multi_V, mask)
        attention = rearrange(muti_attention, "... num_heads seq_len d_head -> ... seq_len (num_heads d_head)")
        return self.Wo(attention)

class TransformerBlock(nn.Moudele):
    def __init__(self, d_model, num_heads, d_ff, device = None, dtype = None, rope: RotaryPositionalEmbedding | None = None):
        """we only pass RoPE from outside here"""
        super().__init__()
        self.muti_head_attention = MultiHeadAttention(d_model, num_heads, device, dtype, rope)
        self.rms_norm1 = RMSNorm(d_model)
        self.rms_norm2 = RMSNorm(d_model)
        self.rope = rope
        self.fnn = Swiglu(d_model, d_ff)     

    def forward(self, x: torch.Tensor):
        x += self.muti_head_attention(self.rms_norm1(x))
        x += self.fnn(self.rms_norm2(x))
        return x  

class TransformerLM(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, vocab_size, context_length, num_layers, 
                 device = None, dtype = None, ropes: list[RotaryPositionalEmbedding] | None = None):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = EmbeddingModule(vocab_size, d_model)
        self.transformer_blocks: list[TransformerBlock] = []
        for i in range(num_layers):
            self.transformer_blocks.append(TransformerBlock(d_model, num_heads, d_ff, device, dtype, ropes[i]))
        self.extra_norm = RMSNorm(d_model)
        self.linear = LinearModule(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        for i in range(self.num_layers):
            x = self.transformer_blocks[i](x)
        x = self.extra_norm(x)
        x = self.linear(x)
        x = MyFunctions.softmax(x, dim = -1)

class MyFunctions:
    def softmax(x: torch.Tensor, dim) -> torch.Tensor:
        max_val = torch.max(x, dim=dim, keepdim=True).values # keepdim很关键，保持原来的张量形状，方便广播
        x = x - max_val
        x = math.e**x
        return x / x.sum(dim = dim, keepdim=True)

    def scaled_dot_product_attention( Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """keys and queries are of shape (batch_size, ..., seq_len, d_k), 
        values (batch_size, ..., seq_len, d_v)"""
        qk = einsum(Q, K, 
                    "batch_size ... seq_len1 d_k, batch_size ... seq_len2 d_k -> batch_size ... seq_len1 seq_len2")
        if mask is not None:
            qk = qk.masked_fill(~mask, float("-inf"))
        scores = qk / math.sqrt(Q.shape[-1])
        scores_sm =  MyFunctions.softmax(scores, -1)
        return einsum(scores_sm, V,
                      "batch_size ... seq_len1 seq_len2, batch_size ... seq_len2 d_v -> batch_size ... seq_len1 d_v")
    
def _softmax_trial():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 4, device=device)
    y = MyFunctions.softmax(x = x, dim = -1)
    """
    x
    tensor([[-1.0103, -0.7683, -1.0513,  1.6759],
            [ 0.6275, -0.5929, -0.3640, -1.2443],
            [ 1.0463, -0.4980, -0.0292,  0.0455],
            [ 0.1790,  0.9046,  0.5002, -1.2187]])

    max_val
    tensor([[1.6759],
            [0.6275],
            [1.0463],
            [0.9046]])

    x
    tensor([[-2.6862, -2.4442, -2.7273,  0.0000],
            [ 0.0000, -1.2204, -0.9915, -1.8719],
            [ 0.0000, -1.5443, -1.0755, -1.0008],
            [-0.7256,  0.0000, -0.4044, -2.1233]])
            
    x
    tensor([[0.0681, 0.0868, 0.0654, 1.0000],
            [1.0000, 0.2951, 0.3710, 0.1538],
            [1.0000, 0.2135, 0.3411, 0.3676],
            [0.4840, 1.0000, 0.6674, 0.1196]])

    y
    tensor([[0.2233, 0.2845, 0.2144, 3.2778],
            [2.1979, 0.6486, 0.8154, 0.3381],
            [2.0810, 0.4442, 0.7099, 0.7649],
            [0.8525, 1.7613, 1.1754, 0.2107]])
    """

