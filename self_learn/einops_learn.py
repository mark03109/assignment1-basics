import torch
from einops import rearrange, einsum

images = torch.randn(2, 2, 2, 3)
dim_by = torch.linspace(start = 0.0, end = 1.0, steps = 10)

dimmed_images = einsum(
    images, dim_by,
    "batch height width channel, dim_value -> batch dim_value height width channel"
)

print(images)
print(dim_by)
print(dimmed_images)
