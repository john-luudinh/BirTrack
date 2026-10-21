"""
FASA: Fully Adaptive Self-Attention
Ported from official implementation of "Lightweight Vision Transformer with Bidirectional Interaction"
Reference: https://arxiv.org/abs/2306.00396
Cleaned and adapted for ORTrack / BiRTrack.
"""
import math
import torch
import torch.nn as nn


class FASA(nn.Module):
    """
    Fully Adaptive Self-Attention (FASA) Module
    """
    def __init__(self, dim: int, kernel_size: int = 5, num_heads: int = 3, window_size: int = 2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.window_size = window_size
        self.scalor = self.dim_head ** -0.5

        self.q = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, stride=1, padding=0)
        self.local_mixer = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=dim, bias=True
        )
        self.mixer = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.pool = self._build_refined_downsample(dim, window_size, kernel_size=5)

    def _build_refined_downsample(self, dim: int, window_size: int, kernel_size: int = 5) -> nn.Module:
        if window_size == 1:
            return nn.Identity()

        steps = int(math.log2(window_size)) if window_size > 0 and (window_size & (window_size - 1)) == 0 else 1
        block = nn.Sequential()
        for num in range(steps):
            block.add_module(
                f"conv{num}",
                nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=2, padding=kernel_size // 2, groups=dim, bias=True)
            )
            block.add_module(f"bn{num}", nn.BatchNorm2d(dim))
            if num != steps - 1:
                block.add_module(f"linear{num}", nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0))
        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        b, c, h, w = x.size()
        H = math.ceil(h / self.window_size)
        W = math.ceil(w / self.window_size)

        q_local = self.q(x)
        q = q_local.reshape(b, self.num_heads, self.dim_head, h * w).transpose(-1, -2).contiguous()  # [b, m, (h*w), d]

        pooled_x = self.pool(x)
        kv = self.kv(pooled_x).reshape(b, 2, self.num_heads, self.dim_head, H * W).permute(1, 0, 2, 4, 3).contiguous()
        k, v = kv[0], kv[1]  # [b, m, (H*W), d]

        # Scaled Dot-Product Attention in down-sampled space
        attn = torch.softmax(self.scalor * (q @ k.transpose(-1, -2)), dim=-1)  # [b, m, (h*w), (H*W)]
        global_feat = (attn @ v).transpose(-1, -2).reshape(b, c, h, w)  # [b, c, h, w]

        # Local context extraction with self-modulation
        local_feat = self.local_mixer(q_local)
        local_weight = torch.sigmoid(local_feat)
        local_feat = local_feat * local_weight

        # Bidirectional adaptation
        local2global = torch.sigmoid(global_feat)
        local_feat = local_feat * local2global

        return self.mixer(local_feat * global_feat)
