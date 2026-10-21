"""
Rep-BLGR: Reparameterized Bidirectional Local-Global Refiner
Combining multi-scale reparameterized local depthwise convolution with
fine-grained adaptive self-attention from FASA.
"""
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RepDWConv(nn.Module):
    """
    Multi-scale Depthwise Convolution with Structural Re-parameterization.
    - Training (deploy=False): Parallel branches (DWConv 5x5 + BN, DWConv 3x3 + BN, Identity + BN)
    - Inference (deploy=True): Single fused DWConv 5x5 layer
    """
    def __init__(self, dim: int, deploy: bool = False):
        super().__init__()
        self.dim = dim
        self.deploy = deploy

        if deploy:
            self.reparam = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=True)
        else:
            self.dw5 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=False),
                nn.BatchNorm2d(dim)
            )
            self.dw3 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim)
            )
            self.idn = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.reparam(x)
        return self.dw5(x) + self.dw3(x) + self.idn(x)

    @staticmethod
    def _fuse(kernel: torch.Tensor, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
        std = (bn.running_var + bn.eps).sqrt()
        fused_weight = kernel * (bn.weight / std).reshape(-1, 1, 1, 1)
        fused_bias = bn.bias - bn.running_mean * bn.weight / std
        return fused_weight, fused_bias

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy:
            return

        # 1. Fuse Conv 5x5 + BN
        k5, b5 = self._fuse(self.dw5[0].weight, self.dw5[1])

        # 2. Fuse Conv 3x3 + BN and zero-pad to 5x5
        k3, b3 = self._fuse(self.dw3[0].weight, self.dw3[1])
        k3_pad = F.pad(k3, [1, 1, 1, 1])

        # 3. Fuse Identity + BN as 5x5 kernel with 1.0 at center (2, 2)
        kid = torch.zeros_like(k5)
        kid[:, 0, 2, 2] = 1.0
        kid_fused, bid = self._fuse(kid, self.idn)

        # 4. Construct single equivalent Conv2d layer
        self.reparam = nn.Conv2d(
            self.dim, self.dim, kernel_size=5, stride=1, padding=2, groups=self.dim, bias=True
        ).to(device=k5.device, dtype=k5.dtype)

        self.reparam.weight.copy_(k5 + k3_pad + kid_fused)
        self.reparam.bias.copy_(b5 + b3 + bid)

        # Remove training-only branches
        del self.dw5
        del self.dw3
        del self.idn
        self.deploy = True


class RepBLGR(nn.Module):
    """
    Reparameterized Bidirectional Local-Global Refiner (Rep-BLGR)
    Designed for search token refinement before prediction head.
    """
    def __init__(
        self,
        dim: int = 192,
        num_heads: int = 3,
        window_size: int = 2,
        bidirectional: bool = False,
        global_type: str = "fasa",
        deploy: bool = False
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5
        self.window_size = window_size
        self.bidirectional = bidirectional
        self.global_type = global_type
        self.deploy = deploy

        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.local = RepDWConv(dim, deploy=deploy)

        if global_type == "fasa":
            self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, stride=1, padding=0)
            if window_size == 1:
                self.pool = nn.Identity()
            elif window_size == 2:
                self.pool = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=5, stride=2, padding=2, groups=dim, bias=True),
                    nn.BatchNorm2d(dim)
                )
            else:
                self.pool = nn.Sequential(
                    nn.AvgPool2d(window_size, stride=window_size)
                )
        elif global_type == "se":
            self.g_proj = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim)
            )
            self.g_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim // 4, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 4, dim, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unsupported global_type: {global_type}. Choose 'fasa' or 'se'.")

        self.mixer = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        # Residual connection with layer-scale gamma initialized to 0 for stable warmup
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def _compute_global(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if self.global_type == "se":
            return self.g_proj(x) * self.g_gate(x)

        # FASA downsampled attention
        xp = self.pool(x)
        H, W = xp.shape[-2:]
        qh = q.reshape(b, self.num_heads, self.dim_head, h * w).transpose(-1, -2).contiguous()  # [b, m, (h*w), d]

        kv = self.kv(xp).reshape(b, 2, self.num_heads, self.dim_head, H * W).permute(1, 0, 2, 4, 3).contiguous()
        k, v = kv[0], kv[1]  # [b, m, (H*W), d]

        attn = torch.softmax(self.scale * (qh @ k.transpose(-1, -2)), dim=-1)  # [b, m, (h*w), (H*W)]
        glob = (attn @ v).transpose(-1, -2).reshape(b, c, h, w)  # [b, c, h, w]
        return glob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        y = self.norm(x)
        q = self.q(y)

        # Local branch with self-modulation via SiLU (x * sigmoid(x))
        local = F.silu(self.local(q))

        # Global branch
        glob = self._compute_global(y, q)

        # Local <-> Global interaction
        if self.bidirectional:
            local = local * torch.sigmoid(glob)
            glob = glob * torch.sigmoid(local)
        else:
            local = local * torch.sigmoid(glob)

        f_out = self.mixer(local * glob)
        return x + self.gamma * f_out

    def switch_to_deploy(self):
        """Convert local reparameterized convolution to single 5x5 DWConv."""
        self.local.switch_to_deploy()
        self.deploy = True
