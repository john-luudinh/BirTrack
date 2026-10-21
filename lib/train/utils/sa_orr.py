"""
Search-Aware Spatial Cox Process Masking (SA-ORR)
Fully implemented in PyTorch for GPU execution without CPU bottlenecks.
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SearchAwareCoxMasking(nn.Module):
    """
    Search-Aware Spatial Cox Process Masking for both Template (Z) and Search (X).
    - Template Z: Bell-shaped Cox intensity centered at (0.5, 0.5)
    - Search X: Bell-shaped Cox intensity centered at Ground-Truth bbox center (xc, yc),
      with sigma scaled to target size and mask count proportional to target patch count.
    """
    def __init__(
        self,
        patch_size: int = 16,
        mask_ratio_z: float = 0.30,
        mask_ratio_x_in_box: float = 0.30,
        sigma_scale: float = 0.5,
        bg_weight: float = 0.2
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio_z = mask_ratio_z
        self.mask_ratio_x_in_box = mask_ratio_x_in_box
        self.sigma_scale = sigma_scale
        self.bg_weight = bg_weight

    def _generate_gaussian_intensity(
        self,
        H_grid: int,
        W_grid: int,
        centers: torch.Tensor,
        sigmas: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        centers: [B, 2] in normalized coordinates [0, 1]
        sigmas: [B, 1] in normalized coordinates
        Returns: [B, H_grid, W_grid] normalized intensity distribution
        """
        B = centers.shape[0]
        y_lin = torch.linspace(0.5 / H_grid, 1.0 - 0.5 / H_grid, H_grid, device=device)
        x_lin = torch.linspace(0.5 / W_grid, 1.0 - 0.5 / W_grid, W_grid, device=device)
        yy, xx = torch.meshgrid(y_lin, x_lin, indexing='ij')  # [H, W]

        xx = xx.unsqueeze(0).expand(B, -1, -1)
        yy = yy.unsqueeze(0).expand(B, -1, -1)

        cx = centers[:, 0].view(B, 1, 1)
        cy = centers[:, 1].view(B, 1, 1)
        sigma2 = (sigmas.view(B, 1, 1) ** 2).clamp(min=1e-4)

        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        intensity = torch.exp(-0.5 * dist_sq / sigma2)
        intensity_sum = intensity.sum(dim=(1, 2), keepdim=True) + 1e-8
        return intensity / intensity_sum

    def _mask_image_by_intensity(
        self,
        img: torch.Tensor,
        intensity: torch.Tensor,
        num_masked_per_sample: torch.Tensor
    ) -> torch.Tensor:
        """
        img: [B, C, H, W]
        intensity: [B, H_g, W_g]
        num_masked_per_sample: [B]
        """
        B, C, H, W = img.shape
        H_g, W_g = intensity.shape[1], intensity.shape[2]
        total_patches = H_g * W_g
        device = img.device

        flat_intensity = intensity.view(B, total_patches)
        # Add stochastic noise for inhomogeneous Poisson process thinning
        noise = torch.rand_like(flat_intensity)
        scores = flat_intensity * noise

        keep_mask = torch.ones((B, total_patches), dtype=img.dtype, device=device)
        for i in range(B):
            k = int(num_masked_per_sample[i].item())
            k = max(1, min(k, total_patches - 1))
            _, topk_idx = torch.topk(scores[i], k=k, largest=True)
            keep_mask[i, topk_idx] = 0.0

        keep_mask = keep_mask.view(B, 1, H_g, W_g)
        keep_mask_pixel = F.interpolate(keep_mask, size=(H, W), mode='nearest')
        return img * keep_mask_pixel

    @torch.no_grad()
    def forward(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
        gt_bbox_search: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inputs:
            template: [B, 3, 128, 128]
            search: [B, 3, 256, 256]
            gt_bbox_search: [B, 4] normalized (x, y, w, h)
        Returns:
            masked_template: [B, 3, 128, 128]
            masked_search: [B, 3, 256, 256]
            token_weights_x: [B, K_x, 1] weight for search token consistency loss
        """
        B, _, H_z, W_z = template.shape
        _, _, H_x, W_x = search.shape
        device = template.device

        # 1. Mask Template (Z): fixed center at (0.5, 0.5)
        Hg_z, Wg_z = H_z // self.patch_size, W_z // self.patch_size
        total_patches_z = Hg_z * Wg_z
        center_z = torch.full((B, 2), 0.5, device=device)
        sigma_z = torch.full((B, 1), 0.25, device=device)
        intensity_z = self._generate_gaussian_gaussian = self._generate_gaussian_intensity(
            Hg_z, Wg_z, center_z, sigma_z, device
        )
        num_masked_z = torch.full(
            (B,), int(round(self.mask_ratio_z * total_patches_z)), dtype=torch.long, device=device
        )
        masked_template = self._mask_image_by_intensity(template, intensity_z, num_masked_z)

        # 2. Mask Search (X): center at GT center, scale sigma to target bbox
        Hg_x, Wg_x = H_x // self.patch_size, W_x // self.patch_size
        total_patches_x = Hg_x * Wg_x

        xc = (gt_bbox_search[:, 0] + 0.5 * gt_bbox_search[:, 2]).clamp(0.05, 0.95)
        yc = (gt_bbox_search[:, 1] + 0.5 * gt_bbox_search[:, 3]).clamp(0.05, 0.95)
        center_x = torch.stack([xc, yc], dim=-1)

        # Target radius in normalized coordinates
        target_size = (gt_bbox_search[:, 2] * gt_bbox_search[:, 3]).clamp(min=1e-4).sqrt()
        sigma_x = (self.sigma_scale * target_size).clamp(min=0.08, max=0.45).unsqueeze(-1)
        intensity_x = self._generate_gaussian_intensity(Hg_x, Wg_x, center_x, sigma_x, device)

        # Patches inside GT box
        target_area_patches = (gt_bbox_search[:, 2] * Hg_x) * (gt_bbox_search[:, 3] * Wg_x)
        num_masked_x = torch.round(self.mask_ratio_x_in_box * target_area_patches).long().clamp(
            min=1, max=int(0.5 * total_patches_x)
        )
        masked_search = self._mask_image_by_intensity(search, intensity_x, num_masked_x)

        # 3. Compute spatial token weights for search tokens
        y_lin = torch.linspace(0.5 / Hg_x, 1.0 - 0.5 / Hg_x, Hg_x, device=device)
        x_lin = torch.linspace(0.5 / Wg_x, 1.0 - 0.5 / Wg_x, Wg_x, device=device)
        gy, gx = torch.meshgrid(y_lin, x_lin, indexing='ij')
        gx = gx.unsqueeze(0).expand(B, -1, -1)
        gy = gy.unsqueeze(0).expand(B, -1, -1)

        x1 = gt_bbox_search[:, 0].view(B, 1, 1)
        y1 = gt_bbox_search[:, 1].view(B, 1, 1)
        x2 = (gt_bbox_search[:, 0] + gt_bbox_search[:, 2]).view(B, 1, 1)
        y2 = (gt_bbox_search[:, 1] + gt_bbox_search[:, 3]).view(B, 1, 1)

        in_box = (gx >= x1) & (gx <= x2) & (gy >= y1) & (gy <= y2)
        token_weights_x = torch.where(
            in_box,
            torch.ones_like(gx),
            torch.full_like(gx, self.bg_weight)
        ).view(B, total_patches_x, 1)

        return masked_template, masked_search, token_weights_x
