"""
MI-AFKD: Mutual Information & Difficulty-Adaptive Feature Knowledge Distillation
Combines adaptive difficulty weighting from ORTrack with softened spatial Jensen-Shannon Divergence.
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifficultyAdaptiveMIDistillationLoss(nn.Module):
    """
    Difficulty-Adaptive Feature Knowledge Distillation combining:
    1. Adaptive sample difficulty weighting: w = tau_0 + rho * ( (1 - giou) - mean(1 - giou) )
    2. Full-token feature MSE distance: || t_S - t_T ||^2
    3. Spatial Jensen-Shannon Divergence (Mutual Information) on softened feature maps (temperature tau)
    """
    def __init__(
        self,
        tau_0: float = 10.0,
        rho: float = 10.0,
        temperature: float = 2.0,
        mi_weight: float = 0.5,
        use_ema: bool = False,
        ema_decay: float = 0.99
    ):
        super().__init__()
        self.tau_0 = tau_0
        self.rho = rho
        self.temperature = temperature
        self.mi_weight = mi_weight
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        if self.use_ema:
            self.register_buffer("mean_giou_loss", torch.tensor(0.4))

    def _compute_softened_jsd(self, feat_s: torch.Tensor, feat_t: torch.Tensor) -> torch.Tensor:
        """
        Compute Jensen-Shannon Divergence across spatial distributions for each channel.
        feat_s, feat_t: [B, C, H, W]
        Returns: [B] JSD loss per sample
        """
        B, C, H, W = feat_s.shape
        s_flat = feat_s.view(B, C, -1) / self.temperature
        t_flat = feat_t.detach().view(B, C, -1) / self.temperature

        p_s = F.softmax(s_flat, dim=-1)
        p_t = F.softmax(t_flat, dim=-1)
        p_m = 0.5 * (p_s + p_t)

        # KL divergence: F.kl_div expects log-probabilities as input, targets as second arg
        log_m = torch.log(p_m + 1e-8)
        kl_s = F.kl_div(log_m, p_s, reduction='none').sum(dim=-1).mean(dim=-1)  # [B]
        kl_t = F.kl_div(log_m, p_t, reduction='none').sum(dim=-1).mean(dim=-1)  # [B]

        jsd_per_sample = 0.5 * (kl_s + kl_t) * (self.temperature ** 2)
        return jsd_per_sample

    def forward(
        self,
        tokens_student: torch.Tensor,
        tokens_teacher: torch.Tensor,
        refined_2d_student: torch.Tensor,
        refined_2d_teacher: torch.Tensor,
        giou_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        tokens_student, tokens_teacher: [B, K, d]
        refined_2d_student, refined_2d_teacher: [B, d, H_s, W_s]
        giou_scores: [B] GIoU values in [-1, 1]
        """
        giou_loss_sample = (1.0 - giou_scores).detach().view(-1)

        # 1. Compute adaptive sample difficulty weight
        if self.use_ema and hasattr(self, "mean_giou_loss"):
            batch_mean = giou_loss_sample.mean()
            if self.training:
                self.mean_giou_loss = self.ema_decay * self.mean_giou_loss + (1.0 - self.ema_decay) * batch_mean
            diff = giou_loss_sample - self.mean_giou_loss
        else:
            diff = giou_loss_sample - giou_loss_sample.mean()

        coef = (self.tau_0 + self.rho * diff).clamp(min=0.1)

        # 2. Token-level MSE loss (with correct gradients via torch vector operations)
        mse_per_sample = ((tokens_student - tokens_teacher.detach()) ** 2).mean(dim=(1, 2))  # [B]

        # 3. Softened spatial JSD Mutual Information loss
        jsd_per_sample = self._compute_softened_jsd(refined_2d_student, refined_2d_teacher)  # [B]

        total_sample = coef * (mse_per_sample + self.mi_weight * jsd_per_sample)

        loss_total = total_sample.mean()
        loss_mse = mse_per_sample.mean()
        loss_mi = jsd_per_sample.mean()
        return loss_total, loss_mse, loss_mi
