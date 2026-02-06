"""
Conditional Latent Synthesis (CLS) Module for GRCL.

Key design: ZERO-INIT GATED RESIDUAL
    y_f = y + gate · (μ_fuse − y)
    gate = sigmoid(γ)   where γ is initialized to −3 → sigmoid(−3) ≈ 0.05

This ensures:
  - At initialization (or when refs are useless) → gate ≈ 0 → y_f ≈ y → baseline
  - After training, gate opens when fusion helps → y_f = fused latent → coding gain
  - Bad / adversarial references → model learns gate → 0 → no harm
  - ref = original → y_a ≈ y → μ_fuse = α·y + (1−α)·y = y → y_f = y (correct)

Inspired by ControlNet zero-init and residual-gating in video codecs (DCVC-HEM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLS(nn.Module):
    """Conditional Latent Synthesis with zero-initialized residual gate."""

    def __init__(self, channels, hidden=128):
        super().__init__()
        # α prediction  (Eq. 14-15 in paper)
        self.weight_net = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )
        # σ² prediction  (for training noise / analysis)
        self.var_net = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, channels, 1), nn.Softplus(),
        )
        # ── ZERO-INIT GATE ──
        # γ initialized to −3 → sigmoid(−3) ≈ 0.047
        # This makes gate nearly closed at init, so the model starts at baseline.
        # During training the gate gradually opens as the model learns to use refs.
        self.gate_logit = nn.Parameter(torch.full((1, channels, 1, 1), -3.0))

    def forward(self, y, y_a):
        """
        Args:
            y:   input latent  [B, C, H, W]
            y_a: aligned ref latent from CLM  [B, C, H, W]
        Returns:
            y_f:  conditional latent  [B, C, H, W]
            info: dict for monitoring
        """
        cat = torch.cat([y, y_a], 1)
        alpha = self.weight_net(cat)            # spatial-adaptive weight
        mu_fuse = alpha * y + (1 - alpha) * y_a # fused mean

        gate = torch.sigmoid(self.gate_logit)   # per-channel gate ∈ (0,1)

        # Gated residual: y_f = y + gate · (fused − y)
        #   gate ≈ 0 → y_f ≈ y          (baseline, no ref influence)
        #   gate → 1 → y_f → mu_fuse    (full ref benefit)
        y_f = y + gate * (mu_fuse - y)

        # Small training-time noise for regularization (disabled at inference)
        if self.training:
            sigma = torch.sqrt(self.var_net(cat) + 1e-8)
            y_f = y_f + sigma * torch.randn_like(y_f) * 0.05

        return y_f, {'alpha': alpha, 'gate': gate.mean().item()}
