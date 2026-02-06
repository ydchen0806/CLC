"""
Conditional Latent Synthesis (CLS) Module for GRCL.

Key design: ZERO-INIT GATED RESIDUAL
    y_f = y + gate · (μ_fuse − y)
    gate = sigmoid(γ)   where γ is initialized to −3 → sigmoid(−3) ≈ 0.05

Properties:
  - gate ≈ 0  → y_f ≈ y  (baseline, no ref influence)
  - gate → 1  → y_f = fused  (full ref benefit)

NOTE on training noise: REMOVED.
  Previous version added σ·ε noise during training but not inference, causing a
  distribution shift between train and test.  The STE quantization in the main
  model already provides sufficient stochastic regularization.  Adding extra
  noise on y_f means the entropy model trains on a noisier distribution than
  what it sees at test time, leading to sub-optimal rate estimation.
  (Similar issue documented in CompressAI issue #164 and ELIC paper Sec. 3.3)
"""

import torch
import torch.nn as nn


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
        # Zero-init gate:  γ = −3 → sigmoid ≈ 0.047
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
        alpha = self.weight_net(torch.cat([y, y_a], 1))   # spatial weight ∈ (0,1)
        mu_fuse = alpha * y + (1 - alpha) * y_a            # fused mean

        gate = torch.sigmoid(self.gate_logit)               # per-channel gate ∈ (0,1)
        y_f = y + gate * (mu_fuse - y)                      # gated residual

        # No training noise — STE quantization in GRCL._slice_loop already
        # provides the stochastic perturbation the entropy model needs.
        # Adding extra noise here would create a train/test mismatch.

        return y_f, {'alpha': alpha, 'gate': gate.mean().item()}
