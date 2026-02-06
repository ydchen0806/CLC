"""
Conditional Latent Synthesis (CLS) Module for GRCL.

Formulation:
    μ(y, y_a) = α ⊙ y + (1 − α) ⊙ y_a
    α = σ{F_w([y, y_a]; θ_f)}      (two 3×3 convs with 128 ch + 1×1 + sigmoid)
    σ²(y, y_a) predicted by separate network
"""

import torch
import torch.nn as nn


class CLS(nn.Module):
    """Conditional Latent Synthesis: adaptive fusion of input and aligned reference."""
    def __init__(self, channels, hidden=128):
        super().__init__()
        # α prediction (paper Eq. 14-15)
        self.weight_net = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )
        # σ² prediction
        self.var_net = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, channels, 1), nn.Softplus(),
        )

    def forward(self, y, y_a):
        """
        Returns:
            y_f: conditional latent (deterministic at eval, noised at train)
            info: dict with alpha, sigma for analysis / regularization
        """
        cat = torch.cat([y, y_a], 1)
        alpha = self.weight_net(cat)
        mu = alpha * y + (1 - alpha) * y_a
        sigma = torch.sqrt(self.var_net(cat) + 1e-8)

        if self.training:
            y_f = mu + sigma * torch.randn_like(mu) * 0.1   # small noise for stability
        else:
            y_f = mu  # deterministic at inference (paper Sec. IV-B note)

        return y_f, {'mu': mu, 'sigma': sigma, 'alpha': alpha}
