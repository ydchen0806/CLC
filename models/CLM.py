"""
Conditional Latent Matching (CLM) Module for GRCL.

Two implementations:
- CLM: Full version with similarity matching + multi-scale deformable alignment.
- SimpleCLM: Lightweight version with cross-attention + grid-based warping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class FeatureTransform(nn.Module):
    """φ(·): 3×3 CNN with BN+ReLU to lift features into a richer space."""
    def __init__(self, in_ch, out_ch=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class SimilarityMatching(nn.Module):
    """S_ij = softmax(<φ(y_i), φ(y_r,j)> / τ);  y_m = Σ_j S_ij · y_r,j"""
    def __init__(self, channels, feat_dim=512, temperature=0.07):
        super().__init__()
        self.tau = temperature
        self.phi = FeatureTransform(channels, feat_dim)

    def forward(self, y, y_ref):
        B, C, H, W = y.shape
        phi_y = self.phi(y).flatten(2)           # [B, D, HW]
        phi_r = self.phi(y_ref).flatten(2)       # [B, D, HW]
        sim = torch.bmm(phi_y.transpose(1, 2), phi_r) / self.tau  # [B, HW, HW]
        sim = F.softmax(sim, dim=-1)
        y_r_flat = y_ref.flatten(2).transpose(1, 2)               # [B, HW, C]
        y_m = torch.bmm(sim, y_r_flat).transpose(1, 2).reshape(B, C, H, W)
        return y_m


class MultiScaleDeformAlign(nn.Module):
    """Cascaded deformable conv at 5×5/3×3/1×1 scales."""
    def __init__(self, ch):
        super().__init__()
        # 5×5 coarse
        self.off5 = nn.Sequential(nn.Conv2d(ch*2, ch, 3, padding=1), nn.GELU(), nn.Conv2d(ch, 2*25, 3, padding=1))
        self.mod5 = nn.Sequential(nn.Conv2d(ch*2, ch, 3, padding=1), nn.GELU(), nn.Conv2d(ch, 25, 3, padding=1))
        self.w5 = nn.Parameter(torch.empty(ch, ch, 5, 5)); nn.init.kaiming_normal_(self.w5)
        self.b5 = nn.Parameter(torch.zeros(ch))
        # 3×3 medium
        self.off3 = nn.Sequential(nn.Conv2d(ch*2, ch, 3, padding=1), nn.GELU(), nn.Conv2d(ch, 2*9, 3, padding=1))
        self.mod3 = nn.Sequential(nn.Conv2d(ch*2, ch, 3, padding=1), nn.GELU(), nn.Conv2d(ch, 9, 3, padding=1))
        self.w3 = nn.Parameter(torch.empty(ch, ch, 3, 3)); nn.init.kaiming_normal_(self.w3)
        self.b3 = nn.Parameter(torch.zeros(ch))
        # 1×1 fine
        self.fine = nn.Sequential(nn.Conv2d(ch*2, ch, 1), nn.GELU(), nn.Conv2d(ch, ch, 1))
        # fusion
        self.fuse = nn.Sequential(nn.Conv2d(ch*3, ch, 1), nn.GELU(), nn.Conv2d(ch, ch, 1))

    def forward(self, y, y_m):
        cat = torch.cat([y, y_m], 1)
        a5 = F.gelu(deform_conv2d(y_m, self.off5(cat), self.w5, self.b5, padding=2, mask=torch.sigmoid(self.mod5(cat))))
        a3 = F.gelu(deform_conv2d(y_m, self.off3(cat), self.w3, self.b3, padding=1, mask=torch.sigmoid(self.mod3(cat))))
        a1 = self.fine(cat)
        return self.fuse(torch.cat([a5, a3, a1], 1))


class CLM(nn.Module):
    """Full CLM: similarity matching → deformable alignment → multi-ref attention fusion."""
    def __init__(self, channels, feat_dim=256, temperature=0.07, num_refs=3):
        super().__init__()
        self.sim_match = SimilarityMatching(channels, feat_dim, temperature)
        self.deform_align = MultiScaleDeformAlign(channels)
        self.ref_attn = nn.Sequential(nn.Conv2d(channels, channels//4, 1), nn.GELU(), nn.Conv2d(channels//4, 1, 1))
        self.fusion = nn.Sequential(nn.Conv2d(channels*2, channels, 3, padding=1), nn.GELU(), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, y, y_refs):
        aligned, weights = [], []
        for yr in y_refs:
            ym = self.sim_match(y, yr)
            ya = self.deform_align(y, ym)
            aligned.append(ya); weights.append(self.ref_attn(ya))
        w = F.softmax(torch.stack(weights, 1), dim=1)           # [B, M, 1, H, W]
        fused = (torch.stack(aligned, 1) * w).sum(1)            # [B, C, H, W]
        return self.fusion(torch.cat([y, fused], 1))


class SimpleCLM(nn.Module):
    """Lightweight CLM with Q/K/V projections + grid-sample warping."""
    def __init__(self, channels, temperature=0.07, num_refs=3):
        super().__init__()
        self.C = channels; self.tau = temperature
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.off = nn.Sequential(nn.Conv2d(channels*2, channels, 3, padding=1), nn.GELU(), nn.Conv2d(channels, 2, 3, padding=1))
        self.ref_attn = nn.Conv2d(channels, 1, 1)
        self.fusion = nn.Sequential(nn.Conv2d(channels*2, channels, 3, padding=1), nn.GELU(), nn.Conv2d(channels, channels, 3, padding=1))

    def _warp(self, feat, offset):
        B, _, H, W = feat.shape
        gy, gx = torch.meshgrid(torch.linspace(-1, 1, H, device=feat.device), torch.linspace(-1, 1, W, device=feat.device), indexing='ij')
        grid = torch.stack([gx, gy], -1).unsqueeze(0).expand(B, -1, -1, -1)
        off_n = offset.permute(0, 2, 3, 1) * 2.0 / torch.tensor([W, H], device=feat.device).float()
        return F.grid_sample(feat, grid + off_n, mode='bilinear', padding_mode='zeros', align_corners=True)

    def forward(self, y, y_refs):
        B, C, H, W = y.shape; aligned, weights = [], []
        query = self.q(y)
        for yr in y_refs:
            key, val = self.k(yr), self.v(yr)
            attn = F.softmax(torch.bmm(query.flatten(2).transpose(1,2), key.flatten(2)) / (C**0.5 * self.tau), dim=-1)
            matched = torch.bmm(attn, val.flatten(2).transpose(1,2)).transpose(1,2).reshape(B, C, H, W)
            al = self._warp(matched, self.off(torch.cat([y, matched], 1)))
            aligned.append(al); weights.append(self.ref_attn(al))
        w = F.softmax(torch.stack(weights, 1), dim=1)
        fused = (torch.stack(aligned, 1) * w).sum(1)
        return self.fusion(torch.cat([y, fused], 1))
