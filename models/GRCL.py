"""
GRCL — Generative Reference of Conditional Latents  (TPAMI 2025)

Three key properties by design:
  ① ANY reference image provides coding gain over baseline.
  ② No reference / bad reference → graceful degradation to baseline level.
  ③ reference = original image → near-lossless at extremely low bitrate.

How each property is achieved:

  ① "Always gain" — The Reference Prior path (ref_prior) provides a per-slice
     predicted mean from the reference latents.  Even weakly-correlated refs
     give the entropy model extra conditioning → lower rate.

  ② "Graceful degradation" — Two zero-init gates:
     (a) CLS gate (γ init −3 → ~0.05): CLM+CLS output starts as identity y.
     (b) ref_prior_gate (init −3 → ~0.05): ref prior starts disabled.
     Training with 10% reference-dropout forces the baseline path to be strong.
     Bad refs → gates stay near 0 → model ≈ baseline.

  ③ "ref=original → near-lossless" —
     When ref = original, g_a(ref) = y exactly.  The ref_prior module projects
     the ref latent to predict μ ≈ y for each slice → residual ≈ 0 → ~0 bpp.
     This is a DIRECT conditioning path available at BOTH encoder and decoder
     (since refs are shared), requiring no extra bits.

Architecture overview:
  Encoder:  x → g_a → y ─┬─→ CLM(y, Y_r) → CLS → y_f → entropy code
                          │                        ↑
  Refs:  r_i → g_a → Y_r ┴──→ ref_adapt → ref_feat ─→ entropy context
                          └──→ ref_combine → ref_prior ─→ per-slice μ prior
  Decoder:  r_i → g_a → Y_r → ref_feat + ref_prior → entropy decode → g_s → x̂
"""

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
from compressai.layers import (
    AttentionBlock, ResidualBlock, ResidualBlockUpsample,
    ResidualBlockWithStride, conv3x3, subpel_conv3x3,
)

import torch, torch.nn as nn, torch.nn.functional as F, math, numpy as np
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_, DropPath

from .CLM import CLM, SimpleCLM
from .CLS import CLS

# ── helpers ──────────────────────────────────────────────────────────────────
SCALES_MIN, SCALES_MAX, SCALES_LEVELS = 0.11, 256, 64
def conv1x1(ci, co, s=1): return nn.Conv2d(ci, co, 1, s)
def conv_k(ci, co, k=5, s=2): return nn.Conv2d(ci, co, k, s, k//2)
def get_scale_table(): return torch.exp(torch.linspace(math.log(SCALES_MIN), math.log(SCALES_MAX), SCALES_LEVELS))
def ste_round(x: Tensor) -> Tensor: return torch.round(x) - x.detach() + x

def _upd_buf(mod, n, key, sd):
    buf = next((b for nm, b in mod.named_buffers() if nm == n), None)
    if buf is None: raise RuntimeError(f'buffer "{n}" not registered')
    if buf.numel() == 0: buf.resize_(sd[key].size())
def update_registered_buffers(mod, pfx, names, sd):
    if not mod: return
    for n in names: _upd_buf(mod, n, f"{pfx}.{n}", sd)

# ── Swin blocks (TCM backbone) ──────────────────────────────────────────────
class WMSA(nn.Module):
    def __init__(self, d, od, hd, ws, tp):
        super().__init__()
        self.d,self.hd,self.ws,self.tp=d,hd,ws,tp; self.sc=hd**-0.5; self.nh=d//hd
        self.emb=nn.Linear(d,3*d,True); self.pj=nn.Linear(d,od)
        rp=nn.Parameter(torch.zeros((2*ws-1)**2,self.nh)); trunc_normal_(rp,std=.02)
        self.rp=nn.Parameter(rp.view(2*ws-1,2*ws-1,self.nh).transpose(1,2).transpose(0,1))
    def _mask(self,h,w,p,s):
        m=torch.zeros(h,w,p,p,p,p,dtype=torch.bool,device=self.rp.device)
        if self.tp=='W': return m
        t=p-s; m[-1,:,:t,:,t:,:]=True;m[-1,:,t:,:,:t,:]=True;m[:,-1,:,:t,:,t:]=True;m[:,-1,:,t:,:,:t]=True
        return rearrange(m,'a b c d e f->1 1 (a b)(c d)(e f)')
    def _rel(self):
        c=torch.tensor([[i,j] for i in range(self.ws) for j in range(self.ws)])
        r=c[:,None,:]-c[None,:,:]+self.ws-1; return self.rp[:,r[:,:,0].long(),r[:,:,1].long()]
    def forward(self,x):
        ws=self.ws
        if self.tp!='W': x=torch.roll(x,(-(ws//2),-(ws//2)),(1,2))
        x=rearrange(x,'b(h p1)(w p2)c->b h w p1 p2 c',p1=ws,p2=ws); H,W=x.size(1),x.size(2)
        x=rearrange(x,'b h w p1 p2 c->b(h w)(p1 p2)c',p1=ws,p2=ws)
        q,k,v=rearrange(self.emb(x),'b n s(t c)->t b n s c',c=self.hd).chunk(3,0)
        s=torch.einsum('hbwpc,hbwqc->hbwpq',q,k)*self.sc+rearrange(self._rel(),'h p q->h 1 1 p q')
        if self.tp!='W': s=s.masked_fill_(self._mask(H,W,ws,ws//2),float('-inf'))
        o=torch.einsum('hbwij,hbwjc->hbwic',F.softmax(s,-1),v)
        o=self.pj(rearrange(o,'h b w p c->b w p(h c)'))
        o=rearrange(o,'b(h w)(p1 p2)c->b(h p1)(w p2)c',h=H,p1=ws)
        if self.tp!='W': o=torch.roll(o,(ws//2,ws//2),(1,2))
        return o

class _Blk(nn.Module):
    def __init__(self,d,hd,ws,dp,tp):
        super().__init__(); self.ln1=nn.LayerNorm(d); self.msa=WMSA(d,d,hd,ws,tp)
        self.dp=DropPath(dp) if dp>0 else nn.Identity(); self.ln2=nn.LayerNorm(d)
        self.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(self,x): x=x+self.dp(self.msa(self.ln1(x))); return x+self.dp(self.mlp(self.ln2(x)))

class SwinBlock(nn.Module):
    def __init__(self,d,hd,ws,dp):
        super().__init__(); self.b1=_Blk(d,hd,ws,dp,'W'); self.b2=_Blk(d,hd,ws,dp,'SW'); self.ws=ws
    def forward(self,x):
        if x.size(-1)<=self.ws or x.size(-2)<=self.ws:
            pr=(self.ws-x.size(-2))//2; pc=(self.ws-x.size(-1))//2; x=F.pad(x,(pc,pc+1,pr,pr+1))
        t=Rearrange('b c h w->b h w c')(x); t=self.b2(self.b1(t)); return Rearrange('b h w c->b c h w')(t)

class CTB(nn.Module):
    def __init__(self,cd,td,hd,ws,dp,tp='W'):
        super().__init__(); self.cd,self.td=cd,td; self.tb=_Blk(td,hd,ws,dp,tp)
        self.c1=nn.Conv2d(cd+td,cd+td,1); self.c2=nn.Conv2d(cd+td,cd+td,1); self.cb=ResidualBlock(cd,cd)
    def forward(self,x):
        cx,tx=torch.split(self.c1(x),(self.cd,self.td),1); cx=self.cb(cx)+cx
        tx=Rearrange('b c h w->b h w c')(tx); tx=self.tb(tx); tx=Rearrange('b h w c->b c h w')(tx)
        return x+self.c2(torch.cat((cx,tx),1))

class SWA(AttentionBlock):
    def __init__(self,id,od,hd,ws,dp,n=192):
        super().__init__(N=n or id); self.nlb=SwinBlock(n or id,hd,ws,dp)
        if n: self.ic=conv1x1(id,n); self.oc=conv1x1(n,od)
    def forward(self,x):
        x=self.ic(x); z=self.nlb(x); return self.oc(self.conv_a(x)*torch.sigmoid(self.conv_b(z))+x)


# ═══════════════════════════════════════════════════════════════════════════════
#  GRCL Model
# ═══════════════════════════════════════════════════════════════════════════════

class GRCL(CompressionModel):

    def __init__(self, config=[2]*6, head_dim=[8,16,32,32,16,8],
                 drop_path_rate=0, N=128, M=320, num_slices=5,
                 max_support_slices=5, num_ref=3, clm_type='simple',
                 ref_dropout=0.1, **kw):
        super().__init__(entropy_bottleneck_channels=N)
        self.num_slices, self.max_ss = num_slices, max_support_slices
        self.num_ref, self.M, self.N, self.ws = num_ref, M, N, 8
        self.ref_dropout = ref_dropout      # prob to drop refs during training
        S = M // num_slices                 # per-slice channels
        ref_dim = 64                        # compact ref feature dim
        dim = N
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        def _ctb(ci, hd):
            return [CTB(dim,dim,hd,self.ws,dpr[i],'W' if not i%2 else 'SW') for i in range(config[ci])]

        # ── g_a / g_s ─────────────────────────────────────────────
        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3,2*N,2), *_ctb(0,head_dim[0]), ResidualBlockWithStride(2*N,2*N,2),
            *_ctb(1,head_dim[1]), ResidualBlockWithStride(2*N,2*N,2),
            *_ctb(2,head_dim[2]), conv3x3(2*N,M,stride=2))
        self.g_s = nn.Sequential(
            ResidualBlockUpsample(M,2*N,2), *_ctb(3,head_dim[3]), ResidualBlockUpsample(2*N,2*N,2),
            *_ctb(4,head_dim[4]), ResidualBlockUpsample(2*N,2*N,2),
            *_ctb(5,head_dim[5]), subpel_conv3x3(2*N,3,2))

        # ── CLM + CLS (encoder-only, with zero-init gate) ─────────
        self.clm = (CLM(M,min(M,256),0.07,num_ref) if clm_type=='full'
                     else SimpleCLM(M,0.07,num_ref))
        self.cls = CLS(M, 128)

        # ── Hyperprior ─────────────────────────────────────────────
        self.h_a = nn.Sequential(
            ResidualBlockWithStride(M,2*N,2),
            *[CTB(N,N,32,4,0,'W' if not i%2 else 'SW') for i in range(config[0])],
            conv3x3(2*N,192,stride=2))
        def _hs():
            return nn.Sequential(ResidualBlockUpsample(192,2*N,2),
                *[CTB(N,N,32,4,0,'W' if not i%2 else 'SW') for i in range(config[3])],
                subpel_conv3x3(2*N,M,2))
        self.h_mean_s = _hs(); self.h_scale_s = _hs()

        # ── Compact ref feature adapter ────────────────────────────
        self.ref_adapt = nn.Sequential(conv1x1(M*num_ref,128), nn.GELU(), conv1x1(128,ref_dim))

        # ══════════════════════════════════════════════════════════
        #  REFERENCE PRIOR — direct conditioning path  (key for ③)
        #  Projects the combined ref latent → per-slice mean prior.
        #  Available at BOTH encoder & decoder (refs are shared).
        #  When ref=original: ref_prior ≈ y_slice → μ ≈ y → rate ≈ 0.
        # ══════════════════════════════════════════════════════════
        self.ref_combine = nn.Sequential(           # combine N refs → 1
            conv1x1(M * num_ref, M), nn.GELU(), conv1x1(M, M))
        self.ref_prior = nn.ModuleList([            # per-slice μ prior
            nn.Sequential(nn.Conv2d(M, 128, 3, padding=1), nn.GELU(),
                          nn.Conv2d(128, S, 3, padding=1))
            for _ in range(num_slices)])
        # Zero-init gate for ref_prior (init −3 → gate ≈ 0.05)
        self.ref_prior_gate = nn.Parameter(torch.full((1, S, 1, 1), -3.0))

        # ── Entropy context transforms ─────────────────────────────
        def _cc(ex):
            return nn.ModuleList(nn.Sequential(
                conv_k(M+S*min(i,5)+ex, 224, 3, 1), nn.GELU(),
                conv_k(224, 128, 3, 1), nn.GELU(),
                conv_k(128, S, 3, 1)) for i in range(num_slices))
        def _lrp(ex):
            return nn.ModuleList(nn.Sequential(
                conv_k(M+S*min(i+1,6)+ex, 224, 3, 1), nn.GELU(),
                conv_k(224, 128, 3, 1), nn.GELU(),
                conv_k(128, S, 3, 1)) for i in range(num_slices))
        def _att():
            return nn.ModuleList(nn.Sequential(
                SWA(M+S*min(i,5), M+S*min(i,5), 16, self.ws, 0, 128))
                for i in range(num_slices))

        self.att_mu = _att(); self.att_sc = _att()
        self.cc_mu  = _cc(0);   self.cc_sc  = _cc(0)       # baseline
        self.rcc_mu = _cc(ref_dim); self.rcc_sc = _cc(ref_dim)  # ref-enhanced
        self.lrp    = _lrp(0);  self.rlrp   = _lrp(ref_dim)

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

    # ── helpers ───────────────────────────────────────────────────────────────

    def update(self, scale_table=None, force=False):
        up = self.gaussian_conditional.update_scale_table(scale_table or get_scale_table(), force)
        return up | super().update(force=force)

    def _encode_refs(self, refs):
        """Encode refs.  Returns (ref_latents, ref_feat, ref_combined).
           ref_combined is the merged single ref latent for the ref_prior path.
           All three are None when refs is None."""
        if refs is None:
            return None, None, None
        lats = [self.g_a(r).detach() for r in refs]
        cat_lats = torch.cat(lats, 1)                        # compute once, reuse
        feat = self.ref_adapt(cat_lats)                       # compact [B, 64, H, W]
        combined = self.ref_combine(cat_lats)                 # full   [B, M,  H, W]
        return lats, feat, combined

    def _clm_cls(self, y, ref_lats):
        """CLM+CLS at encoder side.  Returns y_f identical to y when refs=None."""
        if ref_lats is None:
            return y, None
        y_a = self.clm(y, ref_lats)
        y_f, info = self.cls(y, y_a)
        return y_f, info

    # ── slice loop (shared by forward / compress / decompress) ────────────────

    def _slice_loop(self, y_f, lat_mu, lat_sc, ref_feat, ref_combined, y_shape, mode):
        slices_in = y_f.chunk(self.num_slices, 1) if mode != 'decompress' else [None]*self.num_slices
        hat, mu_l, sc_l, lh_l = [], [], [], []
        has_ref = ref_feat is not None
        rp_gate = torch.sigmoid(self.ref_prior_gate)     # per-slice ref-prior gate

        for i in range(self.num_slices):
            sup = hat if self.max_ss < 0 else hat[:self.max_ss]
            m_sup = self.att_mu[i](torch.cat([lat_mu]+sup, 1))
            s_sup = self.att_sc[i](torch.cat([lat_sc]+sup, 1))

            # ── baseline μ / σ  ──────────────────────────────────
            if has_ref:
                mu = self.rcc_mu[i](torch.cat([m_sup, ref_feat], 1))
                sc = self.rcc_sc[i](torch.cat([s_sup, ref_feat], 1))
            else:
                mu = self.cc_mu[i](m_sup)
                sc = self.cc_sc[i](s_sup)
            mu = mu[:,:,:y_shape[0],:y_shape[1]]
            sc = sc[:,:,:y_shape[0],:y_shape[1]]

            # ── REFERENCE PRIOR (direct path) ────────────────────
            # Adds a gated, per-slice mean correction from the ref latent.
            # When ref = original: ref_prior_i ≈ y_slice → mu + correction ≈ y → rate ≈ 0
            if has_ref:
                rp_i = self.ref_prior[i](ref_combined)
                rp_i = rp_i[:,:,:y_shape[0],:y_shape[1]]
                mu = mu + rp_gate * rp_i     # additive, gated

            mu_l.append(mu); sc_l.append(sc)

            # ── quantize / likelihood ────────────────────────────
            if mode == 'forward':
                _, lh = self.gaussian_conditional(slices_in[i], sc, mu)
                lh_l.append(lh)
                yh = ste_round(slices_in[i] - mu) + mu
            elif mode == 'compress':
                idx = self.gaussian_conditional.build_indexes(sc)
                yq = self.gaussian_conditional.quantize(slices_in[i], "symbols", mu)
                self._sym.extend(yq.reshape(-1).tolist())
                self._idx.extend(idx.reshape(-1).tolist())
                yh = yq + mu
            else:   # decompress
                idx = self.gaussian_conditional.build_indexes(sc)
                rv = self._dec.decode_stream(idx.reshape(-1).tolist(), *self._cdfs)
                rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1]).to(mu.device)
                yh = self.gaussian_conditional.dequantize(rv, mu)

            # ── LRP ──────────────────────────────────────────────
            if has_ref:
                lrp = self.rlrp[i](torch.cat([m_sup, yh, ref_feat], 1))
            else:
                lrp = self.lrp[i](torch.cat([m_sup, yh], 1))
            hat.append(yh + 0.5 * torch.tanh(lrp))

        return hat, mu_l, sc_l, lh_l

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x, ref_frames=None):
        # ── REFERENCE DROPOUT (training only) ────────────────────
        # With probability ref_dropout, ignore all refs.
        # Forces baseline path to stay strong → property ②.
        if self.training and ref_frames is not None and torch.rand(1).item() < self.ref_dropout:
            ref_frames = None

        y = self.g_a(x)
        ref_lats, ref_feat, ref_comb = self._encode_refs(ref_frames)
        y_f, cls_info = self._clm_cls(y, ref_lats)

        z = self.h_a(y_f); _, z_lh = self.entropy_bottleneck(z)
        z_off = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_off) + z_off
        lat_mu = self.h_mean_s(z_hat); lat_sc = self.h_scale_s(z_hat)

        hat, _, _, lh_l = self._slice_loop(y_f, lat_mu, lat_sc, ref_feat, ref_comb, y_f.shape[2:], 'forward')
        return {'x_hat': self.g_s(torch.cat(hat, 1)),
                'likelihoods': {'y': torch.cat(lh_l, 1), 'z': z_lh}}

    # ── compress ─────────────────────────────────────────────────────────────

    def compress(self, x, ref_frames=None):
        y = self.g_a(x)
        ref_lats, ref_feat, ref_comb = self._encode_refs(ref_frames)
        y_f, _ = self._clm_cls(y, ref_lats)

        z = self.h_a(y_f); z_str = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_str, z.size()[-2:])
        lat_mu = self.h_mean_s(z_hat); lat_sc = self.h_scale_s(z_hat)

        self._sym, self._idx = [], []
        self._cdfs = (self.gaussian_conditional.quantized_cdf.tolist(),
                      self.gaussian_conditional.cdf_length.reshape(-1).int().tolist(),
                      self.gaussian_conditional.offset.reshape(-1).int().tolist())
        self._slice_loop(y_f, lat_mu, lat_sc, ref_feat, ref_comb, y_f.shape[2:], 'compress')
        enc = BufferedRansEncoder(); enc.encode_with_indexes(self._sym, self._idx, *self._cdfs)
        return {'strings': [[enc.flush()], z_str], 'shape': z.size()[-2:]}

    # ── decompress ───────────────────────────────────────────────────────────

    def decompress(self, strings, shape, ref_frames=None):
        # Decoder: ref_feat and ref_combined are available (same refs at both sides)
        _, ref_feat, ref_comb = self._encode_refs(ref_frames)
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        lat_mu = self.h_mean_s(z_hat); lat_sc = self.h_scale_s(z_hat)
        y_shape = [z_hat.shape[2]*4, z_hat.shape[3]*4]

        self._cdfs = (self.gaussian_conditional.quantized_cdf.tolist(),
                      self.gaussian_conditional.cdf_length.reshape(-1).int().tolist(),
                      self.gaussian_conditional.offset.reshape(-1).int().tolist())
        self._dec = RansDecoder(); self._dec.set_stream(strings[0][0])
        hat, *_ = self._slice_loop(None, lat_mu, lat_sc, ref_feat, ref_comb, y_shape, 'decompress')
        return {'x_hat': self.g_s(torch.cat(hat, 1)).clamp_(0, 1)}

    # ── state dict ───────────────────────────────────────────────────────────

    def load_state_dict(self, state_dict):
        update_registered_buffers(self.gaussian_conditional, "gaussian_conditional",
                                  ["_quantized_cdf","_offset","_cdf_length","scale_table"], state_dict)
        own = self.state_dict()
        super().load_state_dict({k:v for k,v in state_dict.items() if k in own}, strict=False)
