# -*- coding: utf-8 -*-
"""Conditional flow distributions (spline / tanh-of-affine / log-normal)."""

from __future__ import annotations
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE = torch.float32
SQRT2PI_LOG = 0.5 * math.log(2.0 * math.pi)


# PATCH 2026-04-20 (eps ablation pack):
# Module-level activation switch for the conditioning MLPs. Default preserves
# the original Tanh behaviour. Call `set_activation("relu")` (or "silu",
# "gelu") BEFORE `build_dist(...)` to construct flows with a different
# activation -- this is Test 3 in the eps-ablation study, testing whether
# the Tanh-induced odd symmetry at init contributes to the eps-channel
# collapse.
_ACTIVATION = "tanh"


def set_activation(name: str):
    global _ACTIVATION
    if name not in ("tanh", "relu", "silu", "gelu"):
        raise ValueError(f"unknown activation: {name}")
    _ACTIVATION = name


def _act_cls():
    return {"tanh": nn.Tanh, "relu": nn.ReLU,
            "silu": nn.SiLU, "gelu": nn.GELU}[_ACTIVATION]


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    A = _act_cls()
    return nn.Sequential(
        nn.Linear(in_dim, hidden), A(),
        nn.Linear(hidden, hidden), A(),
        nn.Linear(hidden, out_dim),
    )


# ---------------------------------------------------------------------------
# 1. Conditional log-normal AFT (baseline; same as the original CondLogNormalFlow)
# ---------------------------------------------------------------------------
class CondLogNormal(nn.Module):
    family = "lognormal"

    def __init__(self, ctx_dim: int = 1, hidden: int = 32):
        super().__init__()
        self.head = _mlp(ctx_dim, hidden, 2)

    def params(self, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.head(ctx)
        mu = h[..., 0:1]
        sigma = F.softplus(h[..., 1:2]) + 0.05
        return mu, sigma

    def log_pdf(self, y, ctx):
        y = torch.clamp(y, min=1e-8)
        mu, sigma = self.params(ctx)
        logy = torch.log(y)
        u = (logy - mu) / sigma
        return -SQRT2PI_LOG - torch.log(sigma) - 0.5 * u * u - logy

    def log_survival(self, y, ctx):
        y = torch.clamp(y, min=1e-8)
        mu, sigma = self.params(ctx)
        u = (torch.log(y) - mu) / sigma
        return torch.log(torch.clamp(torch.special.ndtr(-u), min=1e-12))

    def sample(self, ctx, generator=None):
        mu, sigma = self.params(ctx)
        eps = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
        return torch.exp(mu + sigma * eps)


# ---------------------------------------------------------------------------
# 2. Conditional Weibull AFT
#    log T = mu(z) + sigma(z) * eta,  eta ~ standard Gumbel(min)
#    => T ~ Weibull(scale=exp(mu), shape=1/sigma).  Closed-form S, f.
# ---------------------------------------------------------------------------
class CondWeibull(nn.Module):
    family = "weibull"

    def __init__(self, ctx_dim: int = 1, hidden: int = 32):
        super().__init__()
        self.head = _mlp(ctx_dim, hidden, 2)

    def params(self, ctx):
        h = self.head(ctx)
        log_lam = h[..., 0:1]                                # log scale
        log_k = torch.tanh(h[..., 1:2]) * 1.5                # bound shape exponent
        k = torch.exp(log_k)
        return log_lam, k

    def log_pdf(self, y, ctx):
        y = torch.clamp(y, min=1e-8)
        log_lam, k = self.params(ctx)
        # log f(y) = log k - log lam + (k-1)*(log y - log lam) - (y/lam)^k
        logy_minus_loglam = torch.log(y) - log_lam
        return torch.log(k) - log_lam + (k - 1.0) * logy_minus_loglam - torch.exp(k * logy_minus_loglam)

    def log_survival(self, y, ctx):
        y = torch.clamp(y, min=1e-8)
        log_lam, k = self.params(ctx)
        return -torch.exp(k * (torch.log(y) - log_lam))

    def sample(self, ctx, generator=None):
        log_lam, k = self.params(ctx)
        u = torch.rand(log_lam.shape, generator=generator, device=log_lam.device, dtype=log_lam.dtype)
        u = torch.clamp(u, min=1e-7, max=1 - 1e-7)
        # T = lam * (-log(1-u))^(1/k)
        return torch.exp(log_lam) * torch.pow(-torch.log1p(-u), 1.0 / k)


# ---------------------------------------------------------------------------
# 3. Conditional mixture of K log-normals
# ---------------------------------------------------------------------------
class CondMixLogNormal(nn.Module):
    family = "mixln"

    def __init__(self, ctx_dim: int = 1, hidden: int = 32, K: int = 3):
        super().__init__()
        self.K = K
        self.head = _mlp(ctx_dim, hidden, 3 * K)

    def params(self, ctx):
        h = self.head(ctx)                                   # [n, 3K]
        mu = h[..., :self.K]
        sigma = F.softplus(h[..., self.K:2 * self.K]) + 0.05
        log_pi = F.log_softmax(h[..., 2 * self.K:], dim=-1)
        return mu, sigma, log_pi

    def _log_components(self, y, ctx):
        # returns log f_k(y|ctx) and log S_k(y|ctx) of shape [n, K]
        y = torch.clamp(y, min=1e-8).expand(-1, self.K) if y.shape[-1] == 1 else y
        mu, sigma, log_pi = self.params(ctx)
        logy = torch.log(y)
        u = (logy - mu) / sigma
        log_fk = -SQRT2PI_LOG - torch.log(sigma) - 0.5 * u * u - logy
        log_Sk = torch.log(torch.clamp(torch.special.ndtr(-u), min=1e-12))
        return log_fk, log_Sk, log_pi

    def log_pdf(self, y, ctx):
        log_fk, _, log_pi = self._log_components(y, ctx)
        return torch.logsumexp(log_pi + log_fk, dim=-1, keepdim=True)

    def log_survival(self, y, ctx):
        _, log_Sk, log_pi = self._log_components(y, ctx)
        return torch.logsumexp(log_pi + log_Sk, dim=-1, keepdim=True)

    def sample(self, ctx, generator=None, temp: float = 0.0):
        """
        Sample from the conditional mixture of log-normals.

        temp = 0.0 (default): HARD argmax-based sample (non-differentiable
                              w.r.t. mixture weights, identical to prior
                              behaviour).
        temp > 0.0          : STRAIGHT-THROUGH Gumbel-Softmax.  Forward
                              returns a hard sample (identical to argmax);
                              backward passes gradient through a softmax of
                              (log_pi + gumbel)/temp.  This is what lets the
                              InfoNCE MI regulariser push gradients through
                              flow.sample() for mixture flows (argmax alone
                              silently kills those gradients).
        """
        mu, sigma, log_pi = self.params(ctx)                          # [n, K]
        K = mu.shape[-1]

        # Gumbel noise for component selection
        U_g = torch.clamp(
            torch.rand(log_pi.shape, generator=generator,
                       device=mu.device, dtype=mu.dtype),
            min=1e-9, max=1 - 1e-9)
        gumbel = -torch.log(-torch.log(U_g))                          # [n, K]
        logits = log_pi + gumbel                                      # [n, K]

        # Per-component standard-normal noise (one draw per component per
        # subject); allows the whole mixture to be represented upfront.
        eps_comp = torch.randn(mu.shape, generator=generator,
                               device=mu.device, dtype=mu.dtype)      # [n, K]
        t_components = torch.exp(mu + sigma * eps_comp)               # [n, K]

        if temp <= 0.0:
            # Hard argmax sample (non-differentiable). Equivalent to the
            # original implementation up to redundant per-component noise.
            k_idx = torch.argmax(logits, dim=-1, keepdim=True)        # [n, 1]
            return torch.gather(t_components, -1, k_idx)              # [n, 1]

        # Straight-through Gumbel-Softmax: forward = hard, backward = soft.
        weights_soft = F.softmax(logits / temp, dim=-1)               # [n, K]
        k_idx = torch.argmax(logits, dim=-1, keepdim=True)
        weights_hard = torch.zeros_like(weights_soft).scatter_(-1, k_idx, 1.0)
        weights_st = weights_hard.detach() + weights_soft - weights_soft.detach()
        return (weights_st * t_components).sum(dim=-1, keepdim=True)  # [n, 1]


# ---------------------------------------------------------------------------
# 4. Conditional monotone spline flow on log-time
#    log T = a(z) + sum_j w_j(z) * softplus(eta - b_j(z))
#    w_j > 0  =>  log T is monotone increasing in eta  =>  invertible.
#    Inverse and survival via vectorised bisection on the closed-form forward.
# ---------------------------------------------------------------------------
class CondMonotoneSplineFlow(nn.Module):
    """
    YL rewrite 2026-04-09. The original implementation defined the forward
    direction as eta -> log T (sum-of-softplus on eta) and then evaluated
    log_pdf via inverse + bisection. That had three fatal bugs:
      (1) bisection-as-inverse killed gradient flow into theta because the
          comparison `(f_mid < target).to(dtype)` is non-differentiable, so
          the implicit-function term was missing entirely;
      (2) bisection bracket was hardcoded [-8, 8] regardless of where the
          forward image lay at init, so most samples returned the boundary
          and produced log_p_eta = -32-ish, blowing up loss;
      (3) no grad clip + (1)+(2) -> first step nuked params -> NaN forward.
    The result was that 22/30 seeds collapsed at converged_epoch <= 4 with
    best_loss stuck at ~12.5.

    The fix is structural: define the forward direction as log_t -> eta with
    a strictly-monotone sum-of-softplus map, so log_pdf is closed-form and
    NEVER invokes the inverse during training. The inverse is only used for
    sampling, where we reattach gradients via Figurnov 2018 implicit
    reparameterization. Same approach as MonotoneFlow1D in
    uninformative_censoring_flow.py, just with softplus basis instead of
    tanh basis to keep the "spline" naming aligned with the design
    spec.

    Forward map (ctx-conditioned):
        h(log_t; ctx) = a(ctx) * log_t + b(ctx)
                        + sum_{j=1..J} w_j(ctx) * softplus(log_t - mu_j(ctx))
    Constraints (via softplus reparam): a > 0, w_j >= 0
    => h is strictly increasing in log_t, hence invertible on R.

    Base distribution: U = h(log_T; ctx) ~ N(0, 1).
    => log f_T(t|ctx) = log phi(h) + log h' - log t
       log S_T(t|ctx) = log Phi(-h)
    where h' = a + sum_j w_j * sigmoid(log_t - mu_j).
    """
    family = "spline"

    def __init__(self, ctx_dim: int = 1, hidden: int = 32, n_knots: int = 6,
                 log_t_min: float = math.log(1e-4), log_t_max: float = math.log(10.0)):
        super().__init__()
        self.J = n_knots
        self.log_t_min = float(log_t_min)
        self.log_t_max = float(log_t_max)
        # heads: a(1), b(1), w(J), mu(J)  -> 2 + 2J outputs
        self.head = _mlp(ctx_dim, hidden, 2 + 2 * n_knots)

    def params(self, ctx):
        h = self.head(ctx)
        J = self.J
        a_raw = h[..., 0:1]
        b = h[..., 1:2]
        w_raw = h[..., 2:2 + J]
        mu = h[..., 2 + J:2 + 2 * J]
        a = F.softplus(a_raw) + 0.10           # slope baseline > 0
        w = F.softplus(w_raw) * (1.0 / J)      # >= 0, scaled by 1/J for stability
        return a, b, w, mu

    def h_and_dh(self, log_t, ctx):
        """Returns (h, dh/d log_t), both shape [n, 1]."""
        a, b, w, mu = self.params(ctx)
        diff = log_t - mu                              # [n, J]
        sp = F.softplus(diff)                          # [n, J]
        sigm = torch.sigmoid(diff)                     # [n, J]
        h = a * log_t + b + (w * sp).sum(dim=-1, keepdim=True)
        dh = a + (w * sigm).sum(dim=-1, keepdim=True)
        return h, dh

    def log_pdf(self, y, ctx):
        eps = 1e-8
        y_safe = torch.clamp(y, min=eps)
        log_t = torch.log(y_safe)
        h, dh = self.h_and_dh(log_t, ctx)
        log_phi = -SQRT2PI_LOG - 0.5 * h * h
        return log_phi + torch.log(torch.clamp(dh, min=1e-12)) - log_t

    def log_survival(self, y, ctx):
        # 2026-04-09: log_ndtr upgrade — directly compute log Phi(-h)
        # without the clamp(ndtr, 1e-12) tail-saturation.
        y_safe = torch.clamp(y, min=1e-8)
        h, _ = self.h_and_dh(torch.log(y_safe), ctx)
        return torch.special.log_ndtr(-h)

    def aux_penalty(self, ctx):
        # 2026-04-09: gauge fix matching Appendix B (A3) — penalize the
        # redundant Sum_j w_j(ctx) direction. Active on over-parameterized
        # channels (e.g. C side under LogN truth) and inactive elsewhere.
        _, _, w, _ = self.params(ctx)
        return w.sum(dim=-1).pow(2).mean()

    @torch.no_grad()
    def _bisect_log_t(self, u, ctx, n_iter: int = 40):
        lo = torch.full_like(u, self.log_t_min)
        hi = torch.full_like(u, self.log_t_max)
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            h_mid, _ = self.h_and_dh(mid, ctx)
            go_right = (h_mid < u).to(mid.dtype)
            lo = go_right * mid + (1 - go_right) * lo
            hi = (1 - go_right) * mid + go_right * hi
        return 0.5 * (lo + hi)

    def sample(self, ctx, generator=None):
        # Match the API in informative_censoring_impl.py: returns t [n, 1].
        n = ctx.shape[0]
        if generator is None:
            u = torch.randn(n, 1, device=ctx.device, dtype=ctx.dtype)
        else:
            u = torch.randn(n, 1, generator=generator, device=ctx.device, dtype=ctx.dtype)
        log_t_star = self._bisect_log_t(u, ctx)
        # Implicit reparameterization for differentiable sampling (Figurnov 2018):
        h_star, dh_star = self.h_and_dh(log_t_star, ctx)
        log_t_diff = log_t_star.detach() - (h_star - u) / dh_star.detach()
        return torch.exp(log_t_diff)


# ---------------------------------------------------------------------------
# 5. Conditional Rational-Quadratic Spline flow on log-time (Durkan et al. 2019).
#    Closed-form forward, closed-form inverse (no bisection), monotone by
#    construction. log T = RQS(eta; z),  eta ~ N(0,1).
#    Outside the spline interval [-B, B] we extend with identity tails (linear
#    in eta), so the map remains monotone on the whole real line.
# ---------------------------------------------------------------------------
class CondRQSpline(nn.Module):
    family = "rqs"

    def __init__(self, ctx_dim: int = 1, hidden: int = 32, n_bins: int = 8, B: float = 4.0):
        super().__init__()
        self.K = n_bins
        self.B = B
        # per ctx: K widths, K heights, (K-1) inner derivatives
        self.head = _mlp(ctx_dim, hidden, 3 * n_bins - 1)

    def _spline_params(self, ctx):
        h = self.head(ctx)                                                   # [n, 3K-1]
        K = self.K
        unnorm_w = h[..., :K]
        unnorm_h = h[..., K:2 * K]
        unnorm_d = h[..., 2 * K:]                                            # [n, K-1]
        # widths and heights span 2B
        w = F.softmax(unnorm_w, dim=-1) * (2 * self.B)                       # [n, K]
        H = F.softmax(unnorm_h, dim=-1) * (2 * self.B)
        # cumulative knot positions (n, K+1) starting at -B
        cw = torch.cumsum(w, dim=-1)
        ch = torch.cumsum(H, dim=-1)
        x_knots = torch.cat([torch.full_like(cw[..., :1], -self.B), -self.B + cw], dim=-1)
        y_knots = torch.cat([torch.full_like(ch[..., :1], -self.B), -self.B + ch], dim=-1)
        # interior derivatives positive; boundary set to 1 (identity tails)
        d_inner = F.softplus(unnorm_d) + 1e-3                                 # [n, K-1]
        d_left = torch.ones_like(d_inner[..., :1])
        d_right = torch.ones_like(d_inner[..., :1])
        d = torch.cat([d_left, d_inner, d_right], dim=-1)                     # [n, K+1]
        return x_knots, y_knots, w, H, d

    def _eval(self, eta, ctx, inverse: bool):
        # eta: [n, 1] -> log_t [n,1] (or inverse), and log|d log_t / d eta|
        x_knots, y_knots, w, H, d = self._spline_params(ctx)
        e = eta.squeeze(-1)                                                  # [n]
        in_mask = (e > -self.B) & (e < self.B) if not inverse else (e > -self.B) & (e < self.B)
        # bin index via searchsorted
        if not inverse:
            bins = torch.searchsorted(x_knots, e.unsqueeze(-1), right=True).squeeze(-1) - 1
        else:
            bins = torch.searchsorted(y_knots, e.unsqueeze(-1), right=True).squeeze(-1) - 1
        bins = torch.clamp(bins, 0, self.K - 1)
        idx = bins.unsqueeze(-1)
        x_k = torch.gather(x_knots, -1, idx).squeeze(-1)
        x_kp1 = torch.gather(x_knots, -1, idx + 1).squeeze(-1)
        y_k = torch.gather(y_knots, -1, idx).squeeze(-1)
        y_kp1 = torch.gather(y_knots, -1, idx + 1).squeeze(-1)
        d_k = torch.gather(d, -1, idx).squeeze(-1)
        d_kp1 = torch.gather(d, -1, idx + 1).squeeze(-1)
        wb = x_kp1 - x_k
        hb = y_kp1 - y_k
        s = hb / (wb + 1e-9)

        if not inverse:
            xi = (e - x_k) / (wb + 1e-9)
            xi = torch.clamp(xi, 0.0, 1.0)
            num = hb * (s * xi * xi + d_k * xi * (1 - xi))
            den = s + (d_kp1 + d_k - 2 * s) * xi * (1 - xi)
            log_t_in = y_k + num / (den + 1e-9)
            # derivative
            dnum = s * s * (d_kp1 * xi * xi + 2 * s * xi * (1 - xi) + d_k * (1 - xi) ** 2)
            dy_dx_in = dnum / ((den + 1e-9) ** 2)
            # tails (identity outside)
            log_t = torch.where(in_mask, log_t_in, e)
            dy_dx = torch.where(in_mask, dy_dx_in, torch.ones_like(e))
            return log_t.unsqueeze(-1), dy_dx.unsqueeze(-1)
        else:
            # solve quadratic for xi:  (s + (d_kp1+d_k-2s) xi (1-xi)) * (e - y_k)/hb
            #                          = s xi^2 + d_k xi (1-xi)
            yrel = (e - y_k) / (hb + 1e-9)
            yrel = torch.clamp(yrel, 0.0, 1.0)
            a = hb * (s - d_k) + yrel * (d_kp1 + d_k - 2 * s) * hb
            b = hb * d_k - yrel * (d_kp1 + d_k - 2 * s) * hb
            c = -s * yrel * hb
            disc = b * b - 4 * a * c
            # YL patch 2026-04-08: clamp away from 0 (sqrt(0) has +inf gradient,
            # causes NaN backward); was min=0.0 which made training unstable.
            disc = torch.clamp(disc, min=1e-7)
            sqrt_disc = torch.sqrt(disc)
            # YL patch 2026-04-08: numerically stable quadratic root must pick
            # the sign of sqrt according to sign(b), else for b<0 the denominator
            # collapses near 0 and xi blows up. Was hardcoded `-b - sqrt(disc)`.
            denom = -b - torch.sign(b) * sqrt_disc
            denom = denom + torch.where(denom >= 0, torch.full_like(denom, 1e-9), torch.full_like(denom, -1e-9))
            xi = (2 * c) / denom
            xi = torch.clamp(xi, 0.0, 1.0)
            eta_in = x_k + xi * wb
            eta_full = torch.where(in_mask, eta_in, e)                       # tails identity
            return eta_full.unsqueeze(-1), None

    def log_pdf(self, y, ctx):
        eta_logy, _ = self._eval(torch.log(torch.clamp(y, min=1e-8)), ctx, inverse=True)
        log_t_back, dy_dx = self._eval(eta_logy, ctx, inverse=False)         # forward to get jacobian
        log_p_eta = -SQRT2PI_LOG - 0.5 * eta_logy * eta_logy
        return log_p_eta - torch.log(torch.clamp(dy_dx, min=1e-9)) - torch.log(torch.clamp(y, min=1e-8))

    def log_survival(self, y, ctx):
        # 2026-04-09: log_ndtr upgrade for numerical stability in deep tail.
        eta_logy, _ = self._eval(torch.log(torch.clamp(y, min=1e-8)), ctx, inverse=True)
        return torch.special.log_ndtr(-eta_logy)

    def sample(self, ctx, generator=None):
        eta = torch.randn(ctx.shape[0], 1, generator=generator, device=ctx.device, dtype=ctx.dtype)
        log_t, _ = self._eval(eta, ctx, inverse=False)
        return torch.exp(log_t)


# ---------------------------------------------------------------------------
FAMILY_REGISTRY = {
    "lognormal": CondLogNormal,
    "weibull":   CondWeibull,
    "mixln":     CondMixLogNormal,
    "spline":    CondMonotoneSplineFlow,
    "rqs":       CondRQSpline,
}


def build_dist(family: str, ctx_dim: int = 1, hidden: int = 32, **kw) -> nn.Module:
    cls = FAMILY_REGISTRY[family]
    return cls(ctx_dim=ctx_dim, hidden=hidden, **kw)
