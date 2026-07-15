# -*- coding: utf-8 -*-
"""
In-house competing methods for T|Z (no external survival libs needed):
    - Kaplan-Meier             (non-parametric, marginal)
    - Cox PH + Breslow         (semi-parametric)
    - Lognormal AFT (MLE)      (parametric)
    - Weibull   AFT (MLE)      (parametric)

Each method exposes:
    fit(z, x, delta) -> state
    cdf_on_grid(state, z_eval, grid) -> torch.Tensor [grid]   marginal CDF over z_eval
"""
from __future__ import annotations
import math
import torch

DTYPE = torch.float32


# ----------------- KM (marginal, ignores Z) -----------------
def km_fit(x, delta):
    x1 = x.view(-1); d1 = delta.view(-1)
    et = torch.unique(x1[d1 > 0.5])
    et, _ = torch.sort(et)
    if et.numel() == 0:
        return torch.tensor([0.0]), torch.tensor([1.0])
    s = torch.tensor(1.0)
    out = []
    for t in et:
        ar = (x1 >= t).sum().to(DTYPE)
        d_t = ((x1 == t).to(DTYPE) * d1).sum()
        s = s * (1.0 - d_t / torch.clamp(ar, min=1.0))
        out.append(s)
    return et, torch.stack(out)


def km_cdf_on_grid(state, z_eval, grid):
    et, sv = state
    idx = torch.searchsorted(et, grid, right=True) - 1
    s = torch.ones_like(grid)
    valid = idx >= 0
    s[valid] = sv[torch.clamp(idx[valid], min=0, max=sv.numel() - 1)]
    return 1.0 - s


# ----------------- Cox PH 1D + Breslow -----------------
def cox_fit(z, x, delta, max_iter=250, lr=0.05):
    z1 = z.view(-1); x1 = x.view(-1); d1 = delta.view(-1)
    order = torch.argsort(x1, descending=True)
    z_sorted = z1[order]; d_sorted = d1[order]
    beta = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([beta], lr=lr)
    for _ in range(max_iter):
        opt.zero_grad()
        eta = beta * z_sorted
        ee = torch.exp(torch.clamp(eta, -25, 25))
        rc = torch.cumsum(ee, dim=0)
        pll = (d_sorted * (eta - torch.log(torch.clamp(rc, min=1e-8)))).sum()
        (-pll / torch.clamp(d_sorted.sum(), min=1.0)).backward()
        opt.step()
    beta = beta.detach()
    et = torch.unique(x1[d1 > 0.5]); et, _ = torch.sort(et)
    if et.numel() == 0:
        return beta, torch.tensor([0.0]), torch.tensor([0.0])
    ee = torch.exp(torch.clamp(beta * z1, -25, 25))
    inc = []
    for t in et:
        d_t = ((x1 == t).to(DTYPE) * d1).sum()
        denom = ee[x1 >= t].sum()
        inc.append(d_t / torch.clamp(denom, min=1e-8))
    H0 = torch.cumsum(torch.stack(inc), dim=0)
    return beta, et, H0


def cox_cdf_on_grid(state, z_eval, grid):
    beta, et, H0 = state
    idx = torch.searchsorted(et, grid, right=True) - 1
    h0 = torch.zeros_like(grid)
    valid = idx >= 0
    h0[valid] = H0[torch.clamp(idx[valid], min=0, max=H0.numel() - 1)]
    lp = torch.exp(torch.clamp(beta * z_eval.view(-1), -25, 25)).view(-1, 1)
    s = torch.exp(-lp * h0.view(1, -1))
    return 1.0 - s.mean(dim=0)


# ----------------- Lognormal AFT MLE -----------------
def _aft_fit(z, x, delta, family, max_iter=400, lr=0.05):
    """
    family in {"lognormal","weibull"}.
    log T = a + b*z + sigma*eta,  eta ~ N(0,1) (lognormal) or Gumbel-min (weibull).
    """
    z1 = z.view(-1); x1 = x.view(-1); d1 = delta.view(-1)
    a = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    log_sigma = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([a, b, log_sigma], lr=lr)
    for _ in range(max_iter):
        opt.zero_grad()
        sigma = torch.exp(log_sigma).clamp(min=0.05, max=5.0)
        u = (torch.log(torch.clamp(x1, min=1e-8)) - a - b * z1) / sigma
        if family == "lognormal":
            log_f = -0.5 * math.log(2 * math.pi) - torch.log(sigma) - 0.5 * u * u - torch.log(torch.clamp(x1, min=1e-8))
            log_S = torch.log(torch.clamp(torch.special.ndtr(-u), min=1e-12))
        else:  # weibull: eta ~ Gumbel-min, f(u)=exp(u-exp(u))
            log_f = -torch.log(sigma) - torch.log(torch.clamp(x1, min=1e-8)) + u - torch.exp(u)
            log_S = -torch.exp(u)
        ll = (d1 * log_f + (1 - d1) * log_S).sum()
        (-ll / x1.numel()).backward()
        opt.step()
    return a.detach(), b.detach(), torch.exp(log_sigma.detach()).clamp(min=0.05, max=5.0), family


def aft_fit_lognormal(z, x, delta): return _aft_fit(z, x, delta, "lognormal")
def aft_fit_weibull(z, x, delta):   return _aft_fit(z, x, delta, "weibull")


def aft_cdf_on_grid(state, z_eval, grid):
    a, b, sigma, family = state
    z = z_eval.view(-1, 1)
    g = grid.view(1, -1)
    u = (torch.log(torch.clamp(g, min=1e-8)) - a - b * z) / sigma
    if family == "lognormal":
        F = torch.special.ndtr(u)
    else:
        F = 1.0 - torch.exp(-torch.exp(u))
    return F.mean(dim=0)


# convenience dispatcher
COMPETING = {
    "KM":     (lambda z, x, d: km_fit(x, d),       km_cdf_on_grid),
    "Cox":    (cox_fit,                            cox_cdf_on_grid),
    "AFT_LN": (aft_fit_lognormal,                  aft_cdf_on_grid),
    "AFT_W":  (aft_fit_weibull,                    aft_cdf_on_grid),
}
