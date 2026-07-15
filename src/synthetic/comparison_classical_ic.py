# -*- coding: utf-8 -*-
"""T-W1 sampling driver for KM / Cox PH / AFT-LN / AFT-Weibull under informative censoring."""

from __future__ import annotations
import argparse, json, time
from dataclasses import asdict
import numpy as np
import torch

from informative_censoring_impl import ICConfig, gen_informative, DEVICE, DTYPE
from competing_methods import (
    km_fit, cox_fit, aft_fit_lognormal, aft_fit_weibull,
)


# ----------------- Samplers -----------------

def km_sample(state, n, generator):
    """KM is unconditional (no Z). Invert the empirical step CDF."""
    et, sv = state
    if et.numel() == 0:
        return torch.zeros(n, device=DEVICE, dtype=DTYPE) + 1e-4
    u = torch.rand(n, generator=generator, device=DEVICE, dtype=DTYPE)
    F = 1.0 - sv  # increasing step CDF on et
    idx = torch.searchsorted(F, u, right=False)
    idx = torch.clamp(idx, max=et.numel() - 1)
    return et[idx]


def cox_sample(state, z_eval, generator):
    """Cox-PH: invert S(t|z) = exp(-H0(t) * exp(beta z)) on the et grid."""
    beta, et, H0 = state
    n = z_eval.numel()
    if et.numel() == 0:
        return torch.zeros(n, device=DEVICE, dtype=DTYPE) + 1e-4
    u = torch.rand(n, generator=generator, device=DEVICE, dtype=DTYPE)
    lp = torch.exp(torch.clamp(beta * z_eval.view(-1), -25, 25))
    target = -torch.log(torch.clamp(u, min=1e-8)) / lp
    idx = torch.searchsorted(H0, target, right=False)
    idx = torch.clamp(idx, max=et.numel() - 1)
    return et[idx]


def aft_sample(state, z_eval, generator):
    """AFT-LN/Weibull: closed-form inverse CDF of log T = a + b z + sigma * eps."""
    a, b, sigma, family = state
    n = z_eval.numel()
    u = torch.rand(n, generator=generator, device=DEVICE, dtype=DTYPE).clamp(1e-6, 1 - 1e-6)
    mu = a + b * z_eval.view(-1)
    if family == "lognormal":
        eps = torch.special.ndtri(u)
        log_t = mu + sigma * eps
    elif family == "weibull":
        log_t = mu + sigma * torch.log(-torch.log(1.0 - u))
    else:
        raise ValueError(family)
    return torch.exp(log_t)


# ----------------- T-side metrics, sample-based, on a 300-pt grid -----------------

def t_metrics(t_true, t_model, time_min=1e-4, time_max=10.0):
    t = torch.clamp(t_true.view(-1), time_min, time_max)
    m = torch.clamp(t_model.view(-1), time_min, time_max)
    ts, _ = torch.sort(t)
    ms, _ = torch.sort(m)
    pooled = torch.cat([ts, ms])
    gmin = float(torch.quantile(pooled, 0.01).item())
    gmax = float(torch.quantile(pooled, 0.99).item())
    g = torch.linspace(max(1e-5, gmin), max(gmin + 1e-4, gmax),
                       300, device=DEVICE, dtype=DTYPE)
    Ft = torch.searchsorted(ts, g, right=True).float() / ts.numel()
    Fm = torch.searchsorted(ms, g, right=True).float() / ms.numel()
    diff = Ft - Fm
    dg = g[1:] - g[:-1]
    w1 = float((torch.abs(diff[:-1]) * dg).sum().item())
    ks = float(torch.max(torch.abs(diff)).item())
    return w1, ks


# ----------------- Per-seed fit + eval -----------------

METHODS = ["KM", "Cox", "AFT_LN", "AFT_W"]


def fit_and_eval_one(cfg, seed, S_eval=20):
    torch.manual_seed(seed); np.random.seed(seed)
    data = gen_informative(cfg.n_train, cfg, seed)
    z, x, delta = data["z"], data["x"], data["delta"]
    censoring = float((1.0 - delta.mean()).item())

    # Eval data — same seed offset as canonical IC eval (seed + 12345)
    eval_data = gen_informative(cfg.n_eval, cfg, seed + 12345)
    z_e, t_true = eval_data["z"], eval_data["t"]
    z_rep = z_e.unsqueeze(0).expand(S_eval, -1, -1).reshape(-1, 1)

    g_eval = torch.Generator(device=DEVICE); g_eval.manual_seed(seed + 98765)

    out = {"seed": int(seed), "censoring_rate": censoring}

    # --- KM (no Z) ---
    t0 = time.perf_counter()
    state = km_fit(x, delta)
    t_km = km_sample(state, S_eval * cfg.n_eval, g_eval)
    out["KM_fit_time_sec"] = time.perf_counter() - t0
    w1, ks = t_metrics(t_true, t_km)
    out["KM_T_W1"], out["KM_T_KS"] = w1, ks

    # --- Cox PH ---
    t0 = time.perf_counter()
    state = cox_fit(z, x, delta)
    t_cox = cox_sample(state, z_rep, g_eval)
    out["Cox_fit_time_sec"] = time.perf_counter() - t0
    w1, ks = t_metrics(t_true, t_cox)
    out["Cox_T_W1"], out["Cox_T_KS"] = w1, ks

    # --- AFT-LogN ---
    t0 = time.perf_counter()
    state = aft_fit_lognormal(z, x, delta)
    t_aln = aft_sample(state, z_rep, g_eval)
    out["AFT_LN_fit_time_sec"] = time.perf_counter() - t0
    w1, ks = t_metrics(t_true, t_aln)
    out["AFT_LN_T_W1"], out["AFT_LN_T_KS"] = w1, ks

    # --- AFT-Weibull ---
    t0 = time.perf_counter()
    state = aft_fit_weibull(z, x, delta)
    t_aw = aft_sample(state, z_rep, g_eval)
    out["AFT_W_fit_time_sec"] = time.perf_counter() - t0
    w1, ks = t_metrics(t_true, t_aw)
    out["AFT_W_T_W1"], out["AFT_W_T_KS"] = w1, ks

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--mc_runs", type=int, default=50)
    p.add_argument("--n_train", type=int, default=2048)
    p.add_argument("--n_eval", type=int, default=1024)
    p.add_argument("--K_eps", type=int, default=32,
                   help="ignored (classical methods don't use latent), kept for protocol parity")
    p.add_argument("--tag", default="bigN_0504")
    args = p.parse_args()

    cfg = ICConfig(family="spline", rho=args.rho, mc_runs=args.mc_runs,
                   n_train=args.n_train, n_eval=args.n_eval, K_eps=args.K_eps)

    rows = []
    for i in range(args.mc_runs):
        seed = 20260407 + i * 13   # match canonical IC bigN seed sequence
        r = fit_and_eval_one(cfg, seed)
        rows.append(r)
        print(
            f"  run {i+1:02d}/{args.mc_runs} | "
            f"KM W1/KS={r['KM_T_W1']:.4f}/{r['KM_T_KS']:.4f}  "
            f"Cox={r['Cox_T_W1']:.4f}/{r['Cox_T_KS']:.4f}  "
            f"AFT-LN={r['AFT_LN_T_W1']:.4f}/{r['AFT_LN_T_KS']:.4f}  "
            f"AFT-W={r['AFT_W_T_W1']:.4f}/{r['AFT_W_T_KS']:.4f}",
            flush=True,
        )

    summary = {}
    for k in [f"{m}_T_{metric}" for m in METHODS for metric in ("W1", "KS")]:
        vals = [r[k] for r in rows]
        summary[k] = {"mean": float(np.mean(vals)),
                      "std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}

    out_file = f"comp_classical_ic_rho{args.rho}_{args.tag}.json"
    with open(out_file, "w") as f:
        json.dump({"method": "Classical_IC_W1_extraction",
                   "rho": args.rho,
                   "config": asdict(cfg),
                   "results": rows,
                   "summary": summary}, f, indent=2)
    print(f"\nSaved to {out_file}")
    print("\n=== Summary (mean ± std across {} seeds) ===".format(args.mc_runs))
    for k, v in summary.items():
        print(f"  {k:<20s} {v['mean']:.4f} ± {v['std']:.4f}")


if __name__ == "__main__":
    main()
