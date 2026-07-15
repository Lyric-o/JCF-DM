# -*- coding: utf-8 -*-
"""Alternative informative-censoring DGP (LogNormal frailty, Weibull margins) used for real-covariate semi-synthetic experiments."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import torch

DEVICE = torch.device("cpu")
DTYPE = torch.float32


@dataclass
class ICConfig2:
    n_train: int = 2048
    n_eval: int = 1024
    z_std: float = 1.0
    frailty_sigma: float = 0.5  # LogNormal frailty variance
    shape_T: float = 1.5
    shape_C: float = 1.2
    beta_T: float = 0.40
    beta_C: float = -0.30
    rho: float = 1.0
    family: str = "spline"
    hidden: int = 32
    K_eps: int = 32
    s_sim: int = 3
    lambda_w: float = 0.2
    tau_temp: float = 0.25
    soft_alpha: float = 0.20
    grid_size: int = 80
    lr: float = 1e-3
    max_epochs: int = 400
    patience: int = 50
    min_delta: float = 1e-4
    time_min: float = 1e-4
    time_max: float = 10.0
    mc_runs: int = 30


def gen_dgp2(n, cfg, seed):
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    z = torch.normal(0.0, cfg.z_std, size=(n, 1), generator=g, dtype=DTYPE)

    # LogNormal frailty: ξ = exp(σ * η), η ~ N(0,1)
    eta = torch.randn((n, 1), generator=g, dtype=DTYPE)
    xi = torch.exp(cfg.frailty_sigma * eta)

    # Weibull T: T = scale_T * (-log U)^(1/shape_T)
    u1 = torch.rand((n, 1), generator=g, dtype=DTYPE)
    scale_T = torch.exp(cfg.beta_T * z) * xi
    T = scale_T * (-torch.log(torch.clamp(u1, min=1e-12))) ** (1.0 / cfg.shape_T)

    # Weibull C: C = scale_C * (-log U)^(1/shape_C), frailty effect scaled by ρ
    u2 = torch.rand((n, 1), generator=g, dtype=DTYPE)
    scale_C = torch.exp(cfg.beta_C * z) * (xi ** cfg.rho)
    C = scale_C * (-torch.log(torch.clamp(u2, min=1e-12))) ** (1.0 / cfg.shape_C)

    T = torch.clamp(T, cfg.time_min, cfg.time_max)
    C = torch.clamp(C, cfg.time_min, cfg.time_max)
    x = torch.minimum(T, C)
    delta = (T <= C).to(DTYPE)
    return {"z": z, "t": T, "c": C, "x": x, "delta": delta, "xi": xi}


# Reuse infrastructure from existing informative censoring script
from informative_censoring_impl import (
    hard_na_cdf_on_grid, soft_na_cdf_on_grid, w1_curve,
    build_dist, mc_marginal_mle,
)

LAMBDA_ALPHA = 0.01


def train_one(cfg, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = gen_dgp2(cfg.n_train, cfg, seed)
    z, x, delta = data["z"], data["x"], data["delta"]
    censoring = float((1 - delta.mean()).item())

    grid = torch.linspace(
        max(1e-4, float(torch.quantile(x, 0.01).item())),
        max(1e-3, float(torch.quantile(x, 0.99).item())),
        cfg.grid_size, dtype=DTYPE)
    s_obs = hard_na_cdf_on_grid(x, delta, grid)

    flow_t = build_dist(cfg.family, ctx_dim=2, hidden=cfg.hidden).to(DEVICE)
    flow_c = build_dist(cfg.family, ctx_dim=2, hidden=cfg.hidden).to(DEVICE)
    opt = torch.optim.Adam(
        list(flow_t.parameters()) + list(flow_c.parameters()), lr=cfg.lr)

    best_loss = float("inf"); wait = 0; best_epoch = 0
    best_t = best_c = None
    t0 = time.perf_counter()

    for epoch in range(cfg.max_epochs):
        opt.zero_grad()
        loss_mle = mc_marginal_mle(flow_t, flow_c, x, delta, z, cfg.K_eps)

        reg = 0.0
        for _ in range(cfg.s_sim):
            eps_s = torch.randn_like(z)
            ctx_s = torch.cat([z, eps_s], dim=-1)
            t_sim = torch.clamp(flow_t.sample(ctx_s), cfg.time_min, cfg.time_max)
            c_sim = torch.clamp(flow_c.sample(ctx_s), cfg.time_min, cfg.time_max)
            x_sim = torch.minimum(t_sim, c_sim)
            d_soft = torch.sigmoid((c_sim - t_sim) / cfg.tau_temp)
            s_sim = soft_na_cdf_on_grid(x_sim, d_soft, grid, cfg.soft_alpha)
            reg = reg + w1_curve(s_obs, s_sim, grid)
        reg = reg / cfg.s_sim

        if hasattr(flow_c, "aux_penalty"):
            ctx_pen = torch.cat([z, torch.zeros_like(z)], dim=-1)
            aux = flow_c.aux_penalty(ctx_pen)
        else:
            aux = torch.tensor(0.0)

        loss = loss_mle + cfg.lambda_w * reg + LAMBDA_ALPHA * aux
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(flow_t.parameters()) + list(flow_c.parameters()), 2.0)
        opt.step()

        cur = float(loss.item())
        if cur < best_loss - cfg.min_delta:
            best_loss = cur; best_epoch = epoch + 1; wait = 0
            best_t = {k: v.cpu().clone() for k, v in flow_t.state_dict().items()}
            best_c = {k: v.cpu().clone() for k, v in flow_c.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience: break

    train_sec = time.perf_counter() - t0
    if best_t: flow_t.load_state_dict(best_t)
    if best_c: flow_c.load_state_dict(best_c)

    # Evaluation
    S_eval = 20
    g_eval = torch.Generator(device=DEVICE)
    g_eval.manual_seed(seed + 98765)
    eval_data = gen_dgp2(cfg.n_eval, cfg, seed + 12345)
    z_e, t_true, c_true = eval_data["z"], eval_data["t"], eval_data["c"]

    with torch.no_grad():
        z_rep = z_e.unsqueeze(0).expand(S_eval, -1, -1).reshape(-1, z_e.shape[-1])
        eps_e = torch.randn(z_rep.shape[0], 1, generator=g_eval, dtype=DTYPE)
        ctx_e = torch.cat([z_rep, eps_e], dim=-1)
        t_model = torch.clamp(flow_t.sample(ctx_e, generator=g_eval),
                               cfg.time_min, cfg.time_max)
        c_model = torch.clamp(flow_c.sample(ctx_e, generator=g_eval),
                               cfg.time_min, cfg.time_max)

    from uninformative_censoring_impl import cdf_norm_metrics_from_samples
    from uninformative_censoring_flow import cdf_norm_metrics_trimmed

    # t_true is [n_eval, 1], t_model is [S_eval*n_eval, 1].
    # Repeat t_true to match (same true value for each of S_eval draws).
    t_true_rep = t_true.repeat(S_eval, 1)
    c_true_rep = c_true.repeat(S_eval, 1)
    t_m = cdf_norm_metrics_from_samples(t_true_rep, t_model)
    c_m = cdf_norm_metrics_from_samples(c_true_rep, c_model)
    t_trim = cdf_norm_metrics_trimmed(t_true_rep, t_model)

    return {
        "seed": seed, "rho": cfg.rho, "dgp": "lognormal_weibull_frailty",
        "T_KS": t_m["Linf_KS"], "T_W1": t_m["W1_samples"],
        "C_KS": c_m["Linf_KS"], "C_W1": c_m["W1_samples"],
        "T_KS_trim85": t_trim["KS_trim85"],
        "converged_epoch": best_epoch,
        "train_time_sec": train_sec,
        "censoring_rate": censoring,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", default="spline",
                   choices=["lognormal", "spline"])
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--mc_runs", type=int, default=30)
    p.add_argument("--n_train", type=int, default=2048)
    p.add_argument("--n_eval", type=int, default=1024)
    p.add_argument("--K_eps", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=400)
    p.add_argument("--lambda_alpha", type=float, default=0.01)
    p.add_argument("--out", default="ic_dgp2")
    args = p.parse_args()

    global LAMBDA_ALPHA
    LAMBDA_ALPHA = args.lambda_alpha

    cfg = ICConfig2(
        family=args.family, rho=args.rho, mc_runs=args.mc_runs,
        n_train=args.n_train, n_eval=args.n_eval,
        K_eps=args.K_eps, max_epochs=args.max_epochs,
    )

    tag = f"{args.family}_rho{args.rho}"
    rows = []
    for i in range(cfg.mc_runs):
        seed = 20260414 + i * 13
        r = train_one(cfg, seed)
        rows.append(r)
        print(f"[DGP2 {tag}] {i+1:02d}/{cfg.mc_runs} "
              f"T_KS={r['T_KS']:.4f} C_KS={r['C_KS']:.4f} "
              f"cens={r['censoring_rate']:.2f} "
              f"epoch={int(r['converged_epoch'])} "
              f"time={r['train_time_sec']:.1f}s", flush=True)

    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    summary = {k: {"mean": float(np.mean([r[k] for r in rows])),
                    "std": float(np.std([r[k] for r in rows]))}
               for k in keys}

    print(f"\n=== DGP2 (LogN-Weibull frailty) rho={args.rho} ===")
    print(f"T-KS: {summary['T_KS']['mean']:.4f} ± {summary['T_KS']['std']:.4f}")
    print(f"C-KS: {summary['C_KS']['mean']:.4f} ± {summary['C_KS']['std']:.4f}")

    out_file = f"{args.out}_{tag}.json"
    with open(out_file, "w") as f:
        json.dump({"config": asdict(cfg), "results": rows, "summary": summary},
                  f, indent=2)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
