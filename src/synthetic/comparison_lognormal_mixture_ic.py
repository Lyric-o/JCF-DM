# -*- coding: utf-8 -*-
"""DSM-inspired LogNormal mixture evaluated unchanged on the Table 2 DGP.

This is the study-specific ``LogN mixture@IC`` comparison in the paper.  The
``@IC`` suffix denotes evaluation on informative-censoring data; it is not a
claim that Nagpal et al. proposed a method called DSM-IC.

Usage:
    python comparison_lognormal_mixture_ic.py --rho 1.0 --mc_runs 50 --tag bigN_v3
"""

from __future__ import annotations
import argparse, json, math, random, time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_fn

from uninformative_censoring_impl import (
    ExperimentConfig, DEVICE, DTYPE,
    enforce_time_bounds,
    cdf_norm_metrics_from_samples, km_estimator, km_cdf_on_grid,
    cdf_from_samples_on_grid, curve_distance_metrics, summarize_results,
)
from uninformative_censoring_flow import cdf_norm_metrics_trimmed
from comparison_lognormal_mixture import ConditionalLogNormalMixture
from informative_censoring_impl import ICConfig, gen_informative


def train_eval_lognormal_mixture_ic(cfg, rho, seed, K_mix=4, max_epochs=400,
                       patience=50, lr=1e-3, S_eval=20):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ic_cfg = ICConfig(n_train=cfg.n_train, n_eval=cfg.n_eval, rho=rho)
    data = gen_informative(cfg.n_train, ic_cfg, seed)
    z, x, delta = data["z"], data["x"], data["delta"]
    censoring_rate = float((1.0 - delta.mean()).item())

    model = ConditionalLogNormalMixture(in_dim=1, hidden=64, K_mix=K_mix).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_state = None
    wait = 0
    best_epoch = 0
    t0 = time.perf_counter()

    for epoch in range(max_epochs):
        opt.zero_grad()
        log_f = model.log_pdf(x, z)
        log_S = model.log_survival(x, z)
        loss = -(delta * log_f + (1.0 - delta) * log_S).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        cur = float(loss.item())
        if cur < best_loss - 1e-4:
            best_loss = cur
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    train_time = time.perf_counter() - t0
    if best_state:
        model.load_state_dict(best_state)

    # Eval — T-side only
    g_eval = torch.Generator(device=DEVICE)
    g_eval.manual_seed(seed + 98765)
    data_eval = gen_informative(cfg.n_eval, ic_cfg, seed + 12345)
    z_eval = data_eval["z"]
    t_true = data_eval["t"]
    z_rep = z_eval.unsqueeze(0).expand(S_eval, -1, -1).reshape(-1, 1)

    with torch.no_grad():
        t_model = enforce_time_bounds(model.sample(z_rep, generator=g_eval),
                                      cfg)

    t_metrics = cdf_norm_metrics_from_samples(t_true, t_model)
    t_trim = cdf_norm_metrics_trimmed(t_true, t_model, q_high=0.85)

    # KM baseline (marginal, no covariate — also assumes indep censoring)
    grid_t = torch.linspace(
        float(torch.quantile(
            torch.cat([t_true.view(-1), t_model.view(-1)]), 0.01)),
        float(torch.quantile(
            torch.cat([t_true.view(-1), t_model.view(-1)]), 0.995)),
        280, device=DEVICE, dtype=DTYPE,
    )
    true_cdf = cdf_from_samples_on_grid(t_true, grid_t)
    km_times, km_surv = km_estimator(x, delta)
    km_cdf = km_cdf_on_grid(km_times, km_surv, grid_t)
    t_km = curve_distance_metrics(true_cdf, km_cdf, grid_t)

    out = {
        "seed": seed, "method": f"LogNormalMixture_K{K_mix}",
        "best_nll": best_loss, "converged_epoch": float(best_epoch),
        "train_time_sec": train_time, "censoring_rate": censoring_rate,
        "rho": rho,
    }
    out.update({f"T_{k}": v for k, v in t_metrics.items()})
    out.update({f"T_{k}": v for k, v in t_trim.items()})
    out.update({f"T_compKM_{k}": v for k, v in t_km.items()})
    return out


def main():
    p = argparse.ArgumentParser(
        description="Study-specific LogNormal mixture on informative-censoring data")
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--mc_runs", type=int, default=30)
    p.add_argument("--n_train", type=int, default=2048)
    p.add_argument("--n_eval", type=int, default=1024)
    p.add_argument("--K_mix", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=400)
    p.add_argument("--start_index", type=int, default=0,
                   help="First Monte Carlo replication index (for sharding)")
    p.add_argument("--tag", type=str, default="bigN_v3")
    args = p.parse_args()

    cfg = ExperimentConfig()
    cfg.n_train = args.n_train
    cfg.n_eval = args.n_eval

    print(f"=== LogNormal mixture K={args.K_mix} on informative DGP, rho={args.rho}, "
          f"n_train={cfg.n_train}, mc_runs={args.mc_runs} ===")

    results = []
    for local_i in range(args.mc_runs):
        i = args.start_index + local_i
        seed = 20260407 + i * 13
        res = train_eval_lognormal_mixture_ic(cfg, args.rho, seed, K_mix=args.K_mix,
                                max_epochs=args.max_epochs)
        results.append(res)
        print(f"  Run {i+1:02d} (shard {local_i+1:02d}/{args.mc_runs}) | "
              f"T_W1={res['T_W1_samples']:.4f} "
              f"T_KS={res['T_Linf_KS']:.4f} "
              f"epoch={int(res['converged_epoch'])} time={res['train_time_sec']:.1f}s",
              flush=True)

    numeric_results = [{k: v for k, v in r.items()
                        if isinstance(v, (int, float))} for r in results]
    summary = summarize_results(numeric_results)

    out_file = f"lognormal_mixture_at_ic_K{args.K_mix}_rho{args.rho}_{args.tag}.json"
    with open(out_file, "w") as f:
        json.dump({"method": f"LogNormalMixture_K{args.K_mix}", "rho": args.rho,
                    "config": asdict(cfg), "results": results,
                    "summary": summary}, f, indent=2)

    print(f"\n=== Summary: LogNormal mixture K={args.K_mix} rho={args.rho} ===")
    for k in ["T_W1_samples", "T_Linf_KS"]:
        print(f"  {k:22s}: {summary[k]['mean']:.4f} +/- {summary[k]['std']:.4f}")
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
