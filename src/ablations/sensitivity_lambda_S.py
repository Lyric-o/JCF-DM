# -*- coding: utf-8 -*-
"""Sensitivity sweep over the Monte Carlo size S in the Wasserstein regularizer."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "synthetic"))

from uninformative_censoring_impl import (
    ExperimentConfig, DEVICE, DTYPE,
    generate_dataset, build_grid_from_data, hard_na_survival_on_grid,
    soft_na_survival_on_grid, wasserstein_curve_distance,
    enforce_time_bounds, sample_true_given_z,
    cdf_norm_metrics_from_samples, summarize_results,
)
from uninformative_censoring_flow import MonotoneFlow1D, cdf_norm_metrics_trimmed


LAMBDA_ALPHA = 0.01


def mle_loss(flow_t, flow_c, x, delta, z):
    log_f_t = flow_t.log_pdf(x, z)
    log_s_t = flow_t.log_survival(x, z)
    log_f_c = flow_c.log_pdf(x, z)
    log_s_c = flow_c.log_survival(x, z)
    ell = delta * (log_f_t + log_s_c) + (1.0 - delta) * (log_s_t + log_f_c)
    return -ell.mean()


def train_one(cfg, seed, lambda_w, s_sim):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data = generate_dataset(cfg.n_train, cfg, seed=seed)
    z, x, delta = data["z"], data["x"], data["delta"]
    censoring = float((1.0 - delta.mean()).item())

    grid = build_grid_from_data(x, cfg.grid_size)
    s_obs = hard_na_survival_on_grid(x, delta, grid)

    fkw = dict(
        in_dim=1, hidden_dim=cfg.hidden_dim, n_components=4,
        log_t_min=math.log(cfg.time_min), log_t_max=math.log(cfg.time_max),
    )
    flow_t = MonotoneFlow1D(**fkw).to(DEVICE)
    flow_c = MonotoneFlow1D(**fkw).to(DEVICE)
    opt = torch.optim.Adam(
        list(flow_t.parameters()) + list(flow_c.parameters()), lr=cfg.lr
    )

    best_loss = float("inf")
    best_t = best_c = None
    wait = 0
    best_epoch = 0
    t0 = time.perf_counter()

    for epoch in range(cfg.max_epochs):
        opt.zero_grad()
        loss_mle = mle_loss(flow_t, flow_c, x, delta, z)

        reg = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)
        if lambda_w > 0 and s_sim > 0:
            for _ in range(s_sim):
                t_sim = flow_t.sample(z, differentiable=True)
                c_sim = flow_c.sample(z, differentiable=True)
                t_sim = torch.clamp(t_sim, min=cfg.time_min, max=cfg.time_max)
                c_sim = torch.clamp(c_sim, min=cfg.time_min, max=cfg.time_max)
                x_sim = torch.minimum(t_sim, c_sim)
                d_sim_soft = torch.sigmoid((c_sim - t_sim) / cfg.tau_temp)
                s_sim_curve = soft_na_survival_on_grid(
                    x_sim, d_sim_soft, grid, cfg.soft_alpha)
                reg = reg + wasserstein_curve_distance(s_obs, s_sim_curve, grid)
            reg = reg / s_sim

        aux = flow_c.aux_penalty(z)
        loss = loss_mle + lambda_w * reg + LAMBDA_ALPHA * aux
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(flow_t.parameters()) + list(flow_c.parameters()), max_norm=2.0
        )
        opt.step()

        cur = float(loss.item())
        if cur < best_loss - cfg.min_delta:
            best_loss = cur
            best_epoch = epoch + 1
            wait = 0
            best_t = {k: v.detach().cpu().clone()
                      for k, v in flow_t.state_dict().items()}
            best_c = {k: v.detach().cpu().clone()
                      for k, v in flow_c.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    train_sec = time.perf_counter() - t0
    if best_t is not None:
        flow_t.load_state_dict(best_t)
    if best_c is not None:
        flow_c.load_state_dict(best_c)

    # Evaluation
    g_eval = torch.Generator(device=DEVICE)
    g_eval.manual_seed(seed + 98765)
    z_eval = torch.normal(cfg.z_mean, cfg.z_std, size=(cfg.n_eval, 1),
                          generator=g_eval, device=DEVICE, dtype=DTYPE)
    t_true = sample_true_given_z(z_eval, "T", cfg, cfg.scenario_name,
                                 generator=g_eval)
    c_true = sample_true_given_z(z_eval, "C", cfg, cfg.scenario_name,
                                 generator=g_eval)
    with torch.no_grad():
        t_model = enforce_time_bounds(
            flow_t.sample(z_eval, generator=g_eval), cfg)
        c_model = enforce_time_bounds(
            flow_c.sample(z_eval, generator=g_eval), cfg)

    t_m = cdf_norm_metrics_from_samples(t_true, t_model)
    c_m = cdf_norm_metrics_from_samples(c_true, c_model)
    t_trim = cdf_norm_metrics_trimmed(t_true, t_model, q_high=0.85)

    return {
        "seed": seed,
        "lambda_w": lambda_w, "s_sim": s_sim,
        "T_KS": t_m["Linf_KS"],
        "T_W1": t_m["W1_samples"],
        "C_KS": c_m["Linf_KS"],
        "C_W1": c_m["W1_samples"],
        "T_KS_trim85": t_trim["KS_trim85"],
        "T_W1_trim85": t_trim["W1_trim85"],
        "converged_epoch": best_epoch,
        "train_time_sec": train_sec,
        "censoring_rate": censoring,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dgp", default="cox_both",
                   choices=["lognormal_base", "heavy_tail_t",
                            "cox_t_lognormal_c", "cox_both"])
    p.add_argument("--mode", default="lambda", choices=["lambda", "S"])
    p.add_argument("--mc_runs", type=int, default=20)
    p.add_argument("--n_train", type=int, default=2048)
    p.add_argument("--n_eval", type=int, default=1024)
    p.add_argument("--max_epochs", type=int, default=400)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--suffix", default="sens")
    args = p.parse_args()

    cfg = ExperimentConfig()
    cfg.n_train = args.n_train
    cfg.n_eval = args.n_eval
    cfg.max_epochs = args.max_epochs
    cfg.patience = args.patience
    cfg.scenario_name = args.dgp

    if args.mode == "lambda":
        sweep = [
            (0.0, 3), (0.05, 3), (0.1, 3), (0.2, 3), (0.5, 3), (1.0, 3)
        ]
    else:  # S
        sweep = [(0.2, 1), (0.2, 3), (0.2, 5), (0.2, 10)]

    all_results = {}

    for lam, S in sweep:
        key = f"lambda={lam:.2f}_S={S}"
        print(f"\n=== Config: {key} ===", flush=True)
        runs = []
        for r in range(args.mc_runs):
            seed = 20260413 + r * 17
            res = train_one(cfg, seed, lambda_w=lam, s_sim=S)
            runs.append(res)
            print(f"  run {r+1}/{args.mc_runs} | T_KS={res['T_KS']:.4f}, "
                  f"T_W1={res['T_W1']:.4f}, "
                  f"epoch={res['converged_epoch']}, "
                  f"time={res['train_time_sec']:.1f}s", flush=True)

        ks_vals = [r["T_KS"] for r in runs]
        w1_vals = [r["T_W1"] for r in runs]
        all_results[key] = {
            "lambda_w": lam, "S": S,
            "T_KS_mean": float(np.mean(ks_vals)),
            "T_KS_std": float(np.std(ks_vals)),
            "T_W1_mean": float(np.mean(w1_vals)),
            "T_W1_std": float(np.std(w1_vals)),
            "runs": runs,
        }

    # Summary
    print(f"\n\n{'='*70}")
    print(f"Sensitivity ({args.mode}) — {args.dgp} — {args.mc_runs} MC runs")
    print(f"{'='*70}")
    print(f"{'Config':<25s} {'T-KS mean±std':<22s} {'T-W1 mean±std':<22s}")
    print("-" * 70)
    for key, v in all_results.items():
        print(f"{key:<25s} {v['T_KS_mean']:.4f}±{v['T_KS_std']:.4f}     "
              f"{v['T_W1_mean']:.4f}±{v['T_W1_std']:.4f}")

    out_file = f"sensitivity_{args.mode}_{args.dgp}_{args.suffix}.json"
    with open(out_file, "w") as f:
        json.dump({"config": {"dgp": args.dgp, "mc_runs": args.mc_runs,
                               "n_train": args.n_train, "mode": args.mode},
                   "results": all_results}, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
