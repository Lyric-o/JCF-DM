# -*- coding: utf-8 -*-
"""T-only conditional flow ablation under uninformative censoring."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from uninformative_censoring_impl import (
    ExperimentConfig, DEVICE, DTYPE,
    generate_dataset, enforce_time_bounds, sample_true_given_z,
    cdf_norm_metrics_from_samples, summarize_results,
)
from uninformative_censoring_flow import (
    MonotoneFlow1D, N_COMPONENTS, cdf_norm_metrics_trimmed,
)


def censored_mle_t_only(flow_t, x, delta, z):
    """
    Standard censored MLE for T|Z only (no C model):
        ell_i = delta_i * log f_T(x_i|z_i) + (1-delta_i) * log S_T(x_i|z_i)
    This is the Ausset 2021 loss.
    """
    log_f_t = flow_t.log_pdf(x, z)
    log_s_t = flow_t.log_survival(x, z)
    ell = delta * log_f_t + (1.0 - delta) * log_s_t
    return -ell.mean()


def train_eval_ausset(cfg, scenario_name, seed, K=4, max_epochs=400,
                      patience=50, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cfg_s = ExperimentConfig(**{**asdict(cfg), "scenario_name": scenario_name})
    data = generate_dataset(cfg_s.n_train, cfg_s, seed=seed)
    z, x, delta = data["z"], data["x"], data["delta"]
    censoring = float((1.0 - delta.mean()).item())

    flow_t = MonotoneFlow1D(
        in_dim=1, hidden_dim=cfg_s.hidden_dim, n_components=K,
        log_t_min=math.log(cfg_s.time_min),
        log_t_max=math.log(cfg_s.time_max),
    ).to(DEVICE)

    opt = torch.optim.Adam(flow_t.parameters(), lr=lr)

    best_loss = float("inf")
    best_state = None
    wait = 0
    best_epoch = 0
    t0 = time.perf_counter()

    for epoch in range(max_epochs):
        opt.zero_grad()
        loss = censored_mle_t_only(flow_t, x, delta, z)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow_t.parameters(), max_norm=2.0)
        opt.step()

        cur = float(loss.item())
        if cur < best_loss - 1e-4:
            best_loss = cur
            best_epoch = epoch + 1
            wait = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in flow_t.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    train_sec = time.perf_counter() - t0
    if best_state is not None:
        flow_t.load_state_dict(best_state)

    # Evaluation
    g_eval = torch.Generator(device=DEVICE)
    g_eval.manual_seed(seed + 98765)
    z_eval = torch.normal(
        cfg_s.z_mean, cfg_s.z_std, size=(cfg_s.n_eval, 1),
        generator=g_eval, device=DEVICE, dtype=DTYPE,
    )
    t_true = sample_true_given_z(z_eval, "T", cfg_s, scenario_name,
                                 generator=g_eval)
    with torch.no_grad():
        t_model = enforce_time_bounds(
            flow_t.sample(z_eval, generator=g_eval), cfg_s)

    t_m = cdf_norm_metrics_from_samples(t_true, t_model)
    t_trim = cdf_norm_metrics_trimmed(t_true, t_model, q_high=0.85)

    return {
        "seed": seed,
        "scenario": scenario_name,
        "method": "Ausset2021",
        "T_KS": t_m["Linf_KS"],
        "T_W1": t_m["W1_samples"],
        "T_KS_trim85": t_trim["KS_trim85"],
        "T_W1_trim85": t_trim["W1_trim85"],
        "converged_epoch": best_epoch,
        "train_time_sec": train_sec,
        "censoring_rate": censoring,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="lognormal_base",
                   choices=["lognormal_base", "heavy_tail_t",
                            "cox_t_lognormal_c", "cox_both", "weibull_base"])
    p.add_argument("--mc_runs", type=int, default=30)
    p.add_argument("--n_train", type=int, default=2048)
    p.add_argument("--n_eval", type=int, default=1024)
    p.add_argument("--max_epochs", type=int, default=400)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--tag", default="bigN_v4")
    args = p.parse_args()

    cfg = ExperimentConfig()
    cfg.n_train = args.n_train
    cfg.n_eval = args.n_eval

    all_results = []
    for r in range(args.mc_runs):
        seed = 20260407 + r * 13
        res = train_eval_ausset(cfg, args.scenario, seed,
                                K=args.K,
                                max_epochs=args.max_epochs,
                                patience=args.patience)
        all_results.append(res)
        print(f"[{args.scenario}] run {r+1}/{args.mc_runs} | "
              f"T_KS={res['T_KS']:.4f}, T_W1={res['T_W1']:.4f}, "
              f"epoch={res['converged_epoch']}, "
              f"time={res['train_time_sec']:.1f}s", flush=True)

    ks_vals = [r["T_KS"] for r in all_results]
    w1_vals = [r["T_W1"] for r in all_results]
    summary = {
        "T_KS_mean": float(np.mean(ks_vals)),
        "T_KS_std": float(np.std(ks_vals)),
        "T_W1_mean": float(np.mean(w1_vals)),
        "T_W1_std": float(np.std(w1_vals)),
    }

    print(f"\n=== Ausset 2021 baseline on {args.scenario} ===")
    print(f"T-KS: {summary['T_KS_mean']:.4f} ± {summary['T_KS_std']:.4f}")
    print(f"T-W1: {summary['T_W1_mean']:.4f} ± {summary['T_W1_std']:.4f}")

    out_file = f"ausset2021_{args.scenario}_{args.tag}.json"
    with open(out_file, "w") as f:
        json.dump({"config": vars(args), "summary": summary,
                   "results": all_results}, f, indent=2)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
