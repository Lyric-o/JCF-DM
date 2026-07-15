# -*- coding: utf-8 -*-
"""T-only conditional flow ablation (Ausset et al. 2021) under informative censoring."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from survival_dists import build_dist
from uninformative_censoring_impl import (
    DEVICE, DTYPE,
    cdf_norm_metrics_from_samples,
)
from uninformative_censoring_flow import cdf_norm_metrics_trimmed

# ---- DGP1: shared Gamma frailty Cox (same as informative_censoring_impl.py) ----
from informative_censoring_impl import (
    ICConfig, gen_informative,
)

# ---- DGP2: shared LogN-Weibull frailty ----
from informative_censoring_dgp2 import (
    ICConfig2, gen_dgp2,
)


def censored_mle_t_only(flow_t, x, delta, z):
    """
    Standard T-only censored MLE. Assumes T ⊥ C | Z.
    Under informative censoring this is BIASED.
    """
    log_f_t = flow_t.log_pdf(x, z)
    log_s_t = flow_t.log_survival(x, z)
    ell = delta * log_f_t + (1.0 - delta) * log_s_t
    return -ell.mean()


def train_one_ausset_ic(dgp, rho, n_train, n_eval, seed,
                        max_epochs=400, patience=50):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Generate data from informative DGP
    if dgp == "dgp1":
        cfg = ICConfig(n_train=n_train, n_eval=n_eval, rho=rho)
        data = gen_informative(n_train, cfg, seed)
    else:  # dgp2
        cfg = ICConfig2(n_train=n_train, n_eval=n_eval, rho=rho)
        data = gen_dgp2(n_train, cfg, seed)

    z, x, delta = data["z"], data["x"], data["delta"]
    censoring = float((1.0 - delta.mean()).item())

    # T-only flow: ctx_dim=1 (just z, no shared ε)
    flow_t = build_dist("spline", ctx_dim=1, hidden=32).to(DEVICE)
    opt = torch.optim.Adam(flow_t.parameters(), lr=1e-3)

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
            best_state = {k: v.cpu().clone()
                          for k, v in flow_t.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    train_sec = time.perf_counter() - t0
    if best_state is not None:
        flow_t.load_state_dict(best_state)

    # Evaluation: generate fresh true T samples
    if dgp == "dgp1":
        eval_data = gen_informative(n_eval, cfg, seed + 12345)
    else:
        eval_data = gen_dgp2(n_eval, cfg, seed + 12345)

    z_e = eval_data["z"]
    t_true = eval_data["t"]

    # Sample from T-only model (no ε marginalization needed)
    g_eval = torch.Generator(device=DEVICE)
    g_eval.manual_seed(seed + 98765)

    # Draw S_eval=20 samples per individual and pool (matching joint model eval)
    S_eval = 20
    t_model_list = []
    with torch.no_grad():
        for _ in range(S_eval):
            t_s = flow_t.sample(z_e, generator=g_eval)
            t_s = torch.clamp(t_s, 1e-4, 10.0)
            t_model_list.append(t_s)
    t_model = torch.cat(t_model_list, dim=0)
    t_true_rep = t_true.repeat(S_eval, 1)

    t_m = cdf_norm_metrics_from_samples(t_true_rep, t_model)
    t_trim = cdf_norm_metrics_trimmed(t_true_rep, t_model, q_high=0.85)

    return {
        "seed": seed,
        "dgp": dgp,
        "rho": rho,
        "method": "Ausset2021_T_only",
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
    p.add_argument("--dgp", default="dgp1", choices=["dgp1", "dgp2"])
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--mc_runs", type=int, default=30)
    p.add_argument("--n_train", type=int, default=2048)
    p.add_argument("--n_eval", type=int, default=1024)
    p.add_argument("--max_epochs", type=int, default=400)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--suffix", default="bigN_v4")
    args = p.parse_args()

    all_results = []
    for r in range(args.mc_runs):
        seed = 20260407 + r * 13
        res = train_one_ausset_ic(
            args.dgp, args.rho, args.n_train, args.n_eval, seed,
            max_epochs=args.max_epochs, patience=args.patience)
        all_results.append(res)
        print(f"[Ausset-IC {args.dgp} rho={args.rho}] "
              f"run {r+1}/{args.mc_runs} | "
              f"T_KS={res['T_KS']:.4f}, T_W1={res['T_W1']:.4f}, "
              f"epoch={res['converged_epoch']}, "
              f"cens={res['censoring_rate']:.2f}, "
              f"time={res['train_time_sec']:.1f}s", flush=True)

    ks_vals = [r["T_KS"] for r in all_results]
    w1_vals = [r["T_W1"] for r in all_results]
    summary = {
        "T_KS_mean": float(np.mean(ks_vals)),
        "T_KS_std": float(np.std(ks_vals)),
        "T_W1_mean": float(np.mean(w1_vals)),
        "T_W1_std": float(np.std(w1_vals)),
    }

    print(f"\n=== Ausset T-only on {args.dgp} rho={args.rho} ===")
    print(f"T-KS: {summary['T_KS_mean']:.4f} ± {summary['T_KS_std']:.4f}")
    print(f"T-W1: {summary['T_W1_mean']:.4f} ± {summary['T_W1_std']:.4f}")

    out_file = f"ausset_ic_{args.dgp}_rho{args.rho}_{args.suffix}.json"
    with open(out_file, "w") as f:
        json.dump({"config": vars(args), "summary": summary,
                   "results": all_results}, f, indent=2)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
