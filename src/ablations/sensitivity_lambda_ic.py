# -*- coding: utf-8 -*-
"""Sensitivity sweep over the Wasserstein weight lambda_w under informative censoring."""

from __future__ import annotations
import argparse, json, time
from dataclasses import asdict, replace
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "synthetic"))

import informative_censoring_impl as ic_mod
from informative_censoring_impl import ICConfig, train_one


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--family", default="spline",
                   choices=["spline", "lognormal", "weibull", "mixln", "rqs"])
    p.add_argument("--mc_runs", type=int, default=20)
    p.add_argument("--n_train", type=int, default=2048)
    p.add_argument("--n_eval", type=int, default=1024)
    p.add_argument("--K_eps", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=400)
    p.add_argument("--patience", type=int, default=35)
    p.add_argument("--s_sim", type=int, default=3)
    p.add_argument("--lambda_alpha", type=float, default=0.01,
                   help="weight on aux gauge-fix penalty (Appendix B (A3))")
    p.add_argument("--suffix", type=str, default="lam_ic_0504")
    args = p.parse_args()

    # Match uninformative sensitivity grid for direct comparability
    sweep = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]

    # Allow --lambda_alpha override on the IC gauge-fix term
    ic_mod.LAMBDA_ALPHA_INF = args.lambda_alpha

    all_results: Dict[str, Dict] = {}

    for lam in sweep:
        cfg = ICConfig(
            family=args.family, rho=args.rho,
            mc_runs=args.mc_runs, n_train=args.n_train, n_eval=args.n_eval,
            K_eps=args.K_eps, max_epochs=args.max_epochs, patience=args.patience,
            s_sim=args.s_sim, lambda_w=lam,
        )
        key = f"lambda={lam:.2f}_S={args.s_sim}"
        print(f"\n=== Config: {key}  (rho={args.rho}, family={args.family}, "
              f"n_train={args.n_train}, K_eps={args.K_eps}) ===", flush=True)

        runs: List[Dict] = []
        for r in range(args.mc_runs):
            # Use same seed sequence as canonical IC bigN run (20260407 + i*13)
            # so each seed's training data is identical across lambda values.
            seed = 20260407 + r * 13
            t_run0 = time.perf_counter()
            res = train_one(cfg, seed)
            res["lambda_w"] = lam
            res["seed"] = seed
            runs.append(res)
            print(f"  run {r+1:02d}/{args.mc_runs} | T_KS={res['T_KS']:.4f} "
                  f"T_W1={res['T_W1']:.4f} C_KS={res['C_KS']:.4f} "
                  f"epoch={int(res['converged_epoch'])} "
                  f"time={res['train_time_sec']:.1f}s", flush=True)

        ks_vals = [r["T_KS"] for r in runs]
        w1_vals = [r["T_W1"] for r in runs]
        c_ks_vals = [r["C_KS"] for r in runs]
        c_w1_vals = [r["C_W1"] for r in runs]
        all_results[key] = {
            "lambda_w": lam, "S": args.s_sim,
            "T_KS_mean": float(np.mean(ks_vals)),
            "T_KS_std":  float(np.std(ks_vals)),
            "T_W1_mean": float(np.mean(w1_vals)),
            "T_W1_std":  float(np.std(w1_vals)),
            "C_KS_mean": float(np.mean(c_ks_vals)),
            "C_KS_std":  float(np.std(c_ks_vals)),
            "C_W1_mean": float(np.mean(c_w1_vals)),
            "C_W1_std":  float(np.std(c_w1_vals)),
            "runs": runs,
        }

    print(f"\n\n{'='*70}")
    print(f"Sensitivity (lambda) — IC dgp1 rho={args.rho} — {args.mc_runs} MC")
    print(f"{'='*70}")
    print(f"{'Config':<25s} {'T-KS mean±std':<22s} {'T-W1 mean±std':<22s}")
    print("-" * 70)
    for k, v in all_results.items():
        print(f"{k:<25s} {v['T_KS_mean']:.4f}±{v['T_KS_std']:.4f}     "
              f"{v['T_W1_mean']:.4f}±{v['T_W1_std']:.4f}")

    out_file = f"sensitivity_lambda_ic_dgp1_rho{args.rho}_{args.suffix}.json"
    with open(out_file, "w") as f:
        json.dump({"config": {"dgp": "ic_dgp1", "rho": args.rho,
                              "family": args.family, "mc_runs": args.mc_runs,
                              "n_train": args.n_train, "n_eval": args.n_eval,
                              "K_eps": args.K_eps, "s_sim": args.s_sim,
                              "max_epochs": args.max_epochs,
                              "lambda_alpha": args.lambda_alpha,
                              "mode": "lambda_ic"},
                   "results": all_results}, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
