# -*- coding: utf-8 -*-
"""Binned conditional-KM smoother evaluated unchanged on the Table 2 DGP.

This is the study-specific ``Binned KM smoother@IC`` row.  It is inspired by
Wasserstein--Kaplan--Meier regression, but Zhou and Mueller did not propose a
method named WKM-IC.

Usage:
    python comparison_binned_km_ic.py --rho 1.0 --mc_runs 50 --tag bigN_v3
"""

from __future__ import annotations
import argparse, json, random, time
from dataclasses import asdict

import numpy as np
import torch

from uninformative_censoring_impl import (
    ExperimentConfig, DEVICE, DTYPE,
    enforce_time_bounds,
    cdf_norm_metrics_from_samples, km_estimator, km_cdf_on_grid,
    cdf_from_samples_on_grid, curve_distance_metrics, summarize_results,
)
from uninformative_censoring_flow import cdf_norm_metrics_trimmed
from comparison_binned_km import wkm_fit_predict, sample_from_survival
from informative_censoring_impl import ICConfig, gen_informative


def train_eval_binned_km_ic(cfg, rho, seed, n_bins=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Match the canonical joint-flow DGP exactly, including the Torch Gamma
    # draw and the seed schedule used in informative_censoring_impl.py.
    ic_cfg = ICConfig(n_train=cfg.n_train, n_eval=cfg.n_eval, rho=rho)
    data = gen_informative(cfg.n_train, ic_cfg, seed)
    z, x, delta = data["z"], data["x"], data["delta"]
    censoring_rate = float((1.0 - delta.mean()).item())

    t0 = time.perf_counter()

    # Eval
    g_eval = torch.Generator(device=DEVICE)
    g_eval.manual_seed(seed + 98765)
    data_eval = gen_informative(cfg.n_eval, ic_cfg, seed + 12345)
    z_eval = data_eval["z"]
    t_true = data_eval["t"]

    grid_t = torch.linspace(
        float(torch.quantile(t_true.view(-1), 0.01)),
        float(torch.quantile(t_true.view(-1), 0.995)),
        300, device=DEVICE, dtype=DTYPE,
    )

    # WKM predict
    pred_surv_eval = wkm_fit_predict(z, x, delta, z_eval, grid_t,
                                     n_bins=n_bins)
    wkm_marginal_cdf = 1.0 - pred_surv_eval.mean(dim=0)

    fit_time = time.perf_counter() - t0

    true_cdf = cdf_from_samples_on_grid(t_true, grid_t)
    t_wkm = curve_distance_metrics(true_cdf, wkm_marginal_cdf, grid_t)

    # Sample-based metrics (for W1)
    t_model = sample_from_survival(pred_surv_eval, grid_t, n_per_curve=20,
                                   generator=g_eval)
    t_model = enforce_time_bounds(t_model, cfg)
    t_sample_metrics = cdf_norm_metrics_from_samples(t_true, t_model)
    t_trim = cdf_norm_metrics_trimmed(t_true, t_model, q_high=0.85)

    # KM baseline
    km_times, km_surv = km_estimator(x, delta)
    km_cdf = km_cdf_on_grid(km_times, km_surv, grid_t)
    t_km = curve_distance_metrics(true_cdf, km_cdf, grid_t)

    out = {
        "seed": seed, "method": f"BinnedKM_bins{n_bins}",
        "fit_time_sec": fit_time, "censoring_rate": censoring_rate,
        "rho": rho,
    }
    out.update({f"T_cdf_{k}": v for k, v in t_wkm.items()})
    out.update({f"T_{k}": v for k, v in t_sample_metrics.items()})
    out.update({f"T_{k}": v for k, v in t_trim.items()})
    out.update({f"T_compKM_{k}": v for k, v in t_km.items()})
    return out


def main():
    p = argparse.ArgumentParser(
        description="Study-specific binned KM smoother on informative data")
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--mc_runs", type=int, default=30)
    p.add_argument("--n_train", type=int, default=2048)
    p.add_argument("--n_eval", type=int, default=1024)
    p.add_argument("--n_bins", type=int, default=10)
    p.add_argument("--tag", type=str, default="bigN_v3")
    args = p.parse_args()

    cfg = ExperimentConfig()
    cfg.n_train = args.n_train
    cfg.n_eval = args.n_eval

    print(f"=== Binned KM smoother bins={args.n_bins} on informative DGP, rho={args.rho}, "
          f"mc_runs={args.mc_runs} ===")

    results = []
    for i in range(args.mc_runs):
        seed = 20260407 + i * 13
        res = train_eval_binned_km_ic(cfg, args.rho, seed, n_bins=args.n_bins)
        results.append(res)
        print(f"  Run {i+1:02d}/{args.mc_runs} | T_W1={res['T_W1_samples']:.4f} "
              f"T_KS={res['T_Linf_KS']:.4f} "
              f"time={res['fit_time_sec']:.2f}s", flush=True)

    numeric_results = [{k: v for k, v in r.items()
                        if isinstance(v, (int, float))} for r in results]
    summary = summarize_results(numeric_results)

    out_file = f"binned_km_at_ic_bins{args.n_bins}_rho{args.rho}_{args.tag}.json"
    with open(out_file, "w") as f:
        json.dump({"method": f"BinnedKM_bins{args.n_bins}", "rho": args.rho,
                    "config": asdict(cfg), "results": results,
                    "summary": summary}, f, indent=2)

    print(f"\n=== Summary: Binned KM smoother rho={args.rho} ===")
    for k in ["T_W1_samples", "T_Linf_KS"]:
        print(f"  {k:22s}: {summary[k]['mean']:.4f} +/- {summary[k]['std']:.4f}")
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
