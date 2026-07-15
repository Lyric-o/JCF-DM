# -*- coding: utf-8 -*-
"""WKM-inspired binned conditional-KM smoother used in Table 1.

Core idea: treat KM survival curves as objects in Wasserstein-2 space,
then do Fréchet regression (local weighted averaging) to predict S(t|z).

This is a study-specific approximation, not the authors' official WKM
implementation.  For the one-dimensional simulation covariate it:
1. Partition covariate z into B bins via quantiles.
2. Compute a KM curve within each bin.
3. For a new z_eval, average nearby KM curves using Gaussian kernel weights.
4. Sample from the averaged curve via inverse-CDF.

This is T-only (KM doesn't model C separately), same as vanilla KM but
covariate-adaptive.

Usage:
    python comparison_binned_km.py --scenario all --mc_runs 30 --tag bigN_v3
"""

from __future__ import annotations
import argparse, json, math, random, time
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import torch

from uninformative_censoring_impl import (
    ExperimentConfig, DEVICE, DTYPE,
    generate_dataset, enforce_time_bounds, sample_true_given_z,
    cdf_norm_metrics_from_samples, km_estimator, km_cdf_on_grid,
    cdf_from_samples_on_grid, curve_distance_metrics, summarize_results,
)
from uninformative_censoring_flow import cdf_norm_metrics_trimmed


def km_survival_curve(x, delta, grid):
    """Compute KM survival on a fixed grid."""
    times, surv = km_estimator(x, delta)
    # Convert to CDF on grid
    cdf = km_cdf_on_grid(times, surv, grid)
    return 1.0 - cdf  # survival


def wkm_fit_predict(z_train, x_train, delta_train, z_eval, grid,
                     n_bins=10, bandwidth=None):
    """
    Wasserstein-KM survival regression.

    1. Bin z_train into n_bins quantile bins.
    2. Compute KM survival per bin.
    3. For each z_eval, compute Gaussian kernel weights over bin centers.
    4. Return weighted average survival curve for each z_eval.
    """
    z_flat = z_train.view(-1)
    if bandwidth is None:
        bandwidth = float(z_flat.std().item()) * (n_bins ** (-0.2))  # Silverman-like

    # Quantile bins
    quantiles = torch.linspace(0, 1, n_bins + 1, device=DEVICE, dtype=DTYPE)
    bin_edges = torch.quantile(z_flat, quantiles)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # KM per bin
    bin_survivals = []
    for b in range(n_bins):
        if b < n_bins - 1:
            mask = (z_flat >= bin_edges[b]) & (z_flat < bin_edges[b + 1])
        else:
            mask = (z_flat >= bin_edges[b]) & (z_flat <= bin_edges[b + 1])
        if mask.sum() < 5:
            # Too few — use overall KM
            s = km_survival_curve(x_train, delta_train, grid)
        else:
            s = km_survival_curve(x_train[mask.unsqueeze(1)].view(-1, 1),
                                   delta_train[mask.unsqueeze(1)].view(-1, 1),
                                   grid)
        bin_survivals.append(s)
    bin_survivals = torch.stack(bin_survivals, dim=0)  # [n_bins, grid]

    # Kernel weights for each z_eval
    z_e = z_eval.view(-1, 1)  # [n_eval, 1]
    bc = bin_centers.view(1, -1)  # [1, n_bins]
    w = torch.exp(-0.5 * ((z_e - bc) / bandwidth) ** 2)  # [n_eval, n_bins]
    w = w / (w.sum(dim=1, keepdim=True) + 1e-12)

    # Weighted survival curve per z_eval
    pred_surv = torch.matmul(w, bin_survivals)  # [n_eval, grid]
    return pred_surv


def sample_from_survival(surv_curves, grid, n_per_curve=1, generator=None):
    """Inverse-CDF sampling, pooling all draws across evaluation covariates."""
    # surv_curves: [n_eval, grid_size]
    cdf = 1.0 - surv_curves  # [n_eval, grid_size]
    n_eval = cdf.shape[0]

    if generator is None:
        u = torch.rand(n_eval, n_per_curve, device=DEVICE, dtype=DTYPE)
    else:
        u = torch.rand(n_eval, n_per_curve, generator=generator,
                        device=DEVICE, dtype=DTYPE)

    # For each u, find the grid index where CDF crosses u
    samples = []
    for j in range(n_per_curve):
        uj = u[:, j:j+1].contiguous()  # [n_eval, 1]
        # searchsorted along grid dim
        idx = torch.searchsorted(cdf, uj, right=True)  # [n_eval, 1]
        idx = torch.clamp(idx, 1, grid.numel() - 1)
        t_samples = grid[idx.view(-1)]
        samples.append(t_samples.view(-1, 1))

    # Each draw is a valid sample from T|Z.  Pool the draws to estimate the
    # marginal distribution over Z; averaging them would collapse the
    # conditional variance and does not define a sample from T.
    return torch.cat(samples, dim=1).reshape(-1, 1)  # [n_eval*n_per_curve, 1]


def train_eval_binned_km(cfg, scenario_name, seed, n_bins=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data = generate_dataset(cfg.n_train, cfg, seed=seed)
    z, x, delta = data["z"], data["x"], data["delta"]
    censoring_rate = float((1.0 - delta.mean()).item())

    t0 = time.perf_counter()

    # Eval
    g_eval = torch.Generator(device=DEVICE)
    g_eval.manual_seed(seed + 98765)
    z_eval = torch.normal(cfg.z_mean, cfg.z_std, size=(cfg.n_eval, 1),
                          generator=g_eval, device=DEVICE, dtype=DTYPE)
    t_true = sample_true_given_z(z_eval, "T", cfg, scenario_name, generator=g_eval)

    # Evaluation grid based on true samples
    grid_t = torch.linspace(
        float(torch.quantile(t_true.view(-1), 0.01)),
        float(torch.quantile(t_true.view(-1), 0.995)),
        300, device=DEVICE, dtype=DTYPE,
    )

    # WKM: predict per-z_eval survival, then average to get marginal CDF
    pred_surv_eval = wkm_fit_predict(z, x, delta, z_eval, grid_t, n_bins=n_bins)
    # Marginal predicted CDF = 1 - mean survival over z_eval draws
    wkm_marginal_cdf = 1.0 - pred_surv_eval.mean(dim=0)  # [grid]

    fit_time = time.perf_counter() - t0

    # True marginal CDF from samples
    true_cdf = cdf_from_samples_on_grid(t_true, grid_t)

    # WKM vs true CDF
    t_wkm = curve_distance_metrics(true_cdf, wkm_marginal_cdf, grid_t)

    # Also evaluate via sampling (for W1 metric which needs samples)
    t_model = sample_from_survival(pred_surv_eval, grid_t, n_per_curve=20, generator=g_eval)
    t_model = enforce_time_bounds(t_model, cfg)
    t_sample_metrics = cdf_norm_metrics_from_samples(t_true, t_model)
    t_trim = cdf_norm_metrics_trimmed(t_true, t_model, q_high=0.85)

    # KM baseline (marginal, no covariate)
    km_times, km_surv = km_estimator(x, delta)
    km_cdf = km_cdf_on_grid(km_times, km_surv, grid_t)
    t_km = curve_distance_metrics(true_cdf, km_cdf, grid_t)

    out = {
        "seed": seed, "method": f"BinnedKM_bins{n_bins}",
        "fit_time_sec": fit_time, "censoring_rate": censoring_rate,
    }
    # Primary metrics: CDF-level comparison (no sampling noise)
    out.update({f"T_cdf_{k}": v for k, v in t_wkm.items()})
    # Secondary: sample-based metrics (for W1 comparability)
    out.update({f"T_{k}": v for k, v in t_sample_metrics.items()})
    out.update({f"T_{k}": v for k, v in t_trim.items()})
    out.update({f"T_compKM_{k}": v for k, v in t_km.items()})
    return out


def main():
    p = argparse.ArgumentParser(description="Study-specific binned conditional-KM smoother")
    p.add_argument("--scenario", type=str, default="all",
                   choices=["lognormal_base", "heavy_tail_t",
                            "cox_t_lognormal_c", "cox_both", "all"])
    p.add_argument("--mc_runs", type=int, default=30)
    p.add_argument("--n_train", type=int, default=2048)
    p.add_argument("--n_eval", type=int, default=1024)
    p.add_argument("--n_bins", type=int, default=10)
    p.add_argument("--tag", type=str, default="bigN_v3")
    args = p.parse_args()

    cfg = ExperimentConfig()
    cfg.n_train = args.n_train
    cfg.n_eval = args.n_eval

    if args.scenario == "all":
        scenarios = ["lognormal_base", "heavy_tail_t", "cox_t_lognormal_c", "cox_both"]
        mc_counts = [50, 30, 30, 30]
    else:
        scenarios = [args.scenario]
        mc_counts = [args.mc_runs]

    all_output = {}
    for scenario, mc in zip(scenarios, mc_counts):
        cfg_s = ExperimentConfig(**asdict(cfg))
        cfg_s.scenario_name = scenario

        print(f"\n=== Binned KM smoother bins={args.n_bins}, scenario={scenario}, mc={mc} ===")
        results = []
        for i in range(mc):
            seed = 20260407 + i * 13
            res = train_eval_binned_km(cfg_s, scenario, seed, n_bins=args.n_bins)
            results.append(res)
            print(f"  Run {i+1:02d}/{mc} | T_W1={res['T_W1_samples']:.4f} "
                  f"T_KS={res['T_Linf_KS']:.4f} "
                  f"time={res['fit_time_sec']:.2f}s", flush=True)

        numeric_results = [{k: v for k, v in r.items() if isinstance(v, (int, float))}
                           for r in results]
        summary = summarize_results(numeric_results)
        all_output[scenario] = {"results": results, "summary": summary}
        print(f"  Summary: T_W1={summary['T_W1_samples']['mean']:.4f}±"
              f"{summary['T_W1_samples']['std']:.4f}, "
              f"T_KS={summary['T_Linf_KS']['mean']:.4f}±"
              f"{summary['T_Linf_KS']['std']:.4f}")

    out_file = f"binned_km_bins{args.n_bins}_{args.tag}.json"
    with open(out_file, "w") as f:
        json.dump({"method": f"BinnedKM_bins{args.n_bins}", "config": asdict(cfg),
                    "scenarios": all_output}, f, indent=2)
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
