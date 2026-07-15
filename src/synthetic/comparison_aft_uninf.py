# -*- coding: utf-8 -*-
"""AFT-LogN / AFT-Weibull baselines under uninformative censoring."""

from __future__ import annotations
import argparse, json, time
from dataclasses import asdict

import numpy as np
import torch

from uninformative_censoring_impl import (
    ExperimentConfig, DEVICE, DTYPE,
    generate_dataset, sample_true_given_z,
    cdf_norm_metrics_from_samples, cdf_from_samples_on_grid,
    curve_distance_metrics, summarize_results,
)
from competing_methods import (
    aft_fit_lognormal, aft_fit_weibull, aft_cdf_on_grid,
)


def _aft_samples_from_state(state, z_eval, generator):
    """Per-Z sampling using the closed-form AFT inverse-CDF: log T = a + b*z + sigma*eps."""
    a, b, sigma, family = state  # state convention from competing_methods._aft_fit
    mu = a + b * z_eval.view(-1)                              # [n_eval]
    if generator is None:
        u = torch.rand(z_eval.shape[0], device=DEVICE, dtype=DTYPE)
    else:
        u = torch.rand(z_eval.shape[0], generator=generator, device=DEVICE, dtype=DTYPE)
    if family == "lognormal":
        eps = torch.special.ndtri(u.clamp(1e-6, 1 - 1e-6))   # log T = mu + sigma·Φ⁻¹(u)
        log_t = mu + sigma * eps
    elif family == "weibull":
        # log T = mu + sigma · log(-log(1-u))   (Gumbel-min for log T)
        log_t = mu + sigma * torch.log(-torch.log(1.0 - u.clamp(1e-6, 1 - 1e-6)))
    else:
        raise ValueError(family)
    return torch.exp(log_t).view(-1, 1)


def fit_eval_aft(cfg, scenario_name, seed, family):
    np.random.seed(seed); torch.manual_seed(seed)
    data = generate_dataset(cfg.n_train, cfg, seed=seed)
    z, x, delta = data["z"], data["x"], data["delta"]
    censoring_rate = float((1.0 - delta.mean()).item())

    t0 = time.perf_counter()
    if family == "lognormal":
        state = aft_fit_lognormal(z, x, delta)
    else:
        state = aft_fit_weibull(z, x, delta)
    fit_time = time.perf_counter() - t0

    g_eval = torch.Generator(device=DEVICE); g_eval.manual_seed(seed + 98765)
    z_eval = torch.normal(cfg.z_mean, cfg.z_std, size=(cfg.n_eval, 1),
                          generator=g_eval, device=DEVICE, dtype=DTYPE)
    t_true = sample_true_given_z(z_eval, "T", cfg, scenario_name, generator=g_eval)
    t_aft = _aft_samples_from_state(state, z_eval, g_eval)
    t_aft = torch.clamp(t_aft, min=cfg.time_min, max=cfg.time_max)

    t_metrics = cdf_norm_metrics_from_samples(t_true, t_aft)

    # Also compute T-KS via CDF-on-grid (same protocol as embedded comp baselines)
    grid_t = torch.linspace(
        float(torch.quantile(torch.cat([t_true.view(-1), t_aft.view(-1)]), 0.01)),
        float(torch.quantile(torch.cat([t_true.view(-1), t_aft.view(-1)]), 0.995)),
        280, device=DEVICE, dtype=DTYPE,
    )
    true_cdf = cdf_from_samples_on_grid(t_true, grid_t)
    aft_cdf_grid = aft_cdf_on_grid(state, z_eval, grid_t)
    t_curve = curve_distance_metrics(true_cdf, aft_cdf_grid, grid_t)

    out = {
        "seed": seed, "method": f"AFT_{family}",
        "fit_time_sec": fit_time, "censoring_rate": censoring_rate,
    }
    out.update({f"T_{k}": v for k, v in t_metrics.items()})
    out.update({f"T_curve_{k}": v for k, v in t_curve.items()})
    return out


def main():
    p = argparse.ArgumentParser(description="AFT baselines on uninformative DGPs")
    p.add_argument("--scenario", default="all",
                   choices=["lognormal_base", "heavy_tail_t",
                            "cox_t_lognormal_c", "cox_both", "all"])
    p.add_argument("--mc_runs", type=int, default=30)
    p.add_argument("--n_train", type=int, default=2048)
    p.add_argument("--n_eval", type=int, default=1024)
    p.add_argument("--tag", type=str, default="bigN_0504")
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

        for family in ["lognormal", "weibull"]:
            print(f"\n=== AFT-{family}, scenario={scenario}, "
                  f"n_train={cfg.n_train}, mc_runs={mc} ===")

            results = []
            for i in range(mc):
                seed = 20260407 + i * 13
                res = fit_eval_aft(cfg_s, scenario, seed, family)
                results.append(res)
                print(f"  Run {i+1:02d}/{mc} | T_KS={res['T_Linf_KS']:.4f} "
                      f"T_W1={res['T_W1_samples']:.4f} time={res['fit_time_sec']:.2f}s",
                      flush=True)

            numeric = [{k: v for k, v in r.items() if isinstance(v, (int, float))}
                       for r in results]
            summary = summarize_results(numeric)
            key = f"{scenario}__AFT_{family}"
            all_output[key] = {"results": results, "summary": summary}
            print(f"\n  Summary: T_KS={summary['T_Linf_KS']['mean']:.4f}±"
                  f"{summary['T_Linf_KS']['std']:.4f}, "
                  f"T_W1={summary['T_W1_samples']['mean']:.4f}±"
                  f"{summary['T_W1_samples']['std']:.4f}")

    out_file = f"comp_aft_uninf_{args.tag}.json"
    with open(out_file, "w") as f:
        json.dump({"method": "AFT", "config": asdict(cfg),
                   "scenarios": all_output}, f, indent=2)
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
