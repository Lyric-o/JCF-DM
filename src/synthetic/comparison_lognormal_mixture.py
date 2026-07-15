# -*- coding: utf-8 -*-
"""DSM-inspired conditional LogNormal mixture used in Table 1.

The runner follows the mixture-survival idea of Nagpal et al. (2021), but is
a study-specific implementation rather than the official DSM package.

Model:
  f(t|z) = Σ_{k=1}^{K_mix} π_k(z) * f_LN(t; μ_k(z), σ_k(z))
  S(t|z) = Σ_{k=1}^{K_mix} π_k(z) * S_LN(t; μ_k(z), σ_k(z))

Usage:
    python comparison_lognormal_mixture.py --scenario heavy_tail_t --mc_runs 30 --tag bigN_v3
"""

from __future__ import annotations
import argparse, json, math, random, time
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_fn

from uninformative_censoring_impl import (
    ExperimentConfig, DEVICE, DTYPE,
    generate_dataset, enforce_time_bounds, sample_true_given_z,
    cdf_norm_metrics_from_samples, km_estimator, km_cdf_on_grid,
    fit_cox_beta_1d, breslow_baseline_from_beta, cox_marginal_cdf_on_grid,
    cdf_from_samples_on_grid, curve_distance_metrics, summarize_results,
)
from uninformative_censoring_flow import cdf_norm_metrics_trimmed


class ConditionalLogNormalMixture(nn.Module):
    """Mixture of K_mix conditional LogNormals."""

    def __init__(self, in_dim=1, hidden=64, K_mix=4):
        super().__init__()
        self.K_mix = K_mix
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, K_mix * 2 + K_mix),  # mu_k, log_sigma_k, logit_pi_k
        )

    def _params(self, z):
        out = self.net(z)  # [n, K*3]
        K = self.K_mix
        mu = out[..., :K]                           # [n, K]
        sigma = F_fn.softplus(out[..., K:2*K]) + 0.05  # [n, K], > 0.05
        log_pi = F_fn.log_softmax(out[..., 2*K:], dim=-1)  # [n, K]
        return mu, sigma, log_pi

    def log_pdf(self, y, z):
        """log f(t|z) = log Σ_k π_k * f_LN(t; μ_k, σ_k)"""
        mu, sigma, log_pi = self._params(z)         # [n, K] each
        y_safe = torch.clamp(y, min=1e-8)            # [n, 1]
        log_y = torch.log(y_safe)                     # [n, 1]
        u = (log_y - mu) / sigma                      # [n, K]
        # log f_LN per component
        log_f_k = (-0.5 * math.log(2 * math.pi) - torch.log(sigma)
                   - 0.5 * u**2 - log_y)              # [n, K]
        return torch.logsumexp(log_pi + log_f_k, dim=-1, keepdim=True)  # [n, 1]

    def log_survival(self, y, z):
        """log S(t|z) = log Σ_k π_k * S_LN(t; μ_k, σ_k)"""
        mu, sigma, log_pi = self._params(z)
        y_safe = torch.clamp(y, min=1e-8)
        u = (torch.log(y_safe) - mu) / sigma          # [n, K]
        log_S_k = torch.special.log_ndtr(-u)           # [n, K]
        return torch.logsumexp(log_pi + log_S_k, dim=-1, keepdim=True)

    def sample(self, z, generator=None):
        """Sample from the mixture."""
        mu, sigma, log_pi = self._params(z)
        # Gumbel-max trick for component selection
        if generator is None:
            gumbel = -torch.log(-torch.log(torch.rand_like(log_pi).clamp(min=1e-12)))
        else:
            gumbel = -torch.log(-torch.log(
                torch.rand(log_pi.shape, generator=generator,
                           device=log_pi.device, dtype=log_pi.dtype).clamp(min=1e-12)))
        k = (log_pi + gumbel).argmax(dim=-1, keepdim=True)  # [n, 1]
        mu_sel = torch.gather(mu, -1, k)       # [n, 1]
        sigma_sel = torch.gather(sigma, -1, k)  # [n, 1]
        if generator is None:
            eps = torch.randn_like(mu_sel)
        else:
            eps = torch.randn(mu_sel.shape, generator=generator,
                              device=mu_sel.device, dtype=mu_sel.dtype)
        return torch.exp(mu_sel + sigma_sel * eps)


def train_eval_lognormal_mixture(cfg, scenario_name, seed, K_mix=4, max_epochs=400,
                    patience=50, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data = generate_dataset(cfg.n_train, cfg, seed=seed)
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
        # Censored MLE: δ * log f(x|z) + (1-δ) * log S(x|z)
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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    train_time = time.perf_counter() - t0
    if best_state:
        model.load_state_dict(best_state)

    # Evaluation is T-side only; this mixture does not model C.
    g_eval = torch.Generator(device=DEVICE)
    g_eval.manual_seed(seed + 98765)
    z_eval = torch.normal(cfg.z_mean, cfg.z_std, size=(cfg.n_eval, 1),
                          generator=g_eval, device=DEVICE, dtype=DTYPE)
    t_true = sample_true_given_z(z_eval, "T", cfg, scenario_name, generator=g_eval)

    with torch.no_grad():
        t_model = enforce_time_bounds(model.sample(z_eval, generator=g_eval), cfg)

    t_metrics = cdf_norm_metrics_from_samples(t_true, t_model)
    t_trim = cdf_norm_metrics_trimmed(t_true, t_model, q_high=0.85)

    # Baselines
    grid_t = torch.linspace(
        float(torch.quantile(torch.cat([t_true.view(-1), t_model.view(-1)]), 0.01)),
        float(torch.quantile(torch.cat([t_true.view(-1), t_model.view(-1)]), 0.995)),
        280, device=DEVICE, dtype=DTYPE,
    )
    true_cdf = cdf_from_samples_on_grid(t_true, grid_t)
    gen_cdf = cdf_from_samples_on_grid(t_model, grid_t)
    km_times, km_surv = km_estimator(x, delta)
    km_cdf = km_cdf_on_grid(km_times, km_surv, grid_t)
    t_km = curve_distance_metrics(true_cdf, km_cdf, grid_t)

    out = {
        "seed": seed, "method": f"LogNormalMixture_K{K_mix}",
        "best_nll": best_loss, "converged_epoch": float(best_epoch),
        "train_time_sec": train_time, "censoring_rate": censoring_rate,
    }
    out.update({f"T_{k}": v for k, v in t_metrics.items()})
    out.update({f"T_{k}": v for k, v in t_trim.items()})
    out.update({f"T_compKM_{k}": v for k, v in t_km.items()})
    return out


def main():
    p = argparse.ArgumentParser(description="Study-specific conditional LogNormal mixture")
    p.add_argument("--scenario", type=str, default="all",
                   choices=["lognormal_base", "heavy_tail_t",
                            "cox_t_lognormal_c", "cox_both", "all"])
    p.add_argument("--mc_runs", type=int, default=30)
    p.add_argument("--n_train", type=int, default=2048)
    p.add_argument("--n_eval", type=int, default=1024)
    p.add_argument("--K_mix", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=400)
    p.add_argument("--tag", type=str, default="bigN_v3")
    args = p.parse_args()

    cfg = ExperimentConfig()
    cfg.n_train = args.n_train
    cfg.n_eval = args.n_eval

    if args.scenario == "all":
        scenarios = ["lognormal_base", "heavy_tail_t", "cox_t_lognormal_c", "cox_both"]
        mc_counts = [50, 30, 30, 30]  # match our main runs
    else:
        scenarios = [args.scenario]
        mc_counts = [args.mc_runs]

    all_output = {}
    for scenario, mc in zip(scenarios, mc_counts):
        cfg_s = ExperimentConfig(**asdict(cfg))
        cfg_s.scenario_name = scenario

        print(f"\n=== LogNormal mixture K={args.K_mix}, scenario={scenario}, "
              f"n_train={cfg.n_train}, mc_runs={mc} ===")

        results = []
        for i in range(mc):
            seed = 20260407 + i * 13
            res = train_eval_lognormal_mixture(cfg_s, scenario, seed, K_mix=args.K_mix,
                                 max_epochs=args.max_epochs)
            results.append(res)
            print(f"  Run {i+1:02d}/{mc} | T_W1={res['T_W1_samples']:.4f} "
                  f"T_KS={res['T_Linf_KS']:.4f} "
                  f"epoch={int(res['converged_epoch'])} time={res['train_time_sec']:.1f}s",
                  flush=True)

        numeric_results = [{k: v for k, v in r.items() if isinstance(v, (int, float))}
                           for r in results]
        summary = summarize_results(numeric_results)
        all_output[scenario] = {"results": results, "summary": summary}
        print(f"\n  Summary: T_W1={summary['T_W1_samples']['mean']:.4f}±"
              f"{summary['T_W1_samples']['std']:.4f}, "
              f"T_KS={summary['T_Linf_KS']['mean']:.4f}±"
              f"{summary['T_Linf_KS']['std']:.4f}")

    out_file = f"lognormal_mixture_K{args.K_mix}_{args.tag}.json"
    with open(out_file, "w") as f:
        json.dump({"method": f"LogNormalMixture_K{args.K_mix}", "config": asdict(cfg),
                    "scenarios": all_output}, f, indent=2)
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
