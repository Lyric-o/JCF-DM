# -*- coding: utf-8 -*-
"""Study-specific deep LogNormal--Frank comparison for Table 2.

This implementation is motivated by deep copula survival models, but it is
not the authors' official DCSurvival code and should not be read as an exact
reproduction of Zhang et al. (2024).

Model:
  - Marginal T|Z: conditional LogNormal(μ_T(z), σ_T(z)) via MLP
  - Marginal C|Z: conditional LogNormal(μ_C(z), σ_C(z)) via MLP
  - Dependence:   Frank survival copula with learnable parameter α
  - Likelihood:   copula-augmented censored MLE

Usage:
    python comparison_deep_lognormal_frank.py --rho 1.0 --mc_runs 30 --tag bigN_v3
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
    cdf_from_samples_on_grid, curve_distance_metrics,
)
from uninformative_censoring_flow import cdf_norm_metrics_trimmed


# ============================================================================
# Deep marginal model: conditional LogNormal
# ============================================================================
class DeepLogNormal(nn.Module):
    def __init__(self, in_dim=1, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),  # mu, log_sigma
        )

    def _params(self, z):
        out = self.net(z)
        mu = out[..., 0:1]
        sigma = F_fn.softplus(out[..., 1:2]) + 0.05  # > 0.05
        return mu, sigma

    def log_pdf(self, y, z):
        mu, sigma = self._params(z)
        y_safe = torch.clamp(y, min=1e-8)
        u = (torch.log(y_safe) - mu) / sigma
        return -0.5 * math.log(2 * math.pi) - torch.log(sigma) - 0.5 * u**2 - torch.log(y_safe)

    def log_survival(self, y, z):
        mu, sigma = self._params(z)
        y_safe = torch.clamp(y, min=1e-8)
        u = (torch.log(y_safe) - mu) / sigma
        return torch.special.log_ndtr(-u)

    def survival(self, y, z):
        return torch.exp(self.log_survival(y, z))

    def cdf(self, y, z):
        return 1.0 - self.survival(y, z)

    def sample(self, z, generator=None):
        mu, sigma = self._params(z)
        if generator is None:
            eps = torch.randn_like(mu)
        else:
            eps = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
        return torch.exp(mu + sigma * eps)


# ============================================================================
# Frank survival copula
# ============================================================================
class FrankCopula(nn.Module):
    """Frank copula on survival margins: C(u, v; α).
    α > 0 → positive dependence, α → 0 → independence.
    """
    def __init__(self, init_alpha=2.0):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha)))

    @property
    def alpha(self):
        return F_fn.softplus(self.log_alpha) + 0.01  # keep α > 0.01

    def forward(self, u, v):
        """C(u, v; α) — joint survival."""
        a = self.alpha
        eu = torch.exp(-a * u)
        ev = torch.exp(-a * v)
        ea = torch.exp(-a)
        num = (eu - 1.0) * (ev - 1.0)
        return -torch.log(torch.clamp(1.0 + num / (ea - 1.0), min=1e-12)) / a

    def log_dCdu(self, u, v):
        """log ∂C/∂u — needed for δ=1 (event observed)."""
        a = self.alpha
        eu = torch.exp(-a * u)
        ev = torch.exp(-a * v)
        ea = torch.exp(-a)
        denom = (ea - 1.0) + (eu - 1.0) * (ev - 1.0)
        # ∂C/∂u = eu * (ev - 1) / denom
        numer = eu * (ev - 1.0)
        return torch.log(torch.clamp(torch.abs(numer), min=1e-12)) - torch.log(torch.clamp(torch.abs(denom), min=1e-12))

    def log_dCdv(self, u, v):
        """log ∂C/∂v — needed for δ=0 (censored)."""
        return self.log_dCdu(v, u)  # Frank is symmetric


# ============================================================================
# Deep copula comparison model
# ============================================================================
class DeepLogNormalFrank(nn.Module):
    def __init__(self, in_dim=1, hidden=64, init_alpha=2.0):
        super().__init__()
        self.marginal_t = DeepLogNormal(in_dim, hidden)
        self.marginal_c = DeepLogNormal(in_dim, hidden)
        self.copula = FrankCopula(init_alpha)

    def censored_nll(self, x, delta, z):
        """Copula-augmented censored negative log-likelihood."""
        log_f_t = self.marginal_t.log_pdf(x, z)
        log_f_c = self.marginal_c.log_pdf(x, z)
        s_t = self.marginal_t.survival(x, z)
        s_c = self.marginal_c.survival(x, z)

        # Copula log-derivatives
        log_dCdu = self.copula.log_dCdu(s_t, s_c)  # for δ=1
        log_dCdv = self.copula.log_dCdv(s_t, s_c)  # for δ=0

        # Log-likelihood: δ * [log f_T + log ∂C/∂u] + (1-δ) * [log f_C + log ∂C/∂v]
        ll = delta * (log_f_t + log_dCdu) + (1.0 - delta) * (log_f_c + log_dCdv)
        return -ll.mean()

    def sample_t(self, z, generator=None):
        return self.marginal_t.sample(z, generator=generator)

    def sample_c(self, z, generator=None):
        return self.marginal_c.sample(z, generator=generator)


# ============================================================================
# Train + eval one MC run
# ============================================================================
def train_eval_deep_lognormal_frank(cfg, rho, seed, max_epochs=400, patience=50, lr=1e-3,
                      S_eval=20):
    from informative_censoring_impl import ICConfig, gen_informative
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ic_cfg = ICConfig(n_train=cfg.n_train, n_eval=cfg.n_eval, rho=rho)
    data = gen_informative(cfg.n_train, ic_cfg, seed)
    z, x, delta = data["z"], data["x"], data["delta"]
    censoring_rate = float((1.0 - delta.mean()).item())

    model = DeepLogNormalFrank(in_dim=1, hidden=64, init_alpha=2.0).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_state = None
    wait = 0
    best_epoch = 0
    t0 = time.perf_counter()

    for epoch in range(max_epochs):
        opt.zero_grad()
        loss = model.censored_nll(x, delta, z)
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

    # Eval
    g_eval = torch.Generator(device=DEVICE)
    g_eval.manual_seed(seed + 98765)
    data_eval = gen_informative(cfg.n_eval, ic_cfg, seed + 12345)
    z_eval = data_eval["z"]
    t_true = data_eval["t"]
    c_true = data_eval["c"]
    z_rep = z_eval.unsqueeze(0).expand(S_eval, -1, -1).reshape(-1, 1)

    with torch.no_grad():
        t_model = enforce_time_bounds(model.sample_t(z_rep, generator=g_eval), cfg)
        c_model = enforce_time_bounds(model.sample_c(z_rep, generator=g_eval), cfg)

    t_metrics = cdf_norm_metrics_from_samples(t_true, t_model)
    c_metrics = cdf_norm_metrics_from_samples(c_true, c_model)
    t_trim = cdf_norm_metrics_trimmed(t_true, t_model, q_high=0.85)
    c_trim = cdf_norm_metrics_trimmed(c_true, c_model, q_high=0.85)

    # KM + Cox baselines
    grid_t = torch.linspace(
        float(torch.quantile(torch.cat([t_true.view(-1), t_model.view(-1)]), 0.01)),
        float(torch.quantile(torch.cat([t_true.view(-1), t_model.view(-1)]), 0.995)),
        280, device=DEVICE, dtype=DTYPE,
    )
    true_t_cdf = cdf_from_samples_on_grid(t_true, grid_t)
    km_times, km_surv = km_estimator(x, delta)
    km_cdf = km_cdf_on_grid(km_times, km_surv, grid_t)
    t_km = curve_distance_metrics(true_t_cdf, km_cdf, grid_t)

    out = {
        "seed": seed, "method": "DeepLogNormal_Frank_study_specific",
        "best_nll": best_loss, "converged_epoch": float(best_epoch),
        "train_time_sec": train_time, "censoring_rate": censoring_rate,
        "copula_alpha": float(model.copula.alpha.item()),
    }
    out.update({f"T_{k}": v for k, v in t_metrics.items()})
    out.update({f"C_{k}": v for k, v in c_metrics.items()})
    out.update({f"T_{k}": v for k, v in t_trim.items()})
    out.update({f"C_{k}": v for k, v in c_trim.items()})
    out.update({f"T_compKM_{k}": v for k, v in t_km.items()})
    return out


def main():
    p = argparse.ArgumentParser(description="Study-specific deep LogNormal--Frank comparison")
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--mc_runs", type=int, default=30)
    p.add_argument("--n_train", type=int, default=2048)
    p.add_argument("--n_eval", type=int, default=1024)
    p.add_argument("--max_epochs", type=int, default=400)
    p.add_argument("--start_index", type=int, default=0,
                   help="First Monte Carlo replication index (for sharding)")
    p.add_argument("--tag", type=str, default="bigN_v3")
    args = p.parse_args()

    cfg = ExperimentConfig()
    cfg.n_train = args.n_train
    cfg.n_eval = args.n_eval

    print(f"=== Deep LogNormal--Frank, rho={args.rho}, "
          f"n_train={cfg.n_train}, mc_runs={args.mc_runs} ===")

    results = []
    for local_i in range(args.mc_runs):
        i = args.start_index + local_i
        seed = 20260407 + i * 13
        res = train_eval_deep_lognormal_frank(cfg, args.rho, seed, max_epochs=args.max_epochs)
        results.append(res)
        print(f"  Run {i+1:02d} (shard {local_i+1:02d}/{args.mc_runs}) | "
              f"T_W1={res['T_W1_samples']:.4f} "
              f"T_KS={res['T_Linf_KS']:.4f} α={res['copula_alpha']:.2f} "
              f"epoch={int(res['converged_epoch'])} time={res['train_time_sec']:.1f}s",
              flush=True)

    # Summary — strip string fields before calling summarize_results
    # (it expects all non-"seed" values to be numeric)
    from uninformative_censoring_impl import summarize_results
    numeric_results = [{k: v for k, v in r.items() if isinstance(v, (int, float))}
                       for r in results]
    summary = summarize_results(numeric_results)

    out_file = f"deep_lognormal_frank_rho{args.rho}_{args.tag}.json"
    with open(out_file, "w") as f:
        json.dump({"method": "DeepLogNormal_Frank_study_specific", "rho": args.rho,
                    "config": asdict(cfg), "results": results, "summary": summary},
                  f, indent=2)

    print(f"\n=== Summary: Deep LogNormal--Frank rho={args.rho} ===")
    for k in ["T_W1_samples", "T_Linf_KS", "C_W1_samples", "C_Linf_KS"]:
        print(f"  {k:22s}: {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")
    print(f"  copula_alpha          : {np.mean([r['copula_alpha'] for r in results]):.2f}")
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
