# -*- coding: utf-8 -*-
"""Conditional monotone flow for the uninformative-censoring simulations.

The transformation is monotone on log time, has closed-form density and
survival evaluation, and uses implicit differentiation through numerical
inversion for sampling in the Soft--Nelson--Aalen regularizer.

Flow definition
---------------
    h(log_t; z) = a(z) * log_t + b(z)
                  + sum_{k=1..K} alpha_k(z) * tanh((log_t - mu_k(z)) / s_k(z))

with constraints (via softplus):
    a(z) > 0,  alpha_k(z) >= 0,  s_k(z) > 0
=> h is strictly increasing in log_t (and hence in t).

Base distribution: U = h(log_T; Z) ~ N(0, 1).
=> F_T(t|z) = Phi(h),  S_T(t|z) = Phi(-h),  f_T(t|z) = phi(h) * (dh/d log_t) * (1/t)

Sampling: bisect h(log_t; z) = u for u ~ N(0,1) inside [log time_min, log time_max].
Differentiable sampling via implicit reparameterization (Figurnov 2018):
    log_t_diff = log_t*.detach() - (h(log_t*, z, theta) - u) / dh.detach()

The DGPs, classical comparisons, plotting, and Monte Carlo harness are shared
with ``uninformative_censoring_impl.py``.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# gauge-fix penalty weight (Appendix B (A3)). Set via --lambda_alpha.
LAMBDA_ALPHA: float = 0.01
# number of tanh components. Set via --K. K=0 is the LogN ablation.
N_COMPONENTS: int = 4

import uninformative_censoring_impl as base
from uninformative_censoring_impl import (
    ExperimentConfig, DEVICE, DTYPE,
    generate_dataset, build_grid_from_data, hard_na_survival_on_grid,
    soft_na_survival_on_grid, wasserstein_curve_distance, mle_loss,
    enforce_time_bounds, sample_true_given_z,
    cdf_norm_metrics_from_samples, cdf_from_samples_on_grid,
    quantile_abs_errors_from_samples, curve_distance_metrics,
    km_estimator, km_cdf_on_grid, fit_cox_beta_1d, breslow_baseline_from_beta,
    cox_marginal_cdf_on_grid,
    make_selected_seed_comparison_plot, make_summary_plot, summarize_results,
)


# ============================================================================
# Real conditional monotone normalizing flow on log(t).
# ============================================================================
class MonotoneFlow1D(nn.Module):
    def __init__(
        self,
        in_dim: int = 1,
        hidden_dim: int = 32,
        n_components: int = 4,
        log_t_min: float = math.log(1e-4),
        log_t_max: float = math.log(10.0),
    ):
        super().__init__()
        self.K = n_components
        self.log_t_min = float(log_t_min)
        self.log_t_max = float(log_t_max)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # heads: a (1), b (1), alpha (K), mu (K), s (K)
        self.head = nn.Linear(hidden_dim, 2 + 3 * self.K)

    def params(self, z: torch.Tensor):
        h = self.net(z)
        out = self.head(h)
        K = self.K
        a_raw = out[..., 0:1]
        b = out[..., 1:2]
        alpha_raw = out[..., 2:2 + K]
        mu = out[..., 2 + K:2 + 2 * K]
        s_raw = out[..., 2 + 2 * K:2 + 3 * K]

        a = F.softplus(a_raw) + 0.10        # > 0, slope baseline
        alpha = F.softplus(alpha_raw) * 0.5 # >= 0, kept moderate
        s = F.softplus(s_raw) + 0.20        # > 0
        return a, b, alpha, mu, s

    def h_and_dh(self, log_t: torch.Tensor, z: torch.Tensor):
        """Returns (h, dh/d log_t), both shape [n, 1]."""
        a, b, alpha, mu, s = self.params(z)
        # Broadcasting: log_t [n,1], mu/s/alpha [n,K]
        u = (log_t - mu) / s
        tanh_u = torch.tanh(u)
        h = a * log_t + b + (alpha * tanh_u).sum(dim=-1, keepdim=True)
        sech2 = 1.0 - tanh_u ** 2
        dh = a + (alpha * sech2 / s).sum(dim=-1, keepdim=True)
        return h, dh

    def log_pdf(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        y_safe = torch.clamp(y, min=eps)
        log_t = torch.log(y_safe)
        h, dh = self.h_and_dh(log_t, z)
        log_phi = -0.5 * math.log(2.0 * math.pi) - 0.5 * h ** 2
        return log_phi + torch.log(torch.clamp(dh, min=1e-12)) - log_t

    def log_survival(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # use torch.special.log_ndtr (= log Phi) directly
        # instead of log(clamp(ndtr(-h), 1e-12)). The latter clamps in the
        # deep tail and saturates the gradient; log_ndtr is the proper
        # log-CDF and is numerically stable across the full real line.
        y_safe = torch.clamp(y, min=1e-8)
        h, _ = self.h_and_dh(torch.log(y_safe), z)
        return torch.special.log_ndtr(-h)

    def aux_penalty(self, z: torch.Tensor) -> torch.Tensor:
        # gauge-fixing regularizer matching Appendix B (A3).
        # The shift symmetry (b, alpha_k) -> (b + c, alpha_k + c/K) leaves
        # the likelihood unchanged but creates an outlier-prone direction
        # in parameter space (the heavy_tail_t C-side outlier we observed).
        # Penalizing the squared sum of alpha_k(z) over the batch breaks
        # this symmetry without restricting the model class on-shell.
        _, _, alpha, _, _ = self.params(z)
        return alpha.sum(dim=-1).pow(2).mean()

    @torch.no_grad()
    def _bisect_log_t(self, u: torch.Tensor, z: torch.Tensor, n_iter: int = 40) -> torch.Tensor:
        lo = torch.full_like(u, self.log_t_min)
        hi = torch.full_like(u, self.log_t_max)
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            h_mid, _ = self.h_and_dh(mid, z)
            go_right = (h_mid < u).to(mid.dtype)
            lo = go_right * mid + (1 - go_right) * lo
            hi = (1 - go_right) * mid + go_right * hi
        return 0.5 * (lo + hi)

    def sample(self, z: torch.Tensor, n_samples: int = 1, generator=None,
               differentiable: bool = False) -> torch.Tensor:
        if n_samples != 1:
            raise NotImplementedError("Only n_samples=1 supported (matches base API).")
        if generator is None:
            u = torch.randn(z.shape, device=z.device, dtype=z.dtype)
        else:
            u = torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype)

        log_t_star = self._bisect_log_t(u, z)
        if not differentiable:
            return torch.exp(log_t_star)

        # Implicit reparameterization gradient (Figurnov 2018):
        # log_t* solves h(log_t*, z, θ) = u, so d log_t / d θ = -∂h/∂θ / ∂h/∂log_t.
        h_star, dh_star = self.h_and_dh(log_t_star, z)
        log_t_diff = log_t_star.detach() - (h_star - u) / dh_star.detach()
        return torch.exp(log_t_diff)


# ============================================================================
# Trimmed (RMST-style) W1 / KS metrics. YL 2026-04-09.
#
# Restricts the integration / supremum to the interval [0, q_high-quantile of
# the pooled samples], in the spirit of restricted-mean-survival-time
# evaluation. This addresses the W1 inflation observed in cox_t_lognormal_c
# where the bulk CDF offset is small in pointwise max (KS = 0.12) but the
# offset is integrated over a wide [0, t_max] window, producing W1 ~ 0.6
# without indicating a real localized misfit.
# ============================================================================
@torch.no_grad()
def cdf_norm_metrics_trimmed(true_samples: torch.Tensor,
                              model_samples: torch.Tensor,
                              q_high: float = 0.85) -> Dict[str, float]:
    t = true_samples.view(-1)
    m = model_samples.view(-1)
    pooled = torch.cat([t, m], dim=0)
    gmin = float(torch.quantile(pooled, 0.01).item())
    gmax = float(torch.quantile(pooled, q_high).item())
    grid = torch.linspace(max(1e-5, gmin),
                          max(gmin + 1e-4, gmax),
                          300, device=t.device, dtype=t.dtype)

    t_sorted, _ = torch.sort(t)
    m_sorted, _ = torch.sort(m)
    Ft = torch.searchsorted(t_sorted, grid, right=True).to(DTYPE) / t_sorted.numel()
    Fm = torch.searchsorted(m_sorted, grid, right=True).to(DTYPE) / m_sorted.numel()

    diff = Ft - Fm
    dt = grid[1:] - grid[:-1]
    w1_trim = torch.sum(torch.abs(diff[:-1]) * dt)
    ks_trim = torch.max(torch.abs(diff))
    return {
        "W1_trim85": float(w1_trim.item()),
        "KS_trim85": float(ks_trim.item()),
    }


# ============================================================================
# Train + evaluate one MC run. Mirrors base.train_one_experiment but uses
# MonotoneFlow1D and differentiable sampling for the regularizer.
# ============================================================================
def train_one_experiment(cfg: ExperimentConfig, seed: int) -> Dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data_train = generate_dataset(cfg.n_train, cfg, seed=seed)
    z = data_train["z"]
    x = data_train["x"]
    delta = data_train["delta"]
    censoring_rate = float((1.0 - delta.mean()).item())

    grid = build_grid_from_data(x, cfg.grid_size)
    s_obs = hard_na_survival_on_grid(x, delta, grid)

    flow_kwargs = dict(
        in_dim=1,
        hidden_dim=cfg.hidden_dim,
        n_components=N_COMPONENTS,
        log_t_min=math.log(cfg.time_min),
        log_t_max=math.log(cfg.time_max),
    )
    flow_t = MonotoneFlow1D(**flow_kwargs).to(DEVICE)
    flow_c = MonotoneFlow1D(**flow_kwargs).to(DEVICE)

    opt = torch.optim.Adam(
        list(flow_t.parameters()) + list(flow_c.parameters()), lr=cfg.lr
    )

    best_loss = float("inf")
    best_state_t = None
    best_state_c = None
    wait = 0
    best_epoch = 0
    trained_epochs = 0
    train_start = time.perf_counter()

    for _epoch in range(cfg.max_epochs):
        trained_epochs = _epoch + 1
        opt.zero_grad()

        loss_mle = mle_loss(flow_t, flow_c, x, delta, z)

        reg = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)
        for _ in range(cfg.s_sim):
            t_sim = flow_t.sample(z, differentiable=True)
            c_sim = flow_c.sample(z, differentiable=True)
            t_sim = torch.clamp(t_sim, min=cfg.time_min, max=cfg.time_max)
            c_sim = torch.clamp(c_sim, min=cfg.time_min, max=cfg.time_max)
            x_sim = torch.minimum(t_sim, c_sim)
            d_sim_soft = torch.sigmoid((c_sim - t_sim) / cfg.tau_temp)
            s_sim_curve = soft_na_survival_on_grid(x_sim, d_sim_soft, grid, cfg.soft_alpha)
            reg = reg + wasserstein_curve_distance(s_obs, s_sim_curve, grid)
        reg = reg / cfg.s_sim

        # gauge fix ONLY on C-side. The 2×2 ablation
        # (04100046) showed λ_α on flow_t suppresses tanh components that
        # heavy_tail_t's T-side needs (T-W1 0.422→0.301 when removed,
        # p<0.0001). C-side is over-parameterized (truth is LogNormal,
        # K=4 is over-spec) and benefits from the penalty (C-KS 0.148→0.104).
        aux = flow_c.aux_penalty(z)
        loss = loss_mle + cfg.lambda_w * reg + LAMBDA_ALPHA * aux
        loss.backward()
        # tightened from 5.0 to 2.0 for additional outlier
        # protection in over-parameterized regimes.
        torch.nn.utils.clip_grad_norm_(
            list(flow_t.parameters()) + list(flow_c.parameters()), max_norm=2.0
        )
        opt.step()

        cur = float(loss.item())
        if cur < best_loss - cfg.min_delta:
            best_loss = cur
            best_epoch = _epoch + 1
            best_state_t = {k: v.detach().cpu().clone() for k, v in flow_t.state_dict().items()}
            best_state_c = {k: v.detach().cpu().clone() for k, v in flow_c.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    train_time_sec = time.perf_counter() - train_start
    if best_state_t is not None:
        flow_t.load_state_dict(best_state_t)
    if best_state_c is not None:
        flow_c.load_state_dict(best_state_c)

    # ---- Evaluation ----
    g_eval = torch.Generator(device=DEVICE)
    g_eval.manual_seed(seed + 98765)
    z_eval = torch.normal(
        cfg.z_mean, cfg.z_std, size=(cfg.n_eval, 1),
        generator=g_eval, device=DEVICE, dtype=DTYPE,
    )

    t_true = sample_true_given_z(z_eval, "T", cfg, cfg.scenario_name, generator=g_eval)
    c_true = sample_true_given_z(z_eval, "C", cfg, cfg.scenario_name, generator=g_eval)

    with torch.no_grad():
        t_model = enforce_time_bounds(flow_t.sample(z_eval, generator=g_eval), cfg)
        c_model = enforce_time_bounds(flow_c.sample(z_eval, generator=g_eval), cfg)

    t_metrics = cdf_norm_metrics_from_samples(t_true, t_model)
    c_metrics = cdf_norm_metrics_from_samples(c_true, c_model)
    # trimmed (RMST-style) W1 over [0, q_85] of pooled samples
    # to address the wide-time-range CDF-offset inflation in Cox scenarios.
    t_metrics_trim = cdf_norm_metrics_trimmed(t_true, t_model, q_high=0.85)
    c_metrics_trim = cdf_norm_metrics_trimmed(c_true, c_model, q_high=0.85)
    t_qae = quantile_abs_errors_from_samples(t_true, t_model, [0.1, 0.5, 0.9])
    c_qae = quantile_abs_errors_from_samples(c_true, c_model, [0.1, 0.5, 0.9])

    grid_t = torch.linspace(
        float(torch.quantile(torch.cat([t_true.view(-1), t_model.view(-1)]), 0.01).item()),
        float(torch.quantile(torch.cat([t_true.view(-1), t_model.view(-1)]), 0.995).item()),
        280, device=DEVICE, dtype=DTYPE,
    )
    true_t_cdf_grid = cdf_from_samples_on_grid(t_true, grid_t)
    gen_t_cdf_grid = cdf_from_samples_on_grid(t_model, grid_t)

    km_times, km_surv = km_estimator(x, delta)
    km_t_cdf_grid = km_cdf_on_grid(km_times, km_surv, grid_t)
    beta_cox = fit_cox_beta_1d(z, x, delta)
    cox_bt, cox_h0 = breslow_baseline_from_beta(beta_cox, z, x, delta)
    cox_t_cdf_grid = cox_marginal_cdf_on_grid(beta_cox, cox_bt, cox_h0, z_eval, grid_t)

    t_km_metrics = curve_distance_metrics(true_t_cdf_grid, km_t_cdf_grid, grid_t)
    t_cox_metrics = curve_distance_metrics(true_t_cdf_grid, cox_t_cdf_grid, grid_t)

    # AFT baselines (LogNormal + Weibull)
    from competing_methods import aft_fit_lognormal, aft_fit_weibull, aft_cdf_on_grid
    aft_ln_state = aft_fit_lognormal(z, x, delta)
    aft_ln_cdf_grid = aft_cdf_on_grid(aft_ln_state, z_eval, grid_t)
    t_aftln_metrics = curve_distance_metrics(true_t_cdf_grid, aft_ln_cdf_grid, grid_t)
    aft_w_state = aft_fit_weibull(z, x, delta)
    aft_w_cdf_grid = aft_cdf_on_grid(aft_w_state, z_eval, grid_t)
    t_aftw_metrics = curve_distance_metrics(true_t_cdf_grid, aft_w_cdf_grid, grid_t)

    grid_c = torch.linspace(
        float(torch.quantile(torch.cat([c_true.view(-1), c_model.view(-1)]), 0.01).item()),
        float(torch.quantile(torch.cat([c_true.view(-1), c_model.view(-1)]), 0.995).item()),
        280, device=DEVICE, dtype=DTYPE,
    )
    true_c_cdf_grid = cdf_from_samples_on_grid(c_true, grid_c)
    gen_c_cdf_grid = cdf_from_samples_on_grid(c_model, grid_c)

    out = {
        "seed": seed,
        "best_train_loss": best_loss,
        "converged_epoch": float(best_epoch),
        "trained_epochs": float(trained_epochs),
        "train_time_sec": float(train_time_sec),
        "censoring_rate": censoring_rate,
    }
    out.update({f"T_{k}": v for k, v in t_metrics.items()})
    out.update({f"C_{k}": v for k, v in c_metrics.items()})
    out.update({f"T_{k}": v for k, v in t_metrics_trim.items()})
    out.update({f"C_{k}": v for k, v in c_metrics_trim.items()})
    out.update({f"T_{k}": v for k, v in t_qae.items()})
    out.update({f"C_{k}": v for k, v in c_qae.items()})
    out.update({f"T_compKM_{k}": v for k, v in t_km_metrics.items()})
    out.update({f"T_compCox_{k}": v for k, v in t_cox_metrics.items()})
    out.update({f"T_compAFTLN_{k}": v for k, v in t_aftln_metrics.items()})
    out.update({f"T_compAFTW_{k}": v for k, v in t_aftw_metrics.items()})

    out["_plot_payload"] = {
        "grid_t": grid_t.detach().cpu().numpy(),
        "t_true_cdf": true_t_cdf_grid.detach().cpu().numpy(),
        "t_gen_cdf": gen_t_cdf_grid.detach().cpu().numpy(),
        "t_km_cdf": km_t_cdf_grid.detach().cpu().numpy(),
        "t_cox_cdf": cox_t_cdf_grid.detach().cpu().numpy(),
        "grid_c": grid_c.detach().cpu().numpy(),
        "c_true_cdf": true_c_cdf_grid.detach().cpu().numpy(),
        "c_gen_cdf": gen_c_cdf_grid.detach().cpu().numpy(),
    }
    return out


# ============================================================================
# MC harness. Same scenarios + seed schedule as base.main, output suffix _flow.
# ============================================================================
def main():
    # expose CLI args for n_train, n_eval, mc_runs, mc_runs_extra,
    # max_epochs, suffix. Default values reproduce the original sanity config.
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=384)
    p.add_argument("--n_eval", type=int, default=128)
    p.add_argument("--mc_runs", type=int, default=30,
                   help="MC runs for the base scenario (lognormal_base)")
    p.add_argument("--mc_runs_extra", type=int, default=10,
                   help="MC runs per extra scenario (heavy_tail_t, cox_t_lognormal_c, cox_both)")
    p.add_argument("--max_epochs", type=int, default=220)
    p.add_argument("--patience", type=int, default=35)
    p.add_argument("--K", type=int, default=4,
                   help="number of tanh components (0 = LogN ablation)")
    p.add_argument("--lambda_alpha", type=float, default=0.01,
                   help="weight on Sum_k alpha_k^2 gauge fix (Appendix B (A3))")
    p.add_argument("--suffix", default="flow",
                   help="Output filename suffix mc_<scenario>_*_<suffix>")
    args = p.parse_args()

    global LAMBDA_ALPHA, N_COMPONENTS
    LAMBDA_ALPHA = args.lambda_alpha
    N_COMPONENTS = args.K
    SUFFIX = args.suffix
    cfg = ExperimentConfig()
    cfg.n_train = args.n_train
    cfg.n_eval = args.n_eval
    cfg.mc_runs = args.mc_runs
    cfg.mc_runs_extra = args.mc_runs_extra
    cfg.max_epochs = args.max_epochs
    cfg.patience = args.patience
    scenario_runs: List[Tuple[str, int]] = [(cfg.scenario_name, cfg.mc_runs)]
    if cfg.run_extra_scenarios:
        for sc in cfg.extra_scenarios:
            scenario_runs.append((sc, cfg.mc_runs_extra))

    all_payload = {"scenarios": {}}
    report_keys = [
        "T_W1_samples", "T_L1_intCDF", "T_Linf_KS",
        "C_W1_samples", "C_L1_intCDF", "C_Linf_KS",
        "T_compKM_Linf_KS", "T_compCox_Linf_KS",
        "T_compAFTLN_Linf_KS", "T_compAFTW_Linf_KS",
        "converged_epoch", "train_time_sec", "censoring_rate",
    ]

    for scenario_name, run_count in scenario_runs:
        cfg_s = ExperimentConfig(**asdict(cfg))
        cfg_s.scenario_name = scenario_name
        cfg_s.mc_runs = run_count

        all_results: List[Dict[str, float]] = []
        all_plot_payloads: List[Dict[str, object]] = []

        for run_id in range(cfg_s.mc_runs):
            seed = 20260407 + run_id * 13
            res = train_one_experiment(cfg_s, seed)

            payload = res.pop("_plot_payload")
            payload["seed"] = int(seed)
            payload["w1_score"] = float(res["T_W1_samples"] + res["C_W1_samples"])
            all_plot_payloads.append(payload)

            all_results.append(res)
            print(
                f"[{scenario_name}] Run {run_id + 1:02d}/{cfg_s.mc_runs} | "
                f"T_W1={res['T_W1_samples']:.4f}, C_W1={res['C_W1_samples']:.4f}, "
                f"epoch={int(res['converged_epoch'])}, censor={res['censoring_rate']:.3f}, "
                f"time={res['train_time_sec']:.2f}s",
                flush=True,
            )

        summary = summarize_results(all_results)
        result_file = f"mc_{scenario_name}_results_{SUFFIX}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(
                {"config": asdict(cfg_s), "results": all_results, "summary": summary},
                f, ensure_ascii=False, indent=2,
            )

        seed_plot_file = f"mc_{scenario_name}_selected3_seed_TC_{SUFFIX}.png"
        summary_plot_file = f"mc_{scenario_name}_summary_plots_{SUFFIX}.png"

        all_plot_payloads.sort(key=lambda x: float(x["w1_score"]), reverse=True)
        selected_seed_payloads = all_plot_payloads[:3]
        make_selected_seed_comparison_plot(selected_seed_payloads, seed_plot_file)
        make_summary_plot(all_results, summary_plot_file)

        all_payload["scenarios"][scenario_name] = {
            "config": asdict(cfg_s),
            "summary": summary,
            "worst_w1_seeds": [int(p["seed"]) for p in selected_seed_payloads],
            "result_file": result_file,
            "seed_plot": seed_plot_file,
            "summary_plot": summary_plot_file,
        }

    print("\n=== Scenario Summary (mean ± std) ===")
    for scenario_name in all_payload["scenarios"]:
        print(f"\n[{scenario_name}]")
        summary = all_payload["scenarios"][scenario_name]["summary"]
        for k in report_keys:
            m = summary[k]["mean"]
            s = summary[k]["std"]
            print(f"{k:18s}: {m:.6f} ± {s:.6f}")

    all_file = f"mc_all_scenarios_index_{SUFFIX}.json"
    with open(all_file, "w", encoding="utf-8") as f:
        json.dump(all_payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved scenario index to: {all_file}")


if __name__ == "__main__":
    main()
