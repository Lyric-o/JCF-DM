# -*- coding: utf-8 -*-
"""
release modification note:
- Added a full Python implementation for Section 5.1 (Uninformative Censoring).
- Implemented conditional flow-like models for T|Z and C|Z with closed-form density/survival.
- Implemented Soft-NA + Wasserstein regularization and MLE joint training.
- Added 30-run Monte Carlo experiment with convergence (early stopping) and multi-norm evaluation.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# release modification note:
# Device is set to CPU by default for reproducibility in generic environments.
DEVICE = torch.device("cpu")
DTYPE = torch.float32


@dataclass
class ExperimentConfig:
    n_train: int = 384
    n_eval: int = 128
    z_mean: float = 0.0
    z_std: float = 1.0
    s_sim: int = 3
    lambda_w: float = 0.2
    tau_temp: float = 0.25
    soft_alpha: float = 0.20
    grid_size: int = 80
    lr: float = 1e-3
    max_epochs: int = 220
    patience: int = 35
    min_delta: float = 1e-4
    hidden_dim: int = 32
    mc_runs: int = 30
    time_min: float = 1e-4
    time_max: float = 10.0
    scenario_name: str = "lognormal_base"
    run_extra_scenarios: bool = True
    extra_scenarios: Tuple[str, ...] = ("heavy_tail_t", "cox_t_lognormal_c", "cox_both")
    mc_runs_extra: int = 10


# release modification note:
# True data-generating mechanism for uninformative censoring: T ⟂ C | Z.
def true_params_t(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu_t = 0.55 * z + 0.15 * torch.sin(1.2 * z)
    sigma_t = 0.35 + 0.05 * torch.sigmoid(z)
    return mu_t, sigma_t


def true_params_c(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu_c = -0.30 * z + 0.10 * (z ** 2) - 0.05
    sigma_c = 0.40 + 0.04 * torch.sigmoid(-z)
    return mu_c, sigma_c


def enforce_time_bounds(x: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
    return torch.clamp(x, min=cfg.time_min, max=cfg.time_max)


def sample_t_given_z_scenario(z: torch.Tensor, scenario: str, generator: torch.Generator) -> torch.Tensor:
    if scenario == "lognormal_base":
        mu_t, sigma_t = true_params_t(z)
        eps_t = torch.randn(z.shape, generator=generator, device=DEVICE, dtype=DTYPE)
        return torch.exp(mu_t + sigma_t * eps_t)
    if scenario == "cox_t_lognormal_c":
        beta_t = 0.70
        lambda0_t = 0.22
        u = torch.rand(z.shape, generator=generator, device=DEVICE, dtype=DTYPE)
        return -torch.log(torch.clamp(u, min=1e-8)) / (lambda0_t * torch.exp(beta_t * z))
    if scenario == "heavy_tail_t":
        mu_t, sigma_t = true_params_t(z)
        eps_t = torch.distributions.StudentT(df=torch.tensor(3.0, device=DEVICE, dtype=DTYPE)).sample(z.shape)
        eps_t = eps_t.to(device=DEVICE, dtype=DTYPE)
        return torch.exp(mu_t + sigma_t * eps_t)
    if scenario == "cox_both":
        beta_t = 0.75
        lambda0_t = 0.20
        u = torch.rand(z.shape, generator=generator, device=DEVICE, dtype=DTYPE)
        return -torch.log(torch.clamp(u, min=1e-8)) / (lambda0_t * torch.exp(beta_t * z))
    raise ValueError(f"Unknown scenario for T: {scenario}")


def sample_c_given_z_scenario(z: torch.Tensor, scenario: str, generator: torch.Generator) -> torch.Tensor:
    if scenario in {"lognormal_base", "heavy_tail_t", "cox_t_lognormal_c"}:
        mu_c, sigma_c = true_params_c(z)
        eps_c = torch.randn(z.shape, generator=generator, device=DEVICE, dtype=DTYPE)
        return torch.exp(mu_c + sigma_c * eps_c)
    if scenario == "cox_both":
        beta_c = -0.50
        lambda0_c = 0.24
        u = torch.rand(z.shape, generator=generator, device=DEVICE, dtype=DTYPE)
        return -torch.log(torch.clamp(u, min=1e-8)) / (lambda0_c * torch.exp(beta_c * z))
    raise ValueError(f"Unknown scenario for C: {scenario}")


def generate_dataset(n: int, cfg: ExperimentConfig, seed: int) -> Dict[str, torch.Tensor]:
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)

    z = torch.normal(cfg.z_mean, cfg.z_std, size=(n, 1), generator=g, device=DEVICE, dtype=DTYPE)

    t = enforce_time_bounds(sample_t_given_z_scenario(z, cfg.scenario_name, generator=g), cfg)
    c = enforce_time_bounds(sample_c_given_z_scenario(z, cfg.scenario_name, generator=g), cfg)

    x = torch.minimum(t, c)
    delta = (t <= c).to(DTYPE)

    return {
        "z": z,
        "t": t,
        "c": c,
        "x": x,
        "delta": delta,
    }


class CondLogNormalFlow(nn.Module):
    """
    release modification note:
    A conditional monotone transform equivalent to a simple 1D conditional flow:
    log(Y) = mu(z) + sigma(z) * eps, eps ~ N(0,1).
    This yields closed-form f(y|z), S(y|z), inverse sampling, and stable training.
    """

    def __init__(self, in_dim: int = 1, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_sigma_head = nn.Linear(hidden_dim, 1)

    def params(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(z)
        mu = self.mu_head(h)
        # keep sigma strictly positive and well-conditioned
        sigma = F.softplus(self.log_sigma_head(h)) + 0.05
        return mu, sigma

    def log_pdf(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        y_safe = torch.clamp(y, min=eps)
        mu, sigma = self.params(z)
        logy = torch.log(y_safe)
        u = (logy - mu) / sigma
        log_norm = -0.5 * math.log(2.0 * math.pi) - torch.log(sigma)
        log_pdf_logy = log_norm - 0.5 * (u ** 2)
        # Jacobian of log transform: d log(y) / dy = 1/y
        return log_pdf_logy - logy

    def log_survival(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        y_safe = torch.clamp(y, min=1e-8)
        mu, sigma = self.params(z)
        u = (torch.log(y_safe) - mu) / sigma
        # numerically stable: log(1 - Phi(u)) = log(Phi(-u))
        surv = torch.special.ndtr(-u)
        return torch.log(torch.clamp(surv, min=eps))

    def sample(self, z: torch.Tensor, n_samples: int = 1, generator: torch.Generator | None = None) -> torch.Tensor:
        mu, sigma = self.params(z)
        if n_samples == 1:
            eps = torch.randn_like(mu) if generator is None else torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
            return torch.exp(mu + sigma * eps)
        eps = torch.randn((n_samples, z.shape[0], 1), generator=generator, device=z.device, dtype=z.dtype)
        mu_e = mu.unsqueeze(0)
        sigma_e = sigma.unsqueeze(0)
        return torch.exp(mu_e + sigma_e * eps)


# release modification note:
# Hard NA on grid from observed (x, delta). Used as fixed target curve.
def hard_na_survival_on_grid(x: torch.Tensor, delta: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    x_np = x.detach().cpu().numpy().reshape(-1)
    d_np = delta.detach().cpu().numpy().reshape(-1)
    g_np = grid.detach().cpu().numpy().reshape(-1)

    hazard_increments = np.zeros_like(g_np, dtype=np.float64)
    for i, tau in enumerate(g_np):
        at_risk = np.sum(x_np >= tau)
        if at_risk <= 0:
            hazard_increments[i] = 0.0
            continue

        if i < len(g_np) - 1:
            in_bin = (x_np >= g_np[i]) & (x_np < g_np[i + 1])
        else:
            in_bin = x_np >= g_np[i]
        events = np.sum(d_np[in_bin])
        hazard_increments[i] = events / max(at_risk, 1.0)

    cum_hazard = np.cumsum(hazard_increments)
    s = np.exp(-cum_hazard)
    return torch.tensor(s, dtype=DTYPE, device=DEVICE).view(1, -1)


# release modification note:
# Differentiable Soft-NA for generated samples (x_soft, delta_soft).
def soft_na_survival_on_grid(
    x_soft: torch.Tensor,
    delta_soft: torch.Tensor,
    grid: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    # x_soft, delta_soft: [n, 1], grid: [m]
    x = x_soft.view(-1, 1)
    d = delta_soft.view(-1, 1)
    g = grid.view(1, -1)

    risk = torch.sigmoid((x - g) / alpha).sum(dim=0)  # [m]

    g_next = torch.cat([grid[1:], grid[-1:] + (grid[-1] - grid[-2])])
    g0 = grid.view(1, -1)
    g1 = g_next.view(1, -1)

    # soft bin indicator for events in [g_m, g_{m+1})
    in_bin = torch.sigmoid((x - g0) / alpha) - torch.sigmoid((x - g1) / alpha)
    dN = (d * in_bin).sum(dim=0)  # [m]

    hazard_inc = dN / (risk + 1e-6)
    cum_hazard = torch.cumsum(hazard_inc, dim=0)
    s = torch.exp(-cum_hazard)
    return s.view(1, -1)


def wasserstein_curve_distance(s_obs: torch.Tensor, s_sim: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    dt = grid[1:] - grid[:-1]
    diff = torch.abs(s_obs[:, :-1] - s_sim[:, :-1])
    return torch.sum(diff * dt.view(1, -1), dim=1).mean()


def build_grid_from_data(x: torch.Tensor, grid_size: int) -> torch.Tensor:
    xmin = float(torch.quantile(x, 0.01).item())
    xmax = float(torch.quantile(x, 0.99).item())
    xmin = max(1e-4, xmin)
    xmax = max(xmin + 1e-3, xmax)
    return torch.linspace(xmin, xmax, grid_size, device=DEVICE, dtype=DTYPE)


def mle_loss(flow_t: CondLogNormalFlow, flow_c: CondLogNormalFlow, x: torch.Tensor, delta: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    log_f_t = flow_t.log_pdf(x, z)
    log_s_t = flow_t.log_survival(x, z)
    log_f_c = flow_c.log_pdf(x, z)
    log_s_c = flow_c.log_survival(x, z)
    ell = delta * (log_f_t + log_s_c) + (1.0 - delta) * (log_s_t + log_f_c)
    return -ell.mean()


@torch.no_grad()
def sample_true_given_z(
    z: torch.Tensor,
    which: str,
    cfg: ExperimentConfig,
    scenario_name: str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if generator is None:
        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(12345)
    if which == "T":
        return enforce_time_bounds(sample_t_given_z_scenario(z, scenario_name, generator=generator), cfg)
    return enforce_time_bounds(sample_c_given_z_scenario(z, scenario_name, generator=generator), cfg)


@torch.no_grad()
def cdf_norm_metrics_from_samples(true_samples: torch.Tensor, model_samples: torch.Tensor) -> Dict[str, float]:
    t = true_samples.view(-1)
    m = model_samples.view(-1)

    # common grid from pooled quantiles
    pooled = torch.cat([t, m], dim=0)
    gmin = float(torch.quantile(pooled, 0.01).item())
    gmax = float(torch.quantile(pooled, 0.99).item())
    grid = torch.linspace(max(1e-5, gmin), max(gmin + 1e-4, gmax), 300, device=t.device, dtype=t.dtype)

    t_sorted, _ = torch.sort(t)
    m_sorted, _ = torch.sort(m)

    # empirical CDF on grid using searchsorted
    t_idx = torch.searchsorted(t_sorted, grid, right=True).to(DTYPE)
    m_idx = torch.searchsorted(m_sorted, grid, right=True).to(DTYPE)
    Ft = t_idx / t_sorted.numel()
    Fm = m_idx / m_sorted.numel()

    diff = Ft - Fm
    dt = grid[1:] - grid[:-1]

    l1_int = torch.sum(torch.abs(diff[:-1]) * dt)
    l2_int = torch.sqrt(torch.sum((diff[:-1] ** 2) * dt))
    linf = torch.max(torch.abs(diff))

    # On the common evaluation window, the one-dimensional W1 distance is
    # the absolute area between the empirical CDFs.  This remains valid when
    # the true and model sample sizes differ (e.g. S_eval > 1).
    w1 = l1_int

    return {
        "L1_intCDF": float(l1_int.item()),
        "L2_intCDF": float(l2_int.item()),
        "Linf_KS": float(linf.item()),
        "W1_samples": float(w1.item()),
    }


@torch.no_grad()
def cdf_from_samples_on_grid(samples: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    s = torch.sort(samples.view(-1))[0]
    idx = torch.searchsorted(s, grid, right=True).to(DTYPE)
    return idx / s.numel()


def curve_distance_metrics(true_cdf: torch.Tensor, est_cdf: torch.Tensor, grid: torch.Tensor) -> Dict[str, float]:
    diff = true_cdf - est_cdf
    dt = grid[1:] - grid[:-1]
    l1 = torch.sum(torch.abs(diff[:-1]) * dt)
    l2 = torch.sqrt(torch.sum((diff[:-1] ** 2) * dt))
    ks = torch.max(torch.abs(diff))
    return {
        "L1_intCDF": float(l1.item()),
        "L2_intCDF": float(l2.item()),
        "Linf_KS": float(ks.item()),
    }


def km_estimator(x: torch.Tensor, delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x1 = x.view(-1)
    d1 = delta.view(-1)
    event_times = torch.unique(x1[d1 > 0.5])
    event_times, _ = torch.sort(event_times)
    if event_times.numel() == 0:
        return torch.tensor([0.0], dtype=DTYPE, device=DEVICE), torch.tensor([1.0], dtype=DTYPE, device=DEVICE)

    surv_vals = []
    s_cur = torch.tensor(1.0, dtype=DTYPE, device=DEVICE)
    for t in event_times:
        at_risk = torch.sum(x1 >= t).to(DTYPE)
        d_t = torch.sum((x1 == t).to(DTYPE) * d1)
        s_cur = s_cur * (1.0 - d_t / torch.clamp(at_risk, min=1.0))
        surv_vals.append(s_cur)
    return event_times, torch.stack(surv_vals)


def km_cdf_on_grid(event_times: torch.Tensor, surv_vals: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    idx = torch.searchsorted(event_times, grid, right=True) - 1
    idx = torch.clamp(idx, min=-1, max=surv_vals.numel() - 1)
    s = torch.ones_like(grid)
    valid = idx >= 0
    s[valid] = surv_vals[idx[valid]]
    return 1.0 - s


def fit_cox_beta_1d(z: torch.Tensor, x: torch.Tensor, delta: torch.Tensor, max_iter: int = 220, lr: float = 0.05) -> torch.Tensor:
    z1 = z.view(-1)
    x1 = x.view(-1)
    d1 = delta.view(-1)

    order = torch.argsort(x1, descending=True)
    z_sorted = z1[order]
    d_sorted = d1[order]

    beta = torch.tensor([0.0], dtype=DTYPE, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([beta], lr=lr)

    for _ in range(max_iter):
        opt.zero_grad()
        eta = beta * z_sorted
        exp_eta = torch.exp(torch.clamp(eta, min=-25.0, max=25.0))
        risk_cumsum = torch.cumsum(exp_eta, dim=0)
        pll = torch.sum(d_sorted * (eta - torch.log(torch.clamp(risk_cumsum, min=1e-8))))
        loss = -pll / torch.clamp(torch.sum(d_sorted), min=1.0)
        loss.backward()
        opt.step()
    return beta.detach()


def breslow_baseline_from_beta(
    beta: torch.Tensor,
    z: torch.Tensor,
    x: torch.Tensor,
    delta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    z1 = z.view(-1)
    x1 = x.view(-1)
    d1 = delta.view(-1)

    event_times = torch.unique(x1[d1 > 0.5])
    event_times, _ = torch.sort(event_times)
    if event_times.numel() == 0:
        return torch.tensor([0.0], dtype=DTYPE, device=DEVICE), torch.tensor([0.0], dtype=DTYPE, device=DEVICE)

    exp_eta = torch.exp(torch.clamp(beta * z1, min=-25.0, max=25.0))
    increments = []
    for t in event_times:
        d_t = torch.sum((x1 == t).to(DTYPE) * d1)
        denom = torch.sum(exp_eta[x1 >= t])
        increments.append(d_t / torch.clamp(denom, min=1e-8))
    inc = torch.stack(increments)
    h0 = torch.cumsum(inc, dim=0)
    return event_times, h0


def cox_marginal_cdf_on_grid(
    beta: torch.Tensor,
    base_times: torch.Tensor,
    base_h0: torch.Tensor,
    z_eval: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    idx = torch.searchsorted(base_times, grid, right=True) - 1
    idx = torch.clamp(idx, min=-1, max=base_h0.numel() - 1)
    h0_grid = torch.zeros_like(grid)
    valid = idx >= 0
    h0_grid[valid] = base_h0[idx[valid]]
    lp = torch.exp(torch.clamp(beta * z_eval.view(-1), min=-25.0, max=25.0)).view(-1, 1)
    s = torch.exp(-lp * h0_grid.view(1, -1))
    return 1.0 - s.mean(dim=0)


@torch.no_grad()
def quantile_abs_errors_from_samples(
    true_samples: torch.Tensor,
    model_samples: torch.Tensor,
    quantiles: List[float],
) -> Dict[str, float]:
    out = {}
    for q in quantiles:
        q_true = torch.quantile(true_samples.view(-1), q)
        q_model = torch.quantile(model_samples.view(-1), q)
        out[f"QAE_q{int(q * 100):02d}"] = float(torch.abs(q_true - q_model).item())
    return out


def truncate_grid_by_cdf(curves: List[np.ndarray], xgrid: np.ndarray, threshold: float = 0.999) -> Tuple[np.ndarray, List[np.ndarray]]:
    mask = np.zeros_like(xgrid, dtype=bool)
    for c in curves:
        mask = mask | (c < threshold)
    if np.any(mask):
        last = int(np.max(np.where(mask)))
        keep = slice(0, min(last + 2, xgrid.shape[0]))
    else:
        keep = slice(0, xgrid.shape[0])
    return xgrid[keep], [c[keep] for c in curves]


def make_selected_seed_comparison_plot(seed_payloads: List[Dict[str, object]], out_file: str) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    axes = axes.reshape(-1)

    for idx, payload in enumerate(seed_payloads[:3]):
        ax_t = axes[2 * idx]
        ax_c = axes[2 * idx + 1]
        seed = payload["seed"]

        t_x = payload["grid_t"]  # type: ignore[assignment]
        t_true = payload["t_true_cdf"]  # type: ignore[assignment]
        t_gen = payload["t_gen_cdf"]  # type: ignore[assignment]
        t_km = payload["t_km_cdf"]  # type: ignore[assignment]
        t_cox = payload["t_cox_cdf"]  # type: ignore[assignment]
        t_x2, t_curves = truncate_grid_by_cdf([t_true, t_gen, t_km, t_cox], t_x)
        t_true2, t_gen2, t_km2, t_cox2 = t_curves

        c_x = payload["grid_c"]  # type: ignore[assignment]
        c_true = payload["c_true_cdf"]  # type: ignore[assignment]
        c_gen = payload["c_gen_cdf"]  # type: ignore[assignment]
        c_x2, c_curves = truncate_grid_by_cdf([c_true, c_gen], c_x)
        c_true2, c_gen2 = c_curves

        ax_t.plot(t_x2, t_true2, color="#1f77b4", linewidth=1.8, label="True T (DGP)")
        ax_t.plot(t_x2, t_gen2, color="#2ca02c", linewidth=1.8, linestyle="--", label="Generative model g1")
        ax_t.plot(t_x2, t_km2, color="#ff7f0e", linewidth=1.6, linestyle=":", label="Kaplan-Meier")
        ax_t.plot(t_x2, t_cox2, color="#9467bd", linewidth=1.6, linestyle="-.", label="Cox PH + Breslow")
        ax_t.set_title(f"Seed {seed} | T distribution")
        ax_t.set_xlabel("time")
        ax_t.set_ylabel("CDF")
        ax_t.set_ylim(0.0, 1.0)
        ax_t.grid(alpha=0.25)
        ax_t.legend(loc="lower right", frameon=True, fontsize=8)

        ax_c.plot(c_x2, c_true2, color="#d62728", linewidth=1.8, label="True C (DGP)")
        ax_c.plot(c_x2, c_gen2, color="#17becf", linewidth=1.8, linestyle="--", label="Generative model g2")
        ax_c.set_title(f"Seed {seed} | C distribution")
        ax_c.set_xlabel("time")
        ax_c.set_ylabel("CDF")
        ax_c.set_ylim(0.0, 1.0)
        ax_c.grid(alpha=0.25)
        ax_c.legend(loc="lower right", frameon=True, fontsize=8)

    for j in range(2 * len(seed_payloads), 6):
        axes[j].axis("off")

    fig.suptitle("Three-seed CDF comparison (worst combined W1; T/C separated)", y=1.02)
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_summary_plot(results: List[Dict[str, float]], out_file: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # Panel A: W1 and KS distribution across runs
    metric_groups = [
        [r["T_W1_samples"] for r in results],
        [r["C_W1_samples"] for r in results],
        [r["T_Linf_KS"] for r in results],
        [r["C_Linf_KS"] for r in results],
        [r["T_compKM_Linf_KS"] for r in results],
        [r["T_compCox_Linf_KS"] for r in results],
    ]
    axes[0, 0].boxplot(metric_groups, labels=["T_W1", "C_W1", "T_KS", "C_KS", "T_KS_KM", "T_KS_Cox"], showmeans=True)
    axes[0, 0].set_title("Distribution distance across runs")
    axes[0, 0].set_ylabel("value")
    axes[0, 0].grid(axis="y", alpha=0.25)

    # Panel B: convergence and time
    run_ids = np.arange(1, len(results) + 1)
    conv_epochs = np.array([r["converged_epoch"] for r in results], dtype=np.float64)
    train_times = np.array([r["train_time_sec"] for r in results], dtype=np.float64)
    ax_b = axes[0, 1]
    ax_b.plot(run_ids, conv_epochs, marker="o", linewidth=1.2, color="#2ca02c", label="converged epoch")
    ax_b.set_xlabel("run id")
    ax_b.set_ylabel("epoch", color="#2ca02c")
    ax_b.tick_params(axis="y", labelcolor="#2ca02c")
    ax_b.grid(alpha=0.25)
    ax_b2 = ax_b.twinx()
    ax_b2.plot(run_ids, train_times, marker="s", linewidth=1.2, color="#9467bd", label="train time (sec)")
    ax_b2.set_ylabel("sec", color="#9467bd")
    ax_b2.tick_params(axis="y", labelcolor="#9467bd")
    ax_b.set_title("Convergence Epoch and Training Time")

    # Panel C: quantile absolute errors
    q_labels = ["q10", "q50", "q90"]
    t_q = [np.mean([r[f"T_QAE_{q}"] for r in results]) for q in ["q10", "q50", "q90"]]
    c_q = [np.mean([r[f"C_QAE_{q}"] for r in results]) for q in ["q10", "q50", "q90"]]
    x = np.arange(len(q_labels))
    w = 0.35
    axes[1, 0].bar(x - w / 2, t_q, width=w, color="#1f77b4", label="T")
    axes[1, 0].bar(x + w / 2, c_q, width=w, color="#d62728", label="C")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(q_labels)
    axes[1, 0].set_ylabel("absolute error")
    axes[1, 0].set_title("Mean Quantile Absolute Error")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="y", alpha=0.25)

    # Panel D: censoring rate
    censor_rates = [r["censoring_rate"] for r in results]
    axes[1, 1].hist(censor_rates, bins=10, color="#ff7f0e", alpha=0.8, edgecolor="white")
    axes[1, 1].set_title("Censoring Rate Distribution")
    axes[1, 1].set_xlabel("rate")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].grid(axis="y", alpha=0.25)

    fig.suptitle("MC(30) Stability and Fit Summary", y=1.02)
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


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

    flow_t = CondLogNormalFlow(in_dim=1, hidden_dim=cfg.hidden_dim).to(DEVICE)
    flow_c = CondLogNormalFlow(in_dim=1, hidden_dim=cfg.hidden_dim).to(DEVICE)

    opt = torch.optim.Adam(list(flow_t.parameters()) + list(flow_c.parameters()), lr=cfg.lr)

    best_loss = float("inf")
    best_state_t = None
    best_state_c = None
    wait = 0

    train_start = time.perf_counter()
    best_epoch = 0
    trained_epochs = 0

    for _epoch in range(cfg.max_epochs):
        trained_epochs = _epoch + 1
        opt.zero_grad()

        loss_mle = mle_loss(flow_t, flow_c, x, delta, z)

        reg = 0.0
        for _ in range(cfg.s_sim):
            t_sim = enforce_time_bounds(flow_t.sample(z), cfg)
            c_sim = enforce_time_bounds(flow_c.sample(z), cfg)
            x_sim = torch.minimum(t_sim, c_sim)
            d_sim_soft = torch.sigmoid((c_sim - t_sim) / cfg.tau_temp)
            s_sim = soft_na_survival_on_grid(x_sim, d_sim_soft, grid, cfg.soft_alpha)
            reg = reg + wasserstein_curve_distance(s_obs, s_sim, grid)
        reg = reg / cfg.s_sim

        loss = loss_mle + cfg.lambda_w * reg
        loss.backward()
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

    # Evaluation after convergence
    g_eval = torch.Generator(device=DEVICE)
    g_eval.manual_seed(seed + 98765)
    z_eval = torch.normal(cfg.z_mean, cfg.z_std, size=(cfg.n_eval, 1), generator=g_eval, device=DEVICE, dtype=DTYPE)

    t_true = sample_true_given_z(z_eval, "T", cfg, cfg.scenario_name, generator=g_eval)
    c_true = sample_true_given_z(z_eval, "C", cfg, cfg.scenario_name, generator=g_eval)

    with torch.no_grad():
        t_model = enforce_time_bounds(flow_t.sample(z_eval, generator=g_eval), cfg)
        c_model = enforce_time_bounds(flow_c.sample(z_eval, generator=g_eval), cfg)

    t_metrics = cdf_norm_metrics_from_samples(t_true, t_model)
    c_metrics = cdf_norm_metrics_from_samples(c_true, c_model)
    t_qae = quantile_abs_errors_from_samples(t_true, t_model, [0.1, 0.5, 0.9])
    c_qae = quantile_abs_errors_from_samples(c_true, c_model, [0.1, 0.5, 0.9])

    # Competing methods for T based on censored observations
    grid_t = torch.linspace(
        float(torch.quantile(torch.cat([t_true.view(-1), t_model.view(-1)]), 0.01).item()),
        float(torch.quantile(torch.cat([t_true.view(-1), t_model.view(-1)]), 0.995).item()),
        280,
        device=DEVICE,
        dtype=DTYPE,
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

    grid_c = torch.linspace(
        float(torch.quantile(torch.cat([c_true.view(-1), c_model.view(-1)]), 0.01).item()),
        float(torch.quantile(torch.cat([c_true.view(-1), c_model.view(-1)]), 0.995).item()),
        280,
        device=DEVICE,
        dtype=DTYPE,
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
    out.update({f"T_{k}": v for k, v in t_qae.items()})
    out.update({f"C_{k}": v for k, v in c_qae.items()})
    out.update({f"T_compKM_{k}": v for k, v in t_km_metrics.items()})
    out.update({f"T_compCox_{k}": v for k, v in t_cox_metrics.items()})

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


def summarize_results(results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = [k for k in results[0].keys() if k not in {"seed"}]
    arr = {k: np.array([r[k] for r in results], dtype=np.float64) for k in keys}
    summary = {}
    for k, v in arr.items():
        summary[k] = {
            "mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "median": float(np.median(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
        }
    return summary


def main() -> None:
    cfg = ExperimentConfig()
    scenario_runs: List[Tuple[str, int]] = [(cfg.scenario_name, cfg.mc_runs)]
    if cfg.run_extra_scenarios:
        for sc in cfg.extra_scenarios:
            scenario_runs.append((sc, cfg.mc_runs_extra))

    all_payload = {"scenarios": {}}
    report_keys = [
        "T_W1_samples",
        "T_L1_intCDF",
        "T_L2_intCDF",
        "T_Linf_KS",
        "T_QAE_q10",
        "T_QAE_q50",
        "T_QAE_q90",
        "C_W1_samples",
        "C_L1_intCDF",
        "C_L2_intCDF",
        "C_Linf_KS",
        "C_QAE_q10",
        "C_QAE_q50",
        "C_QAE_q90",
        "T_compKM_Linf_KS",
        "T_compCox_Linf_KS",
        "converged_epoch",
        "train_time_sec",
        "censoring_rate",
    ]

    for scenario_name, run_count in scenario_runs:
        cfg_s = ExperimentConfig(**asdict(cfg))
        cfg_s.scenario_name = scenario_name
        cfg_s.mc_runs = run_count

        all_results = []
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
                f"time={res['train_time_sec']:.2f}s"
            )

        summary = summarize_results(all_results)
        result_file = f"mc_{scenario_name}_results_release.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump({"config": asdict(cfg_s), "results": all_results, "summary": summary}, f, ensure_ascii=False, indent=2)

        seed_plot_file = f"mc_{scenario_name}_selected3_seed_TC_release.png"
        summary_plot_file = f"mc_{scenario_name}_summary_plots_release.png"

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

    all_file = "mc_all_scenarios_index_release.json"
    with open(all_file, "w", encoding="utf-8") as f:
        json.dump(all_payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved scenario index to: {all_file}")


if __name__ == "__main__":
    main()
