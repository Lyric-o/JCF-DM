# -*- coding: utf-8 -*-
"""Semi-synthetic protocol used for Table 3.

The SUPPORT and combined Rotterdam--GBSG covariates are retained, while
event and censoring times are replaced by shared-frailty draws.  This makes
the marginal failure-time distribution available for evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
# numpy 2.x compat shim
if not hasattr(np.linalg, 'linalg'):
    np.linalg.linalg = np.linalg

import pandas as pd
import torch

DEVICE = torch.device("cpu")
DTYPE = torch.float32

_DATA_DIR = os.environ.get(
    "JCFDM_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data")
)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_support():
    """SUPPORT benchmark: 8,873 subjects and 14 covariates."""
    df = pd.read_csv(os.path.join(_DATA_DIR, "support.csv"))
    covariates = df.drop(columns=["duration", "event"]).values.astype(np.float32)
    times = df["duration"].values.astype(np.float32)
    events = df["event"].values.astype(np.float32)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    covariates = scaler.fit_transform(covariates)
    return covariates, times, events, "SUPPORT"


def load_gbsg():
    """Combined Rotterdam and GBSG benchmark: 2,232 subjects, 7 covariates."""
    df = pd.read_csv(os.path.join(_DATA_DIR, "gbsg.csv"))
    covariates = df.drop(columns=["duration", "event"]).values.astype(np.float32)
    times = df["duration"].values.astype(np.float32)
    events = df["event"].values.astype(np.float32)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    covariates = scaler.fit_transform(covariates)
    return covariates, times, events, "Rotterdam and GBSG"


LOADERS = {
    "support": load_support,
    "gbsg": load_gbsg,
}


# ── Semi-synthetic DGP ──────────────────────────────────────────────────────

@dataclass
class SemiSynConfig:
    dataset: str = "support"
    n_train: int = 2048
    n_eval: int = 1024
    frailty_sigma: float = 0.5
    shape_T: float = 1.5
    shape_C: float = 1.2
    rho: float = 1.0
    family: str = "spline"
    hidden: int = 64
    lr: float = 1e-3
    max_epochs: int = 800
    patience: int = 70
    min_delta: float = 1e-4
    mc_runs: int = 10
    n_knots: int = 6
    activation: str = "tanh"
    K_eps: int = 32
    time_min: float = 1e-4
    time_max: float = 500.0


def make_betas(d, seed):
    """Generate fixed projection vectors for a given MC run."""
    rng = np.random.default_rng(seed)
    beta_T = rng.standard_normal(d).astype(np.float32)
    beta_T /= np.linalg.norm(beta_T)
    beta_C = rng.standard_normal(d).astype(np.float32)
    beta_C /= np.linalg.norm(beta_C)
    return beta_T, beta_C


def semi_synthetic_generate(Z, cfg, gen_seed, beta_T, beta_C):
    """Given real covariates Z [n, d], generate (T, C) with shared frailty.

    Uses LogNormal-Weibull margins (same as DGP2) but with real Z.
    Beta_T, Beta_C are fixed projection vectors (shared across train/eval).
    gen_seed controls only the random frailty/uniform draws.
    """
    n, d = Z.shape
    g = torch.Generator(device=DEVICE); g.manual_seed(int(gen_seed))

    Z_t = torch.tensor(Z, dtype=DTYPE)
    risk_T = (Z_t @ torch.tensor(beta_T, dtype=DTYPE)).unsqueeze(1)
    risk_C = (Z_t @ torch.tensor(beta_C, dtype=DTYPE)).unsqueeze(1)

    eta = torch.randn((n, 1), generator=g, dtype=DTYPE)
    xi = torch.exp(cfg.frailty_sigma * eta)

    u1 = torch.rand((n, 1), generator=g, dtype=DTYPE)
    u2 = torch.rand((n, 1), generator=g, dtype=DTYPE)

    scale_T = torch.exp(0.4 * risk_T) * xi
    T = scale_T * (-torch.log(u1.clamp(min=1e-12))) ** (1.0 / cfg.shape_T)

    scale_C = torch.exp(-0.3 * risk_C) * (xi ** cfg.rho)
    C = scale_C * (-torch.log(u2.clamp(min=1e-12))) ** (1.0 / cfg.shape_C)

    T = T.clamp(cfg.time_min, cfg.time_max)
    C = C.clamp(cfg.time_min, cfg.time_max)
    x = torch.minimum(T, C)
    delta = (T <= C).to(DTYPE)

    return {"z": Z_t, "t": T, "c": C, "x": x, "delta": delta}


# ── Metrics ──────────────────────────────────────────────────────────────────

def cdf_norm_metrics(true_samples, model_samples):
    t = true_samples.view(-1).float()
    m = model_samples.view(-1).float()
    pooled = torch.cat([t, m])
    gmin = float(torch.quantile(pooled, 0.01))
    gmax = float(torch.quantile(pooled, 0.99))
    grid = torch.linspace(max(1e-5, gmin), max(gmin + 1e-4, gmax), 300)
    t_sorted = torch.sort(t)[0]
    m_sorted = torch.sort(m)[0]
    Ft = torch.searchsorted(t_sorted, grid, right=True).float() / t_sorted.numel()
    Fm = torch.searchsorted(m_sorted, grid, right=True).float() / m_sorted.numel()
    diff = Ft - Fm
    dt = grid[1:] - grid[:-1]
    linf = torch.max(torch.abs(diff))
    n_min = min(t_sorted.numel(), m_sorted.numel())
    w1 = torch.mean(torch.abs(t_sorted[:n_min] - m_sorted[:n_min]))
    return {"T_KS": float(linf), "T_W1": float(w1)}


# ── Inverse-CDF sampling helper ─────────────────────────────────────────────

def _inv_cdf_sample(surv_vals, times, rng, n_samples):
    """Sample times by inverting a discrete survival curve.
    surv_vals: 1-D array, non-increasing.  times: 1-D array of grid points."""
    samples = np.empty(n_samples)
    for j in range(n_samples):
        u = rng.random()
        idx = np.searchsorted(-surv_vals, -u)
        idx = min(idx, len(times) - 1)
        samples[j] = times[idx]
    return samples


# ── Method: Joint flow (our method) ─────────────────────────────────────────

def run_joint_flow(z_tr, x_tr, d_tr, z_ev, t_true_ev, cfg, seed):
    import survival_dists as _sd
    from eps_utils import make_eps
    _sd.set_activation(cfg.activation)
    t0 = time.perf_counter()

    ctx_dim = z_tr.shape[1] + 1
    flow_t = _sd.build_dist(cfg.family, ctx_dim=ctx_dim, hidden=cfg.hidden,
                            n_knots=cfg.n_knots).to(DEVICE)
    flow_c = _sd.build_dist(cfg.family, ctx_dim=ctx_dim, hidden=cfg.hidden,
                            n_knots=cfg.n_knots).to(DEVICE)

    params = list(flow_t.parameters()) + list(flow_c.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)

    best_loss = float("inf"); wait = 0; best_state_t = best_state_c = None
    K = cfg.K_eps
    batch_size = min(512, z_tr.shape[0])

    for epoch in range(cfg.max_epochs):
        perm = torch.randperm(z_tr.shape[0])
        epoch_loss = 0.0; n_seen = 0
        for b_start in range(0, z_tr.shape[0], batch_size):
            b_idx = perm[b_start:b_start + batch_size]
            z_b = z_tr[b_idx]; x_b = x_tr[b_idx]; d_b = d_tr[b_idx]
            nb = z_b.shape[0]

            opt.zero_grad()
            eps = torch.randn(K, nb, 1, dtype=DTYPE)
            z_rep = z_b.unsqueeze(0).expand(K, -1, -1)
            ctx = torch.cat([z_rep, eps], dim=-1).reshape(K * nb, -1)
            x_rep = x_b.unsqueeze(0).expand(K, -1, -1).reshape(K * nb, -1)
            d_rep = d_b.unsqueeze(0).expand(K, -1, -1).reshape(K * nb, -1)

            ll_T = flow_t.log_pdf(x_rep, ctx)
            ll_C = flow_c.log_pdf(x_rep, ctx)
            ls_T = flow_t.log_survival(x_rep, ctx)
            ls_C = flow_c.log_survival(x_rep, ctx)
            ll_per_sample = d_rep * (ll_T + ls_C) + (1 - d_rep) * (ls_T + ll_C)
            ll_per_sample = ll_per_sample.view(K, nb, -1)
            ll_lse = torch.logsumexp(ll_per_sample, dim=0) - np.log(K)
            nll = -torch.mean(ll_lse)

            nll.backward()
            torch.nn.utils.clip_grad_norm_(params, 2.0)
            opt.step()
            epoch_loss += float(nll.item()) * nb
            n_seen += nb

        cur = epoch_loss / max(n_seen, 1)
        if cur < best_loss - cfg.min_delta:
            best_loss = cur; wait = 0
            best_state_t = {k: v.cpu().clone() for k, v in flow_t.state_dict().items()}
            best_state_c = {k: v.cpu().clone() for k, v in flow_c.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    train_sec = time.perf_counter() - t0
    if best_state_t: flow_t.load_state_dict(best_state_t)
    if best_state_c: flow_c.load_state_dict(best_state_c)

    n_samp = 20
    with torch.no_grad():
        eps_eval = torch.randn(n_samp, z_ev.shape[0], 1, dtype=DTYPE)
        z_rep = z_ev.unsqueeze(0).expand(n_samp, -1, -1)
        ctx_eval = torch.cat([z_rep, eps_eval], dim=-1).reshape(n_samp * z_ev.shape[0], -1)
        model_T = flow_t.sample(ctx_eval).clamp(cfg.time_min, cfg.time_max)

    t_true_rep = t_true_ev.repeat(n_samp, 1)
    metrics = cdf_norm_metrics(t_true_rep, model_T)
    metrics["train_time_sec"] = train_sec
    metrics["method"] = "joint_flow"
    return metrics


# ── Method: T-only flow (AUSSET baseline) ───────────────────────────────────

def run_tonly_flow(z_tr, x_tr, d_tr, z_ev, t_true_ev, cfg, seed):
    import survival_dists as _sd
    _sd.set_activation(cfg.activation)
    t0 = time.perf_counter()

    flow_t = _sd.build_dist(cfg.family, ctx_dim=z_tr.shape[1], hidden=cfg.hidden,
                            n_knots=cfg.n_knots).to(DEVICE)
    opt = torch.optim.Adam(flow_t.parameters(), lr=cfg.lr)

    best_loss = float("inf"); wait = 0; best_state = None
    for epoch in range(cfg.max_epochs):
        opt.zero_grad()
        nll = -torch.mean(d_tr * flow_t.log_pdf(x_tr, z_tr)
                          + (1 - d_tr) * flow_t.log_survival(x_tr, z_tr))
        nll.backward()
        torch.nn.utils.clip_grad_norm_(flow_t.parameters(), 2.0)
        opt.step()
        cur = float(nll.item())
        if cur < best_loss - cfg.min_delta:
            best_loss = cur; wait = 0
            best_state = {k: v.cpu().clone() for k, v in flow_t.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    train_sec = time.perf_counter() - t0
    if best_state: flow_t.load_state_dict(best_state)

    n_samp = 20
    z_rep = z_ev.repeat(n_samp, 1)
    with torch.no_grad():
        model_T = flow_t.sample(z_rep).clamp(cfg.time_min, cfg.time_max)
    t_true_rep = t_true_ev.repeat(n_samp, 1)
    metrics = cdf_norm_metrics(t_true_rep, model_T)
    metrics["train_time_sec"] = train_sec
    metrics["method"] = "tonly_flow"
    return metrics


# ── Method: CoxPH baseline (pure torch + Breslow) ──────────────────────────

def run_coxph(z_tr, x_tr, d_tr, z_ev, t_true_ev, cfg, seed):
    t0 = time.perf_counter()

    # numpy arrays
    z_np = z_tr.numpy().astype(np.float64)          # (n, p)
    x_np = x_tr.view(-1).numpy().astype(np.float64)
    d_np = d_tr.view(-1).numpy().astype(np.float64)
    n, p = z_np.shape

    # ----- fit Cox partial likelihood with torch autograd -----
    beta = torch.zeros(p, dtype=torch.float64, requires_grad=True)
    z_t = torch.tensor(z_np, dtype=torch.float64)
    x_t = torch.tensor(x_np, dtype=torch.float64)
    d_t = torch.tensor(d_np, dtype=torch.float64)

    # sort by descending observed time for risk-set computation
    order = torch.argsort(x_t, descending=True)
    z_sorted = z_t[order]
    x_sorted = x_t[order]
    d_sorted = d_t[order]

    optimizer = torch.optim.LBFGS([beta], lr=1.0, max_iter=20,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        eta = z_sorted @ beta                       # (n,)
        # log-cumsum-exp over the risk set (sorted descending by time)
        cumsum_exp = torch.logcumsumexp(eta, dim=0)  # cumulative from top
        # negative partial log-likelihood (Breslow)
        npl = -torch.sum(d_sorted * (eta - cumsum_exp))
        # L2 penalty
        npl = npl + 0.01 * torch.sum(beta ** 2)
        npl.backward()
        return npl

    for _ in range(5):
        optimizer.step(closure)

    beta_hat = beta.detach().numpy()

    # ----- Breslow baseline cumulative hazard -----
    eta_all = z_np @ beta_hat                       # (n,)
    exp_eta = np.exp(eta_all)

    # unique event times
    event_mask = d_np > 0.5
    event_times_set = np.unique(x_np[event_mask])
    event_times_set.sort()

    # risk-set denominator at each event time
    baseline_haz = np.zeros(len(event_times_set))
    for k, t_k in enumerate(event_times_set):
        at_risk = x_np >= t_k
        denom = exp_eta[at_risk].sum()
        d_k = ((x_np == t_k) & (d_np > 0.5)).sum()
        baseline_haz[k] = d_k / max(denom, 1e-12)

    cum_baseline_haz = np.cumsum(baseline_haz)      # H_0(t) at event times
    times = event_times_set                          # grid

    train_sec = time.perf_counter() - t0

    # ----- generate model samples via inverse-CDF -----
    z_ev_np = z_ev.numpy().astype(np.float64)
    rng = np.random.default_rng(seed + 7777)
    n_samp = 20
    n_eval = z_ev_np.shape[0]
    model_T = []
    for i in range(n_eval):
        eta_i = z_ev_np[i] @ beta_hat
        # S(t|z) = exp(-H_0(t) * exp(eta_i))
        surv_vals = np.exp(-cum_baseline_haz * np.exp(eta_i))
        samps = _inv_cdf_sample(surv_vals, times, rng, n_samp)
        model_T.append(samps)

    model_T = torch.tensor(np.concatenate(model_T), dtype=DTYPE)
    t_true_rep = t_true_ev.repeat_interleave(n_samp)
    metrics = cdf_norm_metrics(t_true_rep, model_T)
    metrics["train_time_sec"] = train_sec
    metrics["method"] = "coxph"
    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────

METHODS = {
    "joint_flow": run_joint_flow,
    "tonly_flow": run_tonly_flow,
    "coxph": run_coxph,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="support", choices=list(LOADERS.keys()))
    p.add_argument("--methods", default="joint_flow,tonly_flow,coxph")
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--mc_runs", type=int, default=10)
    p.add_argument("--max_epochs", type=int, default=800)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--out", default="semisyn")
    args = p.parse_args()

    covariates, _, _, ds_name = LOADERS[args.dataset]()
    print(f"Loaded {ds_name}: {covariates.shape[0]} subjects, {covariates.shape[1]} covariates")

    cfg = SemiSynConfig(dataset=args.dataset, rho=args.rho, mc_runs=args.mc_runs,
                        max_epochs=args.max_epochs, hidden=args.hidden)

    n_total = covariates.shape[0]
    methods = [m.strip() for m in args.methods.split(",")]
    all_results = {}

    for method_name in methods:
        if method_name not in METHODS:
            print(f"Unknown method: {method_name}"); continue
        runner = METHODS[method_name]
        rows = []
        for i in range(cfg.mc_runs):
            seed = 20260427 + i * 17
            rng = np.random.default_rng(seed)
            idx = rng.permutation(n_total)
            n_tr = min(cfg.n_train, int(n_total * 0.75))
            n_ev = min(cfg.n_eval, n_total - n_tr)
            Z_tr = covariates[idx[:n_tr]]
            Z_ev = covariates[idx[n_tr:n_tr + n_ev]]

            beta_T, beta_C = make_betas(covariates.shape[1], seed)
            data_tr = semi_synthetic_generate(Z_tr, cfg, seed, beta_T, beta_C)
            data_ev = semi_synthetic_generate(Z_ev, cfg, seed + 50000, beta_T, beta_C)

            try:
                r = runner(data_tr["z"], data_tr["x"], data_tr["delta"],
                          data_ev["z"], data_ev["t"].view(-1), cfg, seed)
                r["seed"] = seed; r["mc_run"] = i
                rows.append(r)
                print(f"[{method_name}] run {i+1:02d}/{cfg.mc_runs} "
                      f"T_KS={r['T_KS']:.4f} T_W1={r['T_W1']:.4f}", flush=True)
            except Exception as e:
                import traceback
                print(f"[{method_name}] run {i+1} FAILED: {e}", flush=True)
                traceback.print_exc()

        if rows:
            summary = {}
            for k in ["T_KS", "T_W1", "train_time_sec"]:
                vals = [r.get(k, float("nan")) for r in rows]
                summary[k] = {"mean": float(np.nanmean(vals)),
                              "std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0}
            all_results[method_name] = {"results": rows, "summary": summary}
            print(f"\n=== {method_name} {ds_name} ρ={cfg.rho} ===")
            print(f"  T_KS: {summary['T_KS']['mean']:.4f} ± {summary['T_KS']['std']:.4f}")
            print(f"  T_W1: {summary['T_W1']['mean']:.4f} ± {summary['T_W1']['std']:.4f}")

    out_file = f"{args.out}_{args.dataset}_rho{args.rho}.json"
    with open(out_file, "w") as f:
        json.dump({"config": asdict(cfg), "dataset": ds_name,
                   "n_covariates": int(covariates.shape[1]),
                   "methods": all_results}, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
