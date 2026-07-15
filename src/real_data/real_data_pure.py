# -*- coding: utf-8 -*-
"""Observational-data protocol used for Table 4."""

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cpu")
DTYPE = torch.float32

_DATA_DIR = os.environ.get(
    "JCFDM_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data")
)


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_seer():
    """Localized pancreatic-cancer cohort: 11,600 subjects, 61 model inputs."""
    seer_path = os.path.join(_DATA_DIR, "seer.csv")
    df = pd.read_csv(seer_path)
    times = df["time"].values.astype(np.float32)
    events = df["cod"].values.astype(np.float32)
    covariates = df.drop(columns=["time", "cod"]).values.astype(np.float32)
    return covariates, times, events, "SEER"


def load_support():
    csv_path = os.path.join(_DATA_DIR, "support.csv")
    df = pd.read_csv(csv_path)
    covariates = df.drop(columns=["duration", "event"]).values.astype(np.float32)
    times = df["duration"].values.astype(np.float32)
    events = df["event"].values.astype(np.float32)
    scaler = StandardScaler()
    covariates = scaler.fit_transform(covariates).astype(np.float32)
    return covariates, times, events, "SUPPORT"


def load_gbsg():
    csv_path = os.path.join(_DATA_DIR, "gbsg.csv")
    df = pd.read_csv(csv_path)
    covariates = df.drop(columns=["duration", "event"]).values.astype(np.float32)
    times = df["duration"].values.astype(np.float32)
    events = df["event"].values.astype(np.float32)
    return covariates, times, events, "Rotterdam and GBSG"


LOADERS = {"seer": load_seer, "support": load_support, "gbsg": load_gbsg}


# ── IPCW helpers (pure numpy) ────────────────────────────────────────────────

def _km_censoring(times, events):
    """Kaplan-Meier estimate of the CENSORING distribution G(t).
    events: 1 = event, 0 = censored.  We flip: censoring indicator = 1-events.
    Returns (unique_times_sorted, G_at_times) as 1-D numpy arrays.
    G(t) = P(C > t) estimated by KM on the censoring indicator.
    """
    cens_indicator = 1.0 - events.astype(np.float64)
    times = times.astype(np.float64)

    # unique times where a censoring "event" happened
    cens_times = np.unique(times[cens_indicator > 0.5])
    cens_times.sort()

    if len(cens_times) == 0:
        # no censoring at all -- G(t)=1 everywhere
        return np.array([0.0]), np.array([1.0])

    g = 1.0
    g_vals = np.ones(len(cens_times), dtype=np.float64)
    for k, tk in enumerate(cens_times):
        at_risk = (times >= tk).sum()
        d_k = ((times == tk) & (cens_indicator > 0.5)).sum()
        if at_risk > 0:
            g *= (1.0 - d_k / at_risk)
        g_vals[k] = g

    return cens_times, g_vals


def _g_at_t(t_query, km_times, km_g):
    """Evaluate censoring survival G(t) at arbitrary query times using step function.
    G(t) = right-continuous step function from KM.
    For t < first event time, G(t)=1.
    """
    idx = np.searchsorted(km_times, t_query, side="right") - 1
    out = np.ones_like(t_query, dtype=np.float64)
    valid = idx >= 0
    out[valid] = km_g[idx[valid]]
    return out


# ── Evaluation metrics (pure numpy IPCW) ─────────────────────────────────────

def concordance_td(et_train, ei_train, et_test, ei_test, surv_preds, pred_times):
    """Time-dependent concordance index (IPCW).
    IPCW censoring distribution estimated from TRAINING set."""
    et_train = et_train.astype(np.float64)
    ei_train = ei_train.astype(np.float64)
    et_test = et_test.astype(np.float64)
    ei_test = ei_test.astype(np.float64)

    # truncation time tau: 90th percentile of observed event times in train
    event_times_train = et_train[ei_train > 0.5]
    if len(event_times_train) == 0:
        return float("nan")
    tau = np.percentile(event_times_train, 90)

    # risk scores at tau: negative survival probability
    mid_idx = np.searchsorted(pred_times, tau, side="right") - 1
    mid_idx = min(max(mid_idx, 0), len(pred_times) - 1)
    risk_scores = -surv_preds[:, mid_idx].astype(np.float64)

    # censoring KM from training set
    km_times, km_g = _km_censoring(et_train, ei_train)

    # IPCW concordance: only consider pairs where subject i has event before tau
    n = len(et_test)
    numerator = 0.0
    denominator = 0.0

    for i in range(n):
        if ei_test[i] < 0.5:
            continue  # i is censored, skip
        if et_test[i] > tau:
            continue  # beyond truncation

        g_ti = max(_g_at_t(np.array([et_test[i]]), km_times, km_g)[0], 1e-8)
        w_i = 1.0 / (g_ti * g_ti)

        for j in range(n):
            if i == j:
                continue
            if et_test[j] <= et_test[i]:
                continue  # need t_j > t_i for a concordant/discordant pair

            # concordant if risk_i > risk_j (higher risk = shorter survival)
            if risk_scores[i] > risk_scores[j]:
                numerator += w_i
            elif risk_scores[i] == risk_scores[j]:
                numerator += 0.5 * w_i
            denominator += w_i

    if denominator < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def integrated_brier_score(et_train, ei_train, et_test, ei_test, surv_preds, pred_times):
    """IPCW Integrated Brier Score. Censoring distribution from TRAINING set."""
    et_train = et_train.astype(np.float64)
    ei_train = ei_train.astype(np.float64)
    et_test = et_test.astype(np.float64)
    ei_test = ei_test.astype(np.float64)

    tau_lo = np.percentile(et_train, 5)
    event_times_train = et_train[ei_train > 0.5]
    if len(event_times_train) == 0:
        return float("nan")
    tau_hi = np.percentile(event_times_train, 90)

    mask = (pred_times >= tau_lo) & (pred_times <= tau_hi)
    if mask.sum() < 5:
        return float("nan")

    grid = pred_times[mask].astype(np.float64)
    surv_at_grid = surv_preds[:, mask].astype(np.float64)
    n_test = len(et_test)

    # censoring KM from training set
    km_times, km_g = _km_censoring(et_train, ei_train)

    # Brier score at each time point
    bs_values = np.zeros(len(grid), dtype=np.float64)

    for k, t_k in enumerate(grid):
        score = 0.0
        for i in range(n_test):
            s_hat = surv_at_grid[i, k]

            if et_test[i] <= t_k and ei_test[i] > 0.5:
                # event before t_k: observed event
                g_xi = max(_g_at_t(np.array([et_test[i]]), km_times, km_g)[0], 1e-8)
                score += (s_hat ** 2) / g_xi
            elif et_test[i] > t_k:
                # still alive/uncensored at t_k
                g_tk = max(_g_at_t(np.array([t_k]), km_times, km_g)[0], 1e-8)
                score += ((1.0 - s_hat) ** 2) / g_tk
            # else: censored before t_k, contributes 0

        bs_values[k] = score / n_test

    # trapezoidal integration over grid, normalized by time range
    if len(grid) < 2:
        return float("nan")
    _trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    ibs = _trapz(bs_values, grid) / (grid[-1] - grid[0])
    return float(ibs)


# ── Method implementations ──────────────────────────────────────────────────

def _make_surv_grid(times, n_points=100):
    return np.linspace(max(times.min(), 0.1), times.max(), n_points).astype(np.float32)


def run_joint_flow(Z_tr, t_tr, e_tr, Z_te, t_te, e_te, cfg, seed):
    """Our shared-ε joint flow."""
    import survival_dists as _sd
    _sd.set_activation("tanh")
    t0 = time.perf_counter()

    ctx_dim = Z_tr.shape[1] + 1
    flow_t = _sd.build_dist("spline", ctx_dim=ctx_dim, hidden=cfg.hidden,
                            n_knots=cfg.n_knots).to(DEVICE)
    flow_c = _sd.build_dist("spline", ctx_dim=ctx_dim, hidden=cfg.hidden,
                            n_knots=cfg.n_knots).to(DEVICE)

    z = torch.tensor(Z_tr, dtype=DTYPE)
    x = torch.tensor(t_tr, dtype=DTYPE).unsqueeze(1)
    delta = torch.tensor(e_tr, dtype=DTYPE).unsqueeze(1)

    params = list(flow_t.parameters()) + list(flow_c.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)

    best_loss = float("inf"); wait = 0
    best_st = best_sc = None
    K = cfg.K_eps
    batch_size = min(512, z.shape[0])

    for epoch in range(cfg.max_epochs):
        perm = torch.randperm(z.shape[0])
        epoch_loss = 0.0
        n_batches = 0
        for b_start in range(0, z.shape[0], batch_size):
            b_idx = perm[b_start:b_start + batch_size]
            z_b = z[b_idx]; x_b = x[b_idx]; d_b = delta[b_idx]
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
            ll_per = d_rep * (ll_T + ls_C) + (1 - d_rep) * (ls_T + ll_C)
            ll_per = ll_per.view(K, nb, -1)
            ll_lse = torch.logsumexp(ll_per, dim=0) - np.log(K)
            nll = -torch.mean(ll_lse)

            nll.backward()
            torch.nn.utils.clip_grad_norm_(params, 2.0)
            opt.step()
            epoch_loss += float(nll.item()) * nb
            n_batches += nb

        cur = epoch_loss / max(n_batches, 1)
        if cur < best_loss - 1e-4:
            best_loss = cur; wait = 0
            best_st = {k: v.cpu().clone() for k, v in flow_t.state_dict().items()}
            best_sc = {k: v.cpu().clone() for k, v in flow_c.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    train_sec = time.perf_counter() - t0
    if best_st: flow_t.load_state_dict(best_st)
    if best_sc: flow_c.load_state_dict(best_sc)

    grid = _make_surv_grid(t_te)
    z_te_t = torch.tensor(Z_te, dtype=DTYPE)
    n_mc = 50
    surv_all = np.zeros((Z_te.shape[0], len(grid)), dtype=np.float64)

    with torch.no_grad():
        for k in range(n_mc):
            eps_k = torch.randn(z_te_t.shape[0], 1, dtype=DTYPE)
            ctx_k = torch.cat([z_te_t, eps_k], dim=-1)
            for j, tj in enumerate(grid):
                tj_t = torch.full((z_te_t.shape[0], 1), float(tj), dtype=DTYPE)
                ls = flow_t.log_survival(tj_t, ctx_k)
                surv_all[:, j] += np.exp(ls.numpy().squeeze())
    surv_all /= n_mc

    c_idx = concordance_td(t_tr, e_tr, t_te, e_te, surv_all, grid)
    ibs = integrated_brier_score(t_tr, e_tr, t_te, e_te, surv_all, grid)

    z_te_rep = z_te_t.repeat(20, 1)
    eps_rep = torch.randn(20 * z_te_t.shape[0], 1, dtype=DTYPE)
    ctx_rep = torch.cat([z_te_rep, eps_rep], dim=-1)
    with torch.no_grad():
        held_nll = -torch.mean(
            torch.tensor(e_te, dtype=DTYPE).repeat(20).unsqueeze(1)
            * flow_t.log_pdf(torch.tensor(t_te, dtype=DTYPE).repeat(20).unsqueeze(1), ctx_rep)
            + (1 - torch.tensor(e_te, dtype=DTYPE).repeat(20).unsqueeze(1))
            * flow_t.log_survival(torch.tensor(t_te, dtype=DTYPE).repeat(20).unsqueeze(1), ctx_rep)
        ).item()

    return {"method": "joint_flow", "C_index": c_idx, "IBS": ibs,
            "held_out_nll": held_nll, "train_time_sec": train_sec}


def run_tonly_flow(Z_tr, t_tr, e_tr, Z_te, t_te, e_te, cfg, seed):
    """T-only conditional NF (AUSSET)."""
    import survival_dists as _sd
    _sd.set_activation("tanh")
    t0 = time.perf_counter()

    flow_t = _sd.build_dist("spline", ctx_dim=Z_tr.shape[1], hidden=cfg.hidden,
                            n_knots=cfg.n_knots).to(DEVICE)
    z = torch.tensor(Z_tr, dtype=DTYPE)
    x = torch.tensor(t_tr, dtype=DTYPE).unsqueeze(1)
    delta = torch.tensor(e_tr, dtype=DTYPE).unsqueeze(1)

    opt = torch.optim.Adam(flow_t.parameters(), lr=cfg.lr)
    best_loss = float("inf"); wait = 0; best_state = None
    batch_size = min(1024, z.shape[0])
    for epoch in range(cfg.max_epochs):
        perm = torch.randperm(z.shape[0])
        epoch_loss = 0.0; n_seen = 0
        for b_start in range(0, z.shape[0], batch_size):
            b_idx = perm[b_start:b_start + batch_size]
            z_b = z[b_idx]; x_b = x[b_idx]; d_b = delta[b_idx]
            opt.zero_grad()
            nll = -torch.mean(d_b * flow_t.log_pdf(x_b, z_b)
                              + (1 - d_b) * flow_t.log_survival(x_b, z_b))
            nll.backward()
            torch.nn.utils.clip_grad_norm_(flow_t.parameters(), 2.0)
            opt.step()
            epoch_loss += float(nll.item()) * z_b.shape[0]
            n_seen += z_b.shape[0]
        cur = epoch_loss / max(n_seen, 1)
        if cur < best_loss - 1e-4:
            best_loss = cur; wait = 0
            best_state = {k: v.cpu().clone() for k, v in flow_t.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    train_sec = time.perf_counter() - t0
    if best_state: flow_t.load_state_dict(best_state)

    grid = _make_surv_grid(t_te)
    z_te_t = torch.tensor(Z_te, dtype=DTYPE)
    surv_all = np.zeros((Z_te.shape[0], len(grid)), dtype=np.float64)
    with torch.no_grad():
        for j, tj in enumerate(grid):
            tj_t = torch.full((z_te_t.shape[0], 1), float(tj), dtype=DTYPE)
            ls = flow_t.log_survival(tj_t, z_te_t)
            surv_all[:, j] = np.exp(ls.numpy().squeeze())

    c_idx = concordance_td(t_tr, e_tr, t_te, e_te, surv_all, grid)
    ibs = integrated_brier_score(t_tr, e_tr, t_te, e_te, surv_all, grid)

    with torch.no_grad():
        held_nll = -torch.mean(
            torch.tensor(e_te, dtype=DTYPE).unsqueeze(1)
            * flow_t.log_pdf(torch.tensor(t_te, dtype=DTYPE).unsqueeze(1), z_te_t)
            + (1 - torch.tensor(e_te, dtype=DTYPE).unsqueeze(1))
            * flow_t.log_survival(torch.tensor(t_te, dtype=DTYPE).unsqueeze(1), z_te_t)
        ).item()

    return {"method": "tonly_flow", "C_index": c_idx, "IBS": ibs,
            "held_out_nll": held_nll, "train_time_sec": train_sec}


# ---------- CoxPH via torch partial likelihood + Breslow --------------------

def run_coxph(Z_tr, t_tr, e_tr, Z_te, t_te, e_te, cfg, seed):
    t0 = time.perf_counter()

    z_np = Z_tr.astype(np.float64)
    x_np = t_tr.astype(np.float64)
    d_np = e_tr.astype(np.float64)
    n, p = z_np.shape

    # fit Cox partial likelihood with torch autograd
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
        eta = z_sorted @ beta
        cumsum_exp = torch.logcumsumexp(eta, dim=0)
        npl = -torch.sum(d_sorted * (eta - cumsum_exp))
        npl = npl + 0.01 * torch.sum(beta ** 2)
        npl.backward()
        return npl

    for _ in range(5):
        optimizer.step(closure)

    beta_hat = beta.detach().numpy()

    # Breslow baseline cumulative hazard
    eta_all = z_np @ beta_hat
    exp_eta = np.exp(eta_all)

    event_mask = d_np > 0.5
    event_times_set = np.unique(x_np[event_mask])
    event_times_set.sort()

    if len(event_times_set) == 0:
        event_times_set = np.array([x_np.mean()])

    baseline_haz = np.zeros(len(event_times_set))
    for k, t_k in enumerate(event_times_set):
        at_risk = x_np >= t_k
        denom = exp_eta[at_risk].sum()
        d_k = ((x_np == t_k) & (d_np > 0.5)).sum()
        baseline_haz[k] = d_k / max(denom, 1e-12)

    cum_baseline_haz = np.cumsum(baseline_haz)
    cox_event_times = event_times_set

    train_sec = time.perf_counter() - t0

    # predict survival curves on the evaluation grid
    grid = _make_surv_grid(t_te)
    z_ev_np = Z_te.astype(np.float64)
    n_test = z_ev_np.shape[0]
    surv_all = np.zeros((n_test, len(grid)), dtype=np.float64)

    for i in range(n_test):
        eta_i = z_ev_np[i] @ beta_hat
        # interpolate cumulative baseline hazard onto the grid
        H0_at_grid = np.interp(grid, cox_event_times, cum_baseline_haz,
                                left=0.0, right=cum_baseline_haz[-1])
        surv_all[i] = np.exp(-H0_at_grid * np.exp(eta_i))

    c_idx = concordance_td(t_tr, e_tr, t_te, e_te, surv_all, grid)
    ibs = integrated_brier_score(t_tr, e_tr, t_te, e_te, surv_all, grid)
    return {"method": "coxph", "C_index": c_idx, "IBS": ibs,
            "train_time_sec": train_sec}


# ---------- RSF via sklearn RandomForest + per-leaf KM ----------------------

def run_rsf(Z_tr, t_tr, e_tr, Z_te, t_te, e_te, cfg, seed):
    from sklearn.ensemble import RandomForestRegressor

    t0 = time.perf_counter()

    z_np = Z_tr.astype(np.float64)
    x_np = t_tr.astype(np.float64)
    d_np = e_tr.astype(np.float64)

    rf = RandomForestRegressor(
        n_estimators=100, max_depth=5,
        min_samples_split=10, min_samples_leaf=6,
        random_state=seed,
    )
    rf.fit(z_np, x_np)

    # leaf assignments for training data
    leaves_train = rf.apply(z_np)  # (n_train, n_trees)

    # Build common time grid from unique event times
    event_times = np.unique(x_np[d_np > 0.5])
    event_times.sort()
    if len(event_times) == 0:
        event_times = np.array([x_np.mean()])
    n_times = len(event_times)

    # Pre-compute per-leaf KM curves for each tree
    n_trees = leaves_train.shape[1]
    tree_leaf_surv = []
    for t_idx in range(n_trees):
        leaf_ids = leaves_train[:, t_idx]
        unique_leaves = np.unique(leaf_ids)
        leaf_km = {}
        for lf in unique_leaves:
            lf_mask = leaf_ids == lf
            lf_x = x_np[lf_mask]
            lf_d = d_np[lf_mask]
            surv = np.ones(n_times)
            s = 1.0
            for k, tk in enumerate(event_times):
                at_risk = (lf_x >= tk).sum()
                d_k = ((lf_x == tk) & (lf_d > 0.5)).sum()
                if at_risk > 0:
                    s *= (1.0 - d_k / at_risk)
                surv[k] = s
            leaf_km[lf] = surv
        tree_leaf_surv.append(leaf_km)

    train_sec = time.perf_counter() - t0

    # Predict survival on the evaluation grid
    grid = _make_surv_grid(t_te)
    z_ev_np = Z_te.astype(np.float64)
    leaves_eval = rf.apply(z_ev_np)  # (n_test, n_trees)
    n_test = z_ev_np.shape[0]

    surv_all = np.zeros((n_test, len(grid)), dtype=np.float64)
    for i in range(n_test):
        # average survival curve across trees
        surv_avg = np.zeros(n_times)
        for t_idx in range(n_trees):
            lf = leaves_eval[i, t_idx]
            surv_avg += tree_leaf_surv[t_idx][lf]
        surv_avg /= n_trees
        # interpolate onto the evaluation grid
        surv_all[i] = np.interp(grid, event_times, surv_avg,
                                 left=1.0, right=surv_avg[-1])

    c_idx = concordance_td(t_tr, e_tr, t_te, e_te, surv_all, grid)
    ibs = integrated_brier_score(t_tr, e_tr, t_te, e_te, surv_all, grid)
    return {"method": "rsf", "C_index": c_idx, "IBS": ibs,
            "train_time_sec": train_sec}


# ---------- DeepHit (pure PyTorch) ------------------------------------------

class _DeepHitNet(torch.nn.Module):
    """Simple MLP that outputs a PMF over discrete time bins."""
    def __init__(self, in_features, num_bins, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden, num_bins),
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


def _deephit_nll(pmf, bin_idx, event):
    """Negative log-likelihood for DeepHit.
    pmf:     (batch, num_bins) -- predicted PMF
    bin_idx: (batch,) long -- discretised time bin index
    event:   (batch,) float -- 1 if event, 0 if censored
    """
    eps = 1e-12
    batch_size = pmf.shape[0]
    p_at_k = pmf[torch.arange(batch_size), bin_idx].clamp(min=eps)
    cdf_at_k = pmf.cumsum(dim=1)[torch.arange(batch_size), bin_idx]
    surv_at_k = (1.0 - cdf_at_k).clamp(min=eps)
    log_lik = event * torch.log(p_at_k) + (1.0 - event) * torch.log(surv_at_k)
    return -log_lik.mean()


def run_deephit(Z_tr, t_tr, e_tr, Z_te, t_te, e_te, cfg, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    z_np = Z_tr.astype(np.float32)
    x_np = t_tr.astype(np.float32)
    d_np = e_tr.astype(np.float32)

    n = len(x_np)
    n_val = n // 5
    idx = np.random.permutation(n)
    tr_idx, val_idx = idx[n_val:], idx[:n_val]

    # discretise time into bins
    num_bins = 100
    cuts = np.linspace(float(x_np.min()), float(x_np.max()), num_bins + 1)
    bin_all = np.digitize(x_np, cuts[1:])
    bin_all = np.clip(bin_all, 0, num_bins - 1)
    bin_centres = 0.5 * (cuts[:-1] + cuts[1:])

    z_tr_t = torch.tensor(z_np[tr_idx], dtype=torch.float32)
    b_tr_t = torch.tensor(bin_all[tr_idx], dtype=torch.long)
    d_tr_t = torch.tensor(d_np[tr_idx], dtype=torch.float32)

    z_val_t = torch.tensor(z_np[val_idx], dtype=torch.float32)
    b_val_t = torch.tensor(bin_all[val_idx], dtype=torch.long)
    d_val_t = torch.tensor(d_np[val_idx], dtype=torch.float32)

    in_features = z_np.shape[1]
    net = _DeepHitNet(in_features, num_bins, hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    t0 = time.perf_counter()
    best_val_loss = float("inf")
    patience, wait = 20, 0
    best_state = None
    batch_size = 256

    for epoch in range(200):
        net.train()
        perm = torch.randperm(z_tr_t.shape[0])
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, z_tr_t.shape[0], batch_size):
            bi = perm[start:start + batch_size]
            pmf = net(z_tr_t[bi])
            loss = _deephit_nll(pmf, b_tr_t[bi], d_tr_t[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        # validation
        net.eval()
        with torch.no_grad():
            pmf_val = net(z_val_t)
            val_loss = _deephit_nll(pmf_val, b_val_t, d_val_t).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    train_sec = time.perf_counter() - t0
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    # predict survival curves for eval subjects
    z_ev_t = torch.tensor(Z_te.astype(np.float32), dtype=torch.float32)
    with torch.no_grad():
        pmf_eval = net(z_ev_t)  # (n_test, num_bins)
    cdf_eval = pmf_eval.cumsum(dim=1).numpy()
    surv_eval = 1.0 - cdf_eval  # (n_test, num_bins)

    # interpolate onto standard evaluation grid
    grid = _make_surv_grid(t_te)
    n_test = Z_te.shape[0]
    surv_all = np.zeros((n_test, len(grid)), dtype=np.float64)
    for i in range(n_test):
        surv_all[i] = np.interp(grid, bin_centres, surv_eval[i],
                                 left=1.0, right=float(surv_eval[i, -1]))

    c_idx = concordance_td(t_tr, e_tr, t_te, e_te, surv_all, grid)
    ibs = integrated_brier_score(t_tr, e_tr, t_te, e_te, surv_all, grid)
    return {"method": "deephit", "C_index": c_idx, "IBS": ibs,
            "train_time_sec": train_sec}


METHODS = {
    "joint_flow": run_joint_flow,
    "tonly_flow": run_tonly_flow,
    "coxph": run_coxph,
    "rsf": run_rsf,
    "deephit": run_deephit,
}


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class PureRealCfg:
    dataset: str = "seer"
    hidden: int = 64
    n_knots: int = 6
    K_eps: int = 16
    lr: float = 1e-3
    max_epochs: int = 600
    patience: int = 50
    mc_runs: int = 5
    test_frac: float = 0.25


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="seer", choices=list(LOADERS.keys()))
    p.add_argument("--methods", default="joint_flow,tonly_flow,coxph,rsf,deephit")
    p.add_argument("--mc_runs", type=int, default=5)
    p.add_argument("--max_epochs", type=int, default=600)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--test_frac", type=float, default=0.25,
                   help="held-out fraction; 0.25 reproduces Table 4")
    p.add_argument("--out", default="pure_real")
    args = p.parse_args()

    covariates, times, events, ds_name = LOADERS[args.dataset]()
    print(f"Loaded {ds_name}: n={covariates.shape[0]}, d={covariates.shape[1]}, "
          f"event_rate={events.mean():.3f}")

    cfg = PureRealCfg(dataset=args.dataset, mc_runs=args.mc_runs,
                       max_epochs=args.max_epochs, hidden=args.hidden,
                       test_frac=args.test_frac)
    methods = [m.strip() for m in args.methods.split(",")]
    all_results = {}

    for method_name in methods:
        if method_name not in METHODS:
            print(f"Unknown: {method_name}"); continue
        runner = METHODS[method_name]
        rows = []
        for i in range(cfg.mc_runs):
            seed = 20260427 + i * 19
            Z_tr, Z_te, t_tr, t_te, e_tr, e_te = train_test_split(
                covariates, times, events,
                test_size=cfg.test_frac, stratify=events, random_state=seed)

            try:
                r = runner(Z_tr, t_tr, e_tr, Z_te, t_te, e_te, cfg, seed)
                r["seed"] = seed; r["mc_run"] = i
                rows.append(r)
                c = r.get("C_index", float("nan"))
                ibs = r.get("IBS", float("nan"))
                nll = r.get("held_out_nll", float("nan"))
                print(f"[{method_name}] run {i+1:02d}/{cfg.mc_runs} "
                      f"C={c:.4f} IBS={ibs:.4f} NLL={nll:.3f} "
                      f"t={r['train_time_sec']:.1f}s", flush=True)
            except Exception as e:
                import traceback
                print(f"[{method_name}] run {i+1} FAILED: {e}", flush=True)
                traceback.print_exc()

        if rows:
            summary = {}
            for k in ["C_index", "IBS", "held_out_nll", "train_time_sec"]:
                vals = [r.get(k, float("nan")) for r in rows]
                vals = [v for v in vals if not np.isnan(v)]
                if vals:
                    summary[k] = {"mean": float(np.mean(vals)),
                                  "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
            all_results[method_name] = {"results": rows, "summary": summary}
            print(f"\n=== {method_name} on {ds_name} ===")
            for k, v in summary.items():
                print(f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f}")

    out_file = f"{args.out}_{args.dataset}.json"
    with open(out_file, "w") as f:
        json.dump({"config": asdict(cfg), "dataset": ds_name,
                   "n_subjects": int(covariates.shape[0]),
                   "n_covariates": int(covariates.shape[1]),
                   "event_rate": float(events.mean()),
                   "methods": all_results}, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
