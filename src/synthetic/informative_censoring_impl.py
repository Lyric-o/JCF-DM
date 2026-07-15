# -*- coding: utf-8 -*-
"""
Section 5.2 — Informative censoring with shared latent epsilon.

T = g1(Z, eps, eta1),  C = g2(Z, eps, eta2)
Two conditional flows take ctx = [Z, eps] as conditioning. Marginal
likelihood is integrated over eps via K-sample LogSumExp Monte Carlo.
Reg term: Soft-NA Wasserstein-1 against observed NA curve, sampling fresh eps.

Designed to plug into the same eval / plotting pipeline as the
uninformative version, with `family` selecting the g1/g2 distribution.
"""
from __future__ import annotations
import argparse, json, math, time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import numpy as np
import torch

from survival_dists import build_dist
from competing_methods import COMPETING

DEVICE = torch.device("cpu")
DTYPE = torch.float32

# gauge-fix penalty weight (Appendix B (A3)). Set via --lambda_alpha.
LAMBDA_ALPHA_INF: float = 0.01


# ----------------- config -----------------
@dataclass
class ICConfig:
    n_train: int = 384
    n_eval: int = 128
    z_std: float = 1.0
    family: str = "lognormal"        # one of survival_dists.FAMILY_REGISTRY
    hidden: int = 32
    K_eps: int = 8                   # MC samples for marginal likelihood
    s_sim: int = 3
    lambda_w: float = 0.2
    tau_temp: float = 0.25
    soft_alpha: float = 0.20
    grid_size: int = 80
    lr: float = 1e-3
    max_epochs: int = 220
    patience: int = 35
    min_delta: float = 1e-4
    time_min: float = 1e-4
    time_max: float = 10.0
    mc_runs: int = 10
    # informative DGP: shared Gamma frailty xi, T ~ Cox with xi*exp(beta_T z), C ~ Cox with xi^rho
    frailty_k: float = 2.0
    beta_T: float = 0.7
    beta_C: float = -0.5
    lam0_T: float = 0.22
    lam0_C: float = 0.24
    rho: float = 1.0                 # 1.0 = strong dep; 0.0 = independence


# ----------------- DGP: shared Gamma frailty Cox -----------------
def gen_informative(n: int, cfg: ICConfig, seed: int):
    g = torch.Generator(device=DEVICE); g.manual_seed(seed)
    z = torch.normal(0.0, cfg.z_std, size=(n, 1), generator=g, device=DEVICE, dtype=DTYPE)
    # Gamma(k, k) with mean 1, var 1/k
    xi = torch._standard_gamma(torch.full((n, 1), cfg.frailty_k, dtype=DTYPE), generator=g) / cfg.frailty_k
    u1 = torch.rand((n, 1), generator=g, dtype=DTYPE)
    u2 = torch.rand((n, 1), generator=g, dtype=DTYPE)
    haz_T = cfg.lam0_T * xi * torch.exp(cfg.beta_T * z)
    haz_C = cfg.lam0_C * (xi ** cfg.rho) * torch.exp(cfg.beta_C * z)
    T = -torch.log(torch.clamp(u1, min=1e-8)) / haz_T
    C = -torch.log(torch.clamp(u2, min=1e-8)) / haz_C
    T = torch.clamp(T, cfg.time_min, cfg.time_max)
    C = torch.clamp(C, cfg.time_min, cfg.time_max)
    x = torch.minimum(T, C)
    delta = (T <= C).to(DTYPE)
    return {"z": z, "t": T, "c": C, "x": x, "delta": delta, "xi": xi}


# ----------------- Soft-NA on grid -----------------
def hard_na_cdf_on_grid(x, delta, grid):
    x_np = x.detach().cpu().numpy().reshape(-1)
    d_np = delta.detach().cpu().numpy().reshape(-1)
    g_np = grid.detach().cpu().numpy().reshape(-1)
    inc = np.zeros_like(g_np, dtype=np.float64)
    for i, tau in enumerate(g_np):
        ar = np.sum(x_np >= tau)
        if ar <= 0: continue
        if i < len(g_np) - 1:
            inb = (x_np >= g_np[i]) & (x_np < g_np[i + 1])
        else:
            inb = x_np >= g_np[i]
        inc[i] = np.sum(d_np[inb]) / max(ar, 1.0)
    s = np.exp(-np.cumsum(inc))
    return torch.tensor(s, dtype=DTYPE).view(1, -1)


def soft_na_cdf_on_grid(x_soft, d_soft, grid, alpha):
    x = x_soft.view(-1, 1); d = d_soft.view(-1, 1); g0 = grid.view(1, -1)
    risk = torch.sigmoid((x - g0) / alpha).sum(dim=0)
    g_next = torch.cat([grid[1:], grid[-1:] + (grid[-1] - grid[-2])])
    inb = torch.sigmoid((x - g0) / alpha) - torch.sigmoid((x - g_next.view(1, -1)) / alpha)
    dN = (d * inb).sum(dim=0)
    haz = dN / (risk + 1e-6)
    return torch.exp(-torch.cumsum(haz, dim=0)).view(1, -1)


def w1_curve(s_obs, s_sim, grid):
    dt = grid[1:] - grid[:-1]
    return (torch.abs(s_obs[:, :-1] - s_sim[:, :-1]) * dt.view(1, -1)).sum(dim=1).mean()


# ----------------- MC marginal MLE over eps -----------------
def mc_marginal_mle(flow_t, flow_c, x, delta, z, K: int):
    # eps shared across (T,C) for the same i, indep across i and k
    n = x.shape[0]
    eps = torch.randn(K, n, 1, device=z.device, dtype=z.dtype)
    z_rep = z.unsqueeze(0).expand(K, -1, -1)                # [K,n,1]
    x_rep = x.unsqueeze(0).expand(K, -1, -1)
    d_rep = delta.unsqueeze(0).expand(K, -1, -1)
    ctx = torch.cat([z_rep, eps], dim=-1).reshape(K * n, -1)
    xv = x_rep.reshape(K * n, 1)
    dv = d_rep.reshape(K * n, 1)
    log_f_T = flow_t.log_pdf(xv, ctx)
    log_S_T = flow_t.log_survival(xv, ctx)
    log_f_C = flow_c.log_pdf(xv, ctx)
    log_S_C = flow_c.log_survival(xv, ctx)
    ll = dv * (log_f_T + log_S_C) + (1 - dv) * (log_S_T + log_f_C)  # [K*n,1]
    ll = ll.view(K, n)
    # log( (1/K) sum_k exp(ll_k) )
    log_marg = torch.logsumexp(ll, dim=0) - math.log(K)             # [n]
    return -log_marg.mean()


# ----------------- training -----------------
def train_one(cfg: ICConfig, seed: int) -> Dict:
    torch.manual_seed(seed); np.random.seed(seed)
    data = gen_informative(cfg.n_train, cfg, seed)
    z, x, delta = data["z"], data["x"], data["delta"]
    censoring = float((1 - delta.mean()).item())

    grid = torch.linspace(
        max(1e-4, float(torch.quantile(x, 0.01).item())),
        max(1e-3, float(torch.quantile(x, 0.99).item())),
        cfg.grid_size, device=DEVICE, dtype=DTYPE,
    )
    s_obs = hard_na_cdf_on_grid(x, delta, grid)

    flow_t = build_dist(cfg.family, ctx_dim=2, hidden=cfg.hidden).to(DEVICE)
    flow_c = build_dist(cfg.family, ctx_dim=2, hidden=cfg.hidden).to(DEVICE)
    opt = torch.optim.Adam(list(flow_t.parameters()) + list(flow_c.parameters()), lr=cfg.lr)

    best_loss = float("inf"); wait = 0; best_epoch = 0
    best_t = best_c = None
    t0 = time.perf_counter()
    for epoch in range(cfg.max_epochs):
        opt.zero_grad()
        loss_mle = mc_marginal_mle(flow_t, flow_c, x, delta, z, cfg.K_eps)

        reg = 0.0
        for _ in range(cfg.s_sim):
            eps_s = torch.randn_like(z)
            ctx_s = torch.cat([z, eps_s], dim=-1)
            t_sim = torch.clamp(flow_t.sample(ctx_s), cfg.time_min, cfg.time_max)
            c_sim = torch.clamp(flow_c.sample(ctx_s), cfg.time_min, cfg.time_max)
            x_sim = torch.minimum(t_sim, c_sim)
            d_soft = torch.sigmoid((c_sim - t_sim) / cfg.tau_temp)
            s_sim = soft_na_cdf_on_grid(x_sim, d_soft, grid, cfg.soft_alpha)
            reg = reg + w1_curve(s_obs, s_sim, grid)
        reg = reg / cfg.s_sim

        # gauge fix C-side only (same logic as uninformative).
        # T-side needs full tanh/spline capacity; C-side is over-parameterized.
        if hasattr(flow_c, "aux_penalty"):
            ctx_for_pen = torch.cat([z, torch.zeros_like(z)], dim=-1)
            aux = flow_c.aux_penalty(ctx_for_pen)
        else:
            aux = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)

        loss = loss_mle + cfg.lambda_w * reg + LAMBDA_ALPHA_INF * aux
        loss.backward()
        # tightened from default to 2.0 for outlier protection.
        torch.nn.utils.clip_grad_norm_(
            list(flow_t.parameters()) + list(flow_c.parameters()), max_norm=2.0
        )
        opt.step()

        cur = float(loss.item())
        if cur < best_loss - cfg.min_delta:
            best_loss = cur; best_epoch = epoch + 1; wait = 0
            best_t = {k: v.detach().cpu().clone() for k, v in flow_t.state_dict().items()}
            best_c = {k: v.detach().cpu().clone() for k, v in flow_c.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience: break
    train_sec = time.perf_counter() - t0
    if best_t is not None: flow_t.load_state_dict(best_t)
    if best_c is not None: flow_c.load_state_dict(best_c)

    # ---- evaluation: marginalise over (Z, eps) ----
    # YL fix 2026-04-09: previously the model marginal was estimated from
    # n_eval=128 samples (1 eps per i), while KM was built from n_train=384
    # observations. That sample-size mismatch alone explains ~0.06 of the
    # T_KS gap to KM. Fix: draw S_eval eps per individual and pool.
    S_eval = 20
    g_eval = torch.Generator(device=DEVICE); g_eval.manual_seed(seed + 98765)
    eval_data = gen_informative(cfg.n_eval, cfg, seed + 12345)
    z_e, t_true, c_true = eval_data["z"], eval_data["t"], eval_data["c"]
    with torch.no_grad():
        z_rep = z_e.unsqueeze(0).expand(S_eval, -1, -1).reshape(-1, z_e.shape[-1])
        eps_e = torch.randn(z_rep.shape[0], 1, generator=g_eval, dtype=DTYPE)
        ctx_e = torch.cat([z_rep, eps_e], dim=-1)
        t_model = torch.clamp(flow_t.sample(ctx_e, generator=g_eval), cfg.time_min, cfg.time_max)
        c_model = torch.clamp(flow_c.sample(ctx_e, generator=g_eval), cfg.time_min, cfg.time_max)

    def metrics(true, model):
        # YL fix 2026-04-09: previous w1 used paired sort which only works when
        # |true|==|model|; with S_eval>1 we now have |model|=S_eval*|true|, so
        # use the CDF-integral form (1D W1 = absolute area between CDFs).
        # Additionally report W1_trim85 / KS_trim85 (RMST-style restriction
        # to [0, q_85]) to suppress wide-tail integration noise on Cox-PH
        # truths whose tails extend beyond the time clip.
        t = true.view(-1); m = model.view(-1)
        ts, _ = torch.sort(t); ms, _ = torch.sort(m)
        pooled = torch.cat([t, m])
        gmin_full = float(torch.quantile(pooled, 0.01).item())
        gmax_full = float(torch.quantile(pooled, 0.99).item())
        gd_full = torch.linspace(max(1e-5, gmin_full), max(gmin_full + 1e-4, gmax_full), 300)
        Ft_full = torch.searchsorted(ts, gd_full, right=True).float() / ts.numel()
        Fm_full = torch.searchsorted(ms, gd_full, right=True).float() / ms.numel()
        diff_full = Ft_full - Fm_full
        dt_full = gd_full[1:] - gd_full[:-1]
        w1_full = float((torch.abs(diff_full[:-1]) * dt_full).sum().item())
        ks_full = float(torch.max(torch.abs(diff_full)).item())

        # trimmed [q_01, q_85]
        gmax_trim = float(torch.quantile(pooled, 0.85).item())
        gd_trim = torch.linspace(max(1e-5, gmin_full), max(gmin_full + 1e-4, gmax_trim), 300)
        Ft_trim = torch.searchsorted(ts, gd_trim, right=True).float() / ts.numel()
        Fm_trim = torch.searchsorted(ms, gd_trim, right=True).float() / ms.numel()
        diff_trim = Ft_trim - Fm_trim
        dt_trim = gd_trim[1:] - gd_trim[:-1]
        w1_trim = float((torch.abs(diff_trim[:-1]) * dt_trim).sum().item())
        ks_trim = float(torch.max(torch.abs(diff_trim)).item())

        return {"W1": w1_full, "KS": ks_full,
                "W1_trim85": w1_trim, "KS_trim85": ks_trim}

    out = {"seed": seed, "censoring_rate": censoring, "converged_epoch": float(best_epoch),
           "train_time_sec": train_sec, "best_loss": best_loss, "family": cfg.family, "rho": cfg.rho}
    for k, v in metrics(t_true, t_model).items(): out[f"T_{k}"] = v
    for k, v in metrics(c_true, c_model).items(): out[f"C_{k}"] = v

    # competing methods (marginal CDF over z_eval) on T
    grid_t = torch.linspace(
        float(torch.quantile(t_true, 0.01).item()),
        float(torch.quantile(t_true, 0.995).item()), 280, dtype=DTYPE)
    Ft_true = torch.searchsorted(torch.sort(t_true.view(-1))[0], grid_t, right=True).float() / t_true.numel()
    for name, (fit, cdf) in COMPETING.items():
        try:
            st = fit(z, x, delta)
            Fm = cdf(st, z_e, grid_t)
            out[f"T_comp{name}_KS"] = float(torch.max(torch.abs(Ft_true - Fm)).item())
        except Exception as e:
            out[f"T_comp{name}_KS"] = float("nan")
    return out


def run_mc(cfg: ICConfig, tag: str):
    rows = []
    for i in range(cfg.mc_runs):
        seed = 20260407 + i * 13
        r = train_one(cfg, seed); rows.append(r)
        print(f"[{tag}] {i+1:02d}/{cfg.mc_runs} family={cfg.family} rho={cfg.rho} "
              f"T_W1={r['T_W1']:.3f} C_W1={r['C_W1']:.3f} cens={r['censoring_rate']:.2f} "
              f"epoch={int(r['converged_epoch'])} time={r['train_time_sec']:.1f}s")
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    summary = {k: {"mean": float(np.mean([r[k] for r in rows])),
                   "std":  float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0}
               for k in keys}
    return rows, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", default="lognormal", choices=["lognormal", "weibull", "mixln", "spline", "rqs"])
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--mc_runs", type=int, default=10)
    p.add_argument("--n_train", type=int, default=384)   # expose for noise-floor sweep
    p.add_argument("--n_eval", type=int, default=128)    # YL 2026-04-09
    p.add_argument("--K_eps", type=int, default=8)       # marginal MLE MC samples
    p.add_argument("--max_epochs", type=int, default=220) # YL 2026-04-09
    p.add_argument("--lambda_alpha", type=float, default=0.01,
                   help="weight on aux gauge-fix penalty (Appendix B (A3))")
    p.add_argument("--out", default="ic_results")
    args = p.parse_args()
    global LAMBDA_ALPHA_INF
    LAMBDA_ALPHA_INF = args.lambda_alpha
    cfg = ICConfig(family=args.family, rho=args.rho, mc_runs=args.mc_runs,
                   n_train=args.n_train, n_eval=args.n_eval,
                   K_eps=args.K_eps, max_epochs=args.max_epochs)
    tag = f"{args.family}_rho{args.rho}"
    rows, summary = run_mc(cfg, tag)
    with open(f"{args.out}_{tag}.json", "w") as f:
        json.dump({"config": asdict(cfg), "results": rows, "summary": summary}, f, indent=2)
    print(f"saved {args.out}_{tag}.json")


if __name__ == "__main__":
    main()
