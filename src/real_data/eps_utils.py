# -*- coding: utf-8 -*-
"""Helper utilities for shared-latent epsilon sampling."""

from __future__ import annotations
import torch
import torch.nn.functional as F


def make_eps(shape, generator, dtype, device, prior: str, transform: str):
    """Sample ε according to (prior, transform) and return a tensor of
    shape `shape`. Uses `generator` for determinism."""
    if prior == "normal":
        raw = torch.randn(shape, generator=generator, dtype=dtype, device=device)
    elif prior == "exp":
        u = torch.rand(shape, generator=generator, dtype=dtype, device=device)
        u = torch.clamp(u, min=1e-7, max=1.0 - 1e-7)
        raw = -torch.log(u)                              # Exp(1), mean=var=1
    elif prior == "halfnormal":
        raw = torch.randn(shape, generator=generator, dtype=dtype, device=device).abs()
    elif prior == "logn":
        eta = torch.randn(shape, generator=generator, dtype=dtype, device=device)
        raw = torch.exp(eta)                             # LogN(0, 1)
    else:
        raise ValueError(f"unknown eps_prior: {prior}")

    if transform == "identity":
        out = raw
    elif transform == "softplus":
        out = F.softplus(raw)
    elif transform == "exp":
        out = torch.exp(raw)
    else:
        raise ValueError(f"unknown eps_transform: {transform}")

    return out


def eps_pair(shape, generator, dtype, device, prior, transform, mode):
    """Draw (eps_t, eps_c) with the correct sharing pattern.

    Shared : returns (eps, eps) -- SAME tensor (identity, not a copy).
    Indep  : returns (eps_t, eps_c) with eps_t drawn first, then eps_c.

    Note for fair comparison across modes: we use a SINGLE generator so
    the random stream is deterministic; in indep mode the second draw
    advances the generator further, which is fine because the flow-
    internal generator is separate (see sampling code).
    """
    eps_t = make_eps(shape, generator, dtype, device, prior, transform)
    if mode == "shared":
        eps_c = eps_t
    elif mode == "indep":
        eps_c = make_eps(shape, generator, dtype, device, prior, transform)
    else:
        raise ValueError(f"unknown latent_mode: {mode}")
    return eps_t, eps_c
