from __future__ import annotations

import numpy as np

from .decomposition import eigenvalues, market_neutralize, pca_scores
from .identifiability import (
    identifiability_margin, margin_null_quantile, margin_null_quantile_block,
)
from .rotation import DEFAULT_LAGS, sobi
from ._types import SelectionResult


def suggest_n_factors(returns, k_max=15, *, neutralize=False, lags=DEFAULT_LAGS,
                      null="gaussian"):
    if null not in ("gaussian", "block"):
        raise ValueError("null must be 'gaussian' or 'block'")
    X = np.asarray(returns, float)
    Xu = market_neutralize(X) if neutralize else X
    ev = eigenvalues(Xu)
    k_max = int(min(k_max, len(ev) - 1))
    explained = np.cumsum(ev) / ev.sum()

    ratios = ev[:k_max] / (ev[1:k_max + 1] + 1e-12)
    eigen_gap_k = int(np.argmax(ratios)) + 1

    noise_floor_k = 1
    for k in range(2, k_max + 1):
        P = pca_scores(Xu, k)
        S, _, _ = sobi(P, lags)
        thr = (margin_null_quantile_block(S, lags=tuple(lags)) if null == "block"
               else margin_null_quantile(len(X), k, tuple(lags)))
        if identifiability_margin(S, len(X), lags) >= thr:
            noise_floor_k = k
        else:
            break

    n = int(min(eigen_gap_k, noise_floor_k))
    n = max(n, 1)
    floor_msg = (f"dynamics stay identifiable up to {noise_floor_k}" if noise_floor_k > 1
                 else "no multi-factor rotation separates from noise (floor 1)")
    reason = (f"eigen-gap suggests {eigen_gap_k}; {floor_msg} -> keep {n} "
              "(both variance-relevant and separable).")
    return SelectionResult(n_suggested=n, eigenvalues=ev, explained=explained,
                           eigen_gap_k=eigen_gap_k, noise_floor_k=noise_floor_k,
                           reason=reason)
