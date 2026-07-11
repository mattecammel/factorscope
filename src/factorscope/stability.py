from __future__ import annotations

import warnings

import numpy as np

from .decomposition import market_neutralize, pca_scores
from .rotation import DEFAULT_LAGS, sobi
from .alignment import matched_corr
from ._types import StabilityResult


def _loadings(scores, X):
    return np.linalg.lstsq(scores, X - X.mean(0), rcond=None)[0].T


def rolling_stability(returns, n_components, *, window=750, step=125,
                      neutralize=False, lags=DEFAULT_LAGS):
    X = np.asarray(returns, float)
    T = len(X)
    if window <= n_components + 1:
        raise ValueError(f"window ({window}) must exceed n_factors+1 ({n_components + 1}) "
                         "so each refit has enough rank to estimate the factors")
    prev = {}
    sobi_s, pca_s = [], []
    for a in range(0, T - window + 1, step):
        Xw = X[a:a + window]
        Xu = market_neutralize(Xw) if neutralize else Xw
        P = pca_scores(Xu, n_components)
        S, _, _ = sobi(P, lags)
        cur = {"pca": _loadings(P, Xw), "sobi": _loadings(S, Xw)}
        for name, store in (("pca", pca_s), ("sobi", sobi_s)):
            if name in prev:
                store.append(matched_corr(cur[name], prev[name], metric="cosine"))
        prev = cur
    if not sobi_s:
        warnings.warn(
            f"no rolling refits produced a comparison: the panel ({T} rows) fits fewer "
            f"than two windows of size {window} at step {step} (need T >= window + step). "
            "The StabilityResult is empty and its means are NaN.", stacklevel=2)
    return StabilityResult(sobi=np.array(sobi_s), pca=np.array(pca_s),
                           window=window, step=step)
