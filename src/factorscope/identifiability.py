from __future__ import annotations

from functools import lru_cache

import numpy as np

from .decomposition import eigenvalues
from .rotation import DEFAULT_LAGS, amuse, signature, sobi, usable_lags
from ._types import TrustReport

NULL_QUANTILE = 0.95
N_NULL_SIMS = 200
_NULL_T_CAP = 1000
DOMINANCE_THRESHOLD = 0.35


def top_pc_share(X):
    ev = eigenvalues(X)
    return float(ev[0] / ev.sum())


def identifiability_margin(S, T, lags=DEFAULT_LAGS):
    sig = signature(S, lags)
    if len(sig) < 2:
        return float("inf")
    d = np.linalg.norm(sig[:, None, :] - sig[None, :, :], axis=2)
    iu = np.triu_indices(len(sig), k=1)
    return float(np.sqrt(T) * d[iu].min())


def per_factor_snr(S, T, lags=DEFAULT_LAGS):
    sig = signature(S, lags)
    if len(sig) < 2:
        return np.full(len(sig), np.inf)
    d = np.linalg.norm(sig[:, None, :] - sig[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return np.sqrt(T) * d.min(axis=1)


def _null_rotate(Z, rotation, eff):
    if rotation == "amuse":
        S, _, _ = amuse(Z, eff[0])
    elif rotation == "none":
        S = Z
    else:
        S, _, _ = sobi(Z, eff)
    return S


@lru_cache(maxsize=128)
def _margin_null_cached(T_sim, K, eff, q, n_sim, seed, rotation):
    rng = np.random.default_rng(seed)
    vals = np.empty(n_sim)
    for i in range(n_sim):
        S = _null_rotate(rng.standard_normal((T_sim, K)), rotation, eff)
        vals[i] = identifiability_margin(S, T_sim, eff)
    return float(np.quantile(vals, q))


def margin_null_quantile(T, K, lags=DEFAULT_LAGS, q=NULL_QUANTILE,
                         n_sim=N_NULL_SIMS, seed=0, rotation="sobi"):
    if K < 2:
        return 0.0
    eff = usable_lags(T, lags)
    T_sim = int(min(T, max(_NULL_T_CAP, 4 * max(eff))))
    return _margin_null_cached(T_sim, K, eff, q, n_sim, seed, rotation)


def _auto_block_length(T, lags):
    b = max(2 * max(lags), int(round(T ** (1.0 / 3.0))))
    return int(np.clip(b, 1, max(1, T // 4)))


def _block_surrogate(cols, T, block, rng):
    Tc, K = cols.shape
    n_blocks = int(np.ceil(T / block))
    col = rng.integers(0, K, size=n_blocks)
    start = rng.integers(0, Tc, size=n_blocks)
    rows = (start[:, None] + np.arange(block)[None, :]) % Tc
    return cols[rows, col[:, None]].ravel()[:T]


def margin_null_quantile_block(source, lags=DEFAULT_LAGS, q=NULL_QUANTILE,
                               n_sim=N_NULL_SIMS, block=None, seed=0, rotation="sobi"):
    S = np.asarray(source, float)
    T, K = S.shape
    if K < 2:
        return 0.0
    eff = usable_lags(T, lags)
    T_sim = int(min(T, max(_NULL_T_CAP, 4 * max(eff))))
    if block is None:
        block = _auto_block_length(T_sim, eff)
    cols = (S - S.mean(0)) / (S.std(0) + 1e-12)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_sim)
    for i in range(n_sim):
        surr = np.column_stack([_block_surrogate(cols, T_sim, block, rng) for _ in range(K)])
        Sb = _null_rotate(surr, rotation, eff)
        vals[i] = identifiability_margin(Sb, T_sim, eff)
    return float(np.quantile(vals, q))


def detect_regime(X, n_components):
    ev = eigenvalues(X)
    share = ev[0] / ev.sum()
    if share >= DOMINANCE_THRESHOLD:
        return "dominated"
    kept = ev[:n_components]
    if kept.min() <= 1e-8 * kept.max():
        return "degenerate"
    return "flat"


def build_trust_report(X_used, S, T, *, neutralized, n_components, lags=DEFAULT_LAGS,
                       null="gaussian", rotation="sobi"):
    if null not in ("gaussian", "block"):
        raise ValueError("null must be 'gaussian' or 'block'")
    share = top_pc_share(X_used)
    margin = identifiability_margin(S, T, lags)
    snr = per_factor_snr(S, T, lags)
    regime = detect_regime(X_used, n_components)
    if null == "block":
        threshold = margin_null_quantile_block(S, lags=tuple(lags), rotation=rotation)
    else:
        threshold = margin_null_quantile(T, S.shape[1], tuple(lags), rotation=rotation)
    margin_ok = margin >= threshold
    warnings = []
    if regime == "dominated":
        if not neutralized:
            warnings.append("Spectrum dominated by one factor: PCA is already identified; "
                            "rotation may hurt. Neutralize the market first.")
        else:
            warnings.append("Spectrum still dominated after neutralization: a dominant factor "
                            "persists (not an equal-weight market), so the rotation is not "
                            "identifiable; factor labels are unreliable.")
    if not margin_ok:
        warnings.append(f"Identifiability margin {margin:.1f} < {threshold:.1f} (95th pct "
                        f"of rotating pure noise, {null} null): factor dynamics are too "
                        "alike to separate reliably; labels are uncertain.")
    if regime == "degenerate":
        warnings.append("Near-degenerate retained eigenvalues: reduce n_factors.")

    if not warnings:
        verdict = "Factors are identifiable and separated -- labels can be trusted."
    elif margin_ok and regime != "dominated":
        verdict = "Usable, with caveats (see warnings)."
    else:
        verdict = "Do NOT trust factor labels as-is (subspace is fine; orientation is not)."

    return TrustReport(regime=regime, top_pc_share=share, margin=margin,
                       margin_threshold=threshold, margin_ok=margin_ok,
                       neutralized=neutralized, per_factor_snr=snr,
                       verdict=verdict, warnings=warnings, null_model=null)
