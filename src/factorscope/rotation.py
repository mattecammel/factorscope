from __future__ import annotations

import warnings

import numpy as np

DEFAULT_LAGS = (1, 2, 3, 5, 8)
MIN_PAIRS = 2
SOFT_MIN_PAIRS = 30


def _warn_if_few_pairs(T, lags):
    pairs = T - max(lags)
    if pairs < SOFT_MIN_PAIRS:
        warnings.warn(
            f"lag {max(lags)} autocovariance rests on only {pairs} overlapping pairs "
            f"(< {SOFT_MIN_PAIRS}): the signature is very noisy and the rotation "
            "unreliable -- prefer more history or shorter `lags=`.", stacklevel=3)


def usable_lags(T, lags, min_pairs=MIN_PAIRS):
    usable = tuple(L for L in lags if L >= 1 and (T - L) >= min_pairs)
    if not usable:
        raise ValueError(
            f"panel too short for time-structure rotation: {T} rows but lags={tuple(lags)}; "
            f"need at least one lag L with T - L >= {min_pairs} "
            f"(supply shorter `lags=` or more history)."
        )
    return usable


def series_autocov(X, lag):
    Xc = X - X.mean(0)
    if lag == 0:
        C = Xc.T @ Xc / len(Xc)
    else:
        C = Xc[lag:].T @ Xc[:-lag] / (len(Xc) - lag)
    return 0.5 * (C + C.T)


def whiten(X):
    C0 = np.cov(X.T)
    C0 = np.atleast_2d(C0)
    w, V = np.linalg.eigh(C0)
    wmax = float(w.max())
    floor = wmax * 1e-12 if wmax > 0 else 1e-12
    w = np.clip(w, floor, None)
    Wh = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    return X @ Wh.T, Wh


def rjd(M, maxiter=200, eps=1e-9):
    A = np.asarray(M, dtype=float).copy()
    k = A.shape[1]
    V = np.eye(k)
    for _ in range(maxiter):
        moved = 0.0
        for p in range(k - 1):
            for q in range(p + 1, k):
                h = np.array([A[:, p, p] - A[:, q, q], A[:, p, q] + A[:, q, p]])
                G = h @ h.T
                ton, toff = G[0, 0] - G[1, 1], G[0, 1] + G[1, 0]
                theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton ** 2 + toff ** 2))
                c, s = np.cos(theta), np.sin(theta)
                if abs(s) > eps:
                    moved += abs(s)
                    cp, cq = A[:, :, p].copy(), A[:, :, q].copy()
                    A[:, :, p] = c * cp + s * cq
                    A[:, :, q] = -s * cp + c * cq
                    rp, rq = A[:, p, :].copy(), A[:, q, :].copy()
                    A[:, p, :] = c * rp + s * rq
                    A[:, q, :] = -s * rp + c * rq
                    vp, vq = V[:, p].copy(), V[:, q].copy()
                    V[:, p] = c * vp + s * vq
                    V[:, q] = -s * vp + c * vq
        if moved < eps:
            break
    return V


def signature(S, lags=DEFAULT_LAGS):
    lags = usable_lags(len(S), lags)
    Sc = S - S.mean(0)
    T = len(Sc)
    c0 = (Sc * Sc).sum(0) / T + 1e-12
    diag = np.array([(Sc[L:] * Sc[:-L]).sum(0) / T for L in lags])
    return (diag / c0).T


def sobi(X, lags=DEFAULT_LAGS):
    eff = usable_lags(len(X), lags)
    if len(eff) < len(tuple(lags)):
        warnings.warn(f"panel has {len(X)} rows; dropped lags too long for it, "
                      f"using lags={eff}", stacklevel=2)
    _warn_if_few_pairs(len(X), eff)
    Xw, Wh = whiten(X)
    Ms = np.stack([series_autocov(Xw, L) for L in eff])
    V = rjd(Ms)
    S = Xw @ V
    W = (V.T @ Wh)
    return S, W, signature(S, eff)


def amuse(X, lag=1):
    usable_lags(len(X), (lag,))
    _warn_if_few_pairs(len(X), (lag,))
    Xw, Wh = whiten(X)
    M = series_autocov(Xw, lag)
    _, U = np.linalg.eigh(M)
    S = Xw @ U
    return S, (U.T @ Wh), signature(S, (lag,))
