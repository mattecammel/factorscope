from __future__ import annotations

import warnings

import numpy as np

from .rotation import whiten


def _nanmean(X, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(X, axis=axis)


def market_neutralize(X):
    X = np.asarray(X, float)
    return X - _nanmean(X, axis=1)[:, None]


def pca_scores(X, n_components):
    Xc = X - X.mean(0)
    U, s, _ = np.linalg.svd(Xc, full_matrices=False)
    P = U[:, :n_components] * s[:n_components]
    return P / (P.std(0) + 1e-12)


def eigenvalues(X):
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    return (s ** 2) / (len(X) - 1)


def em_pca(X, n_components, n_iter=100, tol=1e-7):
    X = np.asarray(X, float)
    mask = np.isnan(X)
    if not mask.any():
        mu = X.mean(0)
        Xc = X - mu
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        return (U[:, :n_components] * s[:n_components],
                Vt[:n_components].T, mu, X.copy())

    col_mean = _nanmean(X, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    filled = np.where(mask, col_mean, X)
    for _ in range(n_iter):
        mu = filled.mean(0)
        Xc = filled - mu
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        recon = (U[:, :n_components] * s[:n_components]) @ Vt[:n_components] + mu
        new = np.where(mask, recon, X)
        delta = np.mean((new[mask] - filled[mask]) ** 2)
        filled = new
        if delta < tol:
            break
    mu = filled.mean(0)
    Xc = filled - mu
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (U[:, :n_components] * s[:n_components],
            Vt[:n_components].T, mu, filled)


def project_scores(X, loadings, mean, ridge=1e-6):
    X = np.asarray(X, float)
    B = np.asarray(loadings, float)
    k = B.shape[1]
    out = np.full((X.shape[0], k), np.nan)
    for t in range(X.shape[0]):
        obs = ~np.isnan(X[t])
        if obs.sum() < k:
            continue
        Bo = B[obs]
        xo = X[t, obs] - mean[obs]
        out[t] = np.linalg.solve(Bo.T @ Bo + ridge * np.eye(k), Bo.T @ xo)
    return out
