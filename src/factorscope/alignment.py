from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._types import AlignmentResult


def _corr(A, B, metric="pearson"):
    A, B = np.asarray(A, float), np.asarray(B, float)
    if metric == "cosine":
        An = A / (np.linalg.norm(A, axis=0) + 1e-12)
        Bn = B / (np.linalg.norm(B, axis=0) + 1e-12)
        return np.nan_to_num(An.T @ Bn, nan=0.0)
    C = np.corrcoef(A.T, B.T)[: A.shape[1], A.shape[1]:]
    return np.nan_to_num(C, nan=0.0)


def matched_corr(S, F, metric="pearson"):
    C = np.abs(_corr(np.asarray(S), np.asarray(F), metric))
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].mean())


def align_factors(source, target):
    source = np.asarray(source, float)
    target = np.asarray(target, float)
    if source.shape[1] != target.shape[1]:
        raise ValueError(f"source and target must have the same number of factors; "
                         f"got {source.shape[1]} and {target.shape[1]}")
    C = _corr(source, target)
    ac = np.abs(C)
    r, c = linear_sum_assignment(-ac)
    K = target.shape[1]
    perm = np.empty(K, int)
    signs = np.ones(K)
    matched = np.zeros(K)
    for si, ti in zip(r, c):
        perm[ti] = si
        signs[ti] = np.sign(C[si, ti]) or 1.0
        matched[ti] = ac[si, ti]
    return AlignmentResult(permutation=perm, signs=signs,
                           matched_corr=matched, mean_corr=float(matched.mean()))
