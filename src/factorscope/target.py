from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._types import TargetRotationResult


def _zscore(df):
    return (df - df.mean()) / (df.std() + 1e-12)


def target_rotation(factors, loadings, reference):
    F = pd.DataFrame(factors)
    R = pd.DataFrame(reference)
    idx = F.index.intersection(R.index)
    ok = (F.loc[idx].notna().all(axis=1) & R.loc[idx].notna().all(axis=1)).to_numpy()
    idx = idx[ok]
    if len(idx) < 2:
        raise ValueError(
            f"factors and reference share only {len(idx)} complete overlapping dates; "
            "cannot rotate. Check that both are time-indexed on the same calendar.")

    Sfull = _zscore(F)
    Sov = Sfull.loc[idx].values
    Rov = _zscore(R.loc[idx]).values
    k, m = Sov.shape[1], Rov.shape[1]
    rnames = list(R.columns)

    A = np.linalg.lstsq(Sov, Rov, rcond=None)[0]
    fitted = Sov @ A
    ss_res = ((Rov - fitted) ** 2).sum(0)
    ss_tot = (Rov ** 2).sum(0)
    subspace_r2 = pd.Series(1.0 - ss_res / (ss_tot + 1e-12), index=rnames)

    r = min(k, m)
    dropped = []
    if m > k:
        sel = np.sort(np.argsort(-subspace_r2.values)[:r])
        dropped = [rnames[i] for i in range(m) if i not in set(sel.tolist())]
        warnings.warn(
            f"{m} references but only {k} factors: aligned the {r} best-captured "
            f"references and dropped {dropped}. Fit more factors, or pass a reference "
            "subset, to align the rest.", stacklevel=3)
    else:
        sel = np.arange(m)

    U, _, Vt = np.linalg.svd(A[:, sel], full_matrices=False)
    Q = U @ Vt
    if r < k:
        Ufull = np.linalg.svd(Q, full_matrices=True)[0]
        Q = np.hstack([Q, Ufull[:, r:]])

    Srot = Sov @ Q
    for j in range(r):
        if np.corrcoef(Srot[:, j], Rov[:, sel[j]])[0, 1] < 0:
            Q[:, j] *= -1
    Srot = Sov @ Q

    aligned = [rnames[i] for i in sel]
    corr = pd.Series(
        [abs(np.corrcoef(Srot[:, j], Rov[:, sel[j]])[0, 1]) for j in range(r)],
        index=aligned)
    names = aligned + [f"Resid{i + 1}" for i in range(k - r)]

    B = np.asarray(loadings, float)
    Frot = pd.DataFrame(Sfull.values @ Q, index=F.index, columns=names)
    Brot = pd.DataFrame(B @ Q, index=pd.DataFrame(loadings).index, columns=names)
    return TargetRotationResult(factors=Frot, loadings=Brot, corr=corr,
                                subspace_r2=subspace_r2, rotation=Q, dropped=dropped)
