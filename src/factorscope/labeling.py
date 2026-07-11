from __future__ import annotations

import numpy as np
import pandas as pd

from ._types import LabelResult


def _standardize(df):
    return (df - df.mean()) / (df.std() + 1e-12)


def label_factors(factors, reference, top_k=2):
    factors = pd.DataFrame(factors)
    reference = pd.DataFrame(reference)
    idx = factors.index.intersection(reference.index)
    Fv, Rv = factors.loc[idx], reference.loc[idx]
    keep = (Fv.notna().all(axis=1) & Rv.notna().all(axis=1)).to_numpy()
    Fv, Rv = Fv[keep], Rv[keep]
    if len(Fv) < 2:
        raise ValueError(
            f"factors and reference share only {len(Fv)} complete overlapping dates; "
            "cannot label. Check that both are time-indexed on the same calendar "
            "(e.g. don't pass integer-indexed returns against Fama-French dates).")
    F = _standardize(Fv).values
    R = _standardize(Rv).values
    rnames = list(reference.columns)

    G = R.T @ R + 1e-8 * np.eye(R.shape[1])
    betas = np.linalg.solve(G, R.T @ F)
    fitted = R @ betas
    ss_res = ((F - fitted) ** 2).sum(0)
    ss_tot = (F ** 2).sum(0)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    table = pd.DataFrame(betas.T, index=factors.columns, columns=rnames)
    names = {}
    for j, f in enumerate(factors.columns):
        order = np.argsort(-np.abs(betas[:, j]))[:top_k]
        parts = []
        for r in order:
            b = betas[r, j]
            if abs(b) < 0.1:
                continue
            parts.append(f"{'+' if b >= 0 else '-'}{abs(b):.2f}*{rnames[r]}")
        names[f] = " ".join(parts) if parts else "(unexplained)"
    return LabelResult(table=table,
                       r2=pd.Series(r2, index=factors.columns),
                       names=pd.Series(names))
