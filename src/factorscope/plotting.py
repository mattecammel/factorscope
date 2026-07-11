from __future__ import annotations

import numpy as np


def _mpl():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("plotting requires matplotlib: pip install factorscope[plot]") from e
    return plt


def plot_labels(label_result, ax=None):
    plt = _mpl()
    tbl = label_result.table
    ax = ax or plt.subplots(figsize=(1.2 * tbl.shape[1] + 2, 0.5 * tbl.shape[0] + 1.5))[1]
    vmax = float(np.nanmax(np.abs(tbl.values))) if tbl.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    im = ax.imshow(tbl.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(tbl.shape[1]), tbl.columns, rotation=45, ha="right")
    ax.set_yticks(range(tbl.shape[0]), tbl.index)
    ax.figure.colorbar(im, ax=ax, label="loading")
    ax.set_title("factor -> reference-style loadings")
    return ax


def plot_stability(stability_result, ax=None):
    plt = _mpl()
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    r = stability_result
    x = np.arange(1, len(r.sobi) + 1)
    ax.plot(x, r.sobi, "-o", label=f"SOBI ({r.sobi_mean:.2f})")
    ax.plot(x, r.pca, "-o", label=f"PCA ({r.pca_mean:.2f})")
    ax.set_xlabel("refit #")
    ax.set_ylabel("persistence vs previous refit")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    ax.set_title("factor identity persistence across refits")
    return ax


def plot_margin(trust_report, ax=None):
    plt = _mpl()
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    tr = trust_report
    snr = np.asarray(tr.per_factor_snr, float)
    finite = np.isfinite(snr)
    x = np.arange(1, len(snr) + 1)[finite]
    vals = snr[finite]
    colors = ["#2b8cbe" if v >= tr.margin_threshold else "#d7301f" for v in vals]
    ax.bar(x, vals, color=colors)
    ax.axhline(tr.margin_threshold, ls="--", color="black",
               label=f"{tr.null_model} noise floor = {tr.margin_threshold:.1f}")
    ax.set_xticks(x, [f"F{i}" for i in x])
    ax.set_xlabel("factor")
    ax.set_ylabel("separation from nearest factor (std errors)")
    ax.legend(frameon=False)
    ax.set_title("identifiability: which factors clear the noise floor?")
    return ax


def plot_scree(selection_result, ax=None, k_max=20):
    plt = _mpl()
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    ev = np.asarray(selection_result.eigenvalues, float)
    share = ev / ev.sum()
    n = int(min(k_max, len(share)))
    x = np.arange(1, n + 1)
    ax.bar(x, share[:n], color="#7fcdbb")
    ax.axvline(selection_result.n_suggested + 0.5, ls="--", color="black",
               label=f"suggested k = {selection_result.n_suggested}")
    ax.set_xlabel("component")
    ax.set_ylabel("variance share")
    ax.legend(frameon=False)
    ax.set_title("eigenvalue spectrum")
    return ax


def plot_target_corr(target_result, ax=None):
    plt = _mpl()
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    c = target_result.corr.sort_values()
    y = np.arange(len(c))
    ax.barh(y, c.values, color="#2b8cbe")
    ax.set_yticks(y, list(c.index))
    ax.set_xlim(0, 1)
    ax.set_xlabel("|corr| between rotated factor and its reference")
    ax.set_title("how much of each reference factor lives in the subspace?")
    for yi, v in zip(y, c.values):
        ax.text(v + 0.02, yi, f"{v:.2f}", va="center", fontsize=9)
    return ax
