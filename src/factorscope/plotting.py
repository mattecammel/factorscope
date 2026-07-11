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
