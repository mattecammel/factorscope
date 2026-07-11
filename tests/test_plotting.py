import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")
import matplotlib
matplotlib.use("Agg")

from factorscope import FactorModel
from factorscope.plotting import plot_labels, plot_stability


def _fitted():
    rng = np.random.default_rng(0)
    T = 1200
    phi = [0.0, 0.4, -0.3, 0.6]
    F = np.zeros((T, 4)); e = rng.standard_normal((T, 4))
    for k in range(4):
        for t in range(1, T):
            F[t, k] = phi[k] * F[t - 1, k] + np.sqrt(1 - phi[k] ** 2) * e[t, k]
    B = rng.standard_normal((25, 4)) / 2
    X = F @ B.T + 0.5 * rng.standard_normal((T, 25))
    idx = pd.date_range("2015-01-01", periods=T, freq="B")
    ref = pd.DataFrame(F, index=idx, columns=["mkt", "size", "val", "mom"])
    fm = FactorModel(n_factors=4, neutralize=False).fit(pd.DataFrame(X, index=idx))
    return fm, ref


def test_plot_labels_runs():
    fm, ref = _fitted()
    ax = plot_labels(fm.label(reference=ref))
    assert ax is not None


def test_plot_stability_runs():
    fm, _ = _fitted()
    ax = plot_stability(fm.stability_report(window=400, step=200))
    assert ax is not None
