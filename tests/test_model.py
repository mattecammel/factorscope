import numpy as np
import pandas as pd
import pytest

from factorscope import FactorModel, suggest_n_factors, em_pca


def _returns(T=1500, N=40, K=4, seed=1, market=False):
    rng = np.random.default_rng(seed)
    phi = np.array([0.0, 0.4, -0.3, 0.6])[:K]
    F = np.zeros((T, K))
    innov = rng.standard_normal((T, K))
    for k in range(K):
        for t in range(1, T):
            F[t, k] = phi[k] * F[t - 1, k] + np.sqrt(1 - phi[k] ** 2) * innov[t, k]
    F = (F - F.mean(0)) / F.std(0)
    B = rng.standard_normal((N, K)) / np.sqrt(K)
    X = F @ B.T + 0.5 * rng.standard_normal((T, N))
    if market:
        X += 3.0 * rng.standard_normal((T, 1))
    idx = pd.date_range("2015-01-01", periods=T, freq="B")
    return pd.DataFrame(X, index=idx, columns=[f"A{i}" for i in range(N)]), F


def test_fit_transform_and_shapes():
    df, _ = _returns()
    fm = FactorModel(n_factors=4).fit(df)
    assert fm.factors_.shape == (len(df), 4)
    assert fm.loadings_.shape == (df.shape[1], 4)
    out = fm.transform(df.iloc[:100])
    assert out.shape == (100, 4)


def test_handles_missing():
    df, _ = _returns()
    d = df.copy()
    rng = np.random.default_rng(0)
    d = d.mask(rng.random(d.shape) < 0.15)
    fm = FactorModel(n_factors=4, missing="em").fit(d)
    assert not fm.factors_.isna().any().any()


def test_trust_report_flags_dominated_market():
    df, _ = _returns(market=True)
    fm = FactorModel(n_factors=4, neutralize="auto").fit(df)
    tr = fm.trust_report()
    assert fm.neutralized_ is True
    assert tr.regime in ("flat", "dominated", "degenerate")


def test_auto_selection():
    df, _ = _returns()
    fm = FactorModel(n_factors=None).fit(df)
    assert 1 <= fm.n_factors_ <= 15
    assert fm.selection_ is not None


def test_stability_and_align():
    df, _ = _returns()
    fm = FactorModel(n_factors=4).fit(df)
    st = fm.stability_report(window=500, step=250)
    assert np.isfinite(st.sobi_mean)
    other = FactorModel(n_factors=4).fit(df)
    al = fm.align_to(other)
    assert al.mean_corr > 0.9


def test_label_synthetic_reference():
    df, F = _returns()
    ref = pd.DataFrame(F, index=df.index, columns=["r0", "r1", "r2", "r3"])
    fm = FactorModel(n_factors=4).fit(df)
    lab = fm.label(reference=ref)
    assert (lab.r2 > 0.3).any()
