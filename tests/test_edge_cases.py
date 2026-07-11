import numpy as np
import pandas as pd
import pytest

from factorscope import FactorModel
from factorscope.alignment import align_factors, matched_corr
from factorscope.identifiability import identifiability_margin, per_factor_snr
from factorscope.rotation import sobi, amuse, usable_lags


def _panel(T=800, N=12, K=3, seed=0, market=False):
    rng = np.random.default_rng(seed)
    phi = np.array([0.0, 0.4, -0.3, 0.6])[:K]
    F = np.zeros((T, K))
    e = rng.standard_normal((T, K))
    for k in range(K):
        for t in range(1, T):
            F[t, k] = phi[k] * F[t - 1, k] + np.sqrt(1 - phi[k] ** 2) * e[t, k]
    F = (F - F.mean(0)) / F.std(0)
    B = rng.standard_normal((N, K)) / np.sqrt(K)
    X = F @ B.T + 0.5 * rng.standard_normal((T, N))
    if market:
        X += 3.0 * rng.standard_normal((T, 1))
    idx = pd.date_range("2010-01-01", periods=T, freq="B")
    return pd.DataFrame(X, index=idx, columns=[f"A{i}" for i in range(N)]), F


def test_amuse_path_end_to_end():
    df, _ = _panel()
    fm = FactorModel(n_factors=3, rotation="amuse").fit(df)
    assert fm.factors_.shape == (len(df), 3)
    assert np.isfinite(fm.factors_.values).all()
    assert np.isfinite(fm.trust_report().margin) or fm.trust_report().margin == float("inf")


def test_rotation_none_is_pca():
    df, _ = _panel()
    fm = FactorModel(n_factors=3, rotation="none").fit(df)
    assert fm.factors_.shape == (len(df), 3)
    al = fm.align_to(fm)
    assert al.mean_corr > 0.999


def test_invalid_rotation_rejected():
    with pytest.raises(ValueError):
        FactorModel(rotation="varimax")


def test_invalid_neutralize_rejected():
    for bad in ("none", "no", "yes", "true"):
        with pytest.raises(ValueError):
            FactorModel(neutralize=bad)
    for ok in ("auto", True, False):
        FactorModel(neutralize=ok)


def test_invalid_missing_rejected():
    with pytest.raises(ValueError):
        FactorModel(missing="drop")
    with pytest.raises(ValueError):
        FactorModel(missing="impute")
    FactorModel(missing="em")
    FactorModel(missing="mean")


def test_missing_mean_path_runs():
    df, _ = _panel()
    d = df.copy()
    rng = np.random.default_rng(0)
    d = d.mask(rng.random(d.shape) < 0.1)
    fm = FactorModel(n_factors=3, missing="mean").fit(d)
    assert np.isfinite(fm.factors_.values).all()


def test_factors_ordered_by_variance_explained():
    df, _ = _panel(N=20)
    for rot in ("sobi", "amuse", "none"):
        fm = FactorModel(n_factors=4, rotation=rot, neutralize=False).fit(df)
        col_var = (fm.loadings_.values ** 2).sum(0)
        assert np.all(np.diff(col_var) <= 1e-9), f"{rot}: F1..Fk not variance-ordered"


def test_all_nan_column():
    df, _ = _panel()
    df = df.copy()
    df["A5"] = np.nan
    fm = FactorModel(n_factors=3, missing="em").fit(df)
    assert np.isfinite(fm.factors_.values).all()
    assert np.isfinite(fm.loadings_.values).all()


def test_all_nan_row_in_transform():
    df, _ = _panel()
    fm = FactorModel(n_factors=3).fit(df)
    probe = df.iloc[:5].copy()
    probe.iloc[0, :] = np.nan
    out = fm.transform(probe)
    assert out.iloc[0].isna().all()
    assert out.iloc[1:].notna().all().all()


def test_constant_column_does_not_crash():
    df, _ = _panel()
    df = df.copy()
    df["A2"] = 0.0
    fm = FactorModel(n_factors=3).fit(df)
    assert np.isfinite(fm.factors_.values).all()


def test_single_factor_trust_report():
    df, _ = _panel()
    fm = FactorModel(n_factors=1).fit(df)
    tr = fm.trust_report()
    assert tr.margin == float("inf")
    assert tr.margin_ok
    assert np.isinf(per_factor_snr(fm._S, fm._T)).all()


def test_too_many_factors_rejected():
    df, _ = _panel(N=8)
    with pytest.raises(ValueError):
        FactorModel(n_factors=8).fit(df)
    with pytest.raises(ValueError):
        FactorModel(n_factors=0).fit(df)


def test_wide_short_panel_rank_bound():
    df = pd.DataFrame(np.random.default_rng(0).standard_normal((8, 50)))
    with pytest.raises(ValueError, match="samples x assets"):
        FactorModel(n_factors=10).fit(df)


def test_fit_is_deterministic():
    df, _ = _panel()
    a = FactorModel(n_factors=3).fit(df).factors_.values
    b = FactorModel(n_factors=3).fit(df).factors_.values
    assert np.array_equal(a, b)


def test_stability_window_too_small_raises():
    df, _ = _panel()
    fm = FactorModel(n_factors=4).fit(df)
    with pytest.raises(ValueError):
        fm.stability_report(window=4, step=2)


def test_matched_corr_nan_safe_with_constant_column():
    rng = np.random.default_rng(0)
    A = np.c_[rng.standard_normal((200, 2)), np.zeros(200)]
    val = matched_corr(A, A)
    assert np.isfinite(val)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        FactorModel().trust_report()


def test_single_asset_rejected():
    df = pd.DataFrame({"A": np.random.default_rng(0).standard_normal(200)})
    with pytest.raises(ValueError):
        FactorModel(n_factors=1).fit(df)


def test_align_recovers_permutation():
    df, _ = _panel()
    fm = FactorModel(n_factors=3, rotation="sobi").fit(df)
    F = fm.factors_.values
    perm = np.array([2, 0, 1])
    signs = np.array([-1.0, 1.0, -1.0])
    scrambled = F[:, perm] * signs
    al = align_factors(scrambled, F)
    assert al.mean_corr > 0.99
    assert list(al.permutation) == list(np.argsort(perm))


def test_usable_lags_drops_too_long():
    assert usable_lags(8, (1, 2, 3, 5, 8)) == (1, 2, 3, 5)
    assert usable_lags(1000, (1, 2, 3, 5, 8)) == (1, 2, 3, 5, 8)


def test_usable_lags_raises_when_too_short():
    with pytest.raises(ValueError):
        usable_lags(2, (1, 2, 3, 5, 8))


def test_sobi_short_panel_warns_but_stays_finite():
    rng = np.random.default_rng(0)
    P = rng.standard_normal((8, 4))
    with pytest.warns(UserWarning):
        S, _, sig = sobi(P)
    assert np.isfinite(S).all() and np.isfinite(sig).all()


def test_signature_never_nan_on_short_panel():
    rng = np.random.default_rng(1)
    S = rng.standard_normal((9, 3))
    assert np.isfinite(identifiability_margin(S, len(S)))


def test_model_raises_on_panel_too_short_for_lags():
    df, _ = _panel(T=6)
    with pytest.raises(ValueError):
        FactorModel(n_factors=3, lags=(5, 8)).fit(df)


def test_noise_floor_reaches_one_on_white_noise():
    from factorscope import suggest_n_factors
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1500, 20))
    sel = suggest_n_factors(X)
    assert sel.noise_floor_k == 1
    assert "floor 1" in sel.reason


def test_dominated_after_neutralize_is_not_trusted():
    from factorscope.identifiability import build_trust_report
    rng = np.random.default_rng(1)
    N, T = 20, 2000
    w = np.r_[np.ones(N // 2), -np.ones(N // 2)]
    Xd = rng.standard_normal((T, 1)) * 5 * w + rng.standard_normal((T, N)) * 0.1
    phi = [0.0, 0.5, -0.4, 0.7, 0.2]
    S = np.zeros((T, 5)); e = rng.standard_normal((T, 5))
    for k in range(5):
        for t in range(1, T):
            S[t, k] = phi[k] * S[t - 1, k] + np.sqrt(1 - phi[k] ** 2) * e[t, k]
    tr = build_trust_report(Xd, S, T, neutralized=True, n_components=5)
    assert tr.regime == "dominated"
    assert tr.verdict.startswith("Do NOT")
    assert any("dominated" in w for w in tr.warnings)


def test_align_factors_size_mismatch_raises():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((100, 3))
    B = rng.standard_normal((100, 4))
    with pytest.raises(ValueError):
        align_factors(A, B)


def test_stability_report_survives_all_nan_column():
    df, _ = _panel(T=1200, N=12)
    df = df.copy(); df["A7"] = np.nan
    fm = FactorModel(n_factors=3).fit(df)
    st = fm.stability_report(window=400, step=200)
    assert np.isfinite(st.sobi_mean)


def test_label_raises_on_nonoverlapping_index():
    df, F = _panel()
    fm = FactorModel(n_factors=3, rotation="none").fit(df)
    ref = pd.DataFrame(F[:, :3], columns=["a", "b", "c"])
    with pytest.raises(ValueError):
        fm.label(reference=ref)


def test_label_handles_nan_rows_in_reference():
    df, F = _panel()
    fm = FactorModel(n_factors=3, rotation="none").fit(df)
    ref = pd.DataFrame(F[:, :3], index=df.index, columns=["a", "b", "c"]).copy()
    ref.iloc[:50] = np.nan
    lab = fm.label(reference=ref)
    assert lab.r2.notna().all()


def test_transform_aligns_permuted_columns():
    df, _ = _panel()
    fm = FactorModel(n_factors=3).fit(df)
    base = fm.transform(df.iloc[:50])
    shuffled = fm.transform(df.iloc[:50, ::-1])
    assert np.allclose(base.values, shuffled.values, atol=1e-10)


def test_transform_missing_assets_by_name():
    df, _ = _panel()
    fm = FactorModel(n_factors=3).fit(df)
    out = fm.transform(df.iloc[:20].drop(columns=["A3", "A7"]))
    assert np.isfinite(out.values).all()


def test_transform_positional_fallback_and_width_check():
    df, _ = _panel()
    fm = FactorModel(n_factors=3).fit(df)
    out = fm.transform(df.iloc[:10].values)
    assert np.isfinite(out.values).all()
    with pytest.raises(ValueError):
        fm.transform(np.zeros((5, df.shape[1] + 3)))


def test_margin_null_quantile_properties():
    from factorscope.identifiability import margin_null_quantile
    thr = margin_null_quantile(1500, 4)
    assert 3.0 < thr < 8.0
    assert margin_null_quantile(1500, 1) == 0.0
    assert margin_null_quantile(1500, 4) == thr


def test_white_noise_margin_not_certified():
    from factorscope.identifiability import identifiability_margin, margin_null_quantile
    rng = np.random.default_rng(3)
    X = rng.standard_normal((1500, 4))
    S, _, _ = sobi(X)
    assert identifiability_margin(S, 1500) < margin_null_quantile(1500, 4)


def test_trust_report_carries_threshold():
    df, _ = _panel()
    tr = FactorModel(n_factors=3).fit(df).trust_report()
    assert np.isfinite(tr.margin_threshold) and tr.margin_threshold > 0
    assert tr.margin_ok == (tr.margin >= tr.margin_threshold)
    assert tr.null_model == "gaussian"


def _garch_white(T, K, seed, omega=0.05, alpha=0.1, beta=0.88):
    rng = np.random.default_rng(seed)
    X = np.empty((T, K))
    for k in range(K):
        sig2 = omega / (1 - alpha - beta)
        prev = 0.0
        for t in range(T):
            sig2 = omega + alpha * prev**2 + beta * sig2
            x = np.sqrt(sig2) * rng.standard_normal()
            X[t, k] = x
            prev = x
    return (X - X.mean(0)) / X.std(0)


def test_margin_null_quantile_block_properties():
    from factorscope.identifiability import margin_null_quantile_block
    S = _garch_white(1500, 4, seed=0)
    thr = margin_null_quantile_block(S)
    assert 3.0 < thr < 12.0
    assert margin_null_quantile_block(S[:, :1]) == 0.0
    assert margin_null_quantile_block(S) == thr


def test_block_null_is_more_conservative_under_heteroskedasticity():
    from factorscope.identifiability import (
        margin_null_quantile, margin_null_quantile_block)
    S = _garch_white(1500, 4, seed=1)
    g = margin_null_quantile(1500, 4)
    b = margin_null_quantile_block(S)
    assert b > g


def test_block_null_still_passes_real_signal():
    from factorscope.identifiability import (
        identifiability_margin, margin_null_quantile_block)
    df, _ = _panel(T=1500, N=20, K=3)
    fm = FactorModel(n_factors=3, neutralize=False).fit(df)
    S = fm._S
    assert identifiability_margin(S, len(df)) >= margin_null_quantile_block(S)


def test_trust_report_block_null_end_to_end():
    df, _ = _panel()
    tr = FactorModel(n_factors=3).fit(df).trust_report(null="block")
    assert tr.null_model == "block"
    assert np.isfinite(tr.margin_threshold) and tr.margin_threshold > 0
    assert tr.margin_ok == (tr.margin >= tr.margin_threshold)


def test_trust_report_rejects_bad_null():
    df, _ = _panel()
    with pytest.raises(ValueError):
        FactorModel(n_factors=3).fit(df).trust_report(null="bootstrap")


def test_plain_integer_index():
    df, _ = _panel()
    df = df.reset_index(drop=True)
    fm = FactorModel(n_factors=3).fit(df)
    assert list(fm.factors_.index) == list(range(len(df)))
