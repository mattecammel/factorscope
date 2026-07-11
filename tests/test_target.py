import numpy as np
import pandas as pd
import pytest

from factorscope import FactorModel, target_rotation
from factorscope.reference import _parse_daily


def _panel(T=1500, N=25, K=4, seed=0):
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
    idx = pd.date_range("2010-01-01", periods=T, freq="B")
    ref = pd.DataFrame(F, index=idx, columns=["r0", "r1", "r2", "r3"])
    return pd.DataFrame(X, index=idx, columns=[f"A{i}" for i in range(N)]), ref


def test_rotation_is_orthogonal_and_preserves_subspace():
    df, ref = _panel()
    fm = FactorModel(n_factors=4, neutralize=False).fit(df)
    tr = fm.rotate_to(ref)
    Q = tr.rotation
    assert np.abs(Q.T @ Q - np.eye(4)).max() < 1e-9
    S0 = ((fm.factors_ - fm.factors_.mean()) / fm.factors_.std()).values
    before = S0 @ fm.loadings_.values.T
    after = tr.factors.values @ tr.loadings.values.T
    assert np.abs(after - before).max() < 1e-8


def test_rotation_recovers_known_factors():
    df, ref = _panel()
    fm = FactorModel(n_factors=4, neutralize=False).fit(df)
    tr = fm.rotate_to(ref)
    assert list(tr.factors.columns) == ["r0", "r1", "r2", "r3"]
    assert (tr.corr > 0.8).all()


def test_rotation_signs_are_positive_against_reference():
    df, ref = _panel()
    fm = FactorModel(n_factors=4, neutralize=False).fit(df)
    tr = fm.rotate_to(ref)
    for name in tr.corr.index:
        c = np.corrcoef(tr.factors[name].values, ref[name].values)[0, 1]
        assert c > 0


def test_more_references_than_factors_drops_worst():
    df, ref = _panel()
    noise = pd.DataFrame(
        np.random.default_rng(9).standard_normal((len(df), 1)),
        index=df.index, columns=["junk"])
    ref2 = ref.join(noise)
    fm = FactorModel(n_factors=4, neutralize=False).fit(df)
    with pytest.warns(UserWarning, match="only 4 factors"):
        tr = fm.rotate_to(ref2)
    assert tr.dropped == ["junk"]
    assert "junk" not in tr.factors.columns


def test_residual_factors_when_fewer_references():
    df, ref = _panel()
    fm = FactorModel(n_factors=4, neutralize=False).fit(df)
    tr = fm.rotate_to(ref[["r0", "r1"]])
    assert list(tr.factors.columns) == ["r0", "r1", "Resid1", "Resid2"]
    assert np.abs(tr.rotation.T @ tr.rotation - np.eye(4)).max() < 1e-9


def test_rotate_to_nonoverlapping_index_raises():
    df, ref = _panel()
    fm = FactorModel(n_factors=4, neutralize=False).fit(df)
    bad = ref.reset_index(drop=True)
    with pytest.raises(ValueError):
        fm.rotate_to(bad)


def test_parse_daily_tolerates_trailing_comma():
    # the Ken French momentum CSV ends every row with a comma; a strict column count
    # rejected every row and silently produced an empty frame
    text = "\n".join([",Mom,", "19261103,0.35,", "19261104,-0.61,"])
    df = _parse_daily(text, 1)
    assert len(df) == 2
    assert df.iloc[0, 0] == pytest.approx(0.35)


def test_parse_daily_still_parses_plain_rows():
    text = "\n".join([",Mkt-RF,SMB", "19700102,0.10,0.20", "19700105,-0.30,0.40"])
    df = _parse_daily(text, 2)
    assert df.shape == (2, 2)
    assert df.iloc[1, 1] == pytest.approx(0.40)
