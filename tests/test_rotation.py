import numpy as np

from factorscope.rotation import sobi, amuse
from factorscope.alignment import matched_corr


def _panel(T=2000, seed=0):
    rng = np.random.default_rng(seed)
    phi = np.array([0.0, 0.4, -0.3, 0.6])
    K = len(phi)
    F = np.zeros((T, K))
    innov = rng.standard_normal((T, K))
    for k in range(K):
        for t in range(1, T):
            F[t, k] = phi[k] * F[t - 1, k] + np.sqrt(1 - phi[k] ** 2) * innov[t, k]
    F = (F - F.mean(0)) / F.std(0)
    Q = np.linalg.qr(rng.standard_normal((K, K)))[0]
    return F, F @ Q


def test_sobi_recovers_rotation():
    F, X = _panel()
    S, W, sig = sobi(X)
    assert matched_corr(S, F) > 0.9


def test_amuse_runs_and_helps():
    F, X = _panel()
    S, _, _ = amuse(X, lag=1)
    assert matched_corr(S, F) > 0.7


def test_signature_shape():
    F, X = _panel()
    _, _, sig = sobi(X, lags=(1, 2, 3))
    assert sig.shape == (4, 3)
