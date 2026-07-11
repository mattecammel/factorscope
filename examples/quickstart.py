import numpy as np
import pandas as pd

from factorscope import FactorModel


def make_returns(T=1500, N=50, K=4, seed=0):
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
    idx = pd.date_range("2015-01-01", periods=T, freq="B")
    return pd.DataFrame(X, index=idx, columns=[f"A{i}" for i in range(N)]), F


if __name__ == "__main__":
    returns, true_F = make_returns()
    fm = FactorModel(n_factors=None).fit(returns)
    print(f"selected {fm.n_factors_} factors\n")
    print(fm.trust_report(), "\n")

    ref = pd.DataFrame(true_F, index=returns.index,
                       columns=["trueF0", "trueF1", "trueF2", "trueF3"])
    print(fm.label(reference=ref), "\n")
    print(fm.stability_report(window=500, step=250))
