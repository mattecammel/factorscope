from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .decomposition import _nanmean, em_pca, market_neutralize, pca_scores, project_scores
from .rotation import DEFAULT_LAGS, amuse, sobi
from .identifiability import build_trust_report, detect_regime
from .labeling import label_factors
from .target import target_rotation
from .stability import rolling_stability
from .alignment import align_factors
from .selection import suggest_n_factors


class FactorModel:
    def __init__(self, n_factors: Optional[int] = None, *, rotation="sobi",
                 lags=DEFAULT_LAGS, neutralize="auto", missing="em"):
        if rotation not in ("sobi", "amuse", "none"):
            raise ValueError("rotation must be 'sobi', 'amuse', or 'none'")
        if not (neutralize == "auto" or neutralize is True or neutralize is False):
            raise ValueError("neutralize must be 'auto', True, or False; "
                             f"got {neutralize!r}")
        if missing not in ("em", "mean"):
            raise ValueError(f"missing must be 'em' or 'mean'; got {missing!r}")
        self.n_factors = n_factors
        self.rotation = rotation
        self.lags = tuple(lags)
        self.neutralize = neutralize
        self.missing = missing

    def fit(self, returns):
        df = pd.DataFrame(returns)
        self.assets_ = list(df.columns)
        self.index_ = df.index
        X = df.values.astype(float)
        if X.shape[1] < 2:
            raise ValueError(f"need at least 2 assets to fit a factor model; "
                             f"got {X.shape[1]}")
        self._returns = X
        ragged = np.isnan(X).any()

        kprov = int(min(10, X.shape[1] - 1))
        if self.missing == "em" and ragged:
            _, _, _, filled = em_pca(X, kprov)
        else:
            filled = np.where(np.isnan(X), _nanmean(X, axis=0), X)
            filled = np.nan_to_num(filled, nan=0.0)

        if self.neutralize == "auto":
            neutralized = detect_regime(filled, kprov) == "dominated"
        else:
            neutralized = bool(self.neutralize)
        self.neutralized_ = neutralized
        Xu = market_neutralize(filled) if neutralized else filled

        if self.n_factors is None:
            self.selection_ = suggest_n_factors(filled, neutralize=neutralized,
                                                lags=self.lags)
            k = self.selection_.n_suggested
        else:
            self.selection_ = None
            k = int(self.n_factors)
        max_k = min(X.shape[0] - 1, X.shape[1] - 1)
        if not 1 <= k <= max_k:
            raise ValueError(f"n_factors ({k}) must be in [1, {max_k}] for a "
                             f"{X.shape[0]}x{X.shape[1]} (samples x assets) panel")
        self.n_factors_ = k

        P = pca_scores(Xu, k)
        if self.rotation == "sobi":
            S, _, sig = sobi(P, self.lags)
        elif self.rotation == "amuse":
            S, _, sig = amuse(P, self.lags[0])
        else:
            S, sig = P, None
        self.mean_ = Xu.mean(0)
        B = np.linalg.lstsq(S, Xu - self.mean_, rcond=None)[0].T
        order = np.argsort(-(B ** 2).sum(0))
        S, B = S[:, order], B[:, order]
        if sig is not None:
            sig = sig[order]
        dom = np.argmax(np.abs(B), axis=0)
        signs = np.sign(B[dom, np.arange(B.shape[1])])
        signs[signs == 0] = 1.0
        S, B = S * signs, B * signs

        self._S = S
        self._Xu = Xu
        self._T = len(X)
        self.signature_ = sig
        cols = [f"F{i+1}" for i in range(k)]
        self.factors_ = pd.DataFrame(S, index=self.index_, columns=cols)
        self.loadings_ = pd.DataFrame(B, index=self.assets_, columns=cols)
        return self

    def transform(self, returns):
        self._check_fitted()
        df = self._align_assets(pd.DataFrame(returns))
        X = df.values.astype(float)
        if self.neutralized_:
            X = market_neutralize(X)
        S = project_scores(X, self.loadings_.values, self.mean_)
        return pd.DataFrame(S, index=df.index, columns=self.factors_.columns)

    def fit_transform(self, returns):
        return self.fit(returns).factors_

    def trust_report(self, *, null="gaussian"):
        self._check_fitted()
        return build_trust_report(self._Xu, self._S, self._T,
                                  neutralized=self.neutralized_,
                                  n_components=self.n_factors_, lags=self.lags,
                                  null=null, rotation=self.rotation)

    def label(self, reference=None, top_k=2):
        self._check_fitted()
        if reference is None:
            from .reference import load_reference_factors
            reference = load_reference_factors("ff5")
        return label_factors(self.factors_, reference, top_k=top_k)

    def rotate_to(self, reference=None):
        self._check_fitted()
        if reference is None:
            from .reference import load_reference_factors
            reference = load_reference_factors("ff5")
        return target_rotation(self.factors_, self.loadings_, reference)

    def stability_report(self, returns=None, *, window=750, step=125):
        self._check_fitted()
        X = self._returns if returns is None else pd.DataFrame(returns).values.astype(float)
        X = np.where(np.isnan(X), _nanmean(X, axis=0), X)
        X = np.nan_to_num(X, nan=0.0)
        return rolling_stability(X, self.n_factors_, window=window, step=step,
                                 neutralize=self.neutralized_, lags=self.lags)

    def align_to(self, other):
        self._check_fitted()
        idx = self.factors_.index.intersection(other.factors_.index)
        return align_factors(self.factors_.loc[idx].values,
                             other.factors_.loc[idx].values)

    def _align_assets(self, df):
        if list(df.columns) == self.assets_:
            return df
        if len(df.columns.intersection(self.assets_)):
            return df.reindex(columns=self.assets_)
        if df.shape[1] == len(self.assets_):
            return df
        raise ValueError(
            f"cannot align input to the fitted universe: {df.shape[1]} columns share no "
            f"names with the {len(self.assets_)} fitted assets. Pass a DataFrame with "
            "the fit-time column names (a subset is fine), or an array with one column "
            "per fitted asset.")

    def _check_fitted(self):
        if not hasattr(self, "factors_"):
            raise RuntimeError("call fit() first")
