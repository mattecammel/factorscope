# factorscope

[![CI](https://github.com/mattecammel/factorscope/actions/workflows/ci.yml/badge.svg)](https://github.com/mattecammel/factorscope/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Interpret, label, and trust the factors from a statistical (PCA) risk model.**

A statistical risk model runs PCA on asset returns and reports "factors". But PCA only
recovers the factor *subspace* — the factors themselves are scrambled by an arbitrary
rotation, so "Factor 3" is an uninterpretable mix of real economic drivers, and it can mean
something different every time you refit. `factorscope` rotates that subspace using the
factors' *time structure* (SOBI), names the result against known style/macro factors, and **tells you when the rotation is trustworthy and when it isn't.**

```python
from factorscope import FactorModel

fm = FactorModel(n_factors=5).fit(returns_df) # ragged panels (NaNs) are fine

fm.trust_report() # is the rotation identifiable, or should you not believe the labels?
fm.label() # -> F1 ~ +0.56*HML -0.41*SMB, F2 ~ +0.70*SMB -0.21*HML, ...
fm.rotate_to(ff) # -> an interpretable basis: SMB |corr| 0.90, HML 0.79, ...
fm.stability_report() # do the factors persist across refits, or churn?
```

**See it on real data:** [`examples/case_study.ipynb`](examples/case_study.ipynb) runs the whole
pipeline on Fama-French portfolios — the guardrail catching a market-dominated panel, blind
rotation returning *blends*, `rotate_to()` recovering size (0.90) and value (0.79), and a
negative control where the tool correctly finds **nothing**.

## Why it exists

`sklearn.decomposition.PCA` is too low-level (no NaNs, no interpretation, no idea if the
rotation is meaningful). Commercial risk models (Barra, Axioma) solve this but are closed and
cost six figures. `factorscope` fills the gap with an honest, sklearn-style API.

## What's in it

| Feature | Method | What it does |
|---|---|---|
| **Factor labeling** | `.label()` | Regress each factor on known style/macro factors → named factors + R² |
| **Interpretable basis** | `.rotate_to()` | Procrustes rotation onto reference factors → *named* axes, without changing the fitted model |
| **Missing-data PCA** | `missing="em"` | EM-PCA builds the subspace from ragged return panels |
| **Trust guardrail** | `.trust_report()` | Regime + identifiability margin → *should you believe the labels?* |
| **Stability monitor** | `.stability_report()` | Rolling-refit persistence of factor identity, SOBI vs PCA |
| **Cross-fit alignment** | `.align_to()` | Sign/permutation match factors across refits or universes |
| **Factor selection** | `n_factors=None` | Eigen-gaps + identifiability noise floor pick the factor count |

## The honest part

Time-structure rotation is **not** magic, and this library says so:

- On a **raw, market-dominated** panel, the market owns ~70% of the variance, PCA is already
  identified, and rotating only adds noise. `factorscope` detects this (`neutralize="auto"`)
  and removes the market first — turning the failure case into the success case.
- **Blind rotation does not hand you named factors.** On real data SOBI returns *blends* — two
  factors that are both size/value mixtures, not one "size" and one "value". That's a genuine
  limitation of the method (SOBI optimizes for *distinct dynamics*, which is not the same thing
  as *economic interpretability*), and the case study shows it rather than hiding it. If you want
  names, use `.rotate_to()`: an orthogonal Procrustes rotation onto reference factors that is
  provably a **change of basis, not a change of model** (the `S Bᵀ` reconstruction is unchanged
  to machine precision).
- On a panel that **doesn't contain** the factors (industry portfolios), it correctly recovers
  *nothing* rather than hallucinating — the negative control in the case study.
- If the factor dynamics are too weak to separate, `.trust_report()` says **"do not trust
  the labels"** instead of silently returning garbage. The "too weak" bar is not a magic
  constant: it is the 95th percentile of the margin you'd get by rotating **pure noise**,
  estimated by simulation. Use `.trust_report(null="block")` for a heteroskedasticity-robust
  bar — a block bootstrap that carries the data's own volatility clustering (which inflates
  the real noise floor on daily returns) into the threshold.

That guardrail — knowing which regime you're in — is the point of the package.

## Install

Not yet on PyPI — until the first release, install from source:

```bash
git clone https://github.com/mattecammel/factorscope
pip install -e ./factorscope # add "[plot]" for plots, "[dev]" to run the tests
```

Core dependencies are numpy, scipy and pandas; the `[plot]` extra adds matplotlib for the
diagnostic plots.

## License

MIT
