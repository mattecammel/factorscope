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
fm.stability_report() # do the factors persist across refits, or churn?
```

## Why it exists

`sklearn.decomposition.PCA` is too low-level (no NaNs, no interpretation, no idea if the
rotation is meaningful). Commercial risk models (Barra, Axioma) solve this but are closed and
cost six figures. `factorscope` fills the gap with an honest, sklearn-style API.

## What's in it

| Feature | Method | What it does |
|---|---|---|
| **Factor labeling** | `.label()` | Regress each factor on known style/macro factors → named factors + R² |
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
- On a **market-neutral** panel with real style structure, SOBI recovers the named factors
  better than raw PCA (see [`examples/fama_french_demo.ipynb`](examples/fama_french_demo.ipynb),
  a full walkthrough on real Fama-French data). On a panel that doesn't contain the factors,
  it correctly recovers *nothing* rather than hallucinating.
- If the factor dynamics are too weak to separate, `.trust_report()` says **"do not trust
  the labels"** instead of silently returning garbage. The "too weak" bar is not a magic
  constant: it is the 95th percentile of the margin you'd get by rotating **pure noise**,
  estimated by simulation. Use `.trust_report(null="block")` for a heteroskedasticity-robust
  bar — a block bootstrap that carries the data's own volatility clustering (which inflates
  the real noise floor on daily returns) into the threshold.

That guardrail — knowing which regime you're in — is the point of the package.

## Install

```bash
pip install factorscope # core: numpy, scipy, pandas
pip install factorscope[plot] # + matplotlib for diagnostic plots
```

Not yet on PyPI — until the first release, install from source:

```bash
git clone https://github.com/mattecammel/factorscope
pip install -e ./factorscope # add "[plot]" for plots, "[dev]" to run the tests
```

## License

MIT
