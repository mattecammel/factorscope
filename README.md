# factorscope

[![CI](https://github.com/mattecammel/factorscope/actions/workflows/ci.yml/badge.svg)](https://github.com/mattecammel/factorscope/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Identification, interpretation, and diagnostics for statistical (PCA) risk factor models.**

`factorscope` addresses the rotational indeterminacy of latent factor models: it resolves the
rotation using the factors' temporal structure, interprets the resulting axes against known style
and macroeconomic factors, and reports quantitative diagnostics for whether the resolved rotation
is identified in the first place.

```python
from factorscope import FactorModel

fm = FactorModel(n_factors=5).fit(returns_df)   # ragged panels (NaNs) are supported

fm.trust_report()      # is the rotation identified? should the labels be believed?
fm.label()             # F1 ~ +0.56*HML -0.41*SMB, F2 ~ +0.70*SMB -0.21*HML, ...
fm.rotate_to(ff)       # interpretable basis: |corr| = 0.90 (SMB), 0.79 (HML), ...
fm.stability_report()  # does factor identity persist across refits?
```

An empirical application to Fama–French portfolio data is developed in
[`examples/case_study.ipynb`](examples/case_study.ipynb).

---

## 1. Motivation

A statistical risk model applies principal component analysis to a panel of asset returns
`X` (T periods × N assets) and reports the leading components as risk factors. The reported
factors are then routinely interpreted — as "the value factor", "the sector factor", and so on.

This interpretation is not licensed by the estimator. PCA identifies the factor *subspace*, not a
basis for it. For any orthogonal matrix `Q`,

```
X ≈ S Bᵀ = (S Q)(B Q)ᵀ
```

so the rotated factors `S Q` with rotated loadings `B Q` reproduce the returns exactly as well as
the original pair. The fit criterion is invariant under the orthogonal group: every rotation of the
subspace is an equally good solution, and PCA singles one out by a purely statistical convention —
successive maximal variance — that has no economic content. Two consequences follow.

- **The reported factors are mixtures.** "Factor 3" is an unidentified linear combination of
  whatever economic drivers span the subspace. Its loadings are interpretable only up to that
  unknown mixing.
- **Factor identity is unstable.** Where eigenvalues are close, the sample eigenvectors are
  poorly determined, so the same "Factor 3" can denote a different combination after each refit.

Note carefully what is *not* affected. The subspace — and therefore the variance forecast that a
risk model derives from it — is untouched by the ambiguity, since every rotation spans the same
space. Rotational indeterminacy is a problem of *orientation*, and hence of *interpretation*, not
of fit. The diagnostics in this package are phrased accordingly: a failed identification check
invalidates the factor *labels*, not the risk model.

### Resolving the indeterminacy

Two additional sources of information can pin down the rotation, and `factorscope` implements both;
a third component tests whether the rotation so obtained is credible in the first place.

**(i) Temporal structure (blind separation).** Second-Order Blind Identification
(SOBI; Belouchrani et al., 1997) assumes the latent sources have
*distinct autocorrelation profiles*. Under that assumption, the rotation that simultaneously
diagonalizes a set of lagged autocovariance matrices `C_L = Cov(s_t, s_{t-L})` is identified up to
sign and permutation. Estimation proceeds by approximate joint diagonalization of the stacked
symmetrized autocovariances (Jacobi sweeps, following Cardoso and Souloumiac). This requires no
data beyond the returns themselves — hence *blind*.

**(ii) External reference factors (target rotation).** Given observed reference series (Fama–French
factors, macroeconomic series), an orthogonal Procrustes rotation
`Q = argmin_{QᵀQ=I} ‖S Q − R‖_F` aligns the estimated subspace with the references. `factorscope`
computes it as the orthogonal matrix nearest — in the polar-decomposition sense — to the
unconstrained least-squares map `A = argmin_A ‖S A − R‖`: with `A = U Σ Vᵀ`, the rotation is
`Q = U Vᵀ`. (Since the estimated factors are standardized and mutually uncorrelated, `SᵀS ≈ (T−1)I`
and this coincides with the classical `SᵀR` form of the Procrustes solution.) Because `Q` is
orthogonal, the operation is a *change of basis, not a change of model*: the reconstruction
`(S Q)(B Q)ᵀ = S Bᵀ` is preserved exactly.

**(iii) Diagnostics.** Both routes have failure modes that are silent unless tested for, and the
diagnostic layer that tests for them is the substantive contribution of this package. It is
described in Section 3.

---

## 2. Interface

| Component | Method | Description |
|---|---|---|
| Factor labeling | `.label()` | Regression of each estimated factor on known style/macro references; returns coefficients, per-factor R², and a name string |
| Interpretable basis | `.rotate_to()` | Orthogonal Procrustes rotation onto reference factors; yields named axes without altering the fitted model |
| Missing-data PCA | `missing="em"` | EM-PCA recovers the subspace from ragged (unbalanced) return panels |
| Identification diagnostic | `.trust_report()` | Spectral regime classification and identifiability margin against a simulated noise floor |
| Stability diagnostic | `.stability_report()` | Rolling-refit persistence of factor identity, SOBI compared against unrotated PCA |
| Cross-fit alignment | `.align_to()` | Sign and permutation matching of factors across refits or universes (Hungarian assignment) |
| Factor-count selection | `n_factors=None` | Minimum of an eigenvalue-gap criterion and an identifiability noise-floor criterion |

The API follows scikit-learn conventions (`fit` / `transform`, trailing-underscore fitted
attributes). Core dependencies are NumPy, SciPy, and pandas.

---

## 3. Scope and limitations of the method

The diagnostics exist because the rotation methods above hold only under conditions that real
return panels frequently violate. Each condition is testable, and `factorscope` tests it rather
than assuming it.

**Blind rotation is only meaningful when the spectrum is flat.** On raw equity returns the market
factor typically accounts for on the order of 70% of total variance. When one eigenvalue is that
dominant, the leading eigenvector is already well separated from the rest and the rotational
ambiguity it was meant to resolve is largely absent; applying a rotation in that regime introduces
estimation noise without recovering identification. `detect_regime` classifies the spectrum as
*dominated*, *degenerate*, or *flat* (the top-PC variance share is compared against a 0.35
threshold), and `neutralize="auto"` cross-sectionally demeans the panel — removing the
equal-weighted market — when the dominated regime is detected, restoring the conditions under
which rotation is informative.

**Blind rotation recovers distinct dynamics, not economic interpretability.** SOBI's objective is
to find the rotation whose sources have maximally distinct autocorrelation signatures. This is not
the same objective as finding the rotation aligned with economically meaningful axes, and no
theorem connects the two: the identification result holds under the *assumption* that the true
economic factors have distinguishable temporal dynamics. Empirically that assumption holds only
partially. On Fama–French portfolios, SOBI returns factors that are *blends* of size and value
rather than one size factor and one value factor (see the case study, §2). The subspace is
correctly estimated; the axes within it are not economically aligned. Where named axes are the
objective, target rotation (`rotate_to`) is the appropriate instrument — with the caveat that it is
interpretation by projection, not blind discovery, and it can only find structure that is already
present in the estimated subspace.

**Identification can fail, and failure is detectable.** If two factors have similar autocorrelation
signatures, the joint-diagonalization problem is near-degenerate and the estimated rotation is
dominated by sampling noise. The `identifiability_margin` measures this as the minimum pairwise
Euclidean distance between rows of the signature matrix, scaled by `√T` — i.e., the separation of
the two most similar factors, expressed in standard errors. Because that statistic has no absolute
scale, it is calibrated against a **simulated noise floor**: the same pipeline is run on data with
no factor structure, and the threshold is the 95th percentile of the resulting margin distribution
(200 replications). A margin below the floor means the estimated rotation is statistically
indistinguishable from one fitted to noise, and `trust_report()` says so explicitly rather than
returning an unqualified label.

Daily returns exhibit volatility clustering, and conditional heteroskedasticity inflates the
apparent temporal structure that SOBI keys on — so a Gaussian null sets the bar too low on real
data. `trust_report(null="block")` therefore offers a heteroskedasticity-robust alternative: a
block bootstrap of the estimated factors that destroys cross-factor and long-range structure while
preserving each block's own volatility dynamics. On simulated GARCH data this raises the threshold
relative to the Gaussian null without raising it so far that genuine factor structure fails to
clear it — both properties are asserted in the test suite.

**Absence is reported as absence.** Applied to a panel that does not contain the reference factors
— 49 industry portfolios, which are not sorted on any style characteristic — the target rotation
returns markedly weaker correlations rather than manufacturing an alignment to satisfy the request.
This negative control is what makes the corresponding positive result on size/book-to-market
portfolios informative rather than mechanical.

---

## 4. Empirical illustration

[`examples/case_study.ipynb`](examples/case_study.ipynb) applies the full pipeline to Fama–French
daily data from 1970 and is organized as an argument rather than a tour:

1. the identification diagnostic declines to certify a raw, market-dominated panel, and the
   verdict reverses once the market is neutralized;
2. blind rotation returns size/value *blends*, illustrating the objective mismatch described above;
3. target rotation recovers size (|corr| = 0.90) and value (0.79) on portfolios sorted on exactly
   those characteristics, and the rotation is verified numerically to leave `S Bᵀ` unchanged to
   machine precision;
4. the identical pipeline on industry portfolios recovers the style factors far more weakly — the
   negative control;
5. factor identity is tracked across rolling refits.

`examples/fama_french_demo.ipynb` gives a shorter walkthrough, and `examples/quickstart.py`
demonstrates the API on synthetic AR(1) factors where ground truth is known.

---

## 5. Installation

Not yet released on PyPI; install from source:

```bash
git clone https://github.com/mattecammel/factorscope
pip install -e ./factorscope        # "[plot]" adds matplotlib diagnostics; "[dev]" adds pytest
```

## 6. References

- Belouchrani, A., Abed-Meraim, K., Cardoso, J.-F., and Moulines, E. (1997). A blind source
  separation technique using second-order statistics. *IEEE Transactions on Signal Processing*,
  45(2), 434–444.
- Cardoso, J.-F. and Souloumiac, A. (1996). Jacobi angles for simultaneous diagonalization.
  *SIAM Journal on Matrix Analysis and Applications*, 17(1), 161–164.
- Schönemann, P. H. (1966). A generalized solution of the orthogonal Procrustes problem.
  *Psychometrika*, 31(1), 1–10.
- Fama, E. F. and French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial
  Economics*, 116(1), 1–22.

## 7. Citation

If this work or its calibration procedure is useful to you:

```bibtex
@software{camellini2026factorscope,
  author  = {Camellini, Matteo},
  title   = {factorscope: Identification, Interpretation, and Diagnostics for
             Statistical Risk Factor Models},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/mattecammel/factorscope}
}
```

## License

MIT 

---

<sub>**Topics:** Statistical risk model · PCA factor model · rotational indeterminacy · factor identification ·
factor interpretation · blind source separation · SOBI (second-order blind identification) ·
AMUSE · joint diagonalization · orthogonal Procrustes rotation · target rotation · factor labeling ·
factor rotation · latent factor model · principal component analysis · EM-PCA · missing data ·
ragged panel · Fama–French factors · size and value factors · style factors · equity risk model ·
market neutralization · eigenvalue spectrum · factor selection · scree criterion · noise floor ·
block bootstrap · surrogate data · heteroskedasticity · volatility clustering · factor stability ·
rolling refit · Hungarian algorithm · quantitative finance · asset pricing · portfolio risk ·
Python · NumPy · SciPy · pandas · scikit-learn-style API</sub>
