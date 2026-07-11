from __future__ import annotations

from .model import FactorModel
from .rotation import sobi, amuse, usable_lags
from .decomposition import em_pca, market_neutralize, pca_scores
from .identifiability import (
    identifiability_margin, margin_null_quantile, margin_null_quantile_block,
    detect_regime, top_pc_share,
)
from .labeling import label_factors
from .target import target_rotation
from .stability import rolling_stability
from .alignment import align_factors, matched_corr
from .selection import suggest_n_factors
from ._types import (
    TrustReport, LabelResult, StabilityResult, AlignmentResult, SelectionResult,
    TargetRotationResult,
)

__version__ = "0.1.0"

__all__ = [
    "FactorModel",
    "sobi", "amuse", "usable_lags",
    "em_pca", "market_neutralize", "pca_scores",
    "identifiability_margin", "margin_null_quantile", "margin_null_quantile_block",
    "detect_regime", "top_pc_share",
    "label_factors",
    "target_rotation",
    "rolling_stability",
    "align_factors", "matched_corr",
    "suggest_n_factors",
    "TrustReport", "LabelResult", "StabilityResult", "AlignmentResult", "SelectionResult",
    "TargetRotationResult",
    "load_reference_factors", "load_portfolios",
    "__version__",
]


def __getattr__(name):
    if name in ("load_reference_factors", "load_portfolios"):
        from . import reference
        return getattr(reference, name)
    raise AttributeError(f"module 'factorscope' has no attribute {name!r}")
