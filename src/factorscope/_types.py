from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TrustReport:
    regime: str
    top_pc_share: float
    margin: float
    margin_threshold: float
    margin_ok: bool
    neutralized: bool
    per_factor_snr: np.ndarray
    verdict: str
    warnings: List[str] = field(default_factory=list)
    null_model: str = "gaussian"

    def __repr__(self):
        lines = ["TrustReport",
                 f"  regime            : {self.regime}",
                 f"  top-PC share      : {self.top_pc_share:.2f}",
                 f"  market neutralized: {self.neutralized}",
                 f"  identif. margin   : {self.margin:.2f}  "
                 f"({'OK' if self.margin_ok else 'LOW'}, {self.null_model} "
                 f"noise floor {self.margin_threshold:.2f})",
                 f"  verdict           : {self.verdict}"]
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


@dataclass
class LabelResult:
    table: pd.DataFrame
    r2: pd.Series
    names: pd.Series

    def __repr__(self):
        rows = [f"  {f:<6} ~ {self.names[f]:<28} (R2={self.r2[f]:.2f})"
                for f in self.names.index]
        return "LabelResult\n" + "\n".join(rows)


@dataclass
class TargetRotationResult:
    factors: pd.DataFrame
    loadings: pd.DataFrame
    corr: pd.Series
    subspace_r2: pd.Series
    rotation: np.ndarray
    dropped: List[str] = field(default_factory=list)

    def __repr__(self):
        rows = [f"  {n:<8} |corr| with reference = {self.corr[n]:.2f}"
                for n in self.corr.index]
        out = ["TargetRotationResult (Procrustes rotation onto reference factors)"] + rows
        resid = [c for c in self.factors.columns if c not in self.corr.index]
        if resid:
            out.append(f"  residual (unaligned) factors: {', '.join(resid)}")
        if self.dropped:
            out.append(f"  references not aligned (fewer factors than references): "
                       f"{', '.join(self.dropped)}")
        return "\n".join(out)


@dataclass
class StabilityResult:
    sobi: np.ndarray
    pca: np.ndarray
    window: int
    step: int

    @property
    def sobi_mean(self):
        return float(np.mean(self.sobi)) if len(self.sobi) else float("nan")

    @property
    def pca_mean(self):
        return float(np.mean(self.pca)) if len(self.pca) else float("nan")

    def __repr__(self):
        return ("StabilityResult\n"
                f"  refits            : {len(self.sobi)}  (window={self.window}, step={self.step})\n"
                f"  SOBI persistence  : {self.sobi_mean:.3f}\n"
                f"  PCA  persistence  : {self.pca_mean:.3f}")


@dataclass
class AlignmentResult:
    permutation: np.ndarray
    signs: np.ndarray
    matched_corr: np.ndarray
    mean_corr: float

    def __repr__(self):
        return (f"AlignmentResult(mean|corr|={self.mean_corr:.3f}, "
                f"perm={self.permutation.tolist()})")


@dataclass
class SelectionResult:
    n_suggested: int
    eigenvalues: np.ndarray
    explained: np.ndarray
    eigen_gap_k: int
    noise_floor_k: int
    reason: str

    def __repr__(self):
        return (f"SelectionResult(n_suggested={self.n_suggested}, "
                f"eigen_gap_k={self.eigen_gap_k}, noise_floor_k={self.noise_floor_k})\n"
                f"  {self.reason}")
