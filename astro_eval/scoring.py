"""
scoring.py - Composite scoring and frame rejection decisions.

Computes session-level statistics, applies rejection criteria,
and produces a normalized composite quality score for each frame.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .metrics import FrameMetrics, EvalConfig, ScoringWeights

logger = logging.getLogger(__name__)

MIN_FRAMES_FOR_STATS = 3


@dataclass
class SessionStats:
    """Descriptive statistics for a session (computed across all frames)."""

    metric_name: str
    count: int
    median: float
    mean: float
    std: float
    min_val: float
    max_val: float

    def __repr__(self) -> str:
        return (
            f"SessionStats({self.metric_name}: "
            f"median={self.median:.4g}, std={self.std:.4g}, "
            f"n={self.count})"
        )


@dataclass
class RejectionFlags:
    """Per-frame rejection decision with per-criterion breakdown."""

    filename: str
    rejected: bool
    flags: Dict[str, bool] = field(default_factory=dict)
    score: float = float("nan")

    @property
    def rejection_reasons(self) -> List[str]:
        return [k for k, v in self.flags.items() if v]


# ---------------------------------------------------------------------------
# Session statistics
# ---------------------------------------------------------------------------

def _collect_values(
    all_metrics: List[FrameMetrics],
    attr: str,
) -> np.ndarray:
    """Collect finite non-NaN values of a metric attribute across all frames."""
    vals = []
    for m in all_metrics:
        v = getattr(m, attr, None)
        if v is not None and math.isfinite(v):
            vals.append(v)
    return np.array(vals, dtype=np.float64)


def _session_stats(
    all_metrics: List[FrameMetrics],
    attr: str,
) -> SessionStats:
    """Compute session statistics for a single metric attribute."""
    vals = _collect_values(all_metrics, attr)
    n = len(vals)
    if n == 0:
        return SessionStats(
            metric_name=attr, count=0,
            median=float("nan"), mean=float("nan"), std=float("nan"),
            min_val=float("nan"), max_val=float("nan"),
        )
    return SessionStats(
        metric_name=attr,
        count=n,
        median=float(np.median(vals)),
        mean=float(np.mean(vals)),
        std=float(np.std(vals, ddof=1)) if n > 1 else 0.0,
        min_val=float(np.min(vals)),
        max_val=float(np.max(vals)),
    )


def compute_session_statistics(
    all_metrics: List[FrameMetrics],
) -> Dict[str, SessionStats]:
    """
    Compute session-level descriptive statistics for all frame metrics.

    Parameters
    ----------
    all_metrics:
        List of FrameMetrics from all frames.

    Returns
    -------
    Dict mapping metric name -> SessionStats.
    """
    if len(all_metrics) < MIN_FRAMES_FOR_STATS:
        logger.warning(
            "Only %d frames available for session statistics (minimum recommended: %d). "
            "Rejection thresholds may be unreliable.",
            len(all_metrics),
            MIN_FRAMES_FOR_STATS,
        )

    attrs = [
        "background_median",
        "background_rms",
        "noise_mad",
        "n_stars",
        "fwhm_median",
        "fwhm_mean",
        "fwhm_std",
        "eccentricity_median",
        "psf_residual_median",
        "snr_weight",
        "psf_signal_weight",
        "wfwhm",
        "moffat_beta",
        "snr_estimate",
        "background_gradient",
    ]

    stats: Dict[str, SessionStats] = {}
    for attr in attrs:
        stats[attr] = _session_stats(all_metrics, attr)
        if stats[attr].count > 0:
            logger.debug("Session %s", stats[attr])

    return stats


# ---------------------------------------------------------------------------
# Rejection flags
# ---------------------------------------------------------------------------

def compute_rejection_flags(
    frame_metrics: FrameMetrics,
    session_stats: Dict[str, SessionStats],
    config: EvalConfig,
) -> RejectionFlags:
    """
    Apply per-frame rejection criteria based on session statistics.

    Star mode criteria:
    - FWHM too large (absolute and statistical)
    - Eccentricity too high
    - Too few stars
    - SNR weight too low

    Gas mode criteria:
    - Background noise too high
    - Background level too high (sky glow)
    - SNR too low
    - Too few stars (transparency)

    Parameters
    ----------
    frame_metrics:
        FrameMetrics for a single frame.
    session_stats:
        Session-level statistics dict from compute_session_statistics().
    config:
        EvalConfig with threshold parameters.

    Returns
    -------
    RejectionFlags with per-criterion breakdown.
    """
    flags: Dict[str, bool] = {}

    def stat(name: str) -> SessionStats:
        return session_stats.get(name, SessionStats(
            name, 0, float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"),
        ))

    if frame_metrics.mode == "gas":
        # --- Gas / narrowband rejection ---

        # High background noise
        s = stat("background_rms")
        if s.count > 0 and math.isfinite(frame_metrics.background_rms):
            thresh = s.median + config.sigma_noise * s.std
            flags["high_noise"] = frame_metrics.background_rms > thresh
        else:
            flags["high_noise"] = False

        # High background level (sky glow, moon contamination)
        s = stat("background_median")
        if s.count > 0 and math.isfinite(frame_metrics.background_median):
            thresh = s.median + config.sigma_bg * s.std
            flags["high_background"] = frame_metrics.background_median > thresh
        else:
            flags["high_background"] = False

        # Low SNR (poor transparency or clouds)
        s = stat("snr_estimate")
        if s.count > 0 and math.isfinite(frame_metrics.snr_estimate) and math.isfinite(s.median):
            flags["low_snr"] = frame_metrics.snr_estimate < s.median * 0.5
        else:
            flags["low_snr"] = False

        # Low star count (poor transparency)
        s = stat("n_stars")
        if s.count > 0 and math.isfinite(s.median) and s.median > 0:
            flags["low_stars"] = frame_metrics.n_stars < s.median * config.star_count_fraction
        else:
            flags["low_stars"] = False

    else:
        # --- Star / broadband rejection ---

        # FWHM too large (statistical)
        s = stat("fwhm_median")
        fwhm = frame_metrics.fwhm_median
        fwhm_rejected = False
        if s.count > 0 and math.isfinite(fwhm) and math.isfinite(s.median):
            stat_thresh = s.median + config.sigma_fwhm * s.std
            fwhm_rejected = fwhm > stat_thresh
        if math.isfinite(fwhm):
            fwhm_rejected = fwhm_rejected or (fwhm > config.fwhm_threshold_arcsec)
        flags["high_fwhm"] = fwhm_rejected

        # High eccentricity (tracking error, wind)
        ecc = frame_metrics.eccentricity_median
        if math.isfinite(ecc):
            flags["high_eccentricity"] = ecc > config.ecc_threshold
        else:
            flags["high_eccentricity"] = False

        # Low star count (clouds or focus shift)
        s = stat("n_stars")
        if s.count > 0 and math.isfinite(s.median) and s.median > 0:
            flags["low_stars"] = frame_metrics.n_stars < s.median * config.star_count_fraction
        else:
            flags["low_stars"] = False

        # Low SNR weight (faint frame, clouds)
        s = stat("snr_weight")
        snr_w = frame_metrics.snr_weight
        if s.count > 0 and math.isfinite(snr_w) and math.isfinite(s.median) and s.median > 0:
            flags["low_snr_weight"] = snr_w < s.median * config.snr_fraction
        else:
            flags["low_snr_weight"] = False

        # High PSF residual (distorted stars, optical aberrations, trailing)
        s = stat("psf_residual_median")
        residual = frame_metrics.psf_residual_median
        if s.count > 0 and math.isfinite(residual) and math.isfinite(s.median):
            thresh = s.median + config.sigma_residual * s.std
            flags["high_residual"] = residual > thresh
        else:
            flags["high_residual"] = False

    # Severe background gradient (sunrise, twilight, cloud edge) — applies to both modes.
    # Uses session-relative sigma criterion (primary) + optional absolute hard cap.
    gradient = frame_metrics.background_gradient
    grad_rejected = False
    if math.isfinite(gradient):
        s = stat("background_gradient")
        if s.count > 0 and math.isfinite(s.median) and math.isfinite(s.std):
            grad_rejected = gradient > s.median + config.sigma_gradient * s.std
        if config.gradient_threshold > 0:
            grad_rejected = grad_rejected or (gradient > config.gradient_threshold)
    flags["high_gradient"] = grad_rejected

    # Trail detection applies regardless of mode.
    # Airplane trails are hard-rejected; satellite trails are informational only.
    flags["airplane_trail"]  = frame_metrics.trail_type == "airplane"
    flags["satellite_trail"] = frame_metrics.trail_type in ("satellite", "unknown")

    # satellite_trail and high_residual are informational — never cause rejection on their own
    _soft_flags = {"satellite_trail", "high_residual"}  # high_gradient is a hard flag
    rejected = any(v for k, v in flags.items() if k not in _soft_flags)
    return RejectionFlags(
        filename=frame_metrics.filename,
        rejected=rejected,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _gradient_multiplier(
    gradient: float,
    s: SessionStats,
    knee: float = 1.2,
    strength: float = 1.0,
) -> float:
    """
    Multiplicative gradient penalty applied after base score computation.

    Works like trail penalties: a bad gradient crushes the entire score
    regardless of how good FWHM/SNR/etc. are.  The additive approach only
    nudged the score by ≤0.15; the multiplicative approach can reduce it to
    near zero for severely burned frames.

    multiplier = 1.0               when gradient ≤ knee × session_median
    multiplier = exp(-strength × excess_ratio)  above the knee
                                   where excess_ratio = (gradient/median) - knee

    With strength=1.0 and knee=1.2:
      gradient = 1.5× median  → multiplier ≈ 0.74  (mild penalty)
      gradient = 2.0× median  → multiplier ≈ 0.45
      gradient = 3.0× median  → multiplier ≈ 0.17
      gradient = 4.0× median  → multiplier ≈ 0.07  (near-zero)
    """
    if not math.isfinite(gradient) or s.count == 0:
        return 1.0
    if not math.isfinite(s.median) or s.median <= 0:
        return 1.0

    ratio = gradient / s.median
    if ratio <= knee:
        return 1.0

    excess = ratio - knee
    return float(max(0.05, math.exp(-strength * excess)))


def _normalize(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize a value to [0, 1] within the given range.

    Returns 0.5 if range is zero or value is not finite.
    """
    if not math.isfinite(value):
        return 0.5
    span = max_val - min_val
    if span <= 0:
        return 0.5
    return float(np.clip((value - min_val) / span, 0.0, 1.0))


def compute_star_score(
    metrics: FrameMetrics,
    session_stats: Dict[str, SessionStats],
    weights: Optional[ScoringWeights] = None,
    config: Optional[EvalConfig] = None,
) -> float:
    """
    Compute composite quality score for a broadband frame.

    Score = w_fwhm*(1-norm_fwhm) + w_ecc*(1-norm_ecc) + w_stars*norm_stars + w_psfsw*norm_psfsw

    snr_weight is retained in the CSV/HTML output for reference but has a
    default weight of 0.0 — PSFSignalWeight supersedes it because it already
    incorporates the amplitude/noise ratio plus a 1/FWHM² correction.

    A gradient multiplier is applied in compute_score() after this base score,
    similar to trail penalties.

    All metrics are normalized to [0, 1] across the session range.
    """
    w = weights or ScoringWeights()

    def s(name: str) -> SessionStats:
        return session_stats.get(name, SessionStats(
            name, 0, float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"),
        ))

    norm_fwhm  = _normalize(metrics.fwhm_median,        s("fwhm_median").min_val,        s("fwhm_median").max_val)
    norm_ecc   = _normalize(metrics.eccentricity_median, s("eccentricity_median").min_val, s("eccentricity_median").max_val)
    norm_stars = _normalize(float(metrics.n_stars),      s("n_stars").min_val,             s("n_stars").max_val)
    norm_snr   = _normalize(metrics.snr_weight,          s("snr_weight").min_val,          s("snr_weight").max_val)
    norm_psfsw = _normalize(metrics.psf_signal_weight,   s("psf_signal_weight").min_val,   s("psf_signal_weight").max_val)

    score = (
        w.star_fwhm  * (1.0 - norm_fwhm)
        + w.star_ecc   * (1.0 - norm_ecc)
        + w.star_stars * norm_stars
        + w.star_snr   * norm_snr          # 0.0 by default; configurable for back-compat
        + w.star_psfsw * norm_psfsw
    )
    return float(np.clip(score, 0.0, 1.0))


def compute_gas_score(
    metrics: FrameMetrics,
    session_stats: Dict[str, SessionStats],
    weights: Optional[ScoringWeights] = None,
    config: Optional[EvalConfig] = None,
) -> float:
    """
    Compute composite quality score for a narrowband frame.

    Score = w_snr*norm_snr + w_noise*(1-norm_noise) + w_bg*(1-norm_bg) + w_stars*norm_stars
    """
    w = weights or ScoringWeights()

    def s(name: str) -> SessionStats:
        return session_stats.get(name, SessionStats(
            name, 0, float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"),
        ))

    norm_snr   = _normalize(metrics.snr_estimate,    s("snr_estimate").min_val,    s("snr_estimate").max_val)
    norm_noise = _normalize(metrics.background_rms,  s("background_rms").min_val,  s("background_rms").max_val)
    norm_bg    = _normalize(metrics.background_median, s("background_median").min_val, s("background_median").max_val)
    norm_stars = _normalize(float(metrics.n_stars),  s("n_stars").min_val,         s("n_stars").max_val)
    norm_psfsw = _normalize(metrics.psf_signal_weight, s("psf_signal_weight").min_val, s("psf_signal_weight").max_val)

    score = (
        w.gas_snr   * norm_snr
        + w.gas_noise * (1.0 - norm_noise)
        + w.gas_bg    * (1.0 - norm_bg)
        + w.gas_stars * norm_stars
        + w.gas_psfsw * norm_psfsw
    )
    return float(np.clip(score, 0.0, 1.0))


# Score multipliers applied after the base score is computed.
# Multiplicative penalties preserve relative ordering among trailed frames
# (a slightly-better airplane frame stays slightly better than a worse one).
_TRAIL_PENALTY: dict = {
    "airplane":  0.15,   # drop to at most 15% of base score — effectively disqualifying
    "satellite": 0.80,   # 20% penalty — visible in rankings but not disqualifying
    "unknown":   0.70,   # treated cautiously: heavier than satellite, lighter than airplane
    "none":      1.00,
}


def compute_score(
    metrics: FrameMetrics,
    session_stats: Dict[str, SessionStats],
    weights: Optional[ScoringWeights] = None,
    config: Optional[EvalConfig] = None,
) -> float:
    """
    Compute composite quality score, then apply trail penalty.

    Trail penalties are multiplicative so they preserve relative ordering
    among frames of the same trail type.
    """
    if metrics.mode == "gas":
        base = compute_gas_score(metrics, session_stats, weights, config)
    else:
        base = compute_star_score(metrics, session_stats, weights, config)

    trail_multiplier = _TRAIL_PENALTY.get(metrics.trail_type, 1.0)

    # Gradient penalty only applies to gas/narrowband mode.
    # In star mode gradients are correctable in post-processing and should not affect the score.
    if metrics.mode == "gas":
        gradient_mult = _gradient_multiplier(
            metrics.background_gradient,
            session_stats.get("background_gradient", SessionStats(
                "background_gradient", 0, float("nan"), float("nan"), float("nan"),
                float("nan"), float("nan"),
            )),
            knee=config.gradient_knee if config else 1.2,
            strength=weights.gradient_penalty_strength if weights else 1.0,
        )
    else:
        gradient_mult = 1.0

    return float(np.clip(base * trail_multiplier * gradient_mult, 0.0, 1.0))


@dataclass
class FrameResult:
    """Combined result for a single frame: metrics + rejection + score."""

    metrics: FrameMetrics
    rejection: RejectionFlags
    score: float


def evaluate_session(
    all_metrics: List[FrameMetrics],
    config: EvalConfig,
    weights: Optional[ScoringWeights] = None,
) -> tuple[Dict[str, SessionStats], List[FrameResult]]:
    """
    Evaluate all frames: compute session statistics, scores, and rejection flags.

    Parameters
    ----------
    all_metrics:
        List of FrameMetrics from all loaded frames.
    config:
        EvalConfig with threshold parameters.

    Returns
    -------
    Tuple of (session_stats, list of FrameResult).
    """
    session_stats = compute_session_statistics(all_metrics)

    min_score = config.min_score if config else 0.0

    results: List[FrameResult] = []
    for m in all_metrics:
        rejection = compute_rejection_flags(m, session_stats, config)
        score = compute_score(m, session_stats, weights, config)
        rejection.score = score
        if min_score > 0 and math.isfinite(score) and score < min_score:
            rejection.flags["low_score"] = True
            rejection.rejected = True
        results.append(FrameResult(metrics=m, rejection=rejection, score=score))

    accepted = sum(1 for r in results if not r.rejection.rejected)
    logger.info(
        "Session evaluation: %d frames, %d accepted, %d rejected.",
        len(results), accepted, len(results) - accepted,
    )
    return session_stats, results
