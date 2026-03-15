"""
metrics.py - Frame quality metric computation.

Provides compute_star_metrics() for broadband/RGB frames and
compute_gas_metrics() for narrowband (Ha/OIII/SII) frames.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .image_loader import FITSData
from .background import estimate_background, BackgroundStats
from .star_detection import detect_stars
from .psf_fitting import fit_psf
from .trail_detection import detect_trails

logger = logging.getLogger(__name__)

# Default pixel scale fallback (Redcat 51 + 3.76 um sensor -> ~3.1 arcsec/px)
DEFAULT_PIXEL_SCALE = 206.265 * 3.76 / 250.0


@dataclass
class EvalConfig:
    """
    Configuration for metric computation and rejection thresholds.

    Passed through the entire pipeline so every module uses consistent settings.
    """
    focal_length_mm: float = 250.0

    # Star detection
    detection_threshold: float = 5.0

    # Rejection thresholds
    fwhm_threshold_arcsec: float = 5.0      # absolute FWHM rejection limit
    ecc_threshold: float = 0.5              # eccentricity rejection limit
    star_count_fraction: float = 0.7        # min stars vs session median
    snr_fraction: float = 0.5              # min SNR vs session median
    sigma_fwhm: float = 2.0                # sigma for FWHM statistical rejection
    sigma_noise: float = 2.5               # sigma for noise statistical rejection
    sigma_bg: float = 3.0                  # sigma for background statistical rejection

    # Mode
    mode: str = "auto"  # 'star', 'gas', or 'auto'

    verbose: bool = False


@dataclass
class FrameMetrics:
    """All quality metrics for a single FITS frame."""

    filename: str
    filepath: str
    mode: str               # 'star' or 'gas'
    filter_name: Optional[str]
    exptime: Optional[float]
    gain: Optional[float]
    ccd_temp: Optional[float]
    pixel_scale: Optional[float]

    # Background
    background_median: float = float("nan")
    background_rms: float = float("nan")
    noise_mad: float = float("nan")

    # Star metrics (star mode and partially gas mode)
    n_stars: int = 0
    fwhm_median: float = float("nan")     # arcseconds
    fwhm_mean: float = float("nan")       # arcseconds
    fwhm_std: float = float("nan")        # arcseconds
    eccentricity_median: float = float("nan")
    psf_residual_median: float = float("nan")
    snr_weight: float = float("nan")

    # Gas / narrowband metrics
    snr_estimate: float = float("nan")

    # Trail detection
    n_trails: int = 0
    trail_length_fraction: float = float("nan")
    trail_type: str = "none"   # 'none', 'satellite', 'airplane', 'unknown'

    # Processing notes
    error: Optional[str] = None
    warnings: list = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if at least basic metrics were computed without a fatal error."""
        return self.error is None


def _compute_snr_weight(
    sources,
    noise_rms: float,
) -> float:
    """
    Compute SNR weight proxy: sum(flux^2) / (noise^2 * n_stars).

    This gives a higher weight to frames with more photons per star relative
    to the noise floor.
    """
    if not sources or noise_rms <= 0:
        return float("nan")
    fluxes = np.array([s.flux for s in sources if s.flux > 0])
    if len(fluxes) == 0:
        return float("nan")
    return float(np.sum(fluxes ** 2) / (noise_rms ** 2 * len(fluxes)))


def _estimate_gas_snr(
    image: np.ndarray,
    bg_stats: BackgroundStats,
    sigma_clip: float = 3.0,
) -> float:
    """
    Estimate SNR for a narrowband frame.

    The 'signal region' is identified by finding pixels significantly
    above the background (via sigma-clipping outliers above bg median).
    SNR = (signal_region_median - background_median) / background_rms.
    """
    if bg_stats.background_rms <= 0:
        return float("nan")

    bg_med = bg_stats.background_median
    bg_rms = max(bg_stats.background_rms, bg_stats.noise_mad, 1.0)

    # Signal pixels: those above background + sigma_clip * rms
    threshold = bg_med + sigma_clip * bg_rms
    signal_mask = image > threshold

    n_signal = int(np.sum(signal_mask))
    n_total = image.size

    if n_signal < 10:
        # Very few signal pixels - might be a blank frame or deep sky with little emission
        return float(0.0)

    signal_median = float(np.median(image[signal_mask]))
    snr = (signal_median - bg_med) / bg_rms
    return max(snr, 0.0)


def compute_star_metrics(
    fits_data: FITSData,
    config: EvalConfig,
) -> FrameMetrics:
    """
    Compute quality metrics for a broadband (RGB/Lum) frame.

    Metrics include PSF FWHM, eccentricity, star count, and SNR weight.

    Parameters
    ----------
    fits_data:
        Loaded FITS data from load_fits().
    config:
        EvalConfig with processing parameters.

    Returns
    -------
    FrameMetrics with computed values (some may be NaN on partial failure).
    """
    image = fits_data.data
    pixel_scale = fits_data.pixel_scale_arcsec or DEFAULT_PIXEL_SCALE

    metrics = FrameMetrics(
        filename=fits_data.filename,
        filepath=fits_data.filepath,
        mode="star",
        filter_name=fits_data.filter_name,
        exptime=fits_data.exptime,
        gain=fits_data.gain,
        ccd_temp=fits_data.ccd_temp,
        pixel_scale=pixel_scale,
    )

    # Step 1: Background estimation
    try:
        bg_stats = estimate_background(image)
        metrics.background_median = bg_stats.background_median
        metrics.background_rms = bg_stats.background_rms
        metrics.noise_mad = bg_stats.noise_mad
    except Exception as exc:
        logger.error("Background estimation failed for %s: %s", fits_data.filename, exc)
        metrics.error = f"Background estimation failed: {exc}"
        return metrics

    # Step 2: Star detection
    try:
        sources = detect_stars(
            image,
            bg_stats=bg_stats,
            detection_threshold=config.detection_threshold,
        )
        metrics.n_stars = len(sources)
        logger.debug("%s: detected %d stars.", fits_data.filename, metrics.n_stars)
    except Exception as exc:
        logger.error("Star detection failed for %s: %s", fits_data.filename, exc)
        metrics.error = f"Star detection failed: {exc}"
        return metrics

    if metrics.n_stars == 0:
        metrics.warnings.append("No stars detected.")
        logger.warning("%s: no stars detected.", fits_data.filename)
        return metrics

    # Step 3: PSF fitting
    try:
        psf = fit_psf(image, sources)
        if psf.n_fitted > 0:
            scale = pixel_scale
            metrics.fwhm_median = psf.fwhm_median * scale
            metrics.fwhm_mean = psf.fwhm_mean * scale
            metrics.fwhm_std = psf.fwhm_std * scale
            metrics.eccentricity_median = psf.eccentricity_median
            metrics.psf_residual_median = psf.psf_residual_median
            logger.debug(
                "%s: PSF FWHM=%.2f\" ecc=%.3f from %d stars.",
                fits_data.filename,
                metrics.fwhm_median,
                metrics.eccentricity_median,
                psf.n_fitted,
            )
        else:
            metrics.warnings.append("PSF fitting returned no valid results.")
            logger.warning("%s: PSF fitting returned no valid results.", fits_data.filename)
    except Exception as exc:
        logger.error("PSF fitting failed for %s: %s", fits_data.filename, exc)
        metrics.warnings.append(f"PSF fitting error: {exc}")

    # Step 4: SNR weight
    try:
        noise = max(bg_stats.background_rms, bg_stats.noise_mad, 1.0)
        metrics.snr_weight = _compute_snr_weight(sources, noise)
    except Exception as exc:
        logger.warning("SNR weight computation failed for %s: %s", fits_data.filename, exc)

    # Step 5: Trail detection
    try:
        trail = detect_trails(
            image,
            background_median=bg_stats.background_median,
            background_rms=bg_stats.background_rms,
        )
        metrics.n_trails = trail.n_trails
        metrics.trail_length_fraction = trail.max_length_fraction
        metrics.trail_type = trail.trail_type
        if trail.detected:
            logger.info("%s: %d trail(s) detected (%s).", fits_data.filename, trail.n_trails, trail.trail_type)
    except Exception as exc:
        logger.warning("Trail detection failed for %s: %s", fits_data.filename, exc)

    return metrics


def compute_gas_metrics(
    fits_data: FITSData,
    config: EvalConfig,
) -> FrameMetrics:
    """
    Compute quality metrics for a narrowband (Ha/OIII/SII) frame.

    Metrics focus on background noise, SNR, and star count as a
    transparency proxy.

    Parameters
    ----------
    fits_data:
        Loaded FITS data from load_fits().
    config:
        EvalConfig with processing parameters.

    Returns
    -------
    FrameMetrics with computed values.
    """
    image = fits_data.data
    pixel_scale = fits_data.pixel_scale_arcsec or DEFAULT_PIXEL_SCALE

    metrics = FrameMetrics(
        filename=fits_data.filename,
        filepath=fits_data.filepath,
        mode="gas",
        filter_name=fits_data.filter_name,
        exptime=fits_data.exptime,
        gain=fits_data.gain,
        ccd_temp=fits_data.ccd_temp,
        pixel_scale=pixel_scale,
    )

    # Step 1: Background estimation
    try:
        bg_stats = estimate_background(image)
        metrics.background_median = bg_stats.background_median
        metrics.background_rms = bg_stats.background_rms
        metrics.noise_mad = bg_stats.noise_mad
    except Exception as exc:
        logger.error("Background estimation failed for %s: %s", fits_data.filename, exc)
        metrics.error = f"Background estimation failed: {exc}"
        return metrics

    # Step 2: SNR estimate for nebula signal
    try:
        metrics.snr_estimate = _estimate_gas_snr(image, bg_stats)
        logger.debug("%s: gas SNR estimate = %.3f", fits_data.filename, metrics.snr_estimate)
    except Exception as exc:
        logger.warning("Gas SNR estimation failed for %s: %s", fits_data.filename, exc)

    # Step 3: Star count (transparency proxy)
    try:
        sources = detect_stars(
            image,
            bg_stats=bg_stats,
            detection_threshold=config.detection_threshold,
        )
        metrics.n_stars = len(sources)
        logger.debug(
            "%s (gas): detected %d stars (transparency proxy).",
            fits_data.filename,
            metrics.n_stars,
        )
    except Exception as exc:
        logger.warning(
            "Star detection failed for gas frame %s: %s", fits_data.filename, exc
        )

    # Step 4: Trail detection
    try:
        trail = detect_trails(
            image,
            background_median=bg_stats.background_median,
            background_rms=bg_stats.background_rms,
        )
        metrics.n_trails = trail.n_trails
        metrics.trail_length_fraction = trail.max_length_fraction
        metrics.trail_type = trail.trail_type
        if trail.detected:
            logger.info("%s: %d trail(s) detected (%s).", fits_data.filename, trail.n_trails, trail.trail_type)
    except Exception as exc:
        logger.warning("Trail detection failed for %s: %s", fits_data.filename, exc)

    return metrics


def compute_metrics(
    fits_data: FITSData,
    config: EvalConfig,
    mode_override: Optional[str] = None,
) -> FrameMetrics:
    """
    Dispatch to star or gas metric computation based on mode.

    Parameters
    ----------
    fits_data:
        Loaded FITS data.
    config:
        EvalConfig.
    mode_override:
        Force a specific mode ('star' or 'gas'), ignoring auto-detection.

    Returns
    -------
    FrameMetrics.
    """
    effective_mode = mode_override or fits_data.mode or "star"

    if effective_mode == "gas":
        return compute_gas_metrics(fits_data, config)
    return compute_star_metrics(fits_data, config)
