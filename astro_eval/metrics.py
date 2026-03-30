"""
metrics.py - Frame quality metric computation.

Provides compute_star_metrics() for broadband/RGB frames and
compute_gas_metrics() for narrowband (Ha/OIII/SII) frames.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

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
class ScoringWeights:
    """
    Configurable weights for the composite quality score equations.

    All weights within a mode should sum to 1.0.  Values outside that range
    are accepted (the score is clipped to [0, 1] regardless), but a warning
    is issued by the config loader if the sum deviates significantly from 1.0.
    """
    # Star / broadband score weights
    star_fwhm:  float = 0.30
    star_ecc:   float = 0.25
    star_stars: float = 0.20
    star_psfsw: float = 0.25   # PSFSignalWeight: combined FWHM+amplitude+noise metric
    star_snr:   float = 0.00   # legacy SNR weight — kept for CSV/HTML output, not used in scoring

    # Gas / narrowband score weights
    gas_snr:   float = 0.30
    gas_noise: float = 0.20
    gas_bg:    float = 0.15
    gas_stars: float = 0.20
    gas_psfsw: float = 0.15

    # Gradient multiplier steepness (applied post-scoring, like trail penalties)
    # Controls how fast the multiplier drops above the knee.
    # Higher = steeper penalty. Default 1.0 gives ~66% penalty at 2× median, ~30% at 3× median.
    gradient_penalty_strength: float = 1.0


@dataclass
class EvalConfig:
    """
    Configuration for metric computation and rejection thresholds.

    Passed through the entire pipeline so every module uses consistent settings.
    """
    bortle: int = 0

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

    sigma_residual: float = 3.0    # sigma for PSF residual statistical rejection
    gradient_threshold: float = 0.0    # hard rejection: gradient in noise σ units; 0 = disabled (sigma_gradient used instead)
    sigma_gradient:    float = 2.0    # session-relative rejection: reject if gradient > median + sigma × std
    gradient_knee:     float = 1.2    # scoring knee: multiplier starts dropping above knee × session_median

    min_score: float = 0.5             # reject if composite score < this value (0 = disabled)

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
    background_gradient: float = float("nan")   # (max-min)/median of 2D bg map; >1.0 = severe gradient

    # Star metrics (star mode and partially gas mode)
    n_stars: int = 0
    fwhm_median: float = float("nan")     # arcseconds
    fwhm_mean: float = float("nan")       # arcseconds
    fwhm_std: float = float("nan")        # arcseconds
    eccentricity_median: float = float("nan")
    psf_residual_median: float = float("nan")
    snr_weight: float = float("nan")
    psf_signal_weight: float = float("nan")  # PSFSignalWeight: combines amplitude, FWHM, noise (1/FWHM² penalty)
    wfwhm: float = float("nan")              # Siril wFWHM = FWHM_arcsec / sqrt(n_stars); lower = better
    moffat_beta: float = float("nan")        # Moffat beta parameter (atmospheric seeing index)

    # Gas / narrowband metrics
    snr_estimate: float = float("nan")

    # Spatial FWHM map: 5×5 grid of median FWHM (arcsec) per image region; NaN = no stars in cell
    fwhm_map: Optional[List[List[float]]] = field(default=None)

    # Elongation direction (circular statistics on fitted PSF theta)
    elongation_direction: float = float("nan")    # mean PSF orientation angle (radians)
    elongation_consistency: float = float("nan")  # R ∈ [0,1]: 0=random, 1=all aligned same direction

    # Trail detection
    n_trails: int = 0
    trail_length_fraction: float = float("nan")
    trail_type: str = "none"   # 'none', 'satellite', 'airplane', 'unknown'

    # Observation time (ISO-8601 string from DATE-OBS header, for trend charts)
    obs_time: Optional[str] = None

    # Telescope altitude above horizon in degrees (0=horizon, 90=zenith)
    altitude_deg: Optional[float] = None

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


def _compute_psf_signal_weight(psf, noise_rms: float) -> float:
    """
    PixInsight-compatible PSFSignalWeight (PSFSW).

    Formula (averaged over n fitted stars):
        PSFSW = Σ_i (A_i² / FWHM_i²) / (2 × noise² × n)

    This matches the PixInsight SubframeSelector definition:
    - Sums squared amplitudes A_i², not the square of their sum (Σ A_i)²;
      (Σ A_i)² ≠ Σ(A_i²) and would grow quadratically with star count.
    - Uses each star's individual FWHM_i (pixels), not the session median,
      so the metric is sensitive to per-star sharpness variation across the field.
    - Divides by n so the result is a per-star average, independent of how
      many stars happen to be detected in a given frame.

    Qualitatively: higher PSFSW means brighter, sharper stars relative to
    the background noise floor.  FWHM² in the denominator penalises poor
    seeing super-linearly (2× worse FWHM → 4× lower PSFSW).
    """
    individual = [
        r for r in psf.individual
        if r.success and math.isfinite(r.amplitude) and math.isfinite(r.fwhm_pix)
        and r.amplitude > 0 and r.fwhm_pix > 0
    ]
    if not individual or noise_rms <= 0:
        return float("nan")

    signal_sum = sum(r.amplitude ** 2 / r.fwhm_pix ** 2 for r in individual)
    n = len(individual)
    return float(signal_sum / (2.0 * noise_rms ** 2 * n))


def _estimate_gas_snr(
    image: np.ndarray,
    bg_stats: BackgroundStats,
) -> float:
    """
    Estimate SNR for a narrowband frame using the 95th-percentile method.

    SNR = (p95 - background_median) / background_rms

    Using the 95th percentile as the signal reference is robust and
    threshold-independent: for a pure-background frame, p95 ≈ bg + 1.6*rms
    (≈1.6 SNR), while a frame with bright nebulosity gives a higher value.
    This avoids the sigma-clip sensitivity of the previous sigma=3 threshold.
    """
    if bg_stats.background_rms <= 0:
        return float("nan")

    bg_med = bg_stats.background_median
    bg_rms = max(bg_stats.background_rms, bg_stats.noise_mad, 1.0)

    p95 = float(np.percentile(image, 95))
    snr = (p95 - bg_med) / bg_rms
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
        obs_time=fits_data.obs_time,
        altitude_deg=fits_data.altitude_deg,
    )

    # Step 1: Background estimation
    try:
        bg_stats = estimate_background(image)
        metrics.background_median = bg_stats.background_median
        metrics.background_rms = bg_stats.background_rms
        metrics.noise_mad = bg_stats.noise_mad
        metrics.background_gradient = bg_stats.background_gradient
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
    psf = None
    try:
        psf = fit_psf(image, sources, image_shape=image.shape)
        if psf.n_fitted > 0:
            scale = pixel_scale
            metrics.fwhm_median = psf.fwhm_median * scale
            metrics.fwhm_mean = psf.fwhm_mean * scale
            metrics.fwhm_std = psf.fwhm_std * scale
            metrics.eccentricity_median = psf.eccentricity_median
            metrics.psf_residual_median = psf.psf_residual_median
            metrics.moffat_beta = psf.beta_median
            # Convert spatial FWHM map from pixels to arcsec
            if psf.fwhm_map is not None:
                metrics.fwhm_map = [
                    [v * scale if math.isfinite(v) else float("nan") for v in row]
                    for row in psf.fwhm_map
                ]
            metrics.elongation_direction = psf.theta_mean
            metrics.elongation_consistency = psf.theta_consistency
            logger.debug(
                "%s: PSF FWHM=%.2f\" ecc=%.3f beta=%.2f from %d stars.",
                fits_data.filename,
                metrics.fwhm_median,
                metrics.eccentricity_median,
                metrics.moffat_beta if math.isfinite(metrics.moffat_beta) else 0.0,
                psf.n_fitted,
            )
        else:
            metrics.warnings.append("PSF fitting returned no valid results.")
            logger.warning("%s: PSF fitting returned no valid results.", fits_data.filename)
    except Exception as exc:
        logger.error("PSF fitting failed for %s: %s", fits_data.filename, exc)
        metrics.warnings.append(f"PSF fitting error: {exc}")
        psf = None

    # Step 4: SNR weight + PSFSignalWeight + wFWHM
    try:
        noise = max(bg_stats.background_rms, bg_stats.noise_mad, 1.0)
        metrics.snr_weight = _compute_snr_weight(sources, noise)
        if psf is not None and psf.n_fitted > 0:
            metrics.psf_signal_weight = _compute_psf_signal_weight(psf, noise)
            if math.isfinite(metrics.fwhm_median) and metrics.n_stars > 0:
                metrics.wfwhm = metrics.fwhm_median / math.sqrt(metrics.n_stars)
    except Exception as exc:
        logger.warning("SNR/PSF weight computation failed for %s: %s", fits_data.filename, exc)

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
        obs_time=fits_data.obs_time,
        altitude_deg=fits_data.altitude_deg,
    )

    # Step 1: Background estimation
    try:
        bg_stats = estimate_background(image)
        metrics.background_median = bg_stats.background_median
        metrics.background_rms = bg_stats.background_rms
        metrics.noise_mad = bg_stats.noise_mad
        metrics.background_gradient = bg_stats.background_gradient
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

    # Step 4: PSF fitting (star shape quality — useful even in narrowband for PSFSW)
    if metrics.n_stars > 0:
        psf = None
        try:
            psf = fit_psf(image, sources, image_shape=image.shape)
            if psf.n_fitted > 0:
                scale = pixel_scale
                metrics.fwhm_median = psf.fwhm_median * scale
                metrics.fwhm_mean = psf.fwhm_mean * scale
                metrics.fwhm_std = psf.fwhm_std * scale
                metrics.eccentricity_median = psf.eccentricity_median
                metrics.psf_residual_median = psf.psf_residual_median
                metrics.moffat_beta = psf.beta_median
                if psf.fwhm_map is not None:
                    metrics.fwhm_map = [
                        [v * scale if math.isfinite(v) else float("nan") for v in row]
                        for row in psf.fwhm_map
                    ]
                metrics.elongation_direction = psf.theta_mean
                metrics.elongation_consistency = psf.theta_consistency
                logger.debug(
                    "%s (gas): PSF FWHM=%.2f\" ecc=%.3f beta=%.2f from %d stars.",
                    fits_data.filename,
                    metrics.fwhm_median,
                    metrics.eccentricity_median,
                    metrics.moffat_beta if math.isfinite(metrics.moffat_beta) else 0.0,
                    psf.n_fitted,
                )
            else:
                metrics.warnings.append("PSF fitting returned no valid results.")
        except Exception as exc:
            logger.warning("PSF fitting failed for gas frame %s: %s", fits_data.filename, exc)
            psf = None

        try:
            noise = max(bg_stats.background_rms, bg_stats.noise_mad, 1.0)
            if psf is not None and psf.n_fitted > 0:
                metrics.psf_signal_weight = _compute_psf_signal_weight(psf, noise)
                if math.isfinite(metrics.fwhm_median) and metrics.n_stars > 0:
                    metrics.wfwhm = metrics.fwhm_median / math.sqrt(metrics.n_stars)
        except Exception as exc:
            logger.warning("PSF weight computation failed for gas frame %s: %s", fits_data.filename, exc)

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
