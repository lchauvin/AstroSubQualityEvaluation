"""
star_detection.py - Star detection using SEP (Source Extractor Python).

Detects point sources in background-subtracted images, applies quality
filters, and returns a list of validated stellar sources.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .background import BackgroundStats, estimate_background, subtract_background

logger = logging.getLogger(__name__)

# Default detection parameters
DEFAULT_DETECT_THRESH = 5.0   # sigma above background
DEFAULT_MIN_AREA = 5          # minimum source area in pixels
DEFAULT_DEBLEND_NTHRESH = 32
DEFAULT_DEBLEND_CONT = 0.005
DEFAULT_EDGE_MARGIN = 50      # pixels from border to exclude
DEFAULT_MAX_ELONGATION = 3.0  # a/b ratio
DEFAULT_MAX_FWHM_PIX = 50.0   # sanity cap


@dataclass
class StarSource:
    """A detected stellar source with measured properties."""

    x: float           # centroid x (pixels, 0-based)
    y: float           # centroid y (pixels, 0-based)
    flux: float        # integrated flux (ADU)
    peak: float        # peak pixel value
    a: float           # semi-major axis (pixels)
    b: float           # semi-minor axis (pixels)
    theta: float       # position angle (radians)
    fwhm_estimate: float   # estimated FWHM (pixels), 2*sqrt(ln2)*2*a
    elongation: float  # a/b ratio
    flag: int          # SEP extraction flag

    @property
    def eccentricity(self) -> float:
        """Eccentricity from semi-axes: sqrt(1 - (b/a)^2)."""
        if self.a <= 0:
            return float("nan")
        ratio = self.b / self.a
        ratio = min(ratio, 1.0)  # guard against numerical noise
        return float(np.sqrt(1.0 - ratio ** 2))


def _sep_to_sources(objects, image_shape: tuple) -> List[StarSource]:
    """Convert SEP extraction result array to list of StarSource objects."""
    sources = []
    for obj in objects:
        # FWHM from SEP's 'a' parameter.
        # SEP 'a' is the RMS of the light distribution along the major axis,
        # which equals Gaussian sigma for a circular Gaussian.
        # FWHM = 2 * sqrt(2 * ln 2) * sigma ≈ 2.355 * a.
        fwhm_est = 2.0 * np.sqrt(2.0 * np.log(2.0)) * float(obj["a"])
        a_val = max(float(obj["a"]), 1e-6)
        b_val = max(float(obj["b"]), 1e-6)
        elongation = a_val / b_val

        sources.append(
            StarSource(
                x=float(obj["x"]),
                y=float(obj["y"]),
                flux=float(obj["flux"]),
                peak=float(obj["peak"]),
                a=a_val,
                b=b_val,
                theta=float(obj["theta"]),
                fwhm_estimate=fwhm_est,
                elongation=elongation,
                flag=int(obj["flag"]),
            )
        )
    return sources


def detect_stars(
    image: np.ndarray,
    bg_stats: Optional[BackgroundStats] = None,
    detection_threshold: float = DEFAULT_DETECT_THRESH,
    min_area: int = DEFAULT_MIN_AREA,
    deblend_nthresh: int = DEFAULT_DEBLEND_NTHRESH,
    deblend_cont: float = DEFAULT_DEBLEND_CONT,
    edge_margin: int = DEFAULT_EDGE_MARGIN,
    max_elongation: float = DEFAULT_MAX_ELONGATION,
) -> List[StarSource]:
    """
    Detect stars in an astronomical image using SEP.

    Parameters
    ----------
    image:
        2D float image array (raw, not background-subtracted).
    bg_stats:
        Pre-computed background statistics. If None, will be estimated.
    detection_threshold:
        Detection threshold in sigma above background RMS.
    min_area:
        Minimum source area in pixels.
    deblend_nthresh:
        Number of deblending thresholds.
    deblend_cont:
        Minimum contrast ratio for deblending.
    edge_margin:
        Pixels from image border to exclude sources.
    max_elongation:
        Maximum allowed a/b elongation ratio.

    Returns
    -------
    List of StarSource objects passing all quality filters.
    """
    import sep

    h, w = image.shape

    # Estimate background if not provided
    if bg_stats is None:
        bg_stats = estimate_background(image)

    # Prepare background-subtracted image for SEP
    # SEP needs C-contiguous native byte-order float64
    data_sub = subtract_background(image, bg_stats)
    data_sep = np.ascontiguousarray(data_sub, dtype=np.float64)
    if data_sep.dtype.byteorder not in ("=", "|", "<" if np.little_endian else ">"):
        data_sep = data_sep.byteswap().view(data_sep.dtype.newbyteorder("="))
        data_sep = np.ascontiguousarray(data_sep)

    # Use background RMS as the noise estimate for thresholding
    noise_level = max(bg_stats.background_rms, bg_stats.noise_mad, 1.0)

    sources = []

    # First attempt with default parameters
    try:
        objects = sep.extract(
            data_sep,
            thresh=detection_threshold,
            err=noise_level,
            minarea=min_area,
            deblend_nthresh=deblend_nthresh,
            deblend_cont=deblend_cont,
        )
        sources = _sep_to_sources(objects, (h, w))
        logger.debug("SEP detected %d raw sources.", len(sources))
    except Exception as exc:
        logger.warning("SEP extraction failed: %s. Retrying with relaxed parameters.", exc)
        try:
            objects = sep.extract(
                data_sep,
                thresh=detection_threshold * 1.5,
                err=noise_level,
                minarea=min_area * 2,
                deblend_nthresh=16,
                deblend_cont=0.01,
            )
            sources = _sep_to_sources(objects, (h, w))
            logger.debug("SEP retry detected %d raw sources.", len(sources))
        except Exception as exc2:
            logger.error("SEP extraction failed on retry: %s", exc2)
            return []

    # --- Quality filtering ---
    filtered = []
    n_edge = n_elongated = n_flux = 0

    for src in sources:
        # Skip edge sources
        if (
            src.x < edge_margin
            or src.x > w - edge_margin
            or src.y < edge_margin
            or src.y > h - edge_margin
        ):
            n_edge += 1
            continue

        # Skip highly elongated sources (cosmic rays, satellite trails, etc.)
        if src.elongation > max_elongation:
            n_elongated += 1
            continue

        # Skip very low flux sources
        min_flux = bg_stats.background_rms * detection_threshold
        if src.flux < min_flux:
            n_flux += 1
            continue

        # Skip sources with flags indicating serious extraction problems
        # SEP flags: 1=aperture truncated, 2=deblended, 4=saturated, 16=memory overflow
        if src.flag & 4:  # saturated
            logger.debug("Skipping saturated source at (%.1f, %.1f)", src.x, src.y)
            continue

        filtered.append(src)

    logger.debug(
        "Filtered: %d edge, %d elongated, %d low-flux. Kept %d / %d sources.",
        n_edge, n_elongated, n_flux, len(filtered), len(sources),
    )

    if len(filtered) == 0 and len(sources) > 0:
        logger.warning(
            "All %d detected sources were filtered out. "
            "Consider reducing detection threshold or edge margin.",
            len(sources),
        )

    return filtered


