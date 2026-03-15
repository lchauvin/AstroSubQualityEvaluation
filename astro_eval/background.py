"""
background.py - Sky background estimation and noise metrics.

Uses sigma-clipped statistics (astropy) and optionally SEP for
robust background estimation suitable for astrophotography frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BackgroundStats:
    """Sky background estimation results."""

    background_median: float
    background_mean: float
    background_rms: float    # standard deviation of background
    noise_mad: float         # MAD-based noise: 1.4826 * MAD
    background_map: Optional[np.ndarray] = None  # 2D background model (from SEP)

    @property
    def snr_proxy(self) -> float:
        """Signal-to-noise proxy: ratio of median to noise."""
        if self.noise_mad > 0:
            return self.background_median / self.noise_mad
        return float("nan")


def _mad(data: np.ndarray) -> float:
    """Median Absolute Deviation of a flattened array."""
    flat = data.ravel()
    med = np.median(flat)
    return float(np.median(np.abs(flat - med)))


def estimate_background_sigma_clip(
    image: np.ndarray,
    sigma: float = 3.0,
    maxiters: int = 5,
) -> BackgroundStats:
    """
    Estimate background using sigma-clipped statistics (astropy).

    Parameters
    ----------
    image:
        2D float image array.
    sigma:
        Clipping sigma level.
    maxiters:
        Maximum clipping iterations.

    Returns
    -------
    BackgroundStats with statistical estimates.
    """
    from astropy.stats import sigma_clipped_stats

    mean, median, std = sigma_clipped_stats(
        image, sigma=sigma, maxiters=maxiters
    )

    mad = _mad(image)
    noise_mad = 1.4826 * mad

    return BackgroundStats(
        background_median=float(median),
        background_mean=float(mean),
        background_rms=float(std),
        noise_mad=float(noise_mad),
        background_map=None,
    )


def estimate_background_sep(
    image: np.ndarray,
    box_size: int = 64,
    filter_size: int = 3,
) -> BackgroundStats:
    """
    Estimate background using SEP (Source Extractor Python).

    SEP creates a spatially varying background model that accounts
    for gradients and large-scale illumination variations.

    Parameters
    ----------
    image:
        2D float image array.
    box_size:
        Size of background mesh boxes in pixels.
    filter_size:
        Median filter size for background smoothing.

    Returns
    -------
    BackgroundStats including 2D background map.
    """
    import sep

    # SEP requires C-contiguous float64 in native byte order
    data = np.ascontiguousarray(image, dtype=np.float64).astype(
        np.float64, copy=False
    )
    if data.dtype.byteorder not in ("=", "|", "<" if np.little_endian else ">"):
        data = data.byteswap().view(data.dtype.newbyteorder("="))
        data = np.ascontiguousarray(data)

    bkg = sep.Background(data, bw=box_size, bh=box_size,
                         fw=filter_size, fh=filter_size)

    bg_map = bkg.back()
    bkg_subtracted = data - bg_map

    # Compute statistics on background-subtracted image
    mad = _mad(bg_map)
    noise_mad = 1.4826 * mad

    return BackgroundStats(
        background_median=float(np.median(bg_map)),
        background_mean=float(np.mean(bg_map)),
        background_rms=float(bkg.globalrms),
        noise_mad=float(noise_mad),
        background_map=bg_map,
    )


def estimate_background(
    image: np.ndarray,
    use_sep: bool = True,
    sigma_clip: float = 3.0,
) -> BackgroundStats:
    """
    Estimate sky background, falling back gracefully between methods.

    Tries SEP first (spatially aware), falls back to sigma-clipped stats.

    Parameters
    ----------
    image:
        2D float image array.
    use_sep:
        If True, attempt SEP-based estimation first.
    sigma_clip:
        Sigma clipping level for astropy fallback.

    Returns
    -------
    BackgroundStats instance.
    """
    if use_sep:
        try:
            return estimate_background_sep(image)
        except Exception as exc:
            logger.warning(
                "SEP background estimation failed (%s); falling back to sigma-clipped stats.",
                exc,
            )

    return estimate_background_sigma_clip(image, sigma=sigma_clip)


def subtract_background(
    image: np.ndarray, bg_stats: BackgroundStats
) -> np.ndarray:
    """
    Subtract background from image.

    Uses the 2D background map if available (from SEP), otherwise
    subtracts the scalar median.

    Parameters
    ----------
    image:
        2D float image array.
    bg_stats:
        BackgroundStats from estimate_background().

    Returns
    -------
    Background-subtracted image (float64).
    """
    image = image.astype(np.float64)
    if bg_stats.background_map is not None:
        return image - bg_stats.background_map
    return image - bg_stats.background_median
