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
    background_gradient: float = float("nan")    # (max-min)/median of 2D bg map; measures spatial non-uniformity

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


def _compute_background_gradient(image: np.ndarray, n_cells: int = 8) -> float:
    """
    Measure sky spatial non-uniformity by dividing the image into an n×n grid,
    computing the sigma-clipped sky median of each cell, then expressing the
    peak-to-valley variation in units of the per-pixel noise (σ).

    Returns (max_cell_bg - min_cell_bg) / noise_rms.

    Normalising by noise rather than by the sky median makes the metric
    camera- and gain-independent: it directly answers "how many noise standard
    deviations does the gradient span?"  This is critical because a gradient
    that looks dramatic after stretching (where the sky level dominates the
    denominator) may only be ~1 % of the sky ADU level but still hundreds of σ
    above noise — and therefore genuinely damaging to the data.

    This approach avoids SEP's background model, which sigma-clips bright
    regions and can produce a spuriously flat map even when part of the image
    is burned by sunrise or a cloud edge.

    Typical values
    --------------
    Uniform sky / good night:        ~5–30 σ
    Normal LP gradient (Bortle 9):   ~20–80 σ
    Severe gradient (sunrise/cloud): ~100–1000+ σ
    """
    h, w = image.shape
    cell_h = max(h // n_cells, 1)
    cell_w = max(w // n_cells, 1)

    cell_medians = []
    all_noise: list = []

    for i in range(n_cells):
        for j in range(n_cells):
            y0 = i * cell_h
            y1 = min((i + 1) * cell_h, h)
            x0 = j * cell_w
            x1 = min((j + 1) * cell_w, w)
            cell = image[y0:y1, x0:x1].ravel().astype(np.float64)
            if len(cell) < 16:
                continue
            # Sigma-clip at 3σ to exclude stars and hot pixels
            med = np.median(cell)
            mad = float(np.median(np.abs(cell - med)))
            sigma = 1.4826 * mad
            if sigma > 0:
                clipped = cell[np.abs(cell - med) < 3.0 * sigma]
            else:
                clipped = cell
            if len(clipped) > 0:
                cell_med = float(np.median(clipped))
                cell_noise = 1.4826 * float(np.median(np.abs(clipped - cell_med)))
                cell_medians.append(cell_med)
                if cell_noise > 0:
                    all_noise.append(cell_noise)

    if len(cell_medians) < 2:
        return float("nan")

    noise_rms = float(np.median(all_noise)) if all_noise else 0.0
    if noise_rms <= 0:
        return float("nan")

    return float((max(cell_medians) - min(cell_medians)) / noise_rms)


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
    gradient = _compute_background_gradient(image)

    return BackgroundStats(
        background_median=float(median),
        background_mean=float(mean),
        background_rms=float(std),
        noise_mad=float(noise_mad),
        background_map=None,
        background_gradient=gradient,
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

    mad = _mad(bg_map)
    noise_mad = 1.4826 * mad

    bg_median = float(np.median(bg_map))
    gradient = _compute_background_gradient(data)

    return BackgroundStats(
        background_median=bg_median,
        background_mean=float(np.mean(bg_map)),
        background_rms=float(bkg.globalrms),
        noise_mad=float(noise_mad),
        background_map=bg_map,
        background_gradient=gradient,
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
