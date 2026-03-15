"""
trail_detection.py - Satellite vs airplane trail detection.

Distinguishes between:
  - Satellite trails: single thin trail, uniform brightness → borderline
  - Airplane trails:  double contrail or strobe pattern     → rejected

Algorithm
---------
1. Downsample + uniform_filter (O(n) separable) to suppress nebula/gradients.
2. Threshold + dilate + label to find candidate elongated components.
3. PCA per component to get principal axis, length, aspect ratio.
4. For each qualifying component:
   a. Parallel-pair check: two components at the same angle → airplane.
   b. Vectorized cross-section sampling: single map_coordinates call across
      all sections simultaneously.
   c. NumPy peak counting on the mean profile: 2 peaks → airplane contrails.
5. Classify: airplane (rejected) vs satellite (borderline) vs none.

Efficiency notes
----------------
- Cross-sections: all N×M sample points issued in one map_coordinates call
  (no Python loop over sections).
- Peak detection: pure NumPy boolean indexing — no scipy.signal overhead.
- Parallel-pair check: O(k²) over k≤5 trail components — negligible.
- GPU not used: after uniform_filter the arrays are tiny (~750×500);
  PCIe transfer overhead would exceed compute time. Frame-level
  parallelism via --workers is the right axis here.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    gaussian_filter1d,
    label,
    map_coordinates,
    uniform_filter,
)

logger = logging.getLogger(__name__)

# Trail type constants
TRAIL_NONE      = "none"
TRAIL_SATELLITE = "satellite"
TRAIL_AIRPLANE  = "airplane"
TRAIL_UNKNOWN   = "unknown"


# ---------------------------------------------------------------------------
# Internal data structure
# ---------------------------------------------------------------------------

@dataclass
class _Component:
    """An elongated connected component that passed the trail shape filter."""
    centroid: np.ndarray   # (row, col) in downsampled image coords
    principal: np.ndarray  # unit vector along principal axis (dy, dx)
    angle_deg: float       # principal axis angle in [0°, 180°)
    length_pix: float      # estimated length in downsampled pixels
    length_fraction: float # length / image diagonal
    aspect: float          # sqrt(λ1/λ2) — elongation measure
    n_pixels: int


# ---------------------------------------------------------------------------
# Public result
# ---------------------------------------------------------------------------

@dataclass
class TrailResult:
    """Result of trail detection and classification for one frame."""
    n_trails: int
    trail_type: str        # one of TRAIL_* constants above
    max_length_fraction: float
    detected: bool         # any trail present
    rejected: bool         # True only for airplane trails


# ---------------------------------------------------------------------------
# Step 1: component detection
# ---------------------------------------------------------------------------

def _detect_components(
    image: np.ndarray,
    background_rms: float,
    downsample: int,
    detection_sigma: float,
    min_length_fraction: float,
    min_aspect_ratio: float,
) -> Tuple[np.ndarray, List[_Component]]:
    """
    Downsample, suppress structure, threshold, label, and return qualifying
    elongated components together with the processed (downsampled) image.

    Returns
    -------
    (small_image, components)
        small_image: the downsampled background-suppressed image (float64)
        components:  list of _Component objects that pass shape filters
    """
    h, w = image.shape
    ds = max(1, downsample)
    h_ds = (h // ds) * ds
    w_ds = (w // ds) * ds
    if h_ds == 0 or w_ds == 0:
        return np.zeros((1, 1)), []

    # Downsample by block-averaging
    small = (
        image[:h_ds, :w_ds]
        .reshape(h_ds // ds, ds, w_ds // ds, ds)
        .mean(axis=(1, 3))
    )
    h_s, w_s = small.shape
    diag = math.sqrt(h_s ** 2 + w_s ** 2)

    # Suppress large-scale structure with a box blur (O(n) separable)
    smooth_size = max(h_s // 15, w_s // 15, 5)
    suppressed = small - uniform_filter(small, size=smooth_size)

    # Noise estimate on the residual
    med = float(np.median(suppressed))
    mad = float(np.median(np.abs(suppressed - med)))
    sigma = 1.4826 * mad
    if sigma <= 0:
        sigma = max(background_rms / ds, 1.0)
    if sigma <= 0:
        return suppressed, []

    # Threshold + dilate
    binary = suppressed > (med + detection_sigma * sigma)
    binary = binary_dilation(binary, iterations=2)

    labeled, n_comps = label(binary)
    if n_comps == 0 or n_comps > 2000:
        if n_comps > 2000:
            logger.warning("Trail detection: %d components (noisy image?); skipping.", n_comps)
        return suppressed, []

    components: List[_Component] = []

    for idx in range(1, n_comps + 1):
        mask = labeled == idx
        n_pix = int(np.sum(mask))
        if n_pix < 15:
            continue

        rows, cols = np.where(mask)
        coords = np.column_stack([rows.astype(np.float64),
                                   cols.astype(np.float64)])
        if len(coords) < 3:
            continue

        centroid = coords.mean(axis=0)
        centred = coords - centroid
        cov = np.cov(centred.T)

        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue

        # eigh returns eigenvalues in ascending order; take the largest
        eigvals = np.abs(eigvals)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        if eigvals[1] < 1e-6:
            aspect = 1e6
        else:
            aspect = math.sqrt(eigvals[0] / eigvals[1])

        # Estimated length = 4 std along principal axis (~95% of component)
        length_pix = 4.0 * math.sqrt(eigvals[0])
        length_fraction = length_pix / diag

        if aspect < min_aspect_ratio or length_fraction < min_length_fraction:
            continue

        # Principal axis direction (unit vector: dy, dx)
        principal = eigvecs[:, 0]  # (dy, dx) in row-col space

        # Angle in [0°, 180°) — normalise direction so dy >= 0
        if principal[0] < 0:
            principal = -principal
        angle_deg = float(np.degrees(np.arctan2(principal[0], principal[1]))) % 180.0

        components.append(_Component(
            centroid=centroid,
            principal=principal,
            angle_deg=angle_deg,
            length_pix=length_pix,
            length_fraction=length_fraction,
            aspect=aspect,
            n_pixels=n_pix,
        ))
        logger.debug(
            "Component %d: n_pix=%d aspect=%.1f len_frac=%.3f angle=%.1f°",
            idx, n_pix, aspect, length_fraction, angle_deg,
        )

    return suppressed, components


# ---------------------------------------------------------------------------
# Step 2: cross-section sampling (fully vectorized)
# ---------------------------------------------------------------------------

def _sample_cross_sections(
    image: np.ndarray,
    comp: _Component,
    n_sections: int = 12,
    half_width: int = 18,
) -> np.ndarray:
    """
    Sample perpendicular cross-sections through a trail component.

    All N×M sample points are issued in a single map_coordinates call —
    no Python loop over individual sections.

    Parameters
    ----------
    image:
        Downsampled image (already background-suppressed).
    comp:
        Trail component to analyse.
    n_sections:
        Number of cross-sections to average (evenly spaced along trail).
    half_width:
        Half-width of each cross-section in pixels (downsampled).

    Returns
    -------
    mean_profile: 1D array of length (2*half_width + 1)
    """
    perp = np.array([-comp.principal[1], comp.principal[0]])  # perpendicular unit vec

    # Section centres: spaced between 20% and 80% of trail length to avoid ends
    half_len = comp.length_pix * 0.4   # stay within ±40% of estimated length
    t_vals = np.linspace(-half_len, half_len, n_sections)  # (n_sections,)

    # Perpendicular sample offsets
    s_vals = np.arange(-half_width, half_width + 1, dtype=np.float64)  # (M,)
    M = len(s_vals)

    # Section centres in image coords: (2, n_sections)
    centers = comp.centroid[:, None] + comp.principal[:, None] * t_vals  # (2, n_sections)

    # Full coordinate grid: (n_sections, M)
    rows = centers[0, :, None] + perp[0] * s_vals[None, :]
    cols = centers[1, :, None] + perp[1] * s_vals[None, :]

    # Single interpolation call for all n_sections × M points
    sampled = map_coordinates(
        image,
        [rows.ravel(), cols.ravel()],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).reshape(n_sections, M)

    return sampled.mean(axis=0)  # (M,)


# ---------------------------------------------------------------------------
# Step 3: NumPy peak counting (no scipy.signal)
# ---------------------------------------------------------------------------

def _count_peaks(
    profile: np.ndarray,
    smooth_sigma: float = 1.5,
    min_prominence: float = 0.25,
    min_separation: int = 3,
) -> int:
    """
    Count significant peaks in a 1D cross-section profile.

    Pure NumPy/scipy-ndimage — no scipy.signal overhead.

    Parameters
    ----------
    profile:
        1D array (mean cross-section).
    smooth_sigma:
        Gaussian smoothing sigma (pixels) to reduce noise before peak finding.
    min_prominence:
        Minimum normalised peak height [0–1] to be considered significant.
    min_separation:
        Minimum pixel gap between accepted peaks (suppresses double-counting).

    Returns
    -------
    Number of significant peaks (1 = satellite, 2 = airplane contrails).
    """
    if len(profile) < 5:
        return 0

    smoothed = gaussian_filter1d(profile.astype(np.float64), sigma=smooth_sigma)

    # Normalise to [0, 1]
    lo, hi = smoothed.min(), smoothed.max()
    if hi - lo < 1e-10:
        return 0
    norm = (smoothed - lo) / (hi - lo)

    # Local maxima: strictly greater than both neighbours (vectorised)
    is_peak = np.zeros(len(norm), dtype=bool)
    is_peak[1:-1] = (norm[1:-1] > norm[:-2]) & (norm[1:-1] > norm[2:])

    # Filter by prominence
    peak_indices = np.where(is_peak & (norm >= min_prominence))[0]

    if len(peak_indices) == 0:
        return 0

    # Enforce minimum separation: greedy scan (already sorted by index)
    accepted = [peak_indices[0]]
    for idx in peak_indices[1:]:
        if idx - accepted[-1] >= min_separation:
            accepted.append(idx)

    return len(accepted)


# ---------------------------------------------------------------------------
# Step 4: parallel-pair check
# ---------------------------------------------------------------------------

def _has_parallel_pair(
    components: List[_Component],
    angle_tol_deg: float = 12.0,
) -> bool:
    """
    Return True if any two components share nearly the same trail angle.

    O(k²) over k components (typically ≤ 3) — negligible cost.
    """
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            diff = abs(components[i].angle_deg - components[j].angle_deg) % 180.0
            diff = min(diff, 180.0 - diff)
            if diff < angle_tol_deg:
                logger.debug(
                    "Parallel pair found: angles %.1f° and %.1f° (diff %.1f°)",
                    components[i].angle_deg, components[j].angle_deg, diff,
                )
                return True
    return False


# ---------------------------------------------------------------------------
# Step 5: classify a single component
# ---------------------------------------------------------------------------

def _classify_component(
    image: np.ndarray,
    comp: _Component,
    n_sections: int,
    half_width: int,
    peak_min_prominence: float,
    peak_min_separation: int,
) -> str:
    """
    Classify one trail component as TRAIL_SATELLITE or TRAIL_AIRPLANE
    based on its cross-section profile.

    Returns TRAIL_UNKNOWN if the cross-section is ambiguous.
    """
    profile = _sample_cross_sections(image, comp,
                                     n_sections=n_sections,
                                     half_width=half_width)
    n_peaks = _count_peaks(profile,
                           min_prominence=peak_min_prominence,
                           min_separation=peak_min_separation)
    logger.debug(
        "Component angle=%.1f° length_frac=%.3f → %d cross-section peak(s)",
        comp.angle_deg, comp.length_fraction, n_peaks,
    )

    if n_peaks >= 2:
        return TRAIL_AIRPLANE
    if n_peaks == 1:
        return TRAIL_SATELLITE
    return TRAIL_UNKNOWN   # profile too flat to decide


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_trails(
    image: np.ndarray,
    background_median: float,
    background_rms: float,
    downsample: int = 4,
    detection_sigma: float = 5.0,
    min_length_fraction: float = 0.15,
    min_aspect_ratio: float = 10.0,
    n_sections: int = 12,
    half_width: int = 18,
    peak_min_prominence: float = 0.25,
    peak_min_separation: int = 3,
    parallel_angle_tol_deg: float = 12.0,
) -> TrailResult:
    """
    Detect and classify satellite / airplane trails in an image.

    Classification logic (in order of confidence):
    1. Two components at the same angle         → AIRPLANE  (parallel contrails)
    2. One component with 2 cross-section peaks → AIRPLANE  (merged contrails)
    3. One component with 1 cross-section peak  → SATELLITE (single trail)
    4. Ambiguous cross-section                  → UNKNOWN   (treated as satellite)

    Only AIRPLANE is flagged for hard rejection. SATELLITE is informational.

    Parameters
    ----------
    image:
        Raw 2D float64 image.
    background_median, background_rms:
        From estimate_background() — used as noise fallback.
    downsample:
        Downsampling factor (default 4). Increase to 6–8 for very large images.
    detection_sigma:
        Threshold above residual noise for trail pixels.
    min_length_fraction:
        Minimum trail length / diagonal. Default 0.15 (15% of diagonal).
    min_aspect_ratio:
        Minimum PCA aspect ratio to qualify as a trail. Default 10.
    n_sections:
        Number of perpendicular cross-sections to average.
    half_width:
        Half-width of each cross-section in downsampled pixels.
    peak_min_prominence:
        Minimum normalised peak height for peak counting.
    peak_min_separation:
        Minimum pixel gap between counted peaks (downsampled units).
    parallel_angle_tol_deg:
        Maximum angle difference between two components to call them parallel.

    Returns
    -------
    TrailResult
    """
    suppressed, components = _detect_components(
        image, background_rms, downsample,
        detection_sigma, min_length_fraction, min_aspect_ratio,
    )

    if not components:
        return TrailResult(
            n_trails=0, trail_type=TRAIL_NONE,
            max_length_fraction=0.0, detected=False, rejected=False,
        )

    n_trails = len(components)
    max_len_frac = max(c.length_fraction for c in components)

    # --- Parallel-pair check (fastest, most reliable for clear double trails) ---
    if len(components) >= 2 and _has_parallel_pair(components, parallel_angle_tol_deg):
        logger.info("Airplane trail: parallel component pair detected (%d components).", n_trails)
        return TrailResult(
            n_trails=n_trails, trail_type=TRAIL_AIRPLANE,
            max_length_fraction=max_len_frac, detected=True, rejected=True,
        )

    # --- Cross-section analysis for each component ---
    # Classify each component; airplane wins over satellite (conservative)
    overall_type = TRAIL_SATELLITE  # default if any trail found

    for comp in components:
        ctype = _classify_component(
            suppressed, comp,
            n_sections=n_sections,
            half_width=half_width,
            peak_min_prominence=peak_min_prominence,
            peak_min_separation=peak_min_separation,
        )
        logger.info(
            "Trail component: length=%.1f%% aspect=%.1f → %s",
            comp.length_fraction * 100, comp.aspect, ctype,
        )
        if ctype == TRAIL_AIRPLANE:
            overall_type = TRAIL_AIRPLANE
            break  # airplane found — no need to check further

    rejected = overall_type == TRAIL_AIRPLANE
    return TrailResult(
        n_trails=n_trails,
        trail_type=overall_type,
        max_length_fraction=max_len_frac,
        detected=True,
        rejected=rejected,
    )
