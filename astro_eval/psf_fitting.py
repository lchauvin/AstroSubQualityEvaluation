"""
psf_fitting.py - PSF fitting for stellar FWHM and eccentricity measurement.

Fits Moffat profiles (with Gaussian fallback) to individual star cutouts
extracted from the image. Aggregates per-star results into session-level
statistics.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning

from .star_detection import StarSource

logger = logging.getLogger(__name__)

# Cutout half-size in pixels (full box = 2*HALF + 1)
DEFAULT_CUTOUT_HALF = 7       # => 15x15 box
MIN_FWHM_PIX = 0.5
MAX_FWHM_PIX = 30.0


@dataclass
class StarFitResult:
    """PSF fit result for a single star."""

    x: float
    y: float
    fwhm_pix: float         # FWHM in pixels
    eccentricity: float     # 0=round, 1=line
    amplitude: float        # fitted peak amplitude
    background: float       # fitted local background
    fit_residual: float     # MAD(fitted - actual) / amplitude
    fit_method: str         # 'moffat' or 'gaussian'
    success: bool
    beta: Optional[float] = None   # Moffat beta parameter (None for Gaussian fits)
    theta: float = 0.0             # PSF orientation angle in radians (major-axis position angle)


@dataclass
class PSFResult:
    """Aggregated PSF statistics from all fitted stars."""

    n_fitted: int
    fwhm_median: float      # pixels
    fwhm_mean: float        # pixels
    fwhm_std: float         # pixels
    eccentricity_median: float
    psf_residual_median: float
    beta_median: float      # Moffat beta median (nan for Gaussian-only sessions)
    individual: List[StarFitResult]

    # Spatial FWHM map: 5×5 grid of median FWHM (pixels) per image region; NaN = no stars in cell
    fwhm_map: Optional[List[List[float]]] = field(default=None)

    # Elongation direction statistics (circular statistics on theta)
    theta_mean: float = float("nan")          # mean PSF orientation angle (radians)
    theta_consistency: float = float("nan")   # mean resultant length R ∈ [0,1]; 1 = all aligned


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

def _moffat_2d(
    xy: Tuple[np.ndarray, np.ndarray],
    amplitude: float,
    x0: float,
    y0: float,
    alpha_x: float,
    alpha_y: float,
    beta: float,
    theta: float,
    background: float,
) -> np.ndarray:
    """
    2D elliptical Moffat profile.

    f(x,y) = A * [1 + ((x'/alpha_x)^2 + (y'/alpha_y)^2)]^(-beta) + B
    where x', y' are rotated coordinates.
    """
    x, y = xy
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    xp = cos_t * (x - x0) + sin_t * (y - y0)
    yp = -sin_t * (x - x0) + cos_t * (y - y0)

    alpha_x = max(abs(alpha_x), 1e-6)
    alpha_y = max(abs(alpha_y), 1e-6)
    beta = max(abs(beta), 0.5)

    r2 = (xp / alpha_x) ** 2 + (yp / alpha_y) ** 2
    return amplitude * (1.0 + r2) ** (-beta) + background


def _moffat_fwhm(alpha: float, beta: float) -> float:
    """FWHM of a circular Moffat profile: 2 * alpha * sqrt(2^(1/beta) - 1)."""
    if beta <= 0.5:
        return float("nan")
    return 2.0 * alpha * np.sqrt(2.0 ** (1.0 / beta) - 1.0)


def _gaussian_2d(
    xy: Tuple[np.ndarray, np.ndarray],
    amplitude: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    background: float,
) -> np.ndarray:
    """2D elliptical Gaussian profile."""
    x, y = xy
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    xp = cos_t * (x - x0) + sin_t * (y - y0)
    yp = -sin_t * (x - x0) + cos_t * (y - y0)

    sigma_x = max(abs(sigma_x), 1e-6)
    sigma_y = max(abs(sigma_y), 1e-6)

    return amplitude * np.exp(
        -0.5 * ((xp / sigma_x) ** 2 + (yp / sigma_y) ** 2)
    ) + background


def _gaussian_fwhm(sigma: float) -> float:
    """FWHM from Gaussian sigma: 2*sqrt(2*ln(2))*sigma."""
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(sigma)


# ---------------------------------------------------------------------------
# Per-star fitting
# ---------------------------------------------------------------------------

def _extract_cutout(
    image: np.ndarray,
    cx: float,
    cy: float,
    half: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Extract a square cutout around a star centre.

    Returns (cutout, xx, yy) meshgrids, or None if out of bounds.
    """
    h, w = image.shape
    ix, iy = int(round(cx)), int(round(cy))

    x0 = ix - half
    x1 = ix + half + 1
    y0 = iy - half
    y1 = iy + half + 1

    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None

    cutout = image[y0:y1, x0:x1].copy()
    xs = np.arange(x0, x1, dtype=np.float64)
    ys = np.arange(y0, y1, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    return cutout, xx, yy


def _fit_moffat(
    cutout: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    cx: float,
    cy: float,
    initial_a: float,
    initial_bg: float,
    initial_alpha: float = 2.0,
    initial_beta: float = 3.0,
) -> Optional[StarFitResult]:
    """Attempt Moffat fit to a cutout."""
    p0 = [initial_a, cx, cy, initial_alpha, initial_alpha,
          initial_beta, 0.0, initial_bg]

    bounds_lo = [0, cx - 5, cy - 5, 0.1, 0.1, 0.5, -np.pi / 2, -np.inf]
    bounds_hi = [np.inf, cx + 5, cy + 5, 20, 20, 20, np.pi / 2, np.inf]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(
                _moffat_2d,
                (xx.ravel(), yy.ravel()),
                cutout.ravel(),
                p0=p0,
                bounds=(bounds_lo, bounds_hi),
                maxfev=2000,
            )
    except Exception:
        return None

    amplitude, x0, y0, alpha_x, alpha_y, beta, theta, bg = popt

    if amplitude <= 0:
        return None

    fwhm_x = _moffat_fwhm(alpha_x, beta)
    fwhm_y = _moffat_fwhm(alpha_y, beta)

    if not (np.isfinite(fwhm_x) and np.isfinite(fwhm_y)):
        return None

    fwhm_avg = (fwhm_x + fwhm_y) / 2.0
    if not (MIN_FWHM_PIX <= fwhm_avg <= MAX_FWHM_PIX):
        return None

    # Eccentricity from semi-axes (alpha_x -> a, alpha_y -> b)
    ax = max(alpha_x, alpha_y)
    bx = min(alpha_x, alpha_y)
    ecc = float(np.sqrt(1.0 - (bx / ax) ** 2)) if ax > 0 else float("nan")

    # Residual
    fitted = _moffat_2d((xx, yy), *popt)
    residual_mad = float(np.median(np.abs(fitted - cutout)))
    norm_residual = residual_mad / amplitude if amplitude > 0 else float("nan")

    return StarFitResult(
        x=x0, y=y0,
        fwhm_pix=fwhm_avg,
        eccentricity=ecc,
        amplitude=amplitude,
        background=bg,
        fit_residual=norm_residual,
        fit_method="moffat",
        success=True,
        beta=float(beta),
        theta=float(theta),
    )


def _fit_gaussian(
    cutout: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    cx: float,
    cy: float,
    initial_a: float,
    initial_bg: float,
    initial_sigma: float = 2.0,
) -> Optional[StarFitResult]:
    """Attempt Gaussian fit to a cutout."""
    p0 = [initial_a, cx, cy, initial_sigma, initial_sigma, 0.0, initial_bg]
    bounds_lo = [0, cx - 5, cy - 5, 0.1, 0.1, -np.pi / 2, -np.inf]
    bounds_hi = [np.inf, cx + 5, cy + 5, 20, 20, np.pi / 2, np.inf]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(
                _gaussian_2d,
                (xx.ravel(), yy.ravel()),
                cutout.ravel(),
                p0=p0,
                bounds=(bounds_lo, bounds_hi),
                maxfev=2000,
            )
    except Exception:
        return None

    amplitude, x0, y0, sigma_x, sigma_y, theta, bg = popt

    if amplitude <= 0:
        return None

    fwhm_x = _gaussian_fwhm(sigma_x)
    fwhm_y = _gaussian_fwhm(sigma_y)
    fwhm_avg = (fwhm_x + fwhm_y) / 2.0

    if not (MIN_FWHM_PIX <= fwhm_avg <= MAX_FWHM_PIX):
        return None

    ax = max(sigma_x, sigma_y)
    bx = min(sigma_x, sigma_y)
    ecc = float(np.sqrt(1.0 - (bx / ax) ** 2)) if ax > 0 else float("nan")

    fitted = _gaussian_2d((xx, yy), *popt)
    residual_mad = float(np.median(np.abs(fitted - cutout)))
    norm_residual = residual_mad / amplitude if amplitude > 0 else float("nan")

    return StarFitResult(
        x=x0, y=y0,
        fwhm_pix=fwhm_avg,
        eccentricity=ecc,
        amplitude=amplitude,
        background=bg,
        fit_residual=norm_residual,
        fit_method="gaussian",
        success=True,
        theta=float(theta),
    )


def fit_star(
    image: np.ndarray,
    source: StarSource,
    cutout_half: int = DEFAULT_CUTOUT_HALF,
) -> Optional[StarFitResult]:
    """
    Fit PSF profile to a single star in the image.

    Attempts Moffat first; falls back to Gaussian on failure.

    Parameters
    ----------
    image:
        Full 2D float image (not background-subtracted; local bg is fitted).
    source:
        StarSource with approximate centroid.
    cutout_half:
        Half-size of the extraction box (full = 2*half+1).

    Returns
    -------
    StarFitResult or None if fitting completely fails.
    """
    result = _extract_cutout(image, source.x, source.y, cutout_half)
    if result is None:
        return None

    cutout, xx, yy = result

    # Sanity check: cutout should have finite values
    if not np.all(np.isfinite(cutout)):
        cutout = np.where(np.isfinite(cutout), cutout, np.nanmedian(cutout))

    # Initial parameter estimates
    bg_est = float(np.percentile(cutout, 10))
    peak_est = float(np.max(cutout)) - bg_est
    if peak_est <= 0:
        return None

    # Try Moffat first
    fit = _fit_moffat(
        cutout, xx, yy,
        cx=source.x, cy=source.y,
        initial_a=peak_est,
        initial_bg=bg_est,
        initial_alpha=max(source.a, 1.0),
        initial_beta=3.0,
    )
    if fit is not None:
        return fit

    # Fallback to Gaussian
    fit = _fit_gaussian(
        cutout, xx, yy,
        cx=source.x, cy=source.y,
        initial_a=peak_est,
        initial_bg=bg_est,
        initial_sigma=max(source.a, 1.0),
    )
    return fit


# ---------------------------------------------------------------------------
# Spatial FWHM map and circular theta statistics
# ---------------------------------------------------------------------------

_FWHM_MAP_GRID = 5   # spatial grid size (NxN cells)


def _compute_fwhm_spatial_map(
    individual: List[StarFitResult],
    image_shape: Tuple[int, int],
    grid: int = _FWHM_MAP_GRID,
) -> List[List[float]]:
    """
    Bin fitted stars into a grid×grid spatial grid and compute median FWHM per cell.

    Returns a (grid × grid) list of floats (pixels); NaN where no stars were fitted.
    Row 0 is the top of the image (low y), column 0 is the left (low x).
    """
    h, w = image_shape
    cell_h = h / grid
    cell_w = w / grid

    cells: List[List[List[float]]] = [[[] for _ in range(grid)] for _ in range(grid)]
    for star in individual:
        if not (np.isfinite(star.fwhm_pix) and np.isfinite(star.x) and np.isfinite(star.y)):
            continue
        ci = min(int(star.y / cell_h), grid - 1)
        cj = min(int(star.x / cell_w), grid - 1)
        cells[ci][cj].append(star.fwhm_pix)

    result: List[List[float]] = []
    for row in cells:
        result.append([
            float(np.median(cell)) if cell else float("nan")
            for cell in row
        ])
    return result


def _compute_theta_stats(individual: List[StarFitResult]) -> Tuple[float, float]:
    """
    Compute mean PSF orientation and alignment consistency using circular statistics.

    Uses the double-angle trick (map θ → 2θ) to handle the π-periodicity of PSF
    orientation (a star elongated at +45° looks identical at −135°).

    Returns
    -------
    (theta_mean_rad, theta_consistency)
        theta_mean_rad    : mean orientation angle in radians
        theta_consistency : mean resultant length R ∈ [0, 1]
                           0 = orientations are random, 1 = all stars elongated identically
    """
    thetas = [r.theta for r in individual if np.isfinite(r.theta)]
    if len(thetas) < 3:
        return float("nan"), float("nan")

    doubled = np.array(thetas) * 2.0
    sin_mean = float(np.mean(np.sin(doubled)))
    cos_mean = float(np.mean(np.cos(doubled)))

    R = float(np.sqrt(sin_mean ** 2 + cos_mean ** 2))
    mean_theta = float(np.arctan2(sin_mean, cos_mean)) / 2.0
    return mean_theta, R


# ---------------------------------------------------------------------------
# Batch fitting
# ---------------------------------------------------------------------------

def fit_psf(
    image: np.ndarray,
    sources: List[StarSource],
    cutout_half: int = DEFAULT_CUTOUT_HALF,
    max_stars: int = 200,
    image_shape: Optional[Tuple[int, int]] = None,
) -> PSFResult:
    """
    Fit PSF profiles to a collection of stars and aggregate results.

    Parameters
    ----------
    image:
        2D float image array.
    sources:
        List of detected stars from detect_stars().
    cutout_half:
        Half-size of per-star extraction boxes.
    max_stars:
        Maximum number of stars to fit (brightest selected).
    image_shape:
        (height, width) of the full image; used for spatial FWHM map computation.
        If None, uses image.shape.

    Returns
    -------
    PSFResult with aggregated statistics and optional spatial FWHM map.
    """
    shape = image_shape if image_shape is not None else image.shape

    if not sources:
        return PSFResult(
            n_fitted=0,
            fwhm_median=float("nan"),
            fwhm_mean=float("nan"),
            fwhm_std=float("nan"),
            eccentricity_median=float("nan"),
            psf_residual_median=float("nan"),
            beta_median=float("nan"),
            individual=[],
        )

    # Sort by flux descending; take brightest N stars for efficiency
    sorted_sources = sorted(sources, key=lambda s: s.flux, reverse=True)
    fitting_sources = sorted_sources[:max_stars]

    logger.debug("Fitting PSF to %d stars (of %d detected).", len(fitting_sources), len(sources))

    individual: List[StarFitResult] = []
    n_failed = 0

    for src in fitting_sources:
        result = fit_star(image, src, cutout_half=cutout_half)
        if result is not None and result.success:
            individual.append(result)
        else:
            n_failed += 1

    logger.debug(
        "PSF fitting: %d succeeded, %d failed.", len(individual), n_failed
    )

    if not individual:
        logger.warning("PSF fitting failed for all stars.")
        return PSFResult(
            n_fitted=0,
            fwhm_median=float("nan"),
            fwhm_mean=float("nan"),
            fwhm_std=float("nan"),
            eccentricity_median=float("nan"),
            psf_residual_median=float("nan"),
            beta_median=float("nan"),
            individual=[],
        )

    fwhms = np.array([r.fwhm_pix for r in individual if np.isfinite(r.fwhm_pix)])
    eccs = np.array([r.eccentricity for r in individual if np.isfinite(r.eccentricity)])
    residuals = np.array([r.fit_residual for r in individual if np.isfinite(r.fit_residual)])
    betas = np.array([r.beta for r in individual if r.beta is not None and np.isfinite(r.beta)])

    fwhm_median = float(np.median(fwhms)) if len(fwhms) > 0 else float("nan")
    fwhm_mean = float(np.mean(fwhms)) if len(fwhms) > 0 else float("nan")
    fwhm_std = float(np.std(fwhms)) if len(fwhms) > 1 else float("nan")
    ecc_median = float(np.median(eccs)) if len(eccs) > 0 else float("nan")
    residual_median = float(np.median(residuals)) if len(residuals) > 0 else float("nan")
    beta_median = float(np.median(betas)) if len(betas) > 0 else float("nan")

    # Spatial FWHM map
    fwhm_map = _compute_fwhm_spatial_map(individual, shape)

    # Elongation direction consistency (circular statistics on theta)
    theta_mean, theta_consistency = _compute_theta_stats(individual)

    return PSFResult(
        n_fitted=len(individual),
        fwhm_median=fwhm_median,
        fwhm_mean=fwhm_mean,
        fwhm_std=fwhm_std,
        eccentricity_median=ecc_median,
        psf_residual_median=residual_median,
        beta_median=beta_median,
        individual=individual,
        fwhm_map=fwhm_map,
        theta_mean=theta_mean,
        theta_consistency=theta_consistency,
    )
