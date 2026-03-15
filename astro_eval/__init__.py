"""
astro_eval - Astrophotography sub-frame quality evaluation tool.

Evaluates FITS sub-frames for quality metrics including:
- PSF / FWHM measurement
- Star count and eccentricity
- Background noise and SNR estimation
- Composite scoring and rejection decisions
"""

__version__ = "0.1.0"
__author__ = "AstroEval"

from .image_loader import load_fits, load_xisf, load_image, FITSData
from .background import estimate_background, BackgroundStats
from .star_detection import detect_stars, StarSource
from .psf_fitting import fit_psf, PSFResult
from .metrics import compute_star_metrics, compute_gas_metrics, FrameMetrics
from .scoring import (
    compute_session_statistics,
    compute_rejection_flags,
    compute_star_score,
    compute_gas_score,
    SessionStats,
    RejectionFlags,
)
from .report import generate_csv_report, generate_html_report, generate_multi_filter_html_report
from .cli import main

__all__ = [
    "__version__",
    "load_fits",
    "load_xisf",
    "load_image",
    "FITSData",
    "estimate_background",
    "BackgroundStats",
    "detect_stars",
    "StarSource",
    "fit_psf",
    "PSFResult",
    "compute_star_metrics",
    "compute_gas_metrics",
    "FrameMetrics",
    "compute_session_statistics",
    "compute_rejection_flags",
    "compute_star_score",
    "compute_gas_score",
    "SessionStats",
    "RejectionFlags",
    "generate_csv_report",
    "generate_html_report",
    "generate_multi_filter_html_report",
    "main",
]
