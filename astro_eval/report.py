"""
report.py - CSV and HTML report generation.

Generates machine-readable CSV output and optional rich HTML report
with distribution plots embedded as base64 images.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import logging
import math
import os
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .metrics import FrameMetrics, ScoringWeights
from .scoring import FrameResult, SessionStats

logger = logging.getLogger(__name__)


def _url_encode(filename: str) -> str:
    """Percent-encode a filename for safe use in a URL path segment."""
    return urllib.parse.quote(filename, safe="")


# CSV column order
CSV_COLUMNS = [
    "filename",
    "mode",
    "filter",
    "exptime_s",
    "gain",
    "ccd_temp_c",
    "pixel_scale_arcsec",
    "n_stars",
    "fwhm_median_arcsec",
    "fwhm_mean_arcsec",
    "fwhm_std_arcsec",
    "eccentricity_median",
    "psf_residual_median",
    "snr_weight",
    "psf_signal_weight",
    "wfwhm_arcsec",
    "moffat_beta",
    "snr_estimate",
    "background_median",
    "background_rms",
    "noise_mad",
    "score",
    "rejected",
    "rejection_reasons",
    # Per-criterion flags
    "n_trails",
    "trail_length_fraction",
    "trail_type",
    "flag_airplane_trail",
    "flag_satellite_trail",
    "flag_high_fwhm",
    "flag_high_eccentricity",
    "flag_low_stars",
    "flag_low_snr_weight",
    "flag_high_residual",
    "flag_high_noise",
    "flag_high_background",
    "flag_low_snr",
    "error",
]


def _fmt(v, precision: int = 4) -> str:
    """Format a numeric value for CSV output."""
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        return f"{v:.{precision}f}"
    return str(v)


def _result_to_row(result: FrameResult) -> dict:
    """Convert a FrameResult to a flat dict for CSV output."""
    m = result.metrics
    r = result.rejection

    row = {
        "filename": m.filename,
        "mode": m.mode,
        "filter": m.filter_name or "",
        "exptime_s": _fmt(m.exptime),
        "gain": _fmt(m.gain),
        "ccd_temp_c": _fmt(m.ccd_temp),
        "pixel_scale_arcsec": _fmt(m.pixel_scale),
        "n_stars": str(m.n_stars),
        "fwhm_median_arcsec": _fmt(m.fwhm_median),
        "fwhm_mean_arcsec": _fmt(m.fwhm_mean),
        "fwhm_std_arcsec": _fmt(m.fwhm_std),
        "eccentricity_median": _fmt(m.eccentricity_median),
        "psf_residual_median": _fmt(m.psf_residual_median),
        "snr_weight": _fmt(m.snr_weight),
        "psf_signal_weight": _fmt(m.psf_signal_weight),
        "wfwhm_arcsec": _fmt(m.wfwhm),
        "moffat_beta": _fmt(m.moffat_beta, precision=3),
        "snr_estimate": _fmt(m.snr_estimate),
        "background_median": _fmt(m.background_median),
        "background_rms": _fmt(m.background_rms),
        "noise_mad": _fmt(m.noise_mad),
        "score": _fmt(result.score),
        "rejected": "1" if r.rejected else "0",
        "rejection_reasons": "|".join(r.rejection_reasons),
        "n_trails": str(m.n_trails),
        "trail_length_fraction": _fmt(m.trail_length_fraction),
        "trail_type": m.trail_type,
        "flag_airplane_trail":  "1" if r.flags.get("airplane_trail") else "0",
        "flag_satellite_trail": "1" if r.flags.get("satellite_trail") else "0",
        "flag_high_fwhm": "1" if r.flags.get("high_fwhm") else "0",
        "flag_high_eccentricity": "1" if r.flags.get("high_eccentricity") else "0",
        "flag_low_stars": "1" if r.flags.get("low_stars") else "0",
        "flag_low_snr_weight": "1" if r.flags.get("low_snr_weight") else "0",
        "flag_high_residual": "1" if r.flags.get("high_residual") else "0",
        "flag_high_noise": "1" if r.flags.get("high_noise") else "0",
        "flag_high_background": "1" if r.flags.get("high_background") else "0",
        "flag_low_snr": "1" if r.flags.get("low_snr") else "0",
        "error": m.error or "",
    }
    return row


def generate_csv_report(
    results: List[FrameResult],
    output_path: str | Path,
) -> None:
    """
    Write evaluation results to a CSV file.

    Parameters
    ----------
    results:
        List of FrameResult objects from evaluate_session().
    output_path:
        Path to the output CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8", errors="replace") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for result in sorted(results, key=lambda r: r.metrics.filename):
            writer.writerow(_result_to_row(result))

    logger.info("CSV report written to %s", output_path)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _make_plot_base64(fig) -> str:
    """Render a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _parse_obs_time(s: Optional[str]):
    """Parse an ISO-8601 DATE-OBS string into a datetime, or return None."""
    if not s:
        return None
    from datetime import datetime
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _plot_quality_trend(
    results: List[FrameResult],
    filter_name: str = "",
) -> Optional[str]:
    """
    Return base64 PNG of quality-over-time trend chart, or None on failure.

    X-axis: observation datetime (DATE-OBS) when available for ≥50% of frames,
            otherwise frame index sorted by filename.
    Left Y-axis:  FWHM (arcsec) for star mode, background RMS (ADU) for gas mode.
    Right Y-axis: composite quality score [0, 1].
    Rejected frames are marked with red circles.
    """
    if not results:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # Sort by filename for consistent ordering
        ordered = sorted(results, key=lambda r: r.metrics.filename)

        # Determine mode from first result
        is_gas = ordered[0].metrics.mode == "gas"

        # Try to use datetime x-axis
        times = [_parse_obs_time(r.metrics.obs_time) for r in ordered]
        use_time = sum(1 for t in times if t is not None) >= len(ordered) * 0.5

        if use_time:
            xs = [t if t is not None else times[i - 1] for i, t in enumerate(times)]
        else:
            xs = list(range(len(ordered)))

        primary_vals = [
            r.metrics.background_rms if is_gas else r.metrics.fwhm_median
            for r in ordered
        ]
        primary_label = "Background RMS (ADU)" if is_gas else "FWHM (arcsec)"
        score_vals = [r.score for r in ordered]
        rejected = [r.rejection.rejected for r in ordered]

        finite_primary = [v for v in primary_vals if math.isfinite(v)]
        if not finite_primary:
            return None

        fig, ax1 = plt.subplots(figsize=(16, 4))
        ax2 = ax1.twinx()

        primary_color = "#5b9bd5"
        score_color   = "#f0ad4e"

        # Primary metric line
        valid_xs = [x for x, v in zip(xs, primary_vals) if math.isfinite(v)]
        valid_ys = [v for v in primary_vals if math.isfinite(v)]
        ax1.plot(valid_xs, valid_ys, color=primary_color, linewidth=1.2, zorder=2)

        # Rejected frame markers
        rej_xs = [x for x, v, r in zip(xs, primary_vals, rejected) if r and math.isfinite(v)]
        rej_ys = [v for v, r in zip(primary_vals, rejected) if r and math.isfinite(v)]
        if rej_xs:
            ax1.scatter(rej_xs, rej_ys, color="#d9534f", s=35, zorder=5,
                        label="Rejected")

        # Session median line
        med = float(np.nanmedian(finite_primary))
        ax1.axhline(med, color=primary_color, linestyle="--", linewidth=1,
                    alpha=0.6, label=f"Median: {med:.2f}")

        # Score line (right axis)
        valid_score_xs = [x for x, v in zip(xs, score_vals) if math.isfinite(v)]
        valid_score_ys = [v for v in score_vals if math.isfinite(v)]
        ax2.plot(valid_score_xs, valid_score_ys, color=score_color,
                 linewidth=1.0, linestyle=":", zorder=3, label="Score")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Quality Score", color=score_color, fontsize=9)
        ax2.tick_params(axis="y", labelcolor=score_color)

        ax1.set_ylabel(primary_label, color=primary_color, fontsize=9)
        ax1.tick_params(axis="y", labelcolor=primary_color)
        ax1.set_xlabel("Observation time" if use_time else "Frame index", fontsize=9)
        title = f"Quality Trend — {filter_name}" if filter_name else "Quality Trend"
        ax1.set_title(title, fontsize=10)
        ax1.grid(axis="both", alpha=0.2)

        if use_time:
            fig.autofmt_xdate()
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

        plt.tight_layout()
        img = _make_plot_base64(fig)
        plt.close(fig)
        return img
    except Exception as exc:
        logger.warning("Failed to generate quality trend plot: %s", exc)
        return None


def _plot_fwhm_distribution(
    results: List[FrameResult],
) -> Optional[str]:
    """Return base64 PNG of FWHM distribution plot, or None on failure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fwhms = [r.metrics.fwhm_median for r in results
                 if math.isfinite(r.metrics.fwhm_median)]
        if not fwhms:
            return None

        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = [
            "#d9534f" if r.rejection.flags.get("high_fwhm") else "#5cb85c"
            for r in results if math.isfinite(r.metrics.fwhm_median)
        ]
        ax.bar(
            range(len(fwhms)),
            fwhms,
            color=colors,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.axhline(np.median(fwhms), color="#f0ad4e", linestyle="--",
                   linewidth=1.5, label=f'Median: {np.median(fwhms):.2f}"')
        ax.set_xlabel("Frame index")
        ax.set_ylabel('FWHM (arcsec)')
        ax.set_title("FWHM Distribution")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        img = _make_plot_base64(fig)
        plt.close(fig)
        return img
    except Exception as exc:
        logger.warning("Failed to generate FWHM plot: %s", exc)
        return None


def _plot_star_count_distribution(
    results: List[FrameResult],
) -> Optional[str]:
    """Return base64 PNG of star count distribution plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        counts = [r.metrics.n_stars for r in results]
        if not counts:
            return None

        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = [
            "#d9534f" if r.rejection.flags.get("low_stars") else "#5cb85c"
            for r in results
        ]
        ax.bar(range(len(counts)), counts, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(np.median(counts), color="#f0ad4e", linestyle="--",
                   linewidth=1.5, label=f"Median: {np.median(counts):.0f}")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Star count")
        ax.set_title("Star Count Distribution")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        img = _make_plot_base64(fig)
        plt.close(fig)
        return img
    except Exception as exc:
        logger.warning("Failed to generate star count plot: %s", exc)
        return None


def _plot_score_distribution(
    results: List[FrameResult],
) -> Optional[str]:
    """Return base64 PNG of composite score distribution."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        scores = [r.score for r in results if math.isfinite(r.score)]
        if not scores:
            return None

        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = [
            "#d9534f" if r.rejection.rejected else
            ("#f0ad4e" if r.score < 0.5 else "#5cb85c")
            for r in results if math.isfinite(r.score)
        ]
        ax.bar(range(len(scores)), scores, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(0.5, color="#aaa", linestyle=":", linewidth=1, label="Score=0.5")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Quality Score")
        ax.set_title("Quality Score Distribution")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        img = _make_plot_base64(fig)
        plt.close(fig)
        return img
    except Exception as exc:
        logger.warning("Failed to generate score plot: %s", exc)
        return None


def _plot_background_distribution(
    results: List[FrameResult],
) -> Optional[str]:
    """Return base64 PNG of background noise distribution."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rms_vals = [r.metrics.background_rms for r in results
                    if math.isfinite(r.metrics.background_rms)]
        if not rms_vals:
            return None

        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = [
            "#d9534f" if r.rejection.flags.get("high_noise") else "#5cb85c"
            for r in results if math.isfinite(r.metrics.background_rms)
        ]
        ax.bar(range(len(rms_vals)), rms_vals, color=colors,
               edgecolor="white", linewidth=0.5)
        ax.axhline(np.median(rms_vals), color="#f0ad4e", linestyle="--",
                   linewidth=1.5, label=f"Median: {np.median(rms_vals):.1f}")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Background RMS (ADU)")
        ax.set_title("Background Noise Distribution")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        img = _make_plot_base64(fig)
        plt.close(fig)
        return img
    except Exception as exc:
        logger.warning("Failed to generate background plot: %s", exc)
        return None


def _row_color(result: FrameResult) -> str:
    """Return CSS background color class for a frame table row."""
    if result.rejection.rejected:
        return "#fde8e8"  # light red
    if result.score < 0.5:
        return "#fff8e1"  # light yellow
    return "#e8f5e9"      # light green


def _format_cell(value, precision: int = 3) -> str:
    """Format a value for HTML table display, with data-value for JS sorting."""
    if value is None:
        return '<td data-value="">—</td>'
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return '<td data-value="">—</td>'
        return f'<td data-value="{value}">{value:.{precision}f}</td>'
    if isinstance(value, bool):
        if value:
            return '<td data-value="1" style="color:#d9534f;font-weight:bold;">YES</td>'
        return '<td data-value="0">—</td>'
    if isinstance(value, int):
        return f'<td data-value="{value}">{value}</td>'
    return f'<td data-value="{value}">{value}</td>'


def _scoring_info_html(results: List[FrameResult], weights: Optional[ScoringWeights]) -> str:
    """
    Return an HTML snippet describing the evaluation mode and scoring equation
    with the actual weights used.
    """
    if not results:
        return ""

    w = weights or ScoringWeights()
    mode = results[0].metrics.mode

    # Determine mode label and build equation terms
    if mode == "gas":
        mode_label = "Gas / Narrowband"
        mode_color = "#8e44ad"
        terms = []
        if w.gas_snr:
            terms.append(f'<span class="eq-term">{w.gas_snr:.2f} × norm(snr_estimate)</span>')
        if w.gas_noise:
            terms.append(f'<span class="eq-term">{w.gas_noise:.2f} × (1 − norm(background_rms))</span>')
        if w.gas_bg:
            terms.append(f'<span class="eq-term">{w.gas_bg:.2f} × (1 − norm(background_median))</span>')
        if w.gas_stars:
            terms.append(f'<span class="eq-term">{w.gas_stars:.2f} × norm(n_stars)</span>')
        if w.gas_psfsw:
            terms.append(f'<span class="eq-term">{w.gas_psfsw:.2f} × norm(psf_signal_weight)</span>')
        desc = (
            "Optimised for faint emission nebulae. snr_estimate and background_rms are the primary "
            "quality discriminators. psf_signal_weight adds a star-sharpness bonus. "
            "n_stars is used as a sky-transparency proxy."
        )
    else:
        mode_label = "Star / Broadband"
        mode_color = "#2980b9"
        terms = []
        if w.star_fwhm:
            terms.append(f'<span class="eq-term">{w.star_fwhm:.2f} × (1 − norm(fwhm_median))</span>')
        if w.star_ecc:
            terms.append(f'<span class="eq-term">{w.star_ecc:.2f} × (1 − norm(eccentricity_median))</span>')
        if w.star_stars:
            terms.append(f'<span class="eq-term">{w.star_stars:.2f} × norm(n_stars)</span>')
        if w.star_psfsw:
            terms.append(f'<span class="eq-term">{w.star_psfsw:.2f} × norm(psf_signal_weight)</span>')
        if w.star_snr:
            terms.append(f'<span class="eq-term">{w.star_snr:.2f} × norm(snr_weight)</span>')
        desc = (
            "Optimised for broadband / RGB imaging. psf_signal_weight combines star amplitude, "
            "noise, and fwhm² in one metric. fwhm_median adds an independent linear seeing penalty."
        )

    total_w = sum([
        w.gas_snr, w.gas_noise, w.gas_bg, w.gas_stars, w.gas_psfsw,
    ] if mode == "gas" else [
        w.star_fwhm, w.star_ecc, w.star_stars, w.star_psfsw, w.star_snr,
    ])
    weight_warn = (
        f' <span style="color:#e74c3c;font-size:0.85em;">'
        f'⚠ weights sum to {total_w:.2f}, not 1.0</span>'
        if abs(total_w - 1.0) > 0.01 else ""
    )

    eq_html = ' <span class="eq-plus">+</span> '.join(terms) if terms else "—"

    return f"""
  <div class="scoring-info">
    <div class="scoring-mode" style="border-left:4px solid {mode_color};">
      <strong>Mode:</strong> {mode_label}
    </div>
    <div class="scoring-eq">
      <strong>Score =</strong> {eq_html}{weight_warn}
    </div>
    <div class="scoring-desc">{desc}</div>
    <div class="scoring-note">
      All metrics are normalised to [0,&nbsp;1] across the session.
      Score&nbsp;=&nbsp;1.0 is the best frame in the session;
      trail penalties are applied multiplicatively after scoring.
    </div>
  </div>"""


def generate_html_report(
    results: List[FrameResult],
    session_stats: Dict[str, SessionStats],
    output_path: str | Path,
    source_dir: Optional[str | Path] = None,
    weights: Optional[ScoringWeights] = None,
) -> None:
    """
    Generate a rich HTML report with summary, per-frame table, and distribution plots.

    Plots are embedded as base64 PNG images — the HTML file is fully self-contained.

    Parameters
    ----------
    results:
        List of FrameResult objects.
    session_stats:
        Session statistics from compute_session_statistics().
    output_path:
        Path to the output HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = len(results)
    n_rejected = sum(1 for r in results if r.rejection.rejected)
    n_accepted = n_total - n_rejected

    # Per-criterion rejection counts
    all_flags = set()
    for r in results:
        all_flags.update(r.rejection.flags.keys())
    flag_counts = {
        flag: sum(1 for r in results if r.rejection.flags.get(flag, False))
        for flag in sorted(all_flags)
    }

    # Generate plots
    fwhm_plot  = _plot_fwhm_distribution(results)
    star_plot  = _plot_star_count_distribution(results)
    score_plot = _plot_score_distribution(results)
    bg_plot    = _plot_background_distribution(results)
    trend_plot = _plot_quality_trend(results)

    def img_tag(b64: Optional[str], alt: str) -> str:
        if b64 is None:
            return f'<p class="no-data">No data for {alt}</p>'
        return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;">'

    # Source directory for the move script (embedded as a JS string literal)
    if source_dir is not None:
        _src = str(Path(source_dir).resolve())
    elif results:
        _src = str(Path(results[0].metrics.filepath).parent.resolve())
    else:
        _src = ""
    # Escape backslashes for embedding in a JS string literal
    source_dir_js = json.dumps(_src)

    scoring_info = _scoring_info_html(results, weights)

    # --- Build HTML ---
    sorted_results = sorted(results, key=lambda r: r.metrics.filename)

    rows_html = []
    for i, result in enumerate(sorted_results):
        m = result.metrics
        r = result.rejection
        bg = _row_color(result)

        status_val = "1" if r.rejected else "0"
        status_cell = (
            f'<td data-value="{status_val}" style="color:#d9534f;font-weight:bold;">REJECTED</td>'
            if r.rejected else
            f'<td data-value="{status_val}" style="color:#5cb85c;font-weight:bold;">ACCEPTED</td>'
        )
        reasons_text = " | ".join(r.rejection_reasons) or "—"
        filter_text = m.filter_name or "—"

        if m.trail_type == "airplane":
            trail_pct = (
                f" ({m.trail_length_fraction * 100:.0f}%)"
                if math.isfinite(m.trail_length_fraction) else ""
            )
            trail_cell = (
                f'<td data-value="2" style="color:#d9534f;font-weight:bold;">'
                f'✈ airplane{trail_pct}</td>'
            )
        elif m.trail_type in ("satellite", "unknown"):
            trail_pct = (
                f" ({m.trail_length_fraction * 100:.0f}%)"
                if math.isfinite(m.trail_length_fraction) else ""
            )
            trail_cell = (
                f'<td data-value="1" style="color:#f0ad4e;font-weight:bold;">'
                f'★ satellite{trail_pct}</td>'
            )
        else:
            trail_cell = '<td data-value="0">—</td>'

        is_rejected_int = 1 if r.rejected else 0
        checked_attr = "checked" if r.rejected else ""
        row = (
            f'<tr style="background-color:{bg};">'
            f'<td style="text-align:center;">'
            f'<input type="checkbox" class="frame-select" '
            f'data-filename="{m.filename}" data-rejected="{is_rejected_int}" {checked_attr}>'
            f'</td>'
            f'<td data-value="{i+1}">{i+1}</td>'
            f'<td data-value="{m.filename}">'
            f'<a class="preview-link" href="/preview/{_url_encode(m.filename)}" target="_blank">{m.filename}</a>'
            f'</td>'
            f'<td data-value="{m.mode}">{m.mode}</td>'
            f'<td data-value="{filter_text}">{filter_text}</td>'
            f"{_format_cell(m.exptime)}"
            f"{_format_cell(m.n_stars)}"
            f"{_format_cell(m.fwhm_median)}"
            f"{_format_cell(m.eccentricity_median)}"
            f"{_format_cell(m.snr_weight, precision=1)}"
            f"{_format_cell(m.psf_signal_weight, precision=1)}"
            f"{_format_cell(m.wfwhm)}"
            f"{_format_cell(m.moffat_beta, precision=2)}"
            f"{_format_cell(m.snr_estimate)}"
            f"{_format_cell(m.background_rms, precision=1)}"
            f"{trail_cell}"
            f"{_format_cell(result.score)}"
            f"{status_cell}"
            f'<td data-value="{reasons_text}">{reasons_text}</td>'
            "</tr>\n"
        )
        rows_html.append(row)

    flag_rows = "".join(
        f"<tr><td>{flag}</td><td>{count}</td></tr>\n"
        for flag, count in flag_counts.items()
    )

    # Session stats table
    stat_rows = []
    for name, ss in session_stats.items():
        if ss.count == 0:
            continue
        stat_rows.append(
            f"<tr>"
            f"<td>{name}</td>"
            f"<td>{ss.count}</td>"
            f"<td>{ss.median:.4g}</td>"
            f"<td>{ss.std:.4g}</td>"
            f"<td>{ss.min_val:.4g}</td>"
            f"<td>{ss.max_val:.4g}</td>"
            "</tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AstroEval Quality Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0; padding: 20px;
      background: #f5f5f5;
      color: #333;
    }}
    h1, h2, h3 {{ color: #2c3e50; }}
    h1 {{ border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
    .summary-cards {{
      display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px;
    }}
    .card {{
      background: white;
      border-radius: 8px;
      padding: 16px 24px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.1);
      min-width: 120px; text-align: center;
    }}
    .card .value {{ font-size: 2em; font-weight: bold; }}
    .card.total .value {{ color: #3498db; }}
    .card.accepted .value {{ color: #27ae60; }}
    .card.rejected .value {{ color: #e74c3c; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: white;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 2px 6px rgba(0,0,0,0.08);
      font-size: 0.88em;
    }}
    th {{
      background: #2c3e50;
      color: white;
      padding: 10px 8px;
      text-align: left;
      white-space: nowrap;
      user-select: none;
    }}
    th.sortable {{
      cursor: pointer;
    }}
    th.sortable:hover {{
      background: #3d5570;
    }}
    th.sort-asc::after  {{ content: " ▲"; font-size: 0.75em; opacity: 0.9; }}
    th.sort-desc::after {{ content: " ▼"; font-size: 0.75em; opacity: 0.9; }}
    td {{
      padding: 7px 8px;
      border-bottom: 1px solid #eee;
      white-space: nowrap;
    }}
    tr:last-child td {{ border-bottom: none; }}
    .plots-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
      gap: 20px;
      margin: 20px 0;
    }}
    .plot-box {{
      background: white;
      border-radius: 8px;
      padding: 12px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }}
    .no-data {{ color: #999; font-style: italic; padding: 20px; text-align: center; }}
    section {{ margin-bottom: 32px; }}
    .session-table {{ max-width: 700px; }}
    .flag-table {{ max-width: 400px; }}
    footer {{ color: #999; font-size: 0.8em; margin-top: 40px; padding-top: 12px;
              border-top: 1px solid #ddd; }}
    .scoring-info {{
      background: white; border-radius: 8px; padding: 14px 18px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.08); margin-top: 16px;
    }}
    .scoring-mode {{ padding: 6px 10px; margin-bottom: 10px;
                     background: #f8f9fa; border-radius: 4px; font-size: 0.92em; }}
    .scoring-eq {{ font-family: "SFMono-Regular", Consolas, monospace;
                   font-size: 0.85em; line-height: 1.9; margin-bottom: 8px;
                   background: #f8f9fa; padding: 8px 12px; border-radius: 4px; }}
    .eq-term {{ display: inline-block; background: #e8f4fd;
                border: 1px solid #bee3f8; border-radius: 3px;
                padding: 1px 6px; margin: 1px; white-space: nowrap; }}
    .eq-plus {{ color: #999; padding: 0 2px; }}
    .scoring-desc {{ color: #555; margin-bottom: 6px; font-size: 0.88em; }}
    .scoring-note {{ color: #999; font-size: 0.82em; font-style: italic; }}
    .frame-controls {{
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 8px; flex-wrap: wrap; gap: 8px;
    }}
    .frame-controls-left {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
    .frame-controls button {{
      padding: 6px 12px; border: 1px solid #ccc; border-radius: 4px;
      background: #f8f9fa; cursor: pointer; font-size: 0.85em;
    }}
    .frame-controls button:hover {{ background: #e9ecef; }}
    .btn-move {{
      background: #e74c3c !important; color: white !important;
      border-color: #c0392b !important; font-weight: bold;
    }}
    .btn-move:hover {{ background: #c0392b !important; }}
    .sel-count {{ font-size: 0.85em; color: #666; }}
    .move-status {{
      font-size: 0.85em; color: #27ae60; margin-bottom: 8px;
      min-height: 1.2em;
    }}
  </style>
</head>
<body>
  <h1>AstroEval Quality Report</h1>

  <section>
    <h2>Summary</h2>
    <div class="summary-cards">
      <div class="card total"><div class="value">{n_total}</div><div>Total Frames</div></div>
      <div class="card accepted"><div class="value">{n_accepted}</div><div>Accepted</div></div>
      <div class="card rejected"><div class="value">{n_rejected}</div><div>Rejected</div></div>
      <div class="card"><div class="value">{n_accepted/n_total*100:.1f}%</div><div>Pass Rate</div></div>
    </div>

    <h3>Rejection Breakdown</h3>
    <table class="flag-table">
      <thead><tr><th>Criterion</th><th>Frames Flagged</th></tr></thead>
      <tbody>{flag_rows}</tbody>
    </table>
    <h3>Scoring</h3>
    {scoring_info}
  </section>

  <section>
    <h2>Distribution Plots</h2>
    <div class="plots-grid">
      <div class="plot-box"><h3>FWHM</h3>{img_tag(fwhm_plot, "FWHM Distribution")}</div>
      <div class="plot-box"><h3>Star Count</h3>{img_tag(star_plot, "Star Count Distribution")}</div>
      <div class="plot-box"><h3>Quality Score</h3>{img_tag(score_plot, "Score Distribution")}</div>
      <div class="plot-box"><h3>Background Noise</h3>{img_tag(bg_plot, "Background Noise Distribution")}</div>
    </div>
    <div class="plot-box" style="margin-top:20px"><h3>Quality Trend Over Time</h3>{'<img src="data:image/png;base64,' + trend_plot + '" alt="Quality Trend" style="width:100%;">' if trend_plot else '<p class="no-data">No data for Quality Trend</p>'}</div>
  </section>

  <section>
    <h2>Session Statistics</h2>
    <table class="session-table">
      <thead>
        <tr><th>Metric</th><th>N</th><th>Median</th><th>Std</th><th>Min</th><th>Max</th></tr>
      </thead>
      <tbody>{"".join(stat_rows)}</tbody>
    </table>
  </section>

  <section>
    <h2>Per-Frame Results</h2>
    <p>
      <span style="background:#e8f5e9;padding:2px 8px;border-radius:3px;">Green</span> = accepted &nbsp;
      <span style="background:#fff8e1;padding:2px 8px;border-radius:3px;">Yellow</span> = borderline &nbsp;
      <span style="background:#fde8e8;padding:2px 8px;border-radius:3px;">Red</span> = rejected
    </p>
    <div class="frame-controls">
      <div class="frame-controls-left">
        <button onclick="selectRejected()">&#9746; Select rejected</button>
        <button onclick="selectAll()">&#9745; Select all</button>
        <button onclick="deselectAll()">&#9744; Deselect all</button>
        <span id="selected-count" class="sel-count">0 selected</span>
      </div>
      <div class="frame-controls-right">
        <button id="btn-move" class="btn-move" onclick="downloadMoveScript()">
          &#128193; Move to _REJECTED&hellip;
        </button>
      </div>
    </div>
    <div id="move-status" class="move-status"></div>
    <table id="frames-table" class="sortable-table">
      <thead>
        <tr>
          <th style="width:32px;text-align:center;" title="Select/deselect all">
            <input type="checkbox" id="cb-toggle-all" title="Toggle all">
          </th>
          <th class="sortable">#</th>
          <th class="sortable">Filename</th>
          <th class="sortable">Mode</th>
          <th class="sortable">Filter</th>
          <th class="sortable">Exp (s)</th>
          <th class="sortable">Stars</th>
          <th class="sortable">FWHM (")</th>
          <th class="sortable">Ecc</th>
          <th class="sortable">SNR wt</th>
          <th class="sortable" title="PSFSignalWeight: combines amplitude, FWHM penalty (1/FWHM²), and noise">PSFSW</th>
          <th class="sortable" title="wFWHM = FWHM / sqrt(n_stars): lower is better (Siril metric)">wFWHM</th>
          <th class="sortable" title="Moffat beta: atmospheric seeing index (typical 2.5–5)">β</th>
          <th class="sortable">SNR est</th>
          <th class="sortable">BG RMS</th>
          <th class="sortable">Trails</th>
          <th class="sortable">Score</th>
          <th class="sortable">Status</th>
          <th class="sortable">Reasons</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows_html)}
      </tbody>
    </table>
  </section>

  <footer>
    Generated by <strong>astro-eval v0.1.0</strong> &mdash;
    Astrophotography Sub-Frame Quality Evaluation Tool
  </footer>

  <script>
    const SOURCE_DIR = {source_dir_js};

    // -----------------------------------------------------------------------
    // Checkbox helpers
    // -----------------------------------------------------------------------
    function updateCount() {{
      const all = document.querySelectorAll('.frame-select');
      const n   = document.querySelectorAll('.frame-select:checked').length;
      document.getElementById('selected-count').textContent =
        n === 0 ? 'none selected' : n + ' selected';
      const tog = document.getElementById('cb-toggle-all');
      tog.indeterminate = n > 0 && n < all.length;
      tog.checked = n === all.length;
    }}

    function selectRejected() {{
      document.querySelectorAll('.frame-select').forEach(function(cb) {{
        cb.checked = cb.dataset.rejected === '1';
      }});
      updateCount();
    }}

    function selectAll() {{
      document.querySelectorAll('.frame-select').forEach(function(cb) {{ cb.checked = true; }});
      updateCount();
    }}

    function deselectAll() {{
      document.querySelectorAll('.frame-select').forEach(function(cb) {{ cb.checked = false; }});
      updateCount();
    }}

    document.getElementById('cb-toggle-all').addEventListener('change', function() {{
      if (this.checked) selectAll(); else deselectAll();
    }});

    document.querySelectorAll('.frame-select').forEach(function(cb) {{
      cb.addEventListener('change', updateCount);
    }});

    updateCount();

    // -----------------------------------------------------------------------
    // Preview links: only active in server mode
    // -----------------------------------------------------------------------
    var SERVER_MODE = (window.location.protocol === 'http:' || window.location.protocol === 'https:');
    if (!SERVER_MODE) {{
      document.querySelectorAll('.preview-link').forEach(function(a) {{
        a.removeAttribute('href');
        a.style.cursor = 'default';
        a.style.color = 'inherit';
        a.style.textDecoration = 'none';
      }});
    }}

    // -----------------------------------------------------------------------
    // Move: direct via server (http://) or .bat download (file://)
    // -----------------------------------------------------------------------

    function downloadMoveScript() {{
      var filenames = Array.from(document.querySelectorAll('.frame-select:checked'))
                          .map(function(cb) {{ return cb.dataset.filename; }});
      if (filenames.length === 0) {{
        alert('No frames selected. Check at least one frame to move.');
        return;
      }}

      if (SERVER_MODE) {{
        // --- Direct move via HTTP server ---
        var btn = document.getElementById('btn-move');
        btn.disabled = true;
        btn.textContent = 'Moving\u2026';

        fetch('/move', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{filenames: filenames}})
        }})
        .then(function(r) {{ return r.json(); }})
        .then(function(data) {{
          var statusEl = document.getElementById('move-status');
          if (data.errors.length === 0) {{
            statusEl.style.color = '#27ae60';
            statusEl.textContent = '\u2713 Moved ' + data.moved.length + ' file(s) to _REJECTED';
          }} else {{
            statusEl.style.color = '#e74c3c';
            statusEl.textContent = 'Moved ' + data.moved.length + ', errors: ' + data.errors.join(' | ');
          }}
          // Dim and uncheck moved rows
          data.moved.forEach(function(fn) {{
            var cb = document.querySelector('.frame-select[data-filename="' + fn + '"]');
            if (cb) {{
              cb.checked = false;
              cb.disabled = true;
              cb.closest('tr').style.opacity = '0.35';
            }}
          }});
          updateCount();
          btn.disabled = false;
          btn.textContent = '\U0001F4C1 Move to _REJECTED\u2026';
        }})
        .catch(function(err) {{
          document.getElementById('move-status').style.color = '#e74c3c';
          document.getElementById('move-status').textContent = 'Error: ' + err;
          btn.disabled = false;
          btn.textContent = '\U0001F4C1 Move to _REJECTED\u2026';
        }});

      }} else {{
        // --- Fallback: generate and download a .bat script ---
        var src  = SOURCE_DIR;
        var dest = src + '\\\\_REJECTED';
        var lines = [
          '@echo off', 'setlocal',
          'set "SRC=' + src + '"', 'set "DEST=' + dest + '"',
          'mkdir "%DEST%" 2>nul', ''
        ];
        filenames.forEach(function(fn) {{
          lines.push('move "%SRC%\\\\' + fn + '" "%DEST%"');
        }});
        lines.push('', 'echo.', 'echo Moved ' + filenames.length + ' file(s) to _REJECTED', 'pause');

        var blob = new Blob([lines.join('\\r\\n')], {{type: 'text/plain'}});
        var url  = URL.createObjectURL(blob);
        var a    = document.createElement('a');
        a.href = url; a.download = 'move_rejected.bat';
        document.body.appendChild(a); a.click();
        document.body.removeChild(a); URL.revokeObjectURL(url);

        document.getElementById('move-status').textContent =
          '\u2713 move_rejected.bat downloaded \u2014 double-click it to move '
          + filenames.length + ' file(s) to _REJECTED';
      }}
    }}

    // -----------------------------------------------------------------------
    // Column sort (per-table state stored in data attributes)
    // -----------------------------------------------------------------------
    (function () {{
      function cellValue(row, col) {{
        var td = row.querySelectorAll('td')[col];
        return td ? (td.dataset.value !== undefined ? td.dataset.value : td.textContent.trim()) : '';
      }}
      function compare(a, b, col) {{
        var av = cellValue(a, col), bv = cellValue(b, col);
        if (av === '' && bv === '') return 0;
        if (av === '') return 1;
        if (bv === '') return -1;
        var an = parseFloat(av), bn = parseFloat(bv);
        if (!isNaN(an) && !isNaN(bn)) return an - bn;
        return av.localeCompare(bv);
      }}
      document.querySelectorAll('.sortable-table thead th.sortable').forEach(function(th) {{
        th.addEventListener('click', function() {{
          var table  = th.closest('table');
          var tbody  = table.querySelector('tbody');
          var rows   = Array.from(tbody.querySelectorAll('tr'));
          var allThs = Array.from(table.querySelectorAll('thead th'));
          var col    = allThs.indexOf(th);
          var sortCol = parseInt(table.dataset.sortCol !== undefined ? table.dataset.sortCol : '-1');
          var sortAsc = table.dataset.sortAsc !== 'false';
          if (sortCol === col) {{ sortAsc = !sortAsc; }}
          else {{ sortCol = col; sortAsc = true; }}
          table.dataset.sortCol = sortCol;
          table.dataset.sortAsc = sortAsc;
          rows.sort(function(a, b) {{
            return sortAsc ? compare(a, b, col) : -compare(a, b, col);
          }});
          rows.forEach(function(r) {{ tbody.appendChild(r); }});
          allThs.forEach(function(h, i) {{
            h.classList.remove('sort-asc', 'sort-desc');
            if (i === col) h.classList.add(sortAsc ? 'sort-asc' : 'sort-desc');
          }});
        }});
      }});
    }})();

    // -----------------------------------------------------------------------
    // Watch mode: auto-refresh via Server-Sent Events
    // -----------------------------------------------------------------------
    if (SERVER_MODE) {{
      var _sse = new EventSource('/events');
      _sse.onmessage = function(e) {{
        if (e.data === 'reload') {{ window.location.reload(); }}
      }};
    }}
  </script>
</body>
</html>
"""

    # Sanitize lone surrogates that some FITS/XISF headers introduce
    html_safe = html.encode("utf-8", errors="replace").decode("utf-8")
    output_path.write_text(html_safe, encoding="utf-8")
    logger.info("HTML report written to %s", output_path)


# ---------------------------------------------------------------------------
# Multi-filter tabbed HTML report
# ---------------------------------------------------------------------------

def _filter_id_safe(name: str) -> str:
    """Make a filter name safe for HTML IDs and JS identifiers."""
    return "".join(c if c.isalnum() else "_" for c in name) or "unknown"


def _build_panel_html(
    fid: str,
    results: List[FrameResult],
    session_stats: Dict[str, SessionStats],
    source_dir_js: str,
    weights: Optional[ScoringWeights] = None,
) -> str:
    """
    Return the inner HTML for one filter's tab panel.
    fid must already be safe for HTML IDs (use _filter_id_safe).
    source_dir_js is a json.dumps()-encoded path string.
    """
    n_total = len(results)
    if n_total == 0:
        return '<p class="no-data">No frames processed for this filter.</p>'

    n_rejected = sum(1 for r in results if r.rejection.rejected)
    n_accepted = n_total - n_rejected

    all_flags: set = set()
    for r in results:
        all_flags.update(r.rejection.flags.keys())
    flag_counts = {
        flag: sum(1 for r in results if r.rejection.flags.get(flag, False))
        for flag in sorted(all_flags)
    }
    flag_rows = "".join(
        f"<tr><td>{flag}</td><td>{count}</td></tr>\n"
        for flag, count in flag_counts.items()
    )

    scoring_info = _scoring_info_html(results, weights)

    fwhm_plot  = _plot_fwhm_distribution(results)
    star_plot  = _plot_star_count_distribution(results)
    score_plot = _plot_score_distribution(results)
    bg_plot    = _plot_background_distribution(results)
    trend_plot = _plot_quality_trend(results, filter_name=fid)

    def img_tag(b64: Optional[str], alt: str) -> str:
        if b64 is None:
            return f'<p class="no-data">No data for {alt}</p>'
        return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;">'

    stat_rows = []
    for name, ss in session_stats.items():
        if ss.count == 0:
            continue
        stat_rows.append(
            f"<tr><td>{name}</td><td>{ss.count}</td><td>{ss.median:.4g}</td>"
            f"<td>{ss.std:.4g}</td><td>{ss.min_val:.4g}</td><td>{ss.max_val:.4g}</td></tr>\n"
        )

    sorted_results = sorted(results, key=lambda r: r.metrics.filename)
    rows_html = []
    for i, result in enumerate(sorted_results):
        m = result.metrics
        r = result.rejection
        bg = _row_color(result)

        status_val = "1" if r.rejected else "0"
        status_cell = (
            f'<td data-value="{status_val}" style="color:#d9534f;font-weight:bold;">REJECTED</td>'
            if r.rejected else
            f'<td data-value="{status_val}" style="color:#5cb85c;font-weight:bold;">ACCEPTED</td>'
        )
        reasons_text = " | ".join(r.rejection_reasons) or "—"
        filter_text  = m.filter_name or "—"

        if m.trail_type == "airplane":
            trail_pct  = f" ({m.trail_length_fraction*100:.0f}%)" if math.isfinite(m.trail_length_fraction) else ""
            trail_cell = f'<td data-value="2" style="color:#d9534f;font-weight:bold;">\u2708 airplane{trail_pct}</td>'
        elif m.trail_type in ("satellite", "unknown"):
            trail_pct  = f" ({m.trail_length_fraction*100:.0f}%)" if math.isfinite(m.trail_length_fraction) else ""
            trail_cell = f'<td data-value="1" style="color:#f0ad4e;font-weight:bold;">\u2605 satellite{trail_pct}</td>'
        else:
            trail_cell = '<td data-value="0">\u2014</td>'

        is_rej     = 1 if r.rejected else 0
        checked    = "checked" if r.rejected else ""
        prev_url   = f"/preview/{_url_encode(fid)}/{_url_encode(m.filename)}"

        rows_html.append(
            f'<tr style="background-color:{bg};">'
            f'<td style="text-align:center;"><input type="checkbox" class="frame-select" '
            f'data-filename="{m.filename}" data-rejected="{is_rej}" {checked}></td>'
            f'<td data-value="{i+1}">{i+1}</td>'
            f'<td data-value="{m.filename}">'
            f'<a class="preview-link" href="{prev_url}" target="_blank">{m.filename}</a></td>'
            f'<td data-value="{m.mode}">{m.mode}</td>'
            f'<td data-value="{filter_text}">{filter_text}</td>'
            f"{_format_cell(m.exptime)}"
            f"{_format_cell(m.n_stars)}"
            f"{_format_cell(m.fwhm_median)}"
            f"{_format_cell(m.eccentricity_median)}"
            f"{_format_cell(m.snr_weight, precision=1)}"
            f"{_format_cell(m.psf_signal_weight, precision=1)}"
            f"{_format_cell(m.wfwhm)}"
            f"{_format_cell(m.moffat_beta, precision=2)}"
            f"{_format_cell(m.snr_estimate)}"
            f"{_format_cell(m.background_rms, precision=1)}"
            f"{trail_cell}"
            f"{_format_cell(result.score)}"
            f"{status_cell}"
            f'<td data-value="{reasons_text}">{reasons_text}</td>'
            "</tr>\n"
        )

    pass_pct = f"{n_accepted/n_total*100:.1f}" if n_total else "0.0"
    return f"""
  <div class="summary-cards">
    <div class="card total"><div class="value">{n_total}</div><div>Total</div></div>
    <div class="card accepted"><div class="value">{n_accepted}</div><div>Accepted</div></div>
    <div class="card rejected"><div class="value">{n_rejected}</div><div>Rejected</div></div>
    <div class="card"><div class="value">{pass_pct}%</div><div>Pass Rate</div></div>
  </div>
  <h3>Rejection Breakdown</h3>
  <table class="flag-table">
    <thead><tr><th>Criterion</th><th>Flagged</th></tr></thead>
    <tbody>{flag_rows}</tbody>
  </table>
  <h3>Scoring</h3>
  {scoring_info}
  <h2>Distribution Plots</h2>
  <div class="plots-grid">
    <div class="plot-box"><h3>FWHM</h3>{img_tag(fwhm_plot, "FWHM")}</div>
    <div class="plot-box"><h3>Star Count</h3>{img_tag(star_plot, "Stars")}</div>
    <div class="plot-box"><h3>Quality Score</h3>{img_tag(score_plot, "Score")}</div>
    <div class="plot-box"><h3>Background Noise</h3>{img_tag(bg_plot, "BG Noise")}</div>
  </div>
  <div class="plot-box" style="margin-top:20px"><h3>Quality Trend Over Time</h3>{'<img src="data:image/png;base64,' + trend_plot + '" alt="Quality Trend" style="width:100%;">' if trend_plot else '<p class="no-data">No data for Quality Trend</p>'}</div>
  <h2>Session Statistics</h2>
  <table class="session-table">
    <thead><tr><th>Metric</th><th>N</th><th>Median</th><th>Std</th><th>Min</th><th>Max</th></tr></thead>
    <tbody>{"".join(stat_rows)}</tbody>
  </table>
  <h2>Per-Frame Results</h2>
  <p>
    <span style="background:#e8f5e9;padding:2px 8px;border-radius:3px;">Green</span> = accepted &nbsp;
    <span style="background:#fff8e1;padding:2px 8px;border-radius:3px;">Yellow</span> = borderline &nbsp;
    <span style="background:#fde8e8;padding:2px 8px;border-radius:3px;">Red</span> = rejected
  </p>
  <div class="frame-controls">
    <div class="frame-controls-left">
      <button onclick="selectRejected('{fid}')">&#9746; Select rejected</button>
      <button onclick="selectAll('{fid}')">&#9745; Select all</button>
      <button onclick="deselectAll('{fid}')">&#9744; Deselect all</button>
      <span id="selected-count-{fid}" class="sel-count">none selected</span>
    </div>
    <div class="frame-controls-right">
      <button id="btn-move-{fid}" class="btn-move" onclick="downloadMoveScript('{fid}')">
        &#128193; Move to _REJECTED&hellip;
      </button>
    </div>
  </div>
  <div id="move-status-{fid}" class="move-status"></div>
  <table id="frames-table-{fid}" class="sortable-table">
    <thead>
      <tr>
        <th style="width:32px;text-align:center;">
          <input type="checkbox" id="cb-toggle-all-{fid}" title="Toggle all"
                 onchange="if(this.checked) selectAll('{fid}'); else deselectAll('{fid}');">
        </th>
        <th class="sortable">#</th>
        <th class="sortable">Filename</th>
        <th class="sortable">Mode</th>
        <th class="sortable">Filter</th>
        <th class="sortable">Exp (s)</th>
        <th class="sortable">Stars</th>
        <th class="sortable">FWHM (&quot;)</th>
        <th class="sortable">Ecc</th>
        <th class="sortable">SNR wt</th>
        <th class="sortable" title="PSFSignalWeight: combines amplitude, FWHM penalty (1/FWHM²), and noise">PSFSW</th>
        <th class="sortable" title="wFWHM = FWHM / sqrt(n_stars): lower is better (Siril metric)">wFWHM</th>
        <th class="sortable" title="Moffat beta: atmospheric seeing index (typical 2.5–5)">β</th>
        <th class="sortable">SNR est</th>
        <th class="sortable">BG RMS</th>
        <th class="sortable">Trails</th>
        <th class="sortable">Score</th>
        <th class="sortable">Status</th>
        <th class="sortable">Reasons</th>
      </tr>
    </thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table>
"""


def _build_summary_html(
    filter_data: Dict[str, tuple],
) -> str:
    """Return HTML for the cross-filter summary tab."""
    all_results = [r for results, _ in filter_data.values() for r in results]
    n_total    = len(all_results)
    n_rejected = sum(1 for r in all_results if r.rejection.rejected)
    n_accepted = n_total - n_rejected
    pass_pct   = f"{n_accepted/n_total*100:.1f}" if n_total else "0.0"

    filter_rows = []
    for fid, (results, session_stats) in filter_data.items():
        nt  = len(results)
        na  = sum(1 for r in results if not r.rejection.rejected)
        nr  = nt - na
        pp  = f"{na/nt*100:.1f}%" if nt else "—"

        fwhms  = [r.metrics.fwhm_median for r in results if math.isfinite(r.metrics.fwhm_median)]
        snrs   = [r.metrics.snr_estimate for r in results if math.isfinite(r.metrics.snr_estimate)]
        scores = [r.score for r in results if math.isfinite(r.score)]

        med_fwhm  = f"{float(np.median(fwhms)):.2f}&quot;" if fwhms else "—"
        med_snr   = f"{float(np.median(snrs)):.2f}"        if snrs  else "—"
        med_score = f"{float(np.median(scores)):.3f}"      if scores else "—"

        filter_rows.append(
            f"<tr>"
            f"<td><strong>{fid}</strong></td>"
            f"<td>{nt}</td><td style='color:#27ae60'>{na}</td>"
            f"<td style='color:#e74c3c'>{nr}</td><td>{pp}</td>"
            f"<td>{med_fwhm}</td><td>{med_snr}</td><td>{med_score}</td>"
            "</tr>\n"
        )

    return f"""
  <div class="summary-cards">
    <div class="card total"><div class="value">{n_total}</div><div>Total Frames</div></div>
    <div class="card accepted"><div class="value">{n_accepted}</div><div>Accepted</div></div>
    <div class="card rejected"><div class="value">{n_rejected}</div><div>Rejected</div></div>
    <div class="card"><div class="value">{pass_pct}%</div><div>Pass Rate</div></div>
  </div>
  <h2>Per-Filter Overview</h2>
  <table style="max-width:800px;">
    <thead>
      <tr>
        <th>Filter</th><th>Total</th><th>Accepted</th><th>Rejected</th>
        <th>Pass Rate</th><th>Median FWHM</th><th>Median SNR</th><th>Median Score</th>
      </tr>
    </thead>
    <tbody>{"".join(filter_rows)}</tbody>
  </table>
"""


_TAB_CSS = """
    .tab-nav {
      display: flex; gap: 4px; flex-wrap: wrap;
      border-bottom: 2px solid #3498db; margin-bottom: 0; padding-top: 8px;
    }
    .tab-btn {
      padding: 10px 20px; border: none; border-radius: 6px 6px 0 0;
      background: #ddd; cursor: pointer; font-size: 0.95em;
      transition: background 0.15s;
    }
    .tab-btn.active { background: #3498db; color: white; font-weight: bold; }
    .tab-btn:hover:not(.active) { background: #bbb; }
    .tab-badge {
      font-size: 0.8em; background: rgba(0,0,0,0.15);
      border-radius: 10px; padding: 1px 7px; margin-left: 4px;
    }
    .tab-pane { padding-top: 16px; }
"""


def generate_multi_filter_html_report(
    filter_data: Dict[str, tuple],
    output_path: str | Path,
    source_dirs: Dict[str, "Path"],
    weights: Optional[ScoringWeights] = None,
) -> None:
    """
    Generate a tabbed HTML report with one tab per filter plus a Summary tab.

    Parameters
    ----------
    filter_data:
        {filter_name: (list[FrameResult], dict[str, SessionStats])}
    output_path:
        Path to the output HTML file.
    source_dirs:
        {filter_name: Path} mapping filter names to their source directories.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON map of filter → source dir for JS move/bat script
    source_dirs_js = json.dumps({
        k: str(Path(v).resolve()) for k, v in source_dirs.items()
    })

    # Build tab navigation buttons
    tab_btns = ['<button class="tab-btn active" data-tab="summary">Summary</button>']
    for fid, (results, _) in filter_data.items():
        na  = sum(1 for r in results if not r.rejection.rejected)
        nt  = len(results)
        fid_safe = _filter_id_safe(fid)
        tab_btns.append(
            f'<button class="tab-btn" data-tab="{fid_safe}">'
            f'{fid} <span class="tab-badge">{na}/{nt}</span></button>'
        )

    # Build per-filter tab panes
    filter_panes = []
    for fid, (results, session_stats) in filter_data.items():
        fid_safe   = _filter_id_safe(fid)
        src_js     = json.dumps(str(Path(source_dirs.get(fid, Path())).resolve()))
        inner_html = _build_panel_html(fid_safe, results, session_stats, src_js, weights=weights)
        filter_panes.append(
            f'<div class="tab-pane" id="tab-{fid_safe}" style="display:none;">\n'
            f'{inner_html}\n</div>'
        )

    # Build per-filter JS initialisation (checkbox listeners + updateCount)
    fid_safes = [_filter_id_safe(fid) for fid in filter_data]
    init_js_parts = []
    for fid_safe in fid_safes:
        init_js_parts.append(f"""
    document.querySelectorAll('#frames-table-{fid_safe} .frame-select').forEach(function(cb) {{
      cb.addEventListener('change', function() {{ updateCount('{fid_safe}'); }});
    }});
    updateCount('{fid_safe}');""")
    init_js = "\n".join(init_js_parts)

    summary_html = _build_summary_html(filter_data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AstroEval Quality Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0; padding: 20px; background: #f5f5f5; color: #333;
    }}
    h1, h2, h3 {{ color: #2c3e50; }}
    h1 {{ border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
    .summary-cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
    .card {{
      background: white; border-radius: 8px; padding: 16px 24px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.1); min-width: 120px; text-align: center;
    }}
    .card .value {{ font-size: 2em; font-weight: bold; }}
    .card.total .value {{ color: #3498db; }}
    .card.accepted .value {{ color: #27ae60; }}
    .card.rejected .value {{ color: #e74c3c; }}
    table {{
      border-collapse: collapse; width: 100%; background: white;
      border-radius: 8px; overflow: hidden;
      box-shadow: 0 2px 6px rgba(0,0,0,0.08); font-size: 0.88em;
    }}
    th {{
      background: #2c3e50; color: white; padding: 10px 8px;
      text-align: left; white-space: nowrap; user-select: none;
    }}
    th.sortable {{ cursor: pointer; }}
    th.sortable:hover {{ background: #3d5570; }}
    th.sort-asc::after  {{ content: " \u25b2"; font-size: 0.75em; opacity: 0.9; }}
    th.sort-desc::after {{ content: " \u25bc"; font-size: 0.75em; opacity: 0.9; }}
    td {{ padding: 7px 8px; border-bottom: 1px solid #eee; white-space: nowrap; }}
    tr:last-child td {{ border-bottom: none; }}
    .plots-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
      gap: 20px; margin: 20px 0;
    }}
    .plot-box {{
      background: white; border-radius: 8px; padding: 12px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }}
    .no-data {{ color: #999; font-style: italic; padding: 20px; text-align: center; }}
    section {{ margin-bottom: 32px; }}
    .session-table {{ max-width: 700px; }}
    .flag-table {{ max-width: 400px; }}
    footer {{ color: #999; font-size: 0.8em; margin-top: 40px; padding-top: 12px;
              border-top: 1px solid #ddd; }}
    .scoring-info {{
      background: white; border-radius: 8px; padding: 14px 18px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.08); margin-top: 16px;
    }}
    .scoring-mode {{ padding: 6px 10px; margin-bottom: 10px;
                     background: #f8f9fa; border-radius: 4px; font-size: 0.92em; }}
    .scoring-eq {{ font-family: "SFMono-Regular", Consolas, monospace;
                   font-size: 0.85em; line-height: 1.9; margin-bottom: 8px;
                   background: #f8f9fa; padding: 8px 12px; border-radius: 4px; }}
    .eq-term {{ display: inline-block; background: #e8f4fd;
                border: 1px solid #bee3f8; border-radius: 3px;
                padding: 1px 6px; margin: 1px; white-space: nowrap; }}
    .eq-plus {{ color: #999; padding: 0 2px; }}
    .scoring-desc {{ color: #555; margin-bottom: 6px; font-size: 0.88em; }}
    .scoring-note {{ color: #999; font-size: 0.82em; font-style: italic; }}
    .frame-controls {{
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 8px; flex-wrap: wrap; gap: 8px;
    }}
    .frame-controls-left {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
    .frame-controls button {{
      padding: 6px 12px; border: 1px solid #ccc; border-radius: 4px;
      background: #f8f9fa; cursor: pointer; font-size: 0.85em;
    }}
    .frame-controls button:hover {{ background: #e9ecef; }}
    .btn-move {{
      background: #e74c3c !important; color: white !important;
      border-color: #c0392b !important; font-weight: bold;
    }}
    .btn-move:hover {{ background: #c0392b !important; }}
    .sel-count {{ font-size: 0.85em; color: #666; }}
    .move-status {{ font-size: 0.85em; color: #27ae60; margin-bottom: 8px; min-height: 1.2em; }}
{_TAB_CSS}
  </style>
</head>
<body>
  <h1>AstroEval Quality Report</h1>

  <div class="tab-nav">
    {chr(10).join(tab_btns)}
  </div>

  <div class="tab-pane active" id="tab-summary">
    {summary_html}
  </div>

  {"".join(filter_panes)}

  <footer>
    Generated by <strong>astro-eval v0.1.0</strong> &mdash;
    Astrophotography Sub-Frame Quality Evaluation Tool
  </footer>

  <script>
    const SOURCE_DIRS = {source_dirs_js};
    const SERVER_MODE = (window.location.protocol === 'http:' || window.location.protocol === 'https:');

    // -----------------------------------------------------------------------
    // Tab switching
    // -----------------------------------------------------------------------
    function switchTab(tabName) {{
      document.querySelectorAll('.tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
      document.querySelectorAll('.tab-pane').forEach(function(p) {{ p.style.display = 'none'; }});
      document.querySelector('.tab-btn[data-tab="' + tabName + '"]').classList.add('active');
      document.getElementById('tab-' + tabName).style.display = 'block';
    }}
    document.querySelectorAll('.tab-btn').forEach(function(btn) {{
      btn.addEventListener('click', function() {{ switchTab(btn.dataset.tab); }});
    }});

    // -----------------------------------------------------------------------
    // Checkbox helpers (scoped by filter ID)
    // -----------------------------------------------------------------------
    function updateCount(fid) {{
      var panel = document.getElementById('tab-' + fid);
      if (!panel) return;
      var all = panel.querySelectorAll('.frame-select');
      var n   = panel.querySelectorAll('.frame-select:checked').length;
      var countEl = document.getElementById('selected-count-' + fid);
      if (countEl) countEl.textContent = n === 0 ? 'none selected' : n + ' selected';
      var tog = document.getElementById('cb-toggle-all-' + fid);
      if (tog) {{
        tog.indeterminate = n > 0 && n < all.length;
        tog.checked = n === all.length && all.length > 0;
      }}
    }}
    function selectRejected(fid) {{
      document.querySelectorAll('#tab-' + fid + ' .frame-select').forEach(function(cb) {{
        cb.checked = cb.dataset.rejected === '1';
      }});
      updateCount(fid);
    }}
    function selectAll(fid) {{
      document.querySelectorAll('#tab-' + fid + ' .frame-select').forEach(function(cb) {{ cb.checked = true; }});
      updateCount(fid);
    }}
    function deselectAll(fid) {{
      document.querySelectorAll('#tab-' + fid + ' .frame-select').forEach(function(cb) {{ cb.checked = false; }});
      updateCount(fid);
    }}

    // Initialise listeners for each filter tab
    {init_js}

    // -----------------------------------------------------------------------
    // Preview links: disable in static mode
    // -----------------------------------------------------------------------
    if (!SERVER_MODE) {{
      document.querySelectorAll('.preview-link').forEach(function(a) {{
        a.removeAttribute('href');
        a.style.cursor = 'default'; a.style.color = 'inherit'; a.style.textDecoration = 'none';
      }});
    }}

    // -----------------------------------------------------------------------
    // Move to _REJECTED
    // -----------------------------------------------------------------------
    function downloadMoveScript(fid) {{
      var panel     = document.getElementById('tab-' + fid);
      var filenames = Array.from(panel.querySelectorAll('.frame-select:checked'))
                          .map(function(cb) {{ return cb.dataset.filename; }});
      if (filenames.length === 0) {{
        alert('No frames selected.');
        return;
      }}
      if (SERVER_MODE) {{
        var btn = document.getElementById('btn-move-' + fid);
        btn.disabled = true;
        btn.textContent = 'Moving\u2026';
        fetch('/move', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{filter: fid, filenames: filenames}})
        }})
        .then(function(r) {{ return r.json(); }})
        .then(function(data) {{
          var st = document.getElementById('move-status-' + fid);
          if (data.errors.length === 0) {{
            st.style.color = '#27ae60';
            st.textContent = '\u2713 Moved ' + data.moved.length + ' file(s) to _REJECTED';
          }} else {{
            st.style.color = '#e74c3c';
            st.textContent = 'Moved ' + data.moved.length + ', errors: ' + data.errors.join(' | ');
          }}
          data.moved.forEach(function(fn) {{
            var cb = panel.querySelector('.frame-select[data-filename="' + fn + '"]');
            if (cb) {{ cb.checked = false; cb.disabled = true; cb.closest('tr').style.opacity = '0.35'; }}
          }});
          updateCount(fid);
          btn.disabled = false;
          btn.textContent = '\U0001F4C1 Move to _REJECTED\u2026';
        }})
        .catch(function(err) {{
          document.getElementById('move-status-' + fid).textContent = 'Error: ' + err;
          document.getElementById('btn-move-' + fid).disabled = false;
          document.getElementById('btn-move-' + fid).textContent = '\U0001F4C1 Move to _REJECTED\u2026';
        }});
      }} else {{
        var src  = SOURCE_DIRS[fid] || '';
        var dest = src + '\\\\_REJECTED';
        var lines = ['@echo off', 'setlocal',
          'set "SRC=' + src + '"', 'set "DEST=' + dest + '"',
          'mkdir "%DEST%" 2>nul', ''];
        filenames.forEach(function(fn) {{ lines.push('move "%SRC%\\\\' + fn + '" "%DEST%"'); }});
        lines.push('', 'echo.', 'echo Moved ' + filenames.length + ' file(s).', 'pause');
        var blob = new Blob([lines.join('\\r\\n')], {{type: 'text/plain'}});
        var url  = URL.createObjectURL(blob);
        var a    = document.createElement('a');
        a.href = url; a.download = 'move_rejected_' + fid + '.bat';
        document.body.appendChild(a); a.click();
        document.body.removeChild(a); URL.revokeObjectURL(url);
        document.getElementById('move-status-' + fid).textContent =
          '\u2713 move_rejected_' + fid + '.bat downloaded';
      }}
    }}

    // -----------------------------------------------------------------------
    // Column sort (per-table state)
    // -----------------------------------------------------------------------
    (function () {{
      function cellValue(row, col) {{
        var td = row.querySelectorAll('td')[col];
        return td ? (td.dataset.value !== undefined ? td.dataset.value : td.textContent.trim()) : '';
      }}
      function compare(a, b, col) {{
        var av = cellValue(a, col), bv = cellValue(b, col);
        if (av === '' && bv === '') return 0;
        if (av === '') return 1; if (bv === '') return -1;
        var an = parseFloat(av), bn = parseFloat(bv);
        if (!isNaN(an) && !isNaN(bn)) return an - bn;
        return av.localeCompare(bv);
      }}
      document.querySelectorAll('.sortable-table thead th.sortable').forEach(function(th) {{
        th.addEventListener('click', function() {{
          var table  = th.closest('table');
          var tbody  = table.querySelector('tbody');
          var rows   = Array.from(tbody.querySelectorAll('tr'));
          var allThs = Array.from(table.querySelectorAll('thead th'));
          var col    = allThs.indexOf(th);
          var sc  = parseInt(table.dataset.sortCol !== undefined ? table.dataset.sortCol : '-1');
          var asc = table.dataset.sortAsc !== 'false';
          if (sc === col) {{ asc = !asc; }} else {{ sc = col; asc = true; }}
          table.dataset.sortCol = sc; table.dataset.sortAsc = asc;
          rows.sort(function(a, b) {{ return asc ? compare(a, b, col) : -compare(a, b, col); }});
          rows.forEach(function(r) {{ tbody.appendChild(r); }});
          allThs.forEach(function(h, i) {{
            h.classList.remove('sort-asc', 'sort-desc');
            if (i === col) h.classList.add(asc ? 'sort-asc' : 'sort-desc');
          }});
        }});
      }});
    }})();

    // -----------------------------------------------------------------------
    // Watch mode: auto-refresh via Server-Sent Events
    // -----------------------------------------------------------------------
    if (SERVER_MODE) {{
      var _sse = new EventSource('/events');
      _sse.onmessage = function(e) {{
        if (e.data === 'reload') {{ window.location.reload(); }}
      }};
    }}
  </script>
</body>
</html>
"""
    html_safe = html.encode("utf-8", errors="replace").decode("utf-8")
    output_path.write_text(html_safe, encoding="utf-8")
    logger.info("Multi-filter HTML report written to %s", output_path)
