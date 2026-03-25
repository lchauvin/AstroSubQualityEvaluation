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

from .metrics import EvalConfig, FrameMetrics, ScoringWeights
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
    "background_gradient",
    "altitude_deg",
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
    "flag_high_gradient",
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
        "background_gradient": _fmt(m.background_gradient, precision=3),
        "altitude_deg": _fmt(m.altitude_deg, precision=1) if m.altitude_deg is not None else "",
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
        "flag_high_gradient": "1" if r.flags.get("high_gradient") else "0",
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


# PixInsight SubFrameSelector columns (compatible with SFS 1.48+)
_SFS_COLUMNS = [
    "Index", "File", "Enabled", "Weight",
    "FWHM", "Eccentricity", "SNRWeight", "Median", "MeanDeviation", "Noise",
    "StarSupport", "StarResidual", "NoiseSupport",
    "FWHMHigh", "EccentricityHigh", "SNRWeightLow",
    "MedianHigh", "MeanDeviationHigh", "NoiseHigh",
]


def generate_subframeselector_csv(
    results: List[FrameResult],
    output_path: str | Path,
) -> None:
    """
    Export evaluation results in PixInsight SubFrameSelector CSV format.

    The Weight column uses the astro-eval composite quality score (0–1).
    Enabled is set to 0 for rejected frames, 1 for accepted.
    All numeric columns use the values computed by astro-eval where available;
    fields not directly computed (MeanDeviation, NoiseSupport) are set to 0.

    Parameters
    ----------
    results:
        List of FrameResult objects from evaluate_session().
    output_path:
        Path to the output CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_results = sorted(results, key=lambda r: r.metrics.filename)

    with open(output_path, "w", newline="", encoding="utf-8", errors="replace") as f:
        writer = csv.DictWriter(f, fieldnames=_SFS_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for i, result in enumerate(sorted_results, 1):
            m = result.metrics
            r = result.rejection
            enabled = 0 if r.rejected else 1
            weight = result.score if math.isfinite(result.score) else 0.0
            fwhm = m.fwhm_median if math.isfinite(m.fwhm_median) else 0.0
            ecc = m.eccentricity_median if math.isfinite(m.eccentricity_median) else 0.0
            snr_weight = m.snr_weight if math.isfinite(m.snr_weight) else 0.0
            bg_median = m.background_median if math.isfinite(m.background_median) else 0.0
            noise = m.background_rms if math.isfinite(m.background_rms) else 0.0
            star_support = m.n_stars
            star_residual = m.psf_residual_median if math.isfinite(m.psf_residual_median) else 0.0
            writer.writerow({
                "Index": i,
                "File": m.filepath,
                "Enabled": enabled,
                "Weight": f"{weight:.6f}",
                "FWHM": f"{fwhm:.4f}",
                "Eccentricity": f"{ecc:.4f}",
                "SNRWeight": f"{snr_weight:.4f}",
                "Median": f"{bg_median:.2f}",
                "MeanDeviation": "0",
                "Noise": f"{noise:.4f}",
                "StarSupport": star_support,
                "StarResidual": f"{star_residual:.6f}",
                "NoiseSupport": "0",
                "FWHMHigh": 1 if r.flags.get("high_fwhm") else 0,
                "EccentricityHigh": 1 if r.flags.get("high_eccentricity") else 0,
                "SNRWeightLow": 1 if r.flags.get("low_snr_weight") else 0,
                "MedianHigh": 1 if r.flags.get("high_background") else 0,
                "MeanDeviationHigh": 0,
                "NoiseHigh": 1 if r.flags.get("high_noise") else 0,
            })

    logger.info("SubFrameSelector CSV written to %s", output_path)


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


# ---------------------------------------------------------------------------
# Per-frame FWHM spatial heatmap (SVG)
# ---------------------------------------------------------------------------

def _fwhm_heatmap_svg(
    fwhm_map: Optional[List[List[float]]],
    global_min: float,
    global_max: float,
    cell_px: int = 9,
) -> str:
    """
    Return an inline SVG string for a 5×5 FWHM spatial heatmap cell.

    Color scale: green (low FWHM = good seeing) → red (high FWHM = bad seeing).
    Gray cells have no star data. Normalised against global_min/global_max so
    all frames in the session use a consistent color scale.

    Returns '—' string if fwhm_map is None.
    """
    if fwhm_map is None:
        return "—"

    grid = len(fwhm_map)
    size = grid * cell_px
    rng = global_max - global_min if global_max > global_min else 1.0

    def cell_color(v: float) -> str:
        if not math.isfinite(v):
            return "#cccccc"
        t = max(0.0, min(1.0, (v - global_min) / rng))
        # green → yellow → red
        r = int(min(255, t * 2 * 255))
        g = int(min(255, (1 - t) * 2 * 255))
        return f"#{r:02x}{g:02x}00"

    rects = []
    for ri, row in enumerate(fwhm_map):
        for ci, val in enumerate(row):
            x = ci * cell_px
            y = ri * cell_px
            color = cell_color(val)
            tip = f"{val:.2f}&quot;" if math.isfinite(val) else "no data"
            rects.append(
                f'<rect x="{x}" y="{y}" width="{cell_px}" height="{cell_px}" '
                f'fill="{color}"><title>{tip}</title></rect>'
            )

    return (
        f'<svg width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" style="display:block;">'
        + "".join(rects)
        + "</svg>"
    )


def _compute_fwhm_map_global_range(results: List[FrameResult]) -> tuple:
    """Return (global_min, global_max) FWHM map values across all frames for consistent coloring."""
    all_vals: List[float] = []
    for r in results:
        if r.metrics.fwhm_map is not None:
            for row in r.metrics.fwhm_map:
                all_vals.extend(v for v in row if math.isfinite(v))
    if not all_vals:
        return 0.0, 5.0
    return float(np.min(all_vals)), float(np.max(all_vals))


# ---------------------------------------------------------------------------
# Interactive chart data + JS
# ---------------------------------------------------------------------------

def _build_chart_data_json(results: List[FrameResult], prefix: str, table_id: str = "") -> str:
    """
    Serialize per-frame data to a JSON object consumed by the inline chart renderer.
    """
    ordered = sorted(results, key=lambda r: r.metrics.filename)
    frames = []
    for r in ordered:
        m = r.metrics
        frames.append({
            "fn": m.filename,
            "fwhm": m.fwhm_median if math.isfinite(m.fwhm_median) else None,
            "stars": m.n_stars,
            "score": r.score if math.isfinite(r.score) else None,
            "bg_rms": m.background_rms if math.isfinite(m.background_rms) else None,
            "alt": m.altitude_deg,
            "mode": m.mode,
            "rejected": r.rejection.rejected,
            "flags": {k: bool(v) for k, v in r.rejection.flags.items()},
        })
    return json.dumps({"frames": frames, "prefix": prefix, "tableId": table_id})


# _CHART_JS is a plain Python string (NOT an f-string) so JavaScript {braces} are preserved verbatim.
# It is embedded into the HTML via an f-string substitution: {_CHART_JS}
_CHART_JS = """
(function() {
  function px(n) { return Math.round(n * 10) / 10; }

  // Adaptive x-axis frame-number labels (bar chart variant: bar centres)
  function xAxisBars(n, ml, iW, mt, iH) {
    var step = n <= 20 ? 1 : n <= 50 ? 5 : n <= 100 ? 10 : n <= 200 ? 20 : 50;
    var out = '', y = mt + iH + 13;
    for (var i = 0; i < n; i++) {
      if (i % step === 0 || i === n - 1) {
        var x = ml + (i + 0.5) * (iW / n);
        out += '<text x="' + px(x) + '" y="' + y + '" fill="#888" font-size="8" text-anchor="middle">' + (i + 1) + '</text>';
      }
    }
    return out;
  }

  // Adaptive x-axis frame-number labels (line chart variant: point positions)
  function xAxisLine(n, ml, iW, mt, iH) {
    var step = n <= 20 ? 1 : n <= 50 ? 5 : n <= 100 ? 10 : n <= 200 ? 20 : 50;
    var out = '', y = mt + iH + 13;
    for (var i = 0; i < n; i++) {
      if (i % step === 0 || i === n - 1) {
        var x = ml + i / Math.max(n - 1, 1) * iW;
        out += '<text x="' + px(x) + '" y="' + y + '" fill="#888" font-size="8" text-anchor="middle">' + (i + 1) + '</text>';
      }
    }
    return out;
  }

  // Wire up click-to-row for [data-fn] elements inside a container
  function addClickListeners(containerId, tableId) {
    var container = document.getElementById(containerId);
    var table = document.getElementById(tableId);
    if (!container || !table) return;
    // Build filename→row lookup to avoid CSS selector escaping issues
    var rowMap = {};
    table.querySelectorAll('tbody tr[data-filename]').forEach(function(r) {
      rowMap[r.getAttribute('data-filename')] = r;
    });
    container.querySelectorAll('[data-fn]').forEach(function(el) {
      el.addEventListener('click', function() {
        var fn = el.getAttribute('data-fn');
        if (!fn) return;
        var row = rowMap[fn];
        if (!row) return;
        row.scrollIntoView({behavior: 'smooth', block: 'center'});
        var prev = row.style.outline;
        row.style.outline = '2px solid #3498db';
        setTimeout(function() { row.style.outline = prev; }, 1500);
      });
    });
  }

  // fns: array of filenames aligned with values (for data-fn on bars)
  function makeBarChart(values, fns, colors, median, title, ylabel, W, H) {
    W = W || 500; H = H || 260;
    var ml = 52, mr = 16, mt = 28, mb = 36;
    var iW = W - ml - mr, iH = H - mt - mb;
    var n = values.length;
    if (!n) return '<p class="no-data">No data</p>';
    var finite = values.filter(function(v) { return v !== null && isFinite(v); });
    if (!finite.length) return '<p class="no-data">No data</p>';
    var maxV = Math.max.apply(null, finite) * 1.08;
    var minV = Math.min(0, Math.min.apply(null, finite));
    var rng = maxV - minV || 1;
    var bW = iW / n * 0.82, gap = iW / n * 0.18;
    var bars = '';
    for (var i = 0; i < n; i++) {
      var v = values[i];
      if (v === null || !isFinite(v)) continue;
      var x = ml + i * (iW / n) + gap / 2;
      var bH = (v - minV) / rng * iH;
      var y = mt + iH - bH;
      var fn = (fns && fns[i]) ? fns[i].replace(/&/g,'&amp;').replace(/"/g,'&quot;') : '';
      bars += '<rect x="' + px(x) + '" y="' + px(y) + '" width="' + px(bW) + '" height="' + px(bH) + '" fill="' + colors[i] + '" opacity="0.88" data-fn="' + fn + '" style="cursor:pointer;"><title>' + (typeof v === 'number' ? v.toFixed(3) : v) + '</title></rect>';
    }
    var med = '';
    if (median !== null && isFinite(median)) {
      var my = mt + iH - (median - minV) / rng * iH;
      med = '<line x1="' + ml + '" y1="' + px(my) + '" x2="' + (ml+iW) + '" y2="' + px(my) + '" stroke="#e67e22" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.85"/><text x="' + (ml+iW-2) + '" y="' + (px(my)-3) + '" fill="#e67e22" font-size="9" text-anchor="end">med:' + median.toFixed(2) + '</text>';
    }
    var yticks = '';
    for (var t = 0; t <= 4; t++) {
      var tv = minV + (maxV - minV) * t / 4;
      var ty = mt + iH - (tv - minV) / rng * iH;
      yticks += '<line x1="' + (ml-3) + '" y1="' + px(ty) + '" x2="' + ml + '" y2="' + px(ty) + '" stroke="#ccc"/><text x="' + (ml-5) + '" y="' + (px(ty)+3.5) + '" fill="#888" font-size="8.5" text-anchor="end">' + tv.toFixed(2) + '</text>';
    }
    var xlabels = xAxisBars(n, ml, iW, mt, iH);
    return '<svg width="100%" viewBox="0 0 ' + W + ' ' + H + '" style="font-family:sans-serif;">' +
      '<text x="' + (W/2) + '" y="16" text-anchor="middle" font-size="11" font-weight="bold" fill="#2c3e50">' + title + '</text>' +
      '<text x="' + px(ml-44) + '" y="' + (H/2) + '" font-size="9.5" fill="#666" text-anchor="middle" transform="rotate(-90 ' + px(ml-44) + ' ' + (H/2) + ')">' + ylabel + '</text>' +
      '<line x1="' + ml + '" y1="' + mt + '" x2="' + ml + '" y2="' + (mt+iH) + '" stroke="#ddd"/>' +
      '<line x1="' + ml + '" y1="' + (mt+iH) + '" x2="' + (ml+iW) + '" y2="' + (mt+iH) + '" stroke="#ddd"/>' +
      yticks + bars + med + xlabels + '</svg>';
  }

  // fns: array of filenames aligned with primVals (for data-fn on rejected dots)
  function makeDualLine(primVals, fns, scoreVals, rejected, title, primLabel, primColor, showMedian, W, H) {
    W = W || 680; H = H || 260;
    var ml = 52, mr = 52, mt = 28, mb = 36;
    var iW = W - ml - mr, iH = H - mt - mb;
    var n = primVals.length;
    if (!n) return '<p class="no-data">No data</p>';
    primColor = primColor || '#5b9bd5';
    var sc = '#f0ad4e';
    var pfin = primVals.filter(function(v) { return v !== null && isFinite(v); });
    var pmax = pfin.length ? Math.max.apply(null, pfin) * 1.08 : 1;
    var pmin = Math.min(0, pfin.length ? Math.min.apply(null, pfin) * 0.9 : 0);
    var prng = pmax - pmin || 1;
    function xp(i) { return ml + i / Math.max(n-1,1) * iW; }
    function yp(v) { return mt + iH - (v - pmin) / prng * iH; }
    function ys(v) { return mt + iH - v * iH; }
    var pp = [], sp = [];
    for (var i = 0; i < n; i++) {
      if (primVals[i] !== null && isFinite(primVals[i])) pp.push(px(xp(i)) + ',' + px(yp(primVals[i])));
      if (scoreVals[i] !== null && isFinite(scoreVals[i])) sp.push(px(xp(i)) + ',' + px(ys(scoreVals[i])));
    }
    var primLine = pp.length > 1 ? '<polyline points="' + pp.join(' ') + '" fill="none" stroke="' + primColor + '" stroke-width="1.5"/>' : '';
    var scoreLine = sp.length > 1 ? '<polyline points="' + sp.join(' ') + '" fill="none" stroke="' + sc + '" stroke-width="1.2" stroke-dasharray="3,2"/>' : '';
    var rejDots = '';
    for (var i = 0; i < n; i++) {
      if (rejected[i] && scoreVals[i] !== null && isFinite(scoreVals[i])) {
        var fn = (fns && fns[i]) ? fns[i].replace(/&/g,'&amp;').replace(/"/g,'&quot;') : '';
        rejDots += '<circle cx="' + px(xp(i)) + '" cy="' + px(ys(scoreVals[i])) + '" r="4" fill="#d9534f" opacity="0.85" data-fn="' + fn + '" style="cursor:pointer;"><title>Rejected: ' + (fns && fns[i] ? fns[i] : '') + '</title></circle>';
      }
    }
    var medLine = '';
    if (showMedian && pfin.length) {
      var med = pfin.slice().sort(function(a,b){return a-b;}); var mi = Math.floor(med.length/2);
      var mv = med.length%2 ? med[mi] : (med[mi-1]+med[mi])/2;
      var my = yp(mv);
      medLine = '<line x1="' + ml + '" y1="' + px(my) + '" x2="' + (ml+iW) + '" y2="' + px(my) + '" stroke="' + primColor + '" stroke-width="1" stroke-dasharray="4,3" opacity="0.55"/>';
    }
    var ytL = '', ytR = '';
    for (var t = 0; t <= 4; t++) {
      var tv = pmin + (pmax-pmin)*t/4, ty = yp(tv);
      ytL += '<line x1="' + (ml-3) + '" y1="' + px(ty) + '" x2="' + ml + '" y2="' + px(ty) + '" stroke="#ccc"/><text x="' + (ml-5) + '" y="' + (px(ty)+3.5) + '" fill="' + primColor + '" font-size="8.5" text-anchor="end">' + tv.toFixed(2) + '</text>';
      var sv = t * 0.25, sy = ys(sv);
      ytR += '<text x="' + (ml+iW+5) + '" y="' + (px(sy)+3.5) + '" fill="' + sc + '" font-size="8.5">' + sv.toFixed(2) + '</text>';
    }
    var xlabels = xAxisLine(n, ml, iW, mt, iH);
    return '<svg width="100%" viewBox="0 0 ' + W + ' ' + H + '" style="font-family:sans-serif;">' +
      '<text x="' + (W/2) + '" y="16" text-anchor="middle" font-size="11" font-weight="bold" fill="#2c3e50">' + title + '</text>' +
      '<text x="' + px(ml-44) + '" y="' + (H/2) + '" font-size="9.5" fill="' + primColor + '" text-anchor="middle" transform="rotate(-90 ' + px(ml-44) + ' ' + (H/2) + ')">' + primLabel + '</text>' +
      '<text x="' + px(ml+iW+44) + '" y="' + (H/2) + '" font-size="9.5" fill="' + sc + '" text-anchor="middle" transform="rotate(90 ' + px(ml+iW+44) + ' ' + (H/2) + ')">Quality Score</text>' +
      '<line x1="' + ml + '" y1="' + mt + '" x2="' + ml + '" y2="' + (mt+iH) + '" stroke="#ddd"/>' +
      '<line x1="' + ml + '" y1="' + (mt+iH) + '" x2="' + (ml+iW) + '" y2="' + (mt+iH) + '" stroke="#ddd"/>' +
      ytL + ytR + medLine + primLine + scoreLine + rejDots + xlabels + '</svg>';
  }

  function median(arr) {
    var v = arr.filter(function(x) { return x !== null && isFinite(x); }).slice().sort(function(a,b){return a-b;});
    if (!v.length) return null;
    var m = Math.floor(v.length/2);
    return v.length%2 ? v[m] : (v[m-1]+v[m])/2;
  }

  // overrideRejected: optional boolean[] aligned with DATA.frames — used by threshold sliders
  window.renderCharts = function(DATA, prefix, overrideRejected) {
    var frames  = DATA.frames;
    var tableId = DATA.tableId || '';
    var fns    = frames.map(function(f) { return f.fn; });
    var fwhms  = frames.map(function(f) { return f.fwhm; });
    var stars  = frames.map(function(f) { return f.stars; });
    var scores = frames.map(function(f) { return f.score; });
    var bgRms  = frames.map(function(f) { return f.bg_rms; });
    var alts   = frames.map(function(f) { return f.alt !== undefined ? f.alt : null; });
    var rej    = overrideRejected || frames.map(function(f) { return f.rejected; });

    function barsFor(flagKey) {
      return frames.map(function(f, i) {
        return (rej[i] || (f.flags && f.flags[flagKey])) ? '#d9534f' : '#5cb85c';
      });
    }
    var scoreColors = frames.map(function(f, i) {
      return rej[i] ? '#d9534f' : (f.score < 0.5 ? '#f0ad4e' : '#5cb85c');
    });

    var charts = [
      { id: prefix + 'chart-fwhm',  html: makeBarChart(fwhms, fns, barsFor('high_fwhm'), median(fwhms), 'FWHM Distribution', 'FWHM (arcsec)') },
      { id: prefix + 'chart-stars', html: makeBarChart(stars, fns, barsFor('low_stars'),  median(stars), 'Star Count', 'Stars') },
      { id: prefix + 'chart-score', html: makeBarChart(scores, fns, scoreColors,          median(scores), 'Quality Score', 'Score') },
      { id: prefix + 'chart-bg',    html: makeBarChart(bgRms,  fns, barsFor('high_noise'), median(bgRms),  'Background Noise', 'BG RMS (ADU)') },
    ];
    charts.forEach(function(c) {
      var el = document.getElementById(c.id);
      if (el) {
        el.innerHTML = c.html;
        if (tableId) addClickListeners(c.id, tableId);
      }
    });

    // Trend chart
    var trendEl = document.getElementById(prefix + 'chart-trend');
    if (trendEl) {
      var isGas = frames.length && frames[0].mode === 'gas';
      var primVals = isGas ? bgRms : fwhms;
      var primLabel = isGas ? 'BG RMS (ADU)' : 'FWHM (arcsec)';
      trendEl.innerHTML = makeDualLine(primVals, fns, scores, rej, 'Quality Trend Over Time', primLabel, '#5b9bd5', true);
      if (tableId) addClickListeners(prefix + 'chart-trend', tableId);
    }

    // Altitude chart
    var altEl = document.getElementById(prefix + 'chart-alt');
    if (altEl) {
      var hasAlt = alts.some(function(a) { return a !== null && isFinite(a); });
      if (hasAlt) {
        altEl.innerHTML = makeDualLine(alts, fns, scores, rej, 'Score & Altitude', 'Altitude (°)', '#5b9bd5', false);
        if (tableId) addClickListeners(prefix + 'chart-alt', tableId);
      } else {
        altEl.innerHTML = '<p class="no-data">No altitude data available</p>';
      }
    }
  };
})();
"""


# _THRESHOLD_JS: interactive rejection threshold sliders (plain string, not f-string)
_THRESHOLD_JS = """
(function() {
  // Per-table threshold panel: re-color rows AND update charts based on slider values.
  // Data attributes on each <tr>: data-filename, data-fwhm, data-ecc, data-score, data-orig-rejected
  // DATA and prefix are optional — when provided, charts are re-rendered on each slider change.
  window.initThresholdPanel = function(tableId, panelId, DATA, prefix) {
    var table = document.getElementById(tableId);
    var panel = document.getElementById(panelId);
    if (!table || !panel) return;

    var rows = Array.from(table.querySelectorAll('tbody tr'));

    function getVal(id) { var el = document.getElementById(id); return el ? parseFloat(el.value) : NaN; }
    function setLabel(id, v, fmt) { var el = document.getElementById(id + '-val'); if (el) el.textContent = fmt ? fmt(v) : v; }

    function applyThresholds() {
      var sigmaFwhm = getVal(panelId + '-sigma-fwhm');
      var eccT      = getVal(panelId + '-ecc');
      var scoreT    = getVal(panelId + '-score');

      setLabel(panelId + '-sigma-fwhm', sigmaFwhm, function(v) { return v.toFixed(1) + '\u03c3'; });
      setLabel(panelId + '-ecc',        eccT,       function(v) { return v.toFixed(2); });
      setLabel(panelId + '-score',      scoreT,     function(v) { return v.toFixed(2); });

      // Compute session-level FWHM median and std from data attributes
      var fwhms = rows.map(function(r) { return parseFloat(r.dataset.fwhm); }).filter(function(v) { return isFinite(v); });
      var fwhmMed = 0, fwhmStd = 0;
      if (fwhms.length) {
        var s = fwhms.slice().sort(function(a,b){return a-b;});
        fwhmMed = s.length%2 ? s[Math.floor(s.length/2)] : (s[Math.floor(s.length/2)-1]+s[Math.floor(s.length/2)])/2;
        var mean = fwhms.reduce(function(a,b){return a+b;},0)/fwhms.length;
        fwhmStd = Math.sqrt(fwhms.reduce(function(a,v){return a+(v-mean)*(v-mean);},0)/fwhms.length);
      }
      var fwhmThresh = fwhmMed + sigmaFwhm * fwhmStd;

      var nVis = 0, nRej = 0;
      // Build filename→rejected map for chart override
      var rejMap = {};
      rows.forEach(function(row) {
        var fwhm    = parseFloat(row.dataset.fwhm);
        var ecc     = parseFloat(row.dataset.ecc);
        var score   = parseFloat(row.dataset.score);
        var origRej = row.dataset.origRejected === '1';
        var fn      = row.dataset.filename || '';

        var rejected = origRej ||
          (isFinite(fwhm)  && fwhm  > fwhmThresh) ||
          (isFinite(ecc)   && ecc   > eccT) ||
          (isFinite(score) && score < scoreT);

        rejMap[fn] = rejected;
        row.setAttribute('data-vis-rejected', rejected ? '1' : '0');

        if (rejected) {
          row.style.backgroundColor = '#fde8e8';
          nRej++;
        } else {
          row.style.backgroundColor = isFinite(score) && score < 0.5 ? '#fff8e1' : '#e8f5e9';
        }
        nVis++;
      });

      var el = document.getElementById(panelId + '-count');
      if (el) el.textContent = 'Preview: ' + (nVis - nRej) + ' accepted / ' + nRej + ' rejected';

      // Re-render charts with override rejection flags
      if (DATA && prefix && window.renderCharts) {
        var overrideRej = DATA.frames.map(function(f) {
          return rejMap.hasOwnProperty(f.fn) ? rejMap[f.fn] : f.rejected;
        });
        window.renderCharts(DATA, prefix, overrideRej);
      }
    }

    panel.querySelectorAll('input[type=range]').forEach(function(inp) {
      inp.addEventListener('input', applyThresholds);
    });
  };
})();
"""


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

        fig, ax1 = plt.subplots(figsize=(9, 4))
        ax2 = ax1.twinx()

        primary_color = "#5b9bd5"
        score_color   = "#f0ad4e"

        # Primary metric line
        valid_xs = [x for x, v in zip(xs, primary_vals) if math.isfinite(v)]
        valid_ys = [v for v in primary_vals if math.isfinite(v)]
        ax1.plot(valid_xs, valid_ys, color=primary_color, linewidth=1.2, zorder=2)

        # Session median line
        med = float(np.nanmedian(finite_primary))
        ax1.axhline(med, color=primary_color, linestyle="--", linewidth=1,
                    alpha=0.6, label=f"Median: {med:.2f}")

        # Score line (right axis)
        valid_score_xs = [x for x, v in zip(xs, score_vals) if math.isfinite(v)]
        valid_score_ys = [v for v in score_vals if math.isfinite(v)]
        ax2.plot(valid_score_xs, valid_score_ys, color=score_color,
                 linewidth=1.0, linestyle=":", zorder=3, label="Score")

        # Rejected markers on the score curve
        rej_xs = [x for x, v, r in zip(xs, score_vals, rejected) if r and math.isfinite(v)]
        rej_ys = [v for v, r in zip(score_vals, rejected) if r and math.isfinite(v)]
        if rej_xs:
            ax2.scatter(rej_xs, rej_ys, color="#d9534f", s=35, zorder=5, label="Rejected")

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

        # Legend: Rejected, Median, Score — Rejected and Score come from ax2, Median from ax1
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # Reorder: Rejected (last of ax2), Median (ax1), Score (first of ax2)
        combined = list(zip(lines2 + lines1, labels2 + labels1))
        rej_entries   = [(h, l) for h, l in combined if l == "Rejected"]
        other_entries = [(h, l) for h, l in combined if l != "Rejected"]
        ordered_entries = rej_entries + other_entries
        ax1.legend([h for h, _ in ordered_entries], [l for _, l in ordered_entries],
                   fontsize=8, loc="upper right")

        plt.tight_layout()
        img = _make_plot_base64(fig)
        plt.close(fig)
        return img
    except Exception as exc:
        logger.warning("Failed to generate quality trend plot: %s", exc)
        return None


def _plot_score_vs_altitude(
    results: List[FrameResult],
) -> Optional[str]:
    """
    Dual-axis plot over frame index, consistent with _plot_quality_trend.
    Left Y (blue):   altitude above horizon [°].
    Right Y (orange dotted): quality score [0, 1].
    Rejected frames marked with red scatter on the altitude line.
    Legend order: Rejected, Altitude, Score.
    Returns base64 PNG, or None if no altitude data is available.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ordered = sorted(results, key=lambda r: r.metrics.filename)

        has_altitude = any(r.metrics.altitude_deg is not None for r in ordered)
        if not has_altitude:
            return None

        xs       = list(range(len(ordered)))
        scores   = [r.score for r in ordered]
        alts     = [r.metrics.altitude_deg for r in ordered]
        rejected = [r.rejection.rejected for r in ordered]

        alt_color   = "#5b9bd5"
        score_color = "#f0ad4e"

        fig, ax1 = plt.subplots(figsize=(9, 4))
        ax2 = ax1.twinx()

        # Altitude line (left axis)
        valid_alt_xs = [x for x, a in zip(xs, alts) if a is not None]
        valid_alts   = [a for a in alts if a is not None]
        ax1.plot(valid_alt_xs, valid_alts, color=alt_color, linewidth=1.2, zorder=2, label="Altitude")

        # Score line (right axis)
        valid_score_xs = [x for x, s in zip(xs, scores) if math.isfinite(s)]
        valid_scores   = [s for s in scores if math.isfinite(s)]
        ax2.plot(valid_score_xs, valid_scores, color=score_color, linewidth=1.0,
                 linestyle=":", zorder=3, label="Score")

        # Rejected markers on the score curve
        rej_xs   = [x for x, s, rj in zip(xs, scores, rejected) if rj and math.isfinite(s)]
        rej_scores = [s for s, rj in zip(scores, rejected) if rj and math.isfinite(s)]
        if rej_xs:
            ax2.scatter(rej_xs, rej_scores, color="#d9534f", s=35, zorder=5, label="Rejected")

        ax1.set_ylabel("Altitude (°)", color=alt_color, fontsize=9)
        ax1.tick_params(axis="y", labelcolor=alt_color)
        ax1.set_ylim(0, 92)
        ax1.set_xlim(-0.5, len(ordered) - 0.5)
        ax1.set_xlabel("Frame index", fontsize=9)

        ax2.set_ylabel("Quality Score", color=score_color, fontsize=9)
        ax2.tick_params(axis="y", labelcolor=score_color)
        ax2.set_ylim(0, 1.05)

        ax1.set_title("Score & Altitude", fontsize=10)
        ax1.grid(axis="both", alpha=0.2)

        # Legend order: Rejected, Altitude, Score
        lines1, labels1 = ax1.get_legend_handles_labels()   # Altitude
        lines2, labels2 = ax2.get_legend_handles_labels()   # Score, Rejected
        combined = list(zip(lines2 + lines1, labels2 + labels1))
        rej_entries   = [(h, l) for h, l in combined if l == "Rejected"]
        other_entries = [(h, l) for h, l in combined if l != "Rejected"]
        ordered_entries = rej_entries + other_entries
        ax1.legend([h for h, _ in ordered_entries], [l for _, l in ordered_entries],
                   fontsize=8, loc="upper right")

        plt.tight_layout()
        img = _make_plot_base64(fig)
        plt.close(fig)
        return img
    except Exception as exc:
        logger.warning("Failed to generate score vs altitude plot: %s", exc)
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
            "A multiplicative gradient penalty is applied after scoring "
            "(like trail penalties) — see gradient_knee in config."
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
            "noise, and fwhm² in one metric. fwhm_median adds an independent linear seeing penalty. "
            "A multiplicative gradient penalty is applied after scoring "
            "(like trail penalties) — see gradient_knee in config."
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


def _rejection_threshold_str(
    name: str,
    ss: SessionStats,
    config: EvalConfig,
    mode: str,
) -> str:
    """Return a human-readable rejection threshold string for a session-stats metric."""
    if not math.isfinite(ss.median) or not math.isfinite(ss.std):
        return "—"
    m, s = ss.median, ss.std

    if name == "fwhm_median":
        stat_t = m + config.sigma_fwhm * s
        parts = [f"reject &gt; {stat_t:.2f}&quot; (stat)"]
        if config.fwhm_threshold_arcsec > 0:
            parts.append(f"&gt; {config.fwhm_threshold_arcsec:.1f}&quot; (abs)")
        return " or ".join(parts)
    if name == "eccentricity_median":
        return f"reject &gt; {config.ecc_threshold:.2f}"
    if name == "n_stars":
        return f"reject &lt; {m * config.star_count_fraction:.0f} (stat)"
    if name == "snr_weight" and mode != "gas":
        return f"reject &lt; {m * config.snr_fraction:.3g} (stat)"
    if name == "psf_residual_median":
        return f"flag &gt; {m + config.sigma_residual * s:.4g} (stat, soft)"
    if name == "background_rms":
        return f"reject &gt; {m + config.sigma_noise * s:.4g} (stat)"
    if name == "background_median":
        return f"reject &gt; {m + config.sigma_bg * s:.4g} (stat)"
    if name == "snr_estimate" and mode == "gas":
        return f"reject &lt; {m * config.snr_fraction:.3g} (stat)"
    if name == "background_gradient":
        parts = [f"reject &gt; {m + config.sigma_gradient * s:.2f} (stat)"]
        if config.gradient_threshold > 0:
            parts.append(f"&gt; {config.gradient_threshold:.0f} (abs)")
        return " or ".join(parts)
    return "—"


def generate_html_report(
    results: List[FrameResult],
    session_stats: Dict[str, SessionStats],
    output_path: str | Path,
    source_dir: Optional[str | Path] = None,
    weights: Optional[ScoringWeights] = None,
    config: Optional[EvalConfig] = None,
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

    # Chart data JSON (replaces matplotlib PNGs)
    chart_data_json = _build_chart_data_json(results, "single-", "frames-table")

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

    # FWHM heatmap global range for consistent coloring
    fwhm_map_min, fwhm_map_max = _compute_fwhm_map_global_range(results)

    # Default threshold values for the interactive slider panel
    cfg = config or EvalConfig()
    default_sigma_fwhm = cfg.sigma_fwhm
    default_ecc = cfg.ecc_threshold
    default_score = cfg.min_score if cfg.min_score > 0 else 0.5

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

        # FWHM spatial heatmap cell
        heatmap_svg = _fwhm_heatmap_svg(m.fwhm_map, fwhm_map_min, fwhm_map_max)
        heatmap_cell = f'<td style="text-align:center;" title="Spatial FWHM map (5×5 grid)">{heatmap_svg}</td>'

        # Elongation direction/consistency cell
        ecc_cons = m.elongation_consistency
        if math.isfinite(ecc_cons):
            ecc_dir_deg = math.degrees(m.elongation_direction) % 180.0 if math.isfinite(m.elongation_direction) else float("nan")
            dir_str = f"{ecc_dir_deg:.0f}°" if math.isfinite(ecc_dir_deg) else "?"
            ecc_color = "#d9534f" if ecc_cons > 0.5 else ("#f0ad4e" if ecc_cons > 0.3 else "#5cb85c")
            ecc_cell = (
                f'<td data-value="{ecc_cons:.2f}" title="R={ecc_cons:.2f} (0=random, 1=all aligned). '
                f'Direction: {dir_str}" style="color:{ecc_color}">'
                f'R={ecc_cons:.2f}<br><small>{dir_str}</small></td>'
            )
        else:
            ecc_cell = '<td data-value="">—</td>'

        is_rejected_int = 1 if r.rejected else 0
        checked_attr = "checked" if r.rejected else ""
        fwhm_data = f'{m.fwhm_median:.4f}' if math.isfinite(m.fwhm_median) else ""
        ecc_data = f'{m.eccentricity_median:.4f}' if math.isfinite(m.eccentricity_median) else ""
        score_data = f'{result.score:.4f}' if math.isfinite(result.score) else ""
        row = (
            f'<tr style="background-color:{bg};" '
            f'data-filename="{m.filename}" '
            f'data-fwhm="{fwhm_data}" data-ecc="{ecc_data}" '
            f'data-score="{score_data}" data-orig-rejected="{is_rejected_int}">'
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
            f"{heatmap_cell}"
            f"{_format_cell(m.eccentricity_median)}"
            f"{ecc_cell}"
            f"{_format_cell(m.snr_weight, precision=1)}"
            f"{_format_cell(m.psf_signal_weight, precision=1)}"
            f"{_format_cell(m.wfwhm)}"
            f"{_format_cell(m.moffat_beta, precision=2)}"
            f"{_format_cell(m.snr_estimate)}"
            f"{_format_cell(m.background_rms, precision=1)}"
            f"{_format_cell(m.background_gradient, precision=2)}"
            f"{trail_cell}"
            f"{_format_cell(m.altitude_deg, precision=1)}"
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
    mode = results[0].metrics.mode if results else "star"
    stat_rows = []
    for name, ss in session_stats.items():
        if ss.count == 0:
            continue
        thresh = _rejection_threshold_str(name, ss, config, mode) if config else "—"
        stat_rows.append(
            f"<tr>"
            f"<td>{name}</td>"
            f"<td>{ss.count}</td>"
            f"<td>{ss.median:.4g}</td>"
            f"<td>{ss.std:.4g}</td>"
            f"<td>{ss.min_val:.4g}</td>"
            f"<td>{ss.max_val:.4g}</td>"
            f"<td>{thresh}</td>"
            "</tr>\n"
        )
    if config and config.min_score > 0:
        stat_rows.append(
            f"<tr><td>score</td><td>{len(results)}</td>"
            f"<td>—</td><td>—</td><td>—</td><td>—</td>"
            f"<td>reject &lt; {config.min_score:.2f}</td></tr>\n"
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
    .threshold-panel {{
      background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px;
      padding: 8px 14px; margin-bottom: 12px;
    }}
    .threshold-panel summary {{ font-size: 0.9em; padding: 2px 0; }}
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

    <h3>Scoring</h3>
    {scoring_info}
    <div style="display:flex;gap:20px;flex-wrap:wrap;margin-top:16px;align-items:stretch;">
      <div style="flex:1;min-width:240px;background:white;border-radius:8px;padding:16px;box-shadow:0 2px 6px rgba(0,0,0,0.08);">
        <h3 style="margin-top:0;margin-bottom:10px;">Rejection Breakdown</h3>
        <table style="box-shadow:none;border-radius:0;">
          <thead><tr><th>Criterion</th><th>Frames Flagged</th></tr></thead>
          <tbody>{flag_rows}</tbody>
        </table>
      </div>
      <div style="flex:3;min-width:420px;background:white;border-radius:8px;padding:16px;box-shadow:0 2px 6px rgba(0,0,0,0.08);overflow-x:auto;">
        <h3 style="margin-top:0;margin-bottom:10px;">Session Statistics</h3>
        <table style="box-shadow:none;border-radius:0;">
          <thead>
            <tr><th>Metric</th><th>N</th><th>Median</th><th>Std</th><th>Min</th><th>Max</th><th>Rejection threshold</th></tr>
          </thead>
          <tbody>{"".join(stat_rows)}</tbody>
        </table>
      </div>
    </div>
  </section>

  <section>
    <h2>Distribution Charts</h2>
    <div class="plots-grid">
      <div class="plot-box" id="single-chart-fwhm"><p class="no-data">Loading chart…</p></div>
      <div class="plot-box" id="single-chart-stars"><p class="no-data">Loading chart…</p></div>
      <div class="plot-box" id="single-chart-score"><p class="no-data">Loading chart…</p></div>
      <div class="plot-box" id="single-chart-bg"><p class="no-data">Loading chart…</p></div>
    </div>
    <div style="display:flex;gap:20px;margin-top:20px;flex-wrap:wrap;">
      <div class="plot-box" style="flex:1;min-width:0;" id="single-chart-trend"><p class="no-data">Loading chart…</p></div>
      <div class="plot-box" style="flex:1;min-width:0;" id="single-chart-alt"><p class="no-data">Loading chart…</p></div>
    </div>
  </section>

  <section>
    <h2>Per-Frame Results</h2>
    <p>
      <span style="background:#e8f5e9;padding:2px 8px;border-radius:3px;">Green</span> = accepted &nbsp;
      <span style="background:#fff8e1;padding:2px 8px;border-radius:3px;">Yellow</span> = borderline &nbsp;
      <span style="background:#fde8e8;padding:2px 8px;border-radius:3px;">Red</span> = rejected
    </p>

    <details class="threshold-panel" id="single-thresholds">
      <summary style="cursor:pointer;font-weight:bold;margin-bottom:10px;">
        &#9881; Interactive Threshold Preview
        <span id="single-thresholds-count" class="sel-count" style="font-weight:normal;margin-left:10px;"></span>
      </summary>
      <div style="background:white;border-radius:6px;padding:12px 16px;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,0.08);display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;">
        <div>
          <label style="font-size:0.85em;color:#555;">FWHM sigma: <strong id="single-thresholds-sigma-fwhm-val">{default_sigma_fwhm:.1f}σ</strong></label><br>
          <input type="range" id="single-thresholds-sigma-fwhm" min="0.5" max="5.0" step="0.1" value="{default_sigma_fwhm:.1f}" style="width:180px;">
        </div>
        <div>
          <label style="font-size:0.85em;color:#555;">Eccentricity max: <strong id="single-thresholds-ecc-val">{default_ecc:.2f}</strong></label><br>
          <input type="range" id="single-thresholds-ecc" min="0.1" max="0.9" step="0.05" value="{default_ecc:.2f}" style="width:180px;">
        </div>
        <div>
          <label style="font-size:0.85em;color:#555;">Min score: <strong id="single-thresholds-score-val">{default_score:.2f}</strong></label><br>
          <input type="range" id="single-thresholds-score" min="0.0" max="0.9" step="0.05" value="{default_score:.2f}" style="width:180px;">
        </div>
        <div style="font-size:0.8em;color:#888;align-self:center;">
          Changes here are visual only — they do not affect the CSV report.
        </div>
      </div>
    </details>

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
          <th title="5×5 spatial FWHM map: green=sharp, red=blurry, grey=no data">FWHM Map</th>
          <th class="sortable">Ecc</th>
          <th class="sortable" title="Elongation direction consistency R ∈ [0,1]. R≈0: random (good), R≈1: all stars elongated same way (tracking/optical issue). Direction = dominant axis.">Elong.</th>
          <th class="sortable">SNR wt</th>
          <th class="sortable" title="PSFSignalWeight: combines amplitude, FWHM penalty (1/FWHM²), and noise">PSFSW</th>
          <th class="sortable" title="wFWHM = FWHM / sqrt(n_stars): lower is better (Siril metric)">wFWHM</th>
          <th class="sortable" title="Moffat beta: atmospheric seeing index (typical 2.5–5)">β</th>
          <th class="sortable">SNR est</th>
          <th class="sortable">BG RMS</th>
          <th class="sortable" title="Background gradient in noise σ units: (max−min)/noise_rms across 8×8 sky cells. Typical: uniform ~5–30σ, LP ~20–80σ, burned >100σ.">Gradient σ</th>
          <th class="sortable">Trails</th>
          <th class="sortable" title="Telescope altitude above horizon in degrees">Alt (°)</th>
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
    const CHART_DATA = {chart_data_json};
    {_CHART_JS}
    {_THRESHOLD_JS}

    // Script is at the bottom of <body>; DOM elements already exist — call directly.
    renderCharts(CHART_DATA, 'single-');
    initThresholdPanel('frames-table', 'single-thresholds', CHART_DATA, 'single-');

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
        var tr = cb.closest('tr');
        var vis = tr ? tr.getAttribute('data-vis-rejected') : null;
        cb.checked = vis !== null ? vis === '1' : cb.dataset.rejected === '1';
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
    config: Optional[EvalConfig] = None,
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

    chart_data_json = _build_chart_data_json(results, fid + "-", f"frames-table-{fid}")

    fwhm_map_min, fwhm_map_max = _compute_fwhm_map_global_range(results)

    cfg = config or EvalConfig()
    default_sigma_fwhm = cfg.sigma_fwhm
    default_ecc = cfg.ecc_threshold
    default_score = cfg.min_score if cfg.min_score > 0 else 0.5
    threshold_panel_id = f"thresh-{fid}"

    mode = results[0].metrics.mode if results else "star"
    stat_rows = []
    for name, ss in session_stats.items():
        if ss.count == 0:
            continue
        thresh = _rejection_threshold_str(name, ss, config, mode) if config else "—"
        stat_rows.append(
            f"<tr><td>{name}</td><td>{ss.count}</td><td>{ss.median:.4g}</td>"
            f"<td>{ss.std:.4g}</td><td>{ss.min_val:.4g}</td><td>{ss.max_val:.4g}</td>"
            f"<td>{thresh}</td></tr>\n"
        )
    if config and config.min_score > 0:
        stat_rows.append(
            f"<tr><td>score</td><td>{len(results)}</td>"
            f"<td>—</td><td>—</td><td>—</td><td>—</td>"
            f"<td>reject &lt; {config.min_score:.2f}</td></tr>\n"
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

        heatmap_svg = _fwhm_heatmap_svg(m.fwhm_map, fwhm_map_min, fwhm_map_max)
        heatmap_cell = f'<td style="text-align:center;" title="Spatial FWHM map (5×5 grid)">{heatmap_svg}</td>'

        ecc_cons = m.elongation_consistency
        if math.isfinite(ecc_cons):
            ecc_dir_deg = math.degrees(m.elongation_direction) % 180.0 if math.isfinite(m.elongation_direction) else float("nan")
            dir_str = f"{ecc_dir_deg:.0f}°" if math.isfinite(ecc_dir_deg) else "?"
            ecc_color = "#d9534f" if ecc_cons > 0.5 else ("#f0ad4e" if ecc_cons > 0.3 else "#5cb85c")
            ecc_cell = (
                f'<td data-value="{ecc_cons:.2f}" title="R={ecc_cons:.2f}. Direction: {dir_str}" '
                f'style="color:{ecc_color}">R={ecc_cons:.2f}<br><small>{dir_str}</small></td>'
            )
        else:
            ecc_cell = '<td data-value="">—</td>'

        fwhm_data = f'{m.fwhm_median:.4f}' if math.isfinite(m.fwhm_median) else ""
        ecc_data = f'{m.eccentricity_median:.4f}' if math.isfinite(m.eccentricity_median) else ""
        score_data = f'{result.score:.4f}' if math.isfinite(result.score) else ""

        rows_html.append(
            f'<tr style="background-color:{bg};" '
            f'data-filename="{m.filename}" '
            f'data-fwhm="{fwhm_data}" data-ecc="{ecc_data}" '
            f'data-score="{score_data}" data-orig-rejected="{is_rej}">'
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
            f"{heatmap_cell}"
            f"{_format_cell(m.eccentricity_median)}"
            f"{ecc_cell}"
            f"{_format_cell(m.snr_weight, precision=1)}"
            f"{_format_cell(m.psf_signal_weight, precision=1)}"
            f"{_format_cell(m.wfwhm)}"
            f"{_format_cell(m.moffat_beta, precision=2)}"
            f"{_format_cell(m.snr_estimate)}"
            f"{_format_cell(m.background_rms, precision=1)}"
            f"{_format_cell(m.background_gradient, precision=2)}"
            f"{trail_cell}"
            f"{_format_cell(m.altitude_deg, precision=1)}"
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
  <h3>Scoring</h3>
  {scoring_info}
  <div style="display:flex;gap:20px;flex-wrap:wrap;margin-top:16px;align-items:stretch;">
    <div style="flex:1;min-width:240px;background:white;border-radius:8px;padding:16px;box-shadow:0 2px 6px rgba(0,0,0,0.08);">
      <h3 style="margin-top:0;margin-bottom:10px;">Rejection Breakdown</h3>
      <table style="box-shadow:none;border-radius:0;">
        <thead><tr><th>Criterion</th><th>Flagged</th></tr></thead>
        <tbody>{flag_rows}</tbody>
      </table>
    </div>
    <div style="flex:3;min-width:420px;background:white;border-radius:8px;padding:16px;box-shadow:0 2px 6px rgba(0,0,0,0.08);overflow-x:auto;">
      <h3 style="margin-top:0;margin-bottom:10px;">Session Statistics</h3>
      <table style="box-shadow:none;border-radius:0;">
        <thead><tr><th>Metric</th><th>N</th><th>Median</th><th>Std</th><th>Min</th><th>Max</th><th>Rejection threshold</th></tr></thead>
        <tbody>{"".join(stat_rows)}</tbody>
      </table>
    </div>
  </div>
  <h2>Distribution Charts</h2>
  <div class="plots-grid">
    <div class="plot-box" id="{fid}-chart-fwhm"><p class="no-data">Loading chart…</p></div>
    <div class="plot-box" id="{fid}-chart-stars"><p class="no-data">Loading chart…</p></div>
    <div class="plot-box" id="{fid}-chart-score"><p class="no-data">Loading chart…</p></div>
    <div class="plot-box" id="{fid}-chart-bg"><p class="no-data">Loading chart…</p></div>
  </div>
  <div style="display:flex;gap:20px;margin-top:20px;flex-wrap:wrap;">
    <div class="plot-box" style="flex:1;min-width:0;" id="{fid}-chart-trend"><p class="no-data">Loading chart…</p></div>
    <div class="plot-box" style="flex:1;min-width:0;" id="{fid}-chart-alt"><p class="no-data">Loading chart…</p></div>
  </div>
  <script>
    // renderCharts / initThresholdPanel are defined in the main <script> block at the bottom of the page.
    // Use DOMContentLoaded so this inline script (parsed earlier) defers until those functions exist.
    document.addEventListener('DOMContentLoaded', function() {{
      var d = {chart_data_json};
      renderCharts(d, '{fid}-');
      initThresholdPanel('frames-table-{fid}', 'thresh-{fid}', d, '{fid}-');
    }});
  </script>
  <h2>Per-Frame Results</h2>
  <p>
    <span style="background:#e8f5e9;padding:2px 8px;border-radius:3px;">Green</span> = accepted &nbsp;
    <span style="background:#fff8e1;padding:2px 8px;border-radius:3px;">Yellow</span> = borderline &nbsp;
    <span style="background:#fde8e8;padding:2px 8px;border-radius:3px;">Red</span> = rejected
  </p>
  <details class="threshold-panel" id="thresh-{fid}">
    <summary style="cursor:pointer;font-weight:bold;margin-bottom:10px;">
      &#9881; Interactive Threshold Preview
      <span id="thresh-{fid}-count" class="sel-count" style="font-weight:normal;margin-left:10px;"></span>
    </summary>
    <div style="background:white;border-radius:6px;padding:12px 16px;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,0.08);display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;">
      <div>
        <label style="font-size:0.85em;color:#555;">FWHM sigma: <strong id="thresh-{fid}-sigma-fwhm-val">{default_sigma_fwhm:.1f}σ</strong></label><br>
        <input type="range" id="thresh-{fid}-sigma-fwhm" min="0.5" max="5.0" step="0.1" value="{default_sigma_fwhm:.1f}" style="width:180px;">
      </div>
      <div>
        <label style="font-size:0.85em;color:#555;">Eccentricity max: <strong id="thresh-{fid}-ecc-val">{default_ecc:.2f}</strong></label><br>
        <input type="range" id="thresh-{fid}-ecc" min="0.1" max="0.9" step="0.05" value="{default_ecc:.2f}" style="width:180px;">
      </div>
      <div>
        <label style="font-size:0.85em;color:#555;">Min score: <strong id="thresh-{fid}-score-val">{default_score:.2f}</strong></label><br>
        <input type="range" id="thresh-{fid}-score" min="0.0" max="0.9" step="0.05" value="{default_score:.2f}" style="width:180px;">
      </div>
      <div style="font-size:0.8em;color:#888;align-self:center;">
        Changes here are visual only — they do not affect the CSV report.
      </div>
    </div>
  </details>
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
        <th title="5×5 spatial FWHM map: green=sharp, red=blurry">FWHM Map</th>
        <th class="sortable">Ecc</th>
        <th class="sortable" title="Elongation direction consistency R ∈ [0,1]. R≈0: random (good), R≈1: all stars elongated same way.">Elong.</th>
        <th class="sortable">SNR wt</th>
        <th class="sortable" title="PSFSignalWeight: combines amplitude, FWHM penalty (1/FWHM²), and noise">PSFSW</th>
        <th class="sortable" title="wFWHM = FWHM / sqrt(n_stars): lower is better (Siril metric)">wFWHM</th>
        <th class="sortable" title="Moffat beta: atmospheric seeing index (typical 2.5–5)">β</th>
        <th class="sortable">SNR est</th>
        <th class="sortable">BG RMS</th>
        <th class="sortable" title="Background gradient: (max−min)/median of the 2D sky background map.">Gradient</th>
        <th class="sortable">Trails</th>
        <th class="sortable" title="Telescope altitude above horizon in degrees">Alt (°)</th>
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
    config: Optional[EvalConfig] = None,
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
        inner_html = _build_panel_html(fid_safe, results, session_stats, src_js, weights=weights, config=config)
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
    .threshold-panel {{
      background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px;
      padding: 8px 14px; margin-bottom: 12px;
    }}
    .threshold-panel summary {{ font-size: 0.9em; padding: 2px 0; }}
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
    {_CHART_JS}
    {_THRESHOLD_JS}

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
        var tr = cb.closest('tr');
        var vis = tr ? tr.getAttribute('data-vis-rejected') : null;
        cb.checked = vis !== null ? vis === '1' : cb.dataset.rejected === '1';
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
