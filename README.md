# astro-eval

**Astrophotography sub-frame quality evaluation tool.**

Evaluates directories of FITS files for quality metrics including PSF FWHM, star eccentricity, background noise, and signal-to-noise ratio. Produces composite quality scores and per-frame rejection decisions, with CSV and optional HTML reports.

## Equipment Support

Tuned for:
- **Telescope:** William Optics Redcat 51 (250 mm focal length, f/4.9) — use `--focal-length` for other scopes
- **Camera:** QHY MiniCam 8M (pixel size read from FITS headers: `XPIXSZ` or `PIXSIZE1`)
- **Pixel scale:** `206.265 × pixel_size_µm / focal_length_mm` arcsec/pixel

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `astropy` | FITS I/O, sigma-clipped statistics |
| `numpy` | Array operations |
| `scipy` | PSF curve fitting (Moffat/Gaussian) |
| `sep` | Source Extractor Python — star detection & background |
| `matplotlib` | Distribution plots in HTML report |

## Quick Start

```bash
# Evaluate a session directory (auto-detects filter mode)
astro-eval /path/to/session --html

# Narrowband Ha session with explicit mode
astro-eval /path/to/ha_session --mode gas --html

# Broadband with custom thresholds
astro-eval /path/to/lum --fwhm-threshold 4.0 --ecc-threshold 0.4

# Custom telescope
astro-eval /path/to/session --focal-length 600
```

## Processing Modes

| Mode | Target | Key Metrics |
|------|--------|-------------|
| `star` | Broadband (L, R, G, B, RGB, Clear) | FWHM, Eccentricity, Star count, SNR weight |
| `gas` | Narrowband (Ha, OIII, SII) | Background noise, SNR estimate, Star count (transparency) |
| `auto` | Auto-detect from `FILTER` header | Dispatches to star or gas |

### Auto-detection filter mapping

| Filter keyword | Canonical | Mode |
|----------------|-----------|------|
| `Ha`, `H-Alpha`, `H_Alpha`, `HAlpha` | `Ha` | gas |
| `OIII`, `O3`, `O-III` | `OIII` | gas |
| `SII`, `S2`, `S-II` | `SII` | gas |
| `R`, `Red`, `G`, `Green`, `B`, `Blue` | R/G/B | star |
| `L`, `Lum`, `Luminance` | L | star |
| Unknown / missing | — | star (with warning) |

## CLI Reference

```
astro-eval INPUT_DIR [OPTIONS]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `INPUT_DIR` | *(required)* | Directory containing `.fits`, `.fit`, or `.fts` files |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode {star,gas,auto}` | `auto` | Evaluation mode |
| `--output DIR` | `INPUT_DIR` | Output directory for reports |
| `--focal-length MM` | `250.0` | Telescope focal length (mm) |
| `--fwhm-threshold ARCSEC` | `5.0` | Absolute FWHM rejection limit (arcsec) |
| `--ecc-threshold VALUE` | `0.5` | Eccentricity rejection threshold [0–1] |
| `--star-fraction FRAC` | `0.7` | Min star count as fraction of session median |
| `--snr-fraction FRAC` | `0.5` | Min SNR weight as fraction of session median |
| `--sigma-fwhm SIGMA` | `2.0` | Sigma multiplier for FWHM statistical rejection |
| `--sigma-noise SIGMA` | `2.5` | Sigma multiplier for noise statistical rejection |
| `--sigma-bg SIGMA` | `3.0` | Sigma multiplier for background level rejection |
| `--detection-threshold SIGMA` | `5.0` | Star detection sigma threshold |
| `--html` | off | Generate HTML report with plots |
| `--verbose` | off | Verbose progress output |
| `--version` | — | Show version and exit |

## Output

### CSV report (`astro_eval_report.csv`)

One row per frame. Columns include:

| Column | Description |
|--------|-------------|
| `filename` | FITS filename |
| `mode` | `star` or `gas` |
| `filter` | Filter name from header |
| `exptime_s` | Exposure time (seconds) |
| `n_stars` | Detected star count |
| `fwhm_median_arcsec` | Median FWHM in arcseconds |
| `eccentricity_median` | Median stellar eccentricity [0–1] |
| `snr_weight` | SNR weight proxy (broadband) |
| `snr_estimate` | Nebula SNR estimate (narrowband) |
| `background_rms` | Background noise RMS (ADU) |
| `score` | Composite quality score [0–1] |
| `rejected` | `1` if frame rejected, `0` if accepted |
| `rejection_reasons` | Pipe-separated rejection criterion names |
| `flag_*` | Per-criterion binary rejection flags |

### HTML report (`astro_eval_report.html`)

Self-contained HTML file (no external dependencies) including:
- Summary cards (total/accepted/rejected/pass rate)
- Per-criterion rejection breakdown table
- Distribution plots: FWHM, star count, quality score, background noise
- Color-coded per-frame results table (green/yellow/red)
- Session statistics table

## Metrics Explained

### Star Mode (Broadband)

**FWHM** — Full Width at Half Maximum of the stellar PSF, fitted using a Moffat profile (Gaussian fallback). Measured in pixels, reported in arcseconds. Lower is better (sharper stars).

**Eccentricity** — Departure from circular PSF: `sqrt(1 - (b/a)²)`. 0 = perfect circle, approaching 1 = elongated. Caused by tracking errors, wind, or collimation issues.

**SNR Weight** — `Σ(flux²) / (noise² × N_stars)`. Higher means brighter stars relative to noise floor.

**Star Count** — Number of detected sources. Drops significantly with clouds or focus shift.

### Gas Mode (Narrowband)

**SNR Estimate** — `(signal_region_median - background_median) / background_rms`, where signal pixels are identified by sigma-clipping above background. Higher is better.

**Background RMS** — Noise level of the sky background. Higher values indicate light pollution, moon contamination, or sky glow.

**Star Count** — Used as a transparency proxy.

## Scoring

### Star Score
```
Score = 0.30×(1 - norm_FWHM) + 0.25×(1 - norm_Ecc) + 0.20×norm_Stars + 0.25×norm_SNR
```

### Gas Score
```
Score = 0.40×norm_SNR + 0.25×(1 - norm_Noise) + 0.15×(1 - norm_BG) + 0.20×norm_Stars
```

All metrics are normalized to [0, 1] across the session range. Score of 1.0 is the best frame in the session.

## Rejection Criteria

### Star Mode

| Criterion | Condition |
|-----------|-----------|
| `high_fwhm` | FWHM > session_median + 2σ **OR** FWHM > `--fwhm-threshold` |
| `high_eccentricity` | Eccentricity > `--ecc-threshold` |
| `low_stars` | Stars < session_median × `--star-fraction` |
| `low_snr_weight` | SNR weight < session_median × `--snr-fraction` |

### Gas Mode

| Criterion | Condition |
|-----------|-----------|
| `high_noise` | Background RMS > session_median + 2.5σ |
| `high_background` | Background median > session_median + 3σ |
| `low_snr` | SNR estimate < session_median × 0.5 |
| `low_stars` | Stars < session_median × 0.7 |

> **Note:** Gradients and vignetting are intentionally **not** rejection criteria — these are correctable in post-processing with calibration frames.

## Technical Notes

- **Minimum frames:** Session statistics require at least 3 frames for reliable thresholds. A warning is issued with fewer.
- **Multi-extension FITS:** Automatically searches image extensions if primary HDU has no data.
- **3D FITS arrays:** RGB (3×H×W) converted to luminance; multi-plane uses first plane.
- **SEP byte order:** `byteswap().newbyteorder()` applied before all SEP calls as required.
- **Saturated stars:** Skipped during PSF fitting (SEP flag bit 4).
- **Edge sources:** Excluded within 50 px of image border by default.

## Architecture

```
astro_eval/
├── __init__.py        Package exports
├── fits_loader.py     FITS I/O + header parsing
├── background.py      Background estimation (SEP + sigma-clip)
├── star_detection.py  SEP star detection + quality filtering
├── psf_fitting.py     Moffat/Gaussian PSF fitting
├── metrics.py         Star + gas metric computation
├── scoring.py         Session stats, rejection flags, composite scores
├── report.py          CSV + HTML report generation
└── cli.py             argparse CLI entry point
```
