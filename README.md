# astro-eval

**Astrophotography sub-frame quality evaluation tool.**

Evaluates directories of FITS/XISF files for quality metrics including PSF FWHM, star eccentricity, background noise, and signal-to-noise ratio. Produces composite quality scores and per-frame rejection decisions, with CSV and interactive HTML reports.

Supports multi-filter sessions (Ha/OIII/SII) with per-filter tabbed reports, a live-updating watch mode, and optional remote file pull from an acquisition PC via SFTP.

## Equipment Support

Tuned for:
- **Telescope:** William Optics Redcat 51 (250 mm focal length, f/4.9) — use `--focal-length` for other scopes
- **Camera:** QHY MiniCam 8M (pixel size read from FITS headers: `XPIXSZ` or `PIXSIZE1`)
- **Pixel scale:** `206.265 × pixel_size_µm / focal_length_mm` arcsec/pixel

## Installation

```bash
pip install -e .
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `astropy` | FITS I/O, sigma-clipped statistics |
| `numpy` | Array operations |
| `scipy` | PSF curve fitting (Moffat/Gaussian), trail detection |
| `sep` | Source Extractor Python — star detection & background |
| `matplotlib` | Distribution plots in HTML report |
| `xisf` | XISF file format support |
| `paramiko` | SFTP remote file pull (`--remote`) |

## Quick Start

```bash
# Single filter session — generates CSV report
astro-eval /path/to/ha_session

# With interactive HTML report
astro-eval /path/to/ha_session --html

# Multi-filter session (subdirs named Ha/, OIII/, SII/ etc.)
astro-eval /path/to/session_root --html

# Live watch mode — updates report every 30s as new frames arrive
astro-eval /path/to/session --watch --html

# Watch + pull new frames from a remote acquisition PC via SFTP
astro-eval ./staging --watch --html \
  --remote astromini \
  --remote-dir "C:\Users\AstroMini\Documents\N.I.N.A\2026-03-09\Soul Nebula\LIGHT" \
  --remote-user AstroMini
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
| `Ha`, `H-Alpha`, `H`, `HAlpha` | `Ha` | gas |
| `OIII`, `O3`, `O-III`, `O` | `OIII` | gas |
| `SII`, `S2`, `S-II`, `S` | `SII` | gas |
| `R`, `Red`, `G`, `Green`, `B`, `Blue` | R/G/B | star |
| `L`, `Lum`, `Luminance` | L | star |
| Unknown / missing | — | star (with warning) |

## Multi-Filter Mode

If `INPUT_DIR` contains subdirectories with filter names (e.g. `Ha/`, `OIII/`, `SII/`), the tool automatically processes each filter independently with its own session statistics and rejection thresholds, and generates a single tabbed HTML report.

```
session_root/
├── Ha/        ← processed as gas mode
├── OIII/      ← processed as gas mode
└── SII/       ← processed as gas mode
```

Each filter's CSV is written as `astro_eval_report_Ha.csv`, `astro_eval_report_OIII.csv`, etc.

## Watch Mode

```bash
astro-eval /path/to/session --watch --html
```

- Polls for new FITS/XISF files every 30 seconds
- Re-evaluates session statistics and regenerates the report when new frames arrive
- Serves the report at `http://127.0.0.1:7420/` and auto-refreshes the browser via SSE
- Press `Ctrl+C` to stop

## Remote Pull (SFTP)

Pull frames from a remote Windows acquisition PC during a live session:

```bash
astro-eval ./staging --watch --html \
  --remote HOSTNAME_OR_IP \
  --remote-dir "C:\path\to\LIGHT" \
  --remote-user USERNAME \
  --remote-key ~/.ssh/id_ed25519   # optional, auto-discovers ~/.ssh/ keys
```

- `INPUT_DIR` is the local staging directory where files are downloaded
- Remote filter subdirectories (e.g. `LIGHT\H\`, `LIGHT\S\`, `LIGHT\O\`) are auto-detected via SFTP
- The SSH connection is established once at startup and reused across polls
- Starting with an empty staging directory is fine — the tool waits for the first SFTP pull

### SSH key setup (one-time)

On the evaluation PC:
```bash
ssh-keygen -t ed25519 -C "astro-eval"
```

On the acquisition PC (if the user is an Administrator, the standard `authorized_keys` location is ignored — use the admin file instead):
```powershell
# Copy public key to the admin-specific location
Copy-Item "$env:USERPROFILE\.ssh\authorized_keys" "C:\ProgramData\ssh\administrators_authorized_keys"

# Fix permissions
icacls "C:\ProgramData\ssh\administrators_authorized_keys" /inheritance:r /grant:r "SYSTEM:F" /grant:r "BUILTIN\Administrators:F"
```

## CLI Reference

```
astro-eval INPUT_DIR [OPTIONS]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `INPUT_DIR` | *(required)* | Directory with FITS/XISF files, filter subdirs, or local staging path |

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
| `--workers N` | `1` | Parallel worker processes for frame evaluation |
| `--html` | off | Generate HTML report with plots |
| `--serve` | off | Serve the HTML report at `http://127.0.0.1:7420/` |
| `--port PORT` | `7420` | HTTP server port |
| `--watch` | off | Watch for new frames and update report every 30s (implies `--serve`) |
| `--remote HOST` | — | Hostname/IP of remote acquisition PC for SFTP pull |
| `--remote-dir DIR` | — | Remote directory to pull frames from (Windows paths OK) |
| `--remote-user USER` | current user | SSH username for remote |
| `--remote-key PATH` | auto | SSH private key path (auto-discovers `~/.ssh/` keys if omitted) |
| `--local-staging DIR` | `INPUT_DIR` | Local directory for downloaded remote files |
| `--verbose` | off | Verbose progress output |
| `--version` | — | Show version and exit |

## Output

### CSV report

One file per filter: `astro_eval_report.csv` (single filter) or `astro_eval_report_Ha.csv` etc. (multi-filter).

| Column | Description |
|--------|-------------|
| `filename` | FITS/XISF filename |
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

### HTML report

Self-contained HTML file (no external dependencies) including:
- Summary cards (total/accepted/rejected/pass rate)
- Per-criterion rejection breakdown
- Distribution plots: FWHM, star count, quality score, background noise
- Color-coded per-frame results table with sortable columns
- Clickable filename previews (asinh-stretched image)
- "Move to _REJECTED" button with downloadable `.bat` script
- Multi-filter: tabbed layout with a cross-filter Summary tab
- Live auto-refresh via SSE when running in watch mode

## Metrics Explained

### Star Mode (Broadband)

**FWHM** — Full Width at Half Maximum of the stellar PSF, fitted using a Moffat profile (Gaussian fallback). Measured in pixels, reported in arcseconds. Lower is better.

**Eccentricity** — Departure from circular PSF: `sqrt(1 - (b/a)²)`. 0 = perfect circle, approaching 1 = elongated. Caused by tracking errors, wind, or collimation issues.

**SNR Weight** — `Σ(flux²) / (noise² × N_stars)`. Higher means brighter stars relative to noise floor.

**Star Count** — Number of detected sources. Drops significantly with clouds or focus shift.

### Gas Mode (Narrowband)

**SNR Estimate** — `(signal_region_median - background_median) / background_rms`. Higher is better.

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

All metrics are normalized to [0, 1] across the session. Score of 1.0 is the best frame in the session. Statistics are computed **independently per filter** in multi-filter mode.

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

## Trail Detection

Satellite and airplane trails are detected automatically:

| Trail type | Classification | Rejection |
|------------|---------------|-----------|
| Satellite | Single thin trail, uniform brightness | Borderline (flagged, not rejected) |
| Airplane | Double contrail or strobe pattern | Hard rejected |

Detection uses PCA on connected components in a downsampled, background-suppressed image, followed by perpendicular cross-section analysis to distinguish single vs double trails.

## Technical Notes

- **Minimum frames:** Session statistics require at least 3 frames for reliable thresholds.
- **Multi-extension FITS:** Automatically searches image extensions if primary HDU has no data.
- **3D FITS arrays:** RGB (3×H×W) converted to luminance; multi-plane uses first plane.
- **Existing report:** If a report already exists, the tool prompts before reprocessing.
- **SEP byte order:** `byteswap().newbyteorder()` applied before all SEP calls as required.
- **Saturated stars:** Skipped during PSF fitting (SEP flag bit 4).
- **Edge sources:** Excluded within 50 px of image border.

## Architecture

```
astro_eval/
├── __init__.py        Package exports
├── image_loader.py    FITS/XISF I/O + header parsing
├── background.py      Background estimation (SEP + sigma-clip)
├── star_detection.py  SEP star detection + quality filtering
├── psf_fitting.py     Moffat/Gaussian PSF fitting
├── metrics.py         Star + gas metric computation
├── trail_detection.py Satellite/airplane trail detection
├── scoring.py         Session stats, rejection flags, composite scores
├── report.py          CSV + HTML report generation
└── cli.py             CLI, HTTP server, watch loop, SFTP sync
```
