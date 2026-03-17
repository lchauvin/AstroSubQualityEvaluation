# astro-eval

**Astrophotography sub-frame quality evaluation tool.**

Evaluates directories of FITS/XISF files for quality metrics including PSF FWHM, star eccentricity, background noise, and signal-to-noise ratio. Produces composite quality scores and per-frame rejection decisions, with CSV and interactive HTML reports.

Supports multi-filter sessions (Ha/OIII/SII) with per-filter tabbed reports, a live-updating watch mode, and optional remote file pull from an acquisition PC via SFTP.

## Equipment Support

Tuned for:
- **Telescope:** William Optics Redcat 51 (250 mm focal length, f/4.9) — use `--focal-length` for other scopes
- **Camera:** QHY MiniCam 8M (pixel size read from FITS headers: `XPIXSZ` or `PIXSIZE1`)
- **Pixel scale:** `206.265 × pixel_size_µm / focal_length_mm` arcsec/pixel

If your camera doesn't write pixel size to FITS headers, set `pixel_size_um` in `astro_eval.toml` (see [Config File](#config-file)) or the pixel scale falls back to the built-in default (3.1 arcsec/px for Redcat 51 + 3.76 µm).

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
| `star` | Broadband (L, R, G, B, RGB, Clear) | FWHM, wFWHM, Eccentricity, Star count, SNR weight, PSFSignalWeight, Moffat β |
| `gas` | Narrowband (Ha, OIII, SII) | Background noise, SNR estimate (p95), Star count (transparency), PSFSignalWeight, wFWHM, Moffat β |
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

## Config File

All CLI options can also be set in a TOML config file, making it easy to store per-scope or per-session defaults. CLI arguments always override the config file.

**Search order:**
1. `--config FILE` (explicit path)
2. `INPUT_DIR/astro_eval.toml`
3. `./astro_eval.toml` (current working directory)

Copy and edit the provided example:

```bash
cp astro_eval.toml.example /path/to/session/astro_eval.toml
```

Key sections:

```toml
[telescope]
focal_length_mm = 250.0

[camera]
pixel_size_um = 3.76   # fallback if not in FITS headers

[rejection]
fwhm_threshold_arcsec = 5.0
sigma_fwhm   = 2.0
sigma_noise  = 2.5
sigma_bg     = 3.0
sigma_residual = 3.0   # PSF residual flag threshold (informational)
gradient_threshold = 100.0  # sky gradient in noise σ units; 0 = disabled
gradient_knee      = 1.2    # scoring knee multiplier (× session median)

[scoring.star]
weight_fwhm  = 0.30
weight_ecc   = 0.25
weight_stars = 0.20
weight_psfsw = 0.25   # PSFSignalWeight (supersedes snr_weight)

[scoring.gas]
weight_snr   = 0.30   # reduced; PSFSignalWeight captures overlapping SNR info
weight_noise = 0.20
weight_bg    = 0.15
weight_stars = 0.20
weight_psfsw = 0.15   # PSFSignalWeight — star sharpness bonus, useful even in NB
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
| `--config FILE` | auto | Path to `astro_eval.toml` config file |
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
| `--sigma-residual SIGMA` | `3.0` | Sigma multiplier for PSF residual flag (informational) |
| `--gradient-threshold SIGMA` | `100` | Background gradient rejection threshold in noise σ units. 0 = disabled |
| `--gradient-knee RATIO` | `1.2` | Scoring knee: penalty steepens above `knee × session_median` gradient |
| `--detection-threshold SIGMA` | `5.0` | Star detection sigma threshold |
| `--workers N` | `0` | Parallel worker processes. 0 = all CPU cores |
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
| `fwhm_mean_arcsec` | Mean FWHM in arcseconds |
| `eccentricity_median` | Median stellar eccentricity [0–1] |
| `psf_residual_median` | Median normalized PSF fit residual |
| `snr_weight` | SNR weight proxy: `Σflux² / (noise² × N)` |
| `psf_signal_weight` | PSFSignalWeight: `ΣA² / (2×noise²×N×FWHM²)` — penalizes FWHM super-linearly |
| `wfwhm_arcsec` | wFWHM = `FWHM / √N_stars` — combined seeing+transparency metric |
| `moffat_beta` | Moffat β parameter (atmospheric seeing index; typical 2.5–5) |
| `snr_estimate` | Nebula SNR estimate via 95th-percentile method (narrowband) |
| `background_median` | Median sky background (ADU) |
| `background_rms` | Background noise RMS (ADU) |
| `noise_mad` | Robust noise estimate via MAD (ADU) |
| `background_gradient` | Sky gradient in noise σ units: `(max−min)/noise_rms` across an 8×8 grid of sigma-clipped sky cells. ~5–30 = uniform, ~20–80 = normal LP gradient, >100 = severe (sunrise/cloud edge) |
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

**FWHM** — Full Width at Half Maximum of the stellar PSF, fitted using a 2D elliptical Moffat profile (Gaussian fallback). Measured in pixels, reported in arcseconds. Lower is better. Typical range: 1.5–5 arcsec.

**wFWHM** — Siril-inspired *weighted FWHM*: `FWHM_arcsec / √n_stars`. Combines seeing quality and sky transparency — a frame with poor seeing or few stars both result in a higher (worse) value. Lower is better.

**Eccentricity** — Departure from circular PSF: `√(1 − (b/a)²)`. 0 = perfect circle, approaching 1 = elongated. Caused by tracking errors, wind, or collimation issues. Typical acceptance threshold: < 0.5.

**SNR Weight** — `Σ(flux²) / (noise² × N)`. Higher means brighter stars relative to noise floor. Useful for detecting thin clouds or transparency loss.

**PSFSignalWeight** — PixInsight-inspired metric: `(Σ amplitude_i)² / (2 × noise² × N × FWHM_px²)`. Unlike SNR weight, penalizes FWHM super-linearly (~1/FWHM²), so frames with sharp stars rank significantly higher than blurry frames with equal total flux. Higher is better.

**Moffat β** — Power-law exponent of the fitted Moffat PSF profile. Reflects atmospheric turbulence: β ≈ 2.5 for strong atmospheric seeing, β ≈ 4–5 for better conditions. Informational only — not used in rejection decisions.

**PSF Residual** — Normalized median absolute deviation between the fitted PSF and the actual pixel data: `MAD(fitted − actual) / amplitude`. High values indicate distorted or trailed stars, optical aberrations, or double stars. Informational by default; flagged if `> session_median + sigma_residual × std`.

**Star Count** — Number of detected sources above the SNR threshold. Drops significantly with clouds or focus shift.

### Gas Mode (Narrowband)

**SNR Estimate** — `(p95 − background_median) / background_rms` where p95 is the 95th percentile pixel value. This threshold-independent method gives SNR ≈ 1.6 for pure background (normal distribution p95 = μ + 1.645σ) and higher values for frames with genuine nebula signal. Higher is better.

**Background RMS** — Noise level of the sky background (ADU). Higher values indicate light pollution, moon contamination, or sky glow. The primary rejection criterion for narrowband.

**Star Count** — Used as a transparency proxy even in narrowband — fewer detected stars indicates reduced sky transparency.

**Background Gradient** — Sky spatial non-uniformity expressed in noise σ units: `(max_cell_bg − min_cell_bg) / noise_rms`, where the image is divided into an 8×8 grid of sigma-clipped sky cells. Normalising by the noise floor (rather than sky level) is critical because auto-stretch dramatically amplifies subtle linear gradients — a gradient that looks enormous in a viewer may only be 1–2% of the sky ADU level but still hundreds of σ above noise. Typical values: uniform sky ~5–30 σ, normal LP gradient ~20–80 σ, severe gradient burning part of the frame ~100–1000+ σ. Values above `gradient_threshold` (default 100) are hard-rejected. Applies to both star and gas modes.

**PSFSignalWeight, wFWHM, Moffat β, FWHM, Eccentricity** — PSF fitting is also run in gas mode (same algorithm as star mode). These metrics are populated in the CSV/HTML output and `psf_signal_weight` contributes to the Gas Score (weight 0.15). They are informational in the context of narrowband imaging — star shape doesn't affect nebula detail — but help discriminate between otherwise similar frames and catch severe tracking or focus issues.

## Scoring

### Star Score
```
Score = 0.25×(1 - norm_FWHM) + 0.20×(1 - norm_Ecc) + 0.20×norm_Stars
      + 0.25×norm_PSFSignalWeight + 0.10×gradient_score(background_gradient)
```
`snr_weight` is retained in the output (CSV/HTML) for reference but has a default weight of 0 — PSFSignalWeight supersedes it because it already captures the amplitude/noise ratio with an additional 1/FWHM² correction. It can be re-enabled via `weight_snr` in `astro_eval.toml` if PSF fitting is unreliable in your data.

### Gas Score
```
Score = 0.25×norm_SNR + 0.20×(1 - norm_Noise) + 0.10×(1 - norm_BG) + 0.15×norm_Stars
      + 0.15×norm_PSFSignalWeight + 0.15×gradient_score(background_gradient)
```
PSF fitting is also run in gas mode to populate `psf_signal_weight`, `wfwhm`, and `moffat_beta`.

### gradient_score non-linear penalty

`gradient_score` is **not** a simple linear normalization. It uses a piecewise function with a knee at **1.2× the session median gradient**:

- `gradient ≤ gradient_knee × median` → score 1.0 → 0.9 (gentle, ≤10% penalty for typical frames)
- `gradient > gradient_knee × median` → score 0.9 → 0.0 (steep, reaches 0 at ~4× median)

This means frames with a normal LP gradient are barely penalised, while frames where one side of the sky is dramatically brighter (sunrise, twilight, cloud edge) are pushed toward 0. The gas mode weight (0.15) is higher than star mode (0.10) because a large gradient is more destructive to faint narrowband signal.

All metrics are normalized to [0, 1] across the session. Score of 1.0 is the best frame in the session. Statistics are computed **independently per filter** in multi-filter mode.

## Rejection Criteria

### Star Mode

| Criterion | Condition | Hard reject? |
|-----------|-----------|-------------|
| `high_fwhm` | FWHM > session_median + `--sigma-fwhm` × σ **OR** FWHM > `--fwhm-threshold` | Yes |
| `high_eccentricity` | Eccentricity > `--ecc-threshold` | Yes |
| `low_stars` | Stars < session_median × `--star-fraction` | Yes |
| `low_snr_weight` | SNR weight < session_median × `--snr-fraction` | Yes |
| `high_residual` | PSF residual > session_median + `--sigma-residual` × σ | No (informational) |

### Gas Mode

| Criterion | Condition | Hard reject? |
|-----------|-----------|-------------|
| `high_noise` | Background RMS > session_median + `--sigma-noise` × σ | Yes |
| `high_background` | Background median > session_median + `--sigma-bg` × σ | Yes |
| `low_snr` | SNR estimate < session_median × `--snr-fraction` | Yes |
| `low_stars` | Stars < session_median × `--star-fraction` | Yes |

### All Modes

| Criterion | Condition | Hard reject? |
|-----------|-----------|-------------|
| `high_gradient` | Background gradient > `--gradient-threshold` | Yes |
| `airplane_trail` | Airplane/double contrail detected | Yes |
| `satellite_trail` | Single satellite trail detected | No (informational) |

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
