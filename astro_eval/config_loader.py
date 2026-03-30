"""
config_loader.py - TOML configuration file support.

Loads an optional `astro_eval.toml` config file and returns a flat dict of
settings that main() merges with CLI arguments (CLI takes precedence).

Search order for the config file:
  1. Explicit path via --config
  2. INPUT_DIR/astro_eval.toml
  3. cwd/astro_eval.toml

Example config (all fields optional):

    [telescope]
    focal_length_mm = 250.0

    [camera]
    pixel_size_um = 3.76      # fallback if not in FITS headers

    [processing]
    mode = "auto"             # "star" | "gas" | "auto"
    detection_threshold = 5.0
    workers = 0

    [rejection]
    fwhm_threshold_arcsec = 5.0
    ecc_threshold = 0.5
    star_count_fraction = 0.7
    snr_fraction = 0.5
    sigma_fwhm = 2.0
    sigma_noise = 2.5
    sigma_bg = 3.0
    sigma_residual = 3.0
    sigma_gradient = 2.0
    gradient_threshold = 0.0
    gradient_knee = 1.2

    [scoring.star]
    weight_fwhm     = 0.30
    weight_ecc      = 0.25
    weight_stars    = 0.20
    weight_psfsw    = 0.25

    [scoring.gas]
    weight_snr      = 0.30
    weight_noise    = 0.20
    weight_bg       = 0.15
    weight_stars    = 0.20
    weight_psfsw    = 0.15

    [output]
    html    = false
    serve   = false
    port    = 7420
    verbose = false

    [analysis]
    # LLM-powered session diagnosis — only runs when --analysis is passed.
    # Model format: "provider/model-id"
    #   Supported providers: anthropic | openai | ollama
    model      = "anthropic/claude-haiku-4-5-20251001"
    # model    = "anthropic/claude-sonnet-4-6"
    # model    = "openai/gpt-4o-mini"
    # model    = "ollama/qwen3:14b"
    ollama_url = "http://localhost:11434"   # only relevant for ollama provider
    max_tokens = 1500

    # API keys go in a .env file next to astro_eval.toml (or in environment):
    #   ANTHROPIC_API_KEY=sk-ant-...
    #   OPENAI_API_KEY=sk-...
    # Ollama runs locally and needs no key.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# tomllib / tomli import (stdlib on 3.11+, backport on 3.9-3.10)
# ---------------------------------------------------------------------------

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib          # type: ignore[no-redef]
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            tomllib = None  # type: ignore[assignment]


def find_config_file(input_dir: Path) -> Optional[Path]:
    """
    Search for astro_eval.toml in order of priority. Returns first found path.

    Search order:
      1. INPUT_DIR/astro_eval.toml          (session-specific override)
      2. %APPDATA%/astro-eval/astro_eval.toml  (user global config, Windows)
         ~/.config/astro_eval/astro_eval.toml  (user global config, Linux/macOS)
      3. Directory of the running executable  (install dir, useful for frozen builds)
      4. cwd/astro_eval.toml                 (fallback for dev usage)
    """
    candidates: list[Path] = [input_dir / "astro_eval.toml"]

    # User-level global config
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "astro-eval" / "astro_eval.toml")
    else:
        candidates.append(Path.home() / ".config" / "astro_eval" / "astro_eval.toml")

    # Next to the executable (works for both frozen PyInstaller builds and editable installs)
    candidates.append(Path(sys.executable).parent / "astro_eval.toml")

    # CWD fallback (useful during development)
    candidates.append(Path.cwd() / "astro_eval.toml")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """
    Load and validate a TOML config file.

    Returns a flat dict with dotted keys (e.g. "telescope.focal_length_mm").
    Returns {} if config_path is None.
    Raises FileNotFoundError if an explicit path was given but not found.
    """
    if config_path is None:
        return {}

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if tomllib is None:
        logger.warning(
            "TOML config file found (%s) but tomllib/tomli is not available. "
            "Install tomli (`pip install tomli`) for Python < 3.11. Ignoring config.",
            config_path,
        )
        return {}

    with open(config_path, "rb") as f:
        raw: dict = tomllib.load(f)

    logger.info("Loaded config from %s", config_path)

    flat: Dict[str, Any] = {}

    def _flatten(d: dict, prefix: str = "") -> None:
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(v, key)
            else:
                flat[key] = v

    _flatten(raw)

    _validate_weights(flat, "scoring.star", ["weight_fwhm", "weight_ecc", "weight_stars", "weight_snr", "weight_psfsw"])
    _validate_weights(flat, "scoring.gas",  ["weight_snr",  "weight_noise", "weight_bg",   "weight_stars", "weight_psfsw"])

    return flat


def _validate_weights(flat: dict, section: str, keys: list) -> None:
    """Warn if scoring weights in a section don't sum to ~1.0."""
    present = [flat[f"{section}.{k}"] for k in keys if f"{section}.{k}" in flat]
    if len(present) == len(keys):
        total = sum(present)
        if abs(total - 1.0) > 0.01:
            logger.warning(
                "Config [%s] weights sum to %.3f (expected 1.0). "
                "Scores will be outside [0,1] before clipping.",
                section, total,
            )
