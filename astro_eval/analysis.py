"""
analysis.py – LLM-powered session diagnostic.

Enabled only when --analysis is passed to the CLI.  All other runs are
completely unaffected — no imports from this module happen otherwise.

Model configuration (astro_eval.toml):

    [analysis]
    model      = "anthropic/claude-haiku-4-5-20251001"   # or openai/... or ollama/...
    ollama_url = "http://localhost:11434"                  # only for ollama provider
    max_tokens = 1500

Model string format: "provider/model-id"
  anthropic/claude-sonnet-4-6
  anthropic/claude-haiku-4-5-20251001
  openai/gpt-4o-mini
  openai/gpt-4o
  ollama/qwen3:14b           (uses OpenAI-compatible API at ollama_url)
  ollama/llama3.2:latest

API keys come from environment variables or a .env file searched in this order:
  1. <input_dir>/.env              (session-specific)
  2. %APPDATA%/astro-eval/.env     (user global, Windows)
     ~/.config/astro_eval/.env     (user global, Linux/macOS)
  3. <cwd>/.env                    (dev fallback)

.env format (standard KEY=VALUE):
  ANTHROPIC_API_KEY=sk-ant-...
  OPENAI_API_KEY=sk-...
  (Ollama runs locally and needs no key)
"""

from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert astrophotographer and imaging session analyst.
You are given quality metrics from an automated sub-frame evaluation tool.
Your task: diagnose the session quality, explain the root causes of any frame
rejections, and give concise, actionable recommendations.

Write in plain prose with labelled sections — do NOT use Markdown, bullet
lists, asterisks, or code blocks.  The output will be embedded in an HTML
report and a plain-text file, so keep formatting minimal.

Structure your response with exactly these section labels on their own lines:
  OVERVIEW
  SEEING
  TRACKING AND GUIDING
  TRANSPARENCY
  ROOT CAUSES
  RECOMMENDATIONS
  QUALITY RATING

Rules:
- Be specific: reference the actual metric values in your diagnosis.
- Keep each section to 2-4 sentences.
- QUALITY RATING must end with one of: Excellent / Good / Fair / Poor / Unusable.
- Total response: 300-500 words.
- Do not repeat the raw metric table back; interpret it.
"""


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def load_dotenv(search_dirs: List[Path]) -> None:
    """
    Load the first .env file found in search_dirs into os.environ.

    Uses python-dotenv when installed; falls back to a simple manual parser
    that handles the KEY=VALUE (and KEY="VALUE") format.  Existing environment
    variables are never overwritten.
    """
    for d in search_dirs:
        env_path = d / ".env"
        if not env_path.is_file():
            continue
        try:
            from dotenv import load_dotenv as _dotenv_load  # type: ignore[import]
            _dotenv_load(env_path, override=False)
            logger.info("Loaded .env via python-dotenv from %s", env_path)
        except ImportError:
            # Manual fallback: parse KEY=VALUE lines
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
            logger.info("Loaded .env (manual parse) from %s", env_path)
        return  # stop at the first .env found


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _fmt(stats: dict, name: str, precision: int = 2) -> str:
    """Format a session stat as 'median ± std' or 'n/a'."""
    s = stats.get(name)
    if s is None or s.count == 0 or not math.isfinite(s.median):
        return "n/a"
    med = s.median
    std = s.std
    if math.isfinite(std) and std > 0:
        return f"{med:.{precision}f} ± {std:.{precision}f}"
    return f"{med:.{precision}f}"


def _build_prompt(filter_data: dict, config) -> str:
    """
    Build the LLM analysis prompt from all-filter session data.

    filter_data: {filter_id: (List[FrameResult], Dict[str, SessionStats])}
    """
    header = [
        "ASTROPHOTOGRAPHY SESSION QUALITY DATA",
        f"Telescope focal length: {config.focal_length_mm:.0f} mm",
        "",
    ]

    filter_sections: List[str] = []

    for fid, (results, stats) in filter_data.items():
        if not results:
            continue

        mode = results[0].metrics.mode
        pixel_scale = next(
            (r.metrics.pixel_scale for r in results if r.metrics.pixel_scale),
            None,
        )
        n_total    = len(results)
        n_rejected = sum(1 for r in results if r.rejection.rejected)
        n_accepted = n_total - n_rejected

        scores = [r.score for r in results if math.isfinite(r.score)]
        score_med = float(np.median(scores)) if scores else float("nan")
        score_p25 = float(np.percentile(scores, 25)) if scores else float("nan")
        score_p75 = float(np.percentile(scores, 75)) if scores else float("nan")

        # Rejection flags — sorted by count descending
        flag_counts: Dict[str, int] = {}
        for r in results:
            for flag, val in r.rejection.flags.items():
                if val:
                    flag_counts[flag] = flag_counts.get(flag, 0) + 1
        rejection_lines = "\n".join(
            f"    {flag}: {cnt} / {n_total} frames ({cnt / n_total * 100:.0f}%)"
            for flag, cnt in sorted(flag_counts.items(), key=lambda x: -x[1])
            if cnt > 0
        ) or "    none"

        # Elongation statistics across all frames
        ecc_vals  = [r.metrics.eccentricity_median   for r in results if math.isfinite(r.metrics.eccentricity_median)]
        cons_vals = [r.metrics.elongation_consistency for r in results if math.isfinite(r.metrics.elongation_consistency)]

        ecc_med  = f"{float(np.median(ecc_vals)):.3f}"  if ecc_vals  else "n/a"
        cons_med = f"{float(np.median(cons_vals)):.3f}" if cons_vals else "n/a"

        # Trail summary
        n_airplane  = sum(1 for r in results if r.metrics.trail_type == "airplane")
        n_satellite = sum(1 for r in results if r.metrics.trail_type in ("satellite", "unknown"))

        # Temporal range
        obs_times = sorted(r.metrics.obs_time for r in results if r.metrics.obs_time)
        time_range_line = ""
        if len(obs_times) >= 2:
            time_range_line = f"  Time range: {obs_times[0]} to {obs_times[-1]}\n"

        filter_label = fid if fid else "single filter"
        ps_str = f"{pixel_scale:.2f} arcsec/px" if pixel_scale else "unknown"

        section = f"""\
--- FILTER / CHANNEL: {filter_label} ---
Mode: {mode} ({'narrowband' if mode == 'gas' else 'broadband'})
Pixel scale: {ps_str}
{time_range_line}\
Frames: {n_total} total | {n_accepted} accepted ({n_accepted / n_total * 100:.0f}%) | {n_rejected} rejected ({n_rejected / n_total * 100:.0f}%)
Composite score: median {score_med:.3f}  [P25={score_p25:.3f}, P75={score_p75:.3f}]  (0=worst, 1=best)

KEY METRICS (session median ± std):
  FWHM:              {_fmt(stats, 'fwhm_median', 2)} arcsec
                     Interpretation: < 2" excellent seeing, 2-4" typical, > 5" poor
  Eccentricity:      {_fmt(stats, 'eccentricity_median', 3)}
                     Interpretation: 0.0=round stars, 0.5=mild (1.15:1), 0.87=severe (2:1)
  Star count:        {_fmt(stats, 'n_stars', 0)}  (transparency proxy; drops with clouds/dew)
  BG gradient:       {_fmt(stats, 'background_gradient', 1)} sigma  (sky non-uniformity; < 20 good, > 80 severe)
  PSFSignalWeight:   {_fmt(stats, 'psf_signal_weight', 1)}  (sharpness x brightness / noise; higher = better)"""

        if mode == "gas":
            section += f"""
  SNR estimate:      {_fmt(stats, 'snr_estimate', 2)}  (narrowband signal; < 2 weak, > 10 strong)
  Background noise:  {_fmt(stats, 'background_rms', 1)} ADU"""

        section += f"""

STAR SHAPE:
  Median eccentricity:     {ecc_med}
  Elongation consistency:  {cons_med}
    (0.0 = stars elongated in random directions = guiding/seeing
     1.0 = all elongated the same way = systematic: PE, polar error, wind)

REJECTION BREAKDOWN:
{rejection_lines}

TRAILS: {n_airplane} airplane | {n_satellite} satellite/unknown
"""
        filter_sections.append(section)

    closing = """\
Please analyse the session data above.  Diagnose the dominant quality factors,
explain the specific rejection causes, interpret the elongation consistency
(systematic vs. random), and give practical advice for the next session.
End with an overall QUALITY RATING."""

    return "\n".join(header + filter_sections + [closing])


# ---------------------------------------------------------------------------
# Provider backends
# ---------------------------------------------------------------------------

def _call_anthropic(
    model: str, system: str, prompt: str, api_key: str, max_tokens: int
) -> str:
    try:
        import anthropic  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is not installed.\n"
            "Install it with:  pip install anthropic"
        )
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _call_openai_compat(
    model: str,
    system: str,
    prompt: str,
    api_key: str,
    max_tokens: int,
    base_url: Optional[str] = None,
) -> str:
    try:
        import openai  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "The 'openai' package is not installed.\n"
            "Install it with:  pip install openai"
        )
    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = openai.OpenAI(**kwargs)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _text_to_html(text: str) -> str:
    """
    Convert the LLM's plain-text output to simple HTML.

    Handles the labelled section headings the system prompt asks for
    (e.g. "OVERVIEW", "SEEING") and wraps other paragraphs in <p> tags.
    """
    _SECTION_LABELS = {
        "OVERVIEW", "SEEING", "TRACKING AND GUIDING", "TRANSPARENCY",
        "ROOT CAUSES", "RECOMMENDATIONS", "QUALITY RATING",
    }

    paragraphs = re.split(r"\n{2,}", text.strip())
    html_parts: List[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Check if the whole paragraph is a section label
        if para.upper() in _SECTION_LABELS:
            html_parts.append(
                f'<h3 style="color:#a0c4f7;margin:1.2rem 0 0.4rem">{para.title()}</h3>'
            )
        else:
            # Inline line-breaks within a paragraph
            para_html = para.replace("\n", "<br>\n")
            html_parts.append(f'<p style="margin:0.4rem 0">{para_html}</p>')

    return "\n".join(html_parts)


def inject_analysis_html(html_path: Path, analysis_text: str, model_str: str) -> None:
    """Inject the AI analysis as a styled section before </body>."""
    html_body = _text_to_html(analysis_text)
    injection = f"""
<section style="max-width:960px;margin:2rem auto;padding:1.5rem 2rem;
                background:#111d35;border-radius:8px;
                border-left:4px solid #7eb8f7;font-family:inherit">
  <h2 style="color:#7eb8f7;margin-top:0">AI Session Analysis</h2>
  <div style="color:#d0dff5;line-height:1.75;font-size:0.95rem">
    {html_body}
  </div>
  <p style="color:#5a6a8a;font-size:0.78rem;margin-bottom:0;margin-top:1rem">
    Generated by {model_str} &mdash; astro-eval --analysis
  </p>
</section>
"""
    try:
        content = html_path.read_text(encoding="utf-8")
        if "</body>" in content:
            content = content.replace("</body>", injection + "\n</body>", 1)
            html_path.write_text(content, encoding="utf-8")
        else:
            logger.warning("Could not inject analysis: </body> not found in %s", html_path)
    except Exception as exc:
        logger.warning("HTML injection failed: %s", exc)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_analysis(
    filter_data: dict,
    model_str: str,
    config,
    ollama_url: Optional[str] = None,
    max_tokens: int = 1500,
) -> str:
    """
    Run LLM analysis on the session and return the diagnosis as plain text.

    Parameters
    ----------
    filter_data:
        {filter_id: (List[FrameResult], Dict[str, SessionStats])}
    model_str:
        "provider/model-id", e.g. "anthropic/claude-haiku-4-5-20251001"
    config:
        EvalConfig (used for focal length in the prompt).
    ollama_url:
        Base URL for Ollama (default: http://localhost:11434).
    max_tokens:
        Maximum tokens in the LLM response.
    """
    provider, sep, model = model_str.partition("/")
    if not sep or not model:
        raise ValueError(
            f"Invalid model string '{model_str}'.\n"
            "Expected format: 'provider/model-id'\n"
            "Examples:\n"
            "  anthropic/claude-haiku-4-5-20251001\n"
            "  openai/gpt-4o-mini\n"
            "  ollama/qwen3:14b"
        )

    prompt = _build_prompt(filter_data, config)
    provider = provider.lower().strip()

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set.\n"
                "Add it to a .env file in your session directory, or set the environment variable."
            )
        return _call_anthropic(model, _SYSTEM_PROMPT, prompt, api_key, max_tokens)

    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set.\n"
                "Add it to a .env file in your session directory, or set the environment variable."
            )
        return _call_openai_compat(model, _SYSTEM_PROMPT, prompt, api_key, max_tokens)

    elif provider == "ollama":
        base = (ollama_url or "http://localhost:11434").rstrip("/")
        return _call_openai_compat(
            model, _SYSTEM_PROMPT, prompt,
            api_key="ollama",       # required by the SDK; Ollama ignores it
            max_tokens=max_tokens,
            base_url=base + "/v1",
        )

    else:
        raise ValueError(
            f"Unknown provider '{provider}'.\n"
            "Supported providers: anthropic, openai, ollama."
        )
