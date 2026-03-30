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

Structure your response with exactly these section labels (surrounder by <b></b> tags) on their own lines:
  \n\nOVERVIEW
  \n\nSEEING
  \n\nTRACKING AND GUIDING
  \n\nTRANSPARENCY
  \n\nROOT CAUSES
  \n\nRECOMMENDATIONS
  \n\nQUALITY RATING

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
    ]

    if config.bortle > 0:
        header.extend(f"Sky Bortle: {config.bortle}")
    header.extend("")

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
Take into consideration in your analysis of the light pollution (Bortle) if precised.
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

_SECTION_LABELS = {
    "OVERVIEW", "SEEING", "TRACKING AND GUIDING", "TRANSPARENCY",
    "ROOT CAUSES", "RECOMMENDATIONS", "QUALITY RATING",
}

# Sections rendered side-by-side in a 2-column grid
_TWO_COL_SECTIONS = {"SEEING", "TRACKING AND GUIDING"}

_RATING_COLORS = {
    "excellent": "#27ae60",
    "good":      "#2ecc71",
    "fair":      "#f39c12",
    "poor":      "#e74c3c",
    "unusable":  "#922b21",
}

_AI_STYLES = """\
<style>
.ae-ai-card{background:#fff;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,.08);
  padding:1.5rem 2rem;margin:1.5rem 0;border-top:4px solid #3498db}
.ae-ai-header{display:flex;align-items:center;gap:.75rem;margin-bottom:1.25rem}
.ae-ai-header h2{margin:0;color:#2c3e50;font-size:1.4rem}
.ae-ai-model{font-family:monospace;font-size:.75rem;background:#eaf2fb;
  color:#2980b9;padding:.2rem .55rem;border-radius:4px;white-space:nowrap}
.ae-ai-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:.75rem 0}
.ae-ai-section h3{font-size:.85rem;font-weight:700;letter-spacing:.06em;
  text-transform:uppercase;color:#2c3e50;margin:1rem 0 .3rem;border-bottom:1px solid #ecf0f1;
  padding-bottom:.25rem}
.ae-ai-section p{margin:.35rem 0;color:#333;line-height:1.7;font-size:.92rem}
.ae-ai-rating{display:inline-block;padding:.35rem 1rem;border-radius:20px;
  color:#fff;font-weight:700;font-size:.9rem;margin-top:.4rem}
.ae-ai-footer{color:#7f8c8d;font-size:.75rem;margin-top:1.25rem;
  padding-top:.75rem;border-top:1px solid #ecf0f1}
</style>"""


def _strip_html_tags(text: str) -> str:
    """Remove HTML tags for label detection."""
    return re.sub(r"<[^>]+>", "", text).strip()


def _text_to_html(text: str) -> str:
    """
    Convert LLM plain-text output to structured HTML cards.

    Handles section labels in plain or <b>LABEL</b> format.
    Pairs SEEING + TRACKING AND GUIDING into a 2-column grid.
    Adds a coloured badge for the QUALITY RATING section.
    """
    # Split on blank lines
    paragraphs = re.split(r"\n{2,}", text.strip())

    # Build list of (label_or_None, content_lines)
    sections: List[tuple] = []  # (label: str|None, html: str)
    current_label: Optional[str] = None
    current_parts: List[str] = []

    def _flush():
        nonlocal current_label, current_parts
        if current_parts or current_label is not None:
            sections.append((current_label, "\n".join(current_parts)))
        current_label = None
        current_parts = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        clean = _strip_html_tags(para).upper()
        if clean in _SECTION_LABELS:
            _flush()
            current_label = clean
        else:
            para_html = para.replace("\n", "<br>\n")
            current_parts.append(f'<p>{para_html}</p>')

    _flush()

    # Render sections
    html_parts: List[str] = []
    i = 0
    while i < len(sections):
        label, body = sections[i]

        if label is None:
            # Preamble text before any section label
            html_parts.append(f'<div class="ae-ai-section">{body}</div>')
            i += 1
            continue

        if label == "QUALITY RATING":
            # Extract rating word for badge colour
            rating_word = ""
            for word in ["excellent", "good", "fair", "poor", "unusable"]:
                if word in body.lower():
                    rating_word = word
                    break
            color = _RATING_COLORS.get(rating_word, "#7f8c8d")
            badge = (
                f'<span class="ae-ai-rating" style="background:{color}">'
                f'{rating_word.title()}</span>' if rating_word else ""
            )
            html_parts.append(
                f'<div class="ae-ai-section">'
                f'<h3>Quality Rating</h3>{body}{badge}</div>'
            )
            i += 1
            continue

        if label in _TWO_COL_SECTIONS:
            # Try to pair with the other 2-col section if it follows immediately
            next_label = sections[i + 1][0] if i + 1 < len(sections) else None
            if next_label in _TWO_COL_SECTIONS and next_label != label:
                _, body2 = sections[i + 1]
                col1 = (
                    f'<div class="ae-ai-section">'
                    f'<h3>{label.title()}</h3>{body}</div>'
                )
                col2 = (
                    f'<div class="ae-ai-section">'
                    f'<h3>{next_label.title()}</h3>{body2}</div>'
                )
                html_parts.append(f'<div class="ae-ai-grid">{col1}{col2}</div>')
                i += 2
                continue

        # Normal full-width section
        html_parts.append(
            f'<div class="ae-ai-section">'
            f'<h3>{label.title()}</h3>{body}</div>'
        )
        i += 1

    return "\n".join(html_parts)


def inject_analysis_html(html_path: Path, analysis_text: str, model_str: str) -> None:
    """
    Inject the AI analysis card into the HTML report.

    Multi-filter (tabbed) reports: injects inside the Summary tab so the card
    is only visible when the Summary tab is active.

    Single-filter reports: injects before <footer>.
    """
    html_body = _text_to_html(analysis_text)

    injection = f"""{_AI_STYLES}
<div class="ae-ai-card">
  <div class="ae-ai-header">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
         xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <circle cx="12" cy="12" r="10" stroke="#3498db" stroke-width="2"/>
      <path d="M8 12h8M12 8v8" stroke="#3498db" stroke-width="2"
            stroke-linecap="round"/>
    </svg>
    <h2>AI Session Analysis</h2>
    <span class="ae-ai-model">{model_str}</span>
  </div>
  {html_body}
  <div class="ae-ai-footer">Generated by astro-eval --analysis</div>
</div>
"""
    try:
        content = html_path.read_text(encoding="utf-8")

        # Multi-filter tabbed report: inject as the first child of the Summary tab
        # pane so it is hidden/shown together with the tab by the existing JS.
        summary_marker = 'id="tab-summary"'
        if summary_marker in content:
            summary_pos = content.find(summary_marker)
            # Move past the opening tag's closing '>'
            open_tag_end = content.find(">", summary_pos) + 1
            if open_tag_end > 0:
                content = content[:open_tag_end] + "\n" + injection + content[open_tag_end:]
                html_path.write_text(content, encoding="utf-8")
                return

        # Single-filter report: inject before the page footer
        if "<footer" in content:
            content = content.replace("<footer", injection + "\n<footer", 1)
        elif "</body>" in content:
            content = content.replace("</body>", injection + "\n</body>", 1)
        else:
            logger.warning("Could not inject analysis: no known anchor found in %s", html_path)
            return
        html_path.write_text(content, encoding="utf-8")
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
