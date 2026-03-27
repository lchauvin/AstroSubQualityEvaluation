"""
astro_eval_siril.py - pySiril integration script for astro-eval.

Runs from within Siril 1.4+ via the pyscript command:
    pyscript /path/to/astro_eval_siril.py

What this script does:
  1. Asks you to pick the folder containing your FITS/XISF light frames
     (Siril's current working directory is pre-selected for convenience)
  2. Runs astro-eval on that folder and generates an HTML quality report
  3. Auto-deselects rejected frames in the currently loaded Siril sequence
  4. Opens the HTML report in your browser

Prerequisites:
  - Siril 1.4+
  - astro-eval installed via the Windows installer (so the exe is on PATH)
    or available at %LOCALAPPDATA%\\Programs\\astro-eval\\astro-eval.exe

Usage:
    pyscript /path/to/astro_eval_siril.py
"""

from __future__ import annotations

import csv
import glob
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

try:
    import sirilpy as s
    from sirilpy import SirilConnectionError
except ImportError:
    print(
        "[astro-eval] ERROR: sirilpy module not found.\n"
        "This script must be run from within Siril 1.4+ via the pyscript command.",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    _HAS_TK = True
except ImportError:
    _HAS_TK = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_exe() -> str | None:
    """Locate the astro-eval executable on PATH or in the default install dir."""
    exe = shutil.which("astro-eval")
    if exe:
        return exe
    # Default Windows installer location
    candidate = os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "Programs", "astro-eval", "astro-eval.exe",
    )
    if os.path.isfile(candidate):
        return candidate
    return None


def ask_directory(default_dir: str) -> str | None:
    """Show a Tkinter folder picker dialog. Returns chosen path or None if cancelled."""
    if not _HAS_TK:
        return None
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes("-topmost", True)
    chosen = filedialog.askdirectory(
        parent=root,
        initialdir=default_dir,
        title="astro-eval — Select FITS/XISF directory to analyze",
    )
    root.destroy()
    return chosen if chosen else None


def run_evaluation(exe: str, input_dir: Path, siril: s.SirilInterface) -> bool:
    """
    Run astro-eval on input_dir. Returns True on success.

    Uses --html to generate the report file without starting the HTTP server
    (which would block). The HTML file is opened directly afterwards.
    """
    siril.log(f"Running astro-eval on: {input_dir}", color=s.LogColor.DEFAULT)
    try:
        proc = subprocess.run(
            [exe, str(input_dir), "--html"],
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        siril.log(f"[astro-eval] Failed to launch exe: {exc}", color=s.LogColor.RED)
        return False

    # Echo stdout to Siril's log
    for line in proc.stdout.splitlines():
        siril.log(f"  {line}")
    for line in proc.stderr.splitlines():
        siril.log(f"  {line}", color=s.LogColor.SALMON)

    if proc.returncode != 0:
        siril.log(
            f"[astro-eval] Process exited with code {proc.returncode}",
            color=s.LogColor.RED,
        )
        return False

    return True


def find_csv_reports(input_dir: Path) -> list[Path]:
    """Return all astro_eval_report*.csv files in input_dir."""
    pattern = str(input_dir / "astro_eval_report*.csv")
    return [Path(p) for p in glob.glob(pattern)]


def parse_rejections(csv_paths: list[Path]) -> set[str]:
    """
    Read all CSV report files and return the set of filenames marked as rejected.
    Handles both single-filter (astro_eval_report.csv) and multi-filter
    (astro_eval_report_Ha.csv, etc.) layouts.
    """
    rejected: set[str] = set()
    for path in csv_paths:
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("rejected", "0").strip() == "1":
                        rejected.add(row.get("filename", "").strip())
        except OSError as exc:
            print(f"[astro-eval] Could not read {path}: {exc}", file=sys.stderr)
    return rejected


def deselect_in_siril(
    siril: s.SirilInterface,
    rejected_files: set[str],
) -> tuple[int, int]:
    """
    Deselect rejected frames in the currently loaded Siril sequence.
    Returns (n_deselected, n_total).
    """
    if not siril.is_sequence_loaded():
        siril.log(
            "[astro-eval] No sequence loaded — skipping frame deselection.",
            color=s.LogColor.DEFAULT,
        )
        return 0, 0

    seq = siril.get_seq()
    n_total = seq.number
    n_deselected = 0

    for i in range(n_total):
        try:
            frame_path = siril.get_seq_frame_filename(i)
            fname = os.path.basename(frame_path)
        except Exception:
            continue

        if fname in rejected_files:
            siril.set_seq_frame_incl(i, False)
            siril.log(f"  Deselected: {fname}", color=s.LogColor.SALMON)
            n_deselected += 1
        else:
            siril.set_seq_frame_incl(i, True)

    return n_deselected, n_total


def open_report(input_dir: Path, siril: s.SirilInterface) -> None:
    """Open the HTML report in the default browser."""
    html_path = input_dir / "astro_eval_report.html"
    if not html_path.exists():
        # Multi-filter: pick the first one found
        candidates = list(input_dir.glob("astro_eval_report*.html"))
        if candidates:
            html_path = candidates[0]

    if html_path.exists():
        siril.log(f"Opening report: {html_path}", color=s.LogColor.GREEN)
        webbrowser.open(html_path.as_uri())
    else:
        siril.log(
            "[astro-eval] HTML report not found — check for errors above.",
            color=s.LogColor.SALMON,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    siril = s.SirilInterface()

    try:
        siril.connect()
    except SirilConnectionError as exc:
        print(f"[astro-eval] Could not connect to Siril: {exc}", file=sys.stderr)
        sys.exit(1)

    siril.log("━━━ astro-eval ━━━", color=s.LogColor.BLUE)

    # Locate the executable
    exe = find_exe()
    if not exe:
        siril.log(
            "[astro-eval] ERROR: astro-eval.exe not found.\n"
            "  Install it from this project's Releases page\n"
            "  Or ensure it is on your system PATH.",
            color=s.LogColor.RED,
        )
        return

    siril.log(f"Using: {exe}", color=s.LogColor.DEFAULT)

    # Ask for input directory
    default_dir = siril.get_siril_wd()
    input_dir_str = ask_directory(default_dir)

    if not input_dir_str:
        siril.log("[astro-eval] Cancelled.", color=s.LogColor.DEFAULT)
        return

    input_dir = Path(input_dir_str)
    if not input_dir.is_dir():
        siril.log(
            f"[astro-eval] Directory not found: {input_dir}",
            color=s.LogColor.RED,
        )
        return

    # Run evaluation
    ok = run_evaluation(exe, input_dir, siril)
    if not ok:
        return

    # Parse rejections from CSV
    csv_paths = find_csv_reports(input_dir)
    if not csv_paths:
        siril.log(
            "[astro-eval] No CSV report found after evaluation.",
            color=s.LogColor.SALMON,
        )
        return

    rejected_files = parse_rejections(csv_paths)

    # Deselect rejected frames in Siril sequence
    n_deselected, n_total = deselect_in_siril(siril, rejected_files)

    # Summary
    siril.log("━━━ Summary ━━━", color=s.LogColor.BLUE)
    if n_total > 0:
        n_accepted = n_total - n_deselected
        siril.log(
            f"  Sequence: {n_accepted}/{n_total} frames kept "
            f"({n_deselected} deselected)",
            color=s.LogColor.GREEN if n_deselected == 0 else s.LogColor.DEFAULT,
        )
    else:
        n_rejected_total = len(rejected_files)
        siril.log(
            f"  Evaluated: {n_rejected_total} frames rejected",
            color=s.LogColor.DEFAULT,
        )

    # Open report in browser
    open_report(input_dir, siril)


if __name__ == "__main__":
    main()
