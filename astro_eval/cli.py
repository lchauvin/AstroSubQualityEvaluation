"""
cli.py - Command-line interface for astro-eval.

Entry point: astro-eval <input_dir> [options]
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import sys
import threading
import time
import urllib.parse
import webbrowser
from concurrent.futures import ProcessPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import List, Optional, Tuple

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

# Filter -> mode mapping
_NARROWBAND_FILTERS = {
    "ha", "h-alpha", "h_alpha", "halpha", "h", "hydrogen-alpha",
    "oiii", "o3", "o-iii", "o", "oxygen",
    "sii", "s2", "s-ii", "s", "sulfur",
    "narrowband", "nb",
}


def _detect_mode_from_filter(filter_name: Optional[str], verbose: bool = False) -> str:
    """Return 'gas' for narrowband filters, 'star' for broadband, with warnings."""
    if filter_name is None:
        if verbose:
            print("  [warn] No FILTER keyword in header; defaulting to star mode.")
        return "star"
    key = filter_name.lower().strip()
    if key in _NARROWBAND_FILTERS:
        return "gas"
    broadband = {"r", "red", "g", "green", "b", "blue", "l", "lum", "luminance", "rgb", "clear"}
    if key in broadband:
        return "star"
    if verbose:
        print(f"  [warn] Unknown filter '{filter_name}'; defaulting to star mode.")
    return "star"


def _process_frame(
    args_tuple: Tuple,
) -> Tuple:
    """
    Module-level worker for parallel frame processing.

    Must be at module level (not nested) so it can be pickled by multiprocessing.

    Parameters
    ----------
    args_tuple:
        (fpath, focal_length_mm, mode_arg, config)

    Returns
    -------
    (FrameMetrics | None, filename_str, error_str | None)
    """
    fpath, focal_length_mm, mode_arg, config = args_tuple

    # Deferred imports so each worker subprocess loads them fresh
    from .image_loader import load_image
    from .metrics import compute_metrics

    try:
        fits_data = load_image(fpath, focal_length_mm=focal_length_mm)
    except Exception as exc:
        return None, str(Path(fpath).name), f"Load error: {exc}"

    effective_mode = (
        _detect_mode_from_filter(fits_data.filter_name)
        if mode_arg == "auto"
        else mode_arg
    )

    try:
        frame_metrics = compute_metrics(fits_data, config, mode_override=effective_mode)
        return frame_metrics, frame_metrics.filename, None
    except Exception as exc:
        return None, str(Path(fpath).name), f"Metric error: {exc}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astro-eval",
        description=(
            "Astrophotography sub-frame quality evaluation tool.\n"
            "Evaluates FITS files for FWHM, eccentricity, star count, "
            "background noise, and SNR."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  astro-eval /data/session1 --html
  astro-eval /data/ha_session --mode gas --html --verbose
  astro-eval /data/lum --fwhm-threshold 4.0 --ecc-threshold 0.4
        """,
    )

    parser.add_argument(
        "input_dir",
        metavar="INPUT_DIR",
        help="Directory containing FITS files to evaluate.",
    )

    parser.add_argument(
        "--mode",
        choices=["star", "gas", "auto"],
        default="auto",
        help=(
            "Evaluation mode: 'star' (broadband/PSF metrics), "
            "'gas' (narrowband/SNR metrics), or 'auto' (detect from FILTER header). "
            "Default: auto"
        ),
    )

    parser.add_argument(
        "--output",
        metavar="OUTPUT_DIR",
        default=None,
        help="Output directory for reports. Default: same as INPUT_DIR.",
    )

    parser.add_argument(
        "--focal-length",
        type=float,
        default=250.0,
        metavar="MM",
        help="Telescope focal length in mm. Default: 250 (Redcat 51).",
    )

    parser.add_argument(
        "--fwhm-threshold",
        type=float,
        default=5.0,
        metavar="ARCSEC",
        help="Absolute FWHM rejection threshold in arcseconds. Default: 5.0",
    )

    parser.add_argument(
        "--ecc-threshold",
        type=float,
        default=0.5,
        metavar="VALUE",
        help="Eccentricity rejection threshold [0-1]. Default: 0.5",
    )

    parser.add_argument(
        "--star-fraction",
        type=float,
        default=0.7,
        metavar="FRAC",
        help="Minimum star count fraction vs session median. Default: 0.7",
    )

    parser.add_argument(
        "--snr-fraction",
        type=float,
        default=0.5,
        metavar="FRAC",
        help="Minimum SNR weight fraction vs session median. Default: 0.5",
    )

    parser.add_argument(
        "--sigma-fwhm",
        type=float,
        default=2.0,
        metavar="SIGMA",
        help="Sigma multiplier for FWHM statistical rejection. Default: 2.0",
    )

    parser.add_argument(
        "--sigma-noise",
        type=float,
        default=2.5,
        metavar="SIGMA",
        help="Sigma multiplier for noise statistical rejection. Default: 2.5",
    )

    parser.add_argument(
        "--sigma-bg",
        type=float,
        default=3.0,
        metavar="SIGMA",
        help="Sigma multiplier for background level rejection. Default: 3.0",
    )

    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=5.0,
        metavar="SIGMA",
        help="Star detection SNR threshold (sigma above background). Default: 5.0",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of parallel worker processes. "
            "0 = use all CPU cores. Default: 1 (serial)."
        ),
    )

    parser.add_argument(
        "--html",
        action="store_true",
        default=False,
        help="Generate HTML report in addition to CSV.",
    )

    parser.add_argument(
        "--serve",
        action="store_true",
        default=False,
        help=(
            "Serve the HTML report on a local HTTP server and open it in the browser. "
            "Enables the 'Move to _REJECTED' button to move files directly without "
            "downloading a script. Implies --html."
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7420,
        metavar="PORT",
        help="Port for the local report server (used with --serve). Default: 7420.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print verbose progress information.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"astro-eval {__version__}",
    )

    return parser


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_progress(current: int, total: int, filename: str) -> None:
    pct = int(100 * current / total)
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    # Truncate filename for display
    name = filename[:40] + "…" if len(filename) > 40 else filename
    print(f"\r  [{bar}] {pct:3d}%  {name:<42}", end="", flush=True)


def _render_preview(filepath: Path) -> bytes:
    """
    Load a FITS/XISF image and return a PNG-encoded bytes object suitable
    for serving as a browser preview.

    Applies an asinh stretch between the 0.5th and 99.5th percentile so
    faint detail and bright stars are both visible.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg

    from .image_loader import load_image

    fits_data = load_image(str(filepath))
    img = fits_data.data.astype(np.float32)

    # Percentile stretch
    lo, hi = np.percentile(img, [0.5, 99.5])
    if hi > lo:
        img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    else:
        img = np.zeros_like(img)

    # Asinh stretch to show faint/bright detail simultaneously
    img = np.arcsinh(img * 3.0) / np.arcsinh(3.0)

    buf = io.BytesIO()
    mpimg.imsave(buf, img, cmap="gray", format="png", origin="lower")
    return buf.getvalue()


def _make_handler(html_path: Path, source_dir: Path):
    """
    Build a request-handler class bound to a specific html_path and source_dir.

    Uses a factory so the handler can access these paths without globals.
    """
    class ReportHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in ("/", "/report"):
                body = html_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/preview/"):
                filename = urllib.parse.unquote(self.path[len("/preview/"):])
                filepath = source_dir / filename
                try:
                    body = _render_preview(filepath)
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except Exception as exc:
                    msg = str(exc).encode()
                    self.send_response(500)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(msg)))
                    self.end_headers()
                    self.wfile.write(msg)
            elif self.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self) -> None:
            if self.path == "/move":
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length))
                filenames = payload.get("filenames", [])

                rejected_dir = source_dir / "_REJECTED"
                rejected_dir.mkdir(exist_ok=True)

                moved, errors = [], []
                for fn in filenames:
                    src = source_dir / fn
                    dst = rejected_dir / fn
                    try:
                        shutil.move(str(src), str(dst))
                        moved.append(fn)
                    except Exception as exc:
                        errors.append(f"{fn}: {exc}")

                resp = json.dumps({"moved": moved, "errors": errors}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, fmt, *args) -> None:  # suppress request logs
            pass

    return ReportHandler


def _serve_report(html_path: Path, source_dir: Path, port: int) -> None:
    """
    Start a local HTTP server, open the report in the browser, and block
    until the user presses Ctrl+C.
    """
    handler = _make_handler(html_path, source_dir)

    # Try the requested port; if taken, find the next available one
    for p in range(port, port + 20):
        try:
            server = HTTPServer(("127.0.0.1", p), handler)
            port = p
            break
        except OSError:
            continue
    else:
        print(f"[error] Could not bind to any port in {port}–{port+19}.", file=sys.stderr)
        return

    url = f"http://127.0.0.1:{port}/"
    print(f"\nReport server running at {url}")
    print("Press Ctrl+C to stop.\n")

    # Open browser after a short delay so the server is ready
    threading.Timer(0.5, webbrowser.open, args=[url]).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the astro-eval CLI.

    Returns
    -------
    Exit code: 0 on success, 1 on error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    # -----------------------------------------------------------------------
    # Validate inputs
    # -----------------------------------------------------------------------
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"[error] Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1
    if not input_dir.is_dir():
        print(f"[error] Input path is not a directory: {input_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output) if args.output else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-define report paths (used both for the existing-report check and later)
    csv_path  = output_dir / "astro_eval_report.csv"
    html_path = output_dir / "astro_eval_report.html"

    # -----------------------------------------------------------------------
    # Check for an existing report and ask the user whether to recompute
    # -----------------------------------------------------------------------
    report_exists = csv_path.exists() or html_path.exists()
    if report_exists:
        existing = html_path if html_path.exists() else csv_path
        try:
            ans = input(
                f"Report already exists ({existing.name}). Recompute? [y/N] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"

        if ans not in ("y", "yes"):
            for p in (csv_path, html_path):
                if p.exists():
                    print(f"Existing report: {p}")
            if args.serve and html_path.exists():
                _serve_report(html_path, input_dir, port=args.port)
            return 0

    # -----------------------------------------------------------------------
    # Imports (deferred so --version / --help are fast)
    # -----------------------------------------------------------------------
    from .image_loader import find_fits_files, load_image  # noqa: F401 (used in worker)
    from .metrics import EvalConfig
    from .scoring import evaluate_session
    from .report import generate_csv_report, generate_html_report

    # -----------------------------------------------------------------------
    # Find FITS files
    # -----------------------------------------------------------------------
    fits_files = find_fits_files(input_dir)
    if not fits_files:
        print(f"[error] No FITS files found in {input_dir}", file=sys.stderr)
        return 1

    n_workers = args.workers if args.workers > 0 else os.cpu_count() or 1

    print(f"astro-eval v{__version__}")
    print(f"Input:   {input_dir}")
    print(f"Output:  {output_dir}")
    print(f"Mode:    {args.mode}")
    print(f"Workers: {n_workers}")
    print(f"Found {len(fits_files)} FITS file(s)")
    print()

    # -----------------------------------------------------------------------
    # Build config
    # -----------------------------------------------------------------------
    config = EvalConfig(
        focal_length_mm=args.focal_length,
        detection_threshold=args.detection_threshold,
        fwhm_threshold_arcsec=args.fwhm_threshold,
        ecc_threshold=args.ecc_threshold,
        star_count_fraction=args.star_fraction,
        snr_fraction=args.snr_fraction,
        sigma_fwhm=args.sigma_fwhm,
        sigma_noise=args.sigma_noise,
        sigma_bg=args.sigma_bg,
        mode=args.mode,
        verbose=args.verbose,
    )

    # -----------------------------------------------------------------------
    # Load and evaluate each frame (serial or parallel)
    # -----------------------------------------------------------------------
    print("Processing frames...")
    t_start = time.monotonic()

    all_metrics = []
    n_load_errors = 0
    n_total_files = len(fits_files)

    worker_args = [
        (fpath, args.focal_length, args.mode, config)
        for fpath in fits_files
    ]

    if n_workers == 1:
        # Serial path — simpler progress and verbose output
        for i, wargs in enumerate(worker_args):
            _print_progress(i, n_total_files, Path(wargs[0]).name)
            frame_metrics, fname, error = _process_frame(wargs)
            if error:
                print(f"\n  [warn] {fname}: {error}")
                n_load_errors += 1
            else:
                all_metrics.append(frame_metrics)
    else:
        # Parallel path — use ProcessPoolExecutor
        # Map futures back to filenames for progress reporting
        future_to_name = {}
        n_done = 0
        deferred_warnings = []

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for wargs in worker_args:
                future = pool.submit(_process_frame, wargs)
                future_to_name[future] = Path(wargs[0]).name

            for future in as_completed(future_to_name):
                n_done += 1
                fname = future_to_name[future]
                _print_progress(n_done, n_total_files, fname)
                try:
                    frame_metrics, fname, error = future.result()
                except Exception as exc:
                    deferred_warnings.append(f"{fname}: unexpected error: {exc}")
                    n_load_errors += 1
                    continue

                if error:
                    deferred_warnings.append(f"{fname}: {error}")
                    n_load_errors += 1
                else:
                    all_metrics.append(frame_metrics)

        # Print warnings after progress bar clears
        if deferred_warnings:
            print()
            for w in deferred_warnings:
                print(f"  [warn] {w}")

    _print_progress(n_total_files, n_total_files, "done")
    print(f"\n\nProcessed {len(all_metrics)} frames "
          f"({n_load_errors} load/compute errors) "
          f"in {time.monotonic() - t_start:.1f}s")

    if not all_metrics:
        print("[error] No frames were successfully processed.", file=sys.stderr)
        return 1

    if len(all_metrics) < 3:
        print(
            f"[warn] Only {len(all_metrics)} frames available for session statistics. "
            "Rejection thresholds may be unreliable (minimum recommended: 3)."
        )

    # -----------------------------------------------------------------------
    # Session evaluation (statistics + scoring + rejection)
    # -----------------------------------------------------------------------
    print("\nComputing session statistics and scores...")
    session_stats, results = evaluate_session(all_metrics, config)

    # -----------------------------------------------------------------------
    # Generate reports
    # -----------------------------------------------------------------------
    generate_csv_report(results, csv_path)
    print(f"CSV report: {csv_path}")

    if args.html or args.serve:
        generate_html_report(results, session_stats, html_path, source_dir=input_dir)
        print(f"HTML report: {html_path}")

    if args.serve:
        _serve_report(html_path, input_dir, port=args.port)

    # -----------------------------------------------------------------------
    # Print summary to stdout
    # -----------------------------------------------------------------------
    n_total = len(results)
    n_rejected = sum(1 for r in results if r.rejection.rejected)
    n_accepted = n_total - n_rejected

    print()
    print("=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(f"  Total frames:    {n_total}")
    print(f"  Accepted:        {n_accepted}  ({n_accepted/n_total*100:.1f}%)")
    print(f"  Rejected:        {n_rejected}  ({n_rejected/n_total*100:.1f}%)")
    print()

    # Session metric summary
    for name, ss in session_stats.items():
        if ss.count == 0:
            continue
        print(f"  {name:30s}: median={ss.median:.4g}  std={ss.std:.4g}  n={ss.count}")

    print()

    # Per-criterion breakdown
    all_flags: dict = {}
    for r in results:
        for flag, val in r.rejection.flags.items():
            if val:
                all_flags[flag] = all_flags.get(flag, 0) + 1

    if all_flags:
        print("  Rejection breakdown:")
        for flag, count in sorted(all_flags.items(), key=lambda x: -x[1]):
            print(f"    {flag:30s}: {count} frame(s)")
        print()

    # List rejected files
    rejected_results = [r for r in results if r.rejection.rejected]
    if rejected_results:
        print("  Rejected frames:")
        for r in sorted(rejected_results, key=lambda x: x.metrics.filename):
            reasons = ", ".join(r.rejection.rejection_reasons)
            score_str = f"{r.score:.3f}" if not (r.score != r.score) else "N/A"
            print(f"    [{score_str}] {r.metrics.filename}  ({reasons})")
    else:
        print("  All frames accepted.")

    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
