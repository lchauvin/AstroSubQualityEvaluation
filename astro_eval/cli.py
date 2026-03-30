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
import queue
import shutil
import sys
import threading
import time
import urllib.parse
import webbrowser
from concurrent.futures import ProcessPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

# Filter -> mode mapping
_NARROWBAND_FILTERS = {
    "ha", "h-alpha", "h_alpha", "halpha", "h", "hydrogen-alpha",
    "oiii", "o3", "o-iii", "o", "oxygen",
    "sii", "s2", "s-ii", "s", "sulfur",
    "narrowband", "nb",
}

# Canonical names for display
_FILTER_CANONICAL = {
    "ha": "Ha", "h-alpha": "Ha", "h_alpha": "Ha", "halpha": "Ha",
    "h": "Ha", "hydrogen-alpha": "Ha",
    "oiii": "OIII", "o3": "OIII", "o-iii": "OIII", "o": "OIII", "oxygen": "OIII",
    "sii": "SII", "s2": "SII", "s-ii": "SII", "s": "SII", "sulfur": "SII",
    "narrowband": "NB", "nb": "NB",
}

_FITS_EXTENSIONS = {".fits", ".fit", ".fts", ".xisf"}


# ---------------------------------------------------------------------------
# SSE broadcaster
# ---------------------------------------------------------------------------

class _SSEBroadcaster:
    """Thread-safe Server-Sent Events broadcaster."""

    def __init__(self) -> None:
        self._clients: List[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            try:
                self._clients.remove(q)
            except ValueError:
                pass

    def broadcast(self, event: str = "reload") -> None:
        with self._lock:
            for q in self._clients:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass


# ---------------------------------------------------------------------------
# Filter directory detection
# ---------------------------------------------------------------------------

def _find_filter_dirs(root: Path) -> Dict[str, Path]:
    """
    Scan root for subdirectories containing FITS/XISF files.

    The canonical filter name is derived from the subdirectory name (matched
    against known aliases), or from the FILTER header of the first file found.

    Returns {canonical_filter_name: path}, or {} if root has no filter subdirs.
    """
    result: Dict[str, Path] = {}

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        fits_in_subdir = [
            f for f in subdir.iterdir()
            if f.is_file() and f.suffix.lower() in _FITS_EXTENSIONS
        ]
        if not fits_in_subdir:
            continue

        # Try to derive canonical name from directory name
        key = subdir.name.lower().strip()
        canonical = _FILTER_CANONICAL.get(key)

        if canonical is None:
            # Fall back to reading FILTER header from first file
            try:
                from .image_loader import load_image
                fd = load_image(str(fits_in_subdir[0]))
                if fd.filter_name:
                    canonical = _FILTER_CANONICAL.get(fd.filter_name.lower().strip(),
                                                       fd.filter_name)
                else:
                    canonical = subdir.name  # use dir name as-is
            except Exception:
                canonical = subdir.name

        result[canonical] = subdir

    return result


def _is_multi_filter_root(input_dir: Path) -> bool:
    """True if input_dir contains filter subdirs rather than FITS files directly."""
    has_direct_fits = any(
        f.suffix.lower() in _FITS_EXTENSIONS
        for f in input_dir.iterdir()
        if f.is_file()
    )
    if has_direct_fits:
        return False
    filter_dirs = _find_filter_dirs(input_dir)
    return len(filter_dirs) > 0


# ---------------------------------------------------------------------------
# Frame processing worker
# ---------------------------------------------------------------------------

def _detect_mode_from_filter(filter_name: Optional[str], verbose: bool = False) -> str:
    """Return 'gas' for narrowband filters, 'star' for broadband."""
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


def _process_frame(args_tuple: Tuple) -> Tuple:
    """
    Module-level worker for parallel frame processing.
    args_tuple: (fpath, focal_length_mm, pixel_size_um, mode_arg, config)
    Returns: (FrameMetrics | None, filename_str, error_str | None)
    """
    fpath, focal_length_mm, pixel_size_um, mode_arg, config = args_tuple

    from .image_loader import load_image
    from .metrics import compute_metrics

    try:
        fits_data = load_image(fpath, focal_length_mm=focal_length_mm, pixel_size_um=pixel_size_um)
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


def _process_directory(
    fits_files: List[str],
    config,
    focal_length: float,
    mode: str,
    n_workers: int,
    label: str = "",
    pixel_size_um: Optional[float] = None,
) -> Tuple[List, int]:
    """
    Process a list of FITS file paths.
    Returns (list[FrameMetrics], n_errors).
    """
    all_metrics: List = []
    n_errors = 0
    n_total = len(fits_files)
    worker_args = [(fp, focal_length, pixel_size_um, mode, config) for fp in fits_files]

    prefix = f"[{label}] " if label else ""

    if n_workers == 1:
        for i, wargs in enumerate(worker_args):
            _print_progress(i, n_total, prefix + Path(wargs[0]).name)
            metrics, fname, error = _process_frame(wargs)
            if error:
                print(f"\n  [warn] {fname}: {error}")
                n_errors += 1
            else:
                all_metrics.append(metrics)
    else:
        future_to_name: Dict = {}
        n_done = 0
        deferred: List[str] = []

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for wargs in worker_args:
                future = pool.submit(_process_frame, wargs)
                future_to_name[future] = Path(wargs[0]).name

            for future in as_completed(future_to_name):
                n_done += 1
                fname = future_to_name[future]
                _print_progress(n_done, n_total, prefix + fname)
                try:
                    metrics, fname, error = future.result()
                except Exception as exc:
                    deferred.append(f"{fname}: unexpected error: {exc}")
                    n_errors += 1
                    continue
                if error:
                    deferred.append(f"{fname}: {error}")
                    n_errors += 1
                else:
                    all_metrics.append(metrics)

        if deferred:
            print()
            for w in deferred:
                print(f"  [warn] {w}")

    _print_progress(n_total, n_total, prefix + "done")
    print()
    return all_metrics, n_errors


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astro-eval",
        description=(
            "Astrophotography sub-frame quality evaluation tool.\n"
            "Evaluates FITS/XISF files for FWHM, eccentricity, star count, "
            "background noise, and SNR."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  astro-eval /data/session1 --html
  astro-eval /data/SHO_root --html --serve          # auto-detects Ha/OIII/SII subdirs
  astro-eval /data/session --serve --watch          # live updates every 30 s
  astro-eval /data/staging --serve --watch \\
      --remote acqpc --remote-dir "C:/Astro/Light" # pull + watch from remote PC
        """,
    )

    parser.add_argument("input_dir", metavar="INPUT_DIR",
                        help="Directory containing FITS files (or filter subdirectories).")
    parser.add_argument("--mode", choices=["star", "gas", "auto"], default="auto",
                        help="Evaluation mode. Default: auto (detect from FILTER header).")
    parser.add_argument("--output", metavar="OUTPUT_DIR", default=None,
                        help="Output directory for reports. Default: same as INPUT_DIR.")
    parser.add_argument("--focal-length", type=float, default=250.0, metavar="MM",
                        help="Telescope focal length in mm. Default: 250.")
    parser.add_argument("--fwhm-threshold", type=float, default=5.0, metavar="ARCSEC",
                        help="Absolute FWHM rejection limit in arcsec. Default: 5.0")
    parser.add_argument("--ecc-threshold", type=float, default=0.5, metavar="VALUE",
                        help="Eccentricity rejection limit [0-1]. Default: 0.5")
    parser.add_argument("--star-fraction", type=float, default=0.7, metavar="FRAC",
                        help="Min star count fraction vs session median. Default: 0.7")
    parser.add_argument("--snr-fraction", type=float, default=0.5, metavar="FRAC",
                        help="Min SNR weight fraction vs session median. Default: 0.5")
    parser.add_argument("--sigma-fwhm", type=float, default=2.0, metavar="SIGMA",
                        help="Sigma multiplier for FWHM statistical rejection. Default: 2.0")
    parser.add_argument("--sigma-noise", type=float, default=2.5, metavar="SIGMA",
                        help="Sigma multiplier for noise statistical rejection. Default: 2.5")
    parser.add_argument("--sigma-bg", type=float, default=3.0, metavar="SIGMA",
                        help="Sigma multiplier for background rejection. Default: 3.0")
    parser.add_argument("--sigma-residual", type=float, default=3.0, metavar="SIGMA",
                        help="Sigma multiplier for PSF residual flag (informational). Default: 3.0")
    parser.add_argument("--sigma-gradient", type=float, default=2.0, metavar="SIGMA",
                        help="Session-relative gradient rejection: reject if gradient > median + sigma × std. Default: 2.0")
    parser.add_argument("--gradient-threshold", type=float, default=0.0, metavar="SIGMA",
                        help="Absolute gradient rejection hard cap in noise σ units. 0 = disabled. Default: 0.")
    parser.add_argument("--gradient-knee", type=float, default=1.2, metavar="RATIO",
                        help="Gradient scoring knee: multiplier starts dropping above knee × session_median. Default: 1.2")
    parser.add_argument("--min-score", type=float, default=0.5, metavar="SCORE",
                        help="Reject frames with composite score below this value. 0 = disabled. Default: 0.5")
    parser.add_argument("--detection-threshold", type=float, default=5.0, metavar="SIGMA",
                        help="Star detection SNR threshold. Default: 5.0")
    parser.add_argument("--workers", type=int, default=0, metavar="N",
                        help="Parallel worker processes. 0 = all CPU cores. Default: 0.")
    parser.add_argument("--html", action="store_true", default=False,
                        help="Generate HTML report in addition to CSV.")
    parser.add_argument("--subframeselector", action="store_true", default=False,
                        help="Export a PixInsight SubFrameSelector-compatible CSV (astro_eval_sfs.csv).")
    parser.add_argument("--serve", action="store_true", default=False,
                        help="Serve HTML report on a local HTTP server. Implies --html.")
    parser.add_argument("--port", type=int, default=7420, metavar="PORT",
                        help="Port for the local report server. Default: 7420.")
    parser.add_argument("--watch", action="store_true", default=False,
                        help="Watch for new frames and update the report every 30 s. Implies --serve.")
    parser.add_argument("--remote", metavar="HOST", default=None,
                        help="Hostname/IP of remote acquisition PC for SFTP file pull (used with --watch).")
    parser.add_argument("--remote-dir", metavar="DIR", default=None,
                        help="Remote directory to pull new frames from (Windows path OK).")
    parser.add_argument("--remote-user", metavar="USER", default=None,
                        help="SSH username for remote. Default: current OS user.")
    parser.add_argument("--remote-key", metavar="PATH", default=None,
                        help="Path to SSH private key. Default: ~/.ssh/id_rsa.")
    parser.add_argument("--local-staging", metavar="DIR", default=None,
                        help="Local directory to stage files downloaded from remote. "
                             "Default: <output_dir>/_staging.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Print verbose progress information.")
    parser.add_argument("--config", metavar="FILE", default=None,
                        help="Path to TOML config file. Default: auto-detect astro_eval.toml "
                             "in INPUT_DIR or current directory.")
    parser.add_argument("--analysis", action="store_true", default=False,
                        help=(
                            "Run LLM-powered session diagnosis after evaluation. "
                            "Configure model in astro_eval.toml [analysis] section "
                            "(e.g. model = \"anthropic/claude-haiku-4-5-20251001\"). "
                            "API keys are read from .env in the session directory "
                            "or from environment variables."
                        ))
    parser.add_argument("--version", action="version", version=f"astro-eval {__version__}")

    return parser


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_progress(current: int, total: int, filename: str) -> None:
    pct    = int(100 * current / total) if total else 100
    filled = int(30 * current / total) if total else 30
    bar    = "#" * filled + "-" * (30 - filled)
    name   = filename[:40] + "\u2026" if len(filename) > 40 else filename
    print(f"\r  [{bar}] {pct:3d}%  {name:<42}", end="", flush=True)


# ---------------------------------------------------------------------------
# Image preview renderer
# ---------------------------------------------------------------------------

def _render_preview(filepath: Path) -> bytes:
    """Load a FITS/XISF image and return a PNG with asinh stretch."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg

    from .image_loader import load_image

    fits_data = load_image(str(filepath))
    img = fits_data.data.astype(np.float32)

    lo, hi = np.percentile(img, [0.5, 99.5])
    if hi > lo:
        img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    else:
        img = np.zeros_like(img)

    img = np.arcsinh(img * 3.0) / np.arcsinh(3.0)

    buf = io.BytesIO()
    mpimg.imsave(buf, img, cmap="gray", format="png", origin="lower")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

def _make_handler(
    html_path: Path,
    source_dirs: Dict[str, Path],
    broadcaster: _SSEBroadcaster,
):
    """
    Build a request-handler class.

    source_dirs: {filter_name: path}
      For single-filter mode pass {"": source_dir}.
      For multi-filter pass {filter_name: filter_dir, ...}.
    """
    class ReportHandler(BaseHTTPRequestHandler):
        @staticmethod
        def _safe_file_path(base_dir: Path, filename: str) -> Optional[Path]:
            """
            Resolve filename under base_dir and reject traversal/invalid paths.
            """
            if not isinstance(filename, str) or not filename:
                return None
            # This API expects plain filenames, not nested paths.
            if Path(filename).name != filename:
                return None
            try:
                base_resolved = base_dir.resolve()
                candidate = (base_resolved / filename).resolve()
                candidate.relative_to(base_resolved)
            except Exception:
                return None
            return candidate

        def do_GET(self) -> None:
            path = urllib.parse.unquote(self.path.split("?")[0])

            if path in ("/", "/report"):
                if not html_path.exists():
                    body = (
                        b"<!DOCTYPE html><html><head><meta charset='utf-8'>"
                        b"<title>astro-eval \xe2\x80\x94 waiting</title></head>"
                        b"<body style='font-family:sans-serif;text-align:center;margin-top:15vh'>"
                        b"<h2>Waiting for first SFTP pull\xe2\x80\xa6</h2>"
                        b"<p>The page will reload automatically when the first report is ready.</p>"
                        b"<script>"
                        b"var es=new EventSource('/events');"
                        b"es.onmessage=function(e){if(e.data==='reload')location.reload();};"
                        b"es.onerror=function(){setTimeout(function(){location.reload();},5000);};"
                        b"</script>"
                        b"</body></html>"
                    )
                    self._send(200, "text/html; charset=utf-8", body)
                    return
                body = html_path.read_bytes()
                self._send(200, "text/html; charset=utf-8", body)

            elif path.startswith("/preview/"):
                # /preview/<filter>/<filename>  or  /preview/<filename> (single-filter)
                parts = path[len("/preview/"):].split("/", 1)
                if len(parts) == 2:
                    fid, filename = parts
                    src_dir = source_dirs.get(fid) or next(iter(source_dirs.values()), None)
                else:
                    filename = parts[0]
                    src_dir = next(iter(source_dirs.values()), None)

                if src_dir is None or not filename:
                    self._send(404, "text/plain", b"Not found")
                    return
                filepath = self._safe_file_path(src_dir, filename)
                if filepath is None or not filepath.is_file():
                    self._send(404, "text/plain", b"Not found")
                    return
                try:
                    body = _render_preview(filepath)
                    self._send(200, "image/png", body)
                except Exception as exc:
                    self._send(500, "text/plain", str(exc).encode())

            elif path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                q = broadcaster.subscribe()
                try:
                    while True:
                        try:
                            event = q.get(timeout=25)
                            self.wfile.write(f"data: {event}\n\n".encode())
                        except queue.Empty:
                            self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    broadcaster.unsubscribe(q)

            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()

            else:
                self._send(404, "text/plain", b"Not found")

        def do_POST(self) -> None:
            if self.path == "/move":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._send_json({"moved": [], "errors": ["Invalid Content-Length"]}, code=400)
                    return
                if length <= 0 or length > 5 * 1024 * 1024:
                    self._send_json({"moved": [], "errors": ["Invalid request body size"]}, code=400)
                    return
                try:
                    payload = json.loads(self.rfile.read(length))
                except Exception:
                    self._send_json({"moved": [], "errors": ["Invalid JSON payload"]}, code=400)
                    return
                if not isinstance(payload, dict):
                    self._send_json({"moved": [], "errors": ["JSON payload must be an object"]}, code=400)
                    return

                fid = payload.get("filter", "")
                filenames = payload.get("filenames", [])
                if not isinstance(filenames, list):
                    self._send_json({"moved": [], "errors": ["'filenames' must be a list"]}, code=400)
                    return

                # Resolve source dir: named filter, or single-filter (""), or first available
                src_dir = (
                    source_dirs.get(fid)
                    or source_dirs.get("")
                    or next(iter(source_dirs.values()), None)
                )
                if src_dir is None:
                    self._send_json({"moved": [], "errors": ["No source directory configured"]})
                    return

                rejected_dir = src_dir / "_REJECTED"
                rejected_dir.mkdir(parents=True, exist_ok=True)

                moved, errors = [], []
                for fn in filenames:
                    if not isinstance(fn, str):
                        errors.append("Invalid filename entry")
                        continue
                    src_path = self._safe_file_path(src_dir, fn)
                    if src_path is None:
                        errors.append(f"{fn}: invalid filename")
                        continue
                    dst_path = rejected_dir / src_path.name
                    try:
                        shutil.move(str(src_path), str(dst_path))
                        moved.append(fn)
                    except Exception as exc:
                        errors.append(f"{fn}: {exc}")

                self._send_json({"moved": moved, "errors": errors})
            else:
                self._send(404, "text/plain", b"Not found")

        def _send(self, code: int, content_type: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, data: dict, code: int = 200) -> None:
            body = json.dumps(data).encode()
            self._send(code, "application/json", body)

        def handle(self) -> None:
            try:
                super().handle()
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                pass  # browser closed the connection mid-request — harmless on Windows

        def log_message(self, fmt, *args) -> None:
            pass  # suppress request logs

    return ReportHandler


def _make_server(
    html_path: Path,
    source_dirs: Dict[str, Path],
    broadcaster: _SSEBroadcaster,
    port: int,
) -> Tuple[Optional[ThreadingHTTPServer], str]:
    """Create a ThreadingHTTPServer. Returns (server, url) or (None, '')."""
    handler = _make_handler(html_path, source_dirs, broadcaster)
    for p in range(port, port + 20):
        try:
            server = ThreadingHTTPServer(("0.0.0.0", p), handler)
            return server, f"http://localhost:{p}/"
        except OSError:
            continue
    return None, ""


def _serve_blocking(
    html_path: Path,
    source_dirs: Dict[str, Path],
    broadcaster: _SSEBroadcaster,
    port: int,
) -> None:
    """Start the server in the main thread (blocking). Used without --watch."""
    server, url = _make_server(html_path, source_dirs, broadcaster, port)
    if server is None:
        print(f"[error] Could not bind to any port in {port}–{port+19}.", file=sys.stderr)
        return

    print(f"\nReport server running at {url}")
    print("Press Ctrl+C to stop.\n")
    threading.Timer(0.5, webbrowser.open, args=[url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


# ---------------------------------------------------------------------------
# Remote SFTP sync
# ---------------------------------------------------------------------------

def _sftp_connect(
    host: str,
    username: str,
    key_path: Optional[Path],
) -> "paramiko.SSHClient":
    """
    Open and return a persistent paramiko SSHClient.
    Auth order: explicit key file → auto key discovery (~/.ssh/) → password prompt.
    """
    import paramiko
    import getpass

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    base_kwargs: Dict = dict(username=username, timeout=15)

    if key_path and key_path.exists():
        # Explicit key provided — use it directly, no password fallback needed
        client.connect(host, key_filename=str(key_path), **base_kwargs)
        return client

    # Try key auto-discovery (~/.ssh/id_rsa, id_ed25519, etc.) first
    try:
        client.connect(host, look_for_keys=True, allow_agent=False, **base_kwargs)
        return client
    except paramiko.AuthenticationException:
        pass

    # Fall back to password / keyboard-interactive
    pw = getpass.getpass(f"SSH password for {username}@{host}: ")
    client.connect(
        host,
        password=pw,
        look_for_keys=False,
        allow_agent=False,
        **base_kwargs,
    )
    return client


def _sftp_discover_filters(
    client: "paramiko.SSHClient",
    remote_base_dir: str,
    local_staging: Path,
) -> Dict[str, str]:
    """
    List the remote base dir and return {canonical_filter: remote_subdir} for any
    subdirectory whose name matches a known filter alias.
    Falls back to an empty dict if no filter subdirs are found.
    """
    remote_base = remote_base_dir.replace("\\", "/").rstrip("/")
    sftp = client.open_sftp()
    try:
        entries = sftp.listdir_attr(remote_base)
    except Exception:
        return {}
    finally:
        sftp.close()

    import stat as stat_mod
    result: Dict[str, str] = {}
    for entry in entries:
        if not stat_mod.S_ISDIR(entry.st_mode):
            continue
        key = entry.filename.lower().strip()
        canonical = _FILTER_CANONICAL.get(key, entry.filename)
        result[canonical] = entry.filename

    return result


def _sftp_sync(
    client: "paramiko.SSHClient",
    remote_base_dir: str,
    local_base_dir: Path,
    filter_subdirs: Dict[str, str],   # {local_filter_name: remote_subdir_name}
    already_synced: set,
) -> List[Path]:
    """
    Download new FITS/XISF files via an already-open SSHClient.
    Returns list of newly downloaded local paths.
    """
    new_files: List[Path] = []
    remote_base = remote_base_dir.replace("\\", "/").rstrip("/")

    sftp = client.open_sftp()
    try:
        for local_filter, remote_subdir in filter_subdirs.items():
            remote_dir = f"{remote_base}/{remote_subdir}"
            local_dir  = local_base_dir / local_filter
            local_dir.mkdir(parents=True, exist_ok=True)

            try:
                remote_files = sftp.listdir(remote_dir)
            except FileNotFoundError:
                continue

            for fname in remote_files:
                if Path(fname).suffix.lower() not in _FITS_EXTENSIONS:
                    continue
                rel = f"{local_filter}/{fname}"
                if rel in already_synced:
                    continue
                try:
                    sftp.get(f"{remote_dir}/{fname}", str(local_dir / fname))
                    already_synced.add(rel)
                    new_files.append(local_dir / fname)
                    print(f"  [sync] {fname}")
                except Exception as exc:
                    print(f"  [warn] Failed to download {fname}: {exc}")
    finally:
        sftp.close()

    return new_files


# ---------------------------------------------------------------------------
# Watch loop
# ---------------------------------------------------------------------------

def _watch_loop(
    input_dirs: Dict[str, Path],
    config,
    args,
    html_path: Path,
    csv_path: Path,
    broadcaster: _SSEBroadcaster,
    initial_filter_metrics: Dict[str, List],
    is_multi_filter: bool,
    weights=None,
    pixel_size_um: Optional[float] = None,
    interval: int = 30,
    remote_host: Optional[str] = None,
    remote_dir: Optional[str] = None,
    remote_user: Optional[str] = None,
    remote_key: Optional[Path] = None,
    local_staging: Optional[Path] = None,
) -> None:
    """Poll for new frames, reprocess session stats, regenerate report, notify browser."""
    from .image_loader import find_fits_files
    from .scoring import evaluate_session
    from .report import generate_html_report, generate_multi_filter_html_report, generate_csv_report, generate_subframeselector_csv

    # Track processed file paths per filter
    processed: Dict[str, set] = {
        fid: {m.filepath for m in metrics}
        for fid, metrics in initial_filter_metrics.items()
    }
    filter_metrics: Dict[str, List] = {
        fid: list(metrics) for fid, metrics in initial_filter_metrics.items()
    }

    # For remote sync: track already-synced relative paths
    sftp_synced: set = set()
    if remote_host and local_staging:
        for fid, ldir in input_dirs.items():
            for f in ldir.glob("*"):
                if f.suffix.lower() in _FITS_EXTENSIONS:
                    sftp_synced.add(f"{fid}/{f.name}")

    if remote_host:
        print(f"Watching remote {remote_host}:{remote_dir or ''}")

    # Establish persistent SSH connection upfront (avoids per-poll reconnect and
    # the paramiko.Agent thread-reuse bug on Windows).
    ssh_client = None
    if remote_host and remote_dir and local_staging:
        try:
            import paramiko  # noqa: F401 — verify installed before loop
            ssh_client = _sftp_connect(
                host=remote_host,
                username=remote_user or os.getlogin(),
                key_path=remote_key,
            )
            print(f"Connected to {remote_host} via SSH.")
            # Discover filter subdirs from remote (e.g. H/, S/, O/)
            remote_filter_subdirs = _sftp_discover_filters(
                ssh_client, remote_dir, local_staging
            )
            if remote_filter_subdirs:
                print(f"  Remote filters: {list(remote_filter_subdirs.keys())}")
                # Ensure local staging dirs exist and are in input_dirs / filter_metrics
                for canonical, remote_sub in remote_filter_subdirs.items():
                    local_dir = local_staging / canonical
                    local_dir.mkdir(parents=True, exist_ok=True)
                    if canonical not in input_dirs:
                        input_dirs[canonical] = local_dir
                    filter_metrics.setdefault(canonical, [])
                    processed.setdefault(canonical, set())
            else:
                # No subdirs found — treat remote base as flat directory
                remote_filter_subdirs = {"": ""}
        except Exception as exc:
            print(f"[error] Could not connect to {remote_host}: {exc}", file=sys.stderr)
            ssh_client = None
            remote_filter_subdirs = {"": ""}
    else:
        remote_filter_subdirs = {fid: fid for fid in input_dirs} or {"": ""}

    print(f"Watch mode active (polling every {interval}s). Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(interval)
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] Checking for new frames...")

            # --- Optional: pull from remote ---
            if ssh_client and remote_dir and local_staging:
                try:
                    new_remote = _sftp_sync(
                        client=ssh_client,
                        remote_base_dir=remote_dir,
                        local_base_dir=local_staging,
                        filter_subdirs=remote_filter_subdirs,
                        already_synced=sftp_synced,
                    )
                except Exception as exc:
                    print(f"  [warn] SFTP error: {exc}. Reconnecting...")
                    try:
                        ssh_client.close()
                    except Exception:
                        pass
                    try:
                        ssh_client = _sftp_connect(
                            host=remote_host,
                            username=remote_user or os.getlogin(),
                            key_path=remote_key,
                        )
                        new_remote = _sftp_sync(
                            client=ssh_client,
                            remote_base_dir=remote_dir,
                            local_base_dir=local_staging,
                            filter_subdirs=remote_filter_subdirs,
                            already_synced=sftp_synced,
                        )
                    except Exception as exc2:
                        print(f"  [warn] Reconnect failed: {exc2}")
                        new_remote = []
                if new_remote:
                    print(f"  Downloaded {len(new_remote)} file(s) from remote")

            # --- Scan local dirs for new files ---
            n_new = 0
            for fid, fdir in input_dirs.items():
                known = processed.setdefault(fid, set())
                new_files = [
                    f for f in find_fits_files(fdir)
                    if str(f) not in known
                ]
                if not new_files:
                    continue

                print(f"  Processing {len(new_files)} new {fid} frame(s)...")
                for fpath in new_files:
                    metrics, fname, error = _process_frame(
                        (str(fpath), args.focal_length, pixel_size_um, args.mode, config)
                    )
                    if error:
                        print(f"    [warn] {fname}: {error}")
                    else:
                        filter_metrics.setdefault(fid, []).append(metrics)
                        n_new += 1
                    known.add(str(fpath))

            if n_new == 0:
                print("  No new frames.")
                continue

            print(f"  {n_new} new frame(s). Regenerating report...")

            # --- Re-evaluate session(s) and regenerate report ---
            try:
                # Use actual filter count — may differ from startup if remote
                # subdirs were discovered after an initially empty staging dir.
                effective_multi = is_multi_filter or len(filter_metrics) > 1
                if effective_multi:
                    filter_data: Dict = {}
                    for fid, metrics in filter_metrics.items():
                        if metrics:
                            stats, results = evaluate_session(metrics, config, weights)
                            filter_data[fid] = (results, stats)
                            fid_safe = "".join(c if c.isalnum() else "_" for c in fid)
                            fp = csv_path.parent / f"astro_eval_report_{fid_safe}.csv"
                            generate_csv_report(results, fp)
                            if args.subframeselector:
                                sfs_path = csv_path.parent / f"astro_eval_sfs_{fid_safe}.csv"
                                generate_subframeselector_csv(results, sfs_path)
                    generate_multi_filter_html_report(filter_data, html_path, input_dirs, weights=weights, config=config)
                else:
                    fid = next(iter(filter_metrics))
                    metrics = filter_metrics[fid]
                    if metrics:
                        stats, results = evaluate_session(metrics, config, weights)
                        generate_html_report(results, stats, html_path,
                                             source_dir=input_dirs[fid], weights=weights, config=config)
                        generate_csv_report(results, csv_path)
                        if args.subframeselector:
                            sfs_path = csv_path.parent / "astro_eval_sfs.csv"
                            generate_subframeselector_csv(results, sfs_path)

                broadcaster.broadcast("reload")
                print("  Report updated. Browser refreshing.")
            except Exception as exc:
                print(f"  [warn] Report regeneration failed: {exc}")

    except KeyboardInterrupt:
        print("\nWatch mode stopped.")
    finally:
        if ssh_client:
            try:
                ssh_client.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    _configure_logging(args.verbose)

    # -----------------------------------------------------------------------
    # TOML config file — apply defaults for any CLI arg left at its default
    # -----------------------------------------------------------------------
    from .config_loader import load_config, find_config_file
    from .metrics import ScoringWeights

    _config_path = Path(args.config) if args.config else find_config_file(Path(args.input_dir))
    try:
        _cfg = load_config(_config_path)
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    if _cfg:
        print(f"Config: {_config_path}")

    # Sentinel defaults — only apply config value when the CLI arg was not explicitly set
    _CLI_DEFAULTS = {
        "focal_length": 250.0, "fwhm_threshold": 5.0, "ecc_threshold": 0.5,
        "star_fraction": 0.7,  "snr_fraction": 0.5,   "sigma_fwhm": 2.0,
        "sigma_noise": 2.5,    "sigma_bg": 3.0,        "sigma_residual": 3.0,
        "sigma_gradient": 2.0,
        "gradient_threshold": 0.0,
        "gradient_knee": 1.2,
        "min_score": 0.5,
        "detection_threshold": 5.0,
        "workers": 0,          "mode": "auto",         "html": False,
        "serve": False,        "port": 7420,           "verbose": False,
    }
    _CFG_MAP = {
        "focal_length":       "telescope.focal_length_mm",
        "fwhm_threshold":     "rejection.fwhm_threshold_arcsec",
        "ecc_threshold":      "rejection.ecc_threshold",
        "star_fraction":      "rejection.star_count_fraction",
        "snr_fraction":       "rejection.snr_fraction",
        "sigma_fwhm":         "rejection.sigma_fwhm",
        "sigma_noise":        "rejection.sigma_noise",
        "sigma_bg":           "rejection.sigma_bg",
        "sigma_residual":     "rejection.sigma_residual",
        "sigma_gradient":     "rejection.sigma_gradient",
        "gradient_threshold": "rejection.gradient_threshold",
        "gradient_knee":      "rejection.gradient_knee",
        "min_score":          "rejection.min_score",
        "detection_threshold":"processing.detection_threshold",
        "workers":            "processing.workers",
        "mode":               "processing.mode",
        "html":               "output.html",
        "serve":              "output.serve",
        "port":               "output.port",
        "verbose":            "output.verbose",
    }
    for attr, cfg_key in _CFG_MAP.items():
        if getattr(args, attr) == _CLI_DEFAULTS[attr] and cfg_key in _cfg:
            setattr(args, attr, _cfg[cfg_key])

    # Pixel size fallback from config (used when FITS headers lack XPIXSZ/PIXSIZE1)
    pixel_size_um: Optional[float] = _cfg.get("camera.pixel_size_um")

    # Build scoring weights from config (falls back to defaults if not set)
    weights = ScoringWeights(
        star_fwhm  = _cfg.get("scoring.star.weight_fwhm",  0.30),
        star_ecc   = _cfg.get("scoring.star.weight_ecc",   0.25),
        star_stars = _cfg.get("scoring.star.weight_stars", 0.20),
        star_snr   = _cfg.get("scoring.star.weight_snr",   0.00),
        star_psfsw = _cfg.get("scoring.star.weight_psfsw", 0.25),
        gas_snr    = _cfg.get("scoring.gas.weight_snr",    0.30),
        gas_noise  = _cfg.get("scoring.gas.weight_noise",  0.20),
        gas_bg     = _cfg.get("scoring.gas.weight_bg",     0.15),
        gas_stars  = _cfg.get("scoring.gas.weight_stars",  0.20),
        gas_psfsw  = _cfg.get("scoring.gas.weight_psfsw",  0.15),
    )

    # --watch implies --serve
    if args.watch:
        args.serve = True
    if args.serve:
        args.html = True

    # -----------------------------------------------------------------------
    # Validate inputs
    # -----------------------------------------------------------------------
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"[error] Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1
    if not input_dir.is_dir():
        print(f"[error] Not a directory: {input_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output) if args.output else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path  = output_dir / "astro_eval_report.csv"
    html_path = output_dir / "astro_eval_report.html"

    # -----------------------------------------------------------------------
    # Check for existing report
    # -----------------------------------------------------------------------
    if not args.watch:  # in watch mode, always recompute
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
                    broadcaster = _SSEBroadcaster()
                    _serve_blocking(html_path, {"": input_dir}, broadcaster, args.port)
                return 0

    # -----------------------------------------------------------------------
    # Deferred imports
    # -----------------------------------------------------------------------
    from .image_loader import find_fits_files
    from .metrics import EvalConfig
    from .scoring import evaluate_session
    from .report import generate_csv_report, generate_html_report, generate_multi_filter_html_report, generate_subframeselector_csv

    # -----------------------------------------------------------------------
    # Detect multi-filter vs single-filter
    # -----------------------------------------------------------------------
    is_multi = _is_multi_filter_root(input_dir)

    remote_watch = args.watch and args.remote

    if is_multi:
        filter_dirs = _find_filter_dirs(input_dir)
        if not filter_dirs:
            if not remote_watch:
                print(f"[error] No FITS files found in subdirectories of {input_dir}",
                      file=sys.stderr)
                return 1
            # Remote watch with empty staging: filter subdirs will be created on first pull
            filter_dirs = {}
        print(f"astro-eval v{__version__}  [multi-filter mode]")
        print(f"Root:    {input_dir}")
        print(f"Output:  {output_dir}")
        for fid, fdir in filter_dirs.items():
            n = sum(1 for f in fdir.iterdir()
                    if f.is_file() and f.suffix.lower() in _FITS_EXTENSIONS)
            print(f"  {fid:8s}  {fdir}  ({n} files)")
    else:
        filter_dirs = {"": input_dir}
        fits_files  = find_fits_files(input_dir)
        if not fits_files:
            if not remote_watch:
                print(f"[error] No FITS files found in {input_dir}", file=sys.stderr)
                return 1
            print(f"astro-eval v{__version__}")
            print(f"Input:   {input_dir}")
            print(f"Output:  {output_dir}")
            print(f"Waiting for remote files from {args.remote} ...")
        else:
            print(f"astro-eval v{__version__}")
            print(f"Input:   {input_dir}")
            print(f"Output:  {output_dir}")
            print(f"Mode:    {args.mode}")
            print(f"Found {len(fits_files)} FITS file(s)")

    n_workers = args.workers if args.workers > 0 else os.cpu_count() or 1
    print(f"Workers: {n_workers}\n")

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
        sigma_residual=args.sigma_residual,
        sigma_gradient=args.sigma_gradient,
        gradient_threshold=args.gradient_threshold,
        gradient_knee=args.gradient_knee,
        min_score=args.min_score,
        mode=args.mode,
        verbose=args.verbose,
    )

    # -----------------------------------------------------------------------
    # Process frames
    # -----------------------------------------------------------------------
    t_start = time.monotonic()
    filter_metrics: Dict[str, List] = {}

    if is_multi:
        for fid, fdir in filter_dirs.items():
            fits_files_f = find_fits_files(fdir)
            if not fits_files_f:
                print(f"  [warn] No FITS files in {fdir}")
                continue
            print(f"Processing {fid} ({len(fits_files_f)} files)...")
            metrics, n_err = _process_directory(
                fits_files_f, config, args.focal_length, args.mode, n_workers,
                label=fid, pixel_size_um=pixel_size_um,
            )
            filter_metrics[fid] = metrics
            print(f"  Done: {len(metrics)} frames ({n_err} errors)  "
                  f"[{time.monotonic()-t_start:.1f}s elapsed]")
    else:
        print("Processing frames...")
        fits_files = find_fits_files(input_dir)
        metrics, n_err = _process_directory(
            fits_files, config, args.focal_length, args.mode, n_workers,
            pixel_size_um=pixel_size_um,
        )
        filter_metrics[""] = metrics
        print(f"\nProcessed {len(metrics)} frames ({n_err} errors) "
              f"in {time.monotonic()-t_start:.1f}s")

    # -----------------------------------------------------------------------
    # Session evaluation
    # -----------------------------------------------------------------------
    broadcaster = _SSEBroadcaster()
    has_initial_frames = any(bool(m) for m in filter_metrics.values())

    if not has_initial_frames and remote_watch:
        # Empty staging at startup — skip initial report; watch loop will generate it
        print("No local frames yet. Waiting for first SFTP pull...")
        filter_data: Dict = {}
    elif is_multi:
        print("\nComputing session statistics...")
        filter_data = {}
        for fid, metrics in filter_metrics.items():
            if not metrics:
                continue
            if len(metrics) < 3:
                print(f"  [warn] {fid}: only {len(metrics)} frames; "
                      "thresholds may be unreliable.")
            stats, results = evaluate_session(metrics, config, weights)
            filter_data[fid] = (results, stats)

        if not filter_data:
            print("[error] No frames processed in any filter.", file=sys.stderr)
            return 1

        # CSV: one file per filter
        for fid, (results, _) in filter_data.items():
            fid_safe = "".join(c if c.isalnum() else "_" for c in fid)
            fp = output_dir / f"astro_eval_report_{fid_safe}.csv"
            generate_csv_report(results, fp)
            print(f"CSV report: {fp}")
            if args.subframeselector:
                sfs_path = output_dir / f"astro_eval_sfs_{fid_safe}.csv"
                generate_subframeselector_csv(results, sfs_path)
                print(f"SubFrameSelector CSV: {sfs_path}")

        if args.html or args.serve:
            generate_multi_filter_html_report(filter_data, html_path, filter_dirs, weights=weights, config=config)
            print(f"HTML report: {html_path}")

        # Summary to stdout
        print()
        print("=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        for fid, (results, _) in filter_data.items():
            nt = len(results)
            nr = sum(1 for r in results if r.rejection.rejected)
            na = nt - nr
            print(f"  {fid:8s}  total={nt}  accepted={na}  rejected={nr}  "
                  f"({na/nt*100:.1f}% pass)")
        print("=" * 60)

    else:
        print("\nComputing session statistics...")
        metrics = filter_metrics.get("", [])
        if not metrics:
            print("[error] No frames were successfully processed.", file=sys.stderr)
            return 1
        if len(metrics) < 3:
            print(f"[warn] Only {len(metrics)} frames; thresholds may be unreliable.")

        session_stats, results = evaluate_session(metrics, config, weights)
        filter_data = {"": (results, session_stats)}

        generate_csv_report(results, csv_path)
        print(f"CSV report: {csv_path}")

        if args.subframeselector:
            sfs_path = output_dir / "astro_eval_sfs.csv"
            generate_subframeselector_csv(results, sfs_path)
            print(f"SubFrameSelector CSV: {sfs_path}")

        if args.html or args.serve:
            generate_html_report(results, session_stats, html_path, source_dir=input_dir, weights=weights, config=config)
            print(f"HTML report: {html_path}")

        # Summary to stdout
        n_total    = len(results)
        n_rejected = sum(1 for r in results if r.rejection.rejected)
        n_accepted = n_total - n_rejected
        print()
        print("=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"  Total:     {n_total}")
        print(f"  Accepted:  {n_accepted}  ({n_accepted/n_total*100:.1f}%)")
        print(f"  Rejected:  {n_rejected}  ({n_rejected/n_total*100:.1f}%)")
        print()
        for name, ss in session_stats.items():
            if ss.count:
                print(f"  {name:30s}: median={ss.median:.4g}  std={ss.std:.4g}  n={ss.count}")
        print("=" * 60)

    # -----------------------------------------------------------------------
    # LLM session analysis  (only when --analysis is passed)
    # -----------------------------------------------------------------------
    if args.analysis and filter_data:
        from .analysis import load_dotenv, run_analysis, inject_analysis_html

        # Search for .env in: session dir, user config dir, cwd
        _env_search = [input_dir, output_dir]
        _appdata = os.environ.get("APPDATA")
        if _appdata:
            _env_search.append(Path(_appdata) / "astro-eval")
        else:
            _env_search.append(Path.home() / ".config" / "astro_eval")
        _env_search.append(Path.cwd())
        load_dotenv(_env_search)

        _model_str   = _cfg.get("analysis.model", "").strip()
        _ollama_url  = _cfg.get("analysis.ollama_url", None)
        _max_tokens  = int(_cfg.get("analysis.max_tokens", 1500))

        if not _model_str:
            print(
                "\n[analysis] No model configured. Add to astro_eval.toml:\n"
                "  [analysis]\n"
                "  model = \"anthropic/claude-haiku-4-5-20251001\"\n"
                "  # or: model = \"openai/gpt-4o-mini\"\n"
                "  # or: model = \"ollama/qwen3:14b\"",
                file=sys.stderr,
            )
        else:
            print(f"\nRunning LLM analysis ({_model_str})...")
            try:
                analysis_text = run_analysis(
                    filter_data, _model_str, config,
                    ollama_url=_ollama_url,
                    max_tokens=_max_tokens,
                )

                # Write plain-text file
                analysis_path = output_dir / "astro_eval_analysis.txt"
                analysis_path.write_text(analysis_text, encoding="utf-8")
                print(f"Analysis saved: {analysis_path}")

                # Print to stdout
                print()
                print("=" * 60)
                print("AI SESSION ANALYSIS")
                print("=" * 60)
                print(analysis_text)
                print("=" * 60)

                # Inject into HTML report if it was generated
                if (args.html or args.serve) and html_path.exists():
                    inject_analysis_html(html_path, analysis_text, _model_str)
                    print(f"Analysis injected into HTML report.")

            except (ValueError, ImportError) as exc:
                print(f"\n[analysis] {exc}", file=sys.stderr)
            except Exception as exc:
                print(f"\n[analysis] Unexpected error: {exc}", file=sys.stderr)
                if args.verbose:
                    import traceback
                    traceback.print_exc()

    # -----------------------------------------------------------------------
    # Serve / watch
    # -----------------------------------------------------------------------
    if args.serve:
        # Build source_dirs map for the server
        if is_multi:
            server_source_dirs = filter_dirs
        else:
            server_source_dirs = {"": input_dir}

        if args.watch:
            # Resolve watch_dirs first so the server and watch loop share the same
            # dict reference — mutations in _watch_loop (e.g. adding filter subdirs
            # discovered via SFTP) are then immediately visible to the HTTP handler.
            local_staging: Optional[Path] = None
            if args.remote:
                local_staging = Path(args.local_staging) if args.local_staging else input_dir
                local_staging.mkdir(parents=True, exist_ok=True)
                if is_multi:
                    watch_dirs = {
                        fid: local_staging / fid for fid in filter_dirs
                    } if filter_dirs else {"": local_staging}
                else:
                    watch_dirs = {"": local_staging}
            else:
                watch_dirs = server_source_dirs

            # Start server in a background thread, run watch loop in main thread
            server, url = _make_server(html_path, watch_dirs, broadcaster, args.port)
            if server is None:
                print(f"[error] Could not bind to port {args.port}.", file=sys.stderr)
                return 1

            srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
            srv_thread.start()
            print(f"\nReport server running at {url}")
            threading.Timer(0.5, webbrowser.open, args=[url]).start()

            _watch_loop(
                input_dirs=watch_dirs,
                config=config,
                args=args,
                html_path=html_path,
                csv_path=csv_path,
                broadcaster=broadcaster,
                initial_filter_metrics=filter_metrics,
                is_multi_filter=is_multi,
                weights=weights,
                pixel_size_um=pixel_size_um,
                interval=30,
                remote_host=args.remote,
                remote_dir=args.remote_dir,
                remote_user=args.remote_user,
                remote_key=Path(args.remote_key) if args.remote_key else None,
                local_staging=local_staging,
            )
            server.shutdown()
        else:
            _serve_blocking(html_path, server_source_dirs, broadcaster, args.port)

    return 0


if __name__ == "__main__":
    sys.exit(main())
