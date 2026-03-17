"""
image_loader.py - FITS and XISF image loading.

Unified loader for both formats. Use load_image() as the single entry point;
it dispatches to load_fits() or load_xisf() based on the file extension.

Supported extensions: .fits, .fit, .fts, .xisf
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default focal length for William Optics Redcat 51
DEFAULT_FOCAL_LENGTH_MM = 250.0

# Known narrowband filter name patterns -> canonical name
_NARROWBAND_FILTERS = {
    "ha": "Ha",
    "h-alpha": "Ha",
    "h_alpha": "Ha",
    "halpha": "Ha",
    "h": "Ha",
    "hydrogen-alpha": "Ha",
    "oiii": "OIII",
    "o3": "OIII",
    "o-iii": "OIII",
    "o": "OIII",
    "oxygen": "OIII",
    "sii": "SII",
    "s2": "SII",
    "s-ii": "SII",
    "s": "SII",
    "sulfur": "SII",
    "narrowband": "narrowband",
    "nb": "narrowband",
}

_BROADBAND_FILTERS = {
    "r": "R",
    "red": "R",
    "g": "G",
    "green": "G",
    "b": "B",
    "blue": "B",
    "l": "L",
    "lum": "L",
    "luminance": "L",
    "rgb": "RGB",
    "clear": "Clear",
    "ha": "Ha",  # also narrowband, but listed for completeness
}


@dataclass
class FITSData:
    """Container for FITS image data and metadata."""

    filepath: str
    filename: str
    data: np.ndarray  # 2D float64 image array

    # Acquisition metadata
    filter_name: Optional[str] = None
    exptime: Optional[float] = None       # seconds
    gain: Optional[float] = None
    ccd_temp: Optional[float] = None
    instrument: Optional[str] = None
    image_type: Optional[str] = None      # e.g. 'Light Frame', 'Master Light'

    # Pixel / optics
    pixel_size_um: Optional[float] = None   # micrometers
    focal_length_mm: float = DEFAULT_FOCAL_LENGTH_MM
    pixel_scale_arcsec: Optional[float] = None  # arcsec/pixel

    # Calibration status
    is_calibrated: bool = False

    # Observation time (ISO-8601 string from DATE-OBS header)
    obs_time: Optional[str] = None

    # Image mode derived from filter
    mode: Optional[str] = None  # 'star' or 'gas'

    # Raw header dict for downstream use
    header: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pixel_scale_arcsec is None and self.pixel_size_um is not None:
            self.pixel_scale_arcsec = (
                206.265 * self.pixel_size_um / self.focal_length_mm
            )
        if self.mode is None and self.filter_name is not None:
            self.mode = _detect_mode_from_filter(self.filter_name)


def _detect_mode_from_filter(filter_name: str) -> str:
    """Return 'gas' for narrowband filters, 'star' for broadband."""
    key = filter_name.lower().strip()
    if key in _NARROWBAND_FILTERS:
        return "gas"
    return "star"


def _parse_header(header) -> dict:
    """Extract relevant keywords from an astropy FITS header into a plain dict."""
    keys = [
        "FILTER", "EXPTIME", "EXPOSURE", "GAIN", "EGAIN",
        "CCD-TEMP", "CCDTEMP", "SET-TEMP",
        "INSTRUME", "IMAGETYP",
        "XPIXSZ", "PIXSIZE1", "PIXSCALE",
        "FOCALLEN", "FOCAL",
        "CALSTAT", "HISTORY",
        "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3",
        "BSCALE", "BZERO",
        "OBJECT", "DATE-OBS",
    ]
    result = {}
    for k in keys:
        try:
            val = header.get(k)
            if val is not None:
                result[k] = val
        except Exception:
            pass
    return result


def _extract_2d(data: np.ndarray) -> np.ndarray:
    """
    Reduce a potentially 3D FITS array to a 2D grayscale image.

    Handles:
    - (H, W)        -> as-is
    - (1, H, W)     -> squeeze first axis
    - (3, H, W)     -> luminance from RGB (weighted average)
    - (N, H, W)     -> first plane
    """
    if data.ndim == 2:
        return data.astype(np.float64)
    if data.ndim == 3:
        nplanes = data.shape[0]
        if nplanes == 1:
            return data[0].astype(np.float64)
        if nplanes == 3:
            # Weighted luminance: 0.2126 R + 0.7152 G + 0.0722 B
            return (
                0.2126 * data[0].astype(np.float64)
                + 0.7152 * data[1].astype(np.float64)
                + 0.0722 * data[2].astype(np.float64)
            )
        # Generic multi-frame: take first plane
        logger.debug("Multi-plane FITS (%d planes), using first plane.", nplanes)
        return data[0].astype(np.float64)
    raise ValueError(f"Unexpected FITS data dimensions: {data.shape}")


def load_fits(
    filepath: str | Path,
    focal_length_mm: float = DEFAULT_FOCAL_LENGTH_MM,
    pixel_size_um: Optional[float] = None,
) -> FITSData:
    """
    Load a FITS file and return a FITSData object.

    Parameters
    ----------
    filepath:
        Path to the FITS file (.fits, .fit, or .fts).
    focal_length_mm:
        Telescope focal length in mm (used for pixel scale calculation).

    Returns
    -------
    FITSData instance with image data and parsed metadata.

    Raises
    ------
    OSError / ValueError on file-level problems (caller should catch).
    """
    from astropy.io import fits as astropy_fits

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"FITS file not found: {filepath}")

    logger.debug("Loading FITS file: %s", filepath)

    with astropy_fits.open(str(filepath), memmap=False) as hdul:
        # Find the primary image extension
        hdu = None
        header = None

        # Try primary HDU first; if it has no data fall through to extensions
        if hdul[0].data is not None and hdul[0].data.ndim >= 2:
            hdu = hdul[0]
            header = hdul[0].header
        else:
            # Search image extensions
            for ext in hdul[1:]:
                if hasattr(ext, "data") and ext.data is not None:
                    if hasattr(ext.data, "ndim") and ext.data.ndim >= 2:
                        hdu = ext
                        # Merge primary header with extension header
                        merged = hdul[0].header.copy()
                        merged.update(ext.header)
                        header = merged
                        break

        if hdu is None:
            raise ValueError(f"No image data found in FITS file: {filepath}")

        raw_data = hdu.data
        header_dict = _parse_header(header)

        # Convert to 2D float64
        image_data = _extract_2d(raw_data)

        # Apply BSCALE / BZERO if present (astropy usually handles this, but be safe)
        bscale = float(header_dict.get("BSCALE", 1.0))
        bzero = float(header_dict.get("BZERO", 0.0))
        if bscale != 1.0 or bzero != 0.0:
            image_data = image_data * bscale + bzero

        # --- Parse metadata ---
        filter_name: Optional[str] = None
        raw_filter = header_dict.get("FILTER")
        if raw_filter is not None:
            filter_name = str(raw_filter).strip()

        exptime: Optional[float] = None
        for key in ("EXPTIME", "EXPOSURE"):
            if key in header_dict:
                try:
                    exptime = float(header_dict[key])
                    break
                except (ValueError, TypeError):
                    pass

        gain: Optional[float] = None
        for key in ("GAIN", "EGAIN"):
            if key in header_dict:
                try:
                    gain = float(header_dict[key])
                    break
                except (ValueError, TypeError):
                    pass

        ccd_temp: Optional[float] = None
        for key in ("CCD-TEMP", "CCDTEMP", "SET-TEMP"):
            if key in header_dict:
                try:
                    ccd_temp = float(header_dict[key])
                    break
                except (ValueError, TypeError):
                    pass

        instrument = header_dict.get("INSTRUME")
        if instrument:
            instrument = str(instrument).strip()

        image_type = header_dict.get("IMAGETYP")
        if image_type:
            image_type = str(image_type).strip()

        # Pixel size: XPIXSZ (arcsec or um depending on software) vs PIXSIZE1 (um)
        header_pixel_size_um: Optional[float] = None
        for key in ("XPIXSZ", "PIXSIZE1"):
            if key in header_dict:
                try:
                    val = float(header_dict[key])
                    if val > 0:
                        header_pixel_size_um = val
                        break
                except (ValueError, TypeError):
                    pass

        # Some software stores pixel size in different keys with different units;
        # XPIXSZ is typically in micrometers (e.g. 2.9 or 3.76)
        # Sanity check: pixel sizes should be between 0.5 and 30 um
        if header_pixel_size_um is not None and not (0.5 <= header_pixel_size_um <= 30.0):
            logger.warning(
                "Suspicious pixel size %s um from header for %s; ignoring.",
                header_pixel_size_um,
                filepath.name,
            )
            header_pixel_size_um = None

        # Use header value if present, else fall back to caller-supplied value
        effective_pixel_size_um = header_pixel_size_um if header_pixel_size_um is not None else pixel_size_um

        # Focal length from header (override CLI default if present)
        fl_from_header: Optional[float] = None
        for key in ("FOCALLEN", "FOCAL"):
            if key in header_dict:
                try:
                    val = float(header_dict[key])
                    if val > 0:
                        fl_from_header = val
                        break
                except (ValueError, TypeError):
                    pass
        effective_fl = fl_from_header if fl_from_header is not None else focal_length_mm

        # Pixel scale
        pixel_scale: Optional[float] = None
        if effective_pixel_size_um is not None:
            pixel_scale = 206.265 * effective_pixel_size_um / effective_fl
        elif "PIXSCALE" in header_dict:
            try:
                pixel_scale = float(header_dict["PIXSCALE"])
            except (ValueError, TypeError):
                pass

        # Calibration detection
        is_calibrated = False
        calstat = header_dict.get("CALSTAT")
        if calstat is not None:
            # CALSTAT like "BDF" means Bias/Dark/Flat calibrated
            cal_str = str(calstat).upper()
            is_calibrated = any(c in cal_str for c in ("B", "D", "F"))
        elif image_type is not None:
            itype = image_type.lower()
            is_calibrated = "master" in itype or "calibrated" in itype

        # Mode detection
        mode = None
        if filter_name:
            mode = _detect_mode_from_filter(filter_name)

        obs_time: Optional[str] = None
        raw_obs = header_dict.get("DATE-OBS")
        if raw_obs is not None:
            obs_time = str(raw_obs).strip()

        fits_data = FITSData(
            filepath=str(filepath),
            filename=filepath.name,
            data=image_data,
            filter_name=filter_name,
            exptime=exptime,
            gain=gain,
            ccd_temp=ccd_temp,
            instrument=instrument,
            image_type=image_type,
            pixel_size_um=effective_pixel_size_um,
            focal_length_mm=effective_fl,
            pixel_scale_arcsec=pixel_scale,
            is_calibrated=is_calibrated,
            obs_time=obs_time,
            mode=mode,
            header=header_dict,
        )

        logger.debug(
            "Loaded %s: shape=%s, filter=%s, pixel_scale=%.3f arcsec/px, mode=%s",
            filepath.name,
            image_data.shape,
            filter_name,
            pixel_scale if pixel_scale is not None else float("nan"),
            mode,
        )

        return fits_data


def _extract_2d_xisf(data: np.ndarray) -> np.ndarray:
    """
    Reduce an XISF image array to 2D grayscale.

    XISF color images use (H, W, C) axis order — opposite to FITS (C, H, W).
    Handles:
    - (H, W)        -> as-is
    - (H, W, 1)     -> squeeze last axis
    - (H, W, 3)     -> weighted luminance
    - (H, W, N)     -> first channel
    """
    if data.ndim == 2:
        return data.astype(np.float64)
    if data.ndim == 3:
        nch = data.shape[2]
        if nch == 1:
            return data[:, :, 0].astype(np.float64)
        if nch == 3:
            return (
                0.2126 * data[:, :, 0].astype(np.float64)
                + 0.7152 * data[:, :, 1].astype(np.float64)
                + 0.0722 * data[:, :, 2].astype(np.float64)
            )
        logger.debug("Multi-channel XISF (%d channels), using first channel.", nch)
        return data[:, :, 0].astype(np.float64)
    raise ValueError(f"Unexpected XISF data dimensions: {data.shape}")


def _xisf_get(meta: dict, *keys, cast=str, default=None):
    """
    Safely retrieve a value from XISF metadata, trying multiple key names.

    Checks both 'FITSKeywords' (embedded FITS) and 'XISFProperties' (native).
    """
    fits_kw  = meta.get("FITSKeywords",  {})
    xisf_props = meta.get("XISFProperties", {})

    for key in keys:
        # FITS keywords: {'FILTER': {'value': 'Ha', 'comment': '...'}}
        if key in fits_kw:
            try:
                return cast(fits_kw[key]["value"])
            except (KeyError, TypeError, ValueError):
                pass
        # XISF properties: {'Instrument:Filter:Name': {'value': 'Ha', ...}}
        if key in xisf_props:
            try:
                return cast(xisf_props[key]["value"])
            except (KeyError, TypeError, ValueError):
                pass

    return default


def load_xisf(
    filepath: str | Path,
    focal_length_mm: float = DEFAULT_FOCAL_LENGTH_MM,
    pixel_size_um: Optional[float] = None,
) -> FITSData:
    """
    Load an XISF file and return a FITSData object.

    Metadata is read from embedded FITS keywords first, then from native
    XISF properties as fallback.

    Parameters
    ----------
    filepath:
        Path to the .xisf file.
    focal_length_mm:
        Telescope focal length in mm (fallback if not in file metadata).

    Returns
    -------
    FITSData instance compatible with the rest of the pipeline.

    Raises
    ------
    ImportError  if the `xisf` package is not installed.
    FileNotFoundError / ValueError on file-level problems.
    """
    try:
        from xisf import XISF
    except ImportError as exc:
        raise ImportError(
            "The 'xisf' package is required to read XISF files. "
            "Install it with: pip install xisf"
        ) from exc

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"XISF file not found: {filepath}")

    logger.debug("Loading XISF file: %s", filepath)

    xf = XISF(str(filepath))
    images_meta = xf.get_images_metadata()

    if not images_meta:
        raise ValueError(f"No images found in XISF file: {filepath}")

    meta = images_meta[0]
    raw_data = xf.read_image(0)

    # Convert to 2D float64 (XISF color is H×W×C, not C×H×W like FITS)
    image_data = _extract_2d_xisf(raw_data)

    # --- Metadata: try FITS keywords first, then XISF native properties ---

    filter_name: Optional[str] = _xisf_get(
        meta,
        "FILTER",                    # FITS keyword
        "Instrument:Filter:Name",    # XISF property
        cast=str,
    )
    if filter_name:
        filter_name = filter_name.strip()

    exptime: Optional[float] = _xisf_get(
        meta,
        "EXPTIME", "EXPOSURE",           # FITS keywords
        "Instrument:ExposureTime",        # XISF property (seconds)
        cast=float,
    )

    gain: Optional[float] = _xisf_get(
        meta,
        "GAIN", "EGAIN",                 # FITS keywords
        "Instrument:Sensor:Gain",        # XISF property
        cast=float,
    )

    ccd_temp: Optional[float] = _xisf_get(
        meta,
        "CCD-TEMP", "CCDTEMP", "SET-TEMP",   # FITS keywords
        "Instrument:Sensor:Temperature",       # XISF property (°C)
        cast=float,
    )

    instrument: Optional[str] = _xisf_get(
        meta,
        "INSTRUME",
        "Instrument:Camera:Name",
        cast=str,
    )
    if instrument:
        instrument = instrument.strip()

    image_type: Optional[str] = _xisf_get(
        meta,
        "IMAGETYP",
        "Observation:Object:Name",   # not quite the same, but better than nothing
        cast=str,
    )

    # Pixel size (µm)
    header_pixel_size_um: Optional[float] = _xisf_get(
        meta,
        "XPIXSZ", "PIXSIZE1",            # FITS keywords (µm)
        "Instrument:Sensor:XPixelSize",   # XISF property (µm)
        cast=float,
    )
    if header_pixel_size_um is not None and not (0.5 <= header_pixel_size_um <= 30.0):
        logger.warning(
            "Suspicious pixel size %.2f µm in %s; ignoring.", header_pixel_size_um, filepath.name
        )
        header_pixel_size_um = None

    # Use header value if present, else fall back to caller-supplied value
    effective_pixel_size_um = header_pixel_size_um if header_pixel_size_um is not None else pixel_size_um

    # Focal length — XISF native property stores it in METERS; FITS keyword in mm
    fl_from_header: Optional[float] = _xisf_get(
        meta,
        "FOCALLEN", "FOCAL",              # FITS keywords (mm)
        cast=float,
    )
    if fl_from_header is None:
        # XISF property: Instrument:Telescope:FocalLength in meters
        fl_m = _xisf_get(meta, "Instrument:Telescope:FocalLength", cast=float)
        if fl_m is not None and fl_m > 0:
            fl_from_header = fl_m * 1000.0   # convert m → mm

    effective_fl = fl_from_header if fl_from_header is not None else focal_length_mm

    # Pixel scale
    pixel_scale: Optional[float] = None
    if effective_pixel_size_um is not None:
        pixel_scale = 206.265 * effective_pixel_size_um / effective_fl
    else:
        pixel_scale = _xisf_get(meta, "PIXSCALE", cast=float)

    # Calibration detection
    calstat = _xisf_get(meta, "CALSTAT", cast=str)
    is_calibrated = False
    if calstat:
        is_calibrated = any(c in calstat.upper() for c in ("B", "D", "F"))
    elif image_type:
        itype = image_type.lower()
        is_calibrated = "master" in itype or "calibrated" in itype

    mode = _detect_mode_from_filter(filter_name) if filter_name else None

    # Store raw metadata as a flat dict for downstream inspection
    header_dict: dict = {}
    for section in ("FITSKeywords", "XISFProperties"):
        for k, v in meta.get(section, {}).items():
            try:
                header_dict[k] = v.get("value", v)
            except AttributeError:
                header_dict[k] = v

    obs_time: Optional[str] = _xisf_get(
        meta,
        "DATE-OBS",
        "Observation:Time:Start",
        cast=str,
    )
    if obs_time:
        obs_time = obs_time.strip()

    fits_data = FITSData(
        filepath=str(filepath),
        filename=filepath.name,
        data=image_data,
        filter_name=filter_name,
        exptime=exptime,
        gain=gain,
        ccd_temp=ccd_temp,
        instrument=instrument,
        image_type=image_type,
        pixel_size_um=effective_pixel_size_um,
        focal_length_mm=effective_fl,
        pixel_scale_arcsec=pixel_scale,
        is_calibrated=is_calibrated,
        obs_time=obs_time,
        mode=mode,
        header=header_dict,
    )

    logger.debug(
        "Loaded %s: shape=%s filter=%s pixel_scale=%s arcsec/px mode=%s",
        filepath.name, image_data.shape, filter_name,
        f"{pixel_scale:.3f}" if pixel_scale is not None else "unknown",
        mode,
    )

    return fits_data


def load_image(
    filepath: str | Path,
    focal_length_mm: float = DEFAULT_FOCAL_LENGTH_MM,
    pixel_size_um: Optional[float] = None,
) -> FITSData:
    """
    Load a FITS or XISF image file, dispatching by extension.

    Supported extensions: .fits, .fit, .fts, .xisf
    """
    filepath = Path(filepath)
    if filepath.suffix.lower() == ".xisf":
        return load_xisf(filepath, focal_length_mm=focal_length_mm, pixel_size_um=pixel_size_um)
    return load_fits(filepath, focal_length_mm=focal_length_mm, pixel_size_um=pixel_size_um)


def find_fits_files(directory: str | Path) -> list[Path]:
    """Return sorted list of FITS and XISF files in directory (non-recursive)."""
    directory = Path(directory)
    extensions = {".fits", ".fit", ".fts", ".xisf"}
    files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )
    logger.debug("Found %d FITS files in %s", len(files), directory)
    return files
