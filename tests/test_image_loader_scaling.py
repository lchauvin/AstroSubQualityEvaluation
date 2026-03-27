from pathlib import Path

import numpy as np
import pytest
astropy_fits = pytest.importorskip("astropy.io.fits", reason="astropy is required for FITS loader tests")

from astro_eval.image_loader import load_fits


def test_load_fits_matches_astropy_scaled_data(tmp_path: Path):
    """
    Regression test: load_fits should return exactly the same data scaling that
    astropy already applies for BSCALE/BZERO-bearing FITS files.
    """
    src = np.array([[1, 2], [3, 4]], dtype=np.int16)
    hdu = astropy_fits.PrimaryHDU(data=src)
    hdu.header["BSCALE"] = 2.0
    hdu.header["BZERO"] = 10.0
    fpath = tmp_path / "scaled.fits"
    hdu.writeto(fpath)

    with astropy_fits.open(fpath, memmap=False) as hdul:
        astropy_data = hdul[0].data.astype(np.float64)

    loaded = load_fits(fpath)
    assert loaded.data.dtype == np.float64
    assert np.allclose(loaded.data, astropy_data)
