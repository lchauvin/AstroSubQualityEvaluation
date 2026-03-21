# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for astro-eval
# Build with: pyinstaller astro-eval.spec
#
# Requires: pip install pyinstaller
#

import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# Collect all submodules that use dynamic imports
hiddenimports = (
    collect_submodules("astropy")
    + collect_submodules("matplotlib")
    + collect_submodules("xisf")
    + collect_submodules("astro_eval")
    + [
        "sep",
        "paramiko",
        "scipy.special._ufuncs_cxx",
        "scipy.linalg.cython_blas",
        "scipy.linalg.cython_lapack",
        "scipy.integrate",
        "scipy.optimize",
        "scipy.ndimage",
        "matplotlib.backends.backend_agg",
        "matplotlib.backends.backend_svg",
        "pkg_resources.py2_compat",
    ]
)

# Collect data files (e.g. astropy IERS tables, matplotlib fonts/styles)
datas = (
    collect_data_files("astropy")
    + collect_data_files("matplotlib")
    + [("astro_eval.toml.example", ".")]
)

a = Analysis(
    ["astro_eval_main.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "PyQt5",
        "PyQt6",
        "wx",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="astro-eval",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,          # Keep console window — shows progress output
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,             # Add an .ico path here if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="astro-eval",
)
