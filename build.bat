@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   astro-eval Windows Build Script
echo ============================================
echo.

:: Check uv is available
where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 'uv' not found. Make sure uv is installed and on PATH.
    echo         https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)

:: Install pyinstaller into the uv project environment if needed
echo [0/3] Ensuring pyinstaller is available in the uv environment...
uv run python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo       Installing pyinstaller...
    uv add --dev pyinstaller
    if errorlevel 1 (
        echo [ERROR] Failed to install pyinstaller.
        pause
        exit /b 1
    )
)

:: Clean previous build artifacts
echo [1/3] Cleaning previous build...
if exist "build" rmdir /s /q "build"
if exist "dist\astro-eval" rmdir /s /q "dist\astro-eval"

:: Run PyInstaller via uv so it sees all project dependencies
echo [2/3] Building exe with PyInstaller...
uv run pyinstaller astro-eval.spec --noconfirm
if errorlevel 1 (
    echo [ERROR] PyInstaller build failed.
    pause
    exit /b 1
)

:: Verify output
if not exist "dist\astro-eval\astro-eval.exe" (
    echo [ERROR] Expected output dist\astro-eval\astro-eval.exe not found.
    pause
    exit /b 1
)

echo.
echo [3/3] Build complete!
echo       Executable: dist\astro-eval\astro-eval.exe
echo.

:: Optionally build the installer if Inno Setup is available
set ISCC=
where iscc >nul 2>&1 && set ISCC=iscc
if not defined ISCC if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" set ISCC="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "C:\Program Files\Inno Setup 6\ISCC.exe"       set ISCC="C:\Program Files\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set ISCC="%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"

if defined ISCC (
    echo [Installer] Inno Setup found. Building installer...
    %ISCC% installer.iss
    if errorlevel 1 (
        echo [WARNING] Inno Setup compilation failed. Exe is still available.
    ) else (
        echo [Installer] Installer built: Output\astro-eval-setup.exe
    )
) else (
    echo [Installer] Inno Setup not found — skipping installer build.
    echo             Download from: https://jrsoftware.org/isinfo.php
    echo             Then re-run this script to also build the installer.
)

echo.
echo Done. Press any key to exit.
pause >nul
