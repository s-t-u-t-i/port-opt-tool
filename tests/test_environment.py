"""
Aspect 9 — Configuration & environment.

Checks the project installs and imports cleanly, the Streamlit config parses,
and the README's run instructions reference files that actually exist.
"""
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from helpers import project_root

ROOT = project_root()

# requirements.txt name -> importable module name (only where they differ)
IMPORT_NAMES = {
    "beautifulsoup4": "bs4",
}


def test_all_core_modules_import_cleanly():
    """No syntax errors, no missing imports — a bare interpreter can load
    every module of the app (this does NOT run the app itself)."""
    code = (
        "import core.data_utils, core.simulation, core.true_optimization, "
        "core.mc_optimization, core.plotting; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr


def test_every_requirements_package_importable():
    """Every package listed in requirements.txt must be importable in the
    venv (a proxy for 'fresh install works' without needing the network)."""
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    pkgs = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            pkgs.append(line.split(">=")[0].split("<")[0].split("==")[0].strip())

    modules = ", ".join(IMPORT_NAMES.get(p, p) for p in pkgs)
    result = subprocess.run(
        [sys.executable, "-c", f"import {modules}; print('ok')"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert len(pkgs) == 13  # full dependency list is still covered


def test_pip_check_reports_no_dependency_conflicts():
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_streamlit_config_toml_is_valid():
    """Parses .streamlit/config.toml and documents the security-relevant
    settings (enableCORS=false / enableXsrfProtection=false are noted in the
    bug report — this test only records that the values are what they are)."""
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))

    theme = cfg["theme"]
    for key in ("primaryColor", "backgroundColor", "secondaryBackgroundColor", "textColor"):
        hexval = theme[key]
        assert hexval.startswith("#") and len(hexval) == 7
        int(hexval[1:], 16)  # valid hex

    server = cfg["server"]
    # Recorded for the report: both hardening flags are DISABLED.
    assert server["enableCORS"] is False
    assert server["enableXsrfProtection"] is False


def test_readme_run_instructions_reference_real_files():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "streamlit run streamlit_app.py" in readme
    assert (ROOT / "streamlit_app.py").exists()
    assert (ROOT / "requirements.txt").exists()
    # README documents the browser URL it expects the app on
    assert "http://localhost:8501" in readme


def test_pytest_listed_in_dev_requirements():
    dev = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "pytest" in dev
    assert "-r requirements.txt" in dev
