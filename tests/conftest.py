"""
Shared test toolbox (conftest.py).

Pytest automatically discovers this file and offers every fixture in it to
all test files. All data is synthetic (fixed seed, no network).
"""
import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make 'core' importable no matter where pytest is launched from.
from helpers import (  # noqa: F401  (helpers sits next to conftest in tests/)
    SYNTHETIC_TICKERS,
    TRADING_DAYS,
    make_prices,
    project_root,
    write_yf_csv,
)

PROJECT_ROOT = project_root()
sys.path.insert(0, str(PROJECT_ROOT))

# Headless matplotlib (no windows pop up during tests) + quiet bare-mode
# Streamlit "missing ScriptRunContext" warnings.
os.environ.setdefault("MPLBACKEND", "Agg")
logging.getLogger("streamlit").setLevel(logging.ERROR)


# --------------------------------------------------------------------------
# Scratch filesystem: tests that trigger hardcoded 'data/...' or 'output/...'
# writes run inside a temp folder so they never touch the repo's real files.
# --------------------------------------------------------------------------
@pytest.fixture
def isolated_fs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Mirror the repo's layout: several functions hardcode 'data/...' paths.
    (tmp_path / "data").mkdir()
    return tmp_path


# --------------------------------------------------------------------------
# Core synthetic market data (session-scoped: built once, reused everywhere)
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def five_asset_prices():
    return make_prices(n_days=500, seed=42, tickers=SYNTHETIC_TICKERS)


@pytest.fixture(scope="session")
def five_asset_log_returns(five_asset_prices):
    prices = five_asset_prices
    return np.log(prices / prices.shift(1)).dropna()


@pytest.fixture(scope="session")
def mc_portfolios(five_asset_log_returns):
    from core.simulation import run_monte_carlo_simulation

    np.random.seed(123)
    return run_monte_carlo_simulation(
        five_asset_log_returns, num_portfolios=2000, risk_free_rate=0.05
    )
