"""
Shared synthetic-data factory for the test suite.

Every number produced here comes from a FIXED random seed, so tests are
deterministic (same data every run) and never need the network.

The "market recipe" below describes 5 imaginary stocks whose daily returns
follow a normal distribution with a known mean vector and covariance matrix.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Ticker-style names so fixture CSVs look like the real yfinance data.
SYNTHETIC_TICKERS = ["AAA.NS", "BBB.NS", "CCC.NS", "DDD.NS", "EEE.NS"]

# Known recipe: daily mean returns and daily volatilities per asset.
MEAN_DAILY = np.array([0.0006, 0.0004, 0.0005, 0.0003, 0.0007])
VOLS_DAILY = np.array([0.012, 0.010, 0.014, 0.008, 0.015])
CORRELATION = 0.30

PROJECT_ROOT_MARKER = True  # helpers.py lives in tests/, two levels up is the repo root


def project_root():
    """Absolute path of the repository root (parent of tests/)."""
    from pathlib import Path
    return Path(__file__).resolve().parents[1]


def tests_dir():
    from pathlib import Path
    return Path(__file__).resolve().parent


def known_daily_cov(n=5):
    """The daily covariance matrix implied by the recipe above (n assets)."""
    if n == 5:
        vols = VOLS_DAILY
    else:
        # Scale the recipe for a different number of assets.
        vols = np.linspace(VOLS_DAILY[0], VOLS_DAILY[-1], n)
    corr = np.full((n, n), CORRELATION)
    np.fill_diagonal(corr, 1.0)
    return corr * np.outer(vols, vols)


def known_mean_daily(n=5):
    if n == 5:
        return MEAN_DAILY
    return np.linspace(MEAN_DAILY[0], MEAN_DAILY[-1], n)


def make_prices(n_days=500, seed=42, start=100.0, tickers=None, n_assets=5,
                nan_from=None):
    """
    Build synthetic price levels by compounding log returns drawn from the
    known recipe.

    nan_from: optional dict {column_index: first_row_index} that turns the
    given columns into NaN from that row onward (simulates a delisted or
    bogus ticker).
    """
    rng = np.random.default_rng(seed)
    names = tickers if tickers is not None else [f"A{i}.NS" for i in range(n_assets)]
    n = len(names)  # asset count follows the ticker list given
    mean = known_mean_daily(n)
    cov = known_daily_cov(n)
    rets = rng.multivariate_normal(mean, cov, size=n_days)
    growth = np.exp(np.cumsum(rets, axis=0))
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    prices = pd.DataFrame(start * growth, index=idx, columns=names)
    if nan_from:
        for col, first_row in nan_from.items():
            prices.iloc[first_row:, prices.columns.get_loc(col)] = np.nan
    return prices


def write_yf_csv(path, prices, fields=("Close", "High", "Low", "Open", "Volume")):
    """
    Write a CSV that looks byte-for-byte like real yfinance output:
    two header rows (Price / Ticker) + a stray 'Date' row, then data.
    """
    frames = {}
    for field in fields:
        for t in prices.columns:
            if field == "Volume":
                frames[(field, t)] = (prices[t] * 1e5).round(0)
            elif field == "High":
                frames[(field, t)] = prices[t] * 1.005
            elif field == "Low":
                frames[(field, t)] = prices[t] * 0.995
            else:
                frames[(field, t)] = prices[t]
    df = pd.DataFrame(frames)
    df.index.name = "Date"
    df.columns = pd.MultiIndex.from_tuples(list(frames.keys()), names=["Price", "Ticker"])
    Path_ = __import__("pathlib").Path
    path = Path_(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return path
