"""
Aspect 3 — Data pipeline edge cases (the highest-risk area found).

yfinance is mocked (monkeypatched) everywhere here, so no network is used.
The mocked behaviors mirror what the real yfinance does:
  - blocked / failed download  -> returns an EMPTY DataFrame (does NOT raise)
  - bad date range             -> returns columns with ZERO rows
"""
import numpy as np
import pandas as pd
import pytest

import core.data_utils as data_utils
from helpers import SYNTHETIC_TICKERS, make_prices, write_yf_csv


@pytest.fixture
def mock_yf_download(monkeypatch):
    """Replace yfinance's download with a controllable stand-in."""
    holder = {}

    def install(return_value=None, raise_exc=None):
        def fake_download(tickers, **kwargs):
            if raise_exc is not None:
                raise raise_exc
            return return_value

        monkeypatch.setattr("core.data_utils.yf.download", fake_download)
        holder["called"] = 0
        return holder

    return install


# ---------------------------------------------------------------------------
# Empty ticker list (app sidebar allows selecting zero tickers)
# ---------------------------------------------------------------------------
def test_empty_ticker_list_produces_empty_csv_then_crash(isolated_fs, mock_yf_download):
    mock_yf_download(return_value=pd.DataFrame())
    data_utils.fetch_and_save_data([], "2022-01-01", "2022-12-31")
    # fetch itself does not raise; the crash happens downstream at load time.
    # (pandas 3: an empty written CSV reloads as ParserError; older pandas
    # raised EmptyDataError — both are the same 'no Close columns' crash.)
    with pytest.raises((pd.errors.ParserError, pd.errors.EmptyDataError)):
        data_utils.load_close_prices_from_csv("data/stock_data.csv")


# ---------------------------------------------------------------------------
# start_date >= end_date -> zero-row download
# ---------------------------------------------------------------------------
def test_zero_row_download_propagates_silently(isolated_fs, mock_yf_download):
    """Documented bug: a zero-row download loads as an EMPTY frame and flows
    through log returns / cov / corr with NO error anywhere."""
    cols = pd.MultiIndex.from_product(
        [["Close", "Open"], SYNTHETIC_TICKERS], names=["Price", "Ticker"]
    )
    zero_rows = pd.DataFrame(columns=cols)
    zero_rows.index.name = "Date"
    mock_yf_download(return_value=zero_rows)

    data_utils.fetch_and_save_data(SYNTHETIC_TICKERS, "2023-01-01", "2022-01-01")
    close = data_utils.load_close_prices_from_csv("data/stock_data.csv")
    assert len(close) == 0                                   # empty data...
    lr = data_utils.calculate_log_returns(close)             # ...no error raised
    assert len(lr) == 0
    cov, _ = data_utils.compute_cov_corr_matrices(lr)        # ...still no error
    assert cov.isna().all().all()                            # silent NaN garbage


# ---------------------------------------------------------------------------
# One bogus / delisted ticker among good ones
# ---------------------------------------------------------------------------
def _prices_with_bogus(n_days=100, bogus_from_row=0):
    prices = make_prices(n_days=n_days, seed=7, tickers=["A", "B", "BOGUS"])
    if bogus_from_row is not None:
        prices.iloc[bogus_from_row:, prices.columns.get_loc("BOGUS")] = np.nan
    return prices


def test_bogus_ticker_from_start_wipes_all_rows(isolated_fs):
    """Documented bug: dropna() is applied to the WHOLE frame, so one ticker
    that is NaN from day 0 wipes every row for every other ticker."""
    lr = data_utils.calculate_log_returns(_prices_with_bogus(bogus_from_row=0))
    assert lr.shape == (0, 3)  # everything gone, silently

    cov, corr = data_utils.compute_cov_corr_matrices(lr)
    assert cov.isna().all().all()  # NaN covariance poisons all downstream math


def test_bogus_ticker_full_chain_ends_in_valueerror(isolated_fs):
    """The full app chain with a bogus ticker: fetch -> load -> returns ->
    MC simulation -> 'optimal portfolio'. Every step is silent NaN garbage
    until get_mc_optimal_portfolio finally blows up."""
    from core.mc_optimization import get_mc_optimal_portfolio
    from core.simulation import run_monte_carlo_simulation

    prices = _prices_with_bogus(bogus_from_row=0)
    write_yf_csv(isolated_fs / "data" / "stock_data.csv", prices)
    close = data_utils.load_close_prices_from_csv("data/stock_data.csv")
    lr = data_utils.calculate_log_returns(close)
    assert len(lr) == 0

    np.random.seed(1)
    portfolios = run_monte_carlo_simulation(lr, num_portfolios=10, risk_free_rate=0.05)
    assert portfolios["Sharpe Ratio"].isna().all()  # garbage metrics, no warning
    with pytest.raises(ValueError):
        get_mc_optimal_portfolio(portfolios)  # the only loud failure, far downstream


def test_half_delisted_ticker_silently_truncates_history(isolated_fs):
    """Documented bug: a ticker that dies halfway through the date range
    silently throws away ALL days after its death for EVERY ticker
    (59 rows kept out of 99 in the probe) — no warning, no error."""
    lr = data_utils.calculate_log_returns(_prices_with_bogus(bogus_from_row=60))
    assert len(lr) == 59  # of 99 possible — 40 days silently discarded


# ---------------------------------------------------------------------------
# Stale / clobbered CSV on disk
# ---------------------------------------------------------------------------
def test_failed_download_overwrites_good_csv_with_empty_file(
    isolated_fs, mock_yf_download
):
    """Documented bug: fetch_and_save_data ALWAYS writes whatever the download
    returned, even an empty frame. One failed download destroys the previously
    good local data file (data_utils.py:30)."""
    good = make_prices(n_days=100, seed=5, tickers=SYNTHETIC_TICKERS)
    write_yf_csv(isolated_fs / "data" / "stock_data.csv", good)
    good_size = (isolated_fs / "data" / "stock_data.csv").stat().st_size
    assert good_size > 10_000

    # Real yfinance returns an empty frame when Yahoo blocks the request.
    mock_yf_download(return_value=pd.DataFrame())
    data_utils.fetch_and_save_data(SYNTHETIC_TICKERS, "2022-01-01", "2022-12-31")

    destroyed = (isolated_fs / "data" / "stock_data.csv").stat().st_size
    print(f"\n[bug] good CSV of {good_size} bytes replaced by {destroyed} bytes")
    assert destroyed < 100  # data destroyed


def test_cached_fetch_can_silently_read_another_sessions_csv(isolated_fs):
    """Documented race: the app caches downloads by (tickers, dates) but ALWAYS
    reads 'data/stock_data.csv' from disk (streamlit_app.py:138-139). If another
    process/session overwrote the file, the cache hit serves the WRONG data
    with no mismatch check."""
    other_session = make_prices(n_days=50, seed=9, tickers=["ZZZ.NS", "YYY.NS"])
    write_yf_csv(isolated_fs / "data" / "stock_data.csv", other_session)

    # Simulating an app-side cache hit: fetch is skipped, disk is read directly.
    close = data_utils.load_close_prices_from_csv("data/stock_data.csv")
    assert list(close.columns) == ["ZZZ.NS", "YYY.NS"]  # ≠ what the session requested


# ---------------------------------------------------------------------------
# fetch_and_save_data path handling
# ---------------------------------------------------------------------------
def test_bare_filename_crashes_on_empty_dirname(isolated_fs):
    """Documented bug: os.makedirs(os.path.dirname(filename)) with a bare
    filename gives os.makedirs('') -> FileNotFoundError (data_utils.py:21).
    Crashes BEFORE the download, so no network needed."""
    with pytest.raises(FileNotFoundError):
        data_utils.fetch_and_save_data(
            SYNTHETIC_TICKERS, "2022-01-01", "2022-12-31", filename="stock_data.csv"
        )
