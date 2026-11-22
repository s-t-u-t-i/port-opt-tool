"""
Aspect 3 — Data pipeline edge cases (fix guards for the highest-risk area).

yfinance is mocked (monkeypatched) everywhere here, so no network is used.
The mocked behaviors mirror what the real yfinance does:
  - blocked / failed download  -> returns an EMPTY DataFrame (does NOT raise)

All of these tests used to document silent-failure bugs; after the fix pass
they guard the FIXED behavior instead.
"""
import numpy as np
import pandas as pd
import pytest

import core.data_utils as data_utils
from helpers import SYNTHETIC_TICKERS, make_prices, write_yf_csv


@pytest.fixture
def mock_yf_download(monkeypatch):
    """Replace yfinance's download with a controllable stand-in."""
    def install(return_value=None, raise_exc=None):
        def fake_download(tickers, **kwargs):
            if raise_exc is not None:
                raise raise_exc
            return return_value

        monkeypatch.setattr("core.data_utils.yf.download", fake_download)

    return install


# ---------------------------------------------------------------------------
# Empty / failed downloads must refuse to destroy good local data (was H2)
# ---------------------------------------------------------------------------
def test_empty_download_raises_and_preserves_good_csv(isolated_fs, mock_yf_download):
    """Fixed (was H2): a failed (empty) download raises a clear error and
    leaves the previously-good CSV untouched instead of clobbering it."""
    good = make_prices(n_days=100, seed=5, tickers=SYNTHETIC_TICKERS)
    write_yf_csv(isolated_fs / "data" / "stock_data.csv", good)
    good_size = (isolated_fs / "data" / "stock_data.csv").stat().st_size
    assert good_size > 10_000

    mock_yf_download(return_value=pd.DataFrame())
    with pytest.raises(ValueError, match="no data"):
        data_utils.fetch_and_save_data(SYNTHETIC_TICKERS, "2022-01-01", "2022-12-31")

    assert (isolated_fs / "data" / "stock_data.csv").stat().st_size == good_size


def test_zero_row_download_raises(isolated_fs, mock_yf_download):
    """Fixed (was H3): a zero-row download (e.g. start >= end) raises instead
    of flowing through the pipeline as silent garbage."""
    cols = pd.MultiIndex.from_product(
        [["Close", "Open"], SYNTHETIC_TICKERS], names=["Price", "Ticker"]
    )
    zero_rows = pd.DataFrame(columns=cols)
    zero_rows.index.name = "Date"
    mock_yf_download(return_value=zero_rows)

    with pytest.raises(ValueError, match="no data"):
        data_utils.fetch_and_save_data(SYNTHETIC_TICKERS, "2023-01-01", "2022-01-01")


# ---------------------------------------------------------------------------
# Bogus / delisted tickers can no longer wipe the panel (was C1)
# ---------------------------------------------------------------------------
def _prices_with_bogus(n_days=100, bogus_from_row=None):
    prices = make_prices(n_days=n_days, seed=7, tickers=["A", "B", "BOGUS"])
    if bogus_from_row is not None:
        prices.iloc[bogus_from_row:, prices.columns.get_loc("BOGUS")] = np.nan
    return prices


def test_bogus_ticker_is_excluded_and_analysis_survives(isolated_fs):
    """Fixed (was C1): a ticker with no data at all is dropped with a warning;
    the remaining tickers keep their FULL history and the covariance is sane."""
    with pytest.warns(UserWarning, match="No price data"):
        lr = data_utils.calculate_log_returns(_prices_with_bogus(bogus_from_row=0))
    assert list(lr.columns) == ["A", "B"]       # bogus excluded
    assert lr.shape == (99, 2)                  # full history kept, not wiped
    assert not lr.isna().any().any()

    cov, corr = data_utils.compute_cov_corr_matrices(lr)
    assert not cov.isna().any().any()           # sane matrix, not NaN garbage


def test_bogus_ticker_full_chain_now_succeeds(isolated_fs):
    """Fixed (was C1): the full chain with a bogus ticker no longer ends in a
    ValueError — the dead ticker is excluded and the MC optimum is computed
    from the surviving tickers."""
    from core.mc_optimization import get_mc_optimal_portfolio
    from core.simulation import run_monte_carlo_simulation

    prices = _prices_with_bogus(bogus_from_row=0)
    write_yf_csv(isolated_fs / "data" / "stock_data.csv", prices)
    close = data_utils.load_close_prices_from_csv("data/stock_data.csv")

    with pytest.warns(UserWarning, match="No price data"):
        lr = data_utils.calculate_log_returns(close)
    assert list(lr.columns) == ["A", "B"]

    np.random.seed(1)
    portfolios = run_monte_carlo_simulation(lr, num_portfolios=100, risk_free_rate=0.05)
    assert portfolios["Sharpe Ratio"].notna().all()
    ret, vol, sharpe, weights = get_mc_optimal_portfolio(portfolios, weight_columns=list(lr.columns))
    assert np.isfinite([ret, vol, sharpe]).all()
    assert len(weights) == 2


def test_half_delisted_ticker_truncates_with_warning(isolated_fs):
    """Fixed (was C1-variant): a ticker that dies halfway still forces the
    aligned-overlap policy, but the discarded history is now announced with a
    warning instead of vanishing silently."""
    with pytest.warns(UserWarning, match="discarded"):
        lr = data_utils.calculate_log_returns(_prices_with_bogus(bogus_from_row=60))
    assert len(lr) == 59  # of 99 possible — same policy, now not silent


# ---------------------------------------------------------------------------
# Stale CSV on disk (H4 is fixed at the app layer; the loader itself stays a
# plain disk reader, which is fine and now only used for offline/fixture use)
# ---------------------------------------------------------------------------
def test_loader_reads_whatever_is_on_disk(isolated_fs):
    """load_close_prices_from_csv reads the disk file as-is. The app no longer
    uses it mid-session (it works from the cached download instead), so a
    stale file can no longer be silently served for the wrong tickers."""
    other_session = make_prices(n_days=50, seed=9, tickers=["ZZZ.NS", "YYY.NS"])
    write_yf_csv(isolated_fs / "data" / "stock_data.csv", other_session)
    close = data_utils.load_close_prices_from_csv("data/stock_data.csv")
    assert list(close.columns) == ["ZZZ.NS", "YYY.NS"]


# ---------------------------------------------------------------------------
# fetch_and_save_data path handling (was M3)
# ---------------------------------------------------------------------------
def test_bare_filename_now_works(isolated_fs, mock_yf_download):
    """Fixed (was M3): a bare filename (no directory part) no longer crashes
    on os.makedirs('') — the CSV is written next to the current directory."""
    mock_yf_download(return_value=make_prices(n_days=50, seed=3, tickers=SYNTHETIC_TICKERS))
    data_utils.fetch_and_save_data(
        SYNTHETIC_TICKERS, "2022-01-01", "2022-12-31", filename="stock_data.csv"
    )
    assert (isolated_fs / "stock_data.csv").exists()
