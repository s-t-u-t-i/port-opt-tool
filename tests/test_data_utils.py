"""
Aspect 2 — Core data_utils unit tests (each Lego brick on its own).
"""
import numpy as np
import pandas as pd
import pytest

import core.data_utils as data_utils
from helpers import SYNTHETIC_TICKERS, write_yf_csv
from helpers import tests_dir as _fixture_dir

FIXTURES = _fixture_dir() / "data"


# ---------------------------------------------------------------------------
# calculate_log_returns
# ---------------------------------------------------------------------------
def test_log_returns_math_matches_manual(five_asset_prices, isolated_fs):
    manual = np.log(five_asset_prices / five_asset_prices.shift(1)).dropna()
    got = data_utils.calculate_log_returns(five_asset_prices)
    pd.testing.assert_frame_equal(got, manual)


def test_log_returns_drops_leading_nan_rows(five_asset_prices, isolated_fs):
    got = data_utils.calculate_log_returns(five_asset_prices)
    # first row has no previous day -> NaN -> dropped
    assert len(got) == len(five_asset_prices) - 1
    assert not got.isna().any().any()


def test_log_returns_zero_price_inf_survives_dropna(isolated_fs):
    """Documented bug: a zero price yields ±inf, and dropna() only removes
    NaN — so ±inf rows flow into downstream covariance math, which turns
    everything into NaN with no error anywhere (data_utils.py:66)."""
    idx = pd.bdate_range("2022-01-03", periods=50)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, (50, 2)), axis=0)),
        index=idx,
        columns=["A", "B"],
    )
    prices.iloc[5] = 0.0  # one zero close
    lr = data_utils.calculate_log_returns(prices)
    assert np.isinf(lr.to_numpy()).any()  # inf silently survives the broom


# ---------------------------------------------------------------------------
# load_close_prices_from_csv
# ---------------------------------------------------------------------------
def test_load_close_prices_roundtrip_with_fixture_csv():
    close = data_utils.load_close_prices_from_csv(FIXTURES / "valid_stock_data.csv")
    assert list(close.columns) == SYNTHETIC_TICKERS
    # 250 price rows; pandas consumes the stray 'Date' row as the index name
    assert len(close) == 250
    assert not close.isna().any().any()
    assert (close > 0).all().all()


def test_load_close_prices_keyerror_on_malformed_csv():
    with pytest.raises(KeyError):
        data_utils.load_close_prices_from_csv(FIXTURES / "malformed_no_close.csv")


def test_load_close_prices_one_ticker_fixture():
    close = data_utils.load_close_prices_from_csv(FIXTURES / "one_ticker_stock_data.csv")
    assert list(close.columns) == ["AAA.NS"]


# ---------------------------------------------------------------------------
# compute_cov_corr_matrices
# ---------------------------------------------------------------------------
def test_cov_corr_symmetry_diagonal_and_range(five_asset_log_returns, isolated_fs):
    cov, corr = data_utils.compute_cov_corr_matrices(
        five_asset_log_returns,
        cov_output="cov_test.csv",
        corr_output="corr_test.csv",
    )
    assert np.allclose(cov, cov.T)                      # symmetric
    assert np.allclose(np.diag(corr), 1.0)              # corr diagonal = 1
    assert ((corr >= -1.0) & (corr <= 1.0)).all().all() # values in [-1, 1]
    assert np.allclose(corr, corr.T)


# ---------------------------------------------------------------------------
# calculate_individual_sharpe_ratios
# ---------------------------------------------------------------------------
def test_individual_sharpe_hand_computed(five_asset_log_returns, isolated_fs):
    lr = five_asset_log_returns
    summary = data_utils.calculate_descriptive_statistics(lr)
    sharpe, ann_mean, ann_std = data_utils.calculate_individual_sharpe_ratios(
        summary, risk_free_rate=0.05
    )
    rf = 0.05
    for i, col in enumerate(lr.columns):
        expected = (
            float(lr[col].mean() * 252 - rf) / float(lr[col].std() * 252 ** 0.5)
        )
        assert float(sharpe.iloc[i]) == pytest.approx(expected, rel=1e-12)
        assert float(ann_mean.iloc[i]) == pytest.approx(
            float(lr[col].mean() * 252), rel=1e-12
        )
        assert float(ann_std.iloc[i]) == pytest.approx(
            float(lr[col].std() * 252 ** 0.5), rel=1e-12
        )


def test_individual_sharpe_zero_std_gives_inf(isolated_fs):
    """Documented bug: a flat price column (std == 0) produces ±inf Sharpe
    with no guard or error (data_utils.py:151)."""
    lr = pd.DataFrame(
        {"FLAT": [0.0001] * 100, "OK": np.random.default_rng(3).normal(0.0005, 0.01, 100)}
    )
    summary = data_utils.calculate_descriptive_statistics(lr)
    sharpe, _, _ = data_utils.calculate_individual_sharpe_ratios(summary, 0.05)
    assert np.isinf(sharpe["FLAT"])
