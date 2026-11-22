"""
Aspect 2 — Simulation + Monte Carlo optimum unit tests.
"""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from core.mc_optimization import get_mc_optimal_portfolio
from core.simulation import run_monte_carlo_simulation


# ---------------------------------------------------------------------------
# run_monte_carlo_simulation
# ---------------------------------------------------------------------------
def test_mc_reproducible_with_same_seed(five_asset_log_returns):
    np.random.seed(123)
    first = run_monte_carlo_simulation(five_asset_log_returns, num_portfolios=50, risk_free_rate=0.05)
    np.random.seed(123)
    second = run_monte_carlo_simulation(five_asset_log_returns, num_portfolios=50, risk_free_rate=0.05)
    assert_frame_equal(first, second)


def test_mc_different_seeds_differ(five_asset_log_returns):
    np.random.seed(1)
    a = run_monte_carlo_simulation(five_asset_log_returns, num_portfolios=50, risk_free_rate=0.05)
    np.random.seed(2)
    b = run_monte_carlo_simulation(five_asset_log_returns, num_portfolios=50, risk_free_rate=0.05)
    assert not np.allclose(a["Return"], b["Return"])


# ---------------------------------------------------------------------------
# get_mc_optimal_portfolio
# ---------------------------------------------------------------------------
def _handcrafted_df():
    return pd.DataFrame(
        {
            "Return": [0.10, 0.20, 0.30],
            "Volatility": [0.20, 0.15, 0.10],
            "Sharpe Ratio": [0.25, 1.00, 2.50],   # row 2 is the winner
            "AAA.NS": [0.5, 0.3, 0.2],
            "BBB.NS": [0.5, 0.7, 0.8],
        }
    )


def test_mc_optimal_picks_max_sharpe_row():
    df = _handcrafted_df()
    ret, vol, sharpe, weights = get_mc_optimal_portfolio(df)
    assert sharpe == 2.50
    assert ret == 0.30
    assert vol == 0.10
    assert list(weights) == [0.2, 0.8]


def test_mc_optimal_weight_slice_is_positional_not_named():
    """Documented fragility (mc_optimization.py:27): weights are taken from
    columns[3:] by POSITION, not by name. Any extra column sitting after
    position 3 leaks into the 'weights' vector."""
    df = pd.DataFrame(
        {
            "Return": [0.1, 0.2],
            "Volatility": [0.2, 0.15],
            "Sharpe Ratio": [0.25, 1.00],
            "INTRUDER": [42.0, 43.0],      # not a weight, sits at position 3
            "AAA.NS": [0.5, 0.3],
            "BBB.NS": [0.5, 0.7],
        }
    )
    _, _, sharpe, weights = get_mc_optimal_portfolio(df)
    assert sharpe == 1.00
    assert 43.0 in weights  # the intruder leaks in — documents the fragility


def test_mc_optimal_all_nan_sharpe_raises():
    df = pd.DataFrame(
        {
            "Return": [0.1],
            "Volatility": [0.2],
            "Sharpe Ratio": [np.nan],
            "AAA.NS": [1.0],
        }
    )
    with pytest.raises(ValueError):
        get_mc_optimal_portfolio(df)


def test_mc_optimal_empty_dataframe_raises():
    df = pd.DataFrame(columns=["Return", "Volatility", "Sharpe Ratio", "AAA.NS"])
    with pytest.raises(ValueError):
        get_mc_optimal_portfolio(df)
