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


def test_mc_optimal_weight_columns_resolved_by_name():
    """Fixed (was L3): weights are resolved BY NAME (everything that isn't a
    metric column), never by position — so reordering or inserting metric
    columns can't corrupt the weights. An unknown extra column still leaks
    under the default, so callers with a known ticker list should pass
    weight_columns explicitly (the app now does)."""
    df = pd.DataFrame(
        {
            "Return": [0.1, 0.2],
            "Volatility": [0.2, 0.15],
            "Sharpe Ratio": [0.25, 1.00],
            "INTRUDER": [42.0, 43.0],      # not a weight column
            "AAA.NS": [0.5, 0.3],
            "BBB.NS": [0.5, 0.7],
        }
    )
    # Explicit list -> clean, intruder excluded (what the app now does)
    _, _, _, weights = get_mc_optimal_portfolio(df, weight_columns=["AAA.NS", "BBB.NS"])
    assert list(weights) == [0.3, 0.7]

    # Default -> name-based exclusion of the three metric columns only
    _, _, sharpe, default_weights = get_mc_optimal_portfolio(df)
    assert sharpe == 1.00
    assert 43.0 in default_weights  # documented default limitation


def test_align_weight_vector_reorders_between_ticker_orders():
    """Fixed (was M1 support): weight vectors can be mapped between ticker
    orders so a re-selected (same-set) ticker list keeps the simulation valid
    while every weight stays attached to the right ticker."""
    from core.mc_optimization import align_weight_vector

    weights = [0.1, 0.2, 0.3, 0.4]
    source = ["A", "B", "C", "D"]
    target = ["D", "A", "C", "B"]
    assert align_weight_vector(weights, source, target) == [0.4, 0.1, 0.3, 0.2]
    # Same order -> unchanged
    assert align_weight_vector(weights, source, source) == weights
    # Different sets are rejected
    with pytest.raises(ValueError):
        align_weight_vector(weights, source, ["A", "B", "C", "E"])


def test_mc_optimal_all_nan_sharpe_raises_clear_error():
    """Fixed (was L6): all-NaN Sharpe raises with an actionable message."""
    df = pd.DataFrame(
        {
            "Return": [0.1],
            "Volatility": [0.2],
            "Sharpe Ratio": [np.nan],
            "AAA.NS": [1.0],
        }
    )
    with pytest.raises(ValueError, match="Sharpe ratio is NaN"):
        get_mc_optimal_portfolio(df, weight_columns=["AAA.NS"])


def test_mc_optimal_empty_dataframe_raises_clear_error():
    """Fixed (was L6): an empty simulation raises with an actionable message."""
    df = pd.DataFrame(columns=["Return", "Volatility", "Sharpe Ratio", "AAA.NS"])
    with pytest.raises(ValueError, match="empty"):
        get_mc_optimal_portfolio(df, weight_columns=["AAA.NS"])
