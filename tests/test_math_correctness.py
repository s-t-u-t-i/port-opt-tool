"""
Aspect 1 — Math & Finance correctness (the heart of the app).

Every expected value here is computed by hand with explicit arithmetic in the
test itself — never by calling the function under test.
"""
import numpy as np
import pandas as pd
import pytest

from core.mc_optimization import get_mc_optimal_portfolio
from core.simulation import run_monte_carlo_simulation
from core.true_optimization import (
    compute_all_true_portfolios,
    negative_sharpe_ratio,
    portfolio_annual_perf,
)
from helpers import SYNTHETIC_TICKERS

RISK_FREE = 0.05

# ---------------------------------------------------------------------------
# Hand-computed 2-asset example (exact arithmetic, no randomness):
#   weights = [0.6, 0.4], daily mean = [0.001, 0.0005]
#   daily cov = [[0.0004, 0.0001], [0.0001, 0.0009]]
# ---------------------------------------------------------------------------
WEIGHTS = np.array([0.6, 0.4])
MEAN_2 = pd.Series([0.001, 0.0005], index=["X", "Y"])
COV_2 = pd.DataFrame(
    [[0.0004, 0.0001], [0.0001, 0.0009]], index=["X", "Y"], columns=["X", "Y"]
)

EXPECTED_RET = (0.001 * 0.6 + 0.0005 * 0.4) * 252  # = 0.2016
EXPECTED_VAR = (
    0.36 * 0.0004 + 2 * 0.6 * 0.4 * 0.0001 + 0.16 * 0.0009
) * 252  # = 0.084672
EXPECTED_VOL = EXPECTED_VAR ** 0.5
EXPECTED_SHARPE = (EXPECTED_RET - RISK_FREE) / EXPECTED_VOL


def test_portfolio_annual_perf_hand_computed():
    ret, vol = portfolio_annual_perf(WEIGHTS, MEAN_2, COV_2)
    assert ret == pytest.approx(EXPECTED_RET, rel=1e-12)
    assert vol == pytest.approx(EXPECTED_VOL, rel=1e-12)


def test_negative_sharpe_matches_hand_computed():
    neg = negative_sharpe_ratio(WEIGHTS, MEAN_2, COV_2, RISK_FREE)
    assert neg == pytest.approx(-EXPECTED_SHARPE, rel=1e-12)


# ---------------------------------------------------------------------------
# Annualization consistency: the *same* recipe (mean*252 / cov*252 / std*√252)
# must come out identical from every hardcoded site in the codebase.
# ---------------------------------------------------------------------------
def test_annualization_consistent_across_all_sites(five_asset_log_returns, isolated_fs):
    lr = five_asset_log_returns
    mean_daily = lr.mean()
    cov_daily = lr.cov()
    w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    # Site 1: core/true_optimization.py::portfolio_annual_perf
    site1_ret, site1_vol = portfolio_annual_perf(w, mean_daily, cov_daily)

    # Site 2: explicit recipe mean*252 / cov*252
    site2_ret = float(np.dot(w, mean_daily * 252))
    site2_vol = float(np.sqrt(w @ (cov_daily * 252) @ w))

    # Site 3: core/simulation.py — read back one MC row using ITS OWN weights
    np.random.seed(7)
    mc = run_monte_carlo_simulation(lr, num_portfolios=1, risk_free_rate=RISK_FREE)
    w0 = mc.iloc[0][lr.columns].to_numpy(float)
    mc_ret = float(mc.iloc[0]["Return"])
    mc_vol = float(mc.iloc[0]["Volatility"])
    mc_recomputed_ret = float(np.dot(w0, mean_daily * 252))
    mc_recomputed_vol = float(np.sqrt(w0 @ (cov_daily * 252) @ w0))

    # Site 4: core/data_utils.py::calculate_individual_sharpe_ratios
    from core.data_utils import (
        calculate_descriptive_statistics,
        calculate_individual_sharpe_ratios,
    )

    summary = calculate_descriptive_statistics(lr)
    _, ann_mean, ann_std = calculate_individual_sharpe_ratios(summary, RISK_FREE)

    assert site1_ret == pytest.approx(site2_ret, rel=1e-12)
    assert site1_vol == pytest.approx(site2_vol, rel=1e-12)
    assert mc_ret == pytest.approx(mc_recomputed_ret, rel=1e-12)
    assert mc_vol == pytest.approx(mc_recomputed_vol, rel=1e-12)

    # Site 4 must use mean*252 and std*sqrt(252), and pandas .std() must agree
    # with sqrt(diag(cov)) so all sites are on the same page (ddof=1).
    assert float(ann_mean.iloc[0]) == pytest.approx(
        float(mean_daily.iloc[0] * 252), rel=1e-12
    )
    assert float(ann_std.iloc[0]) == pytest.approx(
        float(lr.iloc[:, 0].std() * np.sqrt(252)), rel=1e-12
    )
    assert float(np.sqrt(cov_daily.iloc[0, 0])) == pytest.approx(
        float(lr.iloc[:, 0].std()), rel=1e-12
    )


# ---------------------------------------------------------------------------
# Monte Carlo properties
# ---------------------------------------------------------------------------
def test_mc_every_row_weights_sum_to_one(mc_portfolios):
    weights = mc_portfolios[SYNTHETIC_TICKERS].to_numpy(float)
    sums = weights.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-12)


def test_mc_num_portfolios_honored(five_asset_log_returns):
    np.random.seed(1)
    df = run_monte_carlo_simulation(
        five_asset_log_returns, num_portfolios=137, risk_free_rate=RISK_FREE
    )
    assert len(df) == 137


def test_mc_sharpe_formula_per_row(mc_portfolios):
    recomputed = (mc_portfolios["Return"] - RISK_FREE) / mc_portfolios["Volatility"]
    assert np.allclose(recomputed, mc_portfolios["Sharpe Ratio"], rtol=1e-12)


def test_mc_output_shape_and_columns(mc_portfolios):
    # (num_portfolios, 3 summary metrics + k weight columns)
    assert mc_portfolios.shape == (2000, 3 + 5)
    assert list(mc_portfolios.columns) == [
        "Return",
        "Volatility",
        "Sharpe Ratio",
        *SYNTHETIC_TICKERS,
    ]


def test_mc_optimal_sharpe_never_beats_true_optimal(five_asset_log_returns):
    # SLSQP maximizes Sharpe analytically; random darts should never win.
    np.random.seed(0)
    mc = run_monte_carlo_simulation(
        five_asset_log_returns, num_portfolios=5000, risk_free_rate=RISK_FREE
    )
    _, _, mc_sharpe, _ = get_mc_optimal_portfolio(mc)
    out = compute_all_true_portfolios(
        five_asset_log_returns.mean(), five_asset_log_returns.cov(), RISK_FREE
    )
    assert mc_sharpe <= out["true"][2] + 1e-9


def test_gmv_has_lowest_volatility(five_asset_log_returns):
    np.random.seed(2)
    mc = run_monte_carlo_simulation(
        five_asset_log_returns, num_portfolios=3000, risk_free_rate=RISK_FREE
    )
    out = compute_all_true_portfolios(
        five_asset_log_returns.mean(), five_asset_log_returns.cov(), RISK_FREE
    )
    gmv_vol = out["min"][1]
    true_vol = out["true"][1]
    # GMV is the global volatility minimum: nothing (analytic or random)
    # should wobble less.
    assert gmv_vol <= true_vol + 1e-9
    assert gmv_vol < float(mc["Volatility"].min()) + 1e-9


def test_frontier_upper_branch_monotonic(five_asset_log_returns):
    out = compute_all_true_portfolios(
        five_asset_log_returns.mean(), five_asset_log_returns.cov(), RISK_FREE
    )
    frontier = out["frontier"]
    assert len(frontier) > 10

    # target_returns come out of linspace, so Return is already ascending.
    assert (np.diff(frontier["Return"].to_numpy()) >= -1e-12).all()

    # On the upper branch, volatility must never *decrease* as return rises.
    # (tiny negative tolerance for SLSQP numerical wiggle)
    assert (np.diff(frontier["Volatility"].to_numpy()) > -1e-4).all()


def test_custom_portfolio_metrics_match_manual_recomputation(five_asset_log_returns):
    """Mirrors the app's custom-portfolio math (streamlit_app.py:229-231)."""
    lr = five_asset_log_returns
    mean_daily = lr.mean()
    cov_daily = lr.cov()
    custom_weights = np.array([0.10, 0.15, 0.25, 0.30, 0.20])

    ret, vol = portfolio_annual_perf(custom_weights, mean_daily, cov_daily)
    sharpe = (ret - RISK_FREE) / vol

    manual_ret = float(np.sum(mean_daily.to_numpy() * custom_weights) * 252)
    manual_vol = float(
        np.sqrt(custom_weights @ (cov_daily.to_numpy() * 252) @ custom_weights)
    )
    manual_sharpe = (manual_ret - RISK_FREE) / manual_vol

    assert ret == pytest.approx(manual_ret, rel=1e-12)
    assert vol == pytest.approx(manual_vol, rel=1e-12)
    assert sharpe == pytest.approx(manual_sharpe, rel=1e-12)
