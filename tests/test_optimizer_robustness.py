"""
Aspect 4 — Optimizer robustness (fix guards).

Fixed behavior:
- A failed SLSQP solve now RAISES instead of silently returning garbage
  weights (core/true_optimization.py, find_true_optimal_portfolio and
  get_min_variance_portfolio).
- A broken covariance matrix raises a clear ValueError once, up front
  (validate_cov_matrix), instead of producing NaN portfolios.
- The efficient frontier still skips failed targets, and can still come
  back empty (that part was always checked).
"""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult

import core.true_optimization as true_opt
from core.true_optimization import (
    compute_all_true_portfolios,
    compute_efficient_frontier,
    find_true_optimal_portfolio,
    get_min_variance_portfolio,
    portfolio_annual_perf,
)

RISK_FREE = 0.05


# ---------------------------------------------------------------------------
# Broken math inputs
# ---------------------------------------------------------------------------
def test_non_psd_covariance_produces_nan_vol_silently():
    """portfolio_annual_perf is pure math: a non-PSD covariance (negative
    eigenvalue) still yields NaN volatility with no raise. The GUARD lives in
    compute_all_true_portfolios (see next test) — this documents the raw math."""
    bad_cov = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=["X", "Y"], columns=["X", "Y"])
    mean = pd.Series([0.001, 0.0005], index=["X", "Y"])
    with np.errstate(invalid="ignore"):
        ret, vol = portfolio_annual_perf(np.array([0.5, -0.5]), mean, bad_cov)
    assert np.isnan(vol)


def test_non_psd_covariance_is_rejected_up_front(five_asset_log_returns):
    """Fixed (was L4): compute_all_true_portfolios rejects a broken covariance
    matrix with a clear error instead of optimizing on it."""
    n = five_asset_log_returns.shape[1]
    bad = pd.DataFrame(
        np.eye(n) * 0.5, index=five_asset_log_returns.columns,
        columns=five_asset_log_returns.columns,
    )
    bad.iloc[0, 1] = 2.0   # push an eigenvalue negative
    bad.iloc[1, 0] = 2.0
    with pytest.raises(ValueError, match="positive semi-definite"):
        compute_all_true_portfolios(
            five_asset_log_returns.mean(), bad, RISK_FREE
        )


def test_single_asset_zero_variance_divides_by_zero():
    """Raw-math edge: constant prices -> zero variance -> vol 0 -> Sharpe
    divides by zero inside the objective. The optimizer entry points reject
    degenerate covariances before reaching this (see validate_cov_matrix)."""
    mean = pd.Series([0.0], index=["SOLO"])
    cov = pd.DataFrame([[0.0]], index=["SOLO"], columns=["SOLO"])
    with np.errstate(divide="ignore", invalid="ignore"):
        ret, vol = portfolio_annual_perf(np.array([1.0]), mean, cov)
        assert vol == 0.0
        neg_sharpe = true_opt.negative_sharpe_ratio(
            np.array([1.0]), mean, cov, RISK_FREE
        )
    assert np.isinf(neg_sharpe)


# ---------------------------------------------------------------------------
# result.success IS now checked (was C2)
# ---------------------------------------------------------------------------
def _failed_result(x):
    return OptimizeResult(
        x=x, success=False, status=4, message="forced failure", fun=0.0, nit=0
    )


def test_failed_optimizer_result_now_raises(five_asset_log_returns, monkeypatch):
    """Fixed (was C2): a failed SLSQP solve raises a RuntimeError with the
    solver's message instead of handing back garbage weights as 'optimal'."""
    monkeypatch.setattr(
        true_opt,
        "minimize",
        lambda fun, x0, *a, **k: _failed_result(np.full(len(x0), 1.0 / len(x0))),
    )
    with pytest.raises(RuntimeError, match="Sharpe-ratio optimization failed"):
        find_true_optimal_portfolio(
            five_asset_log_returns.mean(), five_asset_log_returns.cov(), RISK_FREE
        )
    with pytest.raises(RuntimeError, match="Minimum-variance optimization failed"):
        get_min_variance_portfolio(
            five_asset_log_returns.mean(), five_asset_log_returns.cov()
        )


def test_real_slsqp_baseline_succeeds_on_good_data(five_asset_log_returns):
    """Sanity baseline: on healthy inputs SLSQP reports success — the raise
    only fires on genuine failures."""
    res = find_true_optimal_portfolio(
        five_asset_log_returns.mean(), five_asset_log_returns.cov(), RISK_FREE
    )
    assert res.success is True
    assert np.isclose(res.x.sum(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Efficient frontier behavior
# ---------------------------------------------------------------------------
def test_frontier_skips_failed_targets(five_asset_log_returns, monkeypatch):
    """compute_efficient_frontier checks result.success and skips failures
    — targets that fail vanish from the frontier (creating holes, which the
    frontier plot renders as gaps rather than wrong data)."""
    real_minimize = true_opt.minimize
    state = {"n": 0}

    def flaky(fun, x0, *a, **k):
        state["n"] += 1
        res = real_minimize(fun, x0, *a, **k)
        if state["n"] % 3 == 0:  # force every 3rd target to fail
            res = OptimizeResult(dict(res))
            res.success = False
        return res

    monkeypatch.setattr(true_opt, "minimize", flaky)
    frontier = compute_efficient_frontier(
        five_asset_log_returns.mean(), five_asset_log_returns.cov(), num_points=24
    )
    assert len(frontier) == 16  # 24 targets, every 3rd silently skipped


def test_frontier_can_come_back_completely_empty(five_asset_log_returns, monkeypatch):
    """compute_efficient_frontier alone returns an empty frame when every
    target fails; compute_all_true_portfolios raises (the true/GMV solve runs
    first and now refuses to report failure as success)."""
    monkeypatch.setattr(
        true_opt,
        "minimize",
        lambda fun, x0, *a, **k: _failed_result(np.full(len(x0), 1.0 / len(x0))),
    )
    frontier = compute_efficient_frontier(
        five_asset_log_returns.mean(), five_asset_log_returns.cov(), num_points=10
    )
    assert frontier.shape == (0, 2)

    with pytest.raises(RuntimeError):
        compute_all_true_portfolios(
            five_asset_log_returns.mean(), five_asset_log_returns.cov(), RISK_FREE
        )


# ---------------------------------------------------------------------------
# Degenerate asset counts
# ---------------------------------------------------------------------------
def test_two_highly_correlated_assets_still_optimize():
    """Near-singular covariance (corr = 0.999): the optimizer survives; the
    up-front PSD validation tolerates valid-but-near-singular matrices."""
    vol = 0.012
    cov_d = np.array([[vol**2, 0.999 * vol**2], [0.999 * vol**2, vol**2]])
    cov = pd.DataFrame(cov_d, index=["A", "B"], columns=["A", "B"])
    mean = pd.Series([0.0004, 0.0006], index=["A", "B"])

    sharp = find_true_optimal_portfolio(mean, cov, RISK_FREE)
    gmv = get_min_variance_portfolio(mean, cov)
    assert sharp.success and gmv.success
    assert np.isclose(sharp.x.sum(), 1.0, atol=1e-6)
    assert np.isclose(gmv.x.sum(), 1.0, atol=1e-6)


def test_single_asset_optimizer_converges_to_all_in():
    """1 asset: the only valid portfolio is 100% of it. Math works fine here —
    and since the plotting fix, the 1-ticker UI case renders correctly too."""
    mean = pd.Series([0.0005], index=["SOLO"])
    cov = pd.DataFrame([[0.0001]], index=["SOLO"], columns=["SOLO"])
    res = find_true_optimal_portfolio(mean, cov, RISK_FREE)
    assert res.success
    assert np.allclose(res.x, [1.0])
