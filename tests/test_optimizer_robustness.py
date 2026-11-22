"""
Aspect 4 — Optimizer robustness.

Key gap documented here: find_true_optimal_portfolio and
get_min_variance_portfolio NEVER check result.success
(core/true_optimization.py:66 and :92), so a failed optimization silently
returns whatever garbage weights the solver left behind. Only the efficient
frontier loop checks success (core/true_optimization.py:135).
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
    """Covariance [[1, 2], [2, 1]] has a negative eigenvalue (-1). A weight
    vector along that direction gives NEGATIVE portfolio variance, sqrt gives
    NaN, and NOTHING raises (true_optimization.py:19)."""
    bad_cov = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=["X", "Y"], columns=["X", "Y"])
    mean = pd.Series([0.001, 0.0005], index=["X", "Y"])
    with np.errstate(invalid="ignore"):
        ret, vol = portfolio_annual_perf(np.array([0.5, -0.5]), mean, bad_cov)
    assert np.isnan(vol)


def test_single_asset_zero_variance_divides_by_zero():
    """Documented edge: constant prices -> zero variance -> vol 0 -> Sharpe
    divides by zero. No guard raises anywhere (true_optimization.py:37)."""
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
# result.success is never checked (the big one)
# ---------------------------------------------------------------------------
def _failed_result(x):
    return OptimizeResult(
        x=x, success=False, status=4, message="forced failure", fun=0.0, nit=0
    )


def test_failed_optimizer_result_flows_through_silently(
    five_asset_log_returns, monkeypatch
):
    """The optimizer is forced to report failure. compute_all_true_portfolios
    must still complete AND hand back the failed weights as if they were the
    answer — proving success is never checked."""
    n = five_asset_log_returns.shape[1]
    monkeypatch.setattr(
        true_opt, "minimize", lambda fun, x0, *a, **k: _failed_result(np.full(len(x0), 1.0 / len(x0)))
    )

    out = compute_all_true_portfolios(
        five_asset_log_returns.mean(), five_asset_log_returns.cov(), RISK_FREE
    )
    # No exception was raised, and the "optimal" weights are the raw failed guess
    assert np.allclose(out["true"][3], np.full(n, 0.2))
    assert np.allclose(out["min"][3], np.full(n, 0.2))


def test_real_slsqp_baseline_succeeds_on_good_data(five_asset_log_returns):
    """Sanity baseline: on healthy inputs SLSQP reports success — so the
    missing check wouldn't matter if failures were impossible. They are not."""
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
    (true_optimization.py:135) — targets that fail simply vanish from the
    frontier, creating invisible holes."""
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
    """If every target fails, the frontier is an EMPTY frame with zero rows —
    and compute_all_true_portfolios still completes without error, handing the
    app an empty frontier to plot (true_optimization.py:175 filter keeps it
    empty)."""
    monkeypatch.setattr(
        true_opt,
        "minimize",
        lambda fun, x0, *a, **k: _failed_result(np.full(len(x0), 1.0 / len(x0))),
    )
    out = compute_all_true_portfolios(
        five_asset_log_returns.mean(), five_asset_log_returns.cov(), RISK_FREE
    )
    assert out["frontier"].shape == (0, 2)


# ---------------------------------------------------------------------------
# Degenerate asset counts
# ---------------------------------------------------------------------------
def test_two_highly_correlated_assets_still_optimize():
    """Near-singular covariance (corr = 0.999). The optimizer survives, but
    this is the regime where tiny numerical noise can flip the 'optimal'
    weights wildly — worth knowing."""
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
    the 1-ticker crash lives in the PLOTTING layer (see test_resource_leaks /
    GUI tests), not in the optimizer."""
    mean = pd.Series([0.0005], index=["SOLO"])
    cov = pd.DataFrame([[0.0001]], index=["SOLO"], columns=["SOLO"])
    res = find_true_optimal_portfolio(mean, cov, RISK_FREE)
    assert res.success
    assert np.allclose(res.x, [1.0])
