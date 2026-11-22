"""
Aspect 7 — Performance & caching.

The timing tests print their measurements (visible with `pytest -s`) and use
generous ceilings — they exist to catch *regressions* (something becoming
10x slower) and to record current numbers, not to fail on slow machines.
"""
import time
import tracemalloc

import numpy as np
import pytest
import streamlit as st

import core.data_utils as data_utils
from core.simulation import run_monte_carlo_simulation
from core.true_optimization import compute_all_true_portfolios
from helpers import SYNTHETIC_TICKERS, make_prices, write_yf_csv

RISK_FREE = 0.05


# ---------------------------------------------------------------------------
# Caching: a second identical call must not re-download
# ---------------------------------------------------------------------------
def test_cache_hit_skips_second_download(isolated_fs, monkeypatch):
    """Mirrors the app's caching pattern (streamlit_app.py:12-14)."""
    calls = {"download": 0}

    def fake_download(tickers, **kwargs):
        calls["download"] += 1
        return make_prices(n_days=100, seed=3, tickers=SYNTHETIC_TICKERS)

    monkeypatch.setattr("core.data_utils.yf.download", fake_download)

    @st.cache_data(show_spinner=False)
    def get_stock_data(tickers, start_date, end_date):
        return data_utils.fetch_and_save_data(tuple(tickers), start_date, end_date)

    try:
        get_stock_data(list(SYNTHETIC_TICKERS), "2022-01-01", "2022-12-31")
        get_stock_data(list(SYNTHETIC_TICKERS), "2022-01-01", "2022-12-31")
        assert calls["download"] == 1  # second call served from cache

        # Different tickers -> cache miss -> real fetch
        get_stock_data(list(SYNTHETIC_TICKERS[:3]), "2022-01-01", "2022-12-31")
        assert calls["download"] == 2
    finally:
        get_stock_data.clear()


# ---------------------------------------------------------------------------
# Max-size simulation (the app's slider goes up to 50,000)
# ---------------------------------------------------------------------------
def test_max_portfolios_50000_timing_and_memory(five_asset_log_returns):
    tracemalloc.start()
    t0 = time.perf_counter()
    df = run_monte_carlo_simulation(five_asset_log_returns, num_portfolios=50_000, risk_free_rate=RISK_FREE)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert df.shape == (50_000, 3 + 5)
    print(f"\n[perf] 50,000-portfolio simulation: {elapsed:.2f}s, peak {peak / 1e6:.1f} MB")
    # Fixed (was L7): the simulation is vectorized — 50k portfolios used to
    # take ~40s in a Python loop; now it must finish in seconds.
    assert elapsed < 10


# ---------------------------------------------------------------------------
# What every slider drag re-executes (the app recomputes this on each rerun,
# streamlit_app.py:138-143)
# ---------------------------------------------------------------------------
def test_every_rerun_pipeline_cost(five_asset_prices, isolated_fs):
    write_yf_csv(isolated_fs / "data" / "stock_data.csv", five_asset_prices)

    t0 = time.perf_counter()
    close = data_utils.load_close_prices_from_csv("data/stock_data.csv")
    lr = data_utils.calculate_log_returns(close)
    data_utils.calculate_descriptive_statistics(lr)
    data_utils.compute_cov_corr_matrices(lr)
    elapsed = time.perf_counter() - t0

    print(f"\n[perf] per-rerun pipeline (load+logreturns+stats+cov): {elapsed * 1000:.0f} ms")
    assert elapsed < 5.0


# ---------------------------------------------------------------------------
# Optimizer: uncached compute vs the app's cached second call
# ---------------------------------------------------------------------------
def test_optimizer_cached_vs_uncached_latency(five_asset_log_returns):
    mean = five_asset_log_returns.mean()
    cov = five_asset_log_returns.cov()

    @st.cache_data(show_spinner=False)
    def get_true_portfolios(mean_daily, cov_daily, rf):
        return compute_all_true_portfolios(mean_daily, cov_daily, rf)

    try:
        t0 = time.perf_counter()
        get_true_portfolios(mean, cov, RISK_FREE)
        uncached = time.perf_counter() - t0

        t0 = time.perf_counter()
        get_true_portfolios(mean, cov, RISK_FREE)
        cached = time.perf_counter() - t0

        print(f"\n[perf] optimizer uncached: {uncached * 1000:.0f} ms | cached: {cached * 1000:.1f} ms")
        assert cached < uncached
        assert uncached < 60.0
    finally:
        get_true_portfolios.clear()


# ---------------------------------------------------------------------------
# Scaling: a 10-ticker universe
# ---------------------------------------------------------------------------
def test_ten_ticker_scaling():
    prices = make_prices(n_days=500, seed=11, n_assets=10,
                         tickers=[f"T{i}.NS" for i in range(10)])
    lr = np.log(prices / prices.shift(1)).dropna()

    t0 = time.perf_counter()
    np.random.seed(4)
    mc = run_monte_carlo_simulation(lr, num_portfolios=10_000, risk_free_rate=RISK_FREE)
    mc_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = compute_all_true_portfolios(lr.mean(), lr.cov(), RISK_FREE)
    opt_time = time.perf_counter() - t0

    assert mc.shape == (10_000, 3 + 10)
    assert out["min"][1] > 0
    assert len(out["frontier"]) > 10
    print(f"\n[perf] 10 tickers: MC(10k) {mc_time:.2f}s | optimizer {opt_time:.2f}s")
