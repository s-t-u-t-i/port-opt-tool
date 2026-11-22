"""
Aspect 8 — Resource leaks.

Documented gap #20: the two matplotlib plotters create figures and NEVER
close them (core/plotting.py:143 and :171 call st.pyplot but there is no
plt.close anywhere). Over a long Streamlit session every slider drag adds
another figure to the heap until memory bloats.
"""
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import core.plotting as plotting


@pytest.fixture(autouse=True)
def _clean_figures():
    """Start and end every test in this file with an empty figure heap."""
    plt.close("all")
    yield
    plt.close("all")


def test_matplotlib_figures_are_closed_after_every_plot(
    five_asset_log_returns, monkeypatch, isolated_fs
):
    """Fixed (was M4 / gap #20): both matplotlib plotters close their figures,
    so 20 simulated reruns leave ZERO figures alive in the global heap."""
    import streamlit as st

    monkeypatch.setattr(st, "pyplot", lambda *a, **k: None)  # skip rendering

    before = len(plt.get_fignums())
    for _ in range(20):
        plotting.plot_log_returns(five_asset_log_returns)
        plotting.plot_correlation_heatmap(five_asset_log_returns.corr())
    after = len(plt.get_fignums())

    leaked = after - before
    assert leaked == 0


def test_single_ticker_plot_now_renders(five_asset_log_returns, isolated_fs):
    """Fixed (was H1): plotting exactly one ticker works — plt.subplots(1, 1)
    returns a bare Axes, which is now handled — and the figure is closed."""
    single = five_asset_log_returns[["AAA.NS"]]
    plotting.plot_log_returns(single)  # must NOT raise
    assert plt.get_fignums() == []     # rendered and closed


def test_plotly_functions_do_not_touch_matplotlib_heap(mc_portfolios):
    """The three Plotly-based plotters return a figure object and leave the
    matplotlib heap untouched — no leak on that side."""
    from helpers import SYNTHETIC_TICKERS

    before = len(plt.get_fignums())
    fig1 = plotting.plot_sharpe_ratio_bar_chart(pd.Series([1.0, 2.0], index=SYNTHETIC_TICKERS[:2]))
    fig2 = plotting.plot_portfolio_weight_comparison(
        pd.DataFrame([[0.5, 0.5]], index=["MC Optimal"], columns=SYNTHETIC_TICKERS[:2])
    )
    fig3 = plotting.plot_metric_comparison(
        pd.DataFrame({"Portfolio": ["A", "B"], "Return": [0.1, 0.2]}), "Return"
    )
    after = len(plt.get_fignums())

    import plotly.graph_objects as go

    assert isinstance(fig1, go.Figure) and isinstance(fig2, go.Figure) and isinstance(fig3, go.Figure)
    assert after == before


def test_app_reset_defaults_deletes_all_simulation_keys():
    """Canary for session-state accumulation: Reset must delete every
    simulation artifact it created (streamlit_app.py:57-73)."""
    from helpers import project_root

    source = (project_root() / "streamlit_app.py").read_text(encoding="utf-8")
    for key in [
        "portfolios_df",
        "mc_result",
        "simulation_tickers",
        "custom_result",
        "simulation_params",
    ]:
        assert f"del st.session_state.{key}" in source, f"reset_defaults no longer clears {key}"
