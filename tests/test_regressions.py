"""
Aspect 6 — Regression tests: bugs that were already fixed stay fixed.

Each test here is a permanent tripwire for a historical bug:
  - Errno 22 savefig crash      (fixed in commit 69288fb6)
  - mc_result AttributeError    (fixed: guard at streamlit_app.py:193)
  - NameError on bad weights    (fixed: metrics inside the else branch,
                                 streamlit_app.py:229-233)
  - duplicate log-returns       (fixed: computed exactly once, :140)
  - simulation.py output_csv    (removed parameter)
"""
import inspect
import numpy as np
import pandas as pd
import pytest

from core.simulation import run_monte_carlo_simulation


def _app_source():
    from helpers import project_root

    return (project_root() / "streamlit_app.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Errno 22: best-effort savefig must never crash the app
# ---------------------------------------------------------------------------
def test_plot_log_returns_survives_savefig_oserror(five_asset_log_returns, monkeypatch):
    import matplotlib.pyplot as plt
    import core.plotting as plotting

    def boom(*args, **kwargs):
        raise OSError(22, "simulated: file locked by another process")

    monkeypatch.setattr(plotting.plt, "savefig", boom)
    try:
        plotting.plot_log_returns(five_asset_log_returns)  # must NOT raise
    finally:
        plt.close("all")


def test_plot_correlation_heatmap_survives_savefig_oserror(
    five_asset_log_returns, monkeypatch
):
    import matplotlib.pyplot as plt
    import core.plotting as plotting

    def boom(*args, **kwargs):
        raise OSError(22, "simulated: file locked by another process")

    monkeypatch.setattr(plotting.plt, "savefig", boom)
    try:
        plotting.plot_correlation_heatmap(five_asset_log_returns.corr())  # must NOT raise
    finally:
        plt.close("all")


# ---------------------------------------------------------------------------
# Simulated state without mc_result must not raise AttributeError
# ---------------------------------------------------------------------------
def test_app_guards_mc_result_access():
    """The 'simulated' flag must only count as valid when mc_result actually
    exists (streamlit_app.py:193: 'if simulated and "mc_result" in session')."""
    source = _app_source()
    assert 'st.session_state.get("simulated", False) and "mc_result" in st.session_state' in source


# ---------------------------------------------------------------------------
# Bad custom weights show an ERROR, not a NameError
# ---------------------------------------------------------------------------
def test_app_computes_custom_metrics_inside_else_branch():
    """Metrics must be computed only after the weights-sum check passes
    (streamlit_app.py:224-233), so a 0.999 sum can never NameError."""
    source = _app_source()
    assert "if not np.isclose(weight_sum, 1.0):" in source
    assert (
        "custom_sharpe = (custom_ret - risk_free_rate) / custom_vol" in source
    )
    assert (
        "st.session_state.custom_result = (custom_ret, custom_vol, custom_sharpe, custom_weights)"
        in source
    )


# ---------------------------------------------------------------------------
# Log returns are computed exactly once per run
# ---------------------------------------------------------------------------
def test_app_computes_log_returns_once():
    assert _app_source().count("calculate_log_returns(") == 1


# ---------------------------------------------------------------------------
# simulation.py no longer accepts output_csv
# ---------------------------------------------------------------------------
def test_simulation_signature_has_no_output_csv():
    params = inspect.signature(run_monte_carlo_simulation).parameters
    assert "output_csv" not in params
    assert list(params) == ["log_returns", "num_portfolios", "risk_free_rate"]
