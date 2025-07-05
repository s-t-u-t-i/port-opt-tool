import streamlit as st
import pandas as pd
import numpy as np
from core.data_utils import fetch_and_save_data, load_close_prices_from_csv, calculate_log_returns, calculate_descriptive_statistics,compute_cov_corr_matrices, calculate_individual_sharpe_ratios

from core.simulation import run_monte_carlo_simulation
from core.mc_optimization import get_mc_optimal_portfolio
from core.true_optimization import portfolio_annual_perf, compute_all_true_portfolios
from core.plotting import plot_log_returns, plot_correlation_heatmap, plot_efficient_frontier, plot_sharpe_ratio_bar_chart, plot_portfolio_weight_comparison, plot_metric_comparison
import plotly.graph_objects as go

st.set_page_config(page_title="Portfolio Optimization Tool", layout="wide")

st.title("Portfolio Optimization & Analysis Tool")

st.markdown("""
This interactive app helps you analyze portfolio performance using real stock data, simulate thousands of portfolios, and understand efficient frontier concepts.

### How to Use
1. Use the sidebar to select stocks and date range.
2. Explore stock performance and summary statistics in the EDA section.
3. Run the Monte Carlo simulation to generate thousands of simulated portfolios.
4. Create your own with custom weights, observe its metrics and the efficient frontier.
5. Compare metrics, weights, and Sharpe ratios across different portfolios.
""")

# ----- Sidebar Controls -----
st.sidebar.header("Settings")
def reset_defaults():
    st.session_state.tickers = ['HDFCBANK.NS', 'ITC.NS', 'RELIANCE.NS', 'SUNPHARMA.NS', 'TCS.NS']
    st.session_state.start_date = '2020-01-01'
    st.session_state.end_date = '2024-12-31'
    st.session_state.risk_free_rate = 0.05
    st.session_state.num_portfolios = 10000
    st.session_state.custom_generated = False
    st.session_state.simulated = False

if 'tickers' not in st.session_state:
    reset_defaults()

if st.sidebar.button("Reset to Default"):
    reset_defaults()

tickers = st.sidebar.multiselect("Select Tickers", options=['HDFCBANK.NS', 'ITC.NS', 'RELIANCE.NS', 'SUNPHARMA.NS', 'TCS.NS'], default=st.session_state.tickers)
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime(st.session_state.start_date))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime(st.session_state.end_date))
risk_free_rate = st.sidebar.number_input("Risk-Free Rate", value=st.session_state.risk_free_rate)
num_portfolios = st.sidebar.number_input("Number of Portfolios", min_value=1000, max_value=50000, value=st.session_state.num_portfolios, step=1000)

simulate_button = st.sidebar.button("Run Simulation")

# ----- Load & Process Data -----
fetch_and_save_data(tickers, start_date, end_date)
close_prices = load_close_prices_from_csv('data/stock_data.csv')
log_returns = calculate_log_returns(close_prices)
summary_df = calculate_descriptive_statistics(log_returns)
cov_matrix_daily, corr_matrix = compute_cov_corr_matrices(log_returns)
mean_returns_daily = log_returns.mean()

# ----- Exploratory Data Analysis ------
st.header("Exploratory Data Analysis (EDA)")

log_returns = calculate_log_returns(close_prices)
_, corr_matrix = compute_cov_corr_matrices(log_returns)

st.subheader("1. Log Returns")

with st.expander("Show Log Returns Chart"):
    st.markdown("This plot shows the daily log returns for each selected stock.")
    plot_log_returns(log_returns)

st.subheader("2. Descriptive Statistics")
with st.expander("Show Descriptive Statistics"):
    st.markdown("**Summary of log returns for selected stocks:**")
    st.dataframe(summary_df.style.format("{:.4f}"))

st.subheader("3. Correlation Heatmap")

with st.expander("Show Correlation Heatmap"):
    st.markdown("Visualizes the correlation between stock returns to highlight diversification potential.")
    plot_correlation_heatmap(corr_matrix)

# Individual Sharpe Ratios
st.subheader("4. Individual Sharpe Ratios")

with st.expander("Show Sharpe Ratio Chart"):
    st.markdown("Higher Sharpe ratios indicate better risk-adjusted performance for individual assets.")
    individual_sharpe_ratios, _, _ = calculate_individual_sharpe_ratios(summary_df, risk_free_rate)
    sharpe_series = individual_sharpe_ratios 
    fig_sharpe = plot_sharpe_ratio_bar_chart(sharpe_series)
    st.plotly_chart(fig_sharpe)

with st.expander("Show Sharpe Ratio Table"):
    st.dataframe(sharpe_series.to_frame("Sharpe Ratio").style.format("{:.2f}"))

# ----- Portfolio Optimization Header -----
st.header("Generate a Custom Portfolio and Compare with Optimized Portfolios")

#  Step 1: If simulation button is clicked, just flag it as done
if simulate_button:
    st.session_state.simulated = True
    st.session_state.custom_generated = False

#  Step 2: If simulation has been run before, load or reuse data
if st.session_state.get("simulated", False):
    if "portfolios_df" not in st.session_state:
        # Only run simulation once, then reuse
        portfolios_df = run_monte_carlo_simulation(log_returns, num_portfolios, risk_free_rate)
        st.session_state.portfolios_df = portfolios_df
        mc_ret, mc_vol, mc_sharpe, mc_weights = get_mc_optimal_portfolio(portfolios_df)
        st.session_state.mc_result = (mc_ret, mc_vol, mc_sharpe, mc_weights)
    else:
        portfolios_df = st.session_state.portfolios_df
        mc_ret, mc_vol, mc_sharpe, mc_weights = st.session_state.mc_result
else:
    portfolios_df = None
    mc_ret = mc_vol = mc_sharpe = mc_weights = None

#  Step 3: Custom Portfolio UI
if not st.session_state.get("simulated", False):
    st.subheader("Custom Portfolio")
    st.warning("⚠️ Please press the 'Run Simulation' button before generating a custom portfolio.")
else:
    # Required values for optimization
    st.subheader("Custom Portfolio")

    results = compute_all_true_portfolios(mean_returns_daily, cov_matrix_daily, risk_free_rate)

    true_ret, true_vol, true_sharpe, true_weights = results["true"]
    min_ret, min_vol, min_sharpe, min_weights = results["min"]
    frontier_df = results["frontier"]


    st.markdown("🎯 Adjust the sliders to design your own portfolio. The weights must sum to 1.0.")

    custom_weights = []
    for ticker in tickers:
        custom_weights.append(
            st.slider(f"{ticker} Weight", min_value=0.0, max_value=1.0, step=0.01, key=f"slider_{ticker}")
        )

    if st.button("Generate Custom Portfolio"):
        weight_sum = sum(custom_weights)
        if not np.isclose(weight_sum, 1.0):
            st.error("🚫 ERROR: The weights must sum to 1.0")
        else:
            custom_ret, custom_vol = portfolio_annual_perf(np.array(custom_weights), mean_returns_daily, cov_matrix_daily)
            custom_sharpe = (custom_ret - risk_free_rate) / custom_vol
            st.session_state.custom_result = (custom_ret, custom_vol, custom_sharpe, custom_weights)
            st.session_state.custom_generated = True
            st.success("✅ Custom portfolio successfully generated!")

        st.markdown(f"""
        **Custom Portfolio Metrics:**

        -  **Expected Return:** {custom_ret:.2%}  
        -  **Risk (Volatility):** {custom_vol:.2%}  
        -  **Sharpe Ratio:** {custom_sharpe:.2f}
        """)

# --- Efficient Frontier Plot ----
st.subheader("Efficient Frontier")

if not st.session_state.get("custom_generated", False):
    st.warning("⚠️ Please generate the custom portfolio first.")
else:
    fig = plot_efficient_frontier(
        portfolios_df=st.session_state.portfolios_df,
        frontier_vols=frontier_df["Volatility"],
        target_returns=frontier_df["Return"],
        mc_opt_vol=mc_vol,
        mc_opt_return=mc_ret,
        mc_opt_sharpe=mc_sharpe,
        opt_volatility=true_vol,
        opt_return=true_ret,
        opt_sharpe=true_sharpe,
        min_vol=min_vol,
        min_return=min_ret,
        min_sharpe=min_sharpe,
        risk_free_rate=risk_free_rate
    )

    # Add custom porfolio to the figure 
    fig.add_trace(go.Scatter(
        x=[custom_vol],
        y=[custom_ret],
        mode='markers',
        name='Custom Portfolio',
        marker=dict(color='orange', size=10, symbol='star'),
        showlegend=True
    ))
    st.plotly_chart(fig)
    st.markdown("*Note: Axes are zoomed in to improve frontier visibility.*")



# ----- Weight Comparison Chart -----
if st.session_state.custom_generated:
    custom_weights = st.session_state.custom_result[-1]
    portfolios = ["MC Optimal", "True Optimal", "GMV", "Custom"]
    weights_data = pd.DataFrame(
        [mc_weights, true_weights, min_weights, custom_weights],
        columns=tickers,
        index=portfolios
    )

    st.subheader("Portfolio Weight Comparison")

    # Transpose and convert to percentage
    weights_data = weights_data.T * 100  # index = stock, columns = portfolios

    st.plotly_chart(plot_portfolio_weight_comparison(weights_data))

# Ensure custom metrics are pulled again (in case 'Run Simulation' was clicked after generating custom portfolio)
if st.session_state.custom_generated:
    custom_ret, custom_vol, custom_sharpe, custom_weights = st.session_state.custom_result
else:
    custom_ret = custom_vol = custom_sharpe = np.nan
    custom_weights = [0] * len(tickers)

# ----- Performance Metrics Comparison -----
st.subheader("Portfolio Metrics Comparison")

if not st.session_state.get("custom_generated", False):
    st.warning("⚠️ Please generate the custom portfolio first to view performance metrics.")
else:
    portfolios = ["MC Optimal", "True Optimal", "GMV", "Custom"]
    perf_df = pd.DataFrame({
        "Portfolio": portfolios,
        "Return": [mc_ret, true_ret, min_ret, custom_ret],
        "Volatility": [mc_vol, true_vol, min_vol, custom_vol],
        "Sharpe Ratio": [mc_sharpe, true_sharpe, min_sharpe, custom_sharpe]
    })
    metrics = ["Return", "Volatility", "Sharpe Ratio"]
    for m in metrics:
        fig = plot_metric_comparison(perf_df, m)
        st.plotly_chart(fig)

