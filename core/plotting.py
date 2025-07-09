import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_efficient_frontier(portfolios_df, frontier_vols, target_returns,
                                mc_opt_vol, mc_opt_return, mc_opt_sharpe,
                                opt_volatility, opt_return, opt_sharpe,
                                min_vol, min_return, min_sharpe,
                                risk_free_rate):
    """
    Plots the efficient frontier, Monte Carlo portfolios, optimal portfolios,
    and the Capital Market Line using Plotly.

    Parameters:
    - portfolios_df (DataFrame): Monte Carlo simulation results.
    - frontier_vols (list): Volatilities from the efficient frontier.
    - target_returns (list): Returns from the efficient frontier.
    - mc_opt_vol (float): Volatility of Monte Carlo optimal portfolio.
    - mc_opt_return (float): Return of Monte Carlo optimal portfolio.
    - mc_opt_sharpe (float): Sharpe ratio of Monte Carlo optimal portfolio.
    - opt_volatility (float): Volatility of true optimal portfolio.
    - opt_return (float): Return of true optimal portfolio.
    - opt_sharpe (float): Sharpe ratio of true optimal portfolio.
    - min_vol (float): Volatility of global minimum variance portfolio.
    - min_return (float): Return of global minimum variance portfolio.
    - min_sharpe (float): Sharpe ratio of global minimum variance portfolio.
    - risk_free_rate (float): Annual risk-free rate used for CAL.
    """
    fig = go.Figure()

    # Plot all simulated portfolios (scatter plot)
    fig.add_trace(go.Scatter(
        x=portfolios_df['Volatility'],
        y=portfolios_df['Return'],
        mode='markers',
        marker=dict(color='rgba(0, 0, 255, 0.3)', size=5),
        name='Simulated Portfolios',
        hovertemplate='Return: %{y:.2%}<br>Volatility: %{x:.2%}<br>'
    ))

    # Plot the efficient frontier (smooth line)
    fig.add_trace(go.Scatter(
        x=frontier_vols,
        y=target_returns,
        mode='lines',
        line=dict(color='black', width=2),
        name='Efficient Frontier'
    ))

    # Highlight Monte Carlo optimal portfolio
    fig.add_trace(go.Scatter(
        x=[mc_opt_vol],
        y=[mc_opt_return],
        mode='markers',
        marker=dict(color='blue', size=9, symbol='diamond'),
        name='Monte Carlo Optimal Portfolio',
        hovertemplate=f"Return: {mc_opt_return:.2%}<br>Volatility: {mc_opt_vol:.2%}<br>Sharpe: {mc_opt_sharpe:.2f}"
    ))

    # Highlight true optimal (Sharpe-maximizing) portfolio from deterministic optimization
    fig.add_trace(go.Scatter(
        x=[opt_volatility],
        y=[opt_return],
        mode='markers',
        marker=dict(color='red', size=8),
        name='True Optimal Portfolio',
        hovertemplate=f"Return: {opt_return:.2%}<br>Volatility: {opt_volatility:.2%}<br>Sharpe: {opt_sharpe:.2f}"
    ))

    # Highlight Global Minimum Variance Portfolio
    fig.add_trace(go.Scatter(
        x=[min_vol],
        y=[min_return],
        mode='markers',
        marker=dict(color='green', size=7),
        name='Global Minimum Variance Portfolio',
        hovertemplate=f"Return: {min_return:.2%}<br>Volatility: {min_vol:.2%}<br>Sharpe: {min_sharpe:.2f}"
    ))

    # Plot Capital Allocation Line (CAL) using slope = Sharpe ratio
    cal_x = np.linspace(0, portfolios_df['Volatility'].max(), 100)
    cal_y = risk_free_rate + opt_sharpe * cal_x
    fig.add_trace(go.Scatter(
        x=cal_x,
        y=cal_y,
        mode='lines',
        line=dict(color='orange', width=2, dash='dot'),
        name='Capital Allocation Line (CAL)'
    ))

    # Set plot layout and styling
    fig.update_layout(
        width=1100,
        height=600,
        title='Efficient Frontier with Monte Carlo Portfolios, Optimal Portfolios, and Capital Market Line',
        title_font=dict(size=20),
        xaxis=dict(title='Volatility (Risk)', range=[0.16, 0.30], title_font=dict(size=16), tickfont=dict(size=14)),
        yaxis=dict(title='Expected Return', range=[0.08, 0.30], title_font=dict(size=16), tickfont=dict(size=14)),
        legend=dict(x=0.75, y=0.05, bgcolor='rgba(255,255,255,0.5)', font=dict(size=12)),
        plot_bgcolor='dark grey',
        hovermode='closest'
    )

    # Create output directory if it doesn't exist and save the plot as an interactive HTML file
    os.makedirs("output", exist_ok=True)

    # Save the plot as an interactive HTML file
    fig.write_html("output/efficient_frontier.html")
    return fig



def plot_log_returns(log_returns: pd.DataFrame):
    """
    Plots daily log returns for each asset using Matplotlib subplots.

    Parameters:
    - log_returns (DataFrame): Daily log returns for each asset.
    """
    num_assets = len(log_returns.columns)
    fig, axes = plt.subplots(num_assets, 1, figsize=(12, 3 * num_assets), sharex=True)

    for i, ticker in enumerate(log_returns.columns):
        axes[i].plot(log_returns.index, log_returns[ticker])
        axes[i].set_title(f"Log Returns: {ticker}")
        axes[i].set_ylabel("Log Return")
        axes[i].grid(True)

    plt.xlabel("Date")
    plt.tight_layout()
    st.pyplot(fig)


def plot_correlation_heatmap(corr_matrix, title='Correlation Matrix'):
    """
    Plots a heatmap of the correlation matrix using Seaborn.

    Parameters:
    - corr_matrix (DataFrame): Correlation matrix of asset returns.
    - title (str): Plot title.
    """
    # Copy the matrix to avoid modifying the original
    corr_matrix = corr_matrix.copy()

    # Remove multi-index labels (if present) for cleaner visualization
    corr_matrix.columns.name = None
    corr_matrix.index.name = None

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f", ax=ax)
    ax.set_title(title)
    plt.tight_layout()

    st.pyplot(fig)

def plot_sharpe_ratio_bar_chart(sharpe_series):
    fig = go.Figure(go.Bar(
        x=sharpe_series.index,
        y=sharpe_series.values,
        marker_color='teal',
        width=0.4
    ))
    fig.update_layout(
        title="Sharpe Ratios of Individual Assets",
        xaxis_title="Stock",
        yaxis_title="Sharpe Ratio",
        height=400
    )
    return fig

def plot_portfolio_weight_comparison(weights_df):
    fig = go.Figure()
    for stock in weights_df.index:
        fig.add_trace(go.Bar(
            x=weights_df.columns,
            y=weights_df.loc[stock],
            name=stock,
            width=0.4
        ))
    fig.update_layout(
        barmode='stack',
        title='Portfolio Weight Composition (Stacked Bar)',
        xaxis_title='Portfolio',
        yaxis_title='Weight (%)',
        legend_title='Stock',
        height=500
    )
    return fig

def plot_metric_comparison(perf_df, metric):
    fig = go.Figure(go.Bar(
        x=perf_df['Portfolio'],
        y=perf_df[metric],
        name=metric,
        width=0.4
    ))
    fig.update_layout(
        title=f"{metric} Comparison",
        yaxis_title=metric,
        xaxis_title="Portfolio"
    )
    return fig
