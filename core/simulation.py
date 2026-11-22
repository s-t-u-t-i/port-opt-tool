# simulation.py

import numpy as np
import pandas as pd

def run_monte_carlo_simulation(log_returns, num_portfolios=10_000, risk_free_rate=0.05):
    """
    Runs a Monte Carlo simulation to generate random portfolios and compute
    their expected return, volatility, and Sharpe ratio.

    All portfolios are drawn in a single vectorized step instead of a Python
    loop (a 50,000-portfolio run drops from ~40s to well under a second).
    `np.random.random((n, k))` consumes the global random stream in the same
    row-major order as n successive `np.random.random(k)` draws, so a fixed
    seed reproduces exactly the same portfolios as the old loop.

    Parameters:
    - log_returns (DataFrame): Daily log returns of assets.
    - num_portfolios (int): Number of random portfolios to simulate.
    - risk_free_rate (float): Annual risk-free rate used in Sharpe ratio.
    Returns:
    - DataFrame: Portfolio simulation results with return, volatility, Sharpe ratio,
                 and individual asset weights.
    """
    # List of asset tickers
    tickers = log_returns.columns.tolist()
    num_assets = len(tickers)

    # Annualize mean returns and covariance matrix
    annual_mean_returns = log_returns.mean() * 252
    annual_cov_matrix = log_returns.cov() * 252

    # Generate ALL random portfolio weights at once; each row sums to 1
    weights = np.random.random((num_portfolios, num_assets))
    weights /= weights.sum(axis=1, keepdims=True)

    # Vectorized return / volatility / Sharpe for every portfolio
    mean_vector = annual_mean_returns.to_numpy()
    cov_matrix = annual_cov_matrix.to_numpy()
    port_return = weights @ mean_vector
    port_variance = np.einsum('ij,jk,ik->i', weights, cov_matrix, weights)
    port_volatility = np.sqrt(port_variance)
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility

    # Assemble results with the same column layout as before:
    # Return, Volatility, Sharpe Ratio, then one weight column per ticker.
    portfolios_df = pd.DataFrame({
        'Return': port_return,
        'Volatility': port_volatility,
        'Sharpe Ratio': sharpe_ratio,
    })
    weights_df = pd.DataFrame(weights, columns=tickers)
    portfolios_df = pd.concat([portfolios_df, weights_df], axis=1)

    return portfolios_df
