# simulation.py

import numpy as np
import pandas as pd

def run_monte_carlo_simulation(log_returns, num_portfolios=10_000, risk_free_rate=0.05, output_csv='data/Monte_Carlo_simulation.csv'):
    """
    Runs a Monte Carlo simulation to generate random portfolios and compute
    their expected return, volatility, and Sharpe ratio.

    Parameters:
    - log_returns (DataFrame): Daily log returns of assets.
    - num_portfolios (int): Number of random portfolios to simulate.
    - risk_free_rate (float): Annual risk-free rate used in Sharpe ratio.
    - output_csv (str): File path to optionally save the simulation results.

    Returns:
    - DataFrame: Portfolio simulation results with return, volatility, Sharpe ratio,
                 and individual asset weights.
    """
    # List of asset tickers
    tickers = log_returns.columns.tolist()

    # Annualize mean returns and covariance matrix
    annual_mean_returns = log_returns.mean() * 252
    annual_cov_matrix = log_returns.cov() * 252

    # Initialize result containers
    results = {
        'Return': [],
        'Volatility': [],
        'Sharpe Ratio': [],
        'Weights': []
    }

    # Monte Carlo simulation loop
    for _ in range(num_portfolios):
        # Generate random portfolio weights that sum to 1
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)

        # Calculate expected portfolio return and volatility
        port_return = np.dot(weights, annual_mean_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix, weights)))

        # Calculate Sharpe ratio
        sharpe_ratio = (port_return - risk_free_rate) / port_volatility

        # Store computed metrics and weights
        results['Return'].append(port_return)
        results['Volatility'].append(port_volatility)
        results['Sharpe Ratio'].append(sharpe_ratio)
        results['Weights'].append(weights)

    # Convert results to DataFrame
    portfolios_df = pd.DataFrame(results)

    # Expand the list of weights into individual columns for each ticker
    weights_df = pd.DataFrame(portfolios_df['Weights'].tolist(), columns=tickers)

    # Merge weights with the main DataFrame and remove original list column
    portfolios_df = pd.concat([portfolios_df.drop('Weights', axis=1), weights_df], axis=1)

    return portfolios_df

