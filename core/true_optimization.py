import numpy as np
import pandas as pd
from scipy.optimize import minimize

def portfolio_annual_perf(weights, mean_returns, cov_matrix):
    """
    Calculates the annualized portfolio return and volatility.

    Parameters:
    - weights (array): Portfolio weights.
    - mean_returns (Series): Daily mean returns.
    - cov_matrix (DataFrame): Daily covariance matrix.

    Returns:
    - Tuple: (Annualized return, Annualized volatility)
    """
    # Annualize return and volatility
    annual_return = np.sum(mean_returns * weights) * 252
    annual_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return annual_return, annual_vol


def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """
    Objective function to minimize: negative Sharpe ratio.

    Parameters:
    - weights (array): Portfolio weights.
    - mean_returns (Series): Daily mean returns.
    - cov_matrix (DataFrame): Daily covariance matrix.
    - risk_free_rate (float): Annual risk-free rate.

    Returns:
    - float: Negative of the annualized Sharpe ratio.
    """
    p_return, p_std_dev = portfolio_annual_perf(weights, mean_returns, cov_matrix)
    sharpe_ratio = (p_return - risk_free_rate) / p_std_dev
    return -sharpe_ratio  # Maximizing Sharpe = Minimizing negative Sharpe


def find_true_optimal_portfolio(mean_returns, cov_matrix, risk_free_rate):
    """
    Finds the portfolio weights that maximize the Sharpe ratio.

    Parameters:
    - mean_returns (Series): Daily mean returns.
    - cov_matrix (DataFrame): Daily covariance matrix.
    - risk_free_rate (float): Annual risk-free rate.

    Returns:
    - OptimizeResult: Output of the scipy optimizer containing weights and status.
    """
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    
    # Equal-weighted initial guess
    init_guess = num_assets * [1. / num_assets]

    # Bounds for each weight: between 0 and 1 (no short-selling)
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Constraint: weights must sum to 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    # Optimize using Sequential Least Squares Programming (SLSQP)
    result = minimize(negative_sharpe_ratio, init_guess, args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)

    return result


def get_min_variance_portfolio(mean_returns, cov_matrix):
    """
    Finds the portfolio with the lowest annualized volatility.

    Parameters:
    - mean_returns (Series): Daily mean returns.
    - cov_matrix (DataFrame): Daily covariance matrix.

    Returns:
    - OptimizeResult: Output of the scipy optimizer containing weights and status.
    """
    num_assets = len(mean_returns)
    init_guess = num_assets * [1. / num_assets]
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    # Objective: Minimize portfolio volatility
    def portfolio_volatility(weights):
        return portfolio_annual_perf(weights, mean_returns, cov_matrix)[1]

    result = minimize(portfolio_volatility, init_guess, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    return result


def compute_efficient_frontier(mean_returns, cov_matrix, num_points=100):
    """
    Computes the efficient frontier by minimizing volatility at various target returns.

    Parameters:
    - mean_returns (Series): Daily mean returns.
    - cov_matrix (DataFrame): Daily covariance matrix.
    - num_points (int): Number of target return levels to compute on the frontier.

    Returns:
    - DataFrame: Contains return and volatility pairs representing the frontier.
    """
    num_assets = len(mean_returns)

    # Containers to hold frontier results
    results = {'Returns': [], 'Volatility': [], 'Weights': []}

    # Define target returns linearly spaced between min and max annualized return
    target_returns = np.linspace(mean_returns.min() * 252, mean_returns.max() * 252, num_points)

    for target in target_returns:
        init_guess = num_assets * [1. / num_assets]
        bounds = tuple((0, 1) for _ in range(num_assets))

        # Two constraints:
        # 1. Weights must sum to 1
        # 2. Expected return must equal the target
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(x * mean_returns) * 252 - target}
        ]

        # Minimize volatility subject to return constraint
        result = minimize(lambda w: portfolio_annual_perf(w, mean_returns, cov_matrix)[1],
                          init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        # Store successful results
        if result.success:
            port_return, port_vol = portfolio_annual_perf(result.x, mean_returns, cov_matrix)
            results['Returns'].append(port_return)
            results['Volatility'].append(port_vol)
            results['Weights'].append(result.x)

    # Assemble results into a DataFrame
    frontier_df = pd.DataFrame({
        'Return': results['Returns'],
        'Volatility': results['Volatility']
    })

    return frontier_df

def compute_all_true_portfolios(mean_returns, cov_matrix, risk_free_rate):
    """
    Computes the True Optimal, Minimum Variance, and Efficient Frontier portfolios.

    Parameters:
    - mean_returns (Series): Daily mean returns.
    - cov_matrix (DataFrame): Daily covariance matrix.
    - risk_free_rate (float): Annual risk-free rate.

    Returns:
    - dict: Contains true optimal, GMV, and frontier data.
    """
    # True Optimal Portfolio
    true_result = find_true_optimal_portfolio(mean_returns, cov_matrix, risk_free_rate)
    true_weights = true_result.x
    true_ret, true_vol = portfolio_annual_perf(true_weights, mean_returns, cov_matrix)
    true_sharpe = (true_ret - risk_free_rate) / true_vol

    # Global Minimum Variance Portfolio
    min_var_result = get_min_variance_portfolio(mean_returns, cov_matrix)
    min_weights = min_var_result.x
    min_ret, min_vol = portfolio_annual_perf(min_weights, mean_returns, cov_matrix)
    min_sharpe = (min_ret - risk_free_rate) / min_vol

    # Efficient Frontier (filter frontier to remove pre-min variance region)
    frontier_df = compute_efficient_frontier(mean_returns, cov_matrix)
    frontier_df = frontier_df[frontier_df['Return'] >= min_ret]

    return {
        "true": (true_ret, true_vol, true_sharpe, true_weights),
        "min": (min_ret, min_vol, min_sharpe, min_weights),
        "frontier": frontier_df
    }
