def get_mc_optimal_portfolio(portfolios_df):
    """
    Identifies the optimal portfolio from the Monte Carlo simulation results,
    based on the maximum Sharpe ratio.

    Parameters:
    - portfolios_df (DataFrame): DataFrame containing portfolio return, volatility,
                                 Sharpe ratio, and weights from simulation.

    Returns:
    - Tuple:
        - opt_return (float): Expected annual return of the optimal portfolio.
        - opt_volatility (float): Annual volatility of the optimal portfolio.
        - opt_sharpe (float): Sharpe ratio of the optimal portfolio.
        - opt_weights (ndarray): Asset weights of the optimal portfolio.
    """
    # Find the index of the portfolio with the highest Sharpe ratio
    max_sharpe_idx = portfolios_df['Sharpe Ratio'].idxmax()

    # Extract the entire row for the optimal portfolio
    opt_port = portfolios_df.loc[max_sharpe_idx]

    # Extract relevant metrics
    opt_return = opt_port['Return']
    opt_volatility = opt_port['Volatility']
    opt_sharpe = float(opt_port['Sharpe Ratio'])  # convert to scalar float
    opt_weights = opt_port[portfolios_df.columns[3:]].values  # weights start from column 4 onward

    return opt_return, opt_volatility, opt_sharpe, opt_weights

