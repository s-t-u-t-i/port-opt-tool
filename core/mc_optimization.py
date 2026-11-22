# The three summary metric columns of a simulation result; everything else is
# a per-ticker weight column.
METRIC_COLUMNS = ("Return", "Volatility", "Sharpe Ratio")


def get_mc_optimal_portfolio(portfolios_df, weight_columns=None):
    """
    Identifies the optimal portfolio from the Monte Carlo simulation results,
    based on the maximum Sharpe ratio.

    Parameters:
    - portfolios_df (DataFrame): DataFrame containing portfolio return, volatility,
                                 Sharpe ratio, and weights from simulation.
    - weight_columns (list, optional): The ticker columns that hold portfolio
                                 weights. When omitted, every column that is
                                 not one of the three metric columns is treated
                                 as a weight column (name-based, so inserting
                                 or reordering metric columns can't corrupt it).

    Returns:
    - Tuple:
        - opt_return (float): Expected annual return of the optimal portfolio.
        - opt_volatility (float): Annual volatility of the optimal portfolio.
        - opt_sharpe (float): Sharpe ratio of the optimal portfolio.
        - opt_weights (ndarray): Asset weights of the optimal portfolio.

    Raises:
    - ValueError: With an actionable message when the simulation produced
      nothing usable (empty input, or every Sharpe ratio is NaN — usually
      unusable price data upstream).
    """
    if portfolios_df is None or len(portfolios_df) == 0:
        raise ValueError(
            "No portfolios to optimize: the simulation DataFrame is empty."
        )
    if not portfolios_df['Sharpe Ratio'].notna().any():
        raise ValueError(
            "No valid portfolios to optimize: every Sharpe ratio is NaN. "
            "This usually means the input price data was unusable (bad "
            "tickers, bad dates, or missing values)."
        )

    # Find the index of the portfolio with the highest Sharpe ratio
    max_sharpe_idx = portfolios_df['Sharpe Ratio'].idxmax()

    # Extract the entire row for the optimal portfolio
    opt_port = portfolios_df.loc[max_sharpe_idx]

    # Extract relevant metrics
    opt_return = opt_port['Return']
    opt_volatility = opt_port['Volatility']
    opt_sharpe = float(opt_port['Sharpe Ratio'])  # convert to scalar float

    # Weights are identified BY NAME, never by position.
    if weight_columns is None:
        weight_columns = [c for c in portfolios_df.columns if c not in METRIC_COLUMNS]
    opt_weights = opt_port[list(weight_columns)].to_numpy(dtype=float)

    return opt_return, opt_volatility, opt_sharpe, opt_weights


def align_weight_vector(weights, source_order, target_order):
    """
    Reorders a weight vector from one ticker order to another.

    Both orders must contain exactly the same tickers. This lets the app keep
    a simulation alive when the user re-selects the same tickers in a
    different order (a set-equality check), while still displaying every
    weight against the right ticker.

    Parameters:
    - weights (sequence): Weights in `source_order` sequence.
    - source_order (list): Ticker order the weights currently follow.
    - target_order (list): Ticker order to rearrange into.

    Returns:
    - list: Weights in `target_order` sequence.
    """
    source_order = list(source_order)
    target_order = list(target_order)
    if set(source_order) != set(target_order):
        raise ValueError(
            "align_weight_vector requires both ticker orders to contain the "
            "same tickers."
        )
    return [float(weights[source_order.index(ticker)]) for ticker in target_order]
