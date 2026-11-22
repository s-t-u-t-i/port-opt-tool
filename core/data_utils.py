import os
import warnings

import yfinance as yf
import pandas as pd
import numpy as np
from curl_cffi import requests

def fetch_and_save_data(tickers, start_date, end_date, filename='data/stock_data.csv'):
    """
    Downloads historical stock data using yfinance and saves it as a CSV.

    Parameters:
    - tickers (list): List of stock ticker symbols.
    - start_date (str): Start date for data download (YYYY-MM-DD).
    - end_date (str): End date for data download (YYYY-MM-DD).
    - filename (str): Path to save the downloaded CSV.

    Returns:
    - DataFrame: Raw downloaded data (multi-index columns).

    Raises:
    - ValueError: If the download comes back empty. The existing CSV file
      is left untouched in that case, so one failed download can never
      destroy previously-good local data.
    """
    # Create the data directory if it doesn't exist (skip for a bare filename,
    # where os.path.dirname is '' and os.makedirs('') would crash).
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Create a session impersonating a Chrome browser (to avoid blocking by Yahoo)
    session = requests.Session(impersonate="chrome")

    # Download historical data for given tickers and date range
    data = yf.download(tickers, start=start_date, end=end_date, session=session)

    # Guard: yfinance reports blocked/failed downloads by returning an EMPTY
    # frame (it does not raise). Refuse to overwrite a good CSV with nothing.
    if data is None or len(data) == 0:
        raise ValueError(
            f"Download returned no data for {list(tickers)} between "
            f"{start_date} and {end_date}. Check the ticker symbols, the date "
            f"order (start must be before end), and your internet connection. "
            f"The existing file '{filename}' was left untouched."
        )

    # Save the complete raw dataset to CSV
    data.to_csv(filename)

    return data


def load_close_prices_from_csv(filename):
    """
    Loads only the 'Close' prices from a CSV with multi-level columns.

    Parameters:
    - filename (str): Path to the CSV file containing raw price data.

    Returns:
    - DataFrame: DataFrame containing only closing prices of each asset.
    """
    # Load CSV with multi-index columns (e.g., 'Open', 'Close', etc.)
    data = pd.read_csv(filename, header=[0, 1], index_col=0, parse_dates=True)

    # Extract the 'Close' level from the multi-index columns
    close_prices = data['Close']

    return close_prices


def calculate_log_returns(close_prices, output_csv='data/Daily_log_returns.csv'):
    """
    Calculates daily log returns from closing prices and saves to CSV.

    Robustness rules (so one broken ticker can't silently corrupt the rest):
    1. Tickers with NO price data at all (bogus symbols, never-listed) are
       dropped, with a warning — they can no longer wipe every row.
    2. Non-positive prices (0 or negative, from data glitches) are treated as
       missing values instead of producing ±inf via log().
    3. Only rows where ALL remaining tickers have data are kept (the overlap
       of their trading days). If that discards real history, a warning says so.

    Parameters:
    - close_prices (DataFrame): DataFrame of closing prices.
    - output_csv (str): Path to save the log returns.

    Returns:
    - DataFrame: Daily log returns for each asset.
    """
    prices = close_prices.copy()

    # 1) Drop tickers that have no usable data whatsoever.
    dead_tickers = prices.columns[prices.isna().all()]
    usable = prices.dropna(axis=1, how='all')
    if len(dead_tickers) > 0:
        warnings.warn(
            f"No price data for {list(dead_tickers)}; these tickers were "
            f"excluded and the analysis continues with the remaining ones.",
            UserWarning,
            stacklevel=2,
        )

    # 2) Treat non-positive prices as missing rather than letting log()
    #    generate ±inf that silently poisons the covariance matrix.
    non_positive = (usable <= 0) & usable.notna()
    if non_positive.to_numpy().any():
        warnings.warn(
            "Non-positive prices were found and treated as missing data.",
            UserWarning,
            stacklevel=2,
        )
        usable = usable.mask(usable <= 0)

    # Compute natural log of relative price changes
    log_returns = np.log(usable / usable.shift(1))

    # 3) Keep only days where every surviving ticker has data, and warn when
    #    that alignment actually discards history (delistings, gaps).
    before_rows = len(log_returns)
    log_returns = log_returns.dropna()
    if len(log_returns) < len(prices) - 1:
        discarded = len(prices) - 1 - len(log_returns)
        warnings.warn(
            f"Only the first {len(log_returns)} trading day(s) where ALL "
            f"tickers have data are used; {discarded} day(s) of history were "
            f"discarded to keep the tickers aligned (a delisted or late-listed "
            f"ticker shortens the analysis for everyone).",
            UserWarning,
            stacklevel=2,
        )

    # Save the computed log returns to CSV
    log_returns.to_csv(output_csv)

    return log_returns


def calculate_descriptive_statistics(log_returns, output_csv='data/descriptive_stats.csv'):
    """
    Computes basic descriptive statistics for log returns and saves to CSV.

    Parameters:
    - log_returns (DataFrame): DataFrame of daily log returns.
    - output_csv (str): Path to save the descriptive statistics.

    Returns:
    - DataFrame: Summary statistics including mean, std, skewness, etc.
    """
    # Calculate statistical metrics for each asset
    summary_stats = {
        'Mean': log_returns.mean(),
        'Median': log_returns.median(),
        'Min': log_returns.min(),
        'Max': log_returns.max(),
        'Std Dev': log_returns.std(),
        '1st Quartile (Q1)': log_returns.quantile(0.25),
        '3rd Quartile (Q3)': log_returns.quantile(0.75),
        'Skewness': log_returns.skew(),
        'Kurtosis': log_returns.kurt()
    }

    # Convert dictionary of Series to DataFrame
    summary_df = pd.DataFrame(summary_stats)

    # Save descriptive statistics to CSV
    summary_df.to_csv(output_csv)

    return summary_df


def compute_cov_corr_matrices(log_returns, cov_output='data/cov_matrix.csv', corr_output='data/corr_matrix.csv'):
    """
    Computes and saves the covariance and correlation matrices of returns.

    Parameters:
    - log_returns (DataFrame): DataFrame of daily log returns.
    - cov_output (str): Path to save the covariance matrix.
    - corr_output (str): Path to save the correlation matrix.

    Returns:
    - Tuple (DataFrame, DataFrame): Covariance matrix, Correlation matrix.
    """
    # Compute sample covariance matrix
    cov_matrix = log_returns.cov()

    # Compute correlation matrix
    corr_matrix = log_returns.corr()

    # Save both matrices to CSV files
    cov_matrix.to_csv(cov_output)
    corr_matrix.to_csv(corr_output)

    return cov_matrix, corr_matrix


def calculate_individual_sharpe_ratios(summary_df, risk_free_rate=0.05, output_csv='data/ind_sharpe_ratios.csv'):
    """
    Calculates annualized Sharpe ratios for individual assets based on daily stats.

    Parameters:
    - summary_df (DataFrame): Descriptive statistics DataFrame.
    - risk_free_rate (float): Annual risk-free rate for Sharpe calculation.
    - output_csv (str): Path to save Sharpe ratios.

    Returns:
    - Tuple: (Series of Sharpe ratios, Series of annual returns, Series of annual std devs)
    """
    # Convert daily mean return to annualized return (assuming 252 trading days)
    annual_mean_returns = summary_df['Mean'] * 252

    # Convert daily standard deviation to annualized volatility
    annual_std = summary_df['Std Dev'] * (252 ** 0.5)

    # A zero-volatility asset (flat prices) cannot have a meaningful Sharpe
    # ratio — return NaN instead of ±inf so charts and tables stay sane.
    zero_vol_assets = annual_std.index[annual_std == 0].tolist()
    if zero_vol_assets:
        warnings.warn(
            f"No Sharpe ratio for zero-volatility asset(s) {zero_vol_assets} "
            f"(their prices never changed).",
            UserWarning,
            stacklevel=2,
        )

    sharpe_ratios = (annual_mean_returns - risk_free_rate) / annual_std.replace(0, np.nan)

    # Save Sharpe ratios to CSV
    sharpe_ratios.to_csv(output_csv)

    return sharpe_ratios, annual_mean_returns, annual_std
