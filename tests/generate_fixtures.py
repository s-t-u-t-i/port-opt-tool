"""
One-off generator for the static fixture CSVs in tests/data/.

Run from the repo root:
    .venv/Scripts/python.exe tests/generate_fixtures.py

Outputs (all deterministic, seed=42):
- valid_stock_data.csv      : 5 tickers, yfinance 2-row-header format
- one_ticker_stock_data.csv : a single-ticker download (for the 1-ticker bug)
- malformed_no_close.csv    : same shape but WITHOUT the 'Close' field
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import SYNTHETIC_TICKERS, make_prices, tests_dir, write_yf_csv  # noqa: E402

OUT = tests_dir() / "data"
OUT.mkdir(parents=True, exist_ok=True)

prices = make_prices(n_days=250, seed=42, tickers=SYNTHETIC_TICKERS)

# 1) Valid multi-ticker CSV in yfinance format
write_yf_csv(OUT / "valid_stock_data.csv", prices)
print("wrote valid_stock_data.csv")

# 2) Single-ticker CSV (yfinance format, only AAA.NS)
write_yf_csv(OUT / "one_ticker_stock_data.csv", prices[["AAA.NS"]])
print("wrote one_ticker_stock_data.csv")

# 3) Malformed CSV: real structure but no 'Close' field at all
write_yf_csv(
    OUT / "malformed_no_close.csv",
    prices,
    fields=("Open", "High", "Low", "Volume"),
)
print("wrote malformed_no_close.csv")
