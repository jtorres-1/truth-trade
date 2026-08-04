# run_backtest.py — Truth Trade walk-forward test runner
#
# Usage:
#   python run_backtest.py --data path/to/1min_ohlcv.csv --split 0.7
#
# Loads 1-min OHLCV data, splits it chronologically into an in-sample
# window (default 70%) and an out-of-sample window (the remaining 30%),
# runs the strategy on each independently, and prints both results side
# by side. The whole point of Truth Trade is the honest comparison: does
# in-sample performance actually hold up out of sample, or was it a
# short-window illusion, same distinction that mattered for both Yash's
# and Dion's builds.

import argparse
import pandas as pd
from backtesting import Backtest

from strategies.orb import ORBStrategy, DEFAULT_PARAMS


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Data file is missing required columns: {missing}")
    return df.sort_index()


def split_data(df: pd.DataFrame, split_ratio: float):
    cutoff = int(len(df) * split_ratio)
    in_sample = df.iloc[:cutoff]
    out_of_sample = df.iloc[cutoff:]
    return in_sample, out_of_sample


def run_single_backtest(df: pd.DataFrame, cash: float = 50000, commission: float = 0.0002, margin: float = 0.05):
    # margin=0.05 tells backtesting.py this is a leveraged/futures
    # instrument, only ~5% of notional is required as buying power per
    # contract, not the full price. Without this, backtesting.py treats
    # every unit of `size` as a full-price stock purchase and silently
    # cancels every order once notional exceeds cash. 0.05 is a rough
    # placeholder — real NQ/ES day-trading margin requirements vary by
    # broker/prop firm and should be set to the actual figure once known,
    # not left at this default for a real report.
    bt = Backtest(df, ORBStrategy, cash=cash, commission=commission, margin=margin, exclusive_orders=True)
    stats = bt.run(finalize_trades=True)
    return stats


def summarize(stats, label: str):
    print(f"\n--- {label} ---")
    print(f"Trades:        {stats['# Trades']}")
    print(f"Win Rate:      {stats['Win Rate [%]']:.2f}%")
    print(f"Profit Factor: {stats.get('Profit Factor', float('nan')):.3f}")
    print(f"Max Drawdown:  {stats['Max. Drawdown [%]']:.2f}%")
    print(f"Return:        {stats['Return [%]']:.2f}%")
    return {
        "trades": int(stats["# Trades"]),
        "win_rate_pct": float(stats["Win Rate [%]"]),
        "profit_factor": float(stats.get("Profit Factor", float("nan"))),
        "max_drawdown_pct": float(stats["Max. Drawdown [%]"]),
        "return_pct": float(stats["Return [%]"]),
    }


def held_up_out_of_sample(in_sample_stats, out_of_sample_stats) -> bool:
    """
    The honest verdict Truth Trade reports actually sell: did it hold up,
    or was the in-sample result a short-window illusion. Simple rule for
    v1 — real profit factor above 1.0 out of sample, not just in-sample.
    """
    return out_of_sample_stats["profit_factor"] > 1.0


def main():
    parser = argparse.ArgumentParser(description="Truth Trade — ORB walk-forward validation")
    parser.add_argument("--data", required=True, help="Path to 1-min OHLCV CSV")
    parser.add_argument("--split", type=float, default=0.7, help="In-sample fraction (default 0.7)")
    args = parser.parse_args()

    df = load_data(args.data)
    in_sample_df, out_of_sample_df = split_data(df, args.split)

    print(f"Loaded {len(df)} bars, {in_sample_df.index[0]} to {df.index[-1]}")
    print(f"In-sample:     {in_sample_df.index[0]} to {in_sample_df.index[-1]} ({len(in_sample_df)} bars)")
    print(f"Out-of-sample: {out_of_sample_df.index[0]} to {out_of_sample_df.index[-1]} ({len(out_of_sample_df)} bars)")

    in_sample_stats = run_single_backtest(in_sample_df)
    out_of_sample_stats = run_single_backtest(out_of_sample_df)

    in_summary = summarize(in_sample_stats, "IN-SAMPLE")
    out_summary = summarize(out_of_sample_stats, "OUT-OF-SAMPLE")

    verdict = held_up_out_of_sample(in_summary, out_summary)
    print(f"\n=== VERDICT: {'HELD UP' if verdict else 'DID NOT HOLD UP'} out of sample ===\n")


if __name__ == "__main__":
    main()
