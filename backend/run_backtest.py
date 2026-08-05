# run_backtest.py — Truth Trade walk-forward test runner
#
# Usage:
#   python run_backtest.py --data path/to/1min_ohlcv.csv --split 0.7 --strategy orb
#   python run_backtest.py --data path/to/1min_ohlcv.csv --split 0.7 --strategy fvg_sweep --cost-per-trade 1.5

import argparse
import pandas as pd
from backtesting import Backtest

from strategies.orb import ORBStrategy
from strategies.fvg_sweep import FVGSweepStrategy

STRATEGIES = {
    "orb": ORBStrategy,
    "fvg_sweep": FVGSweepStrategy,
}

DEFAULT_COST_PER_TRADE_POINTS = 1.5


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


def run_single_backtest(df: pd.DataFrame, strategy_cls, cash: float = 50000, margin: float = 0.05):
    bt = Backtest(df, strategy_cls, cash=cash, commission=0.0, margin=margin, exclusive_orders=True)
    stats = bt.run()
    return stats


def apply_costs_and_summarize(stats, label: str, cost_per_trade_points: float, point_value: float = 1.0):
    trades = stats["_trades"].copy()
    if len(trades) == 0:
        print(f"\n--- {label} ---")
        print("Trades:        0 (no trades to evaluate)")
        return {"trades": 0, "win_rate_pct": float("nan"), "profit_factor": float("nan"),
                "max_drawdown_pct": float(stats["Max. Drawdown [%]"])}

    trades["AdjPnL"] = trades["PnL"] - (cost_per_trade_points * point_value * trades["Size"].abs())

    wins = trades[trades["AdjPnL"] > 0]
    losses = trades[trades["AdjPnL"] <= 0]

    win_rate = 100 * len(wins) / len(trades)
    gross_win = wins["AdjPnL"].sum()
    gross_loss = abs(losses["AdjPnL"].sum())
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    net_pnl = trades["AdjPnL"].sum()

    print(f"\n--- {label} (after {cost_per_trade_points} pt/trade cost) ---")
    print(f"Trades:        {len(trades)}")
    print(f"Win Rate:      {win_rate:.2f}%")
    print(f"Profit Factor: {profit_factor:.3f}")
    print(f"Avg Win:       {wins['AdjPnL'].mean() if len(wins) else 0:.2f}")
    print(f"Avg Loss:      {losses['AdjPnL'].mean() if len(losses) else 0:.2f}")
    print(f"Net PnL (pts): {net_pnl:.2f}")
    print(f"Max Drawdown:  {stats['Max. Drawdown [%]']:.2f}% (pre-cost, from backtesting.py)")

    return {
        "trades": len(trades),
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "net_pnl_points": net_pnl,
        "max_drawdown_pct": float(stats["Max. Drawdown [%]"]),
    }


def held_up_out_of_sample(out_of_sample_stats) -> bool:
    return out_of_sample_stats["profit_factor"] > 1.0


def main():
    parser = argparse.ArgumentParser(description="Truth Trade — walk-forward strategy validation")
    parser.add_argument("--data", required=True, help="Path to 1-min OHLCV CSV")
    parser.add_argument("--split", type=float, default=0.7, help="In-sample fraction (default 0.7)")
    parser.add_argument("--strategy", choices=list(STRATEGIES.keys()), default="orb",
                         help="Which strategy to test (default: orb)")
    parser.add_argument("--cost-per-trade", type=float, default=DEFAULT_COST_PER_TRADE_POINTS,
                         help=f"Fixed cost per trade in points, commission + slippage (default {DEFAULT_COST_PER_TRADE_POINTS})")
    args = parser.parse_args()

    strategy_cls = STRATEGIES[args.strategy]

    df = load_data(args.data)
    in_sample_df, out_of_sample_df = split_data(df, args.split)

    print(f"Strategy:      {args.strategy}")
    print(f"Cost/trade:    {args.cost_per_trade} points (placeholder — set to real commission+slippage)")
    print(f"Loaded {len(df)} bars, {in_sample_df.index[0]} to {df.index[-1]}")
    print(f"In-sample:     {in_sample_df.index[0]} to {in_sample_df.index[-1]} ({len(in_sample_df)} bars)")
    print(f"Out-of-sample: {out_of_sample_df.index[0]} to {out_of_sample_df.index[-1]} ({len(out_of_sample_df)} bars)")

    in_sample_raw = run_single_backtest(in_sample_df, strategy_cls)
    out_of_sample_raw = run_single_backtest(out_of_sample_df, strategy_cls)

    in_summary = apply_costs_and_summarize(in_sample_raw, "IN-SAMPLE", args.cost_per_trade)
    out_summary = apply_costs_and_summarize(out_of_sample_raw, "OUT-OF-SAMPLE", args.cost_per_trade)

    verdict = held_up_out_of_sample(out_summary)
    print(f"\n=== VERDICT: {'HELD UP' if verdict else 'DID NOT HOLD UP'} out of sample (after realistic costs) ===\n")


if __name__ == "__main__":
    main()
