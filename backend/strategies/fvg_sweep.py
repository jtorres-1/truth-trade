# strategies/fvg_sweep.py — 4H Liquidity Sweep + Inverse FVG strategy for Truth Trade
#
# Translates Dion's original 7-point entry checklist into explicit,
# documented rules, same principle as the ORB module: don't leave
# discretionary judgment implicit, turn it into a real testable condition.
#
# Dion's original checklist:
#   1. Find a FRESH 4H swept zone (just touched, not old)
#   2. Zoom to 1 minute
#   3. Locate the 20:40 or 50:10 time panel window
#   4. Find an inverse FVG inside that fresh 4H swept zone
#   5. Confirm the inverse FVG also falls inside the time panel window
#   6. Entry at candle close where the IFVG occurred (or inside it)
#   7. Stop loss = size of the inverse FVG or the candle height where it
#      occurred. Risk/reward confirmed 1:1.
#
# Explicit definitions used here, since "liquidity wick," "swept," and
# "fresh" aren't single testable conditions as originally phrased:
#
#   4H LIQUIDITY WICK = a 4H candle whose upper or lower wick is at least
#                        WICK_THRESHOLD points beyond the candle body,
#                        marking a real liquidity pool.
#   SWEPT             = price (on any lower timeframe bar) trades back
#                        into that wick's price range after it formed.
#   FRESH             = the sweep happened within FRESH_WINDOW_BARS of
#                        the current 1-min bar — old sweeps don't count,
#                        matching "just touched, not old."
#   FVG               = a 3-candle gap: candle1.high < candle3.low
#                        (bullish) or candle1.low > candle3.high
#                        (bearish), sized at least FVG_THRESHOLD points.
#   INVERSE FVG       = an existing FVG that price closes back through
#                        in the opposite direction — the gap gets
#                        "disrespected" and flips role.
#   TIME PANEL WINDOW = minute 20-40 or minute 50 through the next
#                        hour's minute 10, per Dion's original spec.

from backtesting import Strategy
import pandas as pd


DEFAULT_PARAMS = {
    "wick_threshold_points": 35.0,     # 4H liquidity wick size, YM default
    "fvg_threshold_points": 1.09,      # FVG/IFVG minimum size, YM default (109 ticks)
    "fresh_window_bars": 60,           # how recent a sweep must be to count as "fresh"
    "risk_reward": 1.0,                # confirmed 1:1
    "contracts": 1,
}


def compute_4h_wicks(df: pd.DataFrame, wick_threshold: float):
    """
    Resamples to 4H bars and flags each one as having a real liquidity
    wick above and/or below, per the WICK_THRESHOLD rule. Returns the
    4H-level wick high/low levels for bars that qualify.
    """
    h4 = df.resample("4h").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last"
    }).dropna()

    body_top = h4[["Open", "Close"]].max(axis=1)
    body_bottom = h4[["Open", "Close"]].min(axis=1)

    upper_wick = h4["High"] - body_top
    lower_wick = body_bottom - h4["Low"]

    h4["liquidity_high"] = h4["High"].where(upper_wick >= wick_threshold)
    h4["liquidity_low"] = h4["Low"].where(lower_wick >= wick_threshold)

    return h4[["liquidity_high", "liquidity_low"]]


def detect_fvg(df: pd.DataFrame, threshold: float):
    """
    Vectorized 3-candle FVG detection across the full 1-min series.
    Returns two boolean Series (bullish_fvg, bearish_fvg) aligned to the
    bar where the gap COMPLETES (the 3rd candle), plus the gap's
    high/low bounds for later inverse-FVG checks.
    """
    high = df["High"]
    low = df["Low"]

    bull_gap_size = low - high.shift(2)
    bull_fvg = bull_gap_size >= threshold

    bear_gap_size = low.shift(2) - high
    bear_fvg = bear_gap_size >= threshold

    bull_fvg_low = high.shift(2)   # bottom of the bullish gap
    bull_fvg_high = low
    bear_fvg_low = low
    bear_fvg_high = low.shift(2)   # top of the bearish gap (approx)

    return {
        "bull_fvg": bull_fvg,
        "bear_fvg": bear_fvg,
        "bull_fvg_low": bull_fvg_low,
        "bull_fvg_high": bull_fvg_high,
        "bear_fvg_low": bear_fvg_low,
        "bear_fvg_high": bear_fvg_high,
    }


def in_time_panel(ts: pd.Timestamp) -> bool:
    """Minute 20-40, or minute 50 through the next hour's minute 10."""
    m = ts.minute
    return (20 <= m < 40) or (m >= 50) or (m < 10)


class FVGSweepStrategy(Strategy):
    """
    4H liquidity sweep + inverse FVG entry, per Dion's original spec.
    Reference implementation for validation reports — answers "does this
    hold up," doesn't place live orders.
    """

    wick_threshold_points = DEFAULT_PARAMS["wick_threshold_points"]
    fvg_threshold_points = DEFAULT_PARAMS["fvg_threshold_points"]
    fresh_window_bars = DEFAULT_PARAMS["fresh_window_bars"]
    risk_reward = DEFAULT_PARAMS["risk_reward"]
    contracts = DEFAULT_PARAMS["contracts"]

    def init(self):
        df = self.data.df

        h4_zones = compute_4h_wicks(df, self.wick_threshold_points)
        h4_zones_reindexed = h4_zones.reindex(df.index, method="ffill")
        self.liq_high = self.I(lambda: h4_zones_reindexed["liquidity_high"], name="4H Liq High")
        self.liq_low = self.I(lambda: h4_zones_reindexed["liquidity_low"], name="4H Liq Low")

        fvg = detect_fvg(df, self.fvg_threshold_points)
        self.bull_fvg = self.I(lambda: fvg["bull_fvg"], name="Bull FVG")
        self.bear_fvg = self.I(lambda: fvg["bear_fvg"], name="Bear FVG")
        self.bull_fvg_low = self.I(lambda: fvg["bull_fvg_low"], name="Bull FVG Low")
        self.bull_fvg_high = self.I(lambda: fvg["bull_fvg_high"], name="Bull FVG High")
        self.bear_fvg_low = self.I(lambda: fvg["bear_fvg_low"], name="Bear FVG Low")
        self.bear_fvg_high = self.I(lambda: fvg["bear_fvg_high"], name="Bear FVG High")

        # Track recent sweeps so "fresh" can be checked bar to bar.
        self.last_high_sweep_bar = None
        self.last_low_sweep_bar = None

    def next(self):
        i = len(self.data) - 1
        if i < 3:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        ts = self.data.index[-1]

        liq_high = self.liq_high[-1]
        liq_low = self.liq_low[-1]

        # STEP: detect a sweep of a 4H liquidity zone happening right now
        if not pd.isna(liq_high) and high >= liq_high:
            self.last_high_sweep_bar = i
        if not pd.isna(liq_low) and low <= liq_low:
            self.last_low_sweep_bar = i

        if self.position:
            return

        fresh_high_sweep = (
            self.last_high_sweep_bar is not None
            and (i - self.last_high_sweep_bar) <= self.fresh_window_bars
        )
        fresh_low_sweep = (
            self.last_low_sweep_bar is not None
            and (i - self.last_low_sweep_bar) <= self.fresh_window_bars
        )

        if not in_time_panel(ts):
            return

        # SHORT setup: fresh high sweep + inverse of a bearish FVG
        # (price closing back below a bearish gap that formed) = the
        # "disrespected" gap flipping into a short signal, inside a
        # fresh swept high zone, inside the time panel window.
        if fresh_high_sweep and self.bear_fvg[-2] and not pd.isna(self.bear_fvg_low[-2]):
            gap_low = self.bear_fvg_low[-2]
            if price < gap_low:  # inverse FVG confirmed: price closed back through it
                sl_size = max(self.bear_fvg_high[-2] - self.bear_fvg_low[-2], high - low)
                sl = price + sl_size
                risk = sl - price
                if risk > 0:
                    tp = price - risk * self.risk_reward
                    self.sell(size=self.contracts, sl=sl, tp=tp)
                    self.last_high_sweep_bar = None
                return

        # LONG setup: fresh low sweep + inverse of a bullish FVG
        if fresh_low_sweep and self.bull_fvg[-2] and not pd.isna(self.bull_fvg_high[-2]):
            gap_high = self.bull_fvg_high[-2]
            if price > gap_high:  # inverse FVG confirmed
                sl_size = max(self.bull_fvg_high[-2] - self.bull_fvg_low[-2], high - low)
                sl = price - sl_size
                risk = price - sl
                if risk > 0:
                    tp = price + risk * self.risk_reward
                    self.buy(size=self.contracts, sl=sl, tp=tp)
                    self.last_low_sweep_bar = None
                return
