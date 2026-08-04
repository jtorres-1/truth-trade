# strategies/orb.py — Opening Range Breakout strategy for Truth Trade
#
# Implements: mark the opening range high/low on a 15-min candle, drop to
# 1-min, wait for a breakout, wait for a retest of the broken level, wait
# for a rejection at that retest, enter on the confirmation candle close.
#
# The "retest" and "rejection" steps are inherently a little fuzzy in plain
# English, so they're made explicit and documented here rather than left
# implicit, same principle as tightening any other fuzzy rule before
# automating it:
#
#   RETEST   = after a breakout, price trades back to within
#              RETEST_TOLERANCE_PCT of the broken level.
#   REJECTION = the retest candle's high/low crosses back through the
#               level, but the candle CLOSES back on the breakout side
#               of it (i.e. price touched the level and got rejected,
#               not just gapped through).
#   CONFIRMATION = the very next 1-min candle closes further in the
#                  breakout direction than the rejection candle's close.
#
# This is a reference implementation for validation reports, not a live
# execution bot — it answers "does this strategy hold up," it doesn't
# place real orders.

from backtesting import Strategy
from backtesting.lib import crossover
import pandas as pd


# ─── DEFAULT PARAMETERS ────────────────────────────────────────────────────
# Matches Yash's locked build exactly. Every param is overridable per
# report request, these are just the reference defaults.
DEFAULT_PARAMS = {
    "or_start": "09:30",
    "or_end": "09:45",       # opening range window, ET
    "risk_reward": 3.0,      # 1:3
    "contracts": 3,
    "retest_tolerance_pct": 0.0005,  # 0.05% — adjust per instrument tick size
    "allow_long": True,
    "allow_short": True,
    "daily_retrigger_cap": None,     # None = no cap, matches Yash's spec
}


def compute_opening_range(df: pd.DataFrame, or_start: str, or_end: str):
    """
    Returns a Series aligned to df's index giving that day's OR high/low
    for every bar. Computed once per trading day from the 15-min window,
    then forward-filled across the rest of that day's 1-min bars.
    """
    df = df.copy()
    df["date"] = df.index.date
    df["time"] = df.index.time

    or_mask = (df["time"] >= pd.to_datetime(or_start).time()) & \
              (df["time"] < pd.to_datetime(or_end).time())

    daily_or = df[or_mask].groupby("date").agg(
        or_high=("High", "max"),
        or_low=("Low", "min"),
    )

    df = df.join(daily_or, on="date")
    return df["or_high"], df["or_low"]


class ORBStrategy(Strategy):
    """
    Opening Range Breakout with retest + rejection confirmation entry.
    Designed for backtesting.py. Intended for 1-min OHLCV data covering
    NQ or ES futures, but works on any 1-min series with a real opening
    session.
    """

    # backtesting.py requires params as class attributes to support
    # its built-in optimize() grid search later, if that's ever wanted.
    risk_reward = DEFAULT_PARAMS["risk_reward"]
    contracts = DEFAULT_PARAMS["contracts"]
    retest_tolerance_pct = DEFAULT_PARAMS["retest_tolerance_pct"]
    allow_long = DEFAULT_PARAMS["allow_long"]
    allow_short = DEFAULT_PARAMS["allow_short"]
    or_start = DEFAULT_PARAMS["or_start"]
    or_end = DEFAULT_PARAMS["or_end"]

    def init(self):
        or_high, or_low = compute_opening_range(
            self.data.df, self.or_start, self.or_end
        )
        self.or_high = self.I(lambda: or_high, name="OR High")
        self.or_low = self.I(lambda: or_low, name="OR Low")

        # State tracked bar-to-bar, not vectorized — ORB with a
        # retest/rejection/confirmation sequence is inherently a state
        # machine (waiting_for_retest -> waiting_for_rejection ->
        # waiting_for_confirmation -> entered), not something you can
        # reduce to a single crossover condition.
        self.state = None          # None | "broke_up" | "broke_down"
        self.rejection_seen = False
        self.rejection_close = None
        self.last_trade_date = None
        self.trades_today = 0

    def next(self):
        i = len(self.data) - 1
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        or_high = self.or_high[-1]
        or_low = self.or_low[-1]
        current_date = self.data.index[-1].date()

        if pd.isna(or_high) or pd.isna(or_low):
            return  # before today's OR window has been established

        if current_date != self.last_trade_date:
            self.last_trade_date = current_date
            self.trades_today = 0
            self.state = None
            self.rejection_seen = False

        # Skip if we've hit a daily cap (None means uncapped, per spec)
        if DEFAULT_PARAMS["daily_retrigger_cap"] is not None and \
           self.trades_today >= DEFAULT_PARAMS["daily_retrigger_cap"]:
            return

        # If already in a position, let the bracket order (SL/TP) manage
        # the exit — nothing to do here until it's flat again.
        if self.position:
            return

        tol_up = or_high * (1 + self.retest_tolerance_pct)
        tol_down = or_low * (1 - self.retest_tolerance_pct)

        # STEP 1: detect breakout
        if self.state is None:
            if self.allow_long and price > or_high:
                self.state = "broke_up"
                self.rejection_seen = False
            elif self.allow_short and price < or_low:
                self.state = "broke_down"
                self.rejection_seen = False
            return

        # STEP 2 + 3: retest + rejection
        if self.state == "broke_up" and not self.rejection_seen:
            touched_level = low <= tol_up
            rejected = touched_level and price > or_high
            if rejected:
                self.rejection_seen = True
                self.rejection_close = price
            return

        if self.state == "broke_down" and not self.rejection_seen:
            touched_level = high >= tol_down
            rejected = touched_level and price < or_low
            if rejected:
                self.rejection_seen = True
                self.rejection_close = price
            return

        # STEP 4: confirmation candle — enter here
        if self.state == "broke_up" and self.rejection_seen:
            if price > self.rejection_close:
                sl = low  # stop at the rejection candle's low
                risk = price - sl
                tp = price + risk * self.risk_reward
                self.buy(size=self.contracts, sl=sl, tp=tp)
                self.trades_today += 1
                self.state = None
                self.rejection_seen = False
            return

        if self.state == "broke_down" and self.rejection_seen:
            if price < self.rejection_close:
                sl = high  # stop at the rejection candle's high
                risk = sl - price
                tp = price - risk * self.risk_reward
                self.sell(size=self.contracts, sl=sl, tp=tp)
                self.trades_today += 1
                self.state = None
                self.rejection_seen = False
            return
