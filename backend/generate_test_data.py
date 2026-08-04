import pandas as pd
import numpy as np

# Generates synthetic 1-min OHLCV data across several trading days, each
# with a real 9:30-9:45 ET opening range and price action shaped to
# actually trigger the ORB breakout -> retest -> rejection -> confirmation
# sequence at least some of the time. This is NOT meant to produce
# realistic strategy performance numbers — it exists purely to prove the
# pipeline (data loading, OR calculation, state machine, entries, exits,
# walk-forward split, report generation) runs without errors before real
# historical NQ/ES data is sourced.

np.random.seed(42)

def generate_day(date, base_price, session_minutes=390):
    """One trading day, 9:30am to 4:00pm ET, 1-min bars."""
    start = pd.Timestamp(f"{date} 09:30:00")
    timestamps = pd.date_range(start, periods=session_minutes, freq="1min")

    prices = [base_price]
    for i in range(1, session_minutes):
        # Small random walk for most of the day
        drift = np.random.normal(0, 0.6)
        # After minute 15 (end of OR), occasionally inject a directional
        # push so a breakout actually happens some days
        if i == 20:
            drift += np.random.choice([-8, 8]) if np.random.rand() < 0.4 else 0
        prices.append(prices[-1] + drift)

    closes = np.array(prices)
    opens = np.roll(closes, 1)
    opens[0] = closes[0] - np.random.normal(0, 0.3)

    highs = np.maximum(opens, closes) + np.abs(np.random.normal(0.5, 0.4, session_minutes))
    lows = np.minimum(opens, closes) - np.abs(np.random.normal(0.5, 0.4, session_minutes))
    volumes = np.random.randint(200, 2000, session_minutes)

    return pd.DataFrame({
        "Date": timestamps,
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    })


def generate_dataset(start_date="2026-01-05", num_days=40, base_price=18000.0):
    all_days = []
    current_price = base_price
    date = pd.Timestamp(start_date)

    days_added = 0
    while days_added < num_days:
        if date.weekday() < 5:  # skip weekends
            day_df = generate_day(date.strftime("%Y-%m-%d"), current_price)
            current_price = day_df["Close"].iloc[-1]  # carry price forward
            all_days.append(day_df)
            days_added += 1
        date += pd.Timedelta(days=1)

    full = pd.concat(all_days, ignore_index=True)
    full = full.set_index("Date")
    return full


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("test_data.csv")
    print(f"Wrote {len(df)} bars across ~{len(df) // 390} trading days to test_data.csv")
    print(df.head())
