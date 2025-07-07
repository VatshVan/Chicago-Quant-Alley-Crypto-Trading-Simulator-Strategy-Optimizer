import requests
import csv
import os
import time
from datetime import datetime, timedelta, date, timezone
from multiprocessing import Pool, cpu_count

BASE_URL     = "https://api.delta.exchange/v2"
WEIGHT_QUOTA = 10000
CALL_WEIGHT  = 3
INC          = 500
PCT_BAND     = 0.10

def fetch_futures_ohlc(symbol: str, target_date: date) -> dict:
    start_dt = datetime(target_date.year, target_date.month, target_date.day, 0, 0, tzinfo=timezone.utc)
    end_dt   = start_dt + timedelta(days=1)
    start_ts = int(start_dt.timestamp())
    end_ts   = int(end_dt.timestamp())

    resp = requests.get(
        f"{BASE_URL}/history/candles",
        params={
            "symbol": symbol,
            "resolution": "1d",
            "start": str(start_ts),
            "end": str(end_ts),
        },
        headers={"Accept": "application/json"}
    )
    resp.raise_for_status()
    bars = resp.json().get("result", [])
    if not bars:
        raise ValueError(f"No 1d bar {symbol} and {target_date}")
    return bars[0]

def generate_strike_grid_from_ohlc(low: float, high: float, pct_band: float = PCT_BAND) -> list:
    raw_min = low * (1 - pct_band)
    raw_max = high * (1 + pct_band)

    min_strike = int(round(raw_min / INC) * INC)
    max_strike = int(round(raw_max / INC) * INC)

    if min_strike < INC:
        min_strike = INC
    return list(range(min_strike, max_strike + INC, INC))

def fetch_option_candles(symbol: str, target_date: date) -> list:
    start_dt = datetime(target_date.year, target_date.month, target_date.day, 0, 0, tzinfo=timezone.utc)
    end_dt   = start_dt + timedelta(days=1)
    start_ts = int(start_dt.timestamp())
    end_ts   = int(end_dt.timestamp())

    resp = requests.get(
        f"{BASE_URL}/history/candles",
        params={
            "symbol":     symbol,
            "resolution": "5m",
            "start":      str(start_ts),
            "end":        str(end_ts),
        },
        headers={"Accept": "application/json"}
    )
    if resp.status_code != 200:
        return []
    return resp.json().get("result", [])

def process_single_day(args):
    current, underlying, expiry_offsets = args
    window_start = time.time()
    weight_used  = 0

    date_label = current.strftime("%d-%m-%Y")
    os.makedirs(date_label, exist_ok=True)

    try:
        ohlc = fetch_futures_ohlc(f"{underlying}USD", current)
        low_price  = ohlc["low"]
        high_price = ohlc["high"]
    except Exception as e:
        print(f"[ERROR] fetch Failed {underlying}USD on {current}: {e}")
        return

    candidate_strikes = generate_strike_grid_from_ohlc(low_price, high_price)
    expiries     = [current + timedelta(days=offset) for offset in expiry_offsets]
    calls_bucket = {exp.strftime("%d-%m-%Y"): [] for exp in expiries}
    puts_bucket  = {exp.strftime("%d-%m-%Y"): [] for exp in expiries}

    for exp in expiries:
        exp_label  = exp.strftime("%d-%m-%Y")
        exp_suffix = exp.strftime("%d%m%y")
        for strike in candidate_strikes:
            for prefix, bucket in [("MARK:C", calls_bucket), ("MARK:P", puts_bucket)]:
                now     = time.time()
                elapsed = now - window_start
                if elapsed >= 300:
                    window_start = now
                    weight_used  = 0
                if weight_used + CALL_WEIGHT > WEIGHT_QUOTA:
                    sleep_time = 300 - elapsed
                    print(f"[PAUSE] Weight {weight_used}/{WEIGHT_QUOTA} reached on {current}. Sleeping {sleep_time:.1f}s.")
                    time.sleep(sleep_time)
                    window_start = time.time()
                    weight_used  = 0

                symbol = f"{prefix}-{underlying}-{int(strike)}-{exp_suffix}"
                bars   = fetch_option_candles(symbol, current)
                weight_used += CALL_WEIGHT
                if not bars:
                    continue
                for b in bars:
                    bucket[exp_label].append({
                        "symbol": symbol,
                        "time":   b["time"],
                        "open":   b["open"],
                        "high":   b["high"],
                        "low":    b["low"],
                        "close":  b["close"],
                        "volume": b["volume"],
                    })

    fieldnames = ["symbol", "time", "open", "high", "low", "close", "volume"]
    for exp_label, records in calls_bucket.items():
        subfolder = os.path.join(date_label, exp_label)
        os.makedirs(subfolder, exist_ok=True)
        calls_path = os.path.join(subfolder, "calls.csv")
        with open(calls_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    for exp_label, records in puts_bucket.items():
        subfolder = os.path.join(date_label, exp_label)
        os.makedirs(subfolder, exist_ok=True)
        puts_path = os.path.join(subfolder, "puts.csv")
        with open(puts_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

def backfill_using_low_high(start_date: date, end_date: date, underlying: str, expiry_offsets: list):
    all_days = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    args = [(d, underlying, expiry_offsets) for d in all_days]

    # Use half of logical CPUs to prevent overwhelming API/network
    with Pool(processes=max(1, cpu_count() // 2)) as pool:
        pool.map(process_single_day, args)

if __name__ == "__main__":
    start          = date(2025, 5, 21)
    end            = date(2025, 6, 20)
    underlying     = "BTC"
    expiry_offsets = [0, 1, 2, 3]

    backfill_using_low_high(start, end, underlying, expiry_offsets)
