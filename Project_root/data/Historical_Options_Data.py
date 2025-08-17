import requests
import csv
import os
import time
import random
from collections import deque
from datetime import datetime, timedelta, date, timezone
from multiprocessing import Pool, cpu_count, Manager, Lock
import sys  # for clean exit

BASE_URL     = "https://api.delta.exchange/v2"
WEIGHT_QUOTA = 10000       # API weight limit
CALL_WEIGHT  = 3           # weight per request
INC          = 500         # strike increment
PCT_BAND     = 0.10        # strike band around daily OHLC
MAX_RETRIES  = 5
ERROR_LOG    = "api_errors.csv"

manager = Manager()
request_log = manager.list()
log_lock = Lock()

def check_quota(weight: int):
    """Enforce API weight quota in a rolling 300s window (multiprocessing-safe)."""
    global request_log, log_lock
    while True:
        now = time.time()
        with log_lock:
            while request_log and request_log[0][0] < now - 300:
                request_log.pop(0)
            current_weight = sum(w for _, w in request_log)
            if current_weight + weight <= WEIGHT_QUOTA:
                request_log.append((now, weight))
                return
            earliest_timestamp, _ = request_log[0]
            sleep_time = max((earliest_timestamp + 300) - now, 0.1)
            print(f"[PAUSE] Quota {current_weight}/{WEIGHT_QUOTA} reached. Sleeping {sleep_time:.1f}s")
        time.sleep(sleep_time)

def request_with_retry(url: str, params: dict = None, retries: int = MAX_RETRIES) -> dict:
    """Wrapper around requests.get with retry + exponential backoff."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            wait_time = 2 ** attempt + random.uniform(0, 1)
            print(f"[RETRY] Attempt {attempt+1}/{retries} failed for {url} with {e}. Retrying in {wait_time:.1f}s")
            time.sleep(wait_time)
    return None

def log_error(context: str, symbol: str, target_date: date, error_msg: str):
    """Append error details to error log CSV."""
    file_exists = os.path.exists(ERROR_LOG)
    with open(ERROR_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "context", "symbol", "date", "error"])
        writer.writerow([
            datetime.utcnow().isoformat(),
            context,
            symbol,
            target_date.strftime("%Y-%m-%d") if target_date else "",
            error_msg
        ])

def fetch_all_products(retries: int = MAX_RETRIES) -> list:
    """Fetch the list of all products from the API with retry + logging."""
    url = f"{BASE_URL}/products"
    for attempt in range(retries):
        try:
            check_quota(CALL_WEIGHT)
            resp = requests.get(url, headers={"Accept": "application/json"}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("result", [])
        except Exception as e:
            wait_time = 2 ** attempt + random.uniform(0, 1)
            print(f"[RETRY] fetch_all_products attempt {attempt+1}/{retries} failed: {e}. Retrying in {wait_time:.1f}s")
            time.sleep(wait_time)
    log_error("fetch_all_products", "ALL", None, f"Max retries exceeded: {url}")
    print(f"[ERROR] Failed to fetch products after {retries} attempts. Exiting.")
    sys.exit(1)

def fetch_futures_ohlc(symbol: str, target_date: date) -> dict:
    check_quota(CALL_WEIGHT)
    start_dt = datetime(target_date.year, target_date.month, target_date.day, 0, 0, tzinfo=timezone.utc)
    end_dt   = start_dt + timedelta(days=1)
    start_ts, end_ts = int(start_dt.timestamp()), int(end_dt.timestamp())
    url = f"{BASE_URL}/history/candles"
    params = {"symbol": symbol, "resolution": "1d", "start": str(start_ts), "end": str(end_ts)}
    data = request_with_retry(url, params)
    if not data:
        log_error("futures_ohlc", symbol, target_date, "Max retries exceeded")
        return None
    bars = data.get("result", [])
    if not bars:
        log_error("futures_ohlc", symbol, target_date, "No 1d bar returned")
        return None
    return bars[0]

def fetch_option_candles(symbol: str, target_date: date) -> list:
    check_quota(CALL_WEIGHT)
    start_dt = datetime(target_date.year, target_date.month, target_date.day, 0, 0, tzinfo=timezone.utc)
    end_dt   = start_dt + timedelta(days=1)
    start_ts, end_ts = int(start_dt.timestamp()), int(end_dt.timestamp())
    url = f"{BASE_URL}/history/candles"
    params = {"symbol": symbol, "resolution": "5m", "start": str(start_ts), "end": str(end_ts)}
    data = request_with_retry(url, params)
    if not data:
        log_error("option_candles", symbol, target_date, "Max retries exceeded")
        return []
    return data.get("result", [])

def generate_strike_grid_from_ohlc(low: float, high: float, pct_band: float = PCT_BAND) -> list:
    raw_min, raw_max = low * (1 - pct_band), high * (1 + pct_band)
    min_strike = int(round(raw_min / INC) * INC)
    max_strike = int(round(raw_max / INC) * INC)
    if min_strike < INC:
        min_strike = INC
    return list(range(min_strike, max_strike + INC, INC))

def process_single_day(args):
    current, underlying, expiry_offsets = args
    date_label = current.strftime("%d-%m-%Y")
    os.makedirs(date_label, exist_ok=True)
    try:
        ohlc = fetch_futures_ohlc(f"{underlying}USD", current)
        if not ohlc:
            return
        low_price, high_price = ohlc["low"], ohlc["high"]
    except Exception as e:
        log_error("process_single_day", f"{underlying}USD", current, str(e))
        return
    candidate_strikes = generate_strike_grid_from_ohlc(low_price, high_price)
    expiries = [current + timedelta(days=offset) for offset in expiry_offsets]
    calls_bucket = {exp.strftime("%d-%m-%Y"): [] for exp in expiries}
    puts_bucket = {exp.strftime("%d-%m-%Y"): [] for exp in expiries}
    for exp in expiries:
        exp_label  = exp.strftime("%d-%m-%Y")
        exp_suffix = exp.strftime("%d%m%y")
        for strike in candidate_strikes:
            for prefix, bucket in [("MARK:C", calls_bucket), ("MARK:P", puts_bucket)]:
                symbol = f"{prefix}-{underlying}-{int(strike)}-{exp_suffix}"
                bars   = fetch_option_candles(symbol, current)
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
    with Pool(processes=max(1, cpu_count() // 2)) as pool:
        pool.map(process_single_day, args)

if __name__ == "__main__":
    products = fetch_all_products()

    start          = date(2025, 5, 21)
    end            = date(2025, 6, 20)
    underlying     = "BTC"
    expiry_offsets = [0, 1, 2, 3]

    backfill_using_low_high(start, end, underlying, expiry_offsets)
