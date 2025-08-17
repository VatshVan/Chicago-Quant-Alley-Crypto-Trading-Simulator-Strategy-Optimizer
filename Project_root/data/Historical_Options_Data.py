import requests
import csv
import os
import time
import random
from datetime import datetime, timedelta, date, timezone
from multiprocessing import Pool, cpu_count

BASE_URL     = "https://api.delta.exchange/v2"
WEIGHT_QUOTA = 10000
CALL_WEIGHT  = 3
INC          = 500
PCT_BAND     = 0.10
MAX_RETRIES  = 5
ERROR_LOG    = "api_errors.csv"


def request_with_retry(url: str, params: dict, retries: int = MAX_RETRIES) -> dict:
    """Wrapper around requests.get with retry + exponential backoff"""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            wait_time = 2 ** attempt + random.uniform(0, 1)
            print(f"[RETRY] Attempt {attempt+1}/{retries} failed for {url} with {e}. Retrying in {wait_time:.1f}s")
            time.sleep(wait_time)
    # All retries failed → return None
    return None


def log_error(context: str, symbol: str, target_date: date, error_msg: str):
    """Append error details to error log CSV"""
    file_exists = os.path.exists(ERROR_LOG)
    with open(ERROR_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "context", "symbol", "date", "error"])
        writer.writerow([
            datetime.utcnow().isoformat(),
            context,
            symbol,
            target_date.strftime("%Y-%m-%d"),
            error_msg
        ])


def fetch_futures_ohlc(symbol: str, target_date: date) -> dict:
    start_dt = datetime(target_date.year, target_date.month, target_date.day, 0, 0, tzinfo=timezone.utc)
    end_dt   = start_dt + timedelta(days=1)
    start_ts = int(start_dt.timestamp())
    end_ts   = int(end_dt.timestamp())

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
    start_dt = datetime(target_date.year, target_date.month, target_date.day, 0, 0, tzinfo=timezone.utc)
    end_dt   = start_dt + timedelta(days=1)
    start_ts = int(start_dt.timestamp())
    end_ts   = int(end_dt.timestamp())

    url = f"{BASE_URL}/history/candles"
    params = {"symbol": symbol, "resolution": "5m", "start": str(start_ts), "end": str(end_ts)}

    data = request_with_retry(url, params)
    if not data:
        log_error("option_candles", symbol, target_date, "Max retries exceeded")
        return []
    return data.get("result", [])
