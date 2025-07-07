import requests
import time
import csv
import calendar
from datetime import datetime, timedelta

BASE_URL = "https://api.delta.exchange/v2"

def fetch_all_products():
    resp = requests.get(f"{BASE_URL}/products", headers={"Accept": "application/json"})
    resp.raise_for_status()
    return resp.json().get("result", [])

def filter_btc_futures(products):
    return [
        p["symbol"]
        for p in products
        if p.get("symbol", "").upper().startswith("BTC") and "futures" in p.get("description", "").lower()
    ]

def fetch_candles_over(symbol, resolution="5m", days=7):
    dt = datetime(2025, 5, 21, 13, 0, 0)
    start_ts = calendar.timegm(dt.timetuple())
    all_candles = []

    interval_seconds = 6 * 3600  # 6 hours per request
    daily_counter = 0

    for i in range((days * 86400) // interval_seconds):
        t_start = start_ts + i * interval_seconds
        t_end = t_start + interval_seconds

        params = {
            "symbol": symbol,
            "resolution": resolution,
            "start": str(t_start),
            "end": str(t_end)
        }

        try:
            resp = requests.get(f"{BASE_URL}/history/candles", params=params, headers={"Accept": "application/json"})
            resp.raise_for_status()
            chunk = resp.json().get("result", [])
            if chunk:
                all_candles.extend(chunk)
        except Exception as e:
            print(f"[ERROR] Fetch failed for {symbol} from {t_start} to {t_end}: {e}")
        
        # Rate limit buffer
        time.sleep(0.1)

        daily_counter += interval_seconds
        if daily_counter >= 86400:
            time.sleep(1)
            daily_counter = 0

    return all_candles

def main():
    products = fetch_all_products()
    symbols = filter_btc_futures(products)
    seen = set()
    rows = []

    for sym in symbols:
        candles = fetch_candles_over(sym, days=30)
        for c in candles:
            key = (sym, c["time"])
            if key not in seen:
                seen.add(key)
                rows.append({
                    "symbol": sym,
                    "time": c["time"],
                    "open": c["open"],
                    "high": c["high"],
                    "low": c["low"],
                    "close": c["close"],
                    "volume": c["volume"]
                })

    if rows:
        fieldnames = ["symbol", "time", "open", "high", "low", "close", "volume"]
        with open("data/futures.csv", "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

if __name__ == "__main__":
    main()
