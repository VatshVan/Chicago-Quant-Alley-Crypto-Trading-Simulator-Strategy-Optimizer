import requests
import time
import csv
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
        if p.get("symbol", "").upper().startswith("BTC")
           and "futures" in p.get("description", "").lower()]

def fetch_candles_over_week(symbol, resolution="5m", days=7):
    now_ts = int(time.time())
    interval_seconds = 86400
    start_ts = now_ts - days * 86400
    all_candles = []

    for t_start in range(start_ts, now_ts, interval_seconds):
        t_end = min(t_start + interval_seconds, now_ts)
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
            continue
        time.sleep(0.25)
    return all_candles

def main():
    products = fetch_all_products()
    symbols = filter_btc_futures(products)
    rows = []
    for sym in symbols:
        candles = fetch_candles_over_week(sym)
        for c in candles:
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
        with open("btc_futures_1week.csv", "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

if __name__ == "__main__":
    main()
