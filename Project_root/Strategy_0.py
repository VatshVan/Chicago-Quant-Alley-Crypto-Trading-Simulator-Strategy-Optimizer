# -*- coding: utf-8 -*-
'''
Strategy.py
This module implements a trading strategy for options based on futures prices.
It handles entry and exit logic, manages trades, and logs performance.
'''
import pandas as pd
from datetime import datetime

class Strategy:
    def __init__(self, simulator):
        self.sim = simulator
        self.entry_done = False
        self.exit_done = False
        self.entry_price = None
        self.call_symbol = None
        self.put_symbol = None
        self.trade_log = []
        self.current_day = None
        self.last_entry_time = None
        self.min_hold_minutes = 5  # Minimum holding time before exit

        # Load futures data
        self.futures_df = pd.read_csv("data/futures.csv")
        self.futures_df["time"] = pd.to_datetime(self.futures_df["time"], unit="s")
        self.futures_df.set_index("time", inplace=True)

        # Initialize entry time to 13:00 on the simulation start date
        start_date = self.sim.startDate.date()
        self.entry_time = datetime.combine(start_date, datetime.strptime("13:00:00", "%H:%M:%S").time())
        print("[DEBUG] Target entry time:", self.entry_time)

    def onMarketData(self, row):
        row_time = row["time"]
        row_date = row_time.date()
        symbol = row["Symbol"]

        # Check if we are on a new day
        if self.current_day != row_date:
            self.current_day = row_date
            self.entry_time = datetime.combine(row_date, datetime.strptime("13:00:00", "%H:%M:%S").time())

            # Force exit if we had an open position from the previous day
            if self.entry_done and not self.exit_done:
                print(f"[FORCED EXIT] Previous position carried over → Closing on {row_date}")
                self.sim.onOrder(self.call_symbol, "BUY", 0.1, row["close"])
                self.sim.onOrder(self.put_symbol, "BUY", 0.1, row["close"])
                self.exit_done = True

            # Reset state for the new day
            self.entry_done = False
            self.exit_done = False
            self.entry_price = None
            self.call_symbol = None
            self.put_symbol = None
            self.last_entry_time = None
            print(f"\n[NEW DAY] {row_date} → state reset")

        # Entry logic
        if not self.entry_done and not self.exit_done and symbol.startswith("MARK:") and row_time >= self.entry_time:
            fut_price = self.get_futures_price_at(self.entry_time)
            if fut_price is None:
                print(f"[ERROR] No futures price at {self.entry_time}")
                return

            self.entry_price = fut_price
            trade_date = row["TradeDate"]
            call, put = self.get_nearest_call_put(fut_price, trade_date)

            if call is None or put is None:
                print(f"[ERROR] No suitable strikes for ATM {fut_price:.2f}")
                return

            self.call_symbol = call
            self.put_symbol = put

            self.sim.onOrder(self.call_symbol, "SELL", 0.1, row["close"])   # entry
            self.sim.onOrder(self.put_symbol, "SELL", 0.1, row["close"])    # entry
            self.entry_done = True
            self.last_entry_time = row_time # Record entry time for hold logic

            print(f"[ENTRY] {row_date} | Time: {row_time.time()} | ATM = {fut_price:.2f}")
            print(f"        Call: {call} | Put: {put}")

        # Exit logic
        if self.entry_done and not self.exit_done:
            # Ensure we have a valid entry time and check minimum hold time
            if self.last_entry_time is None or (row_time - self.last_entry_time).total_seconds() / 60 < self.min_hold_minutes:
                return

            fut_price = self.get_futures_price_at(row_time)
            if fut_price is None:
                return

            deviation = abs(fut_price - self.entry_price) / self.entry_price
            current_pnl = self.sim.pnl_history[-1]["pnl"] if self.sim.pnl_history else 0.0

            if deviation > 0.01 or current_pnl > 500 or current_pnl < -500:
                self.sim.onOrder(self.call_symbol, "BUY", 0.1, row["close"])  # exit
                self.sim.onOrder(self.put_symbol, "BUY", 0.1, row["close"])  # exit
                self.exit_done = True

                print(f"[EXIT] {row_date} | Time: {row_time.time()} | Dev: {deviation:.4f} | PnL: {current_pnl:.2f}")

    # This method is called when a trade is confirmed
    # It logs the trade details for later analysis
    def onTradeConfirmation(self, symbol, side, quantity, price):
        self.trade_log.append({
            "symbol": symbol,
            "side": side.upper(),
            "quantity": quantity,
            "price": price
        })

    # This method retrieves the futures price at a specific datetime
    # It returns the last available price before or at the given datetime
    def get_futures_price_at(self, dt):
        try:
            available = self.futures_df[self.futures_df.index <= dt]
            if not available.empty:
                return available.iloc[-1]["close"]
            return None
        except Exception as e:
            print(f"[ERROR] get_futures_price_at failed: {e}")
            return None

    # This method finds the nearest call and put options for a given ATM price and trade date
    # It returns the symbols of the nearest call and put options
    def get_nearest_call_put(self, atm_price, trade_date):
        df = self.sim.df
        options_df = df[(df["Symbol"].str.startswith("MARK:")) & (df["TradeDate"] == trade_date)]

        calls = options_df[options_df["Symbol"].str.contains(":C-")]
        puts  = options_df[options_df["Symbol"].str.contains(":P-")]

        if calls.empty or puts.empty:
            print(f"[WARN] No options on {trade_date}")
            return None, None

        nearest_call = calls.iloc[(calls["Strike"] - atm_price).abs().argsort()].head(1)
        nearest_put  = puts.iloc[(puts["Strike"] - atm_price).abs().argsort()].head(1)

        call_symbol = nearest_call["Symbol"].values[0] if not nearest_call.empty else None
        put_symbol  = nearest_put["Symbol"].values[0] if not nearest_put.empty else None

        return call_symbol, put_symbol

    # This method exports the trade log to a CSV file
    # It saves the log of trades executed during the simulation
    def export_log(self, filename="trade_log.csv"):
        if self.trade_log:
            pd.DataFrame(self.trade_log).to_csv(filename, index=False)
            print(f"[INFO] Trade log exported to {filename}")
        else:
            print("[INFO] No trades to export.")
