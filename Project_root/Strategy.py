# -*- coding: utf-8 -*-
'''
Strategy.py
This module implements a trading strategy for options based on futures prices,
using Market Direction Rules for entry and exit.
'''
import os
import pandas as pd
import numpy as np
from datetime import datetime, time

class Strategy:
    def __init__(self, simulator):
        self.sim = simulator
        # State flags
        self.entry_done = False
        self.exit_done = False
        self.entry_price = None
        self.legs = []                # list of (symbol, side)
        self.trade_log = []
        self.current_day = None
        self.last_entry_time = None
        self.min_hold_minutes = 5     # minimum hold time before exit
        
        # Load futures (underlying) data: assume file data/futures.csv with 5-min bars
        fut_path = os.path.join("data", "futures.csv")
        self.futures_df = pd.read_csv(fut_path)
        self.futures_df["time"] = pd.to_datetime(self.futures_df["time"], unit="s")
        self.futures_df.set_index("time", inplace=True)
        
        # Entry time at 13:00 each day
        start = self.sim.startDate.date()
        self.entry_time = datetime.combine(start, time(13, 0))
        print(f"[DEBUG] Strategy will enter at {self.entry_time.time()} each day")

    def onMarketData(self, row):
        now = row["time"]
        today = now.date()
        sym = row["Symbol"]
        
        # New day reset & forced exit
        if today != self.current_day:
            self._on_new_day(today, row["close"])
        
        # Only consider option ticks for entry/exit but use futures for pricing
        if not self.entry_done and not self.exit_done and now >= self.entry_time:
            # get futures price at entry timestamp
            fut_price = self._get_futures_price_at(self.entry_time)
            if fut_price is None:
                print(f"[ERROR] No futures price at {self.entry_time}")
                return
            
            # decide direction
            self.entry_price = fut_price
            direction = self._select_direction(fut_price)
            # extract trade date from this option row
            trade_date = row["TradeDate"]
            
            # build legs for each direction
            if direction == "BULLISH":
                atm, wing = self._get_spread_strikes(trade_date, fut_price)
                if atm is None:
                    print(f"[WARN] No call strikes for {trade_date}")
                    return
                # Bull Call Spread: buy ATM call, sell next OTM call
                self._enter_leg(trade_date, "C", atm, "BUY")
                self._enter_leg(trade_date, "C", wing, "SELL")
            
            elif direction == "BEARISH":
                atm, wing = self._get_spread_strikes(trade_date, fut_price, opt="P")
                if atm is None:
                    print(f"[WARN] No put strikes for {trade_date}")
                    return
                # Bear Put Spread: buy OTM put, sell ATM put
                self._enter_leg(trade_date, "P", wing, "BUY")
                self._enter_leg(trade_date, "P", atm,  "SELL")
            
            else:  # NEUTRAL
                ci, co = self._get_wing_strikes(trade_date, fut_price, "C")
                pi, po = self._get_wing_strikes(trade_date, fut_price, "P")
                if ci is None or pi is None:
                    print(f"[WARN] Not enough strikes for Iron Condor on {trade_date}")
                    return
                # Iron Condor: sell ATM legs, buy wings
                self._enter_leg(trade_date, "C", ci, "SELL")
                self._enter_leg(trade_date, "P", pi, "SELL")
                self._enter_leg(trade_date, "C", co, "BUY")
                self._enter_leg(trade_date, "P", po, "BUY")
            
            self.entry_done = True
            self.last_entry_time = now
            print(f"[ENTRY {direction}] {today} {now.time()} ATM={fut_price:.2f}")
        
        # Exit logic after minimum hold
        if self.entry_done and not self.exit_done:
            held = (now - self.last_entry_time).total_seconds() / 60
            if held < self.min_hold_minutes:
                return
            fut_now = self._get_futures_price_at(now)
            if fut_now is None:
                return
            dev = abs(fut_now - self.entry_price) / self.entry_price
            pnl = self.sim.pnl_history[-1]["pnl"] if self.sim.pnl_history else 0.0
            if dev > 0.01 or pnl > 500 or pnl < -500:
                # unwind all legs
                for leg_sym, leg_side in self.legs:
                    side = "BUY" if leg_side == "SELL" else "SELL"
                    price = self.sim.currentPrice.get(leg_sym, row["close"])
                    self.sim.onOrder(leg_sym, side, 0.1, price)
                self.exit_done = True
                print(f"[EXIT] {today} {now.time()} Dev={dev:.4f} PnL={pnl:.2f}")

    def onTradeConfirmation(self, symbol, side, quantity, price):
        # log trade
        self.trade_log.append({
            "symbol": symbol,
            "side": side.upper(),
            "quantity": quantity,
            "price": price
        })

    def _on_new_day(self, new_day, close_price):
        self.current_day = new_day
        self.entry_time = datetime.combine(new_day, time(13, 0))
        if self.entry_done and not self.exit_done:
            print(f"[FORCED EXIT] Carryover → Closing on {new_day}")
            for leg_sym, leg_side in self.legs:
                side = "BUY" if leg_side=="SELL" else "SELL"
                self.sim.onOrder(leg_sym, side, 0.1, close_price)
            self.exit_done = True
        # reset state
        self.entry_done = False
        self.exit_done = False
        self.entry_price = None
        self.legs.clear()
        self.last_entry_time = None
        print(f"[NEW DAY] {new_day} state reset")

    def _get_futures_price_at(self, dt):
        df = self.futures_df[self.futures_df.index <= dt]
        return df.iloc[-1]["close"] if not df.empty else None

    def _select_direction(self, price):
        # compare current price to entry
        dev = (price - self.entry_price) / self.entry_price if self.entry_price else 0
        if dev > 0.005: return "BULLISH"
        if dev < -0.005: return "BEARISH"
        return "NEUTRAL"

    def _get_spread_strikes(self, trade_date, atm_price, opt="C"):
        df = self.sim.df[(self.sim.df["TradeDate"]==trade_date) & (self.sim.df["Symbol"].str.contains(f":{opt}-"))]
        strikes = sorted(df["Strike"].unique())
        if not strikes: return None, None
        atm = min(strikes, key=lambda k: abs(k-atm_price))
        idx = strikes.index(atm)
        wing = strikes[idx+1] if idx+1<len(strikes) else atm + (strikes[1]-strikes[0])
        return atm, wing

    def _get_wing_strikes(self, trade_date, atm_price, opt):
        df = self.sim.df[(self.sim.df["TradeDate"]==trade_date) & (self.sim.df["Symbol"].str.contains(f":{opt}-"))]
        strikes = sorted(df["Strike"].unique())
        if not strikes: return None, None
        atm = min(strikes, key=lambda k: abs(k-atm_price))
        idx = strikes.index(atm)
        step = strikes[1]-strikes[0] if len(strikes)>1 else 0
        if opt=="C":
            wing = strikes[idx+2] if idx+2<len(strikes) else atm+2*step
        else:
            wing = strikes[idx-2] if idx-2>=0 else atm-2*step
        return atm, wing

    def _enter_leg(self, trade_date, opt, strike, side):
        # symbol format: MARK:<opt>-BTC-<strike>-<ddmmyy>
        date_str = datetime.strptime(trade_date, "%d-%m-%Y").strftime("%d%m%y")
        sym = f"MARK:{opt}-BTC-{strike}-{date_str}"
        price = self.sim.currentPrice.get(sym, 0.0)
        self.sim.onOrder(sym, side, 0.1, price)
        self.legs.append((sym, side))

    def export_log(self, filename="trade_log.csv"):
        if self.trade_log:
            pd.DataFrame(self.trade_log).to_csv(filename, index=False)
            print(f"[INFO] Trade log exported to {filename}")
        else:
            print("[INFO] No trades to export.")
