# -*- coding: utf-8 -*-
"""
Simulator.py
This module simulates a trading environment for options strategies.
It reads market data from CSV files, executes trades based on a strategy, and tracks performance metrics.
"""
import os
import pandas as pd
from datetime import datetime, timedelta
from config import StartDate, EndDate, symbols
from Strategy import Strategy

class Simulator:
    def __init__(self, configFilePath=None):
        ''' 
        Init the simulator with:
        - StartDate: Simulation start date
        - EndDate: Simulation end date
        - symbols: List of option symbols to trade
        - df: DataFrame to hold market data
        - currentPrice: Dictionary to hold latest prices for each symbol
        - slippage: Slippage percentage for order execution
        - turnover_history: List to track turnover over time
        - total_turnover: Total turnover across all trades
        - pnl_history: List to track profit and loss over time
        - strategy: Instance of Strategy class to handle trading logic
        - execution_log: List to track all executed trades
        - trade_lifecycle: Dictionary to track entry/exit times for trades
        - readData: Method to load market data from CSV files
        - startSimulation: Method to begin the simulation process
        - onOrder: Method to simulate order execution
        - printPnl: Method to compute and print the current profit and loss
        '''
        self.startDate = pd.to_datetime(StartDate)
        self.endDate = pd.to_datetime(EndDate)
        self.symbols = symbols

        self.df = pd.DataFrame()
        self.currentPrice = {}
        self.slippage = 0.0001
        self.turnover_history = []
        self.total_turnover = 0.0
        self.pnl_history = []

        self.strategy = Strategy(self)
        self.execution_log = []
        self.trade_lifecycle = {}

        self.readData()
        self.startSimulation()

    def readData(self):
        '''
        Reads market data from CSV files structured by date and expiry.
        - Iterates through each date from startDate to endDate
        - For each date, checks if the directory exists
        - Loads call and put options data from respective CSV files
        - Extracts relevant fields like symbol, strike, expiry, option type, and trade date
        - Concatenates all data into a single DataFrame
        - Handles errors gracefully if files are missing or unreadable
        - Sets self.symbols dynamically based on loaded data
        - Initializes bookkeeping dictionaries for current quantity, buy value, and sell value
        - Sorts the DataFrame by time and resets the index
        - Prints debug information about loaded data
        - If no data is loaded, initializes an empty DataFrame to avoid downstream crashes
        '''
        all_data = []
        current_date = self.startDate

        while current_date <= self.endDate:
            date_str = current_date.strftime("%d-%m-%Y")
            base_folder = os.path.join("data", date_str)

            if not os.path.isdir(base_folder):
                current_date += timedelta(days=1)
                continue

            for expiry_dir in os.listdir(base_folder):
                expiry_path = os.path.join(base_folder, expiry_dir)
                if not os.path.isdir(expiry_path):
                    continue

                for option_type in ["calls.csv", "puts.csv"]:
                    file_path = os.path.join(expiry_path, option_type)

                    if os.path.exists(file_path):
                        try:
                            df = pd.read_csv(file_path)
                            df["Symbol"] = df["symbol"]
                            # Extract strike and expiry from the symbol
                            df["Strike"] = df["symbol"].str.extract(r'-(\d+)-')[0].astype(int)
                            df["Expiry"] = df["symbol"].str.extract(r'-(\d+)$')[0]
                            df["OptionType"] = option_type.split('.')[0].upper()
                            df["TradeDate"] = date_str
                            all_data.append(df)
                        except Exception as e:
                            print(f"[ERROR] Failed to load {file_path}: {e}")

            current_date += timedelta(days=1)
        # Concatenate all loaded data into a single DataFrame
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.sort_values("time", inplace=True)
            df.reset_index(drop=True, inplace=True)
            self.df = df
            # Dynamically set self.symbols from loaded data
            self.symbols = sorted(df["Symbol"].unique())
            self.currQuantity = {symbol: 0.0 for symbol in self.symbols}
            self.buyValue = {symbol: 0.0 for symbol in self.symbols}
            self.sellValue = {symbol: 0.0 for symbol in self.symbols}
        else:
            print("[ERROR] No market data loaded. Check folder structure.")
            self.df = pd.DataFrame()    # To avoid downstream crashes

        # Debug information
        if not self.df.empty:
            btc = self.df[self.df["Symbol"] == "BTCUSDT"]
            formatted_times = btc["time"].dt.strftime("%H:%M:%S").unique()
            print("BTCUSDT unique times:", formatted_times)

    def startSimulation(self):
        '''
        Starts the simulation by iterating through each row of the DataFrame:
        - For each row, extracts the symbol and latest price
        - Updates the current price dictionary
        - Calls the strategy's onMarketData method with the full market row
        - Optionally prints the current PnL for each timestamp
        - Handles the case where no data is available gracefully
        '''
        if self.df is None or self.df.empty:
            print("[ERROR] No data to simulate.")
            return

        for _, row in self.df.iterrows():
            symbol = row["Symbol"]
            price = row["close"]    # Using close price as latest

            self.currentPrice[symbol] = price

            # Update strategy with the full market data row
            self.strategy.onMarketData(row)

            # Print PnL at each timestamp
            self.printPnl(timestamp=row["time"])

    def onOrder(self, symbol, side, quantity, price):
        '''
        Simulates an order execution by:
        - Adjusting the price based on slippage
        - Calculating the trade value
        - Updating the total turnover
        - Initializing bookkeeping for the symbol if not already present
        - Updating current quantity, buy value, and sell value based on the trade side
        - Notifying the strategy of the trade confirmation
        - Tracking execution log with time, symbol, side, quantity, price, and tag
        - Tracking entry/exit time for hold duration
        - Handles errors for invalid trade sides
        - Prints debug information for each trade
        Args:
            symbol (str): The trading symbol (e.g., "BTCUSDT")
            side (str): The trade side ("buy" or "sell")
            quantity (float): The quantity of the asset to trade
            price (float): The price at which the trade is executed
        Returns:
            None
        Raises:
            ValueError: If the trade side is invalid
        '''
        # Adjust price for slippage
        if side.lower() == "buy":
            adjusted_price = price * (1 + self.slippage)
        elif side.lower() == "sell":
            adjusted_price = price * (1 - self.slippage)
        else:
            print(f"[ERROR] Invalid trade side: {side}")
            return

        trade_value = adjusted_price * quantity
        self.total_turnover += abs(trade_value)

        # Initialize bookkeeping for the symbol if not already present
        if symbol not in self.currQuantity:
            self.currQuantity[symbol] = 0.0
            self.buyValue[symbol] = 0.0
            self.sellValue[symbol] = 0.0
            self.symbols.append(symbol)

        # Update current quantity, buy value, and sell value
        if side.lower() == "buy":
            self.currQuantity[symbol] += quantity
            self.buyValue[symbol] += trade_value
        elif side.lower() == "sell":
            self.currQuantity[symbol] -= quantity
            self.sellValue[symbol] += trade_value

        # Notify the strategy of the trade confirmation
        self.strategy.onTradeConfirmation(symbol, side, quantity, adjusted_price)

        # Track execution log
        # Use current time for the execution log
        now = datetime.now()
        tag = "entry" if (side.upper() == "SELL" and symbol not in self.trade_lifecycle) else "exit"

        self.execution_log.append({
            "time": now,
            "symbol": symbol,
            "side": side.upper(),
            "quantity": quantity,
            "price": adjusted_price,
            "tag": tag
        })

        # Track entry/exit time for hold duration
        if side.upper() == "SELL":
            self.trade_lifecycle[symbol] = {
                "entry_time": now,
                "entry_price": adjusted_price
            }
        elif side.upper() == "BUY" and symbol in self.trade_lifecycle:
            self.trade_lifecycle[symbol]["exit_time"] = now
            self.trade_lifecycle[symbol]["exit_price"] = adjusted_price


    def printPnl(self, timestamp=None):
        '''
        Computes and prints the current profit and loss (PnL) for the portfolio.
        - Iterates through each symbol in the current quantity
        - Calculates realized and unrealized PnL for each symbol
        - Sums up the total PnL across all symbols
        - Optionally logs the PnL and turnover history with a timestamp
        - If no timestamp is provided, uses the current time
        Args:
            timestamp (datetime, optional): The timestamp for the PnL calculation. Defaults to None.
        Returns:
            None
        Raises:
            None
        Prints:
            - Total portfolio P&L
        - Timestamp of the P&L calculation
        - Updates the PnL and turnover history lists with the current values
        Example:
        >>> sim.printPnl()  # Prints current P&L and updates history
        >>> sim.printPnl(timestamp=datetime(2025, 5, 21, 12, 0))  # Prints P&L for a specific timestamp
        Note:
        - This method is called at each timestamp during the simulation to track performance.
        - If the current price for a symbol is not available, it skips that symbol.
        '''

        total_pnl = 0.0

        for symbol in self.currQuantity:
            buy = self.buyValue.get(symbol, 0.0)
            sell = self.sellValue.get(symbol, 0.0)
            quantity = self.currQuantity.get(symbol, 0.0)
            current_price = self.currentPrice.get(symbol)

            if current_price is None:
                continue

            realized_pnl = sell - buy
            unrealized_pnl = quantity * current_price
            symbol_pnl = realized_pnl + unrealized_pnl

            total_pnl += symbol_pnl

        # print(f"[PnL|DEBUG] Time: {timestamp} | Total Portfolio P&L: {total_pnl:.2f}")

        if timestamp is not None and self.execution_log:
            self.pnl_history.append({
                "time": timestamp,
                "pnl": total_pnl
            })
            self.turnover_history.append({
                "time": timestamp,
                "turnover": self.total_turnover
            })
        else:
            self.turnover_history.append({
                "time": datetime.now(),
                "turnover": self.total_turnover
            })

# Instantiate and run the simulator
if __name__ == "__main__":
    sim = Simulator()
    
    # Export trade log to CSV
    sim.strategy.export_log("stats/trade_log.csv")

    # Clean PnL history and save
    pd.DataFrame(sim.pnl_history).drop_duplicates(subset="time", keep="last").sort_values("time").to_csv("stats/pnl_history.csv", index=False)

    # Clean turnover history and save
    pd.DataFrame(sim.turnover_history).drop_duplicates(subset="time", keep="last").sort_values("time").to_csv("stats/turnover_history.csv", index=False)

    print("[SIMULATOR] Simulation completed. Logs exported.")

    # Save execution log to CSV
    pd.DataFrame(sim.execution_log).to_csv("stats/execution_log.csv", index=False)
    print("[INFO] Execution log saved to execution_log.csv")
