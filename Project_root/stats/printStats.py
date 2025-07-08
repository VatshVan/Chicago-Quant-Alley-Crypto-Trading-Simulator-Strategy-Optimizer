# -*- coding: utf-8 -*-
'''
PrintStats.py
This script analyzes trading performance by reading PnL and execution logs,
computing various metrics, and generating visualizations.
'''
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# Path configurations
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))  # stats/ folder
PNL_FILE = os.path.join(SAVE_DIR, "pnl_history.csv")
EXEC_FILE = os.path.join(SAVE_DIR, "execution_log.csv")
FUTURE_FILE = os.path.join(SAVE_DIR, "../data/futures.csv")

# Define constants
def compute_drawdown(pnl_series):
    peak = pnl_series.cummax()
    drawdown = pnl_series - peak
    return drawdown

def compute_expected_shortfall(returns, confidence=0.95):
    if len(returns) == 0:
        return 0.0
    var = np.percentile(returns, (1 - confidence) * 100)
    es = returns[returns <= var].mean()
    return es if not np.isnan(es) else 0.0

# Print performance metrics
def print_metrics(df):
    pnl = df["pnl"]
    returns = pnl.diff().dropna()

    print("\n_____ Performance Metrics _____")
    print(f"Final PnL         : {pnl.iloc[-1]:.2f}")
    print(f"Mean PnL          : {pnl.mean():.2f}")
    print(f"Median PnL        : {pnl.median():.2f}")
    print(f"Std Dev of PnL    : {pnl.std():.2f}")

    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 12)  # 5-min bars
    else:
        sharpe = 0.0
    print(f"Sharpe Ratio      : {sharpe:.2f}")

    drawdown = compute_drawdown(pnl)
    print(f"Max Drawdown      : {drawdown.min():.2f}")

    var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0.0
    es_95 = compute_expected_shortfall(returns)
    print(f"Value at Risk (95%): {var_95:.2f}")
    print(f"Expected Shortfall : {es_95:.2f}")

# Compute trade statistics from execution log
def compute_trade_stats(exec_df):
    entries = exec_df[exec_df["tag"] == "entry"].reset_index(drop=True)
    exits = exec_df[exec_df["tag"] == "exit"].reset_index(drop=True)

    paired = min(len(entries), len(exits))
    wins, losses, pnls, holds = 0, 0, [], []

    for i in range(paired):
        e, x = entries.iloc[i], exits.iloc[i]
        if e["side"] == "SELL":
            pnl = (e["price"] - x["price"]) * e["quantity"]
        else:
            pnl = (x["price"] - e["price"]) * e["quantity"]
        pnls.append(pnl)
        wins += pnl > 0
        losses += pnl < 0
        holds.append((pd.to_datetime(x["time"]) - pd.to_datetime(e["time"])).total_seconds() / 60)

    print("\n_____ Trade Stats _____")
    print(f"Total Trades       : {paired}")
    print(f"Winning Trades     : {wins}")
    print(f"Losing Trades      : {losses}")
    print(f"Win Rate           : {(wins / paired * 100):.2f}%")
    print(f"Avg PnL / Trade    : {np.mean(pnls):.2f}")
    print(f"Avg Hold Time (min): {np.mean(holds):.2f}")

# Plot cumulative PnL and drawdown
def plot_pnl(df):
    pnl = df["pnl"]
    drawdown = compute_drawdown(pnl)

    plt.figure(figsize=(10, 4))
    plt.plot(pnl, label="Cumulative PnL", color="blue")
    plt.title("Cumulative PnL")
    plt.xlabel("Time")
    plt.ylabel("PnL")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "pnl_curve.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(drawdown, label="Drawdown", color="red")
    plt.title("Drawdown Curve")
    plt.xlabel("Time")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "drawdown_curve.png"))
    plt.close()

# Plot rolling Sharpe ratio
def plot_rolling_sharpe(df, window_hours=168, avg_window=25):
    """
    Plots a smoothed rolling Sharpe ratio based on resampled hourly returns.
    
    Parameters:
        df : DataFrame
            Must contain a datetime index and a 'pnl' column (cumulative PnL).
        window_hours : int
            Rolling window size in hours for Sharpe ratio calculation.
        avg_window : int
            Window length for Savitzky-Golay smoothing (must be odd).
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Step 1: Resample PnL to hourly and compute returns
    pnl = df["pnl"].resample("1h").mean().interpolate()
    returns = pnl.diff().dropna()

    # Step 2: Rolling Sharpe
    available_points = len(returns)
    N = min(window_hours, available_points)
    if N < 5:
        print("[ERROR] Not enough data to compute rolling Sharpe.")
        return

    roll_mean = returns.rolling(window=N, min_periods=N).mean()
    roll_std = returns.rolling(window=N, min_periods=N).std().clip(lower=1e-4)
    sharpe = (roll_mean / roll_std) * np.sqrt(252 * 24)
    sharpe = sharpe.dropna()

    if sharpe.empty:
        print("[ERROR] No valid Sharpe values to plot.")
        return

    # Step 3: Savitzky-Golay smoothing (only if enough points)
    sg_window = min(avg_window + (1 - avg_window % 2), len(sharpe))
    if sg_window < 5:
        print(f"[WARN] Not enough Sharpe points for smoothing. Plotting raw.")
        sharpe_smooth = sharpe
    else:
        try:
            sharpe_smooth = pd.Series(
                savgol_filter(sharpe.ffill().values, window_length=sg_window, polyorder=2),
                index=sharpe.index
            )
        except Exception as e:
            print(f"[WARN] Smoothing failed: {e}. Plotting raw.")
            sharpe_smooth = sharpe

    # Step 4: Plot
    plt.figure(figsize=(12, 5))
    plt.plot(sharpe, color="orange", alpha=0.3, label="Raw Sharpe")
    plt.plot(sharpe_smooth, color="purple", linewidth=1.5, label="Smoothed Sharpe")
    for level, col in zip([0, 1, 2], ['gray', 'blue', 'green']):
        plt.axhline(level, linestyle="--", color=col)
    plt.ylim(sharpe_smooth.min() - 0.5, sharpe_smooth.max() + 0.5)
    plt.title("Smoothed Rolling Sharpe Ratio", fontsize=14)
    plt.xlabel("Time")
    plt.ylabel("Sharpe")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "rolling_sharpe.png"))
    plt.close()

# Plot histogram of PnL changes
def plot_pnl_histogram(df):
    returns = df["pnl"].diff().dropna()
    plt.figure(figsize=(8, 4))
    plt.hist(returns, bins=30, color="skyblue", edgecolor="black")
    plt.title("Histogram of PnL Changes")
    plt.xlabel("PnL Change")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "pnl_histogram.png"))
    plt.close()

if __name__ == "__main__":
    try:
        df_pnl = pd.read_csv(PNL_FILE, parse_dates=["time"]).drop_duplicates(subset="time").sort_values("time").set_index("time")
    except Exception as e:
        print(f"[ERROR] Failed to load pnl_history.csv: {e}")
        exit(1)

    print_metrics(df_pnl)
    plot_pnl(df_pnl)
    plot_rolling_sharpe(df_pnl)
    plot_pnl_histogram(df_pnl)

    if os.path.exists(EXEC_FILE):
        try:
            df_exec = pd.read_csv(EXEC_FILE, parse_dates=["time"]).sort_values("time")
            compute_trade_stats(df_exec)

            if os.path.exists(FUTURE_FILE):
                future_df = pd.read_csv(FUTURE_FILE)
            else:
                print("[WARN] futures.csv not found for entry/exit plotting.")

        except Exception as e:
            print(f"[ERROR] Failed to process execution_log.csv: {e}")
    else:
        print("[WARN] execution_log.csv not found. Skipping trade stats and entry/exit plot.")
