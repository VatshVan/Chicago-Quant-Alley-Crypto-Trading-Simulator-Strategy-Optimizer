# -*- coding: utf-8 -*-
'''
config.py
This script extracts unique option symbols from CSV files in a specified date range.
It scans through directories structured by date and expiry, reading 'calls.csv' and 'puts.csv' files.
It collects unique symbols and prints the total number of files processed and unique symbols found.
'''
import os
import pandas as pd

StartDate = "2025-05-21"
EndDate   = "2025-06-20"

def extract_symbols(data_root="data", start_date=StartDate, end_date=EndDate):
    symbols = set()
    total_files = 0

    # Convert date strings to datetime objects for comparison
    start = pd.to_datetime(start_date).date()
    end   = pd.to_datetime(end_date).date()

    for date_folder in os.listdir(data_root):
        try:
            date_obj = pd.to_datetime(date_folder, format="%d-%m-%Y").date()
        except Exception:
            continue

        if not (start <= date_obj <= end):
            continue

        date_path = os.path.join(data_root, date_folder)
        if not os.path.isdir(date_path):
            continue

        for expiry_folder in os.listdir(date_path):
            expiry_path = os.path.join(date_path, expiry_folder)
            if not os.path.isdir(expiry_path):
                continue

            for file in ["calls.csv", "puts.csv"]:
                file_path = os.path.join(expiry_path, file)
                if os.path.exists(file_path):
                    total_files += 1
                    try:
                        df = pd.read_csv(file_path)
                        if "symbol" in df.columns:
                            symbols.update(df["symbol"].dropna().unique())
                    except Exception as e:
                        print(f"[ERROR] Failed reading {file_path}: {e}")

    # Add BTCUSDT as a default symbol
    symbols.add("BTCUSDT")

    print(f"[INFO] Loaded {total_files} option files.")
    print(f"[INFO] Extracted {len(symbols)} unique option symbols from {start} to {end}.")
    return set(sorted(list(symbols)))

symbols = extract_symbols("data", StartDate, EndDate)
