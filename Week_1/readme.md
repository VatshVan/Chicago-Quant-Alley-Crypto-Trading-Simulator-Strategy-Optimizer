# BTC Options Data Backfill – Format Documentation

## Objective

This script collects 5-minute OHLCV data for BTC options (Calls and Puts) from Delta Exchange, for a specified date range and a set of expiry offsets. It saves the data in a structured format for further quantitative analysis.

---

## Folder Structure

Data is stored by observation date and expiry:


\<observation\_date>/
|
└── \<expiry\_date>/
|
├── calls.csv
|
└── puts.csv


- `<observation_date>`: Date data was recorded (format: `DD-MM-YYYY`)
- `<expiry_date>`: Option expiry date (same format)

---

## CSV Format

Each `calls.csv` and `puts.csv` file contains the following columns:

| Column   | Description                                              |
|----------|----------------------------------------------------------|
| symbol   | Option identifier (e.g., `MARK:P-BTC-28000-250525`)      |
| time     | UNIX timestamp (UTC) of candle start                     |
| open     | Opening price of the 5-min interval                      |
| high     | Highest price in the interval                            |
| low      | Lowest price in the interval                             |
| close    | Closing price                                            |
| volume   | Trade volume during the interval                         |

---

## Strike & Expiry Generation

- Strikes are derived from daily low/high ±10%, rounded to the nearest 500.
- Expiry dates are computed as `observation_date + expiry_offset` for each value in the offset list.

---
