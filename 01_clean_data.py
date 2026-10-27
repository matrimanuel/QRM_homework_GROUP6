# ============================================================
# 01_clean_data.py  -  Group 6 (Brazil): read raw LSEG exports,
# clean them, merge into one price table and compute log-returns
# ============================================================
import glob, os, re
import numpy as np
import pandas as pd

RAW_DIR = "data_raw"                     # folder with the raw LSEG CSV files
OUT_PRICES = "prices_brazil_clean.csv"
OUT_RET = "logreturns_brazil.csv"

# French/English month abbreviations -> month number
MONTHS = {"janv":1,"jan":1,"févr":2,"fevr":2,"feb":2,"mars":3,"mar":3,"avr":4,"apr":4,
          "mai":5,"may":5,"juin":6,"jun":6,"juil":7,"jul":7,"août":8,"aout":8,"aug":8,
          "sept":9,"sep":9,"oct":10,"nov":11,"déc":12,"dec":12}

def parse_date(s):
    m = re.match(r"^\s*(\d{1,2})-([^-]+)-(\d{2,4})\s*$", s)
    if not m: return pd.NaT
    d, mon, y = int(m[1]), m[2].lower().strip("."), int(m[3])
    y = y + 2000 if y < 100 else y
    return pd.Timestamp(y, MONTHS[mon], d) if mon in MONTHS else pd.NaT

def read_one(path):
    with open(path, encoding="cp1252", errors="replace") as f:
        lines = f.read().splitlines()
    ric = lines[3].split(";")[0].strip()                       # RIC is on line 4
    start = next(i for i,l in enumerate(lines) if l.lower().startswith("exchange date"))
    rows = {}
    for l in lines[start+1:]:                                   # data block only
        p = l.split(";")
        if len(p) < 2: continue
        d = parse_date(p[0])
        try: px = float(p[1].replace(" ", "").replace("\u202f","").replace(",", "."))
        except ValueError: continue
        if pd.notna(d): rows[d] = px
    return pd.Series(rows, name=ric).sort_index()               # oldest -> newest

# ---- 1. Read and merge all assets (dates in rows, one column per asset)
prices = pd.concat([read_one(p) for p in sorted(glob.glob(f"{RAW_DIR}/*.csv"))], axis=1)
prices.index.name = "Date"

# ---- 2. Data quality report
print("Shape:", prices.shape)
print("First valid date per asset:\n", prices.apply(lambda c: c.first_valid_index()).to_string())
print("Missing values per asset:\n", prices.isna().sum().to_string())

# ---- 3. Remove non-trading days: drop dates where (almost) no asset has a price
prices = prices.dropna(how="all")
prices = prices.loc[prices.notna().mean(axis=1) >= 0.5]        # holiday-like rows
prices = prices.ffill(limit=1).dropna()                         # fill isolated gaps, then common calendar
prices = prices.loc[(prices.index >= "2016-09-26") & (prices.index <= "2026-09-30")]

# ---- 4. Stale-price check (zero returns) and log-returns
rets = np.log(prices / prices.shift(1)).dropna()
print("\nShare of zero returns per asset:\n", (rets == 0).mean().round(3).to_string())
print("Largest absolute daily log-return per asset (check splits):\n",
      rets.abs().max().round(3).to_string())

prices.to_csv(OUT_PRICES, date_format="%Y-%m-%d")
rets.to_csv(OUT_RET, date_format="%Y-%m-%d")
print("\nFinal:", prices.shape, prices.index.min().date(), "->", prices.index.max().date())
