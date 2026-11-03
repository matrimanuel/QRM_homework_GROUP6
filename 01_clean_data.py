# ============================================================
# 01_clean_data.py  -  Group 6 (Brazil): read raw LSEG exports,
# clean them, merge into one price table and compute log-returns
# ============================================================
# with this we are creating the csv file in whitch we will store the cleaned data, and the log returns of the data

import glob, os, re
import numpy as np
import pandas as pd

asset_files = sorted(glob.glob("*.SA.csv"))     #read all the raw LSEG exports in the current folder, which are the .csv files with the suffix .SA.csv
OUT_PRICES = "prices_brazil_clean.csv"
OUT_RET = "logreturns_brazil.csv"

#We are converting French/English/italian month abbreviations into month number, because LSEG export the data with this structure
MONTHS = {"janv":1,"jan":1,"févr":2,"fevr":2,"feb":2,"mars":3,"mar":3,"avr":4,"apr":4,
          "mai":5,"may":5,"juin":6,"jun":6,"juil":7,"jul":7,"août":8,"aout":8,"aug":8, 
          "ago":8,"sept":9,"sep":9,"ott":10,"oct":10,"nov":11,"déc":12,"dec":12}

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
prices = pd.concat([read_one(p) for p in asset_files], axis=1)
prices.index.name = "Date"

# ---- 2. Data quality report
print("Shape:", prices.shape)
print("First valid date per asset:\n", prices.apply(lambda c: c.first_valid_index()).to_string())
print("Missing values per asset:\n", prices.isna().sum().to_string())

# ---- 3. Remove non-trading days using official bank holidays
prices = prices.dropna(how="all")

# Load official bank holidays calendar
holidays_df = pd.read_csv("brasilian_bankholidays_2016_2026.csv", sep=None, engine="python")
official_holidays = pd.DatetimeIndex(pd.to_datetime(holidays_df.iloc[:, 0], errors="coerce").dropna())

# Count and drop official holiday dates present in the sample
holidays_in_data = prices.index.intersection(official_holidays)
print(f"Number of official bank holidays dropped: {len(holidays_in_data)}")

prices = prices.loc[~prices.index.isin(official_holidays)]
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
