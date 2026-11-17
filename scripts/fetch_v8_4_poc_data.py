"""
V8.4 Historical Data Proof of Concept - Real Data Ingestion Script
Fetches representative small-scale real market data from:
1. FinMind Open API (TWSE Stock Prices, Institutional Net Buy, Margin/Short, Shares Outstanding, TAIEX, USD/TWD)
2. Yahoo Finance / FRED (S&P 500, NASDAQ, SOX, VIX, US 10Y Treasury Yield)
3. TWSE OpenAPI (MOPS Material Announcements)

Stocks:
- 2330: Taiwan Semiconductor Manufacturing Co. (Semiconductor / Tech Benchmark)
- 2603: Evergreen Marine Corp. (Shipping / Cyclical Volatility)
- 2881: Fubon Financial Holding Co. (Financial / Rates & Real MOPS Announcement)

Period:
- Warm-Up: 2023-06-01 to 2023-12-31 (~140 trading days)
- Test/Snapshot: 2024-01-02 to 2024-03-29 (~60 trading days)
Total ~202 trading days.
"""

import os
import sys
import time
import json
import logging
import requests
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v8_4_poc_fetcher")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "poc_real_data")
os.makedirs(DATA_DIR, exist_ok=True)

START_DATE = "2023-06-01"
END_DATE = "2024-03-29"
STOCKS = ["2330", "2603", "2881"]


def fetch_finmind_dataset(dataset: str, data_id: str, start_date: str = START_DATE, end_date: str = END_DATE) -> pd.DataFrame:
    url = f"https://api.finmindtrade.com/api/v4/data?dataset={dataset}&data_id={data_id}&start_date={start_date}&end_date={end_date}"
    logger.info(f"Fetching FinMind {dataset} for {data_id}...")
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json().get("data", [])
            df = pd.DataFrame(data)
            logger.info(f"  -> Retrieved {len(df)} records.")
            return df
        else:
            logger.warning(f"  -> Failed with status {r.status_code}: {r.text[:100]}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"  -> Error: {e}")
        return pd.DataFrame()


def fetch_yahoo_series(symbol: str, name: str) -> pd.DataFrame:
    # 2023-05-20 to 2024-04-05
    p1 = 1684540800
    p2 = 1712275200
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={p1}&period2={p2}&interval=1d"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    logger.info(f"Fetching Yahoo {name} ({symbol})...")
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            res = r.json()["chart"]["result"][0]
            timestamps = res["timestamp"]
            closes = res["indicators"]["quote"][0]["close"]
            dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("America/New_York")
            df = pd.DataFrame({"close": closes}, index=dates)
            df = df.dropna()
            logger.info(f"  -> Retrieved {len(df)} points for {name}.")
            return df
        else:
            logger.warning(f"  -> Failed with status {r.status_code}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"  -> Error: {e}")
        return pd.DataFrame()


def fetch_fred_series(series_id: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    logger.info(f"Fetching FRED {series_id}...")
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            df = pd.read_csv(pd.io.common.StringIO(r.text))
            df["observation_date"] = pd.to_datetime(df["observation_date"])
            df = df.set_index("observation_date")
            df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
            df = df.dropna()
            df = df.loc[START_DATE:END_DATE]
            logger.info(f"  -> Retrieved {len(df)} points for {series_id}.")
            return df
        else:
            logger.warning(f"  -> Failed with status {r.status_code}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"  -> Error: {e}")
        return pd.DataFrame()


def fetch_twse_mops_announcements() -> list:
    url = "https://openapi.twse.com.tw/v1/opendata/t187ap04_L"
    logger.info("Fetching TWSE OpenAPI MOPS announcements...")
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            items = r.json()
            logger.info(f"  -> Retrieved {len(items)} official announcements.")
            return items
        else:
            logger.warning(f"  -> Failed with status {r.status_code}")
            return []
    except Exception as e:
        logger.error(f"  -> Error: {e}")
        return []


def run_ingestion():
    logger.info(f"Starting V8.4 PoC Data Ingestion into {DATA_DIR}...")

    # 1. Stocks OHLCV, Institutional, Margin, Shares
    for stock in STOCKS:
        # OHLCV
        df_p = fetch_finmind_dataset("TaiwanStockPrice", stock)
        if not df_p.empty:
            df_p.to_csv(os.path.join(DATA_DIR, f"{stock}_ohlcv.csv"), index=False)
        time.sleep(0.5)

        # Institutional
        df_inst = fetch_finmind_dataset("TaiwanStockInstitutionalInvestorsBuySell", stock)
        if not df_inst.empty:
            df_inst.to_csv(os.path.join(DATA_DIR, f"{stock}_inst.csv"), index=False)
        time.sleep(0.5)

        # Margin
        df_margin = fetch_finmind_dataset("TaiwanStockMarginPurchaseShortSale", stock)
        if not df_margin.empty:
            df_margin.to_csv(os.path.join(DATA_DIR, f"{stock}_margin.csv"), index=False)
        time.sleep(0.5)

        # Shares
        df_share = fetch_finmind_dataset("TaiwanStockShareholding", stock)
        if not df_share.empty:
            df_share.to_csv(os.path.join(DATA_DIR, f"{stock}_shareholding.csv"), index=False)
        time.sleep(0.5)

    # 2. Benchmark TAIEX
    df_taiex = fetch_finmind_dataset("TaiwanStockPrice", "TAIEX")
    if not df_taiex.empty:
        df_taiex.to_csv(os.path.join(DATA_DIR, "taiex.csv"), index=False)
    time.sleep(0.5)

    # 3. FX USD/TWD
    df_fx = fetch_finmind_dataset("TaiwanExchangeRate", "USD")
    if not df_fx.empty:
        df_fx.to_csv(os.path.join(DATA_DIR, "fx_usdtwd.csv"), index=False)
    time.sleep(0.5)

    # 4. US Macro series
    df_sp = fetch_yahoo_series("%5EGSPC", "SP500")
    time.sleep(0.5)
    df_ndq = fetch_yahoo_series("%5EIXIC", "NASDAQ")
    time.sleep(0.5)
    df_sox = fetch_yahoo_series("%5ESOX", "SOX")
    time.sleep(0.5)
    df_vix = fetch_fred_series("VIXCLS")
    time.sleep(0.5)
    df_dgs10 = fetch_fred_series("DGS10")

    # Combine US macro into unified dataframe
    # Standardize index to date strings
    us_dates = pd.date_range(START_DATE, "2024-04-05", freq="B")
    df_us = pd.DataFrame(index=us_dates)
    if not df_sp.empty:
        df_sp_norm = df_sp.copy()
        df_sp_norm.index = df_sp.index.floor("D").tz_localize(None)
        df_us["sp500"] = df_sp_norm["close"]
    if not df_ndq.empty:
        df_ndq_norm = df_ndq.copy()
        df_ndq_norm.index = df_ndq.index.floor("D").tz_localize(None)
        df_us["nasdaq"] = df_ndq_norm["close"]
    if not df_sox.empty:
        df_sox_norm = df_sox.copy()
        df_sox_norm.index = df_sox.index.floor("D").tz_localize(None)
        df_us["sox"] = df_sox_norm["close"]
    if not df_vix.empty:
        df_vix_norm = df_vix.copy()
        df_vix_norm.index = df_vix.index.floor("D").tz_localize(None)
        df_us["vix"] = df_vix_norm["VIXCLS"]
    if not df_dgs10.empty:
        df_dgs_norm = df_dgs10.copy()
        df_dgs_norm.index = df_dgs10.index.floor("D").tz_localize(None)
        df_us["us_10y_yield"] = df_dgs_norm["DGS10"]

    df_us = df_us.ffill().bfill()
    df_us.to_csv(os.path.join(DATA_DIR, "us_macro.csv"))
    logger.info(f"Saved US macro table with shape {df_us.shape}.")

    # 5. MOPS Announcements
    mops_items = fetch_twse_mops_announcements()
    if mops_items:
        with open(os.path.join(DATA_DIR, "mops_announcements.json"), "w", encoding="utf-8") as f:
            json.dump(mops_items, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(mops_items)} MOPS announcements.")

    logger.info("Ingestion completed successfully.")


if __name__ == "__main__":
    run_ingestion()
