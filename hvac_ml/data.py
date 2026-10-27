"""
data.py - load and clean the three datasets.

  AI4I 2020      (UCI 601)  machine sensor readings and failures   -> Pillar 2, failure prediction
  Kaggle HVAC    (hourly)   weather, indoor temperature, power     -> Pillar 1, demand forecasting
  Online Retail II (UCI 502) customer purchases (about 1M lines)   -> Pillar 3, churn (stored as one row per invoice)

If a file is missing, load_ai4i() and the retail loader try to download it from UCI.
The Kaggle file has to be downloaded by hand (it needs a Kaggle login): see HELP.md.
"""
import io
import logging
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from hvac_ml import config

log = logging.getLogger(__name__)


def _download_zip(key, member_suffix, dest):
    """Download a UCI zip and extract the first file ending with `member_suffix` to `dest`."""
    url = config.DOWNLOAD_URLS[key]
    log.info("Downloading %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp:
        content = resp.read()
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(member_suffix))
        dest.write_bytes(z.read(name))


# Pillar 2: AI4I
def load_ai4i():
    if not config.AI4I_CSV.exists():
        _download_zip("ai4i", ".csv", config.AI4I_CSV)
    df = pd.read_csv(config.AI4I_CSV)
    df = df.rename(columns={
        "Air temperature [K]": "air_temp_k", "Process temperature [K]": "process_temp_k",
        "Rotational speed [rpm]": "rpm", "Torque [Nm]": "torque_nm", "Tool wear [min]": "tool_wear_min",
        "Machine failure": "failure"})
    assert len(df) == 10000 and df["failure"].isin([0, 1]).all() and df.isna().sum().sum() == 0
    return df


# Pillar 1: HVAC power
def load_hvac():
    if not config.HVAC_CSV.exists():
        raise FileNotFoundError(f"{config.HVAC_CSV} not found. Download it from {config.SOURCES['hvac']} (see HELP.md).")
    df = pd.read_csv(config.HVAC_CSV, parse_dates=["timestamp"])
    df = df.rename(columns={"Month": "month", "Hour": "hour", "Ambient_Temp": "ambient_c", "Indoor_Temp": "indoor_c",
                            "Humidity_pct": "humidity", "HVAC_Mode": "mode", "Power_kW": "power_kw"})
    assert len(df) == 8760 and df.isna().sum().sum() == 0
    return df


# Pillar 3: retail invoices
def _read_transactions():
    """Read the original Excel workbook (about 1 million lines, takes 1-2 minutes) and clean it."""
    if not config.RETAIL_XLSX.exists():
        _download_zip("retail", ".xlsx", config.RETAIL_XLSX)
    log.info("Reading %s", config.RETAIL_XLSX)
    raw = pd.concat(pd.read_excel(config.RETAIL_XLSX, sheet_name=None).values(), ignore_index=True)
    tx = raw.rename(columns={"Invoice": "invoice", "StockCode": "stock_code", "Quantity": "quantity", "InvoiceDate": "date",
                             "Price": "price", "Customer ID": "customer", "Country": "country"})
    tx = tx[["invoice", "stock_code", "quantity", "date", "price", "customer", "country"]].dropna(subset=["customer"])
    tx["customer"] = tx["customer"].astype(int)          # rows without a customer id cannot be followed
    tx["invoice"] = tx["invoice"].astype(str)
    tx["stock_code"] = tx["stock_code"].astype(str)
    tx["is_return"] = tx["invoice"].str.startswith("C")  # invoices starting with C are cancellations / returns
    tx["amount"] = tx["quantity"] * tx["price"]
    return tx[tx["price"] > 0]


def to_invoices(tx):
    """One row per invoice: customer, invoice, date, value, items, products, country, is_return."""
    parts = []
    for is_return, lines in ((False, tx[(~tx["is_return"]) & (tx["quantity"] > 0)]), (True, tx[tx["is_return"]])):
        inv = (lines.groupby(["customer", "invoice"])
               .agg(date=("date", "max"), value=("amount", "sum"), items=("quantity", "sum"),
                    products=("stock_code", "nunique"), country=("country", "first")).reset_index())
        inv["is_return"] = is_return
        parts.append(inv)
    return pd.concat(parts, ignore_index=True).sort_values(["customer", "date"]).reset_index(drop=True)


def load_invoices():
    """Invoice-level table used by Pillar 3. Built from the Excel file the first time, then read from a small CSV."""
    if config.RETAIL_INVOICES.exists():
        return pd.read_csv(config.RETAIL_INVOICES, parse_dates=["date"])
    inv = to_invoices(_read_transactions())
    inv.to_csv(config.RETAIL_INVOICES, index=False)
    log.info("Saved %s (%d invoices)", config.RETAIL_INVOICES, len(inv))
    return inv
