"""Download adjusted closes once and pin them in data/prices.csv so every later run is identical."""
from pathlib import Path
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
START, END = "2022-01-01", "2026-10-31"  # END is exclusive in yfinance


def main():
    px = yf.download(["EWH", "EWZ"], start=START, end=END, auto_adjust=True, progress=False)["Close"]
    px = px[["EWH", "EWZ"]].dropna().round(6)
    px.index.name = "date"
    px.to_csv(ROOT / "data" / "prices.csv")
    print(px.iloc[[0, -1]], len(px))


if __name__ == "__main__":
    main()
