import os

import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def download_history(symbol, start, end):
    df = yf.Ticker(symbol).history(start=start, end=end)
    if df.empty:
        return pd.DataFrame()

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, f'{symbol}.csv'))
    return df
