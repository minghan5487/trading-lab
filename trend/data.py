import json
import os
import time
from datetime import date

import pandas as pd
import requests
import yfinance as yf

from trend.config import DATA_ROOT, MARKET, START_DATE
from trend.universe import current_members, pool_names

INACTIVE_MAX_AGE = 30

FINMIND_URL = 'https://api.finmindtrade.com/api/v4/data'
PRICE_DIR = DATA_ROOT / 'prices'

INCOME_ITEMS = ['Revenue', 'GrossProfit', 'OperatingIncome', 'IncomeAfterTaxes', 'EPS',
                'IncomeFromContinuingOperations', 'TotalConsolidatedProfitForThePeriod']
BALANCE_ITEMS = ['EquityAttributableToOwnersOfParent', 'TotalAssets', 'Liabilities']
CASHFLOW_ITEMS = ['CashFlowsFromOperatingActivities', 'PropertyAndPlantAndEquipment']
INSTITUTIONS = {'Foreign_Investor': 'foreign', 'Investment_Trust': 'trust', 'Dealer_self': 'dealer'}


class RateLimited(Exception):
    pass


def finmind(dataset, code, start=START_DATE):
    params = {'dataset': dataset, 'data_id': code, 'start_date': start}
    if os.environ.get('FINMIND_TOKEN'):
        params['token'] = os.environ['FINMIND_TOKEN']
    resp = requests.get(FINMIND_URL, params=params, timeout=60)
    body = resp.json() if resp.headers.get('content-type', '').startswith('application/json') else {}
    if resp.status_code in (402, 429) or 'data' not in body:
        raise RateLimited(body.get('msg', resp.status_code))
    df = pd.DataFrame(body['data'])
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df


def statement_available(quarter_end):
    return quarter_end + pd.to_timedelta(quarter_end.dt.month.eq(12).map({True: 90, False: 45}), unit='D')


def wide_statement(dataset, items):
    def fetch(code):
        df = finmind(dataset, code)
        if df.empty:
            return df
        df = df[df['type'].isin(items)].pivot_table(index='date', columns='type', values='value')
        df = df.reset_index()
        df['available'] = statement_available(df['date'])
        return df.rename(columns={'date': 'quarter'}).set_index('available')
    return fetch


def fetch_valuation(code):
    df = finmind('TaiwanStockPER', code)
    return df if df.empty else df.set_index('date')[['PER', 'PBR', 'dividend_yield']]


def fetch_revenue(code):
    df = finmind('TaiwanStockMonthRevenue', code)
    if df.empty:
        return df
    df['available'] = df['date'] + pd.Timedelta(days=10)
    return df.set_index('available')[['revenue_year', 'revenue_month', 'revenue']]


def fetch_institutional(code):
    df = finmind('TaiwanStockInstitutionalInvestorsBuySell', code)
    if df.empty:
        return df
    df = df[df['name'].isin(INSTITUTIONS)].assign(net=lambda d: d['buy'] - d['sell'])
    return df.pivot_table(index='date', columns='name', values='net', aggfunc='sum').rename(columns=INSTITUTIONS)


def fetch_margin(code):
    df = finmind('TaiwanStockMarginPurchaseShortSale', code)
    return df if df.empty else df.set_index('date')[['MarginPurchaseTodayBalance', 'ShortSaleTodayBalance']]


def fetch_shareholding(code):
    df = finmind('TaiwanStockShareholding', code)
    return df if df.empty else df.set_index('date')[['ForeignInvestmentSharesRatio', 'NumberOfSharesIssued']]


DATASETS = {
    'valuation': (fetch_valuation, 1),
    'institutional': (fetch_institutional, 1),
    'margin': (fetch_margin, 1),
    'shareholding': (fetch_shareholding, 1),
    'revenue': (fetch_revenue, 7),
    'income': (wide_statement('TaiwanStockFinancialStatements', INCOME_ITEMS), 7),
    'balance': (wide_statement('TaiwanStockBalanceSheet', BALANCE_ITEMS), 7),
    'cashflow': (wide_statement('TaiwanStockCashFlowsStatement', CASHFLOW_ITEMS), 7),
}


UPDATED = DATA_ROOT / 'updated.json'


def load_updated():
    return json.loads(UPDATED.read_text(encoding='utf-8')) if UPDATED.exists() else {}


def mark_updated(key):
    updated = load_updated()
    updated[key] = date.today().isoformat()
    UPDATED.write_text(json.dumps(updated, indent=0), encoding='utf-8')


def age_days(path):
    stamp = load_updated().get(str(path.relative_to(DATA_ROOT)))
    if not path.exists() or stamp is None:
        return None
    return (date.today() - date.fromisoformat(stamp)).days


def download_price(symbol, retries=3):
    for attempt in range(retries):
        try:
            df = yf.Ticker(symbol).history(start=START_DATE, auto_adjust=True)
            if not df.empty:
                df.index = df.index.tz_localize(None).normalize()
                return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            print(f'  {symbol} 第 {attempt + 1} 次失敗：{e}')
        time.sleep(2 * (attempt + 1))
    return pd.DataFrame()


def price_path(symbol):
    return PRICE_DIR / f"{symbol.replace('^', '')}.csv"


def update_prices(force=False):
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    symbols = [MARKET] + [f'{code}.TW' for code in pool_names()]
    for i, symbol in enumerate(symbols, 1):
        path = price_path(symbol)
        if not force and age_days(path) == 0:
            continue
        df = download_price(symbol)
        if df.empty:
            print(f'[股價 {i}/{len(symbols)}] {symbol} 下載失敗，沿用舊資料')
            continue
        df.to_csv(path)
        mark_updated(str(path.relative_to(DATA_ROOT)))
        print(f'[股價 {i}/{len(symbols)}] {symbol} 最新 {df.index[-1].date()}', flush=True)


def update_finmind(force=False, max_wait_minutes=None):
    max_wait_minutes = max_wait_minutes or int(os.environ.get('FINMIND_MAX_WAIT', 90))
    waited = 0
    codes = list(pool_names())
    active = current_members()
    for name, (fetch, max_age) in DATASETS.items():
        folder = DATA_ROOT / name
        folder.mkdir(parents=True, exist_ok=True)
        for i, code in enumerate(codes, 1):
            path = folder / f'{code}.csv'
            age = age_days(path)
            limit = max_age if code in active else max(max_age, INACTIVE_MAX_AGE)
            if not force and age is not None and age < limit:
                continue
            while True:
                try:
                    fetch(code).to_csv(path)
                    mark_updated(str(path.relative_to(DATA_ROOT)))
                    print(f'[{name} {i}/{len(codes)}] {code}', flush=True)
                    break
                except RateLimited as e:
                    if waited >= max_wait_minutes:
                        print(f'已達 FinMind 流量上限（{e}），其餘沿用舊資料')
                        return
                    print(f'FinMind 流量上限（{e}），等待 5 分鐘', flush=True)
                    time.sleep(300)
                    waited += 5
                except Exception as e:
                    print(f'[{name} {i}/{len(codes)}] {code} 下載失敗：{e}')
                    break
            time.sleep(0.3)


def update_all(force=False):
    update_prices(force)
    update_finmind(force)


def load(symbol):
    path = price_path(symbol)
    return pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else None


def load_dataset(name, code):
    path = DATA_ROOT / name / f'{code}.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return None if df.empty else df.sort_index()


def load_valuation(code):
    return load_dataset('valuation', code)


def load_revenue(code):
    return load_dataset('revenue', code)


if __name__ == '__main__':
    import sys
    update_all(force='--force' in sys.argv)
