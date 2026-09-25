import json
import time

import pandas as pd
import requests
import yfinance as yf

from trend.config import DATA_ROOT, MODEL_DIR, START_DATE, STOCKS

LISTED_URL = 'https://openapi.twse.com.tw/v1/opendata/t187ap03_L'
RAW_DIR = DATA_ROOT / 'raw_prices'
UNIVERSE_DIR = DATA_ROOT / 'universe'
TOP_N = 50
POOL_RANK = 70
BATCH = 50


def listed_companies():
    rows = requests.get(LISTED_URL, timeout=60).json()
    df = pd.DataFrame(rows)
    df = df[df['公司代號'].str.fullmatch(r'\d{4}')]
    df['shares'] = pd.to_numeric(df['已發行普通股數或TDR原股發行股數'], errors='coerce')
    return df.set_index('公司代號')[['公司簡稱', 'shares']].rename(columns={'公司簡稱': 'name'}).dropna()


def download_raw_prices(codes, force=False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    todo = [c for c in codes if force or not (RAW_DIR / f'{c}.csv').exists()]
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        for attempt in range(3):
            try:
                df = yf.download([f'{c}.TW' for c in chunk], start=START_DATE, auto_adjust=False,
                                 group_by='ticker', progress=False, threads=True)
                break
            except Exception as e:
                print(f'  批次失敗：{e}', flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            continue
        for c in chunk:
            try:
                close = df[f'{c}.TW']['Close'].dropna()
            except KeyError:
                continue
            if not close.empty:
                close.index = close.index.tz_localize(None) if close.index.tz else close.index
                close.rename('close').to_csv(RAW_DIR / f'{c}.csv')
        print(f'[原始股價] {min(i + BATCH, len(todo))}/{len(todo)}', flush=True)


def quarter_ends(index):
    s = pd.Series(index, index=index)
    return s.groupby(s.dt.to_period('Q')).max().values


def rank_by_cap(closes, shares):
    cap = closes.mul(shares, axis=1)
    return cap.rank(axis=1, ascending=False)


def coarse_pool():
    companies = listed_companies()
    download_raw_prices(companies.index)
    closes = pd.DataFrame({c: pd.read_csv(RAW_DIR / f'{c}.csv', index_col=0, parse_dates=True)['close']
                           for c in companies.index if (RAW_DIR / f'{c}.csv').exists()}).sort_index()
    rebal = quarter_ends(closes.index)
    ranks = rank_by_cap(closes.loc[rebal].ffill(), companies['shares'].reindex(closes.columns))
    pool = sorted(ranks.columns[(ranks <= POOL_RANK).any()])
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    names = companies['name'].reindex(pool).to_dict()
    (MODEL_DIR / 'pool.json').write_text(json.dumps(names, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'候選池 {len(pool)} 檔', flush=True)
    return names


def load_pool():
    path = MODEL_DIR / 'pool.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def pool_names():
    return load_pool() or STOCKS


def current_members():
    members = load_membership()
    if members is None:
        return set(STOCKS)
    return set(members.columns[members.iloc[-1]])


def members_at(members, code, index):
    if members is None:
        return pd.Series(code in STOCKS, index=index)
    if code not in members:
        return pd.Series(False, index=index)
    return members[code].reindex(index).ffill().fillna(False).astype(bool)


def build_membership(load_dataset):
    pool = load_pool()
    closes, shares = {}, {}
    for code in pool:
        path = RAW_DIR / f'{code}.csv'
        holding = load_dataset('shareholding', code)
        if not path.exists() or holding is None or 'NumberOfSharesIssued' not in holding:
            continue
        closes[code] = pd.read_csv(path, index_col=0, parse_dates=True)['close']
        shares[code] = holding['NumberOfSharesIssued']
    closes = pd.DataFrame(closes).sort_index()
    shares = pd.DataFrame(shares).reindex(closes.index).ffill().bfill()
    cap = closes * shares
    rebal = quarter_ends(cap.index)
    ranks = cap.loc[rebal].rank(axis=1, ascending=False)
    members = (ranks <= TOP_N).reindex(cap.index).shift(1).ffill().fillna(False).astype(bool)
    members.to_csv(UNIVERSE_DIR / 'membership.csv')
    counts = members.sum(axis=1)
    print(f'成分股表 {members.index.min().date()} ~ {members.index.max().date()}，'
          f'每日 {counts[counts > 0].min()}～{counts.max()} 檔，曾入選 {int(members.any().sum())} 檔', flush=True)
    return members


def load_membership():
    path = UNIVERSE_DIR / 'membership.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = df.columns.astype(str)
    return df.astype(bool)


if __name__ == '__main__':
    import sys
    if '--members' in sys.argv:
        from trend.data import load_dataset, update_all
        update_all()
        download_raw_prices(list(load_pool()), force=True)
        build_membership(load_dataset)
    else:
        coarse_pool()
