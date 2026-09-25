import numpy as np
import pandas as pd

from trend.config import HORIZON, MARKET
from trend.data import load, load_valuation
from trend.traders import TRADER_FEATURES, add_cross_sectional, stock_trader_features
from trend.universe import load_membership, members_at, pool_names

TECHNICAL = [
    'ret_5', 'ret_20', 'ret_60', 'ma_gap_20', 'ma_gap_60', 'rsi_14', 'macd_hist',
    'volatility_20', 'volume_ratio', 'hl_range', 'rel_strength_20',
]
EXPERT = [
    'faber_trend_200', 'mom_12_1',
    'per', 'pbr', 'dividend_yield', 'per_vs_3y', 'pbr_vs_3y',
    'breadth_200',
    'mkt_drawdown_250', 'mkt_trend_200',
    'high_52w_gap', 'vwap_gap_20',
]
FEATURES = TECHNICAL + EXPERT + TRADER_FEATURES
OPTIONAL = ('per', 'pbr', 'dividend_yield', 'per_vs_3y', 'pbr_vs_3y', 'rev_yoy')
REQUIRED = [f for f in FEATURES if f not in OPTIONAL]


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def price_features(df):
    close, volume = df['Close'], df['Volume']
    ret = close.pct_change()
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    vwap_20 = (close * volume).rolling(20).sum() / volume.rolling(20).sum()

    return pd.DataFrame({
        'ret_5': close.pct_change(5),
        'ret_20': close.pct_change(20),
        'ret_60': close.pct_change(60),
        'ma_gap_20': close / close.rolling(20).mean() - 1,
        'ma_gap_60': close / close.rolling(60).mean() - 1,
        'rsi_14': rsi(close),
        'macd_hist': (macd - macd.ewm(span=9, adjust=False).mean()) / close,
        'volatility_20': ret.rolling(20).std(),
        'volume_ratio': volume / volume.rolling(20).mean(),
        'hl_range': ((df['High'] - df['Low']) / close).rolling(20).mean(),
        'faber_trend_200': close / close.rolling(200).mean() - 1,
        'mom_12_1': close.shift(21) / close.shift(252) - 1,
        'high_52w_gap': close / close.rolling(252).max() - 1,
        'vwap_gap_20': close / vwap_20 - 1,
    }, index=df.index)


def valuation_features(code, index):
    val = load_valuation(code)
    if val is None or val.empty:
        return pd.DataFrame(index=index, columns=['per', 'pbr', 'dividend_yield', 'per_vs_3y', 'pbr_vs_3y'], dtype=float)

    val = val.reindex(index).ffill()
    per = val['PER'].where(val['PER'] > 0)
    pbr = val['PBR'].where(val['PBR'] > 0)
    return pd.DataFrame({
        'per': per,
        'pbr': pbr,
        'dividend_yield': val['dividend_yield'],
        'per_vs_3y': per / per.rolling(750, min_periods=250).median() - 1,
        'pbr_vs_3y': pbr / pbr.rolling(750, min_periods=250).median() - 1,
    }, index=index)


def market_features():
    close = load(MARKET)['Close']
    return pd.DataFrame({
        'mkt_ret_20': close.pct_change(20),
        'mkt_drawdown_250': close / close.rolling(250).max() - 1,
        'mkt_trend_200': close / close.rolling(200).mean() - 1,
        'mkt_ma_gap_50': close / close.rolling(50).mean() - 1,
    })


def build_dataset():
    market = market_features()
    frames = {}

    members = load_membership()
    for code in pool_names():
        df = load(f'{code}.TW')
        if df is None or len(df) < 300:
            continue
        member = members_at(members, code, df.index)
        if not member.any():
            continue
        feats = price_features(df).join(market, how='left')
        feats = feats.join(valuation_features(code, df.index))
        feats = feats.join(stock_trader_features(code, df))
        feats['rel_strength_20'] = feats['ret_20'] - feats['mkt_ret_20']
        feats['future_return'] = df['Close'].shift(-HORIZON) / df['Close'] - 1
        feats['close'] = df['Close']
        feats['code'] = code
        feats['member'] = member
        frames[code] = feats

    above_200 = pd.concat({c: (f['faber_trend_200'] > 0) & f['member'] for c, f in frames.items()}, axis=1)
    valid = pd.concat({c: f['faber_trend_200'].notna() & f['member'] for c, f in frames.items()}, axis=1)
    breadth = above_200.sum(axis=1).astype(float) / valid.sum(axis=1).replace(0, np.nan)

    data = pd.concat(frames.values())
    data.index.name = 'date'
    data = data[data['member']].drop(columns='member')
    data['breadth_200'] = breadth.reindex(data.index).values
    data = add_cross_sectional(data)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=REQUIRED)
    data['label'] = (data['future_return'] > 0).astype(int)

    data = data.sort_index()
    labeled = data[data['future_return'].notna()]
    latest = data[data.index == data.index.max()]
    return labeled, latest, data


def expert_views(row):
    views = {}

    views['Meb Faber'] = ('多', '股價在 200 日均線之上') if row.faber_trend_200 > 0 else ('空', '股價跌破 200 日均線')

    if pd.isna(row.per_vs_3y):
        views['Aswath Damodaran'] = ('—', '無估值資料')
    elif row.per_vs_3y < -0.15:
        views['Aswath Damodaran'] = ('多', f'本益比低於自身 3 年中位數 {abs(row.per_vs_3y):.0%}')
    elif row.per_vs_3y > 0.3:
        views['Aswath Damodaran'] = ('空', f'本益比高於自身 3 年中位數 {row.per_vs_3y:.0%}')
    else:
        views['Aswath Damodaran'] = ('中性', '本益比接近歷史水準')

    if row.breadth_200 > 0.6:
        views['Charlie Bilello'] = ('多', f'{row.breadth_200:.0%} 權值股站上 200 日線，市場廣度健康')
    elif row.breadth_200 < 0.4:
        views['Charlie Bilello'] = ('空', f'僅 {row.breadth_200:.0%} 權值股站上 200 日線')
    else:
        views['Charlie Bilello'] = ('中性', f'{row.breadth_200:.0%} 權值股站上 200 日線')

    if row.mkt_drawdown_250 < -0.15:
        views['Morgan Housel'] = ('多', f'大盤自高點回檔 {abs(row.mkt_drawdown_250):.0%}，長期投資者別恐慌')
    else:
        views['Morgan Housel'] = ('中性', '市場未明顯恐慌，維持長期紀律')

    if row.high_52w_gap > -0.03 and row.vwap_gap_20 > 0:
        views['Peter Brandt'] = ('多', '接近 52 週新高且站上 20 日 VWAP')
    elif row.high_52w_gap < -0.2 and row.vwap_gap_20 < 0:
        views['Peter Brandt'] = ('空', f'距 52 週高點 {abs(row.high_52w_gap):.0%} 且低於 VWAP')
    else:
        views['Peter Brandt'] = ('中性', '未出現明確突破')

    return views
