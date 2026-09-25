import numpy as np
import pandas as pd

from trend.data import load_revenue

TRADERS = {
    'Mark Minervini': '趨勢模板 8 條件全部符合',
    "William O'Neil": 'CAN SLIM：營收成長、近新高、量能、強勢股、大盤多頭',
    'Jesse Livermore': '突破 20 日高點且放量買進，跌破 10 日低點賣出',
    'Steve Burns': '站上 200 日線且 50 日線在 200 日線之上',
}
TRADER_FEATURES = ['minervini_score', 'rs_rating', 'low_52w_gap', 'rev_yoy', 'up_down_volume',
                   'livermore_hold', 'breakout_gap', 'burns_trend']


def revenue_yoy(code, index):
    rev = load_revenue(code)
    if rev is None or rev.empty:
        return pd.Series(np.nan, index=index)
    rev = rev.sort_index()
    yoy = rev['revenue'] / rev['revenue'].shift(12) - 1
    yoy_3m = yoy.rolling(3).mean()
    return yoy_3m.reindex(index.union(yoy_3m.index)).ffill().reindex(index)


def livermore_position(close, volume):
    prior_high = close.shift(1).rolling(20).max()
    prior_low = close.shift(1).rolling(10).min()
    entry = (close > prior_high) & (volume > 1.5 * volume.rolling(50).mean())
    exit_ = close < prior_low

    holding = np.zeros(len(close), dtype=int)
    state = 0
    for i, (e, x) in enumerate(zip(entry.values, exit_.values)):
        if state == 0 and e:
            state = 1
        elif state == 1 and x:
            state = 0
        holding[i] = state
    return pd.Series(holding, index=close.index)


def stock_trader_features(code, df):
    close, volume = df['Close'], df['Volume']
    sma50, sma150, sma200 = (close.rolling(n).mean() for n in (50, 150, 200))
    high_252, low_252 = close.rolling(252).max(), close.rolling(252).min()
    up = close.diff() > 0

    minervini_base = (
        (close > sma150).astype(int) + (close > sma200).astype(int) + (sma150 > sma200).astype(int)
        + (sma200 > sma200.shift(21)).astype(int) + ((sma50 > sma150) & (sma50 > sma200)).astype(int)
        + (close > sma50).astype(int) + (close >= low_252 * 1.3).astype(int) + (close >= high_252 * 0.75).astype(int)
    )

    return pd.DataFrame({
        'minervini_base': minervini_base.where(sma200.shift(21).notna()),
        'rs_raw': (0.4 * close.pct_change(63) + 0.2 * close.pct_change(126)
                   + 0.2 * close.pct_change(189) + 0.2 * close.pct_change(252)),
        'low_52w_gap': close / low_252 - 1,
        'rev_yoy': revenue_yoy(code, df.index),
        'up_down_volume': (volume.where(up, 0).rolling(50).sum() / volume.where(~up, 0).rolling(50).sum()),
        'livermore_hold': livermore_position(close, volume),
        'breakout_gap': close / close.shift(1).rolling(20).max() - 1,
        'burns_trend': ((close > sma200) & (sma50 > sma200)).astype(int).where(sma200.notna()),
    }, index=df.index)


def add_cross_sectional(data):
    data['rs_rating'] = data.groupby(level=0)['rs_raw'].rank(pct=True) * 100
    data['minervini_score'] = data['minervini_base'] + (data['rs_rating'] >= 70).astype(int)
    return data


def trader_signals(data):
    market_up = (data['mkt_trend_200'] > 0) & (data['mkt_ma_gap_50'] > 0)
    return pd.DataFrame({
        'Mark Minervini': data['minervini_score'] >= 8,
        "William O'Neil": ((data['rev_yoy'] > 0.2) & (data['high_52w_gap'] > -0.15) & (data['up_down_volume'] > 1)
                           & (data['rs_rating'] >= 80) & market_up),
        'Jesse Livermore': data['livermore_hold'] == 1,
        'Steve Burns': data['burns_trend'] == 1,
    }, index=data.index)


def trader_views(row):
    views = {}
    score = int(row.minervini_score) if pd.notna(row.minervini_score) else 0
    views['Mark Minervini'] = ('多', '趨勢模板 8/8 全數符合') if score >= 8 else (
        ('中性', f'趨勢模板符合 {score}/8') if score >= 5 else ('空', f'趨勢模板僅符合 {score}/8'))

    checks = [
        ('營收年增>20%', pd.notna(row.rev_yoy) and row.rev_yoy > 0.2),
        ('距新高15%內', row.high_52w_gap > -0.15),
        ('上漲量>下跌量', row.up_down_volume > 1),
        ('相對強度≥80', row.rs_rating >= 80),
        ('大盤多頭', row.mkt_trend_200 > 0 and row.mkt_ma_gap_50 > 0),
    ]
    passed = sum(ok for _, ok in checks)
    missing = '、'.join(name for name, ok in checks if not ok)
    views["William O'Neil"] = ('多', 'CAN SLIM 5 項全數符合') if passed == 5 else (
        ('中性' if passed >= 3 else '空'), f'符合 {passed}/5，未符合：{missing}')

    views['Jesse Livermore'] = ('多', '已突破關鍵點，持有中') if row.livermore_hold == 1 else (
        '中性', f'未突破，距 20 日高點 {row.breakout_gap:+.1%}')

    views['Steve Burns'] = ('多', '站上 200 日線且均線多頭排列') if row.burns_trend == 1 else ('空', '趨勢未確立')
    return views
