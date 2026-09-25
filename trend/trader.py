import numpy as np
import pandas as pd

from trend.config import MARKET, STOCKS
from trend.data import load
from trend.fundamentals import CHIPS, FUNDAMENTAL, chip_features, fundamental_features

COST_PCT = 0.001425 * 2 + 0.003
TARGET_R = 3.0
MAX_HOLD = 60
MIN_RISK, MAX_RISK = 0.02, 0.12

SETUPS = {
    'breakout': '趨勢突破：趨勢模板符合 7 項以上，放量突破 20 日高點（Minervini、Livermore、O\'Neil）',
    'pullback': '多頭回檔：站穩 200 日線、均線多頭排列，回測 50 日線後止跌（Steve Burns）',
}
TECHNICAL = ['rs_rating', 'minervini_score', 'high_52w_gap', 'volatility_20', 'volume_ratio', 'risk_pct', 'rsi_14']
REGIME = ['mkt_trend_200', 'mkt_drawdown_250', 'mkt_ret_20', 'breadth_200']
JUDGE_FEATURES = FUNDAMENTAL + CHIPS + TECHNICAL + REGIME + ['is_breakout']


def indicators(df):
    close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    return pd.DataFrame({
        'sma20': close.rolling(20).mean(), 'sma50': close.rolling(50).mean(), 'sma200': close.rolling(200).mean(),
        'atr': tr.rolling(14).mean(), 'rsi_14': 100 - 100 / (1 + gain / loss),
        'high20': close.shift(1).rolling(20).max(), 'low10': low.shift(1).rolling(10).min(),
        'vol50': volume.rolling(50).mean(), 'ret_5': close.pct_change(5),
    }, index=df.index)


def find_setups(df, ind, ctx):
    close, low, volume = df['Close'], df['Low'], df['Volume']
    breakout = (ctx['minervini_score'] >= 7) & (close > ind['high20']) & (volume > 1.5 * ind['vol50'])
    pullback = ((close > ind['sma200']) & (ind['sma50'] > ind['sma200']) & (low <= ind['sma50'] * 1.01)
                & (close > ind['sma50']) & (ind['ret_5'] < 0) & ind['rsi_14'].between(35, 55))
    kind = pd.Series(np.select([breakout, pullback], ['breakout', 'pullback'], ''), index=df.index)
    return kind[kind != '']


def plan_trade(kind, entry, ind_row):
    if kind == 'breakout':
        stop = max(entry - 2 * ind_row.atr, ind_row.low10)
    else:
        stop = min(ind_row.sma50 * 0.97, entry - 1.5 * ind_row.atr)
    risk = entry - stop
    risk_pct = risk / entry
    if not (MIN_RISK <= risk_pct <= MAX_RISK):
        return None
    return {'stop': stop, 'target': entry + TARGET_R * risk, 'risk': risk, 'risk_pct': risk_pct}


def simulate_exit(df, ind, start, entry, plan):
    stop, target, risk = plan['stop'], plan['target'], plan['risk']
    breakeven = False
    end = min(start + MAX_HOLD, len(df) - 1)
    for i in range(start, end + 1):
        o, h, l, c = df['Open'].iat[i], df['High'].iat[i], df['Low'].iat[i], df['Close'].iat[i]
        if l <= stop:
            price = min(o, stop)
            return i, price, '移動停損（保本）' if breakeven else '停損：跌破停損價'
        if h >= target:
            return i, max(o, target), f'停利：達到 {TARGET_R:.0f}R 目標價'
        if not breakeven and h >= entry + risk:
            stop, breakeven = max(stop, entry), True
        if breakeven and c < ind['sma20'].iat[i]:
            return i, c, '移動停利：獲利後收盤跌破 20 日線'
    if end == len(df) - 1 and end - start < MAX_HOLD:
        return None
    return end, df['Close'].iat[end], f'時間出場：持有滿 {MAX_HOLD} 個交易日'


def build_trades(full):
    market = load(MARKET)
    trades, open_setups = [], []
    for code in STOCKS:
        df = load(f'{code}.TW')
        if df is None:
            continue
        ind = indicators(df)
        ctx = full[full['code'] == code].reindex(df.index)
        fund = fundamental_features(code, df.index)
        chips = chip_features(code, df.index, df['Volume'])

        for date, kind in find_setups(df, ind, ctx).items():
            pos = df.index.get_loc(date)
            row = {'code': code, 'name': STOCKS[code], 'signal_date': date, 'setup': kind,
                   'close': df['Close'].iat[pos],
                   **ctx.loc[date, [c for c in TECHNICAL + REGIME if c in ctx]].to_dict(),
                   **fund.loc[date].to_dict(), **chips.loc[date].to_dict()}

            if pos + 1 >= len(df):
                plan = plan_trade(kind, row['close'], ind.iloc[pos])
                if plan:
                    open_setups.append({**row, **plan, 'entry': row['close'], 'risk_pct': plan['risk_pct']})
                continue

            entry = df['Open'].iat[pos + 1]
            plan = plan_trade(kind, entry, ind.iloc[pos])
            if plan is None:
                continue
            exit_info = simulate_exit(df, ind, pos + 1, entry, plan)
            if exit_info is None:
                continue
            exit_pos, exit_price, exit_reason = exit_info
            pnl = exit_price / entry - 1 - COST_PCT
            trades.append({
                **row, **plan, 'entry_date': df.index[pos + 1], 'entry': entry,
                'exit_date': df.index[exit_pos], 'exit': exit_price, 'exit_reason': exit_reason,
                'days': exit_pos - pos - 1, 'return': pnl, 'r_multiple': pnl * entry / plan['risk'],
            })

    trades = pd.DataFrame(trades)
    trades['is_breakout'] = (trades['setup'] == 'breakout').astype(int)
    trades['win'] = (trades['r_multiple'] > 0).astype(int)
    candidates = pd.DataFrame(open_setups)
    if not candidates.empty:
        candidates['is_breakout'] = (candidates['setup'] == 'breakout').astype(int)
    return trades.sort_values('entry_date').reset_index(drop=True), candidates


def checklist(row):
    def fmt(v, spec):
        return '—' if pd.isna(v) else format(v, spec)

    items = [
        ('基本面', '營收成長', row.rev_yoy_3m > 0.15, f'近 3 月營收年增 {fmt(row.rev_yoy_3m, "+.0%")}'),
        ('基本面', '營收加速', row.rev_accel > 0, f'營收年增率較前季變化 {fmt(row.rev_accel, "+.0%")}'),
        ('基本面', 'EPS 成長', row.eps_ttm_yoy > 0.15, f'近四季 EPS 年增 {fmt(row.eps_ttm_yoy, "+.0%")}'),
        ('基本面', '毛利率提升', row.gross_margin_chg > 0, f'毛利率 {fmt(row.gross_margin, ".1%")}，年變化 {fmt(row.gross_margin_chg, "+.1%")}'),
        ('基本面', 'ROE 佳', row.roe_ttm > 0.12, f'ROE {fmt(row.roe_ttm, ".1%")}'),
        ('基本面', '自由現金流為正', row.fcf_margin > 0, f'自由現金流率 {fmt(row.fcf_margin, ".1%")}'),
        ('基本面', '估值合理', row.per_vs_3y < 0.3, f'本益比 {fmt(row.per, ".1f")}，較 3 年中位數 {fmt(row.per_vs_3y, "+.0%")}'),
        ('籌碼面', '外資買超', row.foreign_20d > 0, f'外資 20 日買賣超佔成交量 {fmt(row.foreign_20d, "+.1%")}'),
        ('籌碼面', '投信買超', row.trust_20d > 0, f'投信 20 日買賣超佔成交量 {fmt(row.trust_20d, "+.1%")}'),
        ('籌碼面', '融資未過熱', not (row.margin_chg_20d > 0.2), f'融資 20 日變化 {fmt(row.margin_chg_20d, "+.0%")}'),
        ('技術面', '相對強勢', row.rs_rating >= 80, f'相對強度 {fmt(row.rs_rating, ".0f")}（前 {fmt(100 - row.rs_rating, ".0f")}%）'),
        ('技術面', '接近新高', row.high_52w_gap > -0.1, f'距 52 週高點 {fmt(row.high_52w_gap, "+.0%")}'),
        ('大盤', '大盤多頭', row.mkt_trend_200 > 0, f'加權指數較 200 日線 {fmt(row.mkt_trend_200, "+.0%")}'),
        ('大盤', '市場廣度佳', row.breadth_200 > 0.5, f'{fmt(row.breadth_200, ".0%")} 權值股站上 200 日線'),
    ]
    return [{'group': g, 'name': n, 'ok': bool(ok) if not pd.isna(ok) else False, 'detail': d} for g, n, ok, d in items]
