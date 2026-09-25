import numpy as np
import pandas as pd

from trend.data import load
from trend.fundamentals import CHIPS, FUNDAMENTAL, chip_features, fundamental_features
from trend.universe import pool_names

COST_PCT = 0.001425 * 2 + 0.003
MIN_RISK, MAX_RISK = 0.02, 0.12

DEFAULT_EXIT = {'target_r': 3.0, 'trail': 'sma20', 'breakeven_r': 1.0, 'max_hold': 60}
TRAIL_NAMES = {'sma20': '20 日線', 'sma50': '50 日線', 'low20': '20 日低點', 'none': '不設移動停利'}

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
        'low20': low.rolling(20).min(), 'atr': tr.rolling(14).mean(), 'rsi_14': 100 - 100 / (1 + gain / loss),
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


def plan_trade(kind, entry, ind_row, exit_rule=DEFAULT_EXIT):
    if kind == 'breakout':
        stop = max(entry - 2 * ind_row.atr, ind_row.low10)
    else:
        stop = min(ind_row.sma50 * 0.97, entry - 1.5 * ind_row.atr)
    risk = entry - stop
    risk_pct = risk / entry
    if not (MIN_RISK <= risk_pct <= MAX_RISK):
        return None
    target = entry + exit_rule['target_r'] * risk if exit_rule['target_r'] else np.inf
    return {'stop': stop, 'target': target, 'risk': risk, 'risk_pct': risk_pct}


def simulate_exit(arrays, start, entry, plan, exit_rule=DEFAULT_EXIT):
    o, h, l, c, trail_line = arrays
    stop, target, risk = plan['stop'], plan['target'], plan['risk']
    trailing = False
    n = len(c)
    end = min(start + exit_rule['max_hold'], n - 1)
    for i in range(start, end + 1):
        if l[i] <= stop:
            return i, min(o[i], stop), '移動停損（保本）' if trailing and stop >= entry else '停損：跌破停損價'
        if h[i] >= target:
            return i, max(o[i], target), f'停利：達到 {exit_rule["target_r"]:.0f}R 目標價'
        if not trailing and h[i] >= entry + exit_rule['breakeven_r'] * risk:
            stop, trailing = max(stop, entry), True
        if trailing and trail_line is not None and c[i] < trail_line[i]:
            return i, c[i], f'移動停利：獲利後收盤跌破{TRAIL_NAMES[exit_rule["trail"]]}'
    if end == n - 1 and end - start < exit_rule['max_hold']:
        return None
    return end, c[end], f'時間出場：持有滿 {exit_rule["max_hold"]} 個交易日'


class SetupBook:
    def __init__(self, full):
        self.stocks, self.setups = {}, []
        names = pool_names()
        for code in full['code'].unique():
            df = load(f'{code}.TW')
            if df is None:
                continue
            ind = indicators(df)
            own = full[full['code'] == code]
            ctx = own.reindex(df.index)
            fund = fundamental_features(code, df.index)
            chips = chip_features(code, df.index, df['Volume'])
            self.stocks[code] = (df, ind)

            setups = find_setups(df, ind, ctx)
            for date, kind in setups[setups.index.isin(own.index)].items():
                pos = df.index.get_loc(date)
                self.setups.append({
                    'code': code, 'name': names.get(code, code), 'signal_date': date, 'setup': kind, 'pos': pos,
                    'close': df['Close'].iat[pos],
                    **ctx.loc[date, [c for c in TECHNICAL + REGIME if c in ctx]].to_dict(),
                    **fund.loc[date].to_dict(), **chips.loc[date].to_dict(),
                })

    def arrays(self, code, trail):
        df, ind = self.stocks[code]
        line = None if trail == 'none' else ind[trail].values
        return df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values, line

    def trades(self, exit_rule=DEFAULT_EXIT):
        trades, candidates = [], []
        cache = {}
        for s in self.setups:
            df, ind = self.stocks[s['code']]
            pos = s['pos']
            if pos + 1 >= len(df):
                plan = plan_trade(s['setup'], s['close'], ind.iloc[pos], exit_rule)
                if plan:
                    candidates.append({**s, **plan, 'entry': s['close']})
                continue

            entry = df['Open'].iat[pos + 1]
            plan = plan_trade(s['setup'], entry, ind.iloc[pos], exit_rule)
            if plan is None:
                continue
            if s['code'] not in cache:
                cache[s['code']] = self.arrays(s['code'], exit_rule['trail'])
            result = simulate_exit(cache[s['code']], pos + 1, entry, plan, exit_rule)
            still_open = result is None
            if still_open:
                exit_pos, exit_price, exit_reason = len(df) - 1, df['Close'].iat[-1], '持有中（以最新收盤價計）'
            else:
                exit_pos, exit_price, exit_reason = result
            pnl = exit_price / entry - 1 - COST_PCT
            trades.append({
                **s, **plan, 'entry_date': df.index[pos + 1], 'entry': entry,
                'exit_date': df.index[exit_pos], 'exit': exit_price, 'exit_reason': exit_reason,
                'days': exit_pos - pos - 1, 'return': pnl, 'r_multiple': pnl * entry / plan['risk'],
                'open': still_open,
            })

        trades = pd.DataFrame(trades)
        trades['is_breakout'] = (trades['setup'] == 'breakout').astype(int)
        trades['win'] = (trades['r_multiple'] > 0).astype(int)
        candidates = pd.DataFrame(candidates)
        if not candidates.empty:
            candidates['is_breakout'] = (candidates['setup'] == 'breakout').astype(int)
        return trades.sort_values('entry_date').reset_index(drop=True), candidates


def build_trades(full, exit_rule=DEFAULT_EXIT):
    return SetupBook(full).trades(exit_rule)


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
