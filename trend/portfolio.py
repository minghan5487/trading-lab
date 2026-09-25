import numpy as np
import pandas as pd

RISK_PER_TRADE = 0.01
MAX_POSITIONS = 10
MAX_WEIGHT = 0.2


def simulate(trades, start_equity=1.0):
    trades = trades.sort_values('entry_date')
    equity = start_equity
    open_pos = []
    curve = []
    taken = []

    events = sorted(set(trades['entry_date']) | set(trades['exit_date']))
    by_entry = trades.groupby('entry_date')

    for day in events:
        still_open = []
        for pos in open_pos:
            if pos['exit_date'] <= day:
                equity += pos['size'] * pos['return']
            else:
                still_open.append(pos)
        open_pos = still_open

        if day in by_entry.groups:
            for _, t in by_entry.get_group(day).sort_values('prob', ascending=False).iterrows():
                if len(open_pos) >= MAX_POSITIONS or any(p['code'] == t.code for p in open_pos):
                    continue
                weight = min(RISK_PER_TRADE / t.risk_pct, MAX_WEIGHT)
                if sum(p['weight'] for p in open_pos) + weight > 1:
                    continue
                open_pos.append({'code': t.code, 'exit_date': t.exit_date, 'return': t['return'],
                                 'weight': weight, 'size': weight * equity})
                taken.append(t.name)
        curve.append((day, equity))

    curve = pd.Series(dict(curve)).sort_index()
    return curve, trades.loc[taken]


def curve_stats(curve):
    if len(curve) < 2:
        return {}
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    return {
        'total_return': float(curve.iloc[-1] / curve.iloc[0] - 1),
        'cagr': float((curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1) if years > 0 else None,
        'max_drawdown': float((curve / curve.cummax() - 1).min()),
    }


def benchmark(prices, start, end):
    p = prices[start:end].dropna(how='all')
    if len(p) < 2:
        return {}
    growth = (1 + p.pct_change().mean(axis=1).fillna(0)).cumprod()
    return curve_stats(growth)
