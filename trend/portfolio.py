import numpy as np
import pandas as pd

from trend.config import STOCKS
from trend.data import load

DEFAULT_SIZING = {'risk': 0.01, 'max_positions': 10, 'max_weight': 0.2}

_PRICES = None


def prices():
    global _PRICES
    if _PRICES is None:
        _PRICES = pd.DataFrame({c: load(f'{c}.TW')['Close'] for c in STOCKS}).sort_index()
    return _PRICES


def position_weight(risk_pct, sizing=DEFAULT_SIZING):
    return min(sizing['risk'] / risk_pct, sizing['max_weight'])


def simulate(trades, sizing=DEFAULT_SIZING, start_equity=1.0):
    if trades.empty:
        return pd.Series(dtype=float), trades, 0.0
    panel = prices()
    days = panel.index[(panel.index >= trades['entry_date'].min()) & (panel.index <= trades['exit_date'].max())]
    col = {c: i for i, c in enumerate(panel.columns)}
    close = panel.loc[days].ffill().values

    entries = {d: g.sort_values('prob', ascending=False) for d, g in trades.groupby('entry_date')}
    cash, positions, taken = start_equity, [], []
    curve, exposure = np.empty(len(days)), np.empty(len(days))

    for i, day in enumerate(days):
        still = []
        for p in positions:
            if p['exit_date'] <= day:
                cash += p['size'] * (1 + p['return'])
            else:
                still.append(p)
        positions = still

        equity = cash + sum(p['size'] * close[i, p['col']] / p['entry'] for p in positions)
        if day in entries:
            for _, t in entries[day].iterrows():
                if len(positions) >= sizing['max_positions'] or any(p['code'] == t.code for p in positions):
                    continue
                size = position_weight(t.risk_pct, sizing) * equity
                if size > cash:
                    continue
                cash -= size
                positions.append({'code': t.code, 'col': col[t.code], 'entry': t.entry, 'size': size,
                                  'exit_date': t.exit_date, 'return': t['return']})
                taken.append(t.name)

        invested = sum(p['size'] * close[i, p['col']] / p['entry'] for p in positions)
        curve[i] = cash + invested
        exposure[i] = invested / curve[i] if curve[i] > 0 else 0

    return pd.Series(curve, index=days), trades.loc[taken], float(exposure.mean())


def curve_stats(curve):
    if len(curve) < 2:
        return {}
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1 if years > 0 else None
    drawdown = (curve / curve.cummax() - 1).min()
    return {
        'total_return': float(curve.iloc[-1] / curve.iloc[0] - 1),
        'cagr': float(cagr) if cagr is not None else None,
        'max_drawdown': float(drawdown),
        'calmar': float(cagr / abs(drawdown)) if cagr is not None and drawdown < 0 else None,
    }


def benchmark(prices_df, start, end):
    p = prices_df[start:end].dropna(how='all')
    if len(p) < 2:
        return {}
    return curve_stats((1 + p.pct_change().mean(axis=1).fillna(0)).cumprod())
