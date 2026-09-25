import json

import numpy as np
import pandas as pd

from trend.config import MARKET, REPORT_DIR, STOCKS
from trend.data import load
from trend.traders import TRADERS, trader_signals

ONE_WAY_COST = (0.001425 * 2 + 0.003) / 2
TRADING_DAYS = 252
REBALANCE_DAYS = 5
ML_TOP_N = 10
ML_HOLD_DAYS = 20


def price_panel():
    closes = {code: load(f'{code}.TW')['Close'] for code in STOCKS}
    return pd.DataFrame(closes).sort_index()


def rebalance(signal, every):
    held = signal.astype(float)
    held.iloc[np.arange(len(held)) % every != 0] = np.nan
    return held.ffill().fillna(0) > 0


def run(holdings, returns):
    holdings = holdings.reindex(returns.index).fillna(False).astype(bool)
    weights = holdings.div(holdings.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    weights = weights.shift(1).fillna(0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    daily = (weights * returns.fillna(0)).sum(axis=1) - turnover * ONE_WAY_COST
    return daily, weights


def metrics(daily, weights=None):
    equity = (1 + daily).cumprod()
    years = len(daily) / TRADING_DAYS
    vol = daily.std() * np.sqrt(TRADING_DAYS)
    result = {
        'total_return': equity.iloc[-1] - 1,
        'cagr': equity.iloc[-1] ** (1 / years) - 1,
        'volatility': vol,
        'sharpe': daily.mean() * TRADING_DAYS / vol if vol else np.nan,
        'max_drawdown': (equity / equity.cummax() - 1).min(),
    }
    if weights is not None:
        result['exposure'] = (weights.sum(axis=1) > 0).mean()
        result['avg_holdings'] = (weights > 0).sum(axis=1)[weights.sum(axis=1) > 0].mean()
        result['turnover'] = weights.diff().abs().sum(axis=1).sum() / 2 / years
    return result


def ml_holdings(oos, index, columns):
    prob = oos.pivot_table(index='date', columns='code', values='prob_up').reindex(columns=columns)
    dates = prob.index
    held = pd.DataFrame(False, index=index, columns=columns)
    for i in range(0, len(dates), ML_HOLD_DAYS):
        top = prob.loc[dates[i]].nlargest(ML_TOP_N).index
        end = dates[min(i + ML_HOLD_DAYS, len(dates)) - 1]
        held.loc[dates[i]:end, top] = True
    return held


def run_all(data, oos):
    closes = price_panel()
    returns = closes.pct_change()
    market = load(MARKET)['Close'].reindex(closes.index).pct_change()

    signals = trader_signals(data)
    signals['code'] = data['code']

    strategies, weights_by = {}, {}
    for name in TRADERS:
        panel = signals.pivot_table(index=signals.index, columns='code', values=name, aggfunc='last')
        panel = panel.reindex(index=closes.index, columns=closes.columns).fillna(False).astype(bool)
        every = 1 if name == 'Jesse Livermore' else REBALANCE_DAYS
        strategies[name], weights_by[name] = run(rebalance(panel, every), returns)

    strategies['機器學習（前 10 名）'], weights_by['機器學習（前 10 名）'] = run(
        ml_holdings(oos, closes.index, closes.columns), returns)
    strategies['50 檔平均持有'] = returns.mean(axis=1).fillna(0)
    strategies['加權指數'] = market.fillna(0)

    start = signals.index.min() + pd.Timedelta(days=30)
    oos_start, oos_end = oos['date'].min(), oos['date'].max()
    windows = {
        'full': (start, closes.index.max()),
        'oos': (oos_start, oos_end),
    }

    report = {}
    for key, (a, b) in windows.items():
        report[key] = {'start': str(a.date()), 'end': str(b.date()), 'strategies': {}}
        for name, daily in strategies.items():
            if key == 'full' and name.startswith('機器學習'):
                continue
            w = weights_by.get(name)
            report[key]['strategies'][name] = metrics(daily[a:b], None if w is None else w[a:b])

    equity = pd.DataFrame({n: (1 + d[start:]).cumprod() for n, d in strategies.items() if not n.startswith('機器學習')})
    oos_equity = pd.DataFrame({n: (1 + d[oos_start:oos_end]).cumprod() for n, d in strategies.items()})

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / 'backtest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float),
                                              encoding='utf-8')
    equity.resample('W').last().round(4).to_csv(REPORT_DIR / 'equity_full.csv')
    oos_equity.resample('W').last().round(4).to_csv(REPORT_DIR / 'equity_oos.csv')
    return report
