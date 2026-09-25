import numpy as np
import pandas as pd

from trend.data import load_dataset

FUNDAMENTAL = ['rev_yoy_3m', 'rev_accel', 'eps_ttm_yoy', 'gross_margin', 'gross_margin_chg', 'op_margin',
               'roe_ttm', 'debt_ratio', 'fcf_margin', 'per', 'per_vs_3y', 'dividend_yield']
CHIPS = ['foreign_20d', 'trust_20d', 'foreign_5d', 'margin_chg_20d', 'foreign_ratio_chg_20d']


def as_of(series, index):
    series = series.dropna()
    series = series[~series.index.duplicated(keep='last')]
    return series.reindex(series.index.union(index)).ffill().reindex(index)


def quarterly(code):
    income = load_dataset('income', code)
    if income is None:
        return None
    balance = load_dataset('balance', code)
    cashflow = load_dataset('cashflow', code)

    q = income.copy()
    for other in (balance, cashflow):
        if other is not None:
            q = q.join(other.drop(columns='quarter', errors='ignore'), how='left')
    q = q[~q.index.duplicated(keep='last')].sort_index()

    def col(name):
        return q[name] if name in q else pd.Series(np.nan, index=q.index)

    ttm = lambda s: s.rolling(4).sum()
    revenue_ttm = ttm(col('Revenue'))
    eps_ttm = ttm(col('EPS'))
    op_cash = col('CashFlowsFromOperatingActivities')
    capex = col('PropertyAndPlantAndEquipment').abs()
    gross_margin = col('GrossProfit') / col('Revenue')

    return pd.DataFrame({
        'eps_ttm_yoy': eps_ttm / eps_ttm.shift(4).where(eps_ttm.shift(4) > 0) - 1,
        'gross_margin': gross_margin,
        'gross_margin_chg': gross_margin - gross_margin.shift(4),
        'op_margin': col('OperatingIncome') / col('Revenue'),
        'roe_ttm': ttm(col('IncomeAfterTaxes')) / col('EquityAttributableToOwnersOfParent'),
        'debt_ratio': col('Liabilities') / col('TotalAssets'),
        'fcf_margin': (op_cash - capex) / col('Revenue'),
        'revenue_ttm': revenue_ttm,
    }, index=q.index)


def fundamental_features(code, index):
    out = pd.DataFrame(index=index)

    rev = load_dataset('revenue', code)
    if rev is not None:
        yoy = rev['revenue'] / rev['revenue'].shift(12) - 1
        yoy_3m = yoy.rolling(3).mean()
        out['rev_yoy_3m'] = as_of(yoy_3m, index)
        out['rev_accel'] = as_of(yoy_3m - yoy_3m.shift(3), index)

    q = quarterly(code)
    if q is not None:
        for c in ['eps_ttm_yoy', 'gross_margin', 'gross_margin_chg', 'op_margin', 'roe_ttm', 'debt_ratio', 'fcf_margin']:
            out[c] = as_of(q[c], index)

    val = load_dataset('valuation', code)
    if val is not None:
        per = val['PER'].where(val['PER'] > 0)
        out['per'] = as_of(per, index)
        out['per_vs_3y'] = as_of(per / per.rolling(750, min_periods=250).median() - 1, index)
        out['dividend_yield'] = as_of(val['dividend_yield'], index)

    return out.reindex(columns=FUNDAMENTAL)


def chip_features(code, index, volume):
    out = pd.DataFrame(index=index)
    avg_volume = volume.rolling(20).mean()

    inst = load_dataset('institutional', code)
    if inst is not None:
        inst = inst.reindex(index).fillna(0)
        for who in ('foreign', 'trust'):
            if who in inst:
                out[f'{who}_20d'] = inst[who].rolling(20).sum() / (avg_volume * 20)
        if 'foreign' in inst:
            out['foreign_5d'] = inst['foreign'].rolling(5).sum() / (avg_volume * 5)

    margin = load_dataset('margin', code)
    if margin is not None:
        bal = as_of(margin['MarginPurchaseTodayBalance'], index)
        out['margin_chg_20d'] = bal / bal.shift(20) - 1

    holding = load_dataset('shareholding', code)
    if holding is not None:
        ratio = as_of(holding['ForeignInvestmentSharesRatio'], index)
        out['foreign_ratio_chg_20d'] = ratio - ratio.shift(20)

    return out.reindex(columns=CHIPS).replace([np.inf, -np.inf], np.nan)
