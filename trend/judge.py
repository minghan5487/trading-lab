import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trend.trader import JUDGE_FEATURES, checklist

LABELS = {
    'rev_yoy_3m': '營收年增', 'rev_accel': '營收加速', 'eps_ttm_yoy': 'EPS 成長', 'gross_margin': '毛利率',
    'gross_margin_chg': '毛利率變化', 'op_margin': '營益率', 'roe_ttm': 'ROE', 'debt_ratio': '負債比',
    'fcf_margin': '自由現金流', 'per': '本益比', 'per_vs_3y': '本益比相對歷史', 'dividend_yield': '殖利率',
    'foreign_20d': '外資 20 日買超', 'trust_20d': '投信 20 日買超', 'foreign_5d': '外資 5 日買超',
    'margin_chg_20d': '融資變化', 'foreign_ratio_chg_20d': '外資持股變化', 'rs_rating': '相對強度',
    'minervini_score': '趨勢模板分數', 'high_52w_gap': '距 52 週高點', 'volatility_20': '波動度',
    'volume_ratio': '量比', 'risk_pct': '停損距離', 'rsi_14': 'RSI', 'mkt_trend_200': '大盤乖離',
    'mkt_drawdown_250': '大盤回檔', 'mkt_ret_20': '大盤 20 日漲跌', 'breadth_200': '市場廣度',
    'is_breakout': '突破型態',
}
HOLDOUT_MONTHS = 6
FIRST_TEST_YEAR = 2016


def new_judge():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000))


def explain(model, row, top=3):
    imputer, scaler, lr = model.named_steps.values()
    z = scaler.transform(imputer.transform(pd.DataFrame([row[JUDGE_FEATURES]], columns=JUDGE_FEATURES)))[0]
    contrib = pd.Series(lr.coef_[0] * z, index=JUDGE_FEATURES).sort_values()
    pros = [f'{LABELS[f]}（+{v:.2f}）' for f, v in contrib[::-1].items() if v > 0.05][:top]
    cons = [f'{LABELS[f]}（{v:.2f}）' for f, v in contrib.items() if v < -0.05][:top]
    return pros, cons


def summarize(trades):
    if trades.empty:
        return {'trades': 0}
    return {
        'trades': int(len(trades)), 'win_rate': float(trades['win'].mean()),
        'avg_r': float(trades['r_multiple'].mean()), 'avg_return': float(trades['return'].mean()),
        'avg_win': float(trades.loc[trades['win'] == 1, 'return'].mean()) if trades['win'].any() else 0.0,
        'avg_loss': float(trades.loc[trades['win'] == 0, 'return'].mean()) if (trades['win'] == 0).any() else 0.0,
        'avg_days': float(trades['days'].mean()),
    }


def walk_forward(trades):
    holdout_start = trades['entry_date'].max() - pd.DateOffset(months=HOLDOUT_MONTHS)
    development = trades[trades['entry_date'] < holdout_start]
    tested = []

    for year in range(FIRST_TEST_YEAR, holdout_start.year + 1):
        start = pd.Timestamp(year=year, month=1, day=1)
        train = development[development['exit_date'] < start]
        test = development[(development['entry_date'] >= start) & (development['entry_date'].dt.year == year)]
        if len(train) < 300 or test.empty:
            continue
        model = new_judge().fit(train[JUDGE_FEATURES], train['win'])
        tested.append(test.assign(prob=model.predict_proba(test[JUDGE_FEATURES])[:, 1],
                                  threshold=train['win'].mean(), year=year))
    tested = pd.concat(tested)

    final_train = trades[trades['exit_date'] < holdout_start]
    holdout = trades[trades['entry_date'] >= holdout_start]
    model = new_judge().fit(final_train[JUDGE_FEATURES], final_train['win'])
    holdout = holdout.assign(prob=model.predict_proba(holdout[JUDGE_FEATURES])[:, 1],
                             threshold=final_train['win'].mean())
    return tested, holdout, holdout_start


def evaluate(tested):
    accepted = tested[tested['prob'] >= tested['threshold']]
    by_year = []
    for year, g in tested.groupby('year'):
        acc = g[g['prob'] >= g['threshold']]
        by_year.append({'year': int(year), 'all': summarize(g), 'accepted': summarize(acc)})
    return {
        'auc': float(roc_auc_score(tested['win'], tested['prob'])) if tested['win'].nunique() > 1 else None,
        'all': summarize(tested), 'accepted': summarize(accepted), 'by_year': by_year,
    }


def reason_stats(tested):
    rows = []
    checks = [checklist(r) for r in tested.itertuples()]
    for i, item in enumerate(checks[0]):
        ok = np.array([c[i]['ok'] for c in checks])
        yes, no = tested[ok], tested[~ok]
        rows.append({
            'group': item['group'], 'name': item['name'], 'share': float(ok.mean()),
            'win_yes': float(yes['win'].mean()) if len(yes) else None,
            'win_no': float(no['win'].mean()) if len(no) else None,
            'r_yes': float(yes['r_multiple'].mean()) if len(yes) else None,
            'r_no': float(no['r_multiple'].mean()) if len(no) else None,
        })
    return rows
