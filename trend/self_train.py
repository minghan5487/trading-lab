import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from trend.config import HORIZON, REPORT_DIR, STOCKS
from trend.traders import TRADERS, trader_signals

EXAM_YEARS = 2
ROUNDS = 500
BATCH = 64
REFIT_EVERY = 25
EXAM_SIZE = 3000
SEED = 42

REGIME = ['mkt_trend_200', 'mkt_drawdown_250', 'mkt_ret_20', 'breadth_200', 'volatility_20']
INDICATORS = ['minervini_score', 'rs_rating', 'rev_yoy', 'up_down_volume', 'breakout_gap', 'high_52w_gap']


def trader_votes(data):
    signals = trader_signals(data).astype(int)
    return pd.DataFrame({
        'Mark Minervini': np.select([data['minervini_score'] >= 8, data['minervini_score'] < 5], [1, -1], 0),
        "William O'Neil": signals["William O'Neil"] * 2 - 1,
        'Jesse Livermore': signals['Jesse Livermore'],
        'Steve Burns': signals['Steve Burns'] * 2 - 1,
    }, index=data.index)


def design_matrix(data):
    votes = trader_votes(data)
    X = votes.copy()
    for col in REGIME + INDICATORS:
        X[col] = data[col].values
    return X, votes


def new_model():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=100,
                                          random_state=SEED)


def grade(model, X, y, votes):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    vote = np.sign(votes.sum(axis=1).values)
    decided = vote != 0
    return {
        'accuracy': float((pred == y).mean()),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'auc': float(roc_auc_score(y, prob)),
        'always_up': float(y.mean()),
        'vote_accuracy': float(((vote[decided] > 0).astype(int) == y[decided]).mean()),
        'vote_balanced': float(balanced_accuracy_score(y[decided], (vote[decided] > 0).astype(int))),
    }


def replay(X, y, pool, rng, exam=None):
    order = rng.permutation(pool)
    memory, practice_curve, exam_curve = [], [], []
    model = None

    for r in range(1, ROUNDS + 1):
        batch = order[(r - 1) * BATCH:r * BATCH]
        if model is not None:
            practice_curve.append({'round': r, 'accuracy': float((model.predict(X.iloc[batch]) == y[batch]).mean())})
        memory.extend(batch)

        if r % REFIT_EVERY == 0:
            model = new_model().fit(X.iloc[memory], y[memory])
            if exam is not None:
                result = grade(model, *exam)
                exam_curve.append({'round': r, 'memory': len(memory), **{k: round(result[k], 4) for k in
                                                                        ('accuracy', 'balanced_accuracy', 'auc')}})
    return model, practice_curve, exam_curve


def run(labeled, latest):
    rng = np.random.default_rng(SEED)
    X, votes = design_matrix(labeled)
    y = labeled['label'].values
    dates = labeled.index

    exam_start = dates.max() - pd.DateOffset(years=EXAM_YEARS)
    unique = np.unique(dates.values)
    practice_end = unique[np.searchsorted(unique, np.datetime64(exam_start)) - HORIZON]

    practice_pool = np.flatnonzero(np.asarray(dates <= practice_end))
    exam_idx = rng.choice(np.flatnonzero(np.asarray(dates >= exam_start)), size=EXAM_SIZE, replace=False)
    exam = (X.iloc[exam_idx], y[exam_idx], votes.iloc[exam_idx])

    model, practice_curve, exam_curve = replay(X, y, practice_pool, rng, exam=exam)
    final = grade(model, *exam)

    today_model, _, _ = replay(X, y, np.arange(len(X)), rng)
    X_latest, votes_latest = design_matrix(latest)
    today = pd.DataFrame({
        'code': latest['code'].values, 'name': latest['code'].map(STOCKS).values,
        'prob_up': today_model.predict_proba(X_latest)[:, 1].round(4),
        'vote': np.sign(votes_latest.sum(axis=1).values),
        **{t: votes_latest[t].values for t in TRADERS},
    }).sort_values('prob_up', ascending=False)

    smooth = pd.Series([p['accuracy'] for p in practice_curve]).rolling(25, min_periods=1).mean()
    report = {
        'practice': {
            'rounds': ROUNDS, 'batch': BATCH, 'start': str(dates.min().date()),
            'end': str(pd.Timestamp(practice_end).date()),
            'curve': [{'round': p['round'], 'accuracy': round(s, 4)} for p, s in zip(practice_curve, smooth)],
        },
        'exam': {'start': str(exam_start.date()), 'end': str(dates.max().date()), 'size': EXAM_SIZE,
                 'curve': exam_curve, 'final': final,
                 'best': max(exam_curve, key=lambda c: c['auc'])},
        'date': str(latest.index.max().date()),
        'today': today.to_dict('records'),
    }
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / 'self_train.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float),
                                                encoding='utf-8')
    return report
