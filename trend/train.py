import json
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trend.backtest import run_all as run_backtest
from trend.config import HORIZON, MODEL_DIR, REPORT_DIR, STOCKS
from trend.data import update_all
from trend.features import EXPERT, FEATURES, TECHNICAL, build_dataset, expert_views
from trend.self_train import run as run_self_train
from trend.traders import trader_views

N_FOLDS = 5
TEST_DAYS = 250
REPLACE_TOLERANCE = 0.005


def candidates():
    return {
        'logistic': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                  LogisticRegression(max_iter=1000)),
        'random_forest': make_pipeline(SimpleImputer(strategy='median'),
                                       RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=200,
                                                              n_jobs=-1, random_state=42)),
        'gradient_boosting': HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                                            min_samples_leaf=200, l2_regularization=1.0,
                                                            random_state=42),
    }


def walk_forward_splits(dates):
    unique = np.sort(dates.unique())
    for k in range(N_FOLDS, 0, -1):
        test_end = len(unique) - (k - 1) * TEST_DAYS
        test_start = test_end - TEST_DAYS
        train_end = test_start - HORIZON
        if train_end < 500:
            continue
        yield unique[:train_end], unique[test_start:test_end]


def evaluate(model, data, features, keep_predictions=False):
    scores, predictions = [], []
    for train_dates, test_dates in walk_forward_splits(data.index):
        train = data[data.index.isin(train_dates)]
        test = data[data.index.isin(test_dates)]

        model.fit(train[features], train['label'])
        prob = model.predict_proba(test[features])[:, 1]
        if keep_predictions:
            predictions.append(pd.DataFrame({'date': test.index, 'code': test['code'].values, 'prob_up': prob}))

        majority = int(train['label'].mean() >= 0.5)
        scores.append({
            'period': f'{pd.Timestamp(test_dates[0]).date()} ~ {pd.Timestamp(test_dates[-1]).date()}',
            'auc': roc_auc_score(test['label'], prob),
            'accuracy': accuracy_score(test['label'], prob >= 0.5),
            'baseline': accuracy_score(test['label'], np.full(len(test), majority)),
            'top_quintile_return': test.loc[prob >= np.quantile(prob, 0.8), 'future_return'].mean(),
            'avg_return': test['future_return'].mean(),
        })
    scores = pd.DataFrame(scores)
    return (scores, pd.concat(predictions, ignore_index=True)) if keep_predictions else scores


def feature_importance(model, data):
    dates = np.sort(data.index.unique())
    split = dates[-TEST_DAYS - HORIZON]
    train, test = data[data.index < split], data[data.index >= dates[-TEST_DAYS]]
    model.fit(train[FEATURES], train['label'])
    result = permutation_importance(model, test[FEATURES], test['label'], scoring='roc_auc',
                                    n_repeats=3, random_state=42, n_jobs=-1)
    return pd.Series(result.importances_mean, index=FEATURES).sort_values(ascending=False)


def load_meta():
    path = MODEL_DIR / 'meta.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def main(skip_download=False, force=False):
    if not skip_download:
        update_all(force=force)

    data, latest, full = build_dataset()
    print(f'資料 {len(data):,} 筆，{data["code"].nunique()} 檔，{data.index.min().date()} ~ {data.index.max().date()}')

    results, oos_by_model = {}, {}
    for name, model in candidates().items():
        results[name], oos_by_model[name] = evaluate(model, data, FEATURES, keep_predictions=True)
        f = results[name]
        print(f'{name:18s} AUC {f.auc.mean():.4f}  準確率 {f.accuracy.mean():.2%}  基準 {f.baseline.mean():.2%}')

    best_name = max(results, key=lambda n: results[n].auc.mean())
    best, oos = results[best_name], oos_by_model[best_name]
    new_auc = best.auc.mean()

    technical_only = evaluate(candidates()[best_name], data, TECHNICAL)
    without_traders = evaluate(candidates()[best_name], data, TECHNICAL + EXPERT)
    print(f'{best_name} 僅技術指標 AUC {technical_only.auc.mean():.4f}，'
          f'技術+專家 {without_traders.auc.mean():.4f}，全部 {new_auc:.4f}')

    meta = load_meta()
    has_model = (MODEL_DIR / 'model.joblib').exists()
    same_features = meta is not None and meta.get('features') == FEATURES
    replaced = not (has_model and same_features) or new_auc >= meta['auc'] - REPLACE_TOLERANCE
    run_time = datetime.now().strftime('%Y-%m-%d %H:%M')

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    if replaced:
        importance = feature_importance(candidates()[best_name], data)
        importance.rename('importance').to_csv(REPORT_DIR / 'importance.csv', index_label='feature')

        model = candidates()[best_name].fit(data[FEATURES], data['label'])
        joblib.dump(model, MODEL_DIR / 'model.joblib')
        meta = {
            'model': best_name, 'auc': round(new_auc, 4), 'accuracy': round(best.accuracy.mean(), 4),
            'baseline': round(best.baseline.mean(), 4), 'auc_technical_only': round(technical_only.auc.mean(), 4),
            'auc_without_traders': round(without_traders.auc.mean(), 4),
            'top_quintile_return': round(best.top_quintile_return.mean(), 4),
            'avg_return': round(best.avg_return.mean(), 4),
            'trained_at': run_time, 'horizon': HORIZON, 'data_end': str(data.index.max().date()),
            'samples': len(data), 'features': FEATURES,
        }
        (MODEL_DIR / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        best.to_csv(REPORT_DIR / 'folds.csv', index=False)
        print(f'採用新模型 {best_name}（AUC {new_auc:.4f}）')
    else:
        print(f'新模型 AUC {new_auc:.4f} 低於現有 {meta["auc"]:.4f}，保留舊模型')

    history_path = REPORT_DIR / 'history.csv'
    row = pd.DataFrame([{
        'run_at': run_time, 'data_end': str(latest.index.max().date()), 'best_model': best_name,
        'auc': round(new_auc, 4), 'accuracy': round(best.accuracy.mean(), 4),
        'baseline': round(best.baseline.mean(), 4), 'auc_technical_only': round(technical_only.auc.mean(), 4),
        'replaced': replaced, **{f'auc_{n}': round(r.auc.mean(), 4) for n, r in results.items()},
    }])
    history = pd.concat([pd.read_csv(history_path), row]) if history_path.exists() else row
    history.to_csv(history_path, index=False)

    predictions = predict_latest(latest)
    predictions.to_csv(REPORT_DIR / 'predictions.csv')

    backtest = run_backtest(full, oos)
    for name, m in backtest['full']['strategies'].items():
        print(f'{name:14s} 年化 {m["cagr"]:+.1%}  最大回檔 {m["max_drawdown"]:.1%}  夏普 {m["sharpe"]:.2f}')

    replay = run_self_train(data, latest)
    exam = replay['exam']['final']
    print(f'AI 復盤訓練：考試準確率 {exam["accuracy"]:.1%}（全猜漲 {exam["always_up"]:.1%}），AUC {exam["auc"]:.3f}')

    write_summary(meta, predictions, history, backtest)


def predict_latest(latest):
    model = joblib.load(MODEL_DIR / 'model.joblib')
    out = latest[['code', 'close', 'ret_20']].copy()
    out['name'] = out['code'].map(STOCKS)
    out['prob_up'] = model.predict_proba(latest[FEATURES])[:, 1].round(4)

    views = []
    for _, row in latest.iterrows():
        opinions = {**expert_views(row), **trader_views(row)}
        views.append({f'{who}|{key}': value
                      for who, (signal, reason) in opinions.items()
                      for key, value in (('signal', signal), ('reason', reason))})
    out = pd.concat([out, pd.DataFrame(views, index=out.index)], axis=1)
    return out.sort_values('prob_up', ascending=False)


def backtest_table(window):
    lines = [f'期間 {window["start"]} ~ {window["end"]}，已扣手續費與證交稅', '',
             '| 策略 | 總報酬 | 年化報酬 | 最大回檔 | 夏普值 |', '|---|---|---|---|---|']
    for name, m in window['strategies'].items():
        lines.append(f'| {name} | {m["total_return"]:+.0%} | {m["cagr"]:+.1%} | {m["max_drawdown"]:.1%} | {m["sharpe"]:.2f} |')
    return lines


def write_summary(meta, predictions, history, backtest):
    date = predictions.index.max().date()
    people = ['Mark Minervini', "William O'Neil", 'Jesse Livermore', 'Steve Burns',
              'Meb Faber', 'Aswath Damodaran', 'Morgan Housel', 'Peter Brandt']
    lines = [
        f'# 走勢預測報告（{date}）',
        '',
        f'預測 0050 成分股未來 {meta["horizon"]} 個交易日上漲的機率。此報告由 GitHub Actions 每個交易日自動產生。',
        '',
        f'- 模型：{meta["model"]}（訓練時間 {meta["trained_at"]}，資料至 {meta["data_end"]}）',
        f'- Walk-forward 驗證：AUC {meta["auc"]:.3f}、準確率 {meta["accuracy"]:.1%}、基準（全猜多數類）{meta["baseline"]:.1%}',
        f'- AUC：僅技術指標 {meta["auc_technical_only"]:.3f} → 加專家因子 {meta.get("auc_without_traders", meta["auc"]):.3f}'
        f' → 再加交易員規則 {meta["auc"]:.3f}',
        f'- 模型挑出的前 20% 股票，未來 {meta["horizon"]} 日平均報酬 {meta["top_quintile_return"]:+.2%}（全體平均 {meta["avg_return"]:+.2%}）',
        '',
        '## 回測：完整期間',
        '',
        *backtest_table(backtest['full']),
        '',
        '## 回測：模型樣本外期間（含機器學習策略）',
        '',
        *backtest_table(backtest['oos']),
        '',
        '## 上漲機率前 10 名',
        '',
        '| 代碼 | 名稱 | 上漲機率 | ' + ' | '.join(p.split(' ')[-1] for p in people) + ' |',
        '|---|---|---|' + '---|' * len(people),
    ]
    for _, r in predictions.head(10).iterrows():
        signals = ' | '.join(r[f'{p}|signal'] for p in people)
        lines.append(f'| {r.code} | {r["name"]} | {r.prob_up:.1%} | {signals} |')

    lines += ['', '## 最近 10 次訓練', '', '| 執行時間 | 資料日期 | 最佳模型 | AUC | 準確率 | 更新模型 |',
              '|---|---|---|---|---|---|']
    for _, r in history.tail(10).iloc[::-1].iterrows():
        lines.append(f'| {r.run_at} | {r.data_end} | {r.best_model} | {r.auc:.3f} | {r.accuracy:.1%} | '
                     f'{"是" if r.replaced else "否"} |')

    lines += ['', '> 僅為機器學習練習，不構成投資建議。專家觀點為依其公開方法論量化的規則，並非本人意見。']
    (REPORT_DIR / 'latest.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main(skip_download='--skip-download' in sys.argv, force='--force' in sys.argv)
