import json

import pandas as pd
from flask import Flask, render_template, request
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from trend.config import MODEL_DIR, REPORT_DIR
from trend.traders import TRADERS
from utils import download_history

app = Flask(__name__)

EXPERTS = {
    'Meb Faber': '趨勢跟隨',
    'Aswath Damodaran': '估值',
    'Charlie Bilello': '市場廣度',
    'Morgan Housel': '長期心態',
    'Peter Brandt': '技術突破',
}
FEATURE_LABELS = {
    'mkt_drawdown_250': '大盤距一年高點（Housel）', 'faber_trend_200': '200 日均線乖離（Faber）',
    'high_52w_gap': '距 52 週高點（Brandt）', 'vwap_gap_20': '20 日 VWAP 乖離（Brandt）',
    'mom_12_1': '12-1 月動能（Faber）', 'breadth_200': '站上 200 日線比例（Bilello）',
    'mkt_trend_200': '大盤 200 日乖離（Housel）', 'per': '本益比（Damodaran）', 'pbr': '淨值比（Damodaran）',
    'dividend_yield': '殖利率（Damodaran）', 'per_vs_3y': '本益比 vs 3 年中位數（Damodaran）',
    'pbr_vs_3y': '淨值比 vs 3 年中位數（Damodaran）', 'ret_5': '5 日報酬', 'ret_20': '20 日報酬',
    'ret_60': '60 日報酬', 'ma_gap_20': '20 日均線乖離', 'ma_gap_60': '60 日均線乖離', 'rsi_14': 'RSI(14)',
    'macd_hist': 'MACD 柱狀體', 'volatility_20': '20 日波動率', 'volume_ratio': '量比',
    'hl_range': '平均振幅', 'rel_strength_20': '相對大盤強弱',
}

FEATURES = ['return', 'ma5_gap', 'ma10_gap', 'hl_range', 'oc_change', 'volume_change']


def read_csv(name, **kwargs):
    path = REPORT_DIR / name
    return pd.read_csv(path, **kwargs) if path.exists() else None


@app.route('/trend')
def trend():
    meta_path = MODEL_DIR / 'meta.json'
    if not meta_path.exists():
        return render_template('trend.html', meta=None)

    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    predictions = read_csv('predictions.csv', dtype={'code': str})
    history = read_csv('history.csv')
    folds = read_csv('folds.csv')
    importance = read_csv('importance.csv').head(10)
    importance['label'] = importance['feature'].map(FEATURE_LABELS).fillna(importance['feature'])

    stocks = []
    for _, r in predictions.iterrows():
        stocks.append({
            'code': r.code, 'name': r['name'], 'close': r.close, 'ret_20': r.ret_20, 'prob_up': r.prob_up,
            'views': [(e, r[f'{e}|signal'], r[f'{e}|reason']) for e in EXPERTS],
        })

    consensus = {e: predictions[f'{e}|signal'].value_counts().to_dict() for e in EXPERTS}
    return render_template('trend.html', meta=meta, stocks=stocks, experts=EXPERTS, consensus=consensus,
                           history=history.to_dict('records'), folds=folds.to_dict('records'),
                           importance=importance.to_dict('records'), data_date=predictions['date'].iloc[0])


def build_features(df):
    close = df['Close']
    df = df.assign(
        **{
            'return': close.pct_change(),
            'ma5_gap': close / close.rolling(5).mean() - 1,
            'ma10_gap': close / close.rolling(10).mean() - 1,
            'hl_range': (df['High'] - df['Low']) / close,
            'oc_change': (close - df['Open']) / df['Open'],
            'volume_change': df['Volume'].pct_change(),
            'next_close': close.shift(-1),
        }
    )
    df['target'] = df['next_close'] / close - 1
    return df.replace([float('inf'), float('-inf')], float('nan')).dropna(subset=FEATURES)


def new_model():
    return DecisionTreeRegressor(max_depth=5, min_samples_leaf=20, random_state=42)


def evaluate(labeled):
    train, test = train_test_split(labeled, test_size=0.2, shuffle=False)
    model = new_model().fit(train[FEATURES], train['target'])
    pred_return = model.predict(test[FEATURES])
    actual = test['next_close']
    predicted = test['Close'] * (1 + pred_return)
    return {
        'direction_acc': ((pred_return > 0) == (test['target'] > 0)).mean(),
        'mae': (actual - predicted).abs().mean(),
        'r2': r2_score(actual, predicted),
        'mse': mean_squared_error(actual, predicted),
        'n_train': len(train),
        'n_test': len(test),
    }


def predict_next(data):
    labeled = data.dropna(subset=['target'])
    model = new_model().fit(labeled[FEATURES], labeled['target'])
    last = data.iloc[[-1]]
    return float(last['Close'].iloc[0] * (1 + model.predict(last[FEATURES])[0]))


@app.route('/backtest')
def backtest():
    path = REPORT_DIR / 'backtest.json'
    if not path.exists():
        return render_template('backtest.html', report=None)

    report = json.loads(path.read_text(encoding='utf-8'))
    curves = {}
    for key in ('full', 'oos'):
        equity = read_csv(f'equity_{key}.csv', index_col=0)
        curves[key] = {'dates': equity.index.tolist(), 'series': {c: equity[c].round(4).tolist() for c in equity}}

    predictions = read_csv('predictions.csv', dtype={'code': str})
    picks = {t: predictions.loc[predictions[f'{t}|signal'] == '多', ['code', 'name']].to_dict('records')
             for t in TRADERS}
    return render_template('backtest.html', report=report, curves=curves, traders=TRADERS, picks=picks)


@app.route('/replay')
def replay():
    path = REPORT_DIR / 'self_train.json'
    report = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    return render_template('replay.html', report=report, traders=TRADERS)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['stock_symbol'].strip().upper()
    try:
        years = int(request.form['years'])
    except ValueError:
        return render_template('index.html', error='請輸入有效的年數')

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)
    df = download_history(symbol, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    if df.empty:
        return render_template('index.html', error=f'查無 {symbol} 的資料（台股請加 .TW，例如 2330.TW）')

    data = build_features(df)
    if len(data) < 100:
        return render_template('index.html', error='資料量不足，請拉長年數')

    last_close = float(data['Close'].iloc[-1])
    predicted = predict_next(data)
    recent = data['Close'].tail(60)

    return render_template(
        'result.html', symbol=symbol, last_date=data.index[-1].date(), last_close=last_close,
        predicted=predicted, change=predicted - last_close, change_pct=(predicted / last_close - 1) * 100,
        dates=[d.strftime('%Y-%m-%d') for d in recent.index], prices=recent.round(2).tolist(),
        metrics=evaluate(data.dropna(subset=['target'])),
    )


if __name__ == '__main__':
    app.run(debug=True)
