import itertools
import json
import time

import pandas as pd

from trend.config import MODEL_DIR, REPORT_DIR, STOCKS
from trend.data import load
from trend.features import build_dataset
from trend.judge import walk_forward
from trend.portfolio import benchmark, curve_stats, simulate
from trend.trader import SetupBook

EXIT_GRID = {
    'target_r': [3.0, 5.0, None],
    'trail': ['sma20', 'sma50', 'low20'],
    'breakeven_r': [1.0, 2.0],
    'max_hold': [60, 120, 250],
}
SIZING_GRID = {'risk': [0.01, 0.02, 0.03], 'max_positions': [5, 10], 'max_weight': [0.25]}
MAX_DRAWDOWN_LIMIT = -0.35
STRATEGY_PATH = MODEL_DIR / 'strategy.json'


def grid(spec):
    keys = list(spec)
    return [dict(zip(keys, values)) for values in itertools.product(*spec.values())]


def price_panel():
    return pd.DataFrame({c: load(f'{c}.TW')['Close'] for c in STOCKS}).sort_index()


def score(stats):
    if not stats or stats['cagr'] is None or stats['max_drawdown'] < MAX_DRAWDOWN_LIMIT:
        return float('-inf')
    return stats['calmar'] or float('-inf')


def run():
    t0 = time.time()
    _, _, full = build_dataset()
    book = SetupBook(full)
    prices = price_panel()
    print(f'型態 {len(book.setups):,} 個，準備 {time.time() - t0:.0f} 秒', flush=True)

    results, cache = [], {}
    exit_rules = grid(EXIT_GRID)
    for i, rule in enumerate(exit_rules, 1):
        trades, _ = book.trades(rule)
        tested, holdout, holdout_start = walk_forward(trades)
        cache[json.dumps(rule)] = (tested, holdout, holdout_start)
        for use_judge in (True, False):
            pool = tested[tested['prob'] >= tested['threshold']] if use_judge else tested
            for sizing in grid(SIZING_GRID):
                curve, taken, exposure = simulate(pool, sizing)
                stats = curve_stats(curve)
                results.append({'exit': rule, 'sizing': sizing, 'use_judge': use_judge, 'dev': stats,
                                'exposure': exposure, 'trades': len(taken), 'score': score(stats)})
        best = max(results, key=lambda r: r['score'])
        print(f'[{i}/{len(exit_rules)}] 目前最佳 年化 {best["dev"]["cagr"]:+.1%} 回檔 {best["dev"]["max_drawdown"]:.1%} '
              f'（{time.time() - t0:.0f} 秒）', flush=True)

    results.sort(key=lambda r: r['score'], reverse=True)
    best = results[0]
    tested, holdout, holdout_start = cache[json.dumps(best['exit'])]
    dev_start, dev_end = tested['entry_date'].min(), holdout_start

    pool = holdout[holdout['prob'] >= holdout['threshold']] if best['use_judge'] else holdout
    hold_curve, hold_taken, hold_exposure = simulate(pool, best['sizing'])
    holdout_stats = curve_stats(hold_curve)
    hold_end = holdout['exit_date'].max()

    strategy = {
        'exit': best['exit'], 'sizing': best['sizing'], 'use_judge': best['use_judge'],
        'selected_at': time.strftime('%Y-%m-%d %H:%M'), 'configs_tested': len(results),
        'dev': {**best['dev'], 'start': str(dev_start.date()), 'end': str(dev_end.date()),
                'exposure': best['exposure'], 'trades': best['trades'],
                'benchmark': benchmark(prices, dev_start, dev_end)},
        'holdout': {**holdout_stats, 'start': str(holdout_start.date()), 'end': str(hold_end.date()),
                    'exposure': hold_exposure, 'trades': len(hold_taken),
                    'benchmark': benchmark(prices, holdout_start, hold_end)},
        'leaderboard': [{k: r[k] for k in ('exit', 'sizing', 'use_judge', 'dev', 'exposure', 'trades')}
                        for r in results[:10]],
        'baseline': next(r for r in results if r['exit'] == {'target_r': 3.0, 'trail': 'sma20', 'breakeven_r': 1.0,
                                                               'max_hold': 60}
                         and r['sizing'] == {'risk': 0.01, 'max_positions': 10, 'max_weight': 0.25}
                         and r['use_judge']),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    STRATEGY_PATH.write_text(json.dumps(strategy, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print(json.dumps({k: strategy[k] for k in ('exit', 'sizing', 'use_judge', 'dev', 'holdout')},
                     ensure_ascii=False, indent=1, default=str))
    print(f'共 {time.time() - t0:.0f} 秒')
    return strategy


def recheck():
    strategy = load_strategy()
    _, _, full = build_dataset()
    book = SetupBook(full)
    prices = price_panel()
    rows = []
    for rank, cfg in enumerate(strategy['leaderboard'], 1):
        trades, _ = book.trades(cfg['exit'])
        _, holdout, holdout_start = walk_forward(trades)
        pool = holdout[holdout['prob'] >= holdout['threshold']] if cfg['use_judge'] else holdout
        curve, taken, exposure = simulate(pool, cfg['sizing'])
        stats = curve_stats(curve)
        end = prices.index.max()
        stats.update({'start': str(holdout_start.date()), 'end': str(end.date()), 'exposure': exposure,
                      'trades': len(taken), 'still_open': int(taken['open'].sum()),
                      'benchmark': benchmark(prices, holdout_start, end)})
        rows.append({**cfg, 'holdout': stats})
        print(f'#{rank} 驗收 {cfg["dev"]["cagr"]:+.1%}/{cfg["dev"]["max_drawdown"]:.1%} → 期末考 '
              f'{stats["total_return"]:+.1%}（回檔 {stats["max_drawdown"]:.1%}，{len(taken)} 筆，其中 {stats["still_open"]} 筆持有中）'
              f'｜同期平均持有 {stats["benchmark"]["total_return"]:+.1%}', flush=True)
    strategy['holdout'] = rows[0]['holdout']
    strategy['leaderboard'] = rows
    STRATEGY_PATH.write_text(json.dumps(strategy, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    return rows


def load_strategy():
    if STRATEGY_PATH.exists():
        return json.loads(STRATEGY_PATH.read_text(encoding='utf-8'))
    return None


if __name__ == '__main__':
    import sys
    recheck() if '--recheck' in sys.argv else run()
