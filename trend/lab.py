import json
import sys
from datetime import datetime

import pandas as pd

from trend.config import MARKET, REPORT_DIR, STOCKS
from trend.data import load, update_all
from trend.features import build_dataset
from trend.judge import (JUDGE_FEATURES, evaluate, explain, new_judge, reason_stats, summarize, walk_forward)
from trend.optimize import load_strategy
from trend.portfolio import DEFAULT_SIZING, benchmark, curve_stats, position_weight, simulate
from trend.trader import DEFAULT_EXIT, SETUPS, TRAIL_NAMES, SetupBook, checklist, plan_trade, simulate_exit

JOURNAL = REPORT_DIR / 'journal.csv'


def price_panel():
    return pd.DataFrame({c: load(f'{c}.TW')['Close'] for c in STOCKS}).sort_index()


def similar_stats(history, setup, prob):
    same = history[(history['setup'] == setup) & (history['prob'].sub(prob).abs() <= 0.05)]
    return summarize(same)


def update_journal(accepted, today):
    journal = pd.read_csv(JOURNAL, parse_dates=['signal_date'], dtype={'code': str}) if JOURNAL.exists() else pd.DataFrame()
    new = accepted.assign(signal_date=today)[['signal_date', 'code', 'name', 'setup', 'prob']]
    if not journal.empty:
        new = new[~new.set_index(['signal_date', 'code']).index.isin(journal.set_index(['signal_date', 'code']).index)]
    journal = pd.concat([journal, new], ignore_index=True)
    journal.to_csv(JOURNAL, index=False)
    return journal


def journal_status(journal, book, exit_rule):
    rows = []
    for j in journal.itertuples():
        df, ind = book.stocks[j.code]
        pos = df.index.searchsorted(j.signal_date)
        if pos + 1 >= len(df):
            rows.append({**j._asdict(), 'status': '等待進場'})
            continue
        entry = df['Open'].iat[pos + 1]
        plan = plan_trade(j.setup, entry, ind.iloc[pos], exit_rule)
        if plan is None:
            rows.append({**j._asdict(), 'status': '放棄（停損距離不合理）'})
            continue
        result = simulate_exit(book.arrays(j.code, exit_rule['trail']), pos + 1, entry, plan, exit_rule)
        base = {**j._asdict(), 'entry_date': df.index[pos + 1].date(), 'entry': entry, **plan}
        if result is None:
            last = df['Close'].iat[-1]
            rows.append({**base, 'status': '持有中', 'price': last, 'r_now': (last - entry) / plan['risk']})
        else:
            i, price, reason = result
            rows.append({**base, 'status': '已出場', 'exit_date': df.index[i].date(), 'exit': price,
                         'exit_reason': reason, 'r_now': (price - entry) / plan['risk']})
    return pd.DataFrame(rows)


def pct(v, spec='+.1%'):
    return '—' if v is None or pd.isna(v) else format(v, spec)


def write_report(ctx):
    m, meta = ctx['market'], ctx['meta']
    L = [f'# 交易日報（{ctx["date"]}）', '',
         f'產生時間 {ctx["run_at"]}・不構成投資建議・[回報問題](../../issues/new/choose)', '',
         '## 大盤', '',
         f'- 加權指數 {m["close"]:,.0f}，距 200 日線 {pct(m["trend_200"])}、距一年高點 {pct(m["drawdown"])}',
         f'- 市場廣度：{pct(m["breadth"], ".0%")} 權值股站上 200 日線', '']

    ex, sz = ctx['exit_rule'], ctx['sizing']
    L += [f'## 今日交易計畫（{len(ctx["plans"])} 筆）', '',
          f'出場規則：' + (f'{ex["target_r"]:.0f}R 停利、' if ex['target_r'] else '不設固定停利、') +
          f'獲利 {ex["breakeven_r"]:.0f}R 後停損移到進場價、之後收盤跌破{TRAIL_NAMES[ex["trail"]]}出場、最多持有 {ex["max_hold"]} 天；'
          f'單筆風險 {sz["risk"]:.0%}、最多 {sz["max_positions"]} 檔', '']
    if not ctx['plans']:
        L += ['今天沒有通過判官的進場訊號。', '']
    for p in ctx['plans']:
        target = (f'{p["target"]:,.2f}（{pct(p["risk_pct"] * ex["target_r"])}，{ex["target_r"]:.0f} 倍風險）'
                  if ex['target_r'] else f'不設固定目標，靠移動停利出場')
        L += [f'### {p["code"]} {p["name"]}｜{p["setup_name"]}｜預估勝率 {p["prob"]:.0%}', '',
              f'- **進場**：明日開盤，參考價 {p["entry"]:,.2f}',
              f'- **停損**：{p["stop"]:,.2f}（{pct(-p["risk_pct"])}）',
              f'- **目標**：{target}',
              f'- **部位**：單筆風險 {sz["risk"]:.0%}，約佔資金 {p["weight"]:.0%}',
              f'- **型態**：{SETUPS[p["setup"]]}',
              f'- **加分理由**：' + ('、'.join(p['pros']) or '無'),
              f'- **扣分理由**：' + ('、'.join(p['cons']) or '無'), '']
        for group in ('基本面', '籌碼面', '技術面', '大盤'):
            items = [c for c in p['checklist'] if c['group'] == group]
            L.append(f'  - {group}：' + '；'.join(f'{"✅" if c["ok"] else "❌"} {c["detail"]}' for c in items))
        s = p['similar']
        if s.get('trades'):
            L.append(f'  - 歷史類似訊號 {s["trades"]} 筆：勝率 {s["win_rate"]:.0%}，平均 {s["avg_r"]:+.2f}R，平均持有 {s["avg_days"]:.0f} 天')
        L.append('')

    if ctx['watch']:
        L += ['## 觀察名單（型態成立但判官不通過）', '',
              '| 股票 | 型態 | 預估勝率 | 主要扣分 |', '|---|---|---|---|']
        L += [f'| {w["code"]} {w["name"]} | {w["setup_name"][:4]} | {w["prob"]:.0%} | {"、".join(w["cons"][:2])} |'
              for w in ctx['watch']]
        L.append('')

    j = ctx['journal']
    L += ['## 模擬帳戶', '']
    if j.empty:
        L += ['尚無紀錄，從今天起開始追蹤。', '']
    else:
        holding = j[j['status'] == '持有中']
        closed = j[j['status'] == '已出場']
        L += [f'持有 {len(holding)} 筆、已出場 {len(closed)} 筆' +
              (f'，已出場勝率 {(closed["r_now"] > 0).mean():.0%}、平均 {closed["r_now"].mean():+.2f}R' if len(closed) else ''), '']
        if len(holding):
            L += ['| 股票 | 進場 | 進場價 | 現價 | 停損 | 目標 | 損益 |', '|---|---|---|---|---|---|---|']
            L += [f'| {r.code} {r.name} | {r.entry_date} | {r.entry:,.2f} | {r.price:,.2f} | {r.stop:,.2f} | {r.target:,.2f} | {r.r_now:+.2f}R |'
                  for r in holding.itertuples()]
            L.append('')
        if len(closed):
            L += ['| 股票 | 進場 → 出場 | 出場原因 | 結果 |', '|---|---|---|---|']
            L += [f'| {r.code} {r.name} | {r.entry_date} → {r.exit_date} | {r.exit_reason} | {r.r_now:+.2f}R |'
                  for r in closed.tail(15).itertuples()]
            L.append('')

    wf, hold = ctx['walk_forward'], ctx['holdout']
    L += ['## 驗收成績（只看沒訓練過的資料）', '',
          f'逐年驗收 {meta["test_start"]} ~ {meta["holdout_start"]}，判官 AUC {pct(wf["auc"], ".3f")}', '',
          '| | 交易數 | 勝率 | 平均 R | 平均報酬 |', '|---|---|---|---|---|',
          f'| 所有型態訊號 | {wf["all"]["trades"]} | {wf["all"]["win_rate"]:.0%} | {wf["all"]["avg_r"]:+.2f} | {wf["all"]["avg_return"]:+.2%} |',
          f'| 判官通過 | {wf["accepted"]["trades"]} | {wf["accepted"]["win_rate"]:.0%} | {wf["accepted"]["avg_r"]:+.2f} | {wf["accepted"]["avg_return"]:+.2%} |',
          f'| **期末考**（最近 6 個月，全程未碰）判官通過 | {hold["accepted"].get("trades", 0)} | {pct(hold["accepted"].get("win_rate"), ".0%")} | {pct(hold["accepted"].get("avg_r"), "+.2f")} | {pct(hold["accepted"].get("avg_return"), "+.2%")} |',
          '']
    pf = ctx['portfolio']
    L += [f'模擬資金（最多 {sz["max_positions"]} 檔、單筆風險 {sz["risk"]:.0%}）與同期間比較：', '',
          '| | 期間 | 年化報酬 | 最大回檔 | 50 檔平均持有 |', '|---|---|---|---|---|']
    for label, key in (('逐年驗收', 'dev'), ('**期末考**', 'holdout')):
        s = pf.get(key) or {}
        b = s.get('benchmark') or {}
        L.append(f'| {label} | {s.get("start", "—")} ~ {s.get("end", "—")} | {pct(s.get("cagr"))} | '
                 f'{pct(s.get("max_drawdown"))} | {pct(b.get("cagr"))}（回檔 {pct(b.get("max_drawdown"))}） |')
    if pf.get('configs_tested'):
        L.append(f'\n規則由 {pf["configs_tested"]} 種組合中，只用逐年驗收期間挑選；期末考不參與挑選。')
    L.append('')

    L += ['## 哪些理由真的有效（驗收期間）', '',
          '| 理由 | 符合時勝率 | 不符合時勝率 | 符合時平均 R |', '|---|---|---|---|']
    L += [f'| {r["group"]}・{r["name"]} | {pct(r["win_yes"], ".0%")} | {pct(r["win_no"], ".0%")} | {pct(r["r_yes"], "+.2f")} |'
          for r in ctx['reasons']]
    L += ['', '> 回測含存活者偏差（使用目前 0050 成分股）。模擬帳戶以隔日開盤價進場，未計滑價。']
    (REPORT_DIR / 'latest.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


def main(skip_download=False):
    if not skip_download:
        update_all()

    strategy = load_strategy() or {}
    exit_rule = strategy.get('exit', DEFAULT_EXIT)
    sizing = strategy.get('sizing', DEFAULT_SIZING)
    use_judge = strategy.get('use_judge', True)

    _, latest, full = build_dataset()
    book = SetupBook(full)
    trades, candidates = book.trades(exit_rule)
    print(f'歷史交易 {len(trades):,} 筆（{trades["entry_date"].min().date()} ~ {trades["exit_date"].max().date()}）')

    tested, holdout, holdout_start = walk_forward(trades)
    wf = evaluate(tested)
    hold = {'all': summarize(holdout), 'accepted': summarize(holdout[holdout['prob'] >= holdout['threshold']])}
    print(f'逐年驗收：全部 {wf["all"]["win_rate"]:.1%} / {wf["all"]["avg_r"]:+.2f}R，'
          f'判官通過 {wf["accepted"]["win_rate"]:.1%} / {wf["accepted"]["avg_r"]:+.2f}R，AUC {wf["auc"]:.3f}')

    if strategy:
        portfolio = {'dev': strategy['dev'], 'holdout': strategy['holdout'],
                     'configs_tested': strategy['configs_tested']}
    else:
        prices = price_panel()
        pool = tested[tested['prob'] >= tested['threshold']] if use_judge else tested
        curve, _, _ = simulate(pool, sizing)
        portfolio = {'dev': {**curve_stats(curve), 'start': str(curve.index[0].date()), 'end': str(curve.index[-1].date()),
                             'benchmark': benchmark(prices, curve.index[0], curve.index[-1])}}
    print(f'模擬資金：{portfolio}')

    closed = trades[~trades['open']]
    judge = new_judge().fit(closed[JUDGE_FEATURES], closed['win'])
    threshold = closed['win'].mean() if use_judge else 0.0
    today = latest.index.max()
    plans, watch = [], []
    if not candidates.empty:
        candidates['prob'] = judge.predict_proba(candidates[JUDGE_FEATURES])[:, 1]
        for _, c in candidates.sort_values('prob', ascending=False).iterrows():
            pros, cons = explain(judge, c)
            item = {**c.to_dict(), 'setup_name': SETUPS[c.setup], 'pros': pros, 'cons': cons,
                    'checklist': checklist(c), 'similar': similar_stats(tested, c.setup, c.prob),
                    'weight': position_weight(c.risk_pct, sizing)}
            (plans if c.prob >= threshold else watch).append(item)

    accepted_today = pd.DataFrame(plans)
    journal = update_journal(accepted_today, today) if plans else (
        pd.read_csv(JOURNAL, parse_dates=['signal_date'], dtype={'code': str}) if JOURNAL.exists() else pd.DataFrame())
    status = journal_status(journal, book, exit_rule) if not journal.empty else pd.DataFrame()

    mkt = load(MARKET)['Close']
    row = latest.iloc[0]
    ctx = {
        'date': str(today.date()), 'run_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'market': {'close': mkt.iloc[-1], 'trend_200': row.mkt_trend_200, 'drawdown': row.mkt_drawdown_250,
                   'breadth': row.breadth_200},
        'plans': plans, 'watch': watch, 'journal': status, 'walk_forward': wf, 'holdout': hold,
        'portfolio': portfolio, 'reasons': reason_stats(tested), 'exit_rule': exit_rule, 'sizing': sizing,
        'meta': {'test_start': str(tested['entry_date'].min().date()), 'holdout_start': str(holdout_start.date())},
    }
    REPORT_DIR.mkdir(exist_ok=True)
    write_report(ctx)
    summary = {k: v for k, v in ctx.items() if k not in ('journal', 'plans', 'watch')}
    summary['plans'] = [{k: v for k, v in p.items() if k in ('code', 'name', 'setup', 'prob', 'entry', 'stop', 'target')}
                        for p in plans]
    (REPORT_DIR / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                             encoding='utf-8')
    print(f'今日計畫 {len(plans)} 筆、觀察 {len(watch)} 筆')


if __name__ == '__main__':
    main(skip_download='--skip-download' in sys.argv)
