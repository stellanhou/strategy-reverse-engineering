"""Independently check reconstructed cycles and build the first-stage report."""
import json
import statistics
from collections import Counter, defaultdict
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/binance/2026-10-02_2026大黑马'
OUT = ROOT / 'analysis/results'
D = Decimal


def read(path):
    return json.loads(path.read_text(), parse_float=D)


def table(headers, rows):
    return '<table><thead><tr>' + ''.join(f'<th>{escape(str(x))}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{escape(str(x))}</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table>'


def cycle_table(cycles, pnl_key):
    def date(ms):
        return datetime.fromtimestamp(ms / 1000, timezone(timedelta(hours=8))).strftime('%m-%d %H:%M:%S') if ms is not None else '未平仓'
    return table(['合约', '方向', '开始（北京）', '结束（北京）', '加仓', '减仓', '盈亏 USDT'], [
        (c['symbol'], {'long': '多', 'short': '空'}[c['direction'].lower()], date(c['start_ms']), date(c['end_ms']), c['add_count'], c['reduce_count'], f"{D(c[pnl_key]):.8f}")
        for c in sorted(cycles, key=lambda x: x['start_ms'], reverse=True)
    ]).replace('<table>', '<table style="min-width:860px;white-space:nowrap">', 1)


def main():
    follower = read(OUT / 'follower_cycles.json')
    leader = read(OUT / 'leader_cycles.json')
    trades = read(DATA / 'trades.json')
    audit = read(DATA / 'audit.json')
    closed = read(DATA / 'closed_positions.json')
    histories = read(DATA / 'leader_position_history_first.json')['data']['list']
    orders = read(DATA / 'leader_orders.json')

    # Recompute source positions independently of either reconstruction script.
    positions = defaultdict(D)
    closed_ends = []
    for t in sorted(trades, key=lambda x: x['time']):
        before = positions[t['symbol']]
        positions[t['symbol']] += D(t['qty']) * (1 if t['side'] == 'BUY' else -1)
        assert before * positions[t['symbol']] >= 0, 'Unexpected reversal in this dataset'
        if before and positions[t['symbol']] == 0:
            closed_ends.append((t['symbol'], t['time']))
    snapshot = {x['symbol']: D(x['positionAmount']) for x in read(DATA / 'open_positions.json')}
    assert {s: q for s, q in positions.items() if q} == snapshot
    assert len(closed_ends) == len(closed) == 224
    assert all(sum(s == c['symbol'] and abs(ms - c['closed']) <= 2000 for s, ms in closed_ends) == 1 for c in closed)
    assert len(follower) == 227
    assert sum(D(c['realized_pnl']) for c in follower) == D(audit['totals_usdt']['realized_trade_pnl'])
    assert sum(D(c['fees']) for c in follower) == D(audit['totals_usdt']['trading_fees'])
    assert sum(D(c['funding']) for c in follower) == D(audit['totals_usdt']['funding'])
    assert all(D(c['net_pnl_before_sharing']) == D(c['closed_position_closing_pnl']) for c in follower if c['status'] == 'closed')

    # Independently anchor leader operations to official completed positions.
    selected, used = [], set()
    actions, add_state, types = Counter(), Counter(), Counter()
    add_cycles = reduce_cycles = 0
    for h in histories:
        ids = [i for i, o in enumerate(orders) if o['symbol'] == h['symbol'] and h['opened'] - 2000 <= o['orderUpdateTime'] <= h['closed'] + 2000]
        if not ids:
            continue
        assert not used.intersection(ids)
        used.update(ids)
        rows = sorted((orders[i] for i in ids), key=lambda x: x['orderUpdateTime'])
        assert abs(rows[0]['orderUpdateTime'] - h['opened']) <= 2000
        q = avg = D(0)
        nadd = nreduce = 0
        for o in rows:
            delta = D(str(o['executedQty'])) * (1 if o['side'] == 'BUY' else -1)
            price = D(str(o['avgPrice']))
            after = q + delta
            if not q:
                action, avg = 'open', price
            elif q * delta > 0:
                action = 'add'
                nadd += 1
                edge = (price - avg) * (1 if q > 0 else -1)
                add_state['underwater' if edge < 0 else 'profitable' if edge > 0 else 'flat'] += 1
                avg = (abs(q) * avg + abs(delta) * price) / abs(after)
            else:
                assert abs(delta) <= abs(q)
                action = 'close' if not after else 'reduce'
                nreduce += bool(after)
            actions[action] += 1
            types[(action, o['type'])] += 1
            q = after
        assert q == 0
        selected.append(h)
        add_cycles += nadd > 0
        reduce_cycles += nreduce > 0
    complete_leader = [c for c in leader if c['status'] == 'complete']
    assert len(complete_leader) == len(selected) == 157
    assert sum(c['add_count'] for c in complete_leader) == actions['add']
    assert sum(c['reduce_count'] for c in complete_leader) == actions['reduce']
    assert len(used) == 451 and len(orders) - len(used) == 3
    directions = Counter(h['side'] for h in selected)
    holds = [(h['closed'] - h['opened']) / 3600000 for h in selected]
    official_pnl = [D(str(h['closingPnl'])) for h in selected]
    verification = {
        'follower_cycles': len(follower), 'follower_closed_cycles': len(closed_ends),
        'follower_open_positions_match': True, 'follower_pnl_fees_funding_match': True,
        'follower_each_closed_cycle_net_matches_official': True,
        'leader_complete_cycles': len(selected), 'leader_complete_events': len(used),
        'leader_remaining_events': len(orders) - len(used),
        'leader_actions': dict(actions), 'leader_add_state': dict(add_state),
        'leader_directions': dict(directions), 'leader_add_cycles': add_cycles,
        'leader_reduce_cycles': reduce_cycles, 'leader_median_hold_hours': statistics.median(holds),
        'leader_official_closing_pnl_sum': str(sum(official_pnl)),
        'leader_official_profitable_cycles': sum(p > 0 for p in official_pnl),
        'leader_official_losing_cycles': sum(p < 0 for p in official_pnl),
    }
    (OUT / 'independent_verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2))
    short_share = 100 * directions['Short'] / len(selected)
    losing_add_share = 100 * add_state['underwater'] / actions['add']
    profile = table(['问题', '已观察到的结果'], [
        ('方向', f"做空 {directions['Short']} 轮，做多 {directions['Long']} 轮；做空占 {short_share:.1f}%"),
        ('加仓', f'{add_cycles} / 157 轮发生加仓，共 {actions["add"]} 次'),
        ('加仓时的价格位置', f'{add_state["underwater"]} 次逆向、{add_state["profitable"]} 次顺向；逆向占 {losing_add_share:.1f}%'),
        ('分批退出', f'{reduce_cycles} / 157 轮有减仓，共 {actions["reduce"]} 次；最终清仓另计'),
        ('下单方式', f'157 次首次开仓、87 次加仓均为市价；最终平仓 {types[("close", "LIMIT")]} 次限价、{types[("close", "MARKET")]} 次市价'),
        ('持仓时长', f'中位数 {statistics.median(holds):.2f} 小时，最长 {max(holds):.2f} 小时'),
        ('官方仓位盈亏口径', f'{sum(p > 0 for p in official_pnl)} 轮盈利、{sum(p < 0 for p in official_pnl)} 轮亏损；合计 {sum(official_pnl):.8f} USDT'),
    ])
    accounting = table(['项目', 'USDT'], [(label, audit['totals_usdt'][key]) for label, key in [
        ('成交已实现盈亏', 'realized_trade_pnl'), ('交易手续费', 'trading_fees'),
        ('资金费用', 'funding'), ('分润支出（扣减项）', 'profit_sharing'),
        ('分润后已实现净额', 'realized_net_after_sharing'),
    ]])
    cycle_tables = '<section><h2>完整交易周期表</h2><p>按开始时间倒序排列，时间均为北京时间。展开后可横向滚动查看。</p><details><summary>你的 227 个周期（224 个已结束、3 个未平仓）</summary><p class="note">盈亏包含已实现成交盈亏、手续费和资金费，未扣账户分润；未平仓周期未计浮动盈亏。</p><div style="overflow-x:auto">' + cycle_table(follower, 'net_pnl_before_sharing') + '</div></details><details><summary>带单员 157 个完整周期</summary><p class="note">盈亏沿用官方仓位历史口径。其他缺少逐笔操作或尚未结束的记录保留在下方结果文件中。</p><div style="overflow-x:auto">' + cycle_table(complete_leader, 'realized_pnl') + '</div></details></section>'
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2026大黑马 · 交易重建与行为画像</title>
<style>body{{margin:0;background:#f4f6f8;color:#1d2939;font:16px/1.75 system-ui,-apple-system,sans-serif}}main{{max-width:1060px;margin:auto;padding:40px 24px}}h1{{font-size:32px;line-height:1.4}}h2{{font-size:23px;margin-top:34px}}p{{max-width:900px}}.eyebrow{{color:#516176;font-size:14px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.card,section{{background:white;border:1px solid #e0e5eb;border-radius:14px;padding:22px;margin:18px 0}}.card strong{{display:block;font-size:30px;color:#105c79}}table{{width:100%;border-collapse:collapse;font-size:15px}}td,th{{text-align:left;padding:13px 10px;border-bottom:1px solid #e5e9ee;vertical-align:top}}th{{color:#526079}}a{{color:#12678c}}.note{{color:#566477;font-size:14px}}.button{{display:inline-block;background:#125f7b;color:white;padding:10px 18px;border-radius:9px;text-decoration:none}}@media(max-width:650px){{.cards{{grid-template-columns:1fr}}main{{padding:20px 12px}}h1{{font-size:27px}}}}</style>
<main><div class="eyebrow">策略逆向建模 / 第一阶段 / 数据截至 2026-10-02（北京时间）</div>
<h1>2026大黑马：交易重建与行为画像</h1>
<p>已完成交易周期重建、订单级加减仓统计和典型交易图。当前最清楚的行为特征是：<b>以做空为主，存在逆向加仓，多数最终退出使用限价单。</b>入场触发条件还需要结合开仓前行情检验。</p>
<div class="cards"><div class="card"><strong>224 + 3</strong>你的已结束 / 未平仓周期</div><div class="card"><strong>157</strong>带单员有完整操作覆盖的已平仓周期</div><div class="card"><strong>{losing_add_share:.1f}%</strong>加仓发生于价格相对持仓均价不利的位置</div></div>
<p><a class="button" href="trade_charts.html">查看典型交易图与逐次操作</a></p>
<section><h2>带单员如何交易</h2>{profile}<p class="note">统计单位为带单员订单操作。逆向 / 顺向按本次成交价相对操作前加权持仓均价，并结合多空方向判定；未扣手续费、资金费，也不代表整个周期最终盈亏。官方仓位盈亏沿用 closingPnl 字段；其与订单 totalPnl 的差额详见带单员核对文件。样本为 30 天窗口内可完整重建的交易。</p></section>
<section><h2>你的实际执行与资金核对</h2><p>676 笔成交对应 623 张已成交委托。逐笔仓位重建得到 224 轮已结束交易，结束时间全部与官方仓位记录在 2 秒内对应；剩余仓位与采集时的 3 个持仓一致。订单级统计为开仓 227 次、加仓 114 次、减仓 58 次、最终平仓 224 次。周期合计的成交盈亏、手续费和资金费用均与原始审计相符。</p>{accounting}<p class="note">分润在账户层扣除，不强行分摊到单轮交易。上表为观察区间已实现项目，包含未平仓周期已经发生的费用等项目，未计入期末浮动盈亏，也不是收益率。成交与委托关联及同秒事件的处理见跟单核对文件。</p></section>
<section><h2>数据覆盖与结论边界</h2><p>你的执行数据覆盖 2026-08-20 至 2026-10-02；带单员操作数据覆盖 2026-09-03 至 2026-10-02。带单员 454 条操作中，451 条属于这 157 轮完整交易，其余 3 条单独保留：BTW、龙虾依据最近已平仓锚点推断为新开并标记右截断；CT 缺少此前仓位锚点，保留为无法确定开仓 / 加减仓的操作。公开仓位最近 200 条中的 43 条不在逐笔操作覆盖范围内，未混入这份行为统计。</p><p>交易图按明确规则选择盈利、亏损、频繁加仓、长短持仓及多头等样例。当前数据足以描述操作行为；要解释为什么此时开仓、为什么选择该币，还需比较未交易时刻及当时其他可选币种。</p></section>
<section><h2>下一阶段</h2><p>围绕“短期上涨后做空、逆向加仓、回落退出”建立候选解释，先检查开仓前 5 分钟、15 分钟和 1 小时的涨跌、成交量及相对 BTC 强弱，再与同币种未交易时刻比较。加仓间距和仓位增量应单独研究，观察是否存在稳定规则。所有这些目前都是待检验假设，本阶段未训练模型或进行策略回测。</p></section>
{cycle_tables}
<section><h2>结果文件</h2><p><a href="follower_cycles.json">跟单交易周期数据</a> · <a href="follower_events.json">跟单订单与资金费事件</a> · <a href="follower_summary.json">跟单统计与核对</a></p><p><a href="leader_cycles.json">带单交易周期数据</a> · <a href="leader_events.json">带单操作事件</a> · <a href="leader_summary.json">带单统计与核对</a></p><p><a href="chart_selection.json">图表选择依据</a> · <a href="independent_verification.json">独立复核结果</a></p></section>
</main></html>'''
    (OUT / 'report.html').write_text(page)
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
