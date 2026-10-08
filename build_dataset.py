"""Build a local research dataset from saved Binance exports and page responses."""
import csv
import io
import json
import sqlite3
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).parent / 'data/binance/2026-10-02_2026大黑马'
D = Decimal


def read(name):
    return json.loads((ROOT / name).read_text(), parse_float=D)


def save(name, obj):
    (ROOT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def page_rows(name):
    pages = read(name)
    rows = []
    for p in pages:
        data = p['response']['data']
        rows.extend(data if isinstance(data, list) else data['list'])
    return rows


def millis(value):
    return int(datetime.strptime(value, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp() * 1000)


def utc(ms):
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).isoformat()


def trade_key(t):
    return (t['time'], t['symbol'], t['side'], D(str(t['price'])), D(str(t['qty'])), D(str(t['fee'])), D(str(t['realizedProfit'])))


z = zipfile.ZipFile(ROOT / 'official_trades.zip')
raw_csv = z.read(z.namelist()[0])
(ROOT / 'official_trades.csv').write_bytes(raw_csv)
csv_rows = list(csv.DictReader(io.StringIO(raw_csv.decode('utf-8-sig'))))
trades = []
for row in csv_rows:
    qty, base = row['Quantity'].split()
    fee, fee_asset = row['Fee'].split()
    pnl, pnl_asset = row['Realized Profit'].split()
    trades.append(dict(time=millis(row['Time(UTC)']), symbol=row['Symbol'], side={'Buy/Long': 'BUY', 'Sell/Short': 'SELL'}[row['Side']], price=D(row['Price']), qty=D(qty), baseAsset=base, fee=D(fee), feeAsset=fee_asset, realizedProfit=D(pnl), realizedProfitAsset=pnl_asset, source='official_csv'))

web_trades = page_rows('trades_raw.json')
export_max = max(t['time'] for t in trades)
overlap = [t for t in web_trades if t['time'] <= export_max]
used = set()
time_deltas = Counter()
for web in overlap:
    candidates = [i for i, t in enumerate(trades) if i not in used and trade_key(t)[1:] == trade_key(web)[1:] and abs(t['time'] - web['time']) <= 1000]
    assert len(candidates) == 1, f'Ambiguous/unmatched overlap: {web}'
    i = candidates[0]
    used.add(i)
    trades[i]['page_time_ms'] = web['time']
    time_deltas[web['time'] - trades[i]['time']] += 1
recent = [dict(t, source='page_response') for t in web_trades if t['time'] > export_max]
trades.extend(recent)
trades.sort(key=lambda t: t['time'])
expected_trades = read('trades_raw.json')[0]['response']['data']['total']
assert len(trades) == expected_trades, (len(trades), expected_trades)
for i, t in enumerate(trades, 1):
    t['record_id'] = f'fill-{i:06d}'
    t['time_utc'] = utc(t['time'])
    t['notional_usdt'] = D(str(t['price'])) * D(str(t['qty']))

orders = page_rows('orders_raw.json')
positions = page_rows('position_history_raw.json')
funding = page_rows('funding_raw.json')
sharing = page_rows('profit_sharing_raw.json')
transfers = read('transfers_raw.json')['data']['list']
failed = read('failed_orders_raw.json')['data']['list']
current = read('positions_raw.json')['data']
assert len(orders) == read('orders_raw.json')[0]['response']['total']
assert len({o['id'] for o in orders}) == len(orders)
for name, rows, key in [('position_history_raw.json', positions, 'positionId'), ('funding_raw.json', funding, 'id')]:
    assert len(rows) == read(name)[0]['response']['data']['total']
    assert len({r[key] for r in rows}) == len(rows)
assert len(sharing) == read('profit_sharing_raw.json')[0]['response']['data']['total']
for order in orders:
    # The string id is authoritative and avoids JS integer precision loss.
    order['orderId'] = order['id'].split('_')[0]
    order['time_utc'] = utc(order['insertTime'])

net = defaultdict(D)
gross = defaultdict(D)
order_gross = defaultdict(D)
for t in trades:
    net[t['symbol']] += D(str(t['qty'])) * (1 if t['side'] == 'BUY' else -1)
    gross[(t['symbol'], t['side'])] += D(str(t['qty']))
for o in orders:
    order_gross[(o['symbol'], o['side'])] += D(o['executedQty'])
current_net = {p['symbol']: D(p['positionAmount']) for p in current}
net_diff = {s: net[s] - current_net.get(s, D(0)) for s in set(net) | set(current_net) if net[s] != current_net.get(s, D(0))}
qty_diff = {str(k): gross[k] - order_gross[k] for k in set(gross) | set(order_gross) if gross[k] != order_gross[k]}

counts = dict(trades=len(trades), orders=len(orders), closed_positions=len(positions), funding=len(funding), profit_sharing=len(sharing), transfers=len(transfers), failed_orders=len(failed), open_positions=len(current))
totals = dict(realized_trade_pnl=sum(D(str(t['realizedProfit'])) for t in trades), trading_fees=sum(D(str(t['fee'])) for t in trades), funding=sum(D(str(r['balanceDeltaStr'])) for r in funding), profit_sharing=sum(D(str(r['sharedAmount'])) for r in sharing))
totals['realized_net_before_sharing'] = totals['realized_trade_pnl'] + totals['trading_fees'] + totals['funding']
totals['realized_net_after_sharing'] = totals['realized_net_before_sharing'] - totals['profit_sharing']
audit = dict(counts=counts, trade_start_utc=utc(trades[0]['time']), trade_end_utc=utc(trades[-1]['time']), symbols=sorted(net), official_csv_rows=len(csv_rows), recent_page_rows_added=len(recent), overlapping_page_rows_checked=len(overlap), page_minus_csv_time_ms=dict(time_deltas), page_trade_rows_saved=len(web_trades), page_trade_total=expected_trades, duplicate_trade_signatures=sum(v-1 for v in Counter(trade_key(t) for t in trades).values() if v>1), net_position_difference=net_diff, order_fill_quantity_difference=qty_diff, totals_usdt=totals, notes=['成交没有交易所成交ID或订单ID；record_id仅为本地行号。相同成交签名不自动去重。', '官方CSV与页面按合约、买卖方向、价格、数量、费用、盈亏完全相同、时间相差不超过1秒逐笔唯一匹配；已观察到页面时间比CSV晚1秒。保留两种原始时间。', 'BUY/SELL是成交方向；开平仓应结合委托reduceOnly和净仓位，不可直接把SELL当作开空。', '原始trades_raw仅保存前30页；标准化trades.json覆盖官方导出和当日补充后的全部676条。'])
audit['order_ids_restored_from_string_id'] = sum(abs(int(o['orderId'])) > 2**53 - 1 for o in orders)
audit['notes'].append('orders_raw为JS解析后的响应快照；大整数orderId可能已舍入。标准化orders.json中的orderId全部由同一响应的无损字符串id恢复。')
save('audit.json', audit)
datasets = dict(trades=trades, orders=orders, closed_positions=positions, funding=funding, profit_sharing=sharing, transfers=transfers, failed_orders=failed, open_positions=current)
for name, rows in datasets.items():
    save(name + '.json', rows)

db = ROOT / 'research.sqlite'
assert not db.exists(), 'Do not overwrite a prior research database'
con = sqlite3.connect(db)
for name, rows in datasets.items():
    con.execute(f'CREATE TABLE {name} (row_number INTEGER PRIMARY KEY, symbol TEXT, time_ms INTEGER, payload_json TEXT NOT NULL)')
    for i, row in enumerate(rows, 1):
        time = row.get('time', row.get('insertTime', row.get('opened', row.get('orderUpdateTime', row.get('updateTime')))))
        con.execute(f'INSERT INTO {name} VALUES (?, ?, ?, ?)', (i, row.get('symbol'), time, json.dumps(row, ensure_ascii=False, default=str)))
    con.execute(f'CREATE INDEX {name}_symbol_time ON {name}(symbol,time_ms)')
con.commit()
con.close()
print(json.dumps(audit, ensure_ascii=False, indent=2, default=str))
