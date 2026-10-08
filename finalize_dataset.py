"""Verify saved market data and add public leader data to the local database."""
import gzip
import json
import sqlite3
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).parent / 'data/binance/2026-10-02_2026大黑马'


def read(name):
    return json.loads((ROOT / name).read_text())


leader = read('leader_orders.json')
positions = read('leader_position_history_first.json')['data']['list']
transfers = [r for p in read('leader_transfers_pages.json') for r in p['response']['data']['list']]
assert len(leader) == 454
assert len({json.dumps(x, sort_keys=True) for x in leader}) == 454
assert len(transfers) == 19
market = read('market/manifest.json')
assert len(market['results']) == 50
assert all(r['status'] == 'complete' for r in market['results'])
assert all(not r['internal_gap_minutes'] and not r['observation_minutes_missing'] for r in market['results'])
trades = read('trades.json')
by_symbol = defaultdict(list)
for t in trades:
    by_symbol[t['symbol']].append(t)
lead_by_symbol = defaultdict(list)
for t in leader:
    lead_by_symbol[t['symbol']].append(t)
price_mismatch, leader_missing = [], []
bar_count = 0
for result in market['results']:
    with gzip.open(ROOT / 'market' / result['file'], 'rt') as f:
        bars = json.load(f)
    assert len(bars) == result['rows']
    assert len({r[0] for r in bars}) == len(bars)
    assert all(len(r) == 12 and r[6] <= result['requested_end_ms'] for r in bars)
    bar_count += len(bars)
    index = {r[0]: r for r in bars}
    for t in by_symbol[result['symbol']]:
        bar = index[t['time'] // 60000 * 60000]
        if not Decimal(bar[3]) <= Decimal(t['price']) <= Decimal(bar[2]):
            price_mismatch.append(dict(record_id=t['record_id'], symbol=t['symbol'], time_ms=t['time'], price=t['price'], low=bar[3], high=bar[2]))
    for t in lead_by_symbol[result['symbol']]:
        if t['orderUpdateTime'] // 60000 * 60000 not in index:
            leader_missing.append(dict(symbol=t['symbol'], orderUpdateTime=t['orderUpdateTime']))

quality = dict(market_symbols=50, kline_rows=bar_count, market_funding_rows=sum(r['funding_rows'] for r in market['results']), duplicate_kline_open_times=0, internal_missing_minutes=0, follower_trade_minutes_missing=0, follower_prices_outside_1m_range=price_mismatch, leader_order_minutes_missing=leader_missing, leader_orders=454, leader_closed_positions_saved=len(positions), leader_closed_positions_reported_total=read('leader_position_history_first.json')['data']['total'], leader_transfers=19)
(ROOT / 'market_quality.json').write_text(json.dumps(quality, ensure_ascii=False, indent=2))
con = sqlite3.connect(ROOT / 'research.sqlite')
for name, rows in [('leader_orders', leader), ('leader_closed_positions', positions), ('leader_transfers', transfers)]:
    con.execute(f'CREATE TABLE {name} (row_number INTEGER PRIMARY KEY, symbol TEXT, time_ms INTEGER, payload_json TEXT NOT NULL)')
    for i, row in enumerate(rows, 1):
        timestamp = row.get('orderUpdateTime', row.get('opened', row.get('time')))
        con.execute(f'INSERT INTO {name} VALUES (?,?,?,?)', (i, row.get('symbol'), timestamp, json.dumps(row, ensure_ascii=False)))
    con.execute(f'CREATE INDEX {name}_symbol_time ON {name}(symbol,time_ms)')
con.execute('CREATE TABLE market_files (symbol TEXT PRIMARY KEY, file TEXT, rows INTEGER, first_open_ms INTEGER, last_open_ms INTEGER, funding_rows INTEGER)')
for r in market['results']:
    con.execute('INSERT INTO market_files VALUES (?,?,?,?,?,?)', (r['symbol'], 'market/' + r['file'], r['rows'], r['first_open_ms'], r['last_open_ms'], r['funding_rows']))
con.commit()
assert con.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
con.close()
print(json.dumps(quality, ensure_ascii=False, indent=2))
