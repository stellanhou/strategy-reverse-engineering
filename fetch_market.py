"""Download public USD-M market context without account credentials."""
import gzip
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent / 'data/binance/2026-10-02_2026大黑马'
OUT = ROOT / 'market'
OUT.mkdir(exist_ok=True)
API = 'https://fapi.binance.com'
lock = threading.Lock()
next_call = 0.0
stop = threading.Event()
sources = []


def get(endpoint, params):
    global next_call
    with lock:
        delay = next_call - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        next_call = time.monotonic() + 0.6
    if stop.is_set():
        raise RuntimeError('Stopped after API rate limit')
    url = API + endpoint + ('?' + urllib.parse.urlencode(params) if params else '')
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            body = r.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (418, 429):
            stop.set()
        raise
    data = json.loads(body)
    if isinstance(data, dict) and 'code' in data:
        raise RuntimeError(str(data))
    with lock:
        sources.append(dict(url=url, fetched_at=datetime.now(timezone.utc).isoformat()))
    return data


trades = json.loads((ROOT / 'trades.json').read_text())
failed = json.loads((ROOT / 'failed_orders.json').read_text())
times = defaultdict(list)
for t in trades:
    times[t['symbol']].append(t['time'])
for t in failed:
    times[t['symbol']].append(t['orderUpdateTime'])
start_all = min(t['time'] for t in trades)
end_all = max(t['time'] for t in trades)
for symbol in ('BTCUSDT', 'ETHUSDT'):
    times[symbol].extend([start_all, end_all])
now = get('/fapi/v1/time', {})['serverTime']
cutoff = now // 60000 * 60000 - 1
metadata = json.loads((OUT / 'exchange_info.json').read_text())
listed = {s['symbol']: s['onboardDate'] for s in metadata['symbols']}
manifest = dict(as_of_utc=datetime.now(timezone.utc).isoformat(), interval='1m', context='Each symbol: 7 days before first observed trade through 1 day after last trade, capped at collection time. BTC/ETH cover full study interval plus the same context. Includes failed-order symbols.', kline_columns=['open_time_ms','open','high','low','close','volume','close_time_ms','quote_volume','trade_count','taker_buy_base_volume','taker_buy_quote_volume','ignore'], results=[])


def fetch_symbol(symbol, observations):
    requested_start = (min(observations) - 7 * 86400000) // 60000 * 60000
    start = max(requested_start, (listed[symbol] + 59999) // 60000 * 60000)
    end = min(cutoff, (max(observations) + 86400000) // 60000 * 60000 - 1)
    filename = OUT / (symbol + '_1m.json.gz')
    assert not filename.exists(), f'Existing output: {filename.name}'
    rows = []
    cur = start
    while cur <= end:
        batch = get('/fapi/v1/klines', dict(symbol=symbol, interval='1m', startTime=cur, endTime=end, limit=1000))
        if not batch:
            break
        assert all(cur <= x[0] <= end for x in batch)
        assert all(batch[i][0] < batch[i+1][0] for i in range(len(batch)-1))
        rows.extend(x for x in batch if x[6] <= end)
        cur = batch[-1][0] + 60000
    with gzip.open(filename, 'wt', encoding='utf-8') as f:
        json.dump(rows, f, separators=(',', ':'))
    funding = []
    cur = start
    while cur <= end:
        batch = get('/fapi/v1/fundingRate', dict(symbol=symbol, startTime=cur, endTime=end, limit=1000))
        if not batch:
            break
        funding.extend(batch)
        cur = batch[-1]['fundingTime'] + 1
        if len(batch) < 1000:
            break
    (OUT / (symbol + '_funding_rates.json')).write_text(json.dumps(funding, ensure_ascii=False, separators=(',', ':')))
    available_minutes = {r[0] for r in rows}
    return dict(symbol=symbol, status='complete', requested_start_ms=requested_start, actual_requested_start_ms=start, requested_end_ms=end, rows=len(rows), first_open_ms=rows[0][0] if rows else None, last_open_ms=rows[-1][0] if rows else None, internal_gap_minutes=sum((rows[i+1][0]-rows[i][0])//60000-1 for i in range(len(rows)-1)), observation_minutes_missing=sorted({t//60000*60000 for t in observations if t//60000*60000 not in available_minutes}), funding_rows=len(funding), file=filename.name)


with ThreadPoolExecutor(max_workers=2) as pool:
    pending = {pool.submit(fetch_symbol, s, ts): s for s, ts in sorted(times.items())}
    for future in as_completed(pending):
        symbol = pending[future]
        try:
            result = future.result()
        except Exception as exc:
            result = dict(symbol=symbol, status='failed', error=str(exc))
        manifest['results'].append(result)
        (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False), flush=True)
(OUT / 'sources.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2))
print('FINISHED', len(manifest['results']), flush=True)
