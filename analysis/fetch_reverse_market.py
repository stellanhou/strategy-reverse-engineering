#!/usr/bin/env python3
"""Fetch missing 1m Klines for history-anchored leader symbols.

Only supplemental rows are written under analysis/results/reverse_market/.
The existing raw market directory is read-only input.
"""

from __future__ import annotations

import concurrent.futures
import gzip
import json
import os
import pathlib
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from collections import defaultdict


ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/binance/2026-10-02_2026大黑马"
BASE_MARKET = DATA_ROOT / "market"
EVENTS_PATH = ROOT / "analysis/results/leader_events.json"
OUT_DIR = ROOT / "analysis/results/reverse_market"
MANIFEST_PATH = OUT_DIR / "manifest.json"

INTERVAL_MS = 60_000
PANEL_START_MS = 1_788_393_600_000  # 2026-09-03 00:00:00 UTC
FETCH_START_MS = PANEL_START_MS - 60 * INTERVAL_MS
CUTOFF_MS = 1_790_940_420_000
END_OPEN_MS = CUTOFF_MS - INTERVAL_MS
REQUEST_INTERVAL_SECONDS = 0.35
WORKERS = 4
PAGE_LIMIT = 1000
API_URL = "https://fapi.binance.com/fapi/v1/klines"

_limiter_lock = threading.Lock()
_next_request_at = 0.0


def paced_request(url: str) -> bytes:
    global _next_request_at
    with _limiter_lock:
        now = time.monotonic()
        wait_for = _next_request_at - now
        if wait_for > 0:
            time.sleep(wait_for)
            now = time.monotonic()
        _next_request_at = max(now, _next_request_at) + REQUEST_INTERVAL_SECONDS
    request = urllib.request.Request(url, headers={"User-Agent": "reverse-research-panel/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def load_base_rows(symbol: str) -> dict[int, list]:
    path = BASE_MARKET / f"{symbol}_1m.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = json.load(stream)
    by_open = {}
    previous = None
    for row in rows:
        if len(row) != 12:
            raise ValueError(f"{symbol}: base Kline row has {len(row)} columns")
        open_ms = int(row[0])
        if previous is not None and open_ms <= previous:
            raise ValueError(f"{symbol}: base Klines are not strictly ordered")
        if open_ms in by_open:
            raise ValueError(f"{symbol}: duplicate base Kline at {open_ms}")
        by_open[open_ms] = row
        previous = open_ms
    if not by_open:
        raise ValueError(f"{symbol}: base Kline file is empty")
    return by_open


def missing_segments(existing_opens: list[int]) -> list[tuple[int, int]]:
    target = sorted(t for t in existing_opens if FETCH_START_MS <= t <= END_OPEN_MS)
    if not target:
        return [(FETCH_START_MS, END_OPEN_MS)]
    segments = []
    if target[0] > FETCH_START_MS:
        segments.append((FETCH_START_MS, target[0] - INTERVAL_MS))
    for left, right in zip(target, target[1:]):
        if right - left > INTERVAL_MS:
            segments.append((left + INTERVAL_MS, right - INTERVAL_MS))
    if target[-1] < END_OPEN_MS:
        segments.append((target[-1] + INTERVAL_MS, END_OPEN_MS))
    return [(start, end) for start, end in segments if start <= end]


def fetch_segment(symbol: str, start_ms: int, end_open_ms: int) -> tuple[list[list], int]:
    rows = []
    pages = 0
    cursor = start_ms
    while cursor <= end_open_ms:
        query = urllib.parse.urlencode(
            {
                "symbol": symbol,
                "interval": "1m",
                "startTime": cursor,
                "endTime": end_open_ms + INTERVAL_MS - 1,
                "limit": PAGE_LIMIT,
            }
        )
        payload = paced_request(f"{API_URL}?{query}")
        decoded = json.loads(payload)
        pages += 1
        if not isinstance(decoded, list):
            raise ValueError(f"{symbol}: unexpected Klines response {decoded!r}")
        if not decoded:
            break
        previous = None
        for row in decoded:
            if not isinstance(row, list) or len(row) != 12:
                raise ValueError(f"{symbol}: malformed Kline row {row!r}")
            open_ms = int(row[0])
            if open_ms < cursor or open_ms > end_open_ms:
                raise ValueError(f"{symbol}: API returned out-of-range open time {open_ms}")
            if previous is not None and open_ms <= previous:
                raise ValueError(f"{symbol}: API Klines are not strictly ordered")
            previous = open_ms
            rows.append(row)
        last_open = int(decoded[-1][0])
        if len(decoded) < PAGE_LIMIT:
            break
        next_cursor = last_open + INTERVAL_MS
        if next_cursor <= cursor:
            raise ValueError(f"{symbol}: Kline pagination did not advance")
        cursor = next_cursor
    return rows, pages


def fetch_symbol(symbol: str, base_by_open: dict[int, list]) -> dict:
    segments = missing_segments(list(base_by_open))
    supplement_by_open = {}
    page_count = 0
    for start_ms, end_ms in segments:
        rows, pages = fetch_segment(symbol, start_ms, end_ms)
        page_count += pages
        for row in rows:
            open_ms = int(row[0])
            previous = supplement_by_open.get(open_ms)
            if previous is not None and previous != row:
                raise ValueError(f"{symbol}: conflicting duplicate supplemental Kline at {open_ms}")
            if open_ms in base_by_open:
                if base_by_open[open_ms] != row:
                    raise ValueError(f"{symbol}: supplemental/base values differ at {open_ms}")
                continue
            supplement_by_open[open_ms] = row
    ordered = [supplement_by_open[t] for t in sorted(supplement_by_open)]
    return {
        "symbol": symbol,
        "segments": [{"start_open_ms": a, "end_open_ms": b} for a, b in segments],
        "pages": page_count,
        "supplement_rows": ordered,
    }


def write_json_atomic(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def write_gzip_rows_atomic(path: pathlib.Path, rows: list[list]) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                compressed.write(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists():
        raise FileExistsError(f"refusing to overwrite existing supplement manifest: {MANIFEST_PATH}")

    events = json.loads(EVENTS_PATH.read_text(encoding="utf-8"))
    symbols = sorted({e["symbol"] for e in events if e["classification"] == "history_anchored"})
    if len(symbols) != 33:
        raise ValueError(f"expected 33 history_anchored symbols, got {len(symbols)}")

    base_rows = {symbol: load_base_rows(symbol) for symbol in symbols}
    manifest = {
        "status": "fetching",
        "source": API_URL,
        "interval": "1m",
        "workers": WORKERS,
        "global_min_request_interval_seconds": REQUEST_INTERVAL_SECONDS,
        "limit": PAGE_LIMIT,
        "panel_start_utc": "2026-09-03T00:00:00Z",
        "fetch_start_utc": "2026-09-02T23:00:00Z",
        "cutoff_ms_exclusive": CUTOFF_MS,
        "end_open_ms_inclusive": END_OPEN_MS,
        "base_market": str(BASE_MARKET.relative_to(ROOT)),
        "symbols": symbols,
        "results": {symbol: {"status": "pending"} for symbol in symbols},
    }
    write_json_atomic(MANIFEST_PATH, manifest)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {
                pool.submit(fetch_symbol, symbol, base_rows[symbol]): symbol
                for symbol in symbols
            }
            for future in concurrent.futures.as_completed(futures):
                symbol = futures[future]
                result = future.result()
                rows = result.pop("supplement_rows")
                filename = f"{symbol}_1m_supplement.json.gz"
                write_gzip_rows_atomic(OUT_DIR / filename, rows)
                opens = [int(row[0]) for row in rows]
                result.update(
                    {
                        "status": "complete",
                        "rows": len(rows),
                        "first_open_ms": min(opens) if opens else None,
                        "last_open_ms": max(opens) if opens else None,
                        "file": filename,
                    }
                )
                manifest["results"][symbol] = result
                manifest["completed_symbols"] = sum(
                    item.get("status") == "complete" for item in manifest["results"].values()
                )
                write_json_atomic(MANIFEST_PATH, manifest)
                print(
                    f"FETCH_DONE {symbol} added={len(rows)} pages={result['pages']} "
                    f"supplement_open=[{result['first_open_ms']},{result['last_open_ms']}]",
                    flush=True,
                )
    except BaseException as exc:
        for future in futures:
            future.cancel()
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_json_atomic(MANIFEST_PATH, manifest)
        raise

    manifest["status"] = "complete"
    manifest["completed_symbols"] = len(symbols)
    manifest["supplement_row_count"] = sum(item["rows"] for item in manifest["results"].values())
    write_json_atomic(MANIFEST_PATH, manifest)
    print(
        f"FETCH_COMPLETE symbols={len(symbols)} rows={manifest['supplement_row_count']} "
        f"cutoff={CUTOFF_MS}",
        flush=True,
    )


if __name__ == "__main__":
    main()
