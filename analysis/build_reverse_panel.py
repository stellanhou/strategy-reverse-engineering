#!/usr/bin/env python3
"""Build the causal minute panel used for leader-behavior research."""

from __future__ import annotations

import collections
import gzip
import json
import os
import pathlib
import tempfile
from decimal import Decimal

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/binance/2026-10-02_2026大黑马"
BASE_MARKET = DATA_ROOT / "market"
SUPPLEMENT_DIR = ROOT / "analysis/results/reverse_market"
SUPPLEMENT_MANIFEST = SUPPLEMENT_DIR / "manifest.json"
RESULTS = ROOT / "analysis/results"
EVENTS_PATH = RESULTS / "leader_events.json"
CYCLES_PATH = RESULTS / "leader_cycles.json"
PANEL_PATH = RESULTS / "reverse_panel.npz"
META_PATH = RESULTS / "reverse_panel_meta.json"
EVENT_FEATURES_PATH = RESULTS / "reverse_event_features.json"

MINUTE_MS = 60_000
PANEL_START_MS = 1_788_393_600_000  # 2026-09-03 00:00:00 UTC
CUTOFF_MS = 1_790_940_420_000  # exclusive; final included minute starts at cutoff - 60s
END_OPEN_MS = CUTOFF_MS - MINUTE_MS
BENCHMARK = "BTCUSDT"

FEATURE_NAMES = [
    "ret5",
    "ret15",
    "ret60",
    "qv5_vs_prev60_mean",
    "volatility60",
    "range15",
    "btc_relative15",
    "btc_relative60",
    "close",
    "avgcost",
    "position",
    "first_qty",
    "age_minutes",
    "minutes_last_action",
    "add_count",
    "mark_return",
    "position_multiple",
]
MARKET_FEATURE_NAMES = FEATURE_NAMES[:9]
POSITION_FEATURE_NAMES = FEATURE_NAMES[9:]
ACTION_CODES = {"open": 1, "add": 2, "reduce": 3, "close": 4}


def read_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
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


def load_kline_file(path: pathlib.Path) -> list[list]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError(f"Kline file is not a row list: {path}")
    return rows


def merge_symbol_rows(symbol: str, supplement_results: dict) -> tuple[dict[int, list], dict]:
    base_path = BASE_MARKET / f"{symbol}_1m.json.gz"
    base_rows = load_kline_file(base_path)
    by_open: dict[int, list] = {}
    for row in base_rows:
        if len(row) != 12:
            raise ValueError(f"{symbol}: base Kline row has {len(row)} fields")
        open_ms = int(row[0])
        if open_ms % MINUTE_MS:
            raise ValueError(f"{symbol}: non-minute-aligned Kline at {open_ms}")
        if open_ms in by_open:
            raise ValueError(f"{symbol}: duplicate base Kline at {open_ms}")
        by_open[open_ms] = row

    supplement = supplement_results.get(symbol)
    supplement_rows = []
    supplement_file = None
    if supplement is not None:
        if supplement.get("status") != "complete":
            raise ValueError(f"{symbol}: supplement result is not complete")
        supplement_file = supplement["file"]
        supplement_rows = load_kline_file(SUPPLEMENT_DIR / supplement_file)
    overlap_equal = 0
    for row in supplement_rows:
        if len(row) != 12:
            raise ValueError(f"{symbol}: supplemental Kline row has {len(row)} fields")
        open_ms = int(row[0])
        if open_ms % MINUTE_MS:
            raise ValueError(f"{symbol}: non-minute-aligned supplemental Kline at {open_ms}")
        previous = by_open.get(open_ms)
        if previous is None:
            by_open[open_ms] = row
        elif previous != row:
            raise ValueError(f"{symbol}: base/supplement values differ at {open_ms}")
        else:
            overlap_equal += 1

    ordered_opens = sorted(by_open)
    if any(b <= a for a, b in zip(ordered_opens, ordered_opens[1:])):
        raise ValueError(f"{symbol}: merged Klines are not unique and ordered")
    gaps = [
        {"start_open_ms": a + MINUTE_MS, "end_open_ms": b - MINUTE_MS, "missing_minutes": (b - a) // MINUTE_MS - 1}
        for a, b in zip(ordered_opens, ordered_opens[1:])
        if b - a > MINUTE_MS
    ]
    coverage = {
        "base_rows": len(base_rows),
        "supplement_rows": len(supplement_rows),
        "supplement_file": supplement_file,
        "equal_overlap_rows": overlap_equal,
        "merged_rows": len(by_open),
        "first_open_ms": ordered_opens[0] if ordered_opens else None,
        "last_open_ms": ordered_opens[-1] if ordered_opens else None,
        "valid_klines_in_panel": sum(PANEL_START_MS <= t <= END_OPEN_MS for t in ordered_opens),
        "first_panel_kline_ms": next((t for t in ordered_opens if PANEL_START_MS <= t <= END_OPEN_MS), None),
        "last_panel_kline_ms": next((t for t in reversed(ordered_opens) if PANEL_START_MS <= t <= END_OPEN_MS), None),
        "internal_gap_count": len(gaps),
        "internal_missing_minutes": sum(g["missing_minutes"] for g in gaps),
        "internal_gaps": gaps,
    }
    return by_open, coverage


def _return(close: np.ndarray, valid: np.ndarray, end_idx: int, lag: int) -> float:
    start_idx = end_idx - lag
    segment = valid[start_idx : end_idx + 1]
    if start_idx < 0 or segment.size != lag + 1 or not segment.all():
        return np.nan
    first = close[start_idx]
    last = close[end_idx]
    if first <= 0 or not np.isfinite(first) or not np.isfinite(last):
        return np.nan
    return last / first - 1.0


def market_features_at(symbol: str, t_ms: int, bars: dict[int, list], btc_bars: dict[int, list]) -> np.ndarray:
    """Features at minute start t; every price/volume bar must close before t."""
    values = np.full(9, np.nan, dtype=np.float64)
    prior_opens = [t_ms - k * MINUTE_MS for k in range(1, 62)]
    rows = [bars.get(open_ms) for open_ms in prior_opens]
    btc_rows = [btc_bars.get(open_ms) for open_ms in prior_opens]
    valid = np.array([row is not None for row in rows], dtype=bool)
    btc_valid = np.array([row is not None for row in btc_rows], dtype=bool)
    close = np.array([float(row[4]) if row is not None else np.nan for row in rows], dtype=np.float64)
    qv = np.array([float(row[7]) if row is not None else np.nan for row in rows], dtype=np.float64)
    high = np.array([float(row[2]) if row is not None else np.nan for row in rows], dtype=np.float64)
    low = np.array([float(row[3]) if row is not None else np.nan for row in rows], dtype=np.float64)
    btc_close = np.array([float(row[4]) if row is not None else np.nan for row in btc_rows], dtype=np.float64)

    # prior_opens is newest first: index 0 is t-1m, index 4 is t-5m, etc.
    values[0] = _return(close[::-1], valid[::-1], len(close) - 1, 5)
    values[1] = _return(close[::-1], valid[::-1], len(close) - 1, 15)
    values[2] = _return(close[::-1], valid[::-1], len(close) - 1, 60)
    if valid[:60].all():
        qv60 = qv[:60]
        mean60 = float(qv60.mean())
        if mean60 > 0:
            values[3] = float(qv[:5].mean()) / mean60
    if valid[:61].all() and np.all(close[:61] > 0):
        chronological = close[:61][::-1]
        returns = chronological[1:] / chronological[:-1] - 1.0
        values[4] = float(np.std(returns, ddof=0))
    if valid[:15].all():
        values[5] = float(np.max(high[:15]) / np.min(low[:15]) - 1.0)
    btc_ret15 = np.nan
    btc_ret60 = np.nan
    if btc_valid[0:16].all():
        btc_ret15 = btc_close[0] / btc_close[15] - 1.0
    if btc_valid[0:61].all():
        btc_ret60 = btc_close[0] / btc_close[60] - 1.0
    if np.isfinite(values[1]) and np.isfinite(btc_ret15):
        values[6] = values[1] - btc_ret15
    if np.isfinite(values[2]) and np.isfinite(btc_ret60):
        values[7] = values[2] - btc_ret60
    if valid[0]:
        values[8] = close[0]
    return values


def panel_market_features(symbol: str, times: np.ndarray, bars: dict[int, list], btc_bars: dict[int, list]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized one-minute features over the dense panel calendar."""
    count = len(times)
    warm_start = PANEL_START_MS - 61 * MINUTE_MS
    dense_count = count + 61
    close = np.full(dense_count, np.nan, dtype=np.float64)
    high = np.full(dense_count, np.nan, dtype=np.float64)
    low = np.full(dense_count, np.nan, dtype=np.float64)
    qv = np.full(dense_count, np.nan, dtype=np.float64)
    btc_close = np.full(dense_count, np.nan, dtype=np.float64)
    valid = np.zeros(dense_count, dtype=bool)
    btc_valid = np.zeros(dense_count, dtype=bool)
    current_kline = np.zeros(count, dtype=bool)

    for open_ms, row in bars.items():
        if warm_start <= open_ms <= END_OPEN_MS:
            i = (open_ms - warm_start) // MINUTE_MS
            close[i] = float(row[4])
            high[i] = float(row[2])
            low[i] = float(row[3])
            qv[i] = float(row[7])
            valid[i] = True
            if PANEL_START_MS <= open_ms <= END_OPEN_MS:
                current_kline[(open_ms - PANEL_START_MS) // MINUTE_MS] = True
    for open_ms, row in btc_bars.items():
        if warm_start <= open_ms <= END_OPEN_MS:
            i = (open_ms - warm_start) // MINUTE_MS
            btc_close[i] = float(row[4])
            btc_valid[i] = True

    x = np.full((count, 9), np.nan, dtype=np.float64)
    # For panel minute k, its current candle slot is k + 61; slot - 1 is last closed.
    for k in range(count):
        i = k + 61
        if valid[i - 1]:
            x[k, 8] = close[i - 1]
        x[k, 0] = _return(close, valid, i - 1, 5)
        x[k, 1] = _return(close, valid, i - 1, 15)
        x[k, 2] = _return(close, valid, i - 1, 60)
        if valid[i - 60 : i].all():
            mean60 = float(qv[i - 60 : i].mean())
            if mean60 > 0:
                x[k, 3] = float(qv[i - 5 : i].mean()) / mean60
        if valid[i - 61 : i].all() and np.all(close[i - 61 : i] > 0):
            returns = close[i - 60 : i] / close[i - 61 : i - 1] - 1.0
            x[k, 4] = float(np.std(returns, ddof=0))
        if valid[i - 15 : i].all():
            x[k, 5] = float(np.max(high[i - 15 : i]) / np.min(low[i - 15 : i]) - 1.0)
        if btc_valid[i - 16 : i].all():
            btc15 = btc_close[i - 1] / btc_close[i - 16] - 1.0
            if np.isfinite(x[k, 1]):
                x[k, 6] = x[k, 1] - btc15
        if btc_valid[i - 61 : i].all():
            btc60 = btc_close[i - 1] / btc_close[i - 61] - 1.0
            if np.isfinite(x[k, 2]):
                x[k, 7] = x[k, 2] - btc60
    prior_kline = np.zeros(count, dtype=bool)
    first_open = min((t for t in bars if t <= END_OPEN_MS), default=None)
    if first_open is not None:
        prior_kline = times > first_open
    return x, current_kline, prior_kline


def decimal_value(value) -> Decimal:
    if value is None:
        return Decimal(0)
    return Decimal(str(value))


def event_sort_key(event: dict):
    return int(event["time_ms"]), event.get("source_index", 0), event["event_id"]


def initial_unknown_until(cycles: list[dict], symbols: list[str]) -> dict[str, int]:
    unknown_until = {symbol: PANEL_START_MS for symbol in symbols}
    by_symbol = collections.defaultdict(list)
    for cycle in cycles:
        if cycle.get("status") == "history_only_outside_order_window":
            by_symbol[cycle["symbol"]].append(cycle)
    for symbol, records in by_symbol.items():
        for cycle in records:
            start = int(cycle["start_ms"])
            end = int(cycle["end_ms"])
            if start <= PANEL_START_MS < end:
                unknown_until[symbol] = max(unknown_until[symbol], end)
    return unknown_until


def apply_event(state: dict, event: dict) -> None:
    before = decimal_value(event.get("position_before"))
    after = decimal_value(event.get("position_after"))
    qty = decimal_value(event["qty"])
    price = decimal_value(event["price"])
    action = event["action"]
    if before != state["position"]:
        raise ValueError(
            f"{event['symbol']} {event['event_id']}: reconstructed position {state['position']} "
            f"does not match event position_before {before}"
        )
    if action == "open":
        if state["position"] != 0 or qty <= 0 or price <= 0:
            raise ValueError(f"{event['event_id']}: invalid open transition")
        direction = 1 if event["direction"].upper() == "LONG" else -1
        state.update(
            {
                "position": after,
                "avgcost": price,
                "first_qty": qty,
                "open_time_ms": int(event["time_ms"]),
                "last_action_time_ms": int(event["time_ms"]),
                "add_count": 0,
                "cycle_id": event["cycle_id"],
                "direction": direction,
                "known": True,
            }
        )
        if after != Decimal(direction) * qty:
            raise ValueError(f"{event['event_id']}: open amount disagrees with position_after")
    elif action == "add":
        if state["position"] == 0 or qty <= 0:
            raise ValueError(f"{event['event_id']}: add without active position")
        direction = state["direction"]
        if after != state["position"] + Decimal(direction) * qty:
            raise ValueError(f"{event['event_id']}: add amount disagrees with position_after")
        state["avgcost"] = (
            abs(state["position"]) * state["avgcost"] + qty * price
        ) / abs(after)
        state["position"] = after
        state["last_action_time_ms"] = int(event["time_ms"])
        state["add_count"] += 1
    elif action == "reduce":
        if state["position"] == 0 or qty <= 0:
            raise ValueError(f"{event['event_id']}: reduce without active position")
        direction = state["direction"]
        if after != state["position"] - Decimal(direction) * qty:
            raise ValueError(f"{event['event_id']}: reduce amount disagrees with position_after")
        if after and (1 if after > 0 else -1) != direction:
            raise ValueError(f"{event['event_id']}: reduce changed position direction")
        state["position"] = after
        state["last_action_time_ms"] = int(event["time_ms"])
        if after == 0:
            state.update(
                {"avgcost": None, "first_qty": None, "open_time_ms": None, "cycle_id": None, "direction": 0, "add_count": 0}
            )
    elif action == "close":
        if state["position"] == 0 or after != 0:
            raise ValueError(f"{event['event_id']}: invalid close transition")
        state.update(
            {
                "position": Decimal(0),
                "avgcost": None,
                "first_qty": None,
                "open_time_ms": None,
                "last_action_time_ms": int(event["time_ms"]),
                "add_count": 0,
                "cycle_id": None,
                "direction": 0,
                "known": True,
            }
        )
    else:
        raise ValueError(f"{event['event_id']}: unsupported history_anchored action {action}")
    if state["position"] != after:
        raise ValueError(f"{event['event_id']}: replay result disagrees with position_after")


def blank_state() -> dict:
    return {
        "position": Decimal(0),
        "avgcost": None,
        "first_qty": None,
        "open_time_ms": None,
        "last_action_time_ms": None,
        "add_count": 0,
        "cycle_id": None,
        "direction": 0,
        "known": True,
    }


def state_features_at(state: dict, t_ms: int, market: np.ndarray) -> np.ndarray:
    out = np.full(8, np.nan, dtype=np.float64)
    if not state["known"] or state["position"] == 0:
        return out
    avg = state["avgcost"]
    pos = state["position"]
    mark = market[8]
    out[0] = float(avg)
    out[1] = float(pos)
    out[2] = float(state["first_qty"])
    out[3] = max(0, (t_ms - state["open_time_ms"]) // MINUTE_MS)
    out[4] = max(0, (t_ms - state["last_action_time_ms"]) // MINUTE_MS)
    out[5] = state["add_count"]
    if np.isfinite(mark):
        out[6] = state["direction"] * (mark / float(avg) - 1.0)
    out[7] = abs(float(pos / state["first_qty"]))
    return out


def feature_object(values: np.ndarray) -> dict:
    return {name: (float(value) if np.isfinite(value) else None) for name, value in zip(FEATURE_NAMES, values)}


def required_features_ready(row: int, features: np.ndarray, direction: np.ndarray) -> bool:
    if not np.isfinite(features[row, :9]).all():
        return False
    # Flat-state position columns are intentionally NaN; held-state columns are required.
    return direction[row] == 0 or bool(np.isfinite(features[row, 9:]).all())


def make_row(symbol_idx: int, minute_idx: int, minute_count: int) -> int:
    return symbol_idx * minute_count + minute_idx


def main() -> None:
    supplement_manifest = read_json(SUPPLEMENT_MANIFEST)
    if supplement_manifest.get("status") != "complete":
        raise ValueError(f"reverse market supplement status is {supplement_manifest.get('status')!r}; require complete")
    supplement_results = supplement_manifest["results"]
    events_all = read_json(EVENTS_PATH)
    cycles_all = read_json(CYCLES_PATH)
    anchored = [e for e in events_all if e["classification"] == "history_anchored"]
    symbols = sorted({e["symbol"] for e in anchored})
    if len(symbols) != 33:
        raise ValueError(f"expected 33 history_anchored symbols, got {len(symbols)}")
    if BENCHMARK in symbols:
        raise ValueError("BTC benchmark must not become an event symbol")
    events = sorted(anchored, key=lambda e: (int(e["order_time_ms"]), int(e["time_ms"]), e["event_id"]))
    if len({e["event_id"] for e in events}) != len(events):
        raise ValueError("history_anchored event_id values are not unique")
    event_index = {e["event_id"]: i for i, e in enumerate(events)}
    cycle_ids = sorted({e["cycle_id"] for e in events})
    cycle_index = {cycle_id: i for i, cycle_id in enumerate(cycle_ids)}
    cycle_records = {c["cycle_id"]: c for c in cycles_all}
    if any(cycle_id not in cycle_records for cycle_id in cycle_ids):
        raise ValueError("an anchored event refers to a missing cycle record")

    all_symbols = symbols + [BENCHMARK]
    market_rows = {}
    coverage = {}
    for symbol in all_symbols:
        market_rows[symbol], coverage[symbol] = merge_symbol_rows(symbol, supplement_results)
    if not market_rows[BENCHMARK]:
        raise ValueError("BTC benchmark has no Klines")
    exchange_info = read_json(BASE_MARKET / "exchange_info.json")
    onboard_dates = {item["symbol"]: int(item["onboardDate"]) for item in exchange_info["symbols"]}
    for symbol in all_symbols:
        onboard_ms = onboard_dates.get(symbol)
        if onboard_ms is None:
            coverage[symbol].update({
                "current_onboard_candidate_ms": None,
                "candidate_grid_minutes": None,
                "observed_candidate_kline_minutes": None,
                "missing_candidate_grid_minutes": None,
            })
            continue
        candidate_start = ((onboard_ms + MINUTE_MS - 1) // MINUTE_MS) * MINUTE_MS
        grid_start = max(PANEL_START_MS, candidate_start)
        candidate_grid = max(0, (CUTOFF_MS - grid_start) // MINUTE_MS)
        opens = market_rows[symbol]
        observed = sum(grid_start <= open_ms <= END_OPEN_MS for open_ms in opens)
        coverage[symbol].update({
            "current_onboard_candidate_ms": onboard_ms,
            "current_onboard_candidate_first_minute_ms": candidate_start,
            "calendar_minutes_before_candidate_in_panel": max(0, (min(CUTOFF_MS, candidate_start) - PANEL_START_MS) // MINUTE_MS),
            "candidate_grid_minutes": candidate_grid,
            "observed_candidate_kline_minutes": observed,
            "missing_candidate_grid_minutes": candidate_grid - observed,
            "observed_kline_minutes_before_candidate": sum(PANEL_START_MS <= t < grid_start for t in opens),
        })

    times = np.arange(PANEL_START_MS, CUTOFF_MS, MINUTE_MS, dtype=np.int64)
    if times.size == 0 or times[-1] != END_OPEN_MS:
        raise ValueError("panel cutoff is not aligned to the minute grid")
    minute_count = len(times)
    row_count = len(symbols) * minute_count
    features = np.full((row_count, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    symbol_idx = np.repeat(np.arange(len(symbols), dtype=np.int16), minute_count)
    time_ms = np.tile(times, len(symbols))
    cycle_idx = np.full(row_count, -1, dtype=np.int32)
    direction = np.zeros(row_count, dtype=np.int8)
    label = np.zeros(row_count, dtype=np.int8)
    label_mask = np.zeros(row_count, dtype=bool)
    fill_label = np.zeros(row_count, dtype=np.int8)
    fill_label_mask = np.zeros(row_count, dtype=bool)
    label_direction = np.zeros(row_count, dtype=np.int8)
    fill_label_direction = np.zeros(row_count, dtype=np.int8)
    event_idx = np.full(row_count, -1, dtype=np.int32)
    fill_event_idx = np.full(row_count, -1, dtype=np.int32)
    state_known_mask = np.zeros(row_count, dtype=bool)
    feature_ready_mask = np.zeros(row_count, dtype=bool)
    open_negative_eligible = np.zeros(row_count, dtype=bool)
    market_exit_risk_eligible = np.zeros(row_count, dtype=bool)
    limit_pending = np.zeros(row_count, dtype=bool)
    next_open = np.full(row_count, np.nan, dtype=np.float32)
    current_kline_mask = np.zeros(row_count, dtype=bool)
    known_tradable_mask = np.zeros(row_count, dtype=bool)

    symbol_index = {symbol: i for i, symbol in enumerate(symbols)}
    events_by_symbol = collections.defaultdict(list)
    for e in events:
        if e["action"] not in ACTION_CODES:
            raise ValueError(f"unsupported action for {e['event_id']}: {e['action']}")
        events_by_symbol[e["symbol"]].append(e)

    market_creation_groups = collections.defaultdict(list)
    market_fill_groups = collections.defaultdict(list)
    any_creation_groups = collections.defaultdict(list)
    any_fill_groups = collections.defaultdict(list)
    for e in events:
        si = symbol_index[e["symbol"]]
        creation_ms = int(e["order_time_ms"])
        fill_ms = int(e["time_ms"])
        if fill_ms < creation_ms:
            raise ValueError(f"{e['event_id']}: fill/update precedes order creation")
        creation_key = (si, (creation_ms // MINUTE_MS) * MINUTE_MS)
        fill_key = (si, (fill_ms // MINUTE_MS) * MINUTE_MS)
        any_creation_groups[creation_key].append(event_index[e["event_id"]])
        any_fill_groups[fill_key].append(event_index[e["event_id"]])
        if e["source_order_type"].upper() == "MARKET":
            market_creation_groups[creation_key].append(event_index[e["event_id"]])
            market_fill_groups[fill_key].append(event_index[e["event_id"]])

    unanchored_tail_events = [
        e for e in events_all
        if e["classification"] != "history_anchored"
        and e["symbol"] in symbols
        and e["classification"] == "inferred_open_after_latest_closed_anchor"
    ]
    unknown_tail_start = {}
    for e in unanchored_tail_events:
        symbol = e["symbol"]
        fill_ms = int(e["time_ms"])
        start_next_minute = (fill_ms // MINUTE_MS) * MINUTE_MS + MINUTE_MS
        unknown_tail_start[symbol] = min(unknown_tail_start.get(symbol, CUTOFF_MS), start_next_minute)

    unknown_until = initial_unknown_until(cycles_all, symbols)
    state_snapshots: dict[tuple[str, int], dict] = {}

    btc_map = market_rows[BENCHMARK]
    needed_snapshot_keys = set()
    for e in events:
        for event_ms in (int(e["order_time_ms"]), int(e["time_ms"])):
            minute = (event_ms // MINUTE_MS) * MINUTE_MS
            if PANEL_START_MS <= minute < CUTOFF_MS:
                needed_snapshot_keys.add((e["symbol"], minute))
    for si, symbol in enumerate(symbols):
        bars = market_rows[symbol]
        market_x, current_mask, tradable_mask = panel_market_features(symbol, times, bars, btc_map)
        row0 = si * minute_count
        row1 = row0 + minute_count
        features[row0:row1, :9] = market_x.astype(np.float32)
        current_kline_mask[row0:row1] = current_mask
        known_tradable_mask[row0:row1] = tradable_mask
        opens = {open_ms: float(row[1]) for open_ms, row in bars.items()}
        for mi, t in enumerate(times):
            if int(t) + MINUTE_MS < CUTOFF_MS and int(t) + MINUTE_MS in opens:
                next_open[row0 + mi] = opens[int(t) + MINUTE_MS]

        state = blank_state()
        state_events = sorted(events_by_symbol[symbol], key=event_sort_key)
        cursor = 0
        history_unknown_end = unknown_until[symbol]
        for mi, t in enumerate(times):
            while cursor < len(state_events) and int(state_events[cursor]["time_ms"]) < int(t):
                apply_event(state, state_events[cursor])
                cursor += 1
            state_known = state["known"] and t >= history_unknown_end
            if symbol in unknown_tail_start and t >= unknown_tail_start[symbol]:
                state_known = False
            row = row0 + mi
            if state_known and state["position"] != 0:
                current_cycle = state["cycle_id"]
                cycle_idx[row] = cycle_index[current_cycle]
                direction[row] = state["direction"]
                state_for_features = state
            else:
                state_known_mask[row] = state_known
                state_for_features = state if state_known else {**state, "known": False}
            if state_known and state["position"] != 0:
                state_known_mask[row] = True
            elif state_known:
                state_known_mask[row] = True
            position = state_features_at(state_for_features, int(t), market_x[mi])
            features[row, 9:] = position.astype(np.float32)
            feature_ready_mask[row] = state_known and required_features_ready(row, features, direction)
            snapshot = dict(state)
            snapshot["known"] = bool(state_known)
            if (symbol, int(t)) in needed_snapshot_keys:
                state_snapshots[(symbol, int(t))] = snapshot
            if state_known and state["position"] != 0:
                market_exit_risk_eligible[row] = feature_ready_mask[row]

        # Event-row links and within-minute fill/position transitions.
        for e in state_events:
            ei = event_index[e["event_id"]]
            creation_minute = (int(e["order_time_ms"]) // MINUTE_MS) * MINUTE_MS
            fill_minute = (int(e["time_ms"]) // MINUTE_MS) * MINUTE_MS
            if PANEL_START_MS <= creation_minute < CUTOFF_MS:
                row = row0 + (creation_minute - PANEL_START_MS) // MINUTE_MS
                if e["source_order_type"].upper() == "MARKET":
                    market_events = market_creation_groups[(si, creation_minute)]
                    if len(market_events) == 1:
                        event_idx[row] = ei
                    else:
                        event_idx[row] = -2
            if PANEL_START_MS <= fill_minute < CUTOFF_MS:
                row = row0 + (fill_minute - PANEL_START_MS) // MINUTE_MS
                if e["source_order_type"].upper() == "MARKET":
                    group = market_fill_groups[(si, fill_minute)]
                    if len(group) == 1:
                        fill_event_idx[row] = ei
                    else:
                        fill_event_idx[row] = -2
            if e["source_order_type"].upper() == "LIMIT":
                create_ms = int(e["order_time_ms"])
                fill_ms = int(e["time_ms"])
                if fill_ms > create_ms:
                    left = max(0, int(np.searchsorted(times, create_ms, side="right")))
                    right = min(minute_count, int(np.searchsorted(times, fill_ms, side="left")))
                    if right > left:
                        limit_pending[row0 + left : row0 + right] = True

    excluded_event_minutes = set()
    for e in events_all:
        if e["classification"] == "history_anchored" or e["symbol"] not in symbol_index:
            continue
        si = symbol_index[e["symbol"]]
        for event_ms, minute_groups in (
            (int(e["order_time_ms"]), any_creation_groups),
            (int(e["time_ms"]), any_fill_groups),
        ):
            minute = (event_ms // MINUTE_MS) * MINUTE_MS
            key = (si, minute)
            minute_groups[key].append(-1)
            excluded_event_minutes.add(key)

    # Attach main protocol labels on order creation time; expose actual-fill labels separately.
    label_exclusions = collections.Counter()
    for (si, minute), indices in market_creation_groups.items():
        if not (PANEL_START_MS <= minute < CUTOFF_MS):
            label_exclusions["market_creation_outside_panel"] += len(indices)
            continue
        row = make_row(si, (minute - PANEL_START_MS) // MINUTE_MS, minute_count)
        if len(indices) != 1:
            label_exclusions["same_minute_multiple_market_creation"] += len(indices)
            continue
        ei = indices[0]
        e = events[ei]
        code = ACTION_CODES[e["action"]]
        label[row] = code
        label_direction[row] = 1 if e["direction"].upper() == "LONG" else -1
        snap = state_snapshots[(symbols[si], minute)]
        state_action_valid = (
            (e["action"] == "open" and snap["position"] == 0 and snap["known"])
            or (e["action"] != "open" and snap["position"] != 0 and snap["cycle_id"] == e["cycle_id"] and snap["known"])
        )
        other_fill_this_minute = any(
            fill_ei >= 0 and fill_ei != ei
            for fill_ei in any_fill_groups.get((si, minute), [])
        )
        if (si, minute) in excluded_event_minutes or other_fill_this_minute:
            label_exclusions["creation_label_same_minute_other_position_event"] += 1
        elif not state_action_valid:
            label_exclusions["creation_label_state_unknown_or_mismatch"] += 1
        elif not feature_ready_mask[row]:
            label_exclusions["creation_label_incomplete_features"] += 1
        else:
            label_mask[row] = True
    for (si, minute), indices in market_fill_groups.items():
        if not (PANEL_START_MS <= minute < CUTOFF_MS):
            label_exclusions["market_fill_outside_panel"] += len(indices)
            continue
        row = make_row(si, (minute - PANEL_START_MS) // MINUTE_MS, minute_count)
        if len(indices) != 1:
            label_exclusions["same_minute_multiple_market_fill"] += len(indices)
            continue
        ei = indices[0]
        fill_label[row] = ACTION_CODES[events[ei]["action"]]
        fill_label_direction[row] = 1 if events[ei]["direction"].upper() == "LONG" else -1
        snap = state_snapshots[(symbols[si], minute)]
        e = events[ei]
        state_action_valid = (
            (e["action"] == "open" and snap["position"] == 0 and snap["known"])
            or (e["action"] != "open" and snap["position"] != 0 and snap["cycle_id"] == e["cycle_id"] and snap["known"])
        )
        other_fill_this_minute = any(fill_ei >= 0 and fill_ei != ei for fill_ei in any_fill_groups.get((si, minute), []))
        if (
            state_action_valid
            and feature_ready_mask[row]
            and not other_fill_this_minute
            and (si, minute) not in excluded_event_minutes
        ):
            fill_label_mask[row] = True
        else:
            label_exclusions["fill_label_state_or_feature_invalid"] += 1

    # Unlabeled rows are no-op only when the position and causal features are known.
    for si in range(len(symbols)):
        row0 = si * minute_count
        row1 = row0 + minute_count
        for mi, minute in enumerate(times):
            row = row0 + mi
            creation_events = any_creation_groups.get((si, int(minute)), [])
            fill_events = any_fill_groups.get((si, int(minute)), [])
            market_at_creation = market_creation_groups.get((si, int(minute)), [])
            if market_at_creation:
                continue
            if creation_events or fill_events:
                continue
            if not state_known_mask[row]:
                continue
            if not feature_ready_mask[row]:
                continue
            label_mask[row] = True
            if direction[row] == 0 and known_tradable_mask[row]:
                open_negative_eligible[row] = True
    # Event minutes are transition/intent minutes, not no-action exit opportunities.
    for (si, minute) in set(any_creation_groups) | set(any_fill_groups):
        if PANEL_START_MS <= minute < CUTOFF_MS:
            row = make_row(si, (minute - PANEL_START_MS) // MINUTE_MS, minute_count)
            valid_exit = (label_mask[row] and label[row] in (3, 4)) or (
                fill_label_mask[row] and fill_label[row] in (3, 4)
            )
            if not valid_exit:
                market_exit_risk_eligible[row] = False
    market_exit_risk_eligible_no_pending = market_exit_risk_eligible & ~limit_pending

    # Per-event causal features at creation and fill minutes, including pre-panel events.
    event_json = []
    for ei, e in enumerate(events):
        symbol = e["symbol"]
        si = symbol_index[symbol]
        creation_ms = int(e["order_time_ms"])
        fill_ms = int(e["time_ms"])
        creation_minute = (creation_ms // MINUTE_MS) * MINUTE_MS
        fill_minute = (fill_ms // MINUTE_MS) * MINUTE_MS
        creation_row = -1
        fill_row = -1
        if PANEL_START_MS <= creation_minute < CUTOFF_MS:
            creation_row = make_row(si, (creation_minute - PANEL_START_MS) // MINUTE_MS, minute_count)
            creation_values = features[creation_row].astype(np.float64)
        else:
            creation_market = market_features_at(symbol, creation_minute, market_rows[symbol], btc_map)
            creation_state = blank_state()
            for previous in sorted(events_by_symbol[symbol], key=event_sort_key):
                if int(previous["time_ms"]) < creation_minute:
                    apply_event(creation_state, previous)
                else:
                    break
            creation_state["known"] = creation_state["known"] and creation_minute >= unknown_until[symbol]
            creation_values = np.concatenate([creation_market, state_features_at(creation_state, creation_minute, creation_market)])
        if PANEL_START_MS <= fill_minute < CUTOFF_MS:
            fill_row = make_row(si, (fill_minute - PANEL_START_MS) // MINUTE_MS, minute_count)
            fill_values = features[fill_row].astype(np.float64)
        else:
            fill_market = market_features_at(symbol, fill_minute, market_rows[symbol], btc_map)
            fill_state = blank_state()
            for previous in sorted(events_by_symbol[symbol], key=event_sort_key):
                if int(previous["time_ms"]) < fill_minute:
                    apply_event(fill_state, previous)
                else:
                    break
            fill_state["known"] = fill_state["known"] and fill_minute >= unknown_until[symbol]
            fill_values = np.concatenate([fill_market, state_features_at(fill_state, fill_minute, fill_market)])
        row_event = dict(e)
        row_event.update(
            {
                "event_idx": ei,
                "cycle_idx": cycle_index[e["cycle_id"]],
                "creation_time_ms": creation_ms,
                "fill_time_ms": fill_ms,
                "creation_minute_ms": creation_minute,
                "fill_minute_ms": fill_minute,
                "creation_row": creation_row,
                "fill_row": fill_row,
                "market_creation_label": ACTION_CODES[e["action"]] if e["source_order_type"].upper() == "MARKET" else None,
                "market_fill_label": ACTION_CODES[e["action"]] if e["source_order_type"].upper() == "MARKET" else None,
                "features_at_creation_minute": feature_object(creation_values),
                "features_at_fill_minute": feature_object(fill_values),
            }
        )
        event_json.append(row_event)

    # Panel-level causal/data checks.
    if len(symbol_idx) != row_count or len(time_ms) != row_count or features.shape != (row_count, len(FEATURE_NAMES)):
        raise ValueError("panel array shapes are inconsistent")
    if np.any(direction[cycle_idx == -1] != 0) or np.any(cycle_idx[direction == 0] != -1):
        raise ValueError("flat cycle/direction state encoding is inconsistent")
    for row in np.flatnonzero(label_mask & (label > 0)):
        ei = int(event_idx[row])
        if ei < 0 or events[ei]["source_order_type"].upper() != "MARKET":
            raise ValueError("MARKET creation label does not map to a MARKET event")
        if int(time_ms[row]) != (int(events[ei]["order_time_ms"]) // MINUTE_MS) * MINUTE_MS:
            raise ValueError("creation label is not on its protocol orderTime minute")
    for row in np.flatnonzero(fill_label_mask & (fill_label > 0)):
        ei = int(fill_event_idx[row])
        if ei < 0 or events[ei]["source_order_type"].upper() != "MARKET":
            raise ValueError("fill label does not map to a MARKET event")
        if int(time_ms[row]) != (int(events[ei]["time_ms"]) // MINUTE_MS) * MINUTE_MS:
            raise ValueError("fill label is not on its MARKET updateTime minute")
    if np.any(open_negative_eligible & ((label != 0) | (direction != 0) | ~state_known_mask | ~known_tradable_mask | ~feature_ready_mask)):
        raise ValueError("open negative eligibility violates flat/known-state requirements")
    causal_close_violations = 0
    next_open_violations = 0
    for si, symbol in enumerate(symbols):
        row0 = si * minute_count
        bars = market_rows[symbol]
        expected_close = np.array(
            [np.float32(float(bars[int(t) - MINUTE_MS][4])) if int(t) - MINUTE_MS in bars else np.nan for t in times],
            dtype=np.float32,
        )
        actual_close = features[row0 : row0 + minute_count, 8]
        expected_valid = np.isfinite(expected_close)
        actual_valid = np.isfinite(actual_close)
        causal_close_violations += int(np.count_nonzero(expected_valid != actual_valid))
        comparable = expected_valid & actual_valid
        if comparable.any():
            causal_close_violations += int(np.count_nonzero(actual_close[comparable] != expected_close[comparable]))
        close_time_violations = sum(
            int(int(bars[int(t) - MINUTE_MS][6]) >= int(t))
            for t in times
            if int(t) - MINUTE_MS in bars
        )
        causal_close_violations += close_time_violations

        expected_next_open = np.array(
            [
                np.float32(float(bars[int(t) + MINUTE_MS][1]))
                if int(t) + MINUTE_MS < CUTOFF_MS and int(t) + MINUTE_MS in bars
                else np.nan
                for t in times
            ],
            dtype=np.float32,
        )
        actual_next_open = next_open[row0 : row0 + minute_count]
        expected_next_valid = np.isfinite(expected_next_open)
        actual_next_valid = np.isfinite(actual_next_open)
        next_open_violations += int(np.count_nonzero(expected_next_valid != actual_next_valid))
        comparable_next = expected_next_valid & actual_next_valid
        next_open_violations += int(
            np.count_nonzero(actual_next_open[comparable_next] != expected_next_open[comparable_next])
        )
    if causal_close_violations or next_open_violations:
        raise ValueError(
            f"causal audit found close violations={causal_close_violations}, "
            f"next_open violations={next_open_violations}"
        )

    npz_tmp = PANEL_PATH.with_name(f".{PANEL_PATH.name}.tmp.npz")
    np.savez_compressed(
        npz_tmp,
        symbol_idx=symbol_idx,
        time_ms=time_ms,
        cycle_idx=cycle_idx,
        direction=direction,
        label=label,
        label_mask=label_mask,
        fill_label=fill_label,
        fill_label_mask=fill_label_mask,
        label_direction=label_direction,
        fill_label_direction=fill_label_direction,
        event_idx=event_idx,
        fill_event_idx=fill_event_idx,
        state_known_mask=state_known_mask,
        feature_ready_mask=feature_ready_mask,
        open_negative_eligible=open_negative_eligible,
        market_exit_risk_eligible=market_exit_risk_eligible,
        market_exit_risk_eligible_no_pending=market_exit_risk_eligible_no_pending,
        limit_pending=limit_pending,
        next_open=next_open,
        next_open_time_ms=np.tile(times + MINUTE_MS, len(symbols)),
        current_kline_mask=current_kline_mask,
        known_tradable_mask=known_tradable_mask,
        X=features,
        feature_names=np.asarray(FEATURE_NAMES, dtype="U48"),
        symbols=np.asarray(symbols, dtype="U32"),
        cycle_ids=np.asarray(cycle_ids, dtype="U64"),
        event_ids=np.asarray([e["event_id"] for e in events], dtype="U48"),
        timeline_ms=times,
    )
    os.replace(npz_tmp, PANEL_PATH)

    cycle_open_ms = {
        cycle_id: int(cycle_records[cycle_id]["start_ms"])
        for cycle_id in cycle_ids
    }
    cross_start_cycles = [cycle_id for cycle_id in cycle_ids if cycle_open_ms[cycle_id] < PANEL_START_MS]
    right_truncated_event_ids = [e["event_id"] for e in unanchored_tail_events]
    feature_nan_counts = {
        name: int(np.count_nonzero(~np.isfinite(features[:, i]))) for i, name in enumerate(FEATURE_NAMES)
    }
    exclusions = {
        "events_total": len(events_all),
        "events_history_anchored_in_panel": len(events),
        "events_excluded_by_classification": len(events_all) - len(events),
        "excluded_event_ids": [e["event_id"] for e in events_all if e["classification"] != "history_anchored"],
        "unanchored_symbols_excluded_entirely": sorted(
            {e["symbol"] for e in events_all if e["classification"] != "history_anchored"} - set(symbols)
        ),
        "unanchored_tail_symbols_retained_with_unknown_state": sorted(unknown_tail_start),
        "market_creation_label_exclusions": dict(label_exclusions),
        "same_minute_multiple_market_creation_event_count": sum(
            len(v) for v in market_creation_groups.values() if len(v) > 1
        ),
        "same_minute_multiple_market_fill_event_count": sum(
            len(v) for v in market_fill_groups.values() if len(v) > 1
        ),
        "limit_events_unlabeled_as_market": sum(e["source_order_type"].upper() == "LIMIT" for e in events),
        "rows_total": row_count,
        "rows_with_known_state": int(state_known_mask.sum()),
        "rows_without_known_tradable_kline_history": int((~known_tradable_mask).sum()),
        "rows_with_incomplete_market_features": int((~np.isfinite(features[:, :9]).all(axis=1)).sum()),
        "held_rows_with_incomplete_position_features": int(
            np.count_nonzero((direction != 0) & ~np.isfinite(features[:, 9:]).all(axis=1))
        ),
        "rows_with_required_features_ready": int(feature_ready_mask.sum()),
        "flat_rows_with_expected_nan_position_features": int(
            np.count_nonzero((direction == 0) & ~np.isfinite(features[:, 9:]).all(axis=1))
        ),
        "open_negative_eligible_rows": int(open_negative_eligible.sum()),
        "market_creation_label_rows": int(np.count_nonzero(label_mask & (label > 0))),
        "market_fill_label_rows": int(np.count_nonzero(fill_label_mask & (fill_label > 0))),
        "pending_limit_rows": int(limit_pending.sum()),
        "market_exit_risk_rows_all": int(market_exit_risk_eligible.sum()),
        "market_exit_risk_rows_no_pending_limit": int(market_exit_risk_eligible_no_pending.sum()),
        "cross_start_cycle_ids": cross_start_cycles,
        "inferred_right_truncated_event_ids_excluded_from_gold_state": right_truncated_event_ids,
        "unknown_tail_start_ms_by_symbol": unknown_tail_start,
        "feature_nan_counts": feature_nan_counts,
        "causal_close_audit_violations": causal_close_violations,
        "next_open_outcome_audit_violations": next_open_violations,
    }
    meta = {
        "schema_version": 1,
        "panel_file": str(PANEL_PATH.relative_to(ROOT)),
        "event_features_file": str(EVENT_FEATURES_PATH.relative_to(ROOT)),
        "frequency": "1m",
        "row_order": "symbol-major, then ascending UTC minute start",
        "start_ms_inclusive": PANEL_START_MS,
        "start_utc_inclusive": "2026-09-03T00:00:00Z",
        "cutoff_ms_exclusive": CUTOFF_MS,
        "cutoff_utc_exclusive": "2026-10-02T11:27:00Z",
        "last_minute_start_ms": END_OPEN_MS,
        "minute_count_per_symbol": minute_count,
        "row_count": row_count,
        "symbols": symbols,
        "symbol_idx_to_symbol": symbols,
        "benchmark_symbol": BENCHMARK,
        "cycle_ids": cycle_ids,
        "cycle_idx_to_cycle_id": cycle_ids,
        "event_ids": [e["event_id"] for e in events],
        "event_idx_to_event_id": [e["event_id"] for e in events],
        "event_features_events_index": "reverse_event_features.json:events[event_idx]",
        "dtypes": {
            "symbol_idx": "int16",
            "time_ms": "int64",
            "cycle_idx": "int32; -1 when flat",
            "direction": "int8; short=-1,long=1,flat=0",
            "label": "int8; MARKET orderTime action open=1/add=2/reduce=3/close=4/noop=0",
            "label_mask": "bool; valid MARKET creation target or known no-op; false for LIMIT-only event minutes, ambiguous actions, unknown state, or missing applicable features",
            "fill_label": "int8; MARKET updateTime fill action, separate from protocol creation label",
            "fill_label_mask": "bool; valid MARKET actual-fill label",
            "label_direction": "int8; direction for all positive MARKET actions, long=1/short=-1/noop=0",
            "fill_label_direction": "int8; direction for all positive MARKET fill actions",
            "feature_ready_mask": "bool; market features complete; held-state features complete when held; flat-state position NaNs are expected",
            "event_idx": "int32; event index for a unique MARKET creation event row, -1 no unique mapped event, -2 multiple MARKET creation events",
            "fill_event_idx": "int32; event index for a unique MARKET fill row, -1 no unique mapped event, -2 multiple MARKET fills",
            "X": "float32[row_count, feature_count]; missing data remains NaN",
            "next_open": "float32; next minute open, execution/outcome only and excluded from X",
        },
        "feature_names": FEATURE_NAMES,
        "feature_groups": {"market": MARKET_FEATURE_NAMES, "position": POSITION_FEATURE_NAMES},
        "feature_definitions": {
            "ret5": "close(t-1m)/close(t-6m)-1 using consecutive completed 1m bars",
            "ret15": "close(t-1m)/close(t-16m)-1 using consecutive completed 1m bars",
            "ret60": "close(t-1m)/close(t-61m)-1 using consecutive completed 1m bars",
            "qv5_vs_prev60_mean": "mean quote_volume of bars [t-5m,t) divided by mean quote_volume of bars [t-60m,t)",
            "volatility60": "population std of 60 consecutive close-to-close 1m returns ending at the last completed bar",
            "range15": "max(high)/min(low)-1 over bars [t-15m,t)",
            "btc_relative15": "symbol ret15 minus BTCUSDT ret15",
            "btc_relative60": "symbol ret60 minus BTCUSDT ret60",
            "close": "last completed close with close_time_ms < t; mark/ledger value only, not an execution price",
            "avgcost": "weighted entry price from successful position events strictly before t",
            "position": "signed base-asset quantity after fills strictly before t",
            "first_qty": "base-asset quantity of the active cycle's first open",
            "age_minutes": "floor((t - first open fill time)/60000)",
            "minutes_last_action": "floor((t - last successful fill time)/60000)",
            "add_count": "number of successful add fills in the active cycle strictly before t",
            "mark_return": "direction * (last completed close / avgcost - 1)",
            "position_multiple": "abs(current position / first_qty)",
        },
        "time_and_state_contract": {
            "panel_time": "UTC minute start; all 33 symbols have one row for every minute in the interval, including minutes with no Kline",
            "market_features": "Only Klines whose close_time_ms < t are used; no missing candle is forward-filled",
            "state_features": "Leader successful fills are replayed by event.time_ms (orderUpdateTime); apply only when event.time_ms < t",
            "protocol_decision_label_time": "MARKET orderTime (creation), floored to its UTC minute; current state row remains pre-fill",
            "actual_fill_label_time": "MARKET orderUpdateTime, floored to its UTC minute and stored separately as fill_label",
            "limit_handling": "LIMIT creation and fill are stored as event timestamps; neither is labeled as a MARKET action",
            "limit_pending": "true where a LIMIT order was created before t and its observed fill/update time is at or after t",
            "mark_price": "X.close is last completed close, never current-minute close or next-open execution price",
            "next_open": "next_open[t] is the raw open of the t+1 minute candle, for downstream execution accounting only; it is not an input feature",
            "no_price_fill": "No missing Kline or feature is replaced with a synthetic price or zero",
            "open_negative_eligible": "Known flat state, prior valid Kline, complete market features, and no observed event in that minute; flat position fields remain NaN",
            "market_exit_risk_eligible": "Known active position with applicable features and no same-minute ambiguity; retains valid MARKET reduce/close positives",
            "market_exit_risk_eligible_no_pending": "market_exit_risk_eligible with pending observed LIMIT intervals removed",
            "universe_boundary": "Only the 33 history_anchored event symbols are modeled; this panel cannot identify full-market coin selection",
        },
        "market_coverage": coverage,
        "exclusions": exclusions,
        "cycle_purge": {
            "cycle_idx_is_group_key": True,
            "cross_panel_start_cycle_ids": cross_start_cycles,
            "purge_rule": "Keep all rows/events from a cycle in one split; purge these IDs from a held-out evaluation when their cycle starts before panel start.",
        },
    }
    event_payload = {
        "schema_version": 1,
        "event_idx_mapping": "events[event_idx]",
        "symbols": symbols,
        "cycle_ids": cycle_ids,
        "feature_names": FEATURE_NAMES,
        "label_contract": "market_creation_label uses orderTime; market_fill_label uses orderUpdateTime; LIMIT has null labels",
        "events": event_json,
        "events_excluded": [e for e in events_all if e["classification"] != "history_anchored"],
    }
    write_json_atomic(EVENT_FEATURES_PATH, event_payload)
    write_json_atomic(META_PATH, meta)

    print(
        json.dumps(
            {
                "status": "complete",
                "rows": row_count,
                "symbols": len(symbols),
                "cycles": len(cycle_ids),
                "events": len(events),
                "market_creation_labels": int(np.count_nonzero(label_mask & (label > 0))),
                "market_fill_labels": int(np.count_nonzero(fill_label_mask & (fill_label > 0))),
                "open_negative_eligible": int(open_negative_eligible.sum()),
                "X_shape": list(features.shape),
                "outputs": [str(PANEL_PATH), str(META_PATH), str(EVENT_FEATURES_PATH)],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def load_panel(panel_path: str | pathlib.Path = PANEL_PATH) -> dict:
    """Load panel arrays and their metadata for downstream rule/model scripts."""
    panel_path = pathlib.Path(panel_path)
    meta_path = panel_path.with_name("reverse_panel_meta.json")
    with np.load(panel_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    return arrays


if __name__ == "__main__":
    main()
