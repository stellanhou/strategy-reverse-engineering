#!/usr/bin/env python3
"""Describe leader event mechanics from reconstructed order sequences.

This first pass deliberately depends only on audited events and cycles. It
reconstructs weighted cost from event quantities/prices, including at exits;
stored exit avg_cost fields are often null and are not used as evidence.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analysis" / "results"
START_MS = 1788393600000  # 2026-09-03 00:00:00 UTC
END_MS = 1790940420000  # 2026-10-02 11:27:00 UTC, exclusive
SEED = 20261002
D = Decimal
ZERO = D("0")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_float=D, parse_int=D)


def _float(v: Any) -> float | None:
    return None if v is None else float(v)


def _q(values: Iterable[float], p: float) -> float | None:
    vals = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not vals:
        return None
    index = (len(vals) - 1) * p
    lo = int(index)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (index - lo)


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return {
        "n": len(vals),
        "median": _q(vals, 0.5),
        "p10": _q(vals, 0.1),
        "p25": _q(vals, 0.25),
        "p75": _q(vals, 0.75),
        "p90": _q(vals, 0.9),
        "mean": sum(vals) / len(vals) if vals else None,
    }


def _utc(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _cycle_events(events: list[dict[str, Any]], cycle_map: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        cid = row["cycle_id"]
        if cid is not None and cid in cycle_map:
            grouped[cid].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda e: (int(e["time_ms"]), str(e["event_id"])))
    return grouped


def reconstruct_cycle(cycle: dict[str, Any], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild signed size and weighted average cost from the order sequence."""
    if cycle["status"] != "complete":
        raise ValueError(f"Cannot calculate full event mechanics for incomplete cycle {cycle['cycle_id']}")
    direction_sign = 1 if cycle["direction"] == "LONG" else -1
    pos = ZERO
    cost = ZERO
    first_qty = ZERO
    first_notional = ZERO
    prior_price: Decimal | None = None
    prior_time: int | None = None
    add_count = 0
    out: list[dict[str, Any]] = []
    realized_total = ZERO

    for event in events:
        action = event["action"]
        qty = D(str(event["qty"]))
        px = D(str(event["price"]))
        signed_qty = qty * D(direction_sign if action in {"open", "add"} else -direction_sign)
        before = pos
        after = before + signed_qty
        if qty <= ZERO or px <= ZERO:
            raise ValueError(f"Non-positive event size/price: {event['event_id']}")

        avg_cost_before: Decimal | None = None
        return_from_cost: Decimal | None = None
        adverse_from_cost: Decimal | None = None
        add_adverse_spacing_from_prev: Decimal | None = None
        realized: Decimal | None = None
        if action == "open":
            if before != ZERO or signed_qty * direction_sign <= ZERO:
                raise ValueError(f"Invalid opening transition in {event['event_id']}")
            pos = after
            cost = px
            first_qty = qty
            first_notional = qty * px
        elif action == "add":
            if before == ZERO or signed_qty * before <= ZERO:
                raise ValueError(f"Invalid add transition in {event['event_id']}")
            avg_cost_before = cost
            return_from_cost = D(direction_sign) * (px / cost - D(1))
            adverse_from_cost = -return_from_cost
            if prior_price is not None:
                add_adverse_spacing_from_prev = -D(direction_sign) * (px / prior_price - D(1))
            cost = (abs(before) * cost + qty * px) / (abs(before) + qty)
            pos = after
            add_count += 1
        elif action in {"reduce", "close"}:
            if before == ZERO or signed_qty * before >= ZERO or after * before < ZERO:
                raise ValueError(f"Invalid reduce/close transition in {event['event_id']}")
            # Exit placement must use the cost reconstructed above, never the
            # source event's nullable avg_cost_before field.
            avg_cost_before = cost
            return_from_cost = D(direction_sign) * (px / cost - D(1))
            closed_qty = min(abs(before), qty)
            realized = (px - cost) * closed_qty * D(1 if before > ZERO else -1)
            realized_total += realized
            pos = after
            if pos == ZERO:
                cost = ZERO
        else:
            raise ValueError(f"Unsupported action in anchored cycle: {action}")

        # Source reconstruction fields are consistency checks only. In
        # particular, null avg_cost_before is expected for close/reduce rows.
        if D(str(event["position_before"])) != before or D(str(event["position_after"])) != after:
            raise ValueError(f"Position transition mismatch: {event['event_id']}")
        stored_cost = event["avg_cost_before"]
        if action in {"add", "reduce", "close"} and stored_cost is not None:
            if abs(D(str(stored_cost)) - avg_cost_before) > D("0.000000000001"):
                raise ValueError(f"Reconstructed/source cost mismatch: {event['event_id']}")

        order_time = int(event["order_time_ms"])
        update_time = int(event["time_ms"])
        order_type = event["source_order_type"]
        out.append({
            "event_id": event["event_id"],
            "cycle_id": event["cycle_id"],
            "symbol": event["symbol"],
            "direction": cycle["direction"],
            "action": action,
            "order_type": order_type,
            "quantity": float(qty),
            "price": float(px),
            "notional": float(qty * px),
            "position_before": float(before),
            "position_after": float(after),
            "weighted_cost_before": _float(avg_cost_before),
            "weighted_cost_after": _float(cost) if pos != ZERO else 0.0,
            "qty_multiple_of_first": float(qty / first_qty) if first_qty else None,
            "notional_multiple_of_first": float(qty * px / first_notional) if first_notional else None,
            "position_multiple_of_first": float(abs(after) / first_qty) if first_qty else None,
            "return_from_cost": _float(return_from_cost),
            "adverse_from_cost": _float(adverse_from_cost),
            "adverse_spacing_from_previous_event_price": _float(add_adverse_spacing_from_prev),
            "minutes_since_previous_event": (update_time - prior_time) / 60000 if prior_time is not None else None,
            "realized_price_pnl": _float(realized),
            "event_time_ms": update_time,
            "order_time_ms": order_time,
            "decision_proxy_time_ms": order_time,
            "fill_time_ms": update_time,
            "order_to_update_delay_ms": update_time - order_time,
        })
        prior_price = px
        prior_time = update_time

    if pos != ZERO:
        raise ValueError(f"Cycle did not return flat: {cycle['cycle_id']}, {pos}")
    source_flow = D(str(cycle["price_flow_pnl"]))
    if abs(realized_total - source_flow) > D("0.000000000001"):
        raise ValueError(f"Rebuilt exit flows do not reconcile: {cycle['cycle_id']}")
    return out


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    adds = [r for r in rows if r["action"] == "add"]
    exits = [r for r in rows if r["action"] in {"reduce", "close"}]
    cycle_open = {r["cycle_id"]: r for r in rows if r["action"] == "open"}
    add_size = [r["notional_multiple_of_first"] for r in adds if r["notional_multiple_of_first"] is not None]
    add_qty_size = [r["qty_multiple_of_first"] for r in adds if r["qty_multiple_of_first"] is not None]
    by_cycle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in adds:
        by_cycle[r["cycle_id"]].append(r)
    total_add_notional_ratio = []
    for cid, items in by_cycle.items():
        opened = cycle_open[cid]
        if opened["notional"] > 0:
            total_add_notional_ratio.append(sum(x["notional"] for x in items) / opened["notional"])
    add_adverse = [r for r in adds if r["adverse_from_cost"] is not None]
    return {
        "event_counts": dict(Counter(r["action"] for r in rows)),
        "add_adverse_vs_weighted_cost_share": (
            sum(r["adverse_from_cost"] > 0 for r in add_adverse) / len(add_adverse) if add_adverse else None
        ),
        "add_at_or_better_than_weighted_cost_count": sum(r["adverse_from_cost"] <= 0 for r in add_adverse),
        "add_adverse_from_cost": _summary(r["adverse_from_cost"] for r in add_adverse),
        "add_adverse_spacing_from_previous_event_price": _summary(
            r["adverse_spacing_from_previous_event_price"] for r in adds
            if r["adverse_spacing_from_previous_event_price"] is not None
        ),
        "add_minutes_since_previous_event": _summary(
            r["minutes_since_previous_event"] for r in adds if r["minutes_since_previous_event"] is not None
        ),
        "add_notional_multiple_of_first": _summary(add_size),
        "add_quantity_multiple_of_first": _summary(add_qty_size),
        "cycle_total_add_notional_multiple_of_first": _summary(total_add_notional_ratio),
        "exit_return_from_reconstructed_weighted_cost": _summary(
            r["return_from_cost"] for r in exits if r["return_from_cost"] is not None
        ),
        "exit_return_by_action": {
            action: _summary(r["return_from_cost"] for r in exits
                             if r["action"] == action and r["return_from_cost"] is not None)
            for action in ("reduce", "close")
        },
        "exit_return_by_order_type": {
            order_type: _summary(r["return_from_cost"] for r in exits
                                 if r["order_type"] == order_type and r["return_from_cost"] is not None)
            for order_type in ("MARKET", "LIMIT")
        },
        "profitable_exit_share_by_price_vs_cost": (
            sum(r["return_from_cost"] > 0 for r in exits if r["return_from_cost"] is not None)
            / sum(r["return_from_cost"] is not None for r in exits)
            if any(r["return_from_cost"] is not None for r in exits) else None
        ),
    }


def _bootstrap(rows: list[dict[str, Any]], span_start: int, span_end: int, block_days: int, reps: int = 4000) -> dict[str, Any]:
    """Calendar-block bootstrap for event-level descriptive shares/medians."""
    import numpy as np

    day_count = max(1, math.ceil((span_end - span_start) / 86_400_000))
    all_days = [
        datetime.fromtimestamp(span_start / 1000, timezone.utc).date().toordinal() + i
        for i in range(day_count)
    ]
    by_day: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ordinal = datetime.fromtimestamp(row["event_time_ms"] / 1000, timezone.utc).date().toordinal()
        by_day[ordinal].append(row)
    day_to_rows = {ordinal: by_day[ordinal] if ordinal in by_day else [] for ordinal in all_days}
    chunks = [all_days[i:i + block_days] for i in range(0, len(all_days), block_days)]
    rng = np.random.default_rng(SEED + block_days)

    def estimate(sample: list[dict[str, Any]]) -> tuple[float | None, float | None]:
        adds = [r for r in sample if r["action"] == "add" and r["adverse_from_cost"] is not None]
        exits = [r for r in sample if r["action"] in {"reduce", "close"} and r["return_from_cost"] is not None]
        add_share = sum(r["adverse_from_cost"] > 0 for r in adds) / len(adds) if adds else None
        exit_median = _q([r["return_from_cost"] for r in exits], 0.5)
        return add_share, exit_median

    observed = estimate(rows)
    boot = [[], []]
    for _ in range(reps):
        selected: list[dict[str, Any]] = []
        sampled_chunks = rng.integers(0, len(chunks), size=len(chunks))
        for idx in sampled_chunks:
            selected.extend(r for ordinal in chunks[int(idx)] for r in day_to_rows[ordinal])
        values = estimate(selected)
        for j, value in enumerate(values):
            if value is not None:
                boot[j].append(value)
    names = ["add_adverse_share", "exit_return_median"]
    result = {"resampling_unit": f"{block_days}-day contiguous UTC calendar blocks", "replicates": reps}
    for j, name in enumerate(names):
        result[name] = {
            "estimate": observed[j],
            "ci95": [_q(boot[j], 0.025), _q(boot[j], 0.975)] if boot[j] else None,
            "valid_replicates": len(boot[j]),
        }
    return result


def _cycle_bootstrap(rows: list[dict[str, Any]], reps: int = 4000) -> dict[str, Any]:
    """Cycle-cluster resampling sensitivity, preserving actions within cycles."""
    import numpy as np

    by_cycle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cycle[row["cycle_id"]].append(row)
    cycle_ids = sorted(by_cycle)
    if not cycle_ids:
        raise ValueError("Cannot bootstrap an empty cycle sample")
    rng = np.random.default_rng(SEED + 17)
    values = {"add_adverse_share": [], "exit_return_median": []}
    for _ in range(reps):
        sampled_ids = rng.choice(cycle_ids, size=len(cycle_ids), replace=True)
        sample = [row for cycle_id in sampled_ids for row in by_cycle[str(cycle_id)]]
        adds = [r for r in sample if r["action"] == "add" and r["adverse_from_cost"] is not None]
        exits = [r for r in sample if r["action"] in {"reduce", "close"} and r["return_from_cost"] is not None]
        if adds:
            values["add_adverse_share"].append(sum(r["adverse_from_cost"] > 0 for r in adds) / len(adds))
        if exits:
            values["exit_return_median"].append(_q([r["return_from_cost"] for r in exits], 0.5))
    return {
        "resampling_unit": "complete position cycle, with all events kept together",
        "replicates": reps,
        **{
            name: {"ci95": [_q(data, 0.025), _q(data, 0.975)], "valid_replicates": len(data)}
            for name, data in values.items()
        },
        "status": "exploratory descriptive sensitivity; not an independent validation interval",
    }


def _folds() -> list[dict[str, int | str]]:
    return [
        {"name": "fold_1", "train_start": 1788393600000, "train_end": 1789257600000, "eval_end": 1789776000000},
        {"name": "fold_2", "train_start": 1788393600000, "train_end": 1789776000000, "eval_end": 1790294400000},
        {"name": "fold_3", "train_start": 1788393600000, "train_end": 1790294400000, "eval_end": END_MS},
    ]


def analyze() -> dict[str, Any]:
    cycles = _read_json(RESULTS / "leader_cycles.json")
    events = _read_json(RESULTS / "leader_events.json")
    cycle_map = {c["cycle_id"]: c for c in cycles}
    grouped = _cycle_events(events, cycle_map)

    eligible_cycles = [
        c for c in cycles
        if c["status"] == "complete"
        and c["start_ms"] >= START_MS
        and c["end_ms"] is not None
        and c["end_ms"] < END_MS
        and c["direction"] in {"SHORT", "LONG"}
    ]
    reconstructed: list[dict[str, Any]] = []
    cycle_metrics: list[dict[str, Any]] = []
    for cycle in eligible_cycles:
        source = grouped[cycle["cycle_id"]]
        if not source:
            raise ValueError(f"Complete anchored cycle has no source events: {cycle['cycle_id']}")
        if any(e["classification"] != "history_anchored" for e in source):
            raise ValueError(f"Unexpected unanchored event in complete history cycle: {cycle['cycle_id']}")
        rows = reconstruct_cycle(cycle, source)
        reconstructed.extend(rows)
        cycle_open = next(r for r in rows if r["action"] == "open")
        adds = [r for r in rows if r["action"] == "add"]
        exits = [r for r in rows if r["action"] in {"reduce", "close"}]
        cycle_metrics.append({
            "cycle_id": cycle["cycle_id"],
            "symbol": cycle["symbol"],
            "direction": cycle["direction"],
            "start_ms": int(cycle["start_ms"]),
            "end_ms": int(cycle["end_ms"]),
            "initial_qty": cycle_open["quantity"],
            "initial_notional": cycle_open["notional"],
            "adds": len(adds),
            "total_add_notional_multiple_of_first": (
                sum(r["notional"] for r in adds) / cycle_open["notional"] if cycle_open["notional"] else None
            ),
            "max_position_multiple_of_first": max((r["position_multiple_of_first"] or 0 for r in rows), default=0),
            "exit_return_from_weighted_cost": next((r["return_from_cost"] for r in exits if r["action"] == "close"), None),
            "exit_price_flow_pnl": sum(r["realized_price_pnl"] or 0 for r in exits),
            "history_closing_pnl": _float(cycle["realized_pnl"]),
        })

    short_rows = [r for r in reconstructed if r["direction"] == "SHORT"]
    long_rows = [r for r in reconstructed if r["direction"] == "LONG"]
    short_cycles = [c for c in cycle_metrics if c["direction"] == "SHORT"]
    long_cycles = [c for c in cycle_metrics if c["direction"] == "LONG"]

    action_order = Counter((r["action"], r["order_type"]) for r in reconstructed)
    delays = {
        kind: _summary(r["order_to_update_delay_ms"] / 60000 for r in reconstructed if r["order_type"] == kind)
        for kind in sorted({r["order_type"] for r in reconstructed})
    }
    by_action_and_type: dict[str, Any] = {}
    for action in ["open", "add", "reduce", "close"]:
        by_action_and_type[action] = {}
        for kind in ["MARKET", "LIMIT"]:
            subset = [r for r in reconstructed if r["action"] == action and r["order_type"] == kind]
            by_action_and_type[action][kind] = {
                "count": len(subset),
                "order_to_update_delay_minutes": _summary(r["order_to_update_delay_ms"] / 60000 for r in subset),
                "action_price_return_from_weighted_cost": _summary(
                    r["return_from_cost"] for r in subset if r["return_from_cost"] is not None
                ),
                "notional_multiple_of_first": _summary(
                    r["notional_multiple_of_first"] for r in subset if r["notional_multiple_of_first"] is not None
                ),
            }

    top_short_symbols = Counter(c["symbol"] for c in short_cycles).most_common(10)
    top_symbol = top_short_symbols[0][0] if top_short_symbols else None
    no_top = [r for r in short_rows if r["symbol"] != top_symbol]
    fold_results = []
    for fold in _folds():
        train_cycles = [c for c in short_cycles if fold["train_start"] <= c["start_ms"] and c["end_ms"] < fold["train_end"]]
        eval_cycles = [c for c in short_cycles if fold["train_end"] <= c["start_ms"] and c["end_ms"] < fold["eval_end"]]
        def subset_rows(which: list[dict[str, Any]]) -> list[dict[str, Any]]:
            ids = {c["cycle_id"] for c in which}
            return [r for r in short_rows if r["cycle_id"] in ids]
        fold_results.append({
            "name": fold["name"],
            "train_cycles": len(train_cycles),
            "eval_cycles": len(eval_cycles),
            "purge_policy": "whole cycles crossing either boundary excluded",
            "train_event_mechanics": _metric_rows(subset_rows(train_cycles)),
            "eval_event_mechanics": _metric_rows(subset_rows(eval_cycles)),
        })

    # Explicit counterexamples retain symbol and event ids so the aggregate
    # claims can be checked against the original order sequence.
    add_exceptions = sorted(
        (r for r in short_rows if r["action"] == "add" and r["adverse_from_cost"] is not None),
        key=lambda r: (r["adverse_from_cost"], r["event_id"]),
    )
    losing_exits = sorted(
        (r for r in short_rows if r["action"] in {"reduce", "close"} and r["return_from_cost"] is not None),
        key=lambda r: (r["return_from_cost"], r["event_id"]),
    )
    counterexamples = {
        "short_adds_at_or_better_than_weighted_cost": [
            r for r in add_exceptions if r["adverse_from_cost"] <= 0
        ][:10],
        "short_adds_most_favorable_vs_weighted_cost": add_exceptions[:5],
        "short_adds_most_adverse_vs_weighted_cost": add_exceptions[-5:][::-1],
        "short_exits_with_negative_price_return_from_cost": [
            r for r in losing_exits if r["return_from_cost"] < 0
        ][:10],
        "long_cycles_as_separate_side_examples": long_cycles[:10],
    }

    split_event_counts = {}
    for name, lo, hi in [
        ("market_decision_proxy", "decision_proxy_time_ms", "MARKET"),
        ("limit_creation", "order_time_ms", "LIMIT"),
        ("limit_fill", "event_time_ms", "LIMIT"),
    ]:
        rows = [r for r in reconstructed if r["order_type"] == hi and START_MS <= r[lo] < END_MS]
        split_event_counts[name] = {
            "event_counts": dict(Counter(r["action"] for r in rows)),
            "n": len(rows),
            "timestamp_field": lo,
        }

    span_start = START_MS
    span_end = END_MS
    results = {
        "version": "reverse-mechanisms-1",
        "status": "exploratory_historical_diagnostics",
        "scope": {
            "start_utc_inclusive": _utc(START_MS),
            "end_utc_exclusive": _utc(END_MS),
            "primary_direction": "SHORT",
            "all_history_exploratory": True,
            "included_complete_history_anchored_cycles": len(cycle_metrics),
            "included_short_cycles": len(short_cycles),
            "included_long_cycles_separate": len(long_cycles),
            "crossing_or_incomplete_cycles_excluded_from_complete_mechanics": sum(
                c["status"] != "complete" or c["start_ms"] < START_MS or c["end_ms"] is None or c["end_ms"] >= END_MS
                for c in cycles
            ),
            "excluded_cycle_counts": {
                "history_only_outside_order_window": sum(c["status"] == "history_only_outside_order_window" for c in cycles),
                "right_truncated": sum(c["status"] == "right_truncated" for c in cycles),
                "complete_but_crossing_analysis_boundary": sum(
                    c["status"] == "complete" and
                    (c["start_ms"] < START_MS or c["end_ms"] is None or c["end_ms"] >= END_MS)
                    for c in cycles
                ),
            },
        },
        "event_reconstruction": {
            "method": "replay signed event quantities/prices in timestamp order; average cost updates on adds and stays constant through partial exits",
            "stored_avg_cost_before_policy": "nullable source values are validation checks only; reconstructed sequence is authoritative for exit placement",
            "position_transition_assertions": "all included position_before/after transitions matched reconstructed quantities",
            "price_flow_pnl_assertions": "all included completed cycles reconciled to reconstructed exit cashflows",
            "action_order_type_counts": {
                f"{action}:{order_type}": count for (action, order_type), count in sorted(action_order.items())
            },
            "order_to_update_delay_minutes_by_type": delays,
            "market_limit_creation_fill_separated": split_event_counts,
            "action_by_order_type": by_action_and_type,
            "reconstructed_events": reconstructed,
            "cycles": cycle_metrics,
        },
        "short_mechanics": {
            "cycle_count": len(short_cycles),
            "cycle_initial_notional": _summary(c["initial_notional"] for c in short_cycles),
            "cycle_initial_quantity": _summary(c["initial_qty"] for c in short_cycles),
            "adds_per_cycle": _summary(c["adds"] for c in short_cycles),
            "cycle_total_add_notional_multiple_of_first": _summary(
                c["total_add_notional_multiple_of_first"] for c in short_cycles
                if c["total_add_notional_multiple_of_first"] is not None
            ),
            "max_position_multiple_of_first": _summary(c["max_position_multiple_of_first"] for c in short_cycles),
            "event_mechanics": _metric_rows(short_rows),
            "fold_diagnostics": fold_results,
            "fold_gate_status": "event-only placement distributions; event-versus-natural-minute mechanism associations are pending the market panel",
            "largest_symbol_by_short_cycle_count": top_short_symbols[0] if top_short_symbols else None,
            "largest_symbol_top_10_cycle_counts": top_short_symbols,
            "remove_largest_symbol_sensitivity": {
                "removed_symbol": top_symbol,
                "remaining_cycles": len({r["cycle_id"] for r in no_top}),
                "remaining_event_mechanics": _metric_rows(no_top),
                "interpretation_limit": "event-only sensitivity describes whether size/spacing/exit distributions change; mechanism association versus non-event minutes requires panel controls",
            },
            "daily_block_bootstrap": _bootstrap(short_rows, span_start, span_end, block_days=1),
            "three_day_block_sensitivity": _bootstrap(short_rows, span_start, span_end, block_days=3),
            "complete_cycle_bootstrap_sensitivity": _cycle_bootstrap(short_rows),
        },
        "long_mechanics_separate": {
            "cycle_count": len(long_cycles),
            "event_mechanics": _metric_rows(long_rows),
            "cycle_initial_notional": _summary(c["initial_notional"] for c in long_cycles),
        },
        "counterexamples": counterexamples,
        "interpretation": [
            "This report describes observed position actions and is exploratory; it does not identify hidden intent or establish a prospective rule.",
            "All 33 event symbols were selected retrospectively from observed leader trades; event-panel findings describe conditional timing, not full-market coin selection.",
            "LIMIT creation timestamps and successful fill timestamps are separate outcomes. The collected successful-order stream does not include unfilled or canceled orders.",
            "Price return from reconstructed weighted cost describes placement at that exit; it is distinct from account-level PnL and excludes fees/funding.",
        ],
    }
    return results


def main() -> None:
    with localcontext() as ctx:
        ctx.prec = 40
        result = analyze()
    path = RESULTS / "reverse_mechanisms.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(path),
        "scope": result["scope"],
        "short_mechanics": {
            "cycle_count": result["short_mechanics"]["cycle_count"],
            "adds": result["short_mechanics"]["event_mechanics"]["event_counts"]["add"],
            "add_adverse_share": result["short_mechanics"]["event_mechanics"]["add_adverse_vs_weighted_cost_share"],
            "exits": result["short_mechanics"]["event_mechanics"]["event_counts"]["close"]
                     + result["short_mechanics"]["event_mechanics"]["event_counts"]["reduce"],
            "exit_profitable_share": result["short_mechanics"]["event_mechanics"]["profitable_exit_share_by_price_vs_cost"],
        },
        "largest_symbol": result["short_mechanics"]["largest_symbol_by_short_cycle_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
