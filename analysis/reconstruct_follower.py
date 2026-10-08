#!/usr/bin/env python3
"""Reconstruct the follower account's one-way position cycles from fills/orders."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/binance/2026-10-02_2026大黑马"
OUT = ROOT / "analysis/results"
BJ = ZoneInfo("Asia/Shanghai")
ZERO = Decimal("0")


def dec(value: Any) -> Decimal:
    return Decimal(str(value))


def dstr(value: Decimal | None) -> str | None:
    if value is None:
        return None
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in {"-0", ""} else rendered


def bj_time(time_ms: int | None) -> str | None:
    if time_ms is None:
        return None
    return datetime.fromtimestamp(time_ms / 1000, timezone.utc).astimezone(BJ).isoformat(
        timespec="milliseconds"
    )


def load_rows(name: str) -> list[dict[str, Any]]:
    rows = json.loads((DATA / name).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"{name} must contain a JSON array")
    return rows


def exact_order_fill_matches(
    orders: list[dict[str, Any]], trades: list[dict[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Match a full order to its fills using time, quantity, and executed quote."""
    fills_by_side: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for fill in trades:
        fills_by_side[(fill["symbol"], fill["side"])].append(fill)

    matched: dict[str, list[dict[str, Any]]] = {}
    used_fill_ids: set[str] = set()
    time_deltas: list[int] = []
    multi_fill_orders = 0
    fill_count_in_multi_orders = 0
    for order in orders:
        order_id = str(order["orderId"])
        candidates = [
            fill
            for fill in fills_by_side[(order["symbol"], order["side"])]
            if abs(int(fill["time"]) - int(order["updateTime"])) <= 1000
        ]
        target_qty = dec(order["executedQty"])
        target_quote = dec(order["executedQuoteQty"])
        solutions: list[tuple[dict[str, Any], ...]] = []

        def search(
            start: int,
            chosen: tuple[dict[str, Any], ...],
            quantity: Decimal,
            quote: Decimal,
        ) -> None:
            if quantity == target_qty:
                if quote == target_quote:
                    solutions.append(chosen)
                return
            if quantity > target_qty:
                return
            for index in range(start, len(candidates)):
                fill = candidates[index]
                fill_qty = dec(fill["qty"])
                if quantity + fill_qty <= target_qty:
                    search(
                        index + 1,
                        chosen + (fill,),
                        quantity + fill_qty,
                        quote + fill_qty * dec(fill["price"]),
                    )

        search(0, (), ZERO, ZERO)
        if len(solutions) != 1:
            raise AssertionError(
                f"order {order_id} has {len(solutions)} exact fill matches "
                f"within ±1 second ({order['symbol']} {order['side']})"
            )
        fills = list(solutions[0])
        fill_ids = [str(fill["record_id"]) for fill in fills]
        if any(fill_id in used_fill_ids for fill_id in fill_ids):
            raise AssertionError(f"a fill was assigned to more than one order: {fill_ids}")
        used_fill_ids.update(fill_ids)
        matched[order_id] = fills
        if len(fills) > 1:
            multi_fill_orders += 1
            fill_count_in_multi_orders += len(fills)
        time_deltas.extend(abs(int(fill["time"]) - int(order["updateTime"])) for fill in fills)

    all_fill_ids = {str(fill["record_id"]) for fill in trades}
    if used_fill_ids != all_fill_ids:
        raise AssertionError(
            f"order matching left {len(all_fill_ids - used_fill_ids)} fills unmatched "
            f"and reused {len(used_fill_ids - all_fill_ids)} unknown fills"
        )
    histogram = Counter(time_deltas)
    return matched, {
        "orders_matched": len(matched),
        "fills_matched": len(used_fill_ids),
        "orders_with_multiple_fills": multi_fill_orders,
        "fills_in_multiple_fill_orders": fill_count_in_multi_orders,
        "fill_order_time_delta_ms": {str(key): histogram[key] for key in sorted(histogram)},
        "max_abs_fill_order_time_delta_ms": max(time_deltas, default=0),
        "matching_rule": "same symbol and side; fill time within ±1000 ms of updateTime; exact executedQty and executedQuoteQty",
    }


def new_cycle(
    symbol: str, sequence: int, direction: str, start_ms: int, order_id: str
) -> dict[str, Any]:
    return {
        "cycle_id": f"{symbol}#{sequence:04d}",
        "symbol": symbol,
        "direction": direction,
        "start_ms": start_ms,
        "start_bj": bj_time(start_ms),
        "end_ms": None,
        "end_bj": None,
        "status": "open",
        "add_count": 0,
        "reduce_count": 0,
        "close_count": 0,
        "opening_order_count": 1,
        "entry_qty": ZERO,
        "exit_qty": ZERO,
        "remaining_qty": ZERO,
        "signed_remaining_position": ZERO,
        "max_abs_position": ZERO,
        "realized_pnl": ZERO,
        "fees": ZERO,
        "funding": ZERO,
        "net_pnl_before_sharing": ZERO,
        "source_order_ids": set(),
        "source_trade_ids": set(),
        "funding_ids": set(),
        "ambiguous": False,
        "ambiguity_reasons": set(),
    }


def finalize_cycle(cycle: dict[str, Any]) -> dict[str, Any]:
    result = dict(cycle)
    result["entry_qty"] = dstr(cycle["entry_qty"])
    result["exit_qty"] = dstr(cycle["exit_qty"])
    result["remaining_qty"] = dstr(cycle["remaining_qty"])
    result["signed_remaining_position"] = dstr(cycle["signed_remaining_position"])
    result["max_abs_position"] = dstr(cycle["max_abs_position"])
    for key in ("realized_pnl", "fees", "funding", "net_pnl_before_sharing"):
        result[key] = dstr(cycle[key])
    for key in ("source_order_ids", "source_trade_ids", "funding_ids", "ambiguity_reasons"):
        result[key] = sorted(cycle[key])
    result["duration_ms"] = (
        cycle["end_ms"] - cycle["start_ms"] if cycle["end_ms"] is not None else None
    )
    return result


def main() -> None:
    trades = load_rows("trades.json")
    orders = load_rows("orders.json")
    closed_positions = load_rows("closed_positions.json")
    open_positions = load_rows("open_positions.json")
    funding_rows = load_rows("funding.json")
    sharing_rows = load_rows("profit_sharing.json")
    audit = json.loads((DATA / "audit.json").read_text(encoding="utf-8"))

    for fill in trades:
        if fill["feeAsset"] != "USDT" or fill["realizedProfitAsset"] != "USDT":
            raise AssertionError(f"non-USDT fee or realized PnL on {fill['record_id']}")
    if any(row.get("asset") != "USDT" for row in funding_rows):
        raise AssertionError("funding contains a non-USDT asset")

    matched, matching_summary = exact_order_fill_matches(orders, trades)
    order_by_id = {str(order["orderId"]): order for order in orders}
    order_events: list[dict[str, Any]] = []
    position_by_symbol: dict[str, Decimal] = defaultdict(lambda: ZERO)
    active_cycle_by_symbol: dict[str, dict[str, Any] | None] = defaultdict(lambda: None)
    sequence_by_symbol: Counter[str] = Counter()
    cycles: list[dict[str, Any]] = []
    timeline: dict[str, list[tuple[int, Decimal, str | None]]] = defaultdict(list)
    action_counts: Counter[str] = Counter()
    first_action_evidence: dict[str, dict[str, Any]] = {}

    order_event_inputs = []
    for order_id, fills in matched.items():
        order = order_by_id[order_id]
        fills = sorted(fills, key=lambda row: (int(row["time"]), str(row["record_id"])))
        fill_times = [int(fill["time"]) for fill in fills]
        order_event_inputs.append(
            {
                "order": order,
                "fills": fills,
                "time_ms": min(fill_times),
                "time_ms_last_fill": max(fill_times),
                "qty": dec(order["executedQty"]),
                "quote": dec(order["executedQuoteQty"]),
                "pnl": sum((dec(fill["realizedProfit"]) for fill in fills), ZERO),
                "fee": sum((dec(fill["fee"]) for fill in fills), ZERO),
            }
        )
    order_event_inputs.sort(
        key=lambda item: (
            item["time_ms"],
            item["order"]["symbol"],
            int(item["order"]["orderId"]),
        )
    )

    same_time_counts = Counter(
        (item["order"]["symbol"], item["time_ms"]) for item in order_event_inputs
    )
    for item in order_event_inputs:
        order = item["order"]
        fills = item["fills"]
        symbol = order["symbol"]
        side = order["side"]
        order_id = str(order["orderId"])
        fill_ids = [str(fill["record_id"]) for fill in fills]
        time_ms = item["time_ms"]
        qty = item["qty"]
        pnl = item["pnl"]
        fee = item["fee"]
        signed_delta = qty if side == "BUY" else -qty
        position_before = position_by_symbol[symbol]
        reduce_only = bool(order["reduceOnly"])
        basis = "reduceOnly" if reduce_only else "one_way_net_position"
        ambiguous_same_time = same_time_counts[(symbol, time_ms)] > 1

        def attach_source(cycle: dict[str, Any]) -> None:
            cycle["source_order_ids"].add(order_id)
            cycle["source_trade_ids"].update(fill_ids)
            if ambiguous_same_time:
                cycle["ambiguous"] = True
                cycle["ambiguity_reasons"].add("multiple_orders_same_symbol_timestamp")

        def record_event(
            action: str,
            cycle: dict[str, Any],
            before: Decimal,
            after: Decimal,
            event_qty: Decimal,
            event_fee: Decimal,
            event_pnl: Decimal,
        ) -> None:
            attach_source(cycle)
            cycle["realized_pnl"] += event_pnl
            cycle["fees"] += event_fee
            cycle["max_abs_position"] = max(cycle["max_abs_position"], abs(before), abs(after))
            event = {
                "cycle_id": cycle["cycle_id"],
                "symbol": symbol,
                "time_ms": time_ms,
                "time_bj": bj_time(time_ms),
                "order_time_ms": int(order["updateTime"]),
                "order_time_bj": bj_time(int(order["updateTime"])),
                "last_fill_time_ms": item["time_ms_last_fill"],
                "action": action,
                "price": str(order["avgPrice"]),
                "qty": dstr(event_qty),
                "position_before": dstr(before),
                "position_after": dstr(after),
                "signed_qty_delta": dstr(after - before),
                "fee": dstr(event_fee),
                "realized_pnl": dstr(event_pnl),
                "reduce_only": reduce_only,
                "classification_basis": basis,
                "source_order_ids": [order_id],
                "source_trade_ids": fill_ids,
                "source_funding_ids": [],
                "ambiguous": ambiguous_same_time,
                "ambiguity_reason": (
                    "multiple_orders_same_symbol_timestamp" if ambiguous_same_time else None
                ),
            }
            order_events.append(event)
            action_counts[action] += 1

        if symbol not in first_action_evidence:
            first_action_evidence[symbol] = {
                "first_time_ms": time_ms,
                "first_order_id": order_id,
                "first_action_side": side,
                "first_order_reduce_only": reduce_only,
                "first_order_realized_pnl": dstr(pnl),
                "starts_from_zero_as_open": position_before == ZERO and not reduce_only,
            }

        if reduce_only:
            if position_before == ZERO or position_before * signed_delta >= ZERO:
                raise AssertionError(
                    f"reduceOnly order {order_id} does not oppose a live position"
                )
            if qty > abs(position_before):
                raise AssertionError(f"reduceOnly order {order_id} exceeds the live position")
            position_after = position_before + signed_delta
            cycle = active_cycle_by_symbol[symbol]
            if cycle is None:
                raise AssertionError(f"no active cycle for reduceOnly order {order_id}")
            if position_after == ZERO:
                action = "close"
                cycle["close_count"] += 1
                cycle["exit_qty"] += qty
                cycle["end_ms"] = time_ms
                cycle["end_bj"] = bj_time(time_ms)
                cycle["status"] = "closed"
                cycle["remaining_qty"] = ZERO
                cycle["signed_remaining_position"] = ZERO
                record_event(action, cycle, position_before, position_after, qty, fee, pnl)
                active_cycle_by_symbol[symbol] = None
            else:
                action = "reduce"
                cycle["reduce_count"] += 1
                cycle["exit_qty"] += qty
                cycle["remaining_qty"] = abs(position_after)
                cycle["signed_remaining_position"] = position_after
                record_event(action, cycle, position_before, position_after, qty, fee, pnl)
            position_by_symbol[symbol] = position_after
        elif position_before == ZERO:
            direction = "long" if signed_delta > ZERO else "short"
            sequence_by_symbol[symbol] += 1
            cycle = new_cycle(symbol, sequence_by_symbol[symbol], direction, time_ms, order_id)
            cycle["entry_qty"] += qty
            cycle["remaining_qty"] = qty
            cycle["signed_remaining_position"] = signed_delta
            cycle["max_abs_position"] = qty
            cycles.append(cycle)
            active_cycle_by_symbol[symbol] = cycle
            position_by_symbol[symbol] = signed_delta
            record_event("open", cycle, ZERO, signed_delta, qty, fee, pnl)
        elif position_before * signed_delta > ZERO:
            cycle = active_cycle_by_symbol[symbol]
            if cycle is None:
                raise AssertionError(f"no active cycle for add order {order_id}")
            position_after = position_before + signed_delta
            cycle["add_count"] += 1
            cycle["entry_qty"] += qty
            cycle["remaining_qty"] = abs(position_after)
            cycle["signed_remaining_position"] = position_after
            position_by_symbol[symbol] = position_after
            record_event("add", cycle, position_before, position_after, qty, fee, pnl)
        elif qty < abs(position_before):
            cycle = active_cycle_by_symbol[symbol]
            if cycle is None:
                raise AssertionError(f"no active cycle for reduce order {order_id}")
            position_after = position_before + signed_delta
            cycle["reduce_count"] += 1
            cycle["exit_qty"] += qty
            cycle["remaining_qty"] = abs(position_after)
            cycle["signed_remaining_position"] = position_after
            position_by_symbol[symbol] = position_after
            record_event("reduce", cycle, position_before, position_after, qty, fee, pnl)
        elif qty == abs(position_before):
            cycle = active_cycle_by_symbol[symbol]
            if cycle is None:
                raise AssertionError(f"no active cycle for close order {order_id}")
            cycle["close_count"] += 1
            cycle["exit_qty"] += qty
            cycle["end_ms"] = time_ms
            cycle["end_bj"] = bj_time(time_ms)
            cycle["status"] = "closed"
            cycle["remaining_qty"] = ZERO
            cycle["signed_remaining_position"] = ZERO
            position_by_symbol[symbol] = ZERO
            record_event("close", cycle, position_before, ZERO, qty, fee, pnl)
            active_cycle_by_symbol[symbol] = None
        else:
            raise AssertionError(f"Unexpected position reversal in audited dataset: {order_id}")

        timeline[symbol].append(
            (time_ms, position_by_symbol[symbol], active_cycle_by_symbol[symbol]["cycle_id"] if active_cycle_by_symbol[symbol] else None)
        )

    trade_totals = {
        "realized_pnl": sum((dec(row["realizedProfit"]) for row in trades), ZERO),
        "fees": sum((dec(row["fee"]) for row in trades), ZERO),
    }
    for cycle in cycles:
        if cycle["status"] == "open":
            cycle["end_ms"] = None
            cycle["end_bj"] = None
        cycle["net_pnl_before_sharing"] = cycle["realized_pnl"] + cycle["fees"] + cycle["funding"]

    # Funding is assigned only when one cycle is unambiguously live strictly between its boundaries.
    funding_events: list[dict[str, Any]] = []
    unmatched_funding: list[dict[str, Any]] = []
    funding_match_count = 0
    for row in funding_rows:
        symbol = row["symbol"]
        time_ms = int(row["time"])
        candidates = [
            cycle
            for cycle in cycles
            if cycle["symbol"] == symbol
            and cycle["start_ms"] < time_ms
            and (cycle["end_ms"] is None or time_ms < cycle["end_ms"])
        ]
        boundary_cycles = [
            cycle
            for cycle in cycles
            if cycle["symbol"] == symbol
            and time_ms in {cycle["start_ms"], cycle["end_ms"]}
        ]
        reason = None
        if len(candidates) == 1 and not boundary_cycles:
            cycle = candidates[0]
            amount = dec(row["balanceDeltaStr"])
            cycle["funding"] += amount
            cycle["net_pnl_before_sharing"] += amount
            cycle["funding_ids"].add(str(row["id"]))
            funding_match_count += 1
            exact_trade_time = any(
                event["symbol"] == symbol and event["time_ms"] == time_ms
                for event in order_events
            )
            position_at_time = None
            if not exact_trade_time:
                previous_steps = [step for step in timeline[symbol] if step[0] < time_ms]
                position_at_time = previous_steps[-1][1] if previous_steps else None
            funding_events.append(
                {
                    "cycle_id": cycle["cycle_id"],
                    "symbol": symbol,
                    "time_ms": time_ms,
                    "time_bj": bj_time(time_ms),
                    "action": "funding",
                    "price": None,
                    "qty": "0",
                    "position_before": dstr(position_at_time),
                    "position_after": dstr(position_at_time),
                    "signed_qty_delta": "0",
                    "amount": dstr(amount),
                    "fee": "0",
                    "realized_pnl": "0",
                    "classification_basis": "unique_symbol_cycle_interval",
                    "source_order_ids": [],
                    "source_trade_ids": [],
                    "source_funding_ids": [str(row["id"])],
                    "ambiguous": exact_trade_time,
                    "ambiguity_reason": (
                        "funding_timestamp_matches_order_timestamp_position_size_uncertain"
                        if exact_trade_time
                        else None
                    ),
                }
            )
        elif boundary_cycles:
            reason = "funding_on_cycle_start_or_end_boundary"
        elif len(candidates) > 1:
            reason = "multiple_live_cycles_for_symbol_time"
        else:
            reason = "no_live_cycle_for_symbol_time"
        if reason is not None:
            unmatched_funding.append(
                {
                    "funding_id": str(row["id"]),
                    "symbol": symbol,
                    "time_ms": time_ms,
                    "time_bj": bj_time(time_ms),
                    "amount": str(row["balanceDeltaStr"]),
                    "reason": reason,
                    "candidate_cycle_ids": [cycle["cycle_id"] for cycle in candidates + boundary_cycles],
                }
            )
            funding_events.append(
                {
                    "cycle_id": None,
                    "symbol": symbol,
                    "time_ms": time_ms,
                    "time_bj": bj_time(time_ms),
                    "action": "funding_unmatched",
                    "price": None,
                    "qty": "0",
                    "position_before": None,
                    "position_after": None,
                    "signed_qty_delta": "0",
                    "amount": str(row["balanceDeltaStr"]),
                    "fee": "0",
                    "realized_pnl": "0",
                    "classification_basis": "symbol_and_time_only",
                    "source_order_ids": [],
                    "source_trade_ids": [],
                    "source_funding_ids": [str(row["id"])],
                    "ambiguous": True,
                    "ambiguity_reason": reason,
                }
            )

    account_realized = trade_totals["realized_pnl"]
    account_fees = trade_totals["fees"]
    account_funding = sum((dec(row["balanceDeltaStr"]) for row in funding_rows), ZERO)
    sharing_paid = sum((dec(row["sharedAmount"]) for row in sharing_rows), ZERO)
    net_before_sharing = account_realized + account_fees + account_funding
    net_after_sharing = net_before_sharing - sharing_paid

    # Closed-position history is an independent cross-check on cycle endpoints.
    unused_closed = {str(row["id"]): row for row in closed_positions}
    closed_crosschecks: list[dict[str, Any]] = []
    for cycle in cycles:
        if cycle["status"] != "closed":
            continue
        candidates = []
        for record_id, row in unused_closed.items():
            if row["symbol"] != cycle["symbol"] or str(row["side"]).lower() != cycle["direction"]:
                continue
            if abs(int(row["opened"]) - cycle["start_ms"]) <= 2000 and abs(
                int(row["closed"]) - cycle["end_ms"]
            ) <= 2000:
                candidates.append((record_id, row))
        if len(candidates) == 1:
            record_id, row = candidates[0]
            del unused_closed[record_id]
            cycle["closed_position_id"] = record_id
            cycle["closed_position_closing_pnl"] = str(row["closingPnl"])
            cycle["closed_position_closed_volume"] = str(row["closedVolume"])
            gross_difference = cycle["realized_pnl"] - dec(row["closingPnl"])
            net_difference = (
                cycle["realized_pnl"] + cycle["fees"] + cycle["funding"]
                - dec(row["closingPnl"])
            )
            closed_crosschecks.append(
                {
                    "cycle_id": cycle["cycle_id"],
                    "closed_position_id": record_id,
                    "open_time_delta_ms": cycle["start_ms"] - int(row["opened"]),
                    "close_time_delta_ms": cycle["end_ms"] - int(row["closed"]),
                    "cycle_exit_qty": dstr(cycle["exit_qty"]),
                    "closed_volume": str(row["closedVolume"]),
                    "gross_realized_pnl_minus_closingPnl": dstr(gross_difference),
                    "net_pnl_before_sharing_minus_closingPnl": dstr(net_difference),
                }
            )
        else:
            closed_crosschecks.append(
                {
                    "cycle_id": cycle["cycle_id"],
                    "matched": False,
                    "candidate_closed_position_ids": [record_id for record_id, _ in candidates],
                }
            )

    # Final position snapshot must agree with the replayed signed positions.
    snapshot_by_symbol = {
        row["symbol"]: dec(row["positionAmount"]) for row in open_positions
    }
    all_symbols = set(position_by_symbol) | set(snapshot_by_symbol)
    final_position_differences = {
        symbol: {
            "replayed": dstr(position_by_symbol[symbol]),
            "snapshot": dstr(snapshot_by_symbol.get(symbol, ZERO)),
        }
        for symbol in sorted(all_symbols)
        if position_by_symbol[symbol] != snapshot_by_symbol.get(symbol, ZERO)
    }
    if final_position_differences:
        raise AssertionError(f"final positions differ from open_positions: {final_position_differences}")

    if account_realized != dec(audit["totals_usdt"]["realized_trade_pnl"]):
        raise AssertionError("trade realized PnL differs from audit")
    if account_fees != dec(audit["totals_usdt"]["trading_fees"]):
        raise AssertionError("trade fees differ from audit")
    if account_funding != dec(audit["totals_usdt"]["funding"]):
        raise AssertionError("funding total differs from audit")
    if net_before_sharing != dec(audit["totals_usdt"]["realized_net_before_sharing"]):
        raise AssertionError("pre-sharing net PnL differs from audit")
    if net_after_sharing != dec(audit["totals_usdt"]["realized_net_after_sharing"]):
        raise AssertionError("post-sharing net PnL differs from audit")

    cycles.sort(key=lambda row: (row["start_ms"], row["symbol"], row["cycle_id"]))
    cycle_results = [finalize_cycle(cycle) for cycle in cycles]
    funding_events.sort(key=lambda row: (row["time_ms"], row["symbol"], row["action"]))
    events = order_events + funding_events
    events.sort(
        key=lambda row: (
            row["time_ms"],
            row["symbol"],
            0 if row["action"] not in {"funding", "funding_unmatched"} else 1,
            row["source_order_ids"][0] if row["source_order_ids"] else "",
        )
    )

    first_fill_by_symbol: dict[str, dict[str, Any]] = {}
    for row in sorted(trades, key=lambda row: (int(row["time"]), row["symbol"])):
        first_fill_by_symbol.setdefault(row["symbol"], row)
    first_fill_realized_zero = sum(
        1 for row in first_fill_by_symbol.values() if dec(row["realizedProfit"]) == ZERO
    )
    first_order_non_reduce = sum(
        1 for row in first_action_evidence.values() if not row["first_order_reduce_only"]
    )
    completed_cycles = [cycle for cycle in cycles if cycle["status"] == "closed"]
    open_cycles = [cycle for cycle in cycles if cycle["status"] == "open"]
    closed_pnl_total = sum((cycle["realized_pnl"] for cycle in completed_cycles), ZERO)
    closed_fees_total = sum((cycle["fees"] for cycle in completed_cycles), ZERO)
    closed_funding_total = sum((cycle["funding"] for cycle in completed_cycles), ZERO)
    closed_net_total = sum(
        (cycle["realized_pnl"] + cycle["fees"] + cycle["funding"] for cycle in completed_cycles),
        ZERO,
    )
    open_realized_total = sum((cycle["realized_pnl"] for cycle in open_cycles), ZERO)
    open_fees_total = sum((cycle["fees"] for cycle in open_cycles), ZERO)
    open_funding_total = sum((cycle["funding"] for cycle in open_cycles), ZERO)
    open_net_total = open_realized_total + open_fees_total + open_funding_total
    closed_position_pnl_total = sum((dec(row["closingPnl"]) for row in closed_positions), ZERO)
    closed_position_matches = sum(1 for row in closed_crosschecks if row.get("closed_position_id"))
    closed_net_pnl_mismatches = sum(
        1
        for row in closed_crosschecks
        if row.get("net_pnl_before_sharing_minus_closingPnl") not in (None, "0")
    )
    closed_volume_mismatches = sum(
        1
        for row in closed_crosschecks
        if row.get("cycle_exit_qty") is not None
        and dec(row["cycle_exit_qty"]) != dec(row["closed_volume"])
    )

    summary = {
        "dataset": str(DATA.relative_to(ROOT)),
        "method": {
            "position_mode": "BOTH one-way net position",
            "initial_position": "zero; supported by each symbol's first fill having zero realized PnL, first order not reduceOnly, and exact final-snapshot replay",
            "fill_to_order_matching": matching_summary,
            "split_fills_are_grouped_by_order": True,
            "same_symbol_same_fill_time_order_clusters": sum(
                1 for count in same_time_counts.values() if count > 1
            ),
            "same_symbol_same_second_fill_groups": sum(
                1
                for count in Counter((row["symbol"], int(row["time"])) for row in trades).values()
                if count > 1
            ),
            "unmatched_or_ambiguous_order_associations": 0,
        },
        "counts": {
            "trade_fills": len(trades),
            "orders": len(orders),
            "order_grouped_trade_events": len(order_events),
            "closed_cycles": len(completed_cycles),
            "open_cycles": len(open_cycles),
            "closed_position_records": len(closed_positions),
            "closed_position_records_matched": closed_position_matches,
            "closed_position_net_pnl_mismatches": closed_net_pnl_mismatches,
            "closed_position_closed_volume_mismatches": closed_volume_mismatches,
            "unmatched_closed_position_records": len(unused_closed),
            "funding_rows": len(funding_rows),
            "funding_rows_matched_to_cycles": funding_match_count,
            "funding_rows_unmatched": len(unmatched_funding),
            "ambiguous_same_time_order_events": sum(
                1 for row in order_events if row["ambiguous"]
            ),
        },
        "order_action_counts": {key: action_counts[key] for key in sorted(action_counts)},
        "cycle_direction_counts": {
            direction: sum(1 for cycle in cycles if cycle["direction"] == direction)
            for direction in ("long", "short")
        },
        "account_totals_usdt": {
            "realized_trade_pnl": dstr(account_realized),
            "trading_fees": dstr(account_fees),
            "funding": dstr(account_funding),
            "net_pnl_before_sharing": dstr(net_before_sharing),
            "profit_sharing_paid_account_level": dstr(sharing_paid),
            "net_pnl_after_sharing": dstr(net_after_sharing),
            "open_cycle_realized_pnl": dstr(open_realized_total),
            "open_cycle_fees": dstr(open_fees_total),
            "open_cycle_funding": dstr(open_funding_total),
            "open_cycle_net_pnl_before_sharing": dstr(open_net_total),
            "closed_cycle_realized_pnl": dstr(closed_pnl_total),
            "closed_cycle_fees": dstr(closed_fees_total),
            "closed_cycle_funding": dstr(closed_funding_total),
            "closed_cycle_fees_plus_funding": dstr(closed_fees_total + closed_funding_total),
            "closed_cycle_net_pnl_before_sharing": dstr(closed_net_total),
            "closed_positions_closing_pnl_sum": dstr(closed_position_pnl_total),
            "closed_cycle_gross_realized_minus_closed_positions_net_pnl": dstr(
                closed_pnl_total - closed_position_pnl_total
            ),
            "closed_cycle_net_minus_closed_positions_pnl": dstr(
                closed_net_total - closed_position_pnl_total
            ),
            "audit_difference": {
                "realized_trade_pnl": dstr(
                    account_realized - dec(audit["totals_usdt"]["realized_trade_pnl"])
                ),
                "trading_fees": dstr(
                    account_fees - dec(audit["totals_usdt"]["trading_fees"])
                ),
                "funding": dstr(account_funding - dec(audit["totals_usdt"]["funding"])),
                "net_before_sharing": dstr(
                    net_before_sharing - dec(audit["totals_usdt"]["realized_net_before_sharing"])
                ),
                "net_after_sharing": dstr(
                    net_after_sharing - dec(audit["totals_usdt"]["realized_net_after_sharing"])
                ),
            },
            "closed_position_closing_pnl_sum_minus_audit_realized_trade_pnl": dstr(
                closed_position_pnl_total - account_realized
            ),
        },
        "checks": {
            "final_positions_match_open_positions": not final_position_differences,
            "final_position_differences": final_position_differences,
            "first_fill_realized_pnl_zero_symbols": first_fill_realized_zero,
            "first_fill_symbol_count": len(first_fill_by_symbol),
            "first_order_non_reduce_only_symbols": first_order_non_reduce,
            "first_order_symbol_count": len(first_action_evidence),
            "first_order_evidence_by_symbol": first_action_evidence,
            "closed_cycle_count_matches_closed_position_count": len(completed_cycles) == len(closed_positions),
            "closed_cycle_endpoint_matches": closed_position_matches == len(completed_cycles),
            "closed_cycle_net_pnl_matches_closed_positions": closed_net_total == closed_position_pnl_total,
            "closed_cycle_net_pnl_matches_each_closed_position": closed_net_pnl_mismatches == 0,
            "closed_cycle_exit_qty_matches_each_closed_volume": closed_volume_mismatches == 0,
            "funding_allocated_plus_unmatched_equals_source_count": funding_match_count + len(unmatched_funding) == len(funding_rows),
            "audit_totals_match": True,
        },
        "closed_position_crosschecks": closed_crosschecks,
        "unmatched_closed_position_ids": sorted(unused_closed),
        "unmatched_funding": unmatched_funding,
        "notes": [
            "A cycle is one continuous non-zero signed position for a symbol; each fully flat-to-flat interval is a separate cycle.",
            "An order is one action. Fills are grouped only after exact quantity and executed-quote reconciliation; 676 fills therefore do not become 676 action counts.",
            "Funding is allocated only to one cycle strictly open at that symbol and timestamp. Boundary or missing-interval records remain unassigned.",
            "Profit sharing remains an account-level expense and is not apportioned across cycles.",
            "Cycle net PnL includes realized PnL, fees, and matched funding; it excludes unrealized PnL and profit sharing.",
            "closed_positions.closingPnl is a net cycle figure: realized PnL plus fees plus funding. The gross realized-PnL comparison alone is not an equivalent basis.",
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "follower_cycles.json").write_text(
        json.dumps(cycle_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "follower_events.json").write_text(
        json.dumps(events, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "follower_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
