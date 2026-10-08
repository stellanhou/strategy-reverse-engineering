#!/usr/bin/env python3
"""Reconstruct observable leader position cycles from the saved Binance data."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from decimal import Decimal, localcontext
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "binance" / "2026-10-02_2026大黑马"
RESULTS_DIR = ROOT / "analysis" / "results"
ORDER_MATCH_TOLERANCE_MS = 2_000
D = Decimal
ZERO = D("0")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, parse_float=D, parse_int=D)


def dec(value: Any) -> Decimal | None:
    if value is None:
        return None
    return value if isinstance(value, Decimal) else D(str(value))


def required_dec(value: Any, field: str) -> Decimal:
    parsed = dec(value)
    if parsed is None:
        raise ValueError(f"Missing required decimal field: {field}")
    return parsed


def dec_string(value: Decimal | None) -> str | None:
    if value is None:
        return None
    if value == ZERO:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return dec_string(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def direction_sign(direction: str) -> int:
    if direction == "Long":
        return 1
    if direction == "Short":
        return -1
    raise ValueError(f"Unsupported history direction: {direction!r}")


def order_signed_qty(order: dict[str, Any]) -> Decimal:
    quantity = dec(order["executedQty"])
    if quantity is None or quantity <= ZERO:
        raise ValueError(f"Non-positive executedQty in order: {order}")
    if order["side"] == "BUY":
        return quantity
    if order["side"] == "SELL":
        return -quantity
    raise ValueError(f"Unsupported order side: {order['side']!r}")


def source_order_fields(order: dict[str, Any], source_index: int) -> dict[str, Any]:
    required_dec(order.get("executedQty"), f"order[{source_index}].executedQty")
    required_dec(order.get("avgPrice"), f"order[{source_index}].avgPrice")
    required_dec(order.get("totalPnl"), f"order[{source_index}].totalPnl")
    return {
        "event_id": f"leader-order-{source_index:04d}",
        "source_index": source_index,
        "symbol": order["symbol"],
        "side": order["side"],
        "position_side": order.get("positionSide"),
        "order_type": order.get("type"),
        "raw_type": order.get("type"),
        "order_id": None,
        "reduce_only": None,
        "time_ms": int(order["orderUpdateTime"]),
        "order_time_ms": int(order["orderTime"]),
        "timestamp_basis": "orderUpdateTime",
        "price": dec_string(dec(order.get("avgPrice"))),
        "qty": dec_string(dec(order.get("executedQty"))),
        "leader_reported_total_pnl": dec_string(dec(order.get("totalPnl"))),
    }


def history_cycle_base(history: dict[str, Any], cycle_id: str) -> dict[str, Any]:
    return {
        "cycle_id": cycle_id,
        "symbol": history["symbol"],
        "direction": history["side"].upper(),
        "start_ms": int(history["opened"]),
        "end_ms": int(history["closed"]),
        "status": "history_only_outside_order_window",
        "add_count": None,
        "reduce_count": None,
        "realized_pnl": dec_string(required_dec(history.get("closingPnl"), "history.closingPnl")),
        "realized_pnl_source": "leader_position_history_first.closingPnl",
        "price_flow_pnl": None,
        "order_total_pnl": None,
        "event_count": 0,
        "open_count": None,
        "close_count": None,
        "duration_ms": int(history["closed"]) - int(history["opened"]),
        "history_position_id": str(history["positionId"]),
        "history_status": history.get("status"),
        "history_avg_cost": dec_string(dec(history.get("avgCost"))),
        "history_avg_close_price": dec_string(dec(history.get("avgClosePrice"))),
        "history_max_open_interest": dec_string(dec(history.get("maxOpenInterest"))),
        "history_closed_volume": dec_string(dec(history.get("closedVolume"))),
        "start_evidence": "position history opened timestamp",
        "end_evidence": "position history closed timestamp; order actions predate the saved 30-day order window",
        "notes": "Complete closed position-history record; per-order operations are outside leader_orders.json coverage.",
    }


def assign_anchored_cycle(
    history: dict[str, Any], orders: list[tuple[int, dict[str, Any]]], cycle_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    direction = history["side"]
    sign = direction_sign(direction)
    direction_name = direction.upper()
    ordered = sorted(orders, key=lambda row: (int(row[1]["orderUpdateTime"]), row[0]))
    first_time = int(ordered[0][1]["orderUpdateTime"])
    last_time = int(ordered[-1][1]["orderUpdateTime"])
    opened = int(history["opened"])
    closed = int(history["closed"])
    if abs(first_time - opened) > ORDER_MATCH_TOLERANCE_MS:
        raise ValueError(f"First order is not anchored to open time: {history['symbol']} {cycle_id}")
    if abs(last_time - closed) > ORDER_MATCH_TOLERANCE_MS:
        raise ValueError(f"Last order is not anchored to close time: {history['symbol']} {cycle_id}")
    if history.get("status") != "All Closed":
        raise ValueError(f"Expected a closed history anchor: {cycle_id}")

    position = ZERO
    average_cost = ZERO
    add_count = 0
    reduce_count = 0
    open_count = 0
    close_count = 0
    flow_pnl = ZERO
    reported_order_pnl = ZERO
    events: list[dict[str, Any]] = []

    for source_index, order in ordered:
        base = source_order_fields(order, source_index)
        quantity = dec(order["executedQty"])
        price = dec(order["avgPrice"])
        if quantity is None or price is None:
            raise ValueError(f"Missing quantity or price: {cycle_id} {base['event_id']}")
        signed_qty = order_signed_qty(order)
        before = position
        after = before + signed_qty
        flow_pnl += quantity * price * (D(1) if order["side"] == "SELL" else D(-1))
        reported_order_pnl += required_dec(order.get("totalPnl"), "order.totalPnl")

        avg_cost_before: Decimal | None = None
        add_mtm_before: Decimal | None = None
        add_state: str | None = None
        realized_flow: Decimal | None = None
        if before == ZERO:
            if signed_qty * sign <= ZERO:
                raise ValueError(f"Opening order conflicts with history side: {cycle_id}")
            action = "open"
            open_count += 1
            average_cost = price
        elif signed_qty * before > ZERO:
            action = "add"
            add_count += 1
            avg_cost_before = average_cost
            add_mtm_before = (price - average_cost) * before
            add_state = "profit" if add_mtm_before > ZERO else "loss" if add_mtm_before < ZERO else "at_cost"
            average_cost = (abs(before) * average_cost + quantity * price) / (abs(before) + quantity)
        else:
            if after == ZERO:
                action = "close"
                close_count += 1
            elif after * before > ZERO:
                action = "reduce"
                reduce_count += 1
            else:
                raise ValueError(f"Order reverses a position inside a history cycle: {cycle_id}")
            closed_quantity = min(abs(before), quantity)
            realized_flow = (price - average_cost) * closed_quantity * (D(1) if before > ZERO else D(-1))
            if after == ZERO:
                average_cost = ZERO

        position = after
        base.update(
            {
                "cycle_id": cycle_id,
                "direction": direction_name,
                "action": action,
                "classification": "history_anchored",
                "price": dec_string(price),
                "qty": dec_string(quantity),
                "position_before": dec_string(before),
                "position_after": dec_string(after),
                "avg_cost_before": dec_string(avg_cost_before),
                "unrealized_pnl_before_add": dec_string(add_mtm_before),
                "add_pnl_state": add_state,
                "realized_pnl_cashflow": dec_string(realized_flow),
                "source_order_type": order.get("type"),
                "leader_reported_total_pnl": dec_string(dec(order.get("totalPnl"))),
                "source_history_position_id": str(history["positionId"]),
            }
        )
        events.append(base)

    if position != ZERO or close_count != 1:
        raise ValueError(f"Anchored sequence did not finish flat exactly once: {cycle_id}, {position}")
    if open_count != 1:
        raise ValueError(f"Expected one opening event in the anchored sequence: {cycle_id}")
    realized_by_events = ZERO
    for event in events:
        if event["action"] in {"reduce", "close"}:
            realized_by_events += required_dec(
                event["realized_pnl_cashflow"], "event.realized_pnl_cashflow"
            )
        elif event["realized_pnl_cashflow"] is not None:
            raise ValueError(f"Unexpected realized cash flow on {event['action']}: {cycle_id}")
    if abs(realized_by_events - flow_pnl) > D("0.000000000001"):
        raise ValueError(f"Weighted-cost realized PnL does not reconcile to price cash flow: {cycle_id}")

    cycle = {
        "cycle_id": cycle_id,
        "symbol": history["symbol"],
        "direction": direction_name,
        "start_ms": opened,
        "end_ms": closed,
        "status": "complete",
        "add_count": add_count,
        "reduce_count": reduce_count,
        "realized_pnl": dec_string(required_dec(history.get("closingPnl"), "history.closingPnl")),
        "realized_pnl_source": "leader_position_history_first.closingPnl",
        "price_flow_pnl": dec_string(flow_pnl),
        "order_total_pnl": dec_string(reported_order_pnl),
        "order_total_pnl_minus_price_flow": dec_string(reported_order_pnl - flow_pnl),
        "price_flow_minus_history_closing_pnl": dec_string(
            flow_pnl - required_dec(history.get("closingPnl"), "history.closingPnl")
        ),
        "event_count": len(events),
        "open_count": open_count,
        "close_count": close_count,
        "duration_ms": closed - opened,
        "history_position_id": str(history["positionId"]),
        "history_status": history.get("status"),
        "history_avg_cost": dec_string(dec(history.get("avgCost"))),
        "history_avg_close_price": dec_string(dec(history.get("avgClosePrice"))),
        "history_max_open_interest": dec_string(dec(history.get("maxOpenInterest"))),
        "history_closed_volume": dec_string(dec(history.get("closedVolume"))),
        "start_evidence": f"position history opened; first orderUpdateTime {first_time} is within {ORDER_MATCH_TOLERANCE_MS} ms",
        "end_evidence": f"position history closed; last orderUpdateTime {last_time} is within {ORDER_MATCH_TOLERANCE_MS} ms and reconstructed position returns to zero",
        "notes": "All saved order events for this anchored position cycle are classified. PnL basis is the source history closingPnl field.",
    }
    return cycle, events


def right_truncated_cycle(
    order: dict[str, Any], source_index: int, previous_history: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    time_ms = int(order["orderUpdateTime"])
    previous_closed = int(previous_history["closed"])
    if time_ms <= previous_closed:
        raise ValueError(f"Expected new order after the latest closed anchor: {order['symbol']}")
    if order["type"] != "MARKET" or dec(order.get("totalPnl")) != ZERO:
        raise ValueError(f"Unexpected unanchored event evidence: {order}")
    side_sign = 1 if order["side"] == "BUY" else -1
    direction = "LONG" if side_sign == 1 else "SHORT"
    cycle_id = f"local-{order['symbol']}-{time_ms}"
    base = source_order_fields(order, source_index)
    quantity = required_dec(order.get("executedQty"), "order.executedQty")
    price = required_dec(order.get("avgPrice"), "order.avgPrice")
    event = {
        **base,
        "cycle_id": cycle_id,
        "direction": direction,
        "action": "open",
        "classification": "inferred_open_after_latest_closed_anchor",
        "price": dec_string(price),
        "qty": dec_string(quantity),
        "position_before": "0",
        "position_after": dec_string(quantity * D(side_sign)),
        "avg_cost_before": None,
        "unrealized_pnl_before_add": None,
        "add_pnl_state": None,
        "realized_pnl_cashflow": None,
        "source_order_type": order.get("type"),
        "leader_reported_total_pnl": dec_string(dec(order.get("totalPnl"))),
        "source_history_position_id": str(previous_history["positionId"]),
    }
    cycle = {
        "cycle_id": cycle_id,
        "symbol": order["symbol"],
        "direction": direction,
        "start_ms": time_ms,
        "end_ms": None,
        "status": "right_truncated",
        "add_count": 0,
        "reduce_count": 0,
        "realized_pnl": None,
        "realized_pnl_source": None,
        "price_flow_pnl": None,
        "order_total_pnl": None,
        "event_count": 1,
        "open_count": 1,
        "close_count": 0,
        "duration_ms": None,
        "history_position_id": None,
        "history_status": None,
        "history_avg_cost": None,
        "history_avg_close_price": None,
        "history_max_open_interest": None,
        "history_closed_volume": None,
        "latest_closed_history_position_id": str(previous_history["positionId"]),
        "latest_closed_history_closed_ms": previous_closed,
        "start_evidence": "first observed same-symbol order after latest fully closed history anchor; no intervening same-symbol order is present",
        "end_evidence": "no later matching position-history close; right-censored at collection time",
        "notes": "Open classification is inferred from the immediately preceding closed anchor and saved order stream; cycle completion and realized PnL are unknown.",
    }
    return cycle, event


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)
        stream.write("\n")


def main() -> None:
    with localcontext() as context:
        context.prec = 50
        orders: list[dict[str, Any]] = read_json(DATA_DIR / "leader_orders.json")
        history_response: dict[str, Any] = read_json(DATA_DIR / "leader_position_history_first.json")
        history = history_response["data"]["list"]
        history_total = int(history_response["data"]["total"])
        if len(orders) != 454 or len(history) != 200 or history_total != 1196:
            raise ValueError(
                f"Unexpected audited source sizes: orders={len(orders)}, history={len(history)}, total={history_total}"
            )

        order_window_start = min(int(order["orderUpdateTime"]) for order in orders)
        order_window_end = max(int(order["orderUpdateTime"]) for order in orders)
        histories_by_id: dict[str, dict[str, Any]] = {}
        history_orders: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        unmatched: list[tuple[int, dict[str, Any]]] = []
        for history_record in history:
            position_id = str(history_record["positionId"])
            if position_id in histories_by_id:
                raise ValueError(f"Duplicate history positionId: {position_id}")
            histories_by_id[position_id] = history_record

        for source_index, order in enumerate(orders, start=1):
            event_time = int(order["orderUpdateTime"])
            matches = [
                record
                for record in history
                if record["symbol"] == order["symbol"]
                and int(record["opened"]) - ORDER_MATCH_TOLERANCE_MS
                <= event_time
                <= int(record["closed"]) + ORDER_MATCH_TOLERANCE_MS
            ]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous history anchor for order {source_index}: {order}")
            if matches:
                history_orders[str(matches[0]["positionId"])].append((source_index, order))
            else:
                unmatched.append((source_index, order))

        cycles: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        for record in history:
            position_id = str(record["positionId"])
            assigned = history_orders.get(position_id, [])
            if assigned:
                cycle, cycle_events = assign_anchored_cycle(
                    record, assigned, f"leader-position-{position_id}"
                )
                cycles.append(cycle)
                events.extend(cycle_events)
            else:
                if int(record["closed"]) >= order_window_start:
                    raise ValueError(
                        f"History cycle overlaps the saved order window but has no matching orders: {position_id}"
                    )
                cycles.append(history_cycle_base(record, f"leader-position-{position_id}"))

        unmatched_by_symbol: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for row in unmatched:
            unmatched_by_symbol[row[1]["symbol"]].append(row)
        expected_unmatched_symbols = {"BTWUSDT", "CTUSDT", "龙虾USDT"}
        if set(unmatched_by_symbol) != expected_unmatched_symbols or any(
            len(rows) != 1 for rows in unmatched_by_symbol.values()
        ):
            raise ValueError(f"Unexpected unmatched order rows: {unmatched_by_symbol}")

        for symbol in ("BTWUSDT", "龙虾USDT"):
            source_index, order = unmatched_by_symbol[symbol][0]
            prior = [
                record
                for record in history
                if record["symbol"] == symbol and int(record["closed"]) < int(order["orderUpdateTime"])
            ]
            if not prior:
                raise ValueError(f"Expected a latest closed history anchor for {symbol}")
            latest = max(prior, key=lambda record: int(record["closed"]))
            intervening = [
                earlier
                for earlier in orders
                if earlier["symbol"] == symbol
                and int(latest["closed"]) < int(earlier["orderUpdateTime"]) < int(order["orderUpdateTime"])
            ]
            if intervening:
                raise ValueError(f"Unexpected intervening order before inferred open: {symbol}")
            cycle, event = right_truncated_cycle(order, source_index, latest)
            cycles.append(cycle)
            events.append(event)

        source_index, unclassified = unmatched_by_symbol["CTUSDT"][0]
        if any(record["symbol"] == "CTUSDT" for record in history):
            raise ValueError("CTUSDT gained a position-history anchor; update the explicit edge-case review")
        if sum(order["symbol"] == "CTUSDT" for order in orders) != 1:
            raise ValueError("CTUSDT has more than one order; the current unclassified treatment no longer applies")
        ct_event = {
            **source_order_fields(unclassified, source_index),
            "cycle_id": None,
            "direction": "unknown",
            "action": "unknown",
            "classification": "unanchored_right_edge",
            "position_before": None,
            "position_after": None,
            "avg_cost_before": None,
            "unrealized_pnl_before_add": None,
            "add_pnl_state": None,
            "realized_pnl_cashflow": None,
            "source_order_type": unclassified.get("type"),
            "source_history_position_id": None,
            "notes": "No prior position-history anchor and only one observed 30-day order; open/add/reduce/close cannot be distinguished.",
        }
        events.append(ct_event)

        cycles.sort(key=lambda cycle: (cycle["start_ms"], cycle["symbol"], cycle["cycle_id"]))
        events.sort(key=lambda event: (event["time_ms"], event["event_id"]))
        if len(cycles) != 202 or len(events) != 454:
            raise ValueError(f"Unexpected output sizes: cycles={len(cycles)}, events={len(events)}")

        complete = [cycle for cycle in cycles if cycle["status"] == "complete"]
        history_only = [
            cycle for cycle in cycles if cycle["status"] == "history_only_outside_order_window"
        ]
        right_truncated = [cycle for cycle in cycles if cycle["status"] == "right_truncated"]
        if (len(complete), len(history_only), len(right_truncated), len(unmatched)) != (157, 43, 2, 3):
            raise ValueError("Source coverage no longer matches the verified 157/43/2 cycle split")

        complete_events = [event for event in events if event["classification"] == "history_anchored"]
        action_counts = Counter(event["action"] for event in events)
        complete_action_order_types = Counter(
            (event["action"], event["order_type"]) for event in complete_events
        )
        add_states = Counter(
            event["add_pnl_state"] for event in complete_events if event["action"] == "add"
        )
        durations = sorted(cycle["duration_ms"] for cycle in complete)
        cycle_pnls = [required_dec(cycle["realized_pnl"], "cycle.realized_pnl") for cycle in complete]
        flow_sum = sum(
            (required_dec(cycle["price_flow_pnl"], "cycle.price_flow_pnl") for cycle in complete), ZERO
        )
        order_pnl_sum = sum(
            (required_dec(cycle["order_total_pnl"], "cycle.order_total_pnl") for cycle in complete), ZERO
        )
        history_pnl_sum = sum(
            (required_dec(cycle["realized_pnl"], "cycle.realized_pnl") for cycle in complete), ZERO
        )
        turnover_sum = ZERO
        for event in complete_events:
            turnover_sum += required_dec(event["price"], "event.price") * required_dec(
                event["qty"], "event.qty"
            )
        abs_flow_history_diffs = [
            abs(
                required_dec(cycle["price_flow_pnl"], "cycle.price_flow_pnl")
                - required_dec(cycle["realized_pnl"], "cycle.realized_pnl")
            )
            for cycle in complete
        ]
        duration_mean = sum((D(value) for value in durations), ZERO) / D(len(durations))
        median_duration = D(str(median(durations)))
        sorted_pnls = sorted(cycle_pnls)
        median_pnl = sorted_pnls[len(sorted_pnls) // 2]

        summary = {
            "dataset": "2026-10-02_2026大黑马",
            "source_files": {
                "orders": "data/binance/2026-10-02_2026大黑马/leader_orders.json",
                "position_history": "data/binance/2026-10-02_2026大黑马/leader_position_history_first.json",
            },
            "source_coverage": {
                "leader_orders_count": len(orders),
                "leader_orders_time_min_ms": order_window_start,
                "leader_orders_time_max_ms": order_window_end,
                "leader_position_history_returned": len(history),
                "leader_position_history_reported_total": history_total,
                "anchor_match_tolerance_ms": ORDER_MATCH_TOLERANCE_MS,
                "order_time_field_used_for_matching_and_events": "orderUpdateTime",
                "order_time_field_preserved_separately": "orderTime",
                "time_zone": "UTC",
            },
            "reconstruction_counts": {
                "cycle_records": len(cycles),
                "closed_history_cycles_with_full_order_sequences": len(complete),
                "closed_history_cycles_outside_order_window": len(history_only),
                "right_truncated_local_cycles": len(right_truncated),
                "event_records": len(events),
                "history_anchored_classified_events": len(complete_events),
                "right_truncated_inferred_open_events": 2,
                "unclassified_unanchored_events": action_counts["unknown"],
                "unclassified_event_symbols": ["CTUSDT"],
                "unmatched_events_total": len(unmatched),
                "ambiguous_anchor_events": 0,
                "cycle_status_counts": dict(Counter(cycle["status"] for cycle in cycles)),
            },
            "status_definitions": {
                "complete": "Closed history anchor plus a fully matched operation sequence that begins at the history open and returns to zero at close.",
                "history_only_outside_order_window": "Closed history anchor predates the earliest saved order event; individual actions are unavailable.",
                "right_truncated": "A likely new position inferred after the latest closed anchor; no close appears before collection ended.",
                "unclassified_right_edge": "An order with no history anchor and insufficient observations to decide open/add/reduce/close.",
            },
            "complete_cycle_statistics": {
                "cycle_count": len(complete),
                "direction_counts": dict(Counter(cycle["direction"] for cycle in complete)),
                "action_counts": dict(Counter(event["action"] for event in complete_events)),
                "order_type_counts": dict(Counter(event["order_type"] for event in complete_events)),
                "action_by_order_type": {
                    f"{action}:{order_type}": count
                    for (action, order_type), count in sorted(complete_action_order_types.items())
                },
                "duration_ms_median": dec_string(median_duration),
                "duration_ms_mean": dec_string(duration_mean),
                "closing_pnl_positive_cycles": sum(pnl > ZERO for pnl in cycle_pnls),
                "closing_pnl_negative_cycles": sum(pnl < ZERO for pnl in cycle_pnls),
                "closing_pnl_zero_cycles": sum(pnl == ZERO for pnl in cycle_pnls),
                "closing_pnl_median": dec_string(median_pnl),
                "add_state_by_pre_add_weighted_cost": dict(add_states),
                "add_count": sum(cycle["add_count"] for cycle in complete),
                "reduce_count": sum(cycle["reduce_count"] for cycle in complete),
            },
            "pnl_reconciliation": {
                "matched_complete_cycle_count": len(complete),
                "price_cashflow_pnl_sum": dec_string(flow_sum),
                "order_totalPnl_sum": dec_string(order_pnl_sum),
                "history_closingPnl_sum": dec_string(history_pnl_sum),
                "order_totalPnl_minus_price_cashflow": dec_string(order_pnl_sum - flow_sum),
                "price_cashflow_minus_history_closingPnl": dec_string(flow_sum - history_pnl_sum),
                "turnover_sum_from_saved_order_qty_times_price": dec_string(turnover_sum),
                "cycles_abs_price_flow_vs_closingPnl_gt_0_01": sum(
                    value > D("0.01") for value in abs_flow_history_diffs
                ),
                "notes": [
                    "Price cash flow is SELL quantity×price minus BUY quantity×price across each flat reconstructed cycle; it is a gross trade-flow measure.",
                    "The sum of order totalPnl differs from price cash flow by a small amount; the history closingPnl total differs by a larger amount. The package does not expose the components needed to fully explain that gap.",
                    "realized_pnl in cycles copies the public position-history closingPnl field. The leader's fee, funding, and profit-sharing components are not independently exposed for reconciliation here; the official closingPnl value is retained as reported.",
                ],
            },
            "interpretation_notes": [
                "Order events use orderUpdateTime as the saved update/fill time; orderTime is preserved separately. A LIMIT order's update time cannot be treated as its placement or strategy-decision time.",
                "MARKET/LIMIT is the leader order's public order type. It does not establish maker/taker status or the follower account's execution type.",
                "Open positions are signed in position_before/position_after: positive for LONG, negative for SHORT. Quantity is always a positive absolute order size.",
                "Pre-add profit/loss compares the add price with the weighted average cost immediately before that add, without fees or funding.",
                "No exchange order ID or reduceOnly field is available in the standardized leader order rows; both are null rather than inferred.",
                "The position-history endpoint returned 200 of 1,196 records; these records are not the leader's full history.",
            ],
        }

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        write_json(RESULTS_DIR / "leader_cycles.json", cycles)
        write_json(RESULTS_DIR / "leader_events.json", events)
        write_json(RESULTS_DIR / "leader_summary.json", summary)
        print(
            json.dumps(
                {
                    "cycles": len(cycles),
                    "events": len(events),
                    "complete_cycles": len(complete),
                    "history_only_cycles": len(history_only),
                    "right_truncated_cycles": len(right_truncated),
                    "unclassified_events": action_counts["unknown"],
                    "actions": dict(action_counts),
                    "pnl": summary["pnl_reconciliation"],
                    "results_dir": str(RESULTS_DIR),
                },
                ensure_ascii=False,
                indent=2,
                default=json_default,
            )
        )


if __name__ == "__main__":
    main()
