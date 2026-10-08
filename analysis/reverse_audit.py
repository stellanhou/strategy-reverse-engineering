#!/usr/bin/env python3
"""Read-only audit of leader-order timing, position coverage, and market support."""

from __future__ import annotations

import gzip
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "binance" / "2026-10-02_2026大黑马"
RESULTS = ROOT / "analysis" / "results"
OUT = RESULTS / "reverse_audit.json"
MINUTE_MS = 60_000
ANCHOR_TOLERANCE_MS = 2_000
D = Decimal
ZERO = D("0")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, parse_float=D)


def as_decimal(value: Any) -> Decimal:
    return value if isinstance(value, Decimal) else D(str(value))


def utc(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat(timespec="milliseconds")


def number(value: Decimal | int | float | None) -> str | int | float | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        rendered = format(value, "f")
        if "." in rendered:
            rendered = rendered.rstrip("0").rstrip(".")
        return rendered or "0"
    return value


def signature(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)


def quantile_summary(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "p90_nearest_rank": None, "max": None}
    ordered = sorted(values)
    p90_index = max(0, (9 * len(ordered) + 9) // 10 - 1)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90_nearest_rank": ordered[p90_index],
        "max": ordered[-1],
    }


def decimal_summary(values: list[Decimal]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": number(ordered[0]),
        "median": number(D(str(statistics.median(ordered)))),
        "max": number(ordered[-1]),
    }


def main() -> None:
    orders: list[dict[str, Any]] = read_json(DATA / "leader_orders.json")
    raw_pages: list[dict[str, Any]] = read_json(DATA / "leader_orders_raw.json")
    tail_pages: list[dict[str, Any]] = read_json(DATA / "leader_orders_tail_raw.json")
    history_response: dict[str, Any] = read_json(DATA / "leader_position_history_first.json")
    history: list[dict[str, Any]] = history_response["data"]["list"]
    history_total = int(history_response["data"]["total"])
    market_manifest: dict[str, Any] = read_json(DATA / "market" / "manifest.json")
    market_quality: dict[str, Any] = read_json(DATA / "market_quality.json")
    exchange_info: dict[str, Any] = read_json(DATA / "market" / "exchange_info.json")
    saved_events: list[dict[str, Any]] = read_json(RESULTS / "leader_events.json")
    saved_cycles: list[dict[str, Any]] = read_json(RESULTS / "leader_cycles.json")
    saved_summary: dict[str, Any] = read_json(RESULTS / "leader_summary.json")

    raw_rows = [row for page in raw_pages for row in page["response"]["data"]["list"]]
    tail_rows = [row for page in tail_pages for row in page["response"]["data"]["list"]]
    main_signatures = {signature(row) for row in raw_rows}
    tail_signatures = {signature(row) for row in tail_rows}
    normalized_signatures = {signature(row) for row in orders}

    required_fields = ("symbol", "side", "type", "executedQty", "avgPrice", "totalPnl", "orderTime", "orderUpdateTime")
    missing_required = {
        field: sum(field not in row or row[field] is None for row in orders)
        for field in required_fields
    }
    absent_lifecycle_fields = ("orderId", "status", "origQty", "reduceOnly", "clientOrderId")
    lifecycle_field_presence = {
        field: sum(field in row and row[field] is not None for row in orders)
        for field in absent_lifecycle_fields
    }
    position_side_counts = Counter(str(row.get("positionSide")) for row in orders)

    first_update = min(int(row["orderUpdateTime"]) for row in orders)
    last_update = max(int(row["orderUpdateTime"]) for row in orders)
    first_order_time = min(int(row["orderTime"]) for row in orders)
    last_order_time = max(int(row["orderTime"]) for row in orders)
    study_minute_start = first_update // MINUTE_MS * MINUTE_MS
    study_minute_end = last_update // MINUTE_MS * MINUTE_MS
    study_minute_count = (study_minute_end - study_minute_start) // MINUTE_MS + 1

    # Match source orders to the returned position-history records independently.
    history_orders: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    unmatched_orders: list[tuple[int, dict[str, Any]]] = []
    ambiguous_orders: list[int] = []
    for source_index, order in enumerate(orders, start=1):
        event_ms = int(order["orderUpdateTime"])
        matches = [
            row
            for row in history
            if row["symbol"] == order["symbol"]
            and int(row["opened"]) - ANCHOR_TOLERANCE_MS <= event_ms <= int(row["closed"]) + ANCHOR_TOLERANCE_MS
        ]
        if len(matches) > 1:
            ambiguous_orders.append(source_index)
        elif matches:
            history_orders[str(matches[0]["positionId"])].append((source_index, order))
        else:
            unmatched_orders.append((source_index, order))

    histories_with_orders = 0
    histories_without_orders_before_window = 0
    histories_without_orders_in_window = []
    cycle_rows: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    action_type_counts: Counter[str] = Counter()
    action_lags: dict[str, list[int]] = defaultdict(list)
    replay_violations: list[dict[str, Any]] = []
    anchor_open_diffs: list[int] = []
    anchor_close_diffs: list[int] = []
    pnl_cycle_gaps: list[tuple[Decimal, dict[str, Any]]] = []
    closed_volume_diffs: list[Decimal] = []
    max_position_diffs: list[Decimal] = []
    avg_close_price_diffs: list[Decimal] = []
    event_source_index_action: dict[int, tuple[str, str]] = {}

    for position in history:
        position_id = str(position["positionId"])
        assigned = history_orders.get(position_id, [])
        if not assigned:
            if int(position["closed"]) < first_update:
                histories_without_orders_before_window += 1
            else:
                histories_without_orders_in_window.append(position_id)
            continue

        histories_with_orders += 1
        direction_sign = D(1) if position["side"] == "Long" else D(-1)
        signed_position = ZERO
        average_cost = ZERO
        maximum_abs_position = ZERO
        total_closed_qty = ZERO
        total_closed_notional = ZERO
        order_flow_pnl = ZERO
        reported_order_pnl = ZERO
        opens = adds = reduces = closes = 0
        ordered = sorted(assigned, key=lambda pair: (int(pair[1]["orderUpdateTime"]), pair[0]))
        anchor_open_diffs.append(int(ordered[0][1]["orderUpdateTime"]) - int(position["opened"]))
        anchor_close_diffs.append(int(ordered[-1][1]["orderUpdateTime"]) - int(position["closed"]))

        for source_index, order in ordered:
            quantity = as_decimal(order["executedQty"])
            price = as_decimal(order["avgPrice"])
            signed_qty = quantity if order["side"] == "BUY" else -quantity
            before = signed_position
            after = before + signed_qty
            if before == ZERO:
                action = "open"
                opens += 1
                if signed_qty * direction_sign <= ZERO:
                    replay_violations.append({"position_id": position_id, "source_index": source_index, "issue": "opening side conflicts with history direction"})
                average_cost = price
            elif signed_qty * before > ZERO:
                action = "add"
                adds += 1
                average_cost = (abs(before) * average_cost + quantity * price) / (abs(before) + quantity)
            else:
                if after == ZERO:
                    action = "close"
                    closes += 1
                elif after * before > ZERO:
                    action = "reduce"
                    reduces += 1
                else:
                    action = "reverse_inside_cycle"
                    replay_violations.append({"position_id": position_id, "source_index": source_index, "issue": "order reverses position inside history cycle"})
                closing_quantity = min(abs(before), quantity)
                total_closed_qty += closing_quantity
                total_closed_notional += closing_quantity * price
                order_flow_pnl += (price - average_cost) * closing_quantity * (D(1) if before > ZERO else D(-1))
                if after == ZERO:
                    average_cost = ZERO

            action_counts[action] += 1
            action_type_counts[f"{action}:{order['type']}"] += 1
            action_lags[action].append(int(order["orderUpdateTime"]) - int(order["orderTime"]))
            event_source_index_action[source_index] = (action, order["type"])
            signed_position = after
            maximum_abs_position = max(maximum_abs_position, abs(signed_position))
            reported_order_pnl += as_decimal(order["totalPnl"])

        if signed_position != ZERO or opens != 1 or closes != 1:
            replay_violations.append({
                "position_id": position_id,
                "issue": "sequence does not open once, close once, and finish flat",
                "ending_position": number(signed_position),
                "opens": opens,
                "closes": closes,
            })

        official_closing_pnl = as_decimal(position["closingPnl"])
        history_closed_volume = as_decimal(position["closedVolume"])
        history_max_interest = as_decimal(position["maxOpenInterest"])
        history_avg_close = as_decimal(position["avgClosePrice"])
        independent_avg_close = total_closed_notional / total_closed_qty if total_closed_qty else ZERO
        closed_volume_diffs.append(abs(total_closed_qty - history_closed_volume))
        max_position_diffs.append(abs(maximum_abs_position - history_max_interest))
        avg_close_price_diffs.append(abs(independent_avg_close - history_avg_close))
        pnl_cycle_gaps.append((abs(order_flow_pnl - official_closing_pnl), {
            "position_id": position_id,
            "symbol": position["symbol"],
            "direction": position["side"].upper(),
            "orders_price_flow_pnl": number(order_flow_pnl),
            "sum_order_totalPnl": number(reported_order_pnl),
            "history_closingPnl": number(official_closing_pnl),
            "abs_flow_vs_history_gap": number(abs(order_flow_pnl - official_closing_pnl)),
        }))
        cycle_rows.append({
            "position_id": position_id,
            "symbol": position["symbol"],
            "direction": position["side"].upper(),
            "event_count": len(ordered),
            "open_count": opens,
            "add_count": adds,
            "reduce_count": reduces,
            "close_count": closes,
            "duration_ms": int(position["closed"]) - int(position["opened"]),
            "order_price_flow_pnl": order_flow_pnl,
            "order_totalPnl": reported_order_pnl,
            "history_closingPnl": official_closing_pnl,
            "max_abs_position": maximum_abs_position,
            "closed_volume": total_closed_qty,
        })

    unmatched_by_symbol: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for pair in unmatched_orders:
        unmatched_by_symbol[pair[1]["symbol"]].append(pair)
    inferred_unmatched: list[dict[str, Any]] = []
    for symbol, rows in unmatched_by_symbol.items():
        if symbol == "CTUSDT":
            continue
        for source_index, order in rows:
            previous_closed = [
                item for item in history
                if item["symbol"] == symbol and int(item["closed"]) < int(order["orderUpdateTime"])
            ]
            if not previous_closed:
                continue
            latest = max(previous_closed, key=lambda item: int(item["closed"]))
            intervening = [
                other for other in orders
                if other["symbol"] == symbol
                and int(latest["closed"]) < int(other["orderUpdateTime"]) < int(order["orderUpdateTime"])
            ]
            if not intervening and order["type"] == "MARKET" and as_decimal(order["totalPnl"]) == ZERO:
                inferred_unmatched.append({"source_index": source_index, "symbol": symbol, "position_id": str(latest["positionId"])})

    saved_event_by_index = {int(event["source_index"]): event for event in saved_events}
    event_mismatches = []
    for source_index, order in enumerate(orders, start=1):
        event = saved_event_by_index.get(source_index)
        if event is None:
            event_mismatches.append({"source_index": source_index, "issue": "event missing"})
            continue
        comparisons = {
            "symbol": order["symbol"],
            "side": order["side"],
            "order_type": order["type"],
            "time_ms": int(order["orderUpdateTime"]),
            "order_time_ms": int(order["orderTime"]),
            "qty": number(as_decimal(order["executedQty"])),
            "price": number(as_decimal(order["avgPrice"])),
        }
        if any(event.get(field) != expected for field, expected in comparisons.items()):
            event_mismatches.append({"source_index": source_index, "issue": "event differs from normalized source row"})
    for source_index in set(saved_event_by_index) - set(range(1, len(orders) + 1)):
        event_mismatches.append({"source_index": source_index, "issue": "event has no source row"})

    source_index_actions = Counter(f"{action}:{order_type}" for action, order_type in event_source_index_action.values())
    lag_by_action_type: dict[str, Any] = {}
    for key, values in sorted(defaultdict(list, {
        f"{action}:{order_type}": [
            int(order["orderUpdateTime"]) - int(order["orderTime"])
            for source_index, order in enumerate(orders, start=1)
            if event_source_index_action.get(source_index) == (action, order_type)
        ]
        for action, order_type in event_source_index_action.values()
    }).items()):
        lag_by_action_type[key] = quantile_summary(values)

    order_type_counts = Counter(row["type"] for row in orders)
    complete_direction_counts = Counter(row["direction"] for row in cycle_rows)
    lag_by_type: dict[str, Any] = {}
    for order_type in sorted(order_type_counts):
        rows = [row for row in orders if row["type"] == order_type]
        diffs = [int(row["orderUpdateTime"]) - int(row["orderTime"]) for row in rows]
        lag_by_type[order_type] = {
            **quantile_summary(diffs),
            "equal_timestamps": sum(value == 0 for value in diffs),
            "strictly_later_updates": sum(value > 0 for value in diffs),
            "updates_over_1_minute": sum(value > MINUTE_MS for value in diffs),
            "updates_over_1_hour": sum(value > 60 * MINUTE_MS for value in diffs),
        }

    collision_summary: dict[str, Any] = {}
    for field in ("orderTime", "orderUpdateTime"):
        grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
        for source_index, order in enumerate(orders, start=1):
            grouped[(order["symbol"], int(order[field]) // MINUTE_MS)].append(source_index)
        collisions = [indices for indices in grouped.values() if len(indices) > 1]
        collision_summary[field] = {
            "symbol_minutes_with_multiple_orders": len(collisions),
            "orders_in_those_symbol_minutes": sum(len(indices) for indices in collisions),
            "max_orders_in_one_symbol_minute": max((len(indices) for indices in collisions), default=0),
        }

    manifest_by_symbol = {row["symbol"]: row for row in market_manifest["results"]}
    kline_open_times: dict[str, set[int]] = {}
    kline_times_by_symbol: dict[str, list[int]] = {}
    market_by_symbol: dict[str, dict[str, Any]] = {}
    for symbol, record in manifest_by_symbol.items():
        with gzip.open(DATA / "market" / record["file"], "rt", encoding="utf-8") as stream:
            candles = json.load(stream, parse_float=D)
        times = [int(candle[0]) for candle in candles]
        kline_times_by_symbol[symbol] = times
        kline_open_times[symbol] = set(times)
        in_study = [time_ms for time_ms in times if study_minute_start <= time_ms <= study_minute_end]
        market_by_symbol[symbol] = {
            "rows": len(times),
            "first_open_ms": times[0],
            "last_open_ms": times[-1],
            "internal_gap_minutes": record.get("internal_gap_minutes"),
            "study_span_bar_count": len(in_study),
            "study_span_coverage_pct": round(100 * len(in_study) / study_minute_count, 6),
        }

    event_minute_missing: dict[str, list[int]] = {"orderTime": [], "orderUpdateTime": []}
    warmup_missing: dict[str, dict[str, list[int]]] = {
        field: {str(horizon): [] for horizon in (5, 15, 60)}
        for field in ("orderTime", "orderUpdateTime")
    }
    for source_index, order in enumerate(orders, start=1):
        opens = kline_open_times.get(order["symbol"], set())
        if not opens:
            for field in event_minute_missing:
                event_minute_missing[field].append(source_index)
            continue
        for field in event_minute_missing:
            event_ms = int(order[field])
            boundary = event_ms // MINUTE_MS * MINUTE_MS
            if boundary not in opens:
                event_minute_missing[field].append(source_index)
            for horizon in (5, 15, 60):
                # A return spanning H one-minute intervals needs H+1 completed close observations.
                required = range(boundary - (horizon + 1) * MINUTE_MS, boundary, MINUTE_MS)
                if any(bar_open not in opens for bar_open in required):
                    warmup_missing[field][str(horizon)].append(source_index)

    history_symbol_counts = Counter(row["symbol"] for row in history)
    complete_cycle_symbol_counts = Counter(row["symbol"] for row in cycle_rows)
    leader_symbols = sorted({row["symbol"] for row in orders})
    complete_symbols = sorted(complete_cycle_symbol_counts)
    panel_rectangles = {}
    for label, symbols in (
        ("leader_order_symbols", leader_symbols),
        ("history_anchored_complete_symbols", complete_symbols),
        ("all_monitored_market_symbols", sorted(manifest_by_symbol)),
    ):
        total_slots = len(symbols) * study_minute_count
        covered_slots = sum(market_by_symbol.get(symbol, {}).get("study_span_bar_count", 0) for symbol in symbols)
        panel_rectangles[label] = {
            "symbols": len(symbols),
            "expected_symbol_minutes_over_full_observed_order_span": total_slots,
            "available_market_minutes": covered_slots,
            "coverage_pct": round(100 * covered_slots / total_slots, 6) if total_slots else None,
        }

    requested_start_mismatches = []
    for record in market_manifest["results"]:
        requested = record.get("requested_start_ms")
        actual = record.get("actual_requested_start_ms", record.get("first_open_ms"))
        if requested is not None and actual is not None and int(actual) != int(requested):
            requested_start_mismatches.append({
                "symbol": record["symbol"],
                "manifest_status": record["status"],
                "requested_start_utc": utc(int(requested)),
                "first_returned_bar_utc": utc(int(actual)),
                "delay_minutes": (int(actual) - int(requested)) // MINUTE_MS,
                "internal_gap_minutes": record.get("internal_gap_minutes"),
            })

    history_pnl_sum = sum((as_decimal(row["history_closingPnl"]) for row in cycle_rows), ZERO)
    order_pnl_sum = sum((as_decimal(row["order_totalPnl"]) for row in cycle_rows), ZERO)
    flow_pnl_sum = sum((as_decimal(row["order_price_flow_pnl"]) for row in cycle_rows), ZERO)
    sorted_gaps = sorted(pnl_cycle_gaps, key=lambda item: item[0], reverse=True)
    abs_pnl_gaps = [gap for gap, _ in pnl_cycle_gaps]
    complete_durations = sorted(int(row["duration_ms"]) for row in cycle_rows)
    duration_min = complete_durations[0] if complete_durations else None
    duration_max = complete_durations[-1] if complete_durations else None

    cycle_status_counts = Counter(row["status"] for row in saved_cycles)
    cycle_status_by_symbol: dict[str, Counter[str]] = defaultdict(Counter)
    for cycle in saved_cycles:
        cycle_status_by_symbol[cycle["symbol"]][cycle["status"]] += 1
    saved_event_classifications = Counter(row.get("classification") for row in saved_events)
    saved_action_counts = Counter(row.get("action") for row in saved_events)
    saved_reconstruction_counts = saved_summary.get("reconstruction_counts", {})
    saved_pnl_reconciliation = saved_summary.get("pnl_reconciliation", {})
    raw_page_declared_totals = sorted({int(page["response"]["data"]["total"]) for page in raw_pages})
    tail_page_declared_totals = sorted({int(page["response"]["data"]["total"]) for page in tail_pages})

    exchange_symbols = exchange_info.get("symbols", [])
    exchange_status = Counter(row.get("status") for row in exchange_symbols)
    exchange_time = exchange_info.get("serverTime")

    report = {
        "audit": {
            "purpose": "阶段0/1数据与时间语义审计；只读现有文件；历史结果属于已查看的探索样本。",
            "script": "analysis/reverse_audit.py",
            "source_as_of_utc": market_manifest.get("as_of_utc"),
            "time_zone_for_epoch_fields": "UTC",
            "limits": [
                "原始公开行没有订单ID、状态、原始委托量或reduceOnly；executedQty大于零不能证明委托完整成交，也不能拆分部分成交时点。",
                "orderTime与orderUpdateTime的名字和观测关系支持创建/更新代理，但保存的响应没有字段定义、委托生命周期或隐藏策略决策时刻，故真实决策时间未独立验证。",
                "行情是按已观察交易币种和区间采集的局部面板；没有完整历史可交易币池，不能构造全市场选币负样本。",
                "公开仓位closingPnl与订单价格现金流差异无法由当前lead数据分解到费用、资金费或其他调整项。",
            ],
        },
        "source_integrity": {
            "normalized_orders": len(orders),
            "main_raw_pages": len(raw_pages),
            "main_raw_rows": len(raw_rows),
            "main_raw_unique_rows": len(main_signatures),
            "main_raw_declared_total_values": raw_page_declared_totals,
            "tail_raw_pages": len(tail_pages),
            "tail_raw_rows": len(tail_rows),
            "tail_raw_unique_rows": len(tail_signatures),
            "tail_raw_declared_total_values": tail_page_declared_totals,
            "tail_rows_also_in_main_raw": len(main_signatures & tail_signatures),
            "tail_rows_added_beyond_main_raw": len(tail_signatures - main_signatures),
            "normalized_rows_not_in_main_raw": len(normalized_signatures - main_signatures),
            "normalized_rows_not_in_main_or_tail": len(normalized_signatures - (main_signatures | tail_signatures)),
            "tail_rows_not_in_normalized": len(tail_signatures - normalized_signatures),
            "required_field_null_or_missing_counts": missing_required,
            "executed_qty_nonpositive": sum(as_decimal(row["executedQty"]) <= ZERO for row in orders),
            "avg_price_nonpositive": sum(as_decimal(row["avgPrice"]) <= ZERO for row in orders),
            "order_type_counts": dict(Counter(row["type"] for row in orders)),
            "side_counts": dict(Counter(row["side"] for row in orders)),
            "position_side_counts": dict(position_side_counts),
            "lifecycle_fields_present_row_counts": lifecycle_field_presence,
            "interpretation": "主raw快照只保存44×10=440条；尾部22条中14条补齐，8条重叠；raw+tail可逐行覆盖454条标准化记录。返回的行均有正executedQty，但缺原始量和状态，完成/部分成交无法区分。",
        },
        "time_semantics": {
            "order_update_time_span_utc": {"start": utc(first_update), "end": utc(last_update)},
            "order_time_span_utc": {"start": utc(first_order_time), "end": utc(last_order_time)},
            "order_time_after_update_rows": sum(int(row["orderTime"]) > int(row["orderUpdateTime"]) for row in orders),
            "update_minus_order_time_ms_by_type": lag_by_type,
            "update_minus_order_time_ms_by_complete_action_and_type": lag_by_action_type,
            "source_action_order_type_counts": dict(source_index_actions),
            "same_symbol_minute_collisions": collision_summary,
            "decision_time_recommendation": {
                "market_orders": "将orderTime作为可观察的创建/决策代理，将orderUpdateTime作为更新/成交代理；309条MARKET中307条相等，另2条只差1ms。不能据此声称恢复了策略内部决策时刻或在线到达时延。",
                "limit_orders": "分开建立创建时刻与执行/更新时刻事件流。145条LIMIT中123条orderTime早于orderUpdateTime；仅成功返回的成交行可见，挂单创建样本存在成交选择偏差，且缺少未成交/撤单/部分成交生命周期。不要把orderUpdateTime直接当作限价策略发起决策时刻。",
                "position_ledger": "只有在用于可观察仓位账本时，才按orderUpdateTime排序应用executedQty；它是最后更新/成交代理，部分成交精确时刻仍未知。",
            },
            "minute_boundary_recommendation": {
                "grid": "UTC Unix整分钟边界t。事件时间落入[t,t+60000)；同币同分钟多个事件保留为多事件/多标签，不合并成单一动作。",
                "features": "边界t的特征只用在t之前已结束的K线；不读取标签所在分钟的OHLCV。5/15/60分钟输入按固定已结束bar窗口生成。",
                "pre_decision_state": "样本状态只应用事件代理时间严格早于t的仓位变化，避免把当前标签事件放进其自身状态。",
                "holding_interval": "以position history精确锚点按左闭右开[opened, closed)定义持仓资格；分钟网格资格为opened_ms<=t<closed_ms。",
                "limit_order_state": "当前数据不能重建分钟级挂单待成交、部分成交或撤单状态；限价订单创建研究只对观测到最终成交的订单作条件描述。",
            },
        },
        "independent_position_replay": {
            "history_records_returned": len(history),
            "history_records_reported_total": history_total,
            "returned_share_of_total_pct": round(100 * len(history) / history_total, 6) if history_total else None,
            "history_records_with_matched_orders": histories_with_orders,
            "history_only_closed_before_order_window": histories_without_orders_before_window,
            "unmatched_orders": len(unmatched_orders),
            "unmatched_order_symbols": {symbol: len(rows) for symbol, rows in sorted(unmatched_by_symbol.items())},
            "unmatched_market_rows_meeting_inferred_open_heuristic": len(inferred_unmatched),
            "inferred_open_source_rows": inferred_unmatched,
            "ambiguous_order_to_history_matches": len(ambiguous_orders),
            "history_records_without_orders_inside_order_window": histories_without_orders_in_window,
            "complete_replayed_cycles": len(cycle_rows),
            "replayed_complete_symbols": len(complete_symbols),
            "complete_cycle_direction_counts": dict(complete_direction_counts),
            "order_symbols_vs_cycle_status": {
                "symbols_with_any_order": len(leader_symbols),
                "symbols_with_complete_anchored_cycles": len(complete_symbols),
                "unmatched_symbols": sorted(unmatched_by_symbol),
                "right_truncated_inferred_symbols": sorted(row["symbol"] for row in inferred_unmatched),
                "cycle_counts_by_symbol": {
                    symbol: dict(counts)
                    for symbol, counts in sorted(cycle_status_by_symbol.items())
                },
            },
            "replayed_action_counts": dict(action_counts),
            "replayed_action_by_type": dict(action_type_counts),
            "all_replayed_cycles_open_once_close_once_finish_flat": bool(cycle_rows) and not replay_violations,
            "replay_violations": replay_violations,
            "anchor_time_differences_ms": {
                "first_order_update_minus_history_open": quantile_summary(anchor_open_diffs),
                "last_order_update_minus_history_close": quantile_summary(anchor_close_diffs),
                "max_abs_open_difference": max(map(abs, anchor_open_diffs), default=None),
                "max_abs_close_difference": max(map(abs, anchor_close_diffs), default=None),
                "tolerance_ms": ANCHOR_TOLERANCE_MS,
            },
            "quantity_reconciliation": {
                "cycles_exactly_matching_history_closed_volume": sum(value == ZERO for value in closed_volume_diffs),
                "max_abs_closed_volume_difference": number(max(closed_volume_diffs, default=ZERO)),
                "cycles_exactly_matching_history_max_open_interest": sum(value == ZERO for value in max_position_diffs),
                "max_abs_max_open_interest_difference": number(max(max_position_diffs, default=ZERO)),
                "cycles_avg_close_price_within_0_000001": sum(value <= D("0.000001") for value in avg_close_price_diffs),
                "max_abs_avg_close_price_difference": number(max(avg_close_price_diffs, default=ZERO)),
            },
            "cycle_duration_minutes": {
                "count": len(complete_durations),
                "min": number(D(duration_min) / D(MINUTE_MS)) if duration_min is not None else None,
                "median": number(D(str(statistics.median(complete_durations))) / D(MINUTE_MS)) if complete_durations else None,
                "mean": number(D(sum(complete_durations)) / D(len(complete_durations)) / D(MINUTE_MS)) if complete_durations else None,
                "max": number(D(duration_max) / D(MINUTE_MS)) if duration_max is not None else None,
            },
            "pnl_reconciliation_usdt": {
                "replayed_order_price_cashflow_sum": number(flow_pnl_sum),
                "replayed_order_totalPnl_sum": number(order_pnl_sum),
                "position_history_closingPnl_sum": number(history_pnl_sum),
                "order_totalPnl_minus_price_cashflow": number(order_pnl_sum - flow_pnl_sum),
                "price_cashflow_minus_history_closingPnl": number(flow_pnl_sum - history_pnl_sum),
                "cycles_abs_priceflow_vs_history_closingPnl_gt_0_01": sum(value > D("0.01") for value in abs_pnl_gaps),
                "median_abs_cycle_gap": number(D(str(statistics.median(abs_pnl_gaps)))) if abs_pnl_gaps else None,
                "max_abs_cycle_gap": number(max(abs_pnl_gaps, default=ZERO)),
                "largest_cycle_gaps": [row for _, row in sorted_gaps[:8]],
                "finding": "订单数量与方向可独立重放并回到零仓；但订单价格现金流与官方closingPnl在55/157周期相差超过0.01 USDT，最大单轮差2.90060427 USDT。缺少lead端费用、资金费及其他分项，官方closingPnl不能由当前订单字段逐轮复算。",
            },
        },
        "market_coverage": {
            "monitored_symbols": len(market_manifest["results"]),
            "kline_rows": sum(int(row["rows"]) for row in market_manifest["results"]),
            "study_order_update_span_utc": {"start": utc(study_minute_start), "end": utc(study_minute_end)},
            "study_span_minutes": study_minute_count,
            "symbol_minute_panel_rectangle_coverage": panel_rectangles,
            "market_event_minutes_missing_by_time_field": {field: len(indices) for field, indices in event_minute_missing.items()},
            "complete_lookback_missing_by_time_field_and_horizon": {
                field: {horizon: len(indices) for horizon, indices in horizons.items()}
                for field, horizons in warmup_missing.items()
            },
            "requested_start_not_equal_first_returned_bar": requested_start_mismatches,
            "manifest_status_counts": dict(Counter(row["status"] for row in market_manifest["results"])),
            "market_quality_saved": market_quality,
            "symbol_coverage": {
                symbol: market_by_symbol.get(symbol)
                for symbol in sorted(set(leader_symbols) | set(manifest_by_symbol))
            },
            "historical_universe": {
                "available": False,
                "evidence": f"Market files cover {len(manifest_by_symbol)} monitored symbols; {len(leader_symbols)} appear in the 454-order sample. The current exchangeInfo snapshot contains {len(exchange_symbols)} symbols (serverTime={utc(int(exchange_time)) if exchange_time is not None else None}) but does not provide a time-indexed roster of every historically eligible/listed symbol or delisting state.",
                "current_exchange_info_status_counts": dict(exchange_status),
                "consequence": "全市场选币和全局未操作负样本不可验收；本地可做的历史时机解释只能限定在逐币实际行情覆盖及可观察币种内。",
            },
        },
        "existing_reconstruction_crosscheck": {
            "saved_event_rows": len(saved_events),
            "saved_cycle_rows": len(saved_cycles),
            "saved_cycle_status_counts": dict(cycle_status_counts),
            "saved_event_classification_counts": dict(saved_event_classifications),
            "saved_event_action_counts": dict(saved_action_counts),
            "saved_events_vs_normalized_source_row_mismatches": event_mismatches,
            "saved_complete_cycle_count_equals_independent_replay": cycle_status_counts.get("complete", 0) == len(cycle_rows),
            "saved_summary_vs_independent_checks": {
                "order_count_matches": saved_summary.get("source_coverage", {}).get("leader_orders_count") == len(orders),
                "history_returned_count_matches": saved_summary.get("source_coverage", {}).get("leader_position_history_returned") == len(history),
                "history_reported_total_matches": saved_summary.get("source_coverage", {}).get("leader_position_history_reported_total") == history_total,
                "complete_cycle_count_matches": saved_reconstruction_counts.get("closed_history_cycles_with_full_order_sequences") == len(cycle_rows),
                "history_only_count_matches": saved_reconstruction_counts.get("closed_history_cycles_outside_order_window") == histories_without_orders_before_window,
                "right_truncated_count_matches": saved_reconstruction_counts.get("right_truncated_local_cycles") == len(inferred_unmatched),
                "independent_order_flow_total_matches_saved_within_1e_10": abs(
                    as_decimal(saved_pnl_reconciliation.get("price_cashflow_pnl_sum", "0")) - flow_pnl_sum
                ) <= D("0.0000000001"),
                "independent_official_closing_pnl_total_matches_saved_within_1e_10": abs(
                    as_decimal(saved_pnl_reconciliation.get("history_closingPnl_sum", "0")) - history_pnl_sum
                ) <= D("0.0000000001"),
            },
            "note": "本审计从标准化订单与position-history重新匹配并重放；不把既有summary中的通过结论当作输入证明。",
        },
        "stage0_stage1_recommendations": {
            "primary_behavior_scope": f"先研究空头；完整锚定样本为空头{complete_direction_counts.get('SHORT', 0)}轮、多头{complete_direction_counts.get('LONG', 0)}轮，多头仅描述。",
            "causal_decision_proxy": "MARKET以orderTime作创建/信号代理、orderUpdateTime作执行/状态变更代理；LIMIT分别记录创建和执行，不将二者压成一个事件时间。所有结论称可观察代理关联，不称隐藏决策因果。",
            "opening_denominator": "只在当分钟行情与所需预热齐全、并且该合约在分钟边界t处净仓为空的symbol-minute评估开仓；目前只代表被观察币种条件样本。",
            "held_action_denominator": "加仓/减仓/平仓只在边界t已持仓、分钟行情齐全、且状态来自早于t的已观察更新时评估；将同币同分钟多动作保留，不把缺K线或右截断尾部编码成无动作。",
            "limit_denominator": "缺未成交/撤单订单，不能形成无选择偏差的限价挂单正负样本集；限价创建标签仅作填单条件下探索。",
            "historical_coin_selection": "暂不对历史全市场币池选币打分；50个监控合约不等于当时完整可交易宇宙，按计划先做已观察币种内时机解释。",
            "historical_status": "全部历史用于假设提出/筛选；阶段6.1中的历史切分是探索性诊断，不是独立最终测试。",
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, default=str, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "output": str(OUT),
        "orders": len(orders),
        "replayed_complete_cycles": len(cycle_rows),
        "replay_violations": len(replay_violations),
        "MARKET_lag": lag_by_type.get("MARKET"),
        "LIMIT_lag": lag_by_type.get("LIMIT"),
        "pnl_gap_gt_0_01": sum(value > D("0.01") for value in abs_pnl_gaps),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
