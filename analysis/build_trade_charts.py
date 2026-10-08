#!/usr/bin/env python3
"""Build selected leader-cycle SVG charts and a browsable HTML index."""

from __future__ import annotations

import datetime as dt
import gzip
import html
import json
import math
import re
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "binance" / "2026-10-02_2026大黑马"
RESULTS = ROOT / "analysis" / "results"
CHARTS = RESULTS / "charts"
BEIJING = ZoneInfo("Asia/Shanghai")
PRICE_SAMPLE_BUDGET = 1500


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def number(value):
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value: {value!r}")
    return result


def bj_time(epoch_ms: int, seconds: bool = True) -> str:
    value = dt.datetime.fromtimestamp(epoch_ms / 1000, tz=dt.timezone.utc).astimezone(BEIJING)
    return value.strftime("%Y-%m-%d %H:%M:%S" if seconds else "%m-%d %H:%M")


def slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value)).strip("-")
    return result[:80] or "cycle"


def load_inputs():
    cycles_path = RESULTS / "leader_cycles.json"
    events_path = RESULTS / "leader_events.json"
    if not cycles_path.exists() or not events_path.exists():
        missing = [str(p.relative_to(ROOT)) for p in (cycles_path, events_path) if not p.exists()]
        raise FileNotFoundError("Missing leader reconstruction output: " + ", ".join(missing))
    cycles = read_json(cycles_path)
    events = read_json(events_path)
    if not isinstance(cycles, list) or not isinstance(events, list):
        raise TypeError("leader_cycles.json and leader_events.json must be top-level arrays")
    cycle_fields = {"cycle_id", "symbol", "direction", "start_ms", "end_ms", "status", "add_count", "reduce_count", "realized_pnl"}
    event_fields = {"event_id", "cycle_id", "symbol", "time_ms", "action", "price", "qty", "position_before", "position_after", "order_type", "classification"}
    allowed_statuses = {"complete", "history_only_outside_order_window", "right_truncated"}
    cycle_ids = set()
    for index, cycle in enumerate(cycles):
        if not isinstance(cycle, dict) or cycle_fields - cycle.keys():
            raise ValueError(f"Invalid leader cycle record at index {index}: missing {sorted(cycle_fields - cycle.keys()) if isinstance(cycle, dict) else 'object'}")
        if cycle["cycle_id"] is None or cycle["cycle_id"] in cycle_ids:
            raise ValueError(f"Leader cycle ID is missing or duplicated at index {index}")
        if cycle["status"] not in allowed_statuses:
            raise ValueError(f"Leader cycle {cycle['cycle_id']} has unknown status: {cycle['status']!r}")
        cycle_ids.add(cycle["cycle_id"])
    event_ids = set()
    for index, event in enumerate(events):
        if not isinstance(event, dict) or event_fields - event.keys():
            raise ValueError(f"Invalid leader event record at index {index}: missing {sorted(event_fields - event.keys()) if isinstance(event, dict) else 'object'}")
        if event["event_id"] is None or event["event_id"] in event_ids:
            raise ValueError(f"Leader event ID is missing or duplicated at index {index}")
        event_ids.add(event["event_id"])
        if event["action"] not in {"open", "add", "reduce", "close", "unknown"}:
            raise ValueError(f"Leader event {event['event_id']} has unknown action: {event['action']!r}")
        if number(event["time_ms"]) is None:
            raise ValueError(f"Leader event {event['event_id']} has no event timestamp")
        if event["cycle_id"] is None:
            if event["action"] != "unknown":
                raise ValueError(f"Unanchored leader event {event['event_id']} has a classified action")
        elif event["cycle_id"] not in cycle_ids:
            raise ValueError(f"Leader event {event['event_id']} references unknown cycle {event['cycle_id']!r}")
    return cycles, events


def complete_cycles(cycles):
    """Keep only explicitly completed, non-truncated cycles with a full interval."""
    result = []
    for cycle in cycles:
        status = cycle["status"]
        if status != "complete":
            continue
        start = number(cycle["start_ms"])
        end = number(cycle["end_ms"])
        if start is None or end is None or not start.is_integer() or not end.is_integer() or end <= start:
            raise ValueError(f"Complete cycle {cycle['cycle_id']} has invalid start/end times")
        if cycle["cycle_id"] is None or cycle["symbol"] is None:
            raise ValueError("Complete cycle is missing its ID or symbol")
        if cycle["direction"] not in {"LONG", "SHORT", "unknown"}:
            raise ValueError(f"Complete cycle {cycle['cycle_id']} has unknown direction: {cycle['direction']!r}")
        counts = (number(cycle["add_count"]), number(cycle["reduce_count"]))
        if any(value is None or not value.is_integer() for value in counts) or number(cycle["realized_pnl"]) is None:
            raise ValueError(f"Complete cycle {cycle['cycle_id']} is missing counts or official PnL")
        result.append(cycle)
    return result


def duration_ms(cycle) -> float:
    return float(cycle["end_ms"]) - float(cycle["start_ms"])


def select_examples(cycles):
    """Select exact extremes for distinct research questions, merging overlaps."""
    if not cycles:
        raise ValueError("No complete leader cycles are available for chart selection")

    chosen: dict[str, dict] = {}

    def add_choice(cycle, category, reason):
        key = str(cycle["cycle_id"])
        if key not in chosen:
            chosen[key] = {"cycle": cycle, "categories": [], "reasons": []}
        chosen[key]["categories"].append(category)
        chosen[key]["reasons"].append(reason)

    c = max(cycles, key=lambda x: (number(x["add_count"]), str(x["cycle_id"])))
    add_choice(c, "最多加仓", f"加仓 {int(number(c['add_count']))} 次（完整周期中最多）")

    loser = min(cycles, key=lambda x: (number(x["realized_pnl"]), str(x["cycle_id"])))
    add_choice(loser, "最大亏损", f"官方仓位盈亏 {cents(number(loser['realized_pnl']))} USDT（完整周期中最低）")
    winner = max(cycles, key=lambda x: (number(x["realized_pnl"]), str(x["cycle_id"])))
    add_choice(winner, "最高盈利", f"官方仓位盈亏 {cents(number(winner['realized_pnl']))} USDT（完整周期中最高）")

    durations = sorted(duration_ms(c) for c in cycles)
    middle_duration = durations[(len(durations) - 1) // 2]
    shortest = min(cycles, key=lambda c: (duration_ms(c), str(c["cycle_id"])))
    longest = max(cycles, key=lambda c: (duration_ms(c), str(c["cycle_id"])))
    add_choice(shortest, "最短持仓", f"持仓 {duration_label(duration_ms(shortest))}（完整周期中最短）")
    add_choice(longest, "最长持仓", f"持仓 {duration_label(duration_ms(longest))}（完整周期中最长）")

    # Include a long-direction cycle near the median holding duration.
    long_cycles = [c for c in cycles if c["direction"] == "LONG"]
    if not long_cycles:
        raise ValueError("No complete LONG cycle is available for the requested representative sample")
    central = min(long_cycles, key=lambda c: (abs(duration_ms(c) - middle_duration), str(c["cycle_id"])))
    reason = f"方向为多，持仓 {duration_label(duration_ms(central))}，在多头周期中最接近总体持仓时长中位值"
    add_choice(central, "多头周期", reason)

    # If criteria overlap heavily, fill with the remaining cycle closest to median PnL.
    remaining = [c for c in cycles if str(c["cycle_id"]) not in chosen]
    remaining = sorted(
        remaining,
        key=lambda c: (
            abs(number(c["realized_pnl"]) - pnl_median(cycles)),
            abs(duration_ms(c) - middle_duration),
            str(c["cycle_id"]),
        ),
    )
    while len(chosen) < 6 and remaining:
        cycle = remaining.pop(0)
        add_choice(cycle, "中位盈亏", f"官方仓位盈亏 {cents(number(cycle['realized_pnl']))} USDT，最接近完整周期盈亏中位值；持仓时长仅用于打破并列")

    values = list(chosen.values())
    for item in values:
        item["reasons"] = list(dict.fromkeys(item["reasons"]))
        item["categories"] = list(dict.fromkeys(item["categories"]))
    return values[:6]


def pnl_median(cycles):
    values = sorted(number(c["realized_pnl"]) for c in cycles)
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def cents(value):
    return f"{value:,.4f}"


def duration_label(value_ms):
    seconds = round(value_ms / 1000)
    if seconds < 60:
        return f"{seconds}秒"
    minutes = seconds // 60
    days, minutes = divmod(minutes, 24 * 60)
    hours, minutes = divmod(minutes, 60)
    pieces = []
    if days:
        pieces.append(f"{days}天")
    if hours:
        pieces.append(f"{hours}小时")
    if minutes or not pieces:
        pieces.append(f"{minutes}分钟")
    return "".join(pieces)


def direction_label(value):
    return {"LONG": "多头", "SHORT": "空头", "unknown": "方向未知"}[value]


def events_for(cycle, events):
    cycle_id = cycle["cycle_id"]
    result = [e for e in events if e["cycle_id"] == cycle_id]
    return sorted(result, key=lambda e: (number(e["time_ms"]), str(e["event_id"])))


def read_market(symbol):
    path = DATA / "market" / f"{symbol}_1m.json.gz"
    if not path.exists():
        raise FileNotFoundError(f"No 1m market file for {symbol}: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        raw = json.load(f)
    # Binance futures klines: close price at index 4, close time at index 6.
    return [(int(row[6]), number(row[4])) for row in raw]


def market_window(candles, cycle):
    start, end = int(cycle["start_ms"]), int(cycle["end_ms"])
    duration = end - start
    pad = max(30 * 60_000, min(12 * 60 * 60_000, int(duration * 0.2)))
    if start < candles[0][0] or end > candles[-1][0]:
        raise ValueError(f"1m market coverage does not contain complete cycle {cycle['cycle_id']}")
    left, right = max(candles[0][0], start - pad), min(candles[-1][0], end + pad)
    window = [point for point in candles if left <= point[0] <= right]
    if not window:
        raise ValueError(f"No 1m candles intersect cycle {cycle['cycle_id']}")
    return left, right, window


def sample_points(points, budget=PRICE_SAMPLE_BUDGET):
    if len(points) <= budget:
        return points
    stride = math.ceil((len(points) - 1) / (budget - 1))
    sampled = points[::stride]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def fmt_price(value):
    if value is None:
        return "—"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:,.2f}"
    if magnitude >= 1:
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return f"{value:.8f}".rstrip("0").rstrip(".")


def action_label(value):
    labels = {"open": "开仓", "add": "加仓", "reduce": "减仓", "close": "平仓", "unknown": "未分类"}
    return labels[value]


def action_color(value):
    colors = {"open": "#16804a", "add": "#2468c5", "reduce": "#ca7b12", "close": "#c43f40", "unknown": "#596579"}
    return colors[value]


def order_type_label(value):
    if value is None:
        return "—"
    return {"MARKET": "市价", "LIMIT": "限价"}[value]


def svg_chart(cycle, events, candles):
    left, right, candles = market_window(candles, cycle)
    points = sample_points(candles)
    event_prices = [(int(e["time_ms"]), number(e["price"])) for e in events if number(e["price"]) is not None]
    price_values = [price for _, price in points] + [price for _, price in event_prices]
    low, high = min(price_values), max(price_values)
    margin = max((high - low) * 0.06, abs(high) * 0.002, 1e-12)
    low -= margin
    high += margin

    width, height = 1160, 500
    x0, x1, y0, y1 = 90, width - 32, 44, height - 68
    sx = lambda t: x0 + (t - left) / max(1, right - left) * (x1 - x0)
    sy = lambda p: y1 - (p - low) / max(1e-15, high - low) * (y1 - y0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(cycle["symbol"])} price chart">',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#263244}.axis{font-size:12px;fill:#64748b}.grid{stroke:#e5eaf0;stroke-width:1}.price{fill:none;stroke:#334b6c;stroke-width:1.65;stroke-linejoin:round;stroke-linecap:round}.window{fill:#e5f3ed}.context{fill:#f5f7fa}.event-label{font-size:10px;font-weight:700;fill:#fff;text-anchor:middle;dominant-baseline:central}</style>',
        f'<rect x="{x0}" y="{y0}" width="{sx(int(cycle["start_ms"])) - x0:.1f}" height="{y1-y0}" class="context"/>',
        f'<rect x="{sx(int(cycle["start_ms"])):.1f}" y="{y0}" width="{max(0, sx(int(cycle["end_ms"]))-sx(int(cycle["start_ms"]))):.1f}" height="{y1-y0}" class="window"/>',
        f'<rect x="{sx(int(cycle["end_ms"])):.1f}" y="{y0}" width="{x1-sx(int(cycle["end_ms"])):.1f}" height="{y1-y0}" class="context"/>',
    ]
    for i in range(5):
        value = low + (high - low) * i / 4
        y = sy(value)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{x0-10}" y="{y+4:.1f}" class="axis" text-anchor="end">{html.escape(fmt_price(value))}</text>')
    for i in range(6):
        t = int(left + (right - left) * i / 5)
        x = sx(t)
        parts.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{y1+24}" class="axis" text-anchor="middle">{bj_time(t, False)}</text>')

    d = " ".join(("M" if i == 0 else "L") + f"{sx(t):.1f},{sy(p):.1f}" for i, (t, p) in enumerate(points))
    parts.append(f'<path d="{d}" class="price"/>')

    start_x, end_x = sx(int(cycle["start_ms"])), sx(int(cycle["end_ms"]))
    parts.append(f'<line x1="{start_x:.1f}" y1="{y0}" x2="{start_x:.1f}" y2="{y1}" stroke="#16804a" stroke-dasharray="4 4"/>')
    parts.append(f'<line x1="{end_x:.1f}" y1="{y0}" x2="{end_x:.1f}" y2="{y1}" stroke="#c43f40" stroke-dasharray="4 4"/>')
    for idx, event in enumerate(events, start=1):
        t, price = number(event["time_ms"]), number(event["price"])
        if price is None or t < left or t > right:
            continue
        x, y, color = sx(t), sy(price), action_color(event["action"])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{color}" stroke="#fff" stroke-width="1.6"/>')
        parts.append(f'<text x="{x:.1f}" y="{y:.1f}" class="event-label">{idx}</text>')
    parts.append(f'<text x="{x0+4}" y="{y0+17}" font-size="11" fill="#617286">灰底：持仓前后行情　浅绿：周期持仓区间</text>')
    parts.append('</svg>')
    return "\n".join(parts), len(candles), len(points)


def op_table(events):
    if not events:
        raise ValueError("A selected complete cycle has no operation events")
    rows = []
    for idx, event in enumerate(events, start=1):
        event_ms = number(event["time_ms"])
        time = bj_time(int(event_ms))
        action = action_label(event["action"])
        color = action_color(event["action"])
        qty = event["qty"]
        before, after = event["position_before"], event["position_after"]
        rows.append(
            "<tr>"
            f"<td>{idx}</td><td>{html.escape(time)}</td>"
            f'<td><span class="action" style="--action-color:{color}">{html.escape(action)}</span></td>'
            f"<td>{html.escape(fmt_price(number(event['price'])))}</td>"
            f"<td>{html.escape(str(qty if qty is not None else '—'))}</td>"
            f"<td>{html.escape(str(before if before is not None else '—'))} → {html.escape(str(after if after is not None else '—'))}</td>"
            f"<td>{html.escape(order_type_label(event['order_type']))}</td>"
            "</tr>"
        )
    return '<div class="table-wrap"><table><thead><tr><th>#</th><th>时间（北京时间）</th><th>操作</th><th>价格</th><th>数量</th><th>仓位变化</th><th>委托类型</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>"


def build():
    cycles, events = load_inputs()
    full = complete_cycles(cycles)
    selected = select_examples(full)
    CHARTS.mkdir(parents=True, exist_ok=True)

    event_map = {}
    for event in events:
        event_map.setdefault(str(event["cycle_id"]), []).append(event)
    for cycle in full:
        if str(cycle["cycle_id"]) not in event_map:
            raise ValueError(f"Complete cycle {cycle['cycle_id']} has no operation events")

    cards = []
    selection = []
    market_cache = {}
    for index, item in enumerate(selected, start=1):
        cycle = item["cycle"]
        cycle_id = str(cycle["cycle_id"])
        ops = sorted(event_map[cycle_id], key=lambda e: (number(e["time_ms"]), str(e["event_id"])))
        if cycle["symbol"] not in market_cache:
            market_cache[cycle["symbol"]] = read_market(cycle["symbol"])
        svg, candle_count, plotted_count = svg_chart(cycle, ops, market_cache[cycle["symbol"]])
        filename = f"{index:02d}_{slug(cycle_id)}_{slug(cycle['symbol'])}.svg"
        (CHARTS / filename).write_text(svg, encoding="utf-8")
        categories = "、".join(item["categories"])
        reasons = "；".join(item["reasons"])
        duration = duration_ms(cycle)
        pnl = number(cycle["realized_pnl"])
        direction = direction_label(cycle["direction"])
        title = f"{cycle['symbol']} · {direction}"
        selection.append({
            "rank": index,
            "cycle_id": cycle_id,
            "symbol": cycle["symbol"],
            "direction": cycle["direction"],
            "status": cycle["status"],
            "start_ms": int(cycle["start_ms"]),
            "end_ms": int(cycle["end_ms"]),
            "duration_ms": int(duration),
            "add_count": cycle["add_count"],
            "reduce_count": cycle["reduce_count"],
            "official_closing_pnl": cycle["realized_pnl"],
            "categories": item["categories"],
            "reasons": item["reasons"],
            "chart": f"charts/{filename}",
            "event_count": len(ops),
            "market_candles_in_window": candle_count,
            "market_points_plotted": plotted_count,
        })
        cards.append(
            f'<article class="card" data-search="{html.escape((title+" "+categories+" "+reasons).lower(), quote=True)}">'
            f'<div class="card-head"><div><p class="eyebrow">样例 {index} · {html.escape(categories)}</p><h2>{html.escape(title)}</h2></div>'
            f'<span class="status">完整周期</span></div>'
            f'<p class="reason">选择理由：{html.escape(reasons)}</p>'
            f'<div class="meta"><span>持仓：{html.escape(duration_label(duration))}</span>'
            f'<span>加仓 {html.escape(str(cycle["add_count"]))} 次</span>'
            f'<span>减仓 {html.escape(str(cycle["reduce_count"]))} 次</span>'
            f'<span>官方仓位盈亏（closingPnl）：{html.escape(cents(pnl))} USDT</span></div>'
            f'<p class="time">北京时间：{bj_time(int(cycle["start_ms"]))} — {bj_time(int(cycle["end_ms"]))}</p>'
            f'<details open><summary>查看价格图</summary><p class="chart-link"><a href="charts/{html.escape(filename, quote=True)}" target="_blank" rel="noopener">打开独立大图 ↗</a></p><img class="chart" src="charts/{html.escape(filename, quote=True)}" alt="{html.escape(title)} 周期价格图"></details>'
            f'<details><summary>操作记录（{len(ops)} 条，标号与图中对应）</summary>{op_table(ops)}</details>'
            '</article>'
        )

    selection_doc = {
        "source_cycles": "analysis/results/leader_cycles.json",
        "source_events": "analysis/results/leader_events.json",
        "eligible_complete_cycle_count": len(full),
        "selected_count": len(selection),
        "selection_note": "只使用status=complete的完整锚定周期。按add_count、官方closingPnl及持仓时长极值选例，类别重叠时合并；多头样例在LONG周期中选持仓时长最接近全体中位值者。不足六例时按closingPnl与全体中位值的距离补充，持仓时长仅用于打破并列。样例不代表总体胜率或跟单账户净收益。",
        "price_line": {
            "source": "Binance USDT-margined futures 1m klines",
            "value_field": "close",
            "timestamp_field": "close_time_ms",
            "sampling": "time-ordered uniform stride, preserving the final point",
            "max_points_per_chart": PRICE_SAMPLE_BUDGET,
        },
        "operation_event_time": "leader_events.time_ms (orderUpdateTime)",
        "items": selection,
    }
    (RESULTS / "chart_selection.json").write_text(json.dumps(selection_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    html_doc = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>2026大黑马｜典型交易周期图</title>
<style>
:root{{color-scheme:light;--ink:#182435;--muted:#64748b;--line:#e4e9ef;--bg:#f4f6f9;--green:#16804a}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}}
main{{max-width:1200px;margin:0 auto;padding:34px 22px 70px}}header{{margin-bottom:24px}}h1{{font-size:30px;line-height:1.2;margin:0 0 12px}}h2{{font-size:21px;margin:0}}p{{margin:8px 0}}.intro{{color:#405065;max-width:900px}}.notice{{background:#fff7e6;border:1px solid #f0d799;border-radius:10px;padding:13px 16px;margin:18px 0;color:#684d12}}.toolbar{{position:sticky;top:0;background:rgba(244,246,249,.96);padding:10px 0 14px;display:flex;gap:12px;z-index:2}}input{{width:min(520px,100%);padding:11px 13px;border:1px solid #ccd5df;border-radius:8px;font:inherit}}.count{{align-self:center;color:var(--muted)}}.card{{background:#fff;border:1px solid var(--line);border-radius:14px;margin:16px 0;padding:20px;box-shadow:0 2px 8px #1824350a}}.card-head{{display:flex;justify-content:space-between;gap:16px;align-items:center}}.eyebrow{{color:#476d9c;font-size:12px;font-weight:700;letter-spacing:.04em;margin:0 0 4px}}.status{{color:#13663e;background:#e8f5ed;padding:5px 9px;border-radius:999px;font-size:12px;white-space:nowrap}}.reason{{color:#31445d;margin-top:12px}}.meta{{display:flex;flex-wrap:wrap;gap:7px;margin:10px 0}}.meta span{{font-size:13px;color:#405065;background:#f1f4f7;border-radius:6px;padding:4px 8px}}.time{{color:var(--muted);font-size:13px}}details{{margin-top:14px;border-top:1px solid #edf0f3;padding-top:10px}}summary{{cursor:pointer;font-weight:650}}.chart{{display:block;width:100%;height:auto;margin:12px 0 4px}}.chart-link{{font-size:13px;text-align:right;margin:2px 0}}.table-wrap{{overflow:auto;margin-top:8px}}table{{width:100%;border-collapse:collapse;font-size:13px;white-space:nowrap}}th,td{{padding:8px 9px;border-bottom:1px solid #edf0f3;text-align:left}}th{{color:#637287;background:#f8fafb;font-weight:600}}.action{{color:var(--action-color);font-weight:700}}footer{{color:var(--muted);font-size:13px;margin-top:24px}}
</style></head><body><main>
<header><h1>2026大黑马｜典型交易周期</h1>
<p class="intro">本页从 {len(full)} 个完整带单员周期中选取 {len(selection)} 个样例，展示其操作与对应合约行情，供逐例检查操作节奏、持仓时长和盈亏分布。</p>
<div class="notice"><strong>阅读说明：</strong>这些是按加仓次数、公开仓位盈亏和持仓时长挑选的案例，不代表总体胜率，也不代表跟单账户净收益。价格线来自币安1分钟K线收盘价，点位标在每分钟收盘时刻并按时间等间隔抽样；操作标记与表格来自带单员重建的实际操作事件，价格取事件记录，不从价格线上推算。操作时间使用订单更新时间，不等于限价委托的策略决策时点。浅绿区域为完整锚定持仓周期，灰底为前后行情上下文。所有时间均为北京时间。图中盈亏为公开仓位历史记录的closingPnl。</div>
</header>
<div class="toolbar"><input id="search" type="search" placeholder="按合约、方向或选择理由筛选"><span class="count" id="count">{len(selection)} 个样例</span></div>
{''.join(cards)}
<footer>图中盈亏是带单员公开仓位历史 closingPnl；仓位变化或事件价格缺失时按原数据标为“—”。</footer>
</main><script>
const search=document.getElementById('search'),cards=[...document.querySelectorAll('.card')],count=document.getElementById('count');
search.addEventListener('input',()=>{{const q=search.value.trim().toLowerCase();let visible=0;for(const c of cards){{const show=!q||c.dataset.search.includes(q);c.hidden=!show;if(show)visible++}}count.textContent=visible+' 个样例'}});
</script></body></html>'''
    (RESULTS / "trade_charts.html").write_text(html_doc, encoding="utf-8")
    print(f"Wrote {len(selection)} charts from {len(full)} complete cycles to {RESULTS}")


if __name__ == "__main__":
    build()
