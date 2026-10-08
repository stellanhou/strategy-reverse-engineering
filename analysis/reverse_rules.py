#!/usr/bin/env python3
"""Deterministic rule candidates for the exploratory leader reconstruction.

The public Rule.step() interface is the autonomous state-machine contract.
The separate one_step_decisions() path accepts an observed position state for
local event explanation and must never be called by autonomous replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
MINUTE_MS = 60_000
SHORT = -1

RET15_THRESHOLDS = (0.01, 0.02, 0.03, 0.05, 0.08)
RELATIVE60_THRESHOLDS = (0.01, 0.02, 0.03, 0.05, 0.08)
PROFIT_THRESHOLDS = (0.005, 0.01)
AGE_BASELINE_MINUTES = (30, 60, 120, 240, 480)
TIME_FREQUENCY_MINUTES = (60, 120, 240, 360, 720)
ADVERSE_ADD_THRESHOLD = 0.02
COOLDOWN_MINUTES = 30
MAX_ADDS = 2


@dataclass(frozen=True)
class RuleConfig:
    """All tunable values for one registered deterministic configuration."""

    config_id: str
    family: str
    entry_threshold: float | None = None
    profit_exit_threshold: float | None = None
    max_hold_minutes: int | None = None
    schedule_minutes: int | None = None
    add_adverse_threshold: float = ADVERSE_ADD_THRESHOLD
    cooldown_minutes: int = COOLDOWN_MINUTES
    max_adds: int = MAX_ADDS
    incremental_add_qty_first_qty: float = 1.0
    allow_adds: bool = True

    def __post_init__(self) -> None:
        families = {"ret15_surge_short", "ret15_fixed_age", "relative60_surge_short", "time_frequency"}
        if self.family not in families:
            raise ValueError(f"Unsupported rule family: {self.family}")
        if self.family in {"ret15_surge_short", "ret15_fixed_age", "relative60_surge_short"}:
            if self.entry_threshold is None or not isfinite(self.entry_threshold) or self.entry_threshold < 0:
                raise ValueError(f"{self.family} requires a finite non-negative entry threshold")
        if self.family in {"ret15_surge_short", "relative60_surge_short"}:
            if self.profit_exit_threshold is None or not isfinite(self.profit_exit_threshold) or self.profit_exit_threshold <= 0:
                raise ValueError(f"{self.family} requires a positive profit-exit threshold")
        if self.family in {"ret15_fixed_age", "time_frequency"}:
            if self.max_hold_minutes is None or self.max_hold_minutes <= 0:
                raise ValueError(f"{self.family} requires a positive max_hold_minutes")
        if self.family == "time_frequency" and (self.schedule_minutes is None or self.schedule_minutes <= 0):
            raise ValueError("time_frequency requires a positive schedule_minutes")
        if not isfinite(self.add_adverse_threshold) or self.add_adverse_threshold <= 0:
            raise ValueError("add_adverse_threshold must be finite and positive")
        if self.cooldown_minutes < 0 or self.max_adds < 0:
            raise ValueError("cooldown_minutes and max_adds must be non-negative")
        if not isfinite(self.incremental_add_qty_first_qty) or self.incremental_add_qty_first_qty <= 0:
            raise ValueError("incremental_add_qty_first_qty must be finite and positive")

    def signal_value(self, features: dict[str, float]) -> float | None:
        if self.family in {"ret15_surge_short", "ret15_fixed_age"}:
            value = features["ret15"]
            return value if isfinite(value) else None
        if self.family == "relative60_surge_short":
            value = features["btc_relative60"]
            return value if isfinite(value) else None
        return None


@dataclass
class RuleState:
    """Persisted autonomous state. One instance belongs to one symbol."""

    direction: int = 0
    quantity: float = 0.0
    weighted_cost: float = 0.0
    first_quantity: float = 0.0
    add_count: int = 0
    opened_at_ms: int | None = None
    last_action_ms: int | None = None
    previous_signal_above: bool | None = None
    previous_mark_return: float | None = None
    last_seen_ms: int | None = None
    cycle_key: int | str | None = None

    @property
    def is_flat(self) -> bool:
        return self.direction == 0 and self.quantity == 0.0


@dataclass(frozen=True)
class RuleBar:
    """Information available at a decision minute.

    ``mark_price`` is the most recently completed close available at time_ms.
    ``execution_price`` is the next observable open used to fill actions.
    """

    time_ms: int
    symbol: str
    features: dict[str, float]
    mark_price: float
    execution_price: float


@dataclass(frozen=True)
class RuleAction:
    time_ms: int
    symbol: str
    direction: int
    action: str
    quantity: float
    reason: str
    mark_price: float
    execution_price: float
    weighted_cost_before: float | None
    state_before_qty: float

    @property
    def fill_time_ms(self) -> int:
        """The following minute's open represented by RuleBar.execution_price."""
        return self.time_ms + MINUTE_MS


@dataclass
class RuleSignalMemory:
    """Prediction-only history for one-step scoring; never a position ledger."""

    previous_entry_above: bool | None = None
    previous_mark_return: float | None = None
    previous_age_minutes: float | None = None
    previous_direction: int | None = None
    previous_weighted_cost: float | None = None
    previous_add_count: int | None = None
    previous_cycle_key: int | str | None = None
    last_signal_ms: int | None = None
    last_seen_ms: int | None = None


@dataclass
class Rule:
    """One deterministic rule and its independent per-symbol state."""

    config: RuleConfig
    state: RuleState = field(default_factory=RuleState)

    def step(self, bar: RuleBar) -> list[RuleAction]:
        """Advance autonomous state using this bar; never reads leader labels.

        Calls must be in strictly increasing time order. Missing bars do not
        synthesize prices. An open position carries across a data gap, while a
        threshold-crossing signal cannot bridge that gap.
        """
        if self.state.last_seen_ms is not None and bar.time_ms <= self.state.last_seen_ms:
            raise ValueError("Rule.step requires strictly increasing decision times")
        if bar.time_ms % MINUTE_MS != 0:
            raise ValueError("Rule.step decision times must be UTC minute starts")
        if self.state.direction not in {0, SHORT}:
            raise ValueError(f"Autonomous SHORT rule received unsupported state direction: {self.state.direction}")
        if self.state.direction == 0 and (
            self.state.quantity != 0 or self.state.weighted_cost != 0 or self.state.first_quantity != 0
        ):
            raise ValueError("Flat RuleState must have zero quantity, weighted cost, and first quantity")
        if self.state.direction == SHORT and (
            self.state.quantity <= 0 or self.state.weighted_cost <= 0 or self.state.first_quantity <= 0
            or self.state.opened_at_ms is None or self.state.last_action_ms is None
        ):
            raise ValueError("Open SHORT RuleState must have positive size/cost and action timestamps")
        if self.state.add_count < 0 or self.state.add_count > self.config.max_adds:
            raise ValueError("RuleState add_count exceeds the registered rule limit")
        if self.state.last_seen_ms is not None and bar.time_ms != self.state.last_seen_ms + MINUTE_MS:
            self.state.previous_signal_above = None
            self.state.previous_mark_return = None

        actions: list[RuleAction] = []
        px = bar.execution_price
        mark = bar.mark_price
        if not isfinite(px) or not isfinite(mark) or px <= 0 or mark <= 0:
            raise ValueError(f"Non-positive mark or execution price for {bar.symbol} at {bar.time_ms}")
        cfg = self.config
        st = self.state
        acted_this_bar = False
        fill_time_ms = bar.time_ms + MINUTE_MS

        if st.direction == SHORT and st.quantity > 0:
            mark_return = st.direction * (mark / st.weighted_cost - 1.0)
            previous_mark_return = st.previous_mark_return
            mark_is_after_last_fill = st.last_action_ms is not None and bar.time_ms > st.last_action_ms
            age = ((bar.time_ms - st.opened_at_ms) / MINUTE_MS) if st.opened_at_ms is not None else 0.0
            close_reason: str | None = None
            if cfg.family in {"ret15_fixed_age", "time_frequency"} and cfg.max_hold_minutes is not None and age >= cfg.max_hold_minutes:
                close_reason = "fixed_age_exit"
            elif (
                cfg.profit_exit_threshold is not None
                and mark_is_after_last_fill
                and previous_mark_return is not None
                and previous_mark_return < cfg.profit_exit_threshold <= mark_return
            ):
                close_reason = "profit_threshold_close"
            if close_reason is not None:
                actions.append(RuleAction(
                    bar.time_ms, bar.symbol, SHORT, "close", st.quantity, close_reason,
                    mark, px, st.weighted_cost, st.quantity,
                ))
                st.quantity = 0.0
                st.direction = 0
                st.weighted_cost = 0.0
                st.first_quantity = 0.0
                st.add_count = 0
                st.opened_at_ms = None
                st.last_action_ms = fill_time_ms
                st.previous_mark_return = None
                acted_this_bar = True
            elif (
                previous_mark_return is not None
                and mark_is_after_last_fill
                and previous_mark_return > -cfg.add_adverse_threshold >= mark_return
                and cfg.allow_adds
                and st.add_count < cfg.max_adds
                and st.last_action_ms is not None
                and bar.time_ms - st.last_action_ms >= cfg.cooldown_minutes * MINUTE_MS
            ):
                add_qty = st.first_quantity * cfg.incremental_add_qty_first_qty
                old_qty = st.quantity
                cost_before = st.weighted_cost
                st.weighted_cost = (old_qty * cost_before + add_qty * px) / (old_qty + add_qty)
                st.quantity = old_qty + add_qty
                st.add_count += 1
                st.last_action_ms = fill_time_ms
                st.previous_mark_return = st.direction * (px / st.weighted_cost - 1.0)
                acted_this_bar = True
                actions.append(RuleAction(
                    bar.time_ms, bar.symbol, SHORT, "add", add_qty, "adverse_mark_return_and_cooldown",
                    mark, px, cost_before, old_qty,
                ))

        if st.is_flat and not acted_this_bar:
            signal = self._entry_signal(bar)
            if signal:
                if st.last_action_ms is None or bar.time_ms - st.last_action_ms >= cfg.cooldown_minutes * MINUTE_MS:
                    qty = 1.0
                    st.direction = SHORT
                    st.quantity = qty
                    st.weighted_cost = px
                    st.first_quantity = qty
                    st.add_count = 0
                    st.opened_at_ms = fill_time_ms
                    st.last_action_ms = fill_time_ms
                    st.previous_mark_return = st.direction * (px / st.weighted_cost - 1.0)
                    acted_this_bar = True
                    actions.append(RuleAction(
                        bar.time_ms, bar.symbol, SHORT, "open", qty, "entry_signal_crossing",
                        mark, px, None, 0.0,
                    ))

        self.state.previous_signal_above = self._signal_above(bar)
        if (
            st.direction == SHORT and st.quantity > 0 and not acted_this_bar
            and st.last_action_ms is not None and bar.time_ms > st.last_action_ms
        ):
            st.previous_mark_return = st.direction * (mark / st.weighted_cost - 1.0)
        self.state.last_seen_ms = bar.time_ms
        return actions

    def _signal_above(self, bar: RuleBar) -> bool | None:
        value = self.config.signal_value(bar.features)
        if value is None or self.config.entry_threshold is None:
            return None
        return bool(value >= self.config.entry_threshold)

    def _entry_signal(self, bar: RuleBar) -> bool:
        cfg = self.config
        if cfg.family == "time_frequency":
            if cfg.schedule_minutes is None:
                raise ValueError("time_frequency config requires schedule_minutes")
            return bar.time_ms % (cfg.schedule_minutes * MINUTE_MS) == 0
        above = self._signal_above(bar)
        if above is None:
            return False
        return above and self.state.previous_signal_above is False


def registered_configs() -> tuple[list[RuleConfig], list[RuleConfig], list[RuleConfig]]:
    """Return main ret15, baselines, and optional relative60 grids.

    The relative60 family stays preregistered but disabled until the train-only
    mechanism gate supports trying it. Volatility-normalized reversal is logged
    as not attempted in this round; it is outside the frozen 30-config budget.
    """
    main = [
        RuleConfig(
            config_id=f"ret15_{entry:.3f}_close_{profit:.3f}",
            family="ret15_surge_short",
            entry_threshold=entry,
            profit_exit_threshold=profit,
        )
        for entry in RET15_THRESHOLDS for profit in PROFIT_THRESHOLDS
    ]
    baseline = [
        RuleConfig(
            config_id=f"ret15_{entry:.3f}_fixed_age_{age}m",
            family="ret15_fixed_age",
            entry_threshold=entry,
            max_hold_minutes=age,
            allow_adds=False,
        )
        for entry, age in zip(RET15_THRESHOLDS, AGE_BASELINE_MINUTES)
    ] + [
        RuleConfig(
            config_id=f"time_frequency_{period}m_hold_120m",
            family="time_frequency",
            schedule_minutes=period,
            max_hold_minutes=120,
            allow_adds=False,
        )
        for period in TIME_FREQUENCY_MINUTES
    ]
    relative = [
        RuleConfig(
            config_id=f"relative60_{entry:.3f}_close_{profit:.3f}",
            family="relative60_surge_short",
            entry_threshold=entry,
            profit_exit_threshold=profit,
        )
        for entry in RELATIVE60_THRESHOLDS for profit in PROFIT_THRESHOLDS
    ]
    return main, baseline, relative


def one_step_decisions(
    config: RuleConfig,
    bar: RuleBar,
    observed_state: RuleState,
    memory: RuleSignalMemory,
) -> list[RuleAction]:
    """Predict next action from one observed state without changing it.

    This function is for one-step explanation only. It intentionally uses the
    supplied leader position and cannot be used by Rule.step() autonomous runs.
    The separate memory object holds only signal crossings and prediction
    cooldown, not observed or model-generated position state.
    """
    px = bar.execution_price
    mark = bar.mark_price
    if not isfinite(px) or not isfinite(mark) or px <= 0 or mark <= 0:
        raise ValueError(f"Non-positive mark or execution price for {bar.symbol} at {bar.time_ms}")
    if bar.time_ms % MINUTE_MS != 0:
        raise ValueError("one_step_decisions times must be UTC minute starts")
    if observed_state.direction not in {-1, 0, 1}:
        raise ValueError(f"Observed position has unsupported direction: {observed_state.direction}")
    if observed_state.direction == 0 and observed_state.quantity != 0:
        raise ValueError("Observed flat state must have zero quantity")
    if observed_state.direction == SHORT and (
        observed_state.quantity <= 0 or observed_state.weighted_cost <= 0 or observed_state.first_quantity <= 0
    ):
        raise ValueError("Observed SHORT state requires positive quantity, cost, and first quantity")
    if memory.last_seen_ms is not None and bar.time_ms <= memory.last_seen_ms:
        raise ValueError("one_step_decisions requires strictly increasing symbol times")
    if memory.last_seen_ms is not None and bar.time_ms != memory.last_seen_ms + MINUTE_MS:
        memory.previous_entry_above = None
        memory.previous_mark_return = None
        memory.previous_age_minutes = None
    same_observed_cycle = (
        observed_state.direction == SHORT
        and memory.previous_direction == SHORT
        and observed_state.cycle_key == memory.previous_cycle_key
    )
    if not same_observed_cycle:
        memory.previous_mark_return = None
        memory.previous_age_minutes = None
    elif (
        observed_state.weighted_cost != memory.previous_weighted_cost
        or observed_state.add_count != memory.previous_add_count
    ):
        memory.previous_mark_return = None

    actions: list[RuleAction] = []
    cooldown_times = [t for t in (observed_state.last_action_ms, memory.last_signal_ms) if t is not None]
    cooldown_ready = not cooldown_times or bar.time_ms - max(cooldown_times) >= config.cooldown_minutes * MINUTE_MS

    if observed_state.is_flat:
        if config.family == "time_frequency":
            fire = config.schedule_minutes is not None and bar.time_ms % (config.schedule_minutes * MINUTE_MS) == 0
        else:
            value = config.signal_value(bar.features)
            above = value is not None and config.entry_threshold is not None and value >= config.entry_threshold
            fire = bool(above and memory.previous_entry_above is False)
        if fire and cooldown_ready:
            actions.append(RuleAction(bar.time_ms, bar.symbol, SHORT, "open", 1.0,
                                      "single_step_entry_signal", mark, px, None, 0.0))
    elif observed_state.direction == SHORT and observed_state.weighted_cost > 0:
        mark_return = SHORT * (mark / observed_state.weighted_cost - 1.0)
        observed_age = (
            (bar.time_ms - observed_state.opened_at_ms) / MINUTE_MS
            if observed_state.opened_at_ms is not None else 0.0
        )
        age_crossed = (
            memory.previous_age_minutes is not None
            and config.max_hold_minutes is not None
            and memory.previous_age_minutes < config.max_hold_minutes <= observed_age
        )
        profit_crossed = (
            config.profit_exit_threshold is not None
            and memory.previous_mark_return is not None
            and memory.previous_mark_return < config.profit_exit_threshold <= mark_return
        )
        if (
            config.family in {"ret15_fixed_age", "time_frequency"}
            and config.max_hold_minutes is not None and age_crossed
        ):
            actions.append(RuleAction(bar.time_ms, bar.symbol, SHORT, "close", observed_state.quantity,
                                      "single_step_fixed_age_exit", mark, px,
                                      observed_state.weighted_cost, observed_state.quantity))
        elif profit_crossed:
            actions.append(RuleAction(bar.time_ms, bar.symbol, SHORT, "close", observed_state.quantity,
                                      "single_step_profit_threshold", mark, px,
                                      observed_state.weighted_cost, observed_state.quantity))
        elif (
            config.allow_adds
            and memory.previous_mark_return is not None
            and memory.previous_mark_return > -config.add_adverse_threshold >= mark_return
            and observed_state.add_count < config.max_adds and cooldown_ready
        ):
            qty = observed_state.first_quantity * config.incremental_add_qty_first_qty
            actions.append(RuleAction(bar.time_ms, bar.symbol, SHORT, "add", qty,
                                      "single_step_adverse_mark_return_and_cooldown", mark, px,
                                      observed_state.weighted_cost, observed_state.quantity))

    value = config.signal_value(bar.features) if config.family != "time_frequency" else None
    memory.previous_entry_above = (
        bool(value >= config.entry_threshold)
        if value is not None and config.entry_threshold is not None else None
    )
    if observed_state.direction == SHORT:
        memory.previous_mark_return = SHORT * (mark / observed_state.weighted_cost - 1.0)
        memory.previous_age_minutes = (
            (bar.time_ms - observed_state.opened_at_ms) / MINUTE_MS
            if observed_state.opened_at_ms is not None else 0.0
        )
    else:
        memory.previous_mark_return = None
        memory.previous_age_minutes = None
    memory.previous_direction = observed_state.direction
    memory.previous_weighted_cost = observed_state.weighted_cost
    memory.previous_add_count = observed_state.add_count
    memory.previous_cycle_key = observed_state.cycle_key
    if actions:
        memory.last_signal_ms = bar.time_ms + MINUTE_MS
    memory.last_seen_ms = bar.time_ms
    return actions
