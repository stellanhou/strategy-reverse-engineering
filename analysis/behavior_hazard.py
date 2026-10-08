"""Bounded observable-event pilot. No exchange/account access or private intent.

The forecasting API receives only market values and a supplied prior state.
Autonomous rollout constructs that state from its own executed virtual events.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math
import warnings

import numpy as np
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

M = 60000
ACTIONS = ('open', 'add', 'reduce', 'close')
MARKET = ('ret5', 'ret15', 'ret60', 'qv5_vs_prev60_mean', 'volatility60',
          'range15', 'btc_relative15', 'btc_relative60')
POSITION = ('age_minutes', 'minutes_last_action', 'add_count', 'mark_return', 'position_multiple')
warnings.filterwarnings('error', category=ConvergenceWarning)


def stable_key(symbol, action, now):
    return hashlib.sha256(f'{symbol}|{action}|{now}'.encode()).hexdigest()


def corrected_sample(y, limit=20000, seed=20261003):
    """All events plus sampled negatives, with inverse inclusion weights.

    Sum of weights reproduces each natural class count. No class balancing.
    """
    y = np.asarray(y, dtype=int)
    positive, negative = np.flatnonzero(y), np.flatnonzero(y == 0)
    chosen = np.random.default_rng(seed).choice(negative, min(len(negative), limit), replace=False)
    rows = np.sort(np.r_[positive, chosen])
    weights = np.ones(len(rows))
    if len(chosen):
        weights[y[rows] == 0] = len(negative) / len(chosen)
    return rows, weights


def offset_calibration(probability, y):
    """Slope=1, one intercept; temporal calibration only, natural frequency."""
    p = np.clip(np.asarray(probability, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, int)
    target = (y.sum() + .5) / (len(y) + 1)
    z = logit(p)
    low, high = -50., 50.
    for _ in range(80):
        mid = (low + high) / 2
        if expit(z + mid).mean() < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def probability_card(y, p):
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    if not len(y):
        return dict(minutes=0, events=0)
    n = int(y.sum())
    bins = np.array([0, .0001, .001, .01, .05, .2, 1.000001])
    at = np.digitize(p, bins) - 1
    reliability = [dict(lower=float(bins[i]), upper=float(bins[i + 1]),
                        minutes=int((at == i).sum()),
                        predicted=float(p[at == i].mean()), observed=float(y[at == i].mean()))
                   for i in range(len(bins) - 1) if (at == i).any()]
    return dict(minutes=len(y), events=n, expected_events=float(p.sum()),
                event_mass_ratio=float(p.sum() / n) if n else None,
                log_loss=float(-(y * np.log(p) + (1-y) * np.log1p(-p)).mean()),
                brier=float(np.square(p-y).mean()),
                average_precision=float(average_precision_score(y, p)) if n else None,
                calibration=reliability)


class BinaryHead:
    def __init__(self, constant=False):
        self.constant = constant
        self.offset = 0.

    def fit(self, x, y):
        y = np.asarray(y, int)
        self.prior = float((y.sum() + .5) / (len(y) + 1))
        self.details = dict(natural_minutes=len(y), natural_events=int(y.sum()))
        self.learned = not self.constant and len(np.unique(y)) == 2
        if self.learned:
            rows, weights = corrected_sample(y)
            self.scaler = StandardScaler().fit(x[rows], sample_weight=weights)
            self.model = LogisticRegression(C=.1, class_weight=None, solver='liblinear',
                                            max_iter=1000, random_state=20261003)
            self.model.fit(self.scaler.transform(x[rows]), y[rows], sample_weight=weights)
            self.details.update(sampled_minutes=len(rows), weight_sum=float(weights.sum()),
                                event_weight_sum=float(weights[y[rows] == 1].sum()),
                                non_event_weight_sum=float(weights[y[rows] == 0].sum()),
                                intercept=self.model.intercept_.tolist(), coefficients=self.model.coef_.tolist(),
                                scaler_mean=self.scaler.mean_.tolist(), scaler_scale=self.scaler.scale_.tolist())
        return self

    def raw(self, x):
        return (self.model.predict_proba(self.scaler.transform(x))[:, 1] if self.learned
                else np.full(len(x), self.prior))

    def calibrate(self, x, y):
        self.offset = offset_calibration(self.raw(x), y)
        self.details.update(calibration_offset=self.offset, calibration_minutes=len(y),
                            calibration_events=int(np.asarray(y).sum()))

    def predict(self, x):
        return expit(logit(np.clip(self.raw(x), 1e-12, 1 - 1e-12)) + self.offset)


class Forecast:
    """Competing SHORT operations: P(action)=P(any)*P(action|any)."""
    def __init__(self, variant):
        if variant not in ('state_clock', 'market_state'):
            raise ValueError(variant)
        self.variant = variant

    def inputs(self, market, state, held):
        if not held:
            return market if self.variant == 'market_state' else np.zeros((len(market), 1))
        return np.column_stack([market, state]) if self.variant == 'market_state' else state

    def fit(self, market, state, labels, flat, held):
        self.open = BinaryHead(constant=self.variant == 'state_clock').fit(
            self.inputs(market[flat], state[flat], False), labels[flat] == 1)
        hx = self.inputs(market[held], state[held], True)
        hy = labels[held]
        self.hazard = BinaryHead().fit(hx, hy > 1)
        positive = hy > 1
        self.action_classes = np.unique(hy[positive])
        self.action_scaler = StandardScaler().fit(hx[positive])
        if len(self.action_classes) > 1:
            self.action_model = LogisticRegression(C=.1, class_weight=None, solver='lbfgs',
                                                    max_iter=1000, random_state=20261003)
            self.action_model.fit(self.action_scaler.transform(hx[positive]), hy[positive])
        self.details = dict(variant=self.variant, open=self.open.details, hazard=self.hazard.details,
                            action_events=int(positive.sum()), action_classes=self.action_classes.tolist())
        return self

    def calibrate(self, market, state, labels, flat, held):
        self.open.calibrate(self.inputs(market[flat], state[flat], False), labels[flat] == 1)
        self.hazard.calibrate(self.inputs(market[held], state[held], True), labels[held] > 1)

    def predict(self, market, state, is_held):
        market, state = np.asarray(market), np.asarray(state)
        is_held = np.asarray(is_held, bool)
        out = np.zeros((len(market), 4))
        flat = ~is_held
        if flat.any():
            out[flat, 0] = self.open.predict(self.inputs(market[flat], state[flat], False))
        if is_held.any():
            x = self.inputs(market[is_held], state[is_held], True)
            h = self.hazard.predict(x)
            conditional = (self.action_model.predict_proba(self.action_scaler.transform(x))
                           if len(self.action_classes) > 1 else np.ones((len(x), 1)))
            for j, code in enumerate(self.action_classes):
                out[is_held, int(code)-1] = h * conditional[:, j]
        return out


@dataclass
class Position:
    qty: float = 0.
    cost: float = 0.
    first_qty: float = 0.
    opened: int = 0
    last: int = 0
    adds: int = 0

    def features(self, now, mark):
        if self.qty == 0:
            return np.full(5, np.nan)
        return np.array([max(0, (now-self.opened)//M), max(0, (now-self.last)//M),
                         self.adds, 1-mark/self.cost, self.qty/self.first_qty], float)

    def apply(self, action, quantity, price, now):
        if quantity <= 0 or price <= 0 or not math.isfinite(quantity + price):
            raise ValueError('non-positive/non-finite execution')
        if action == 'open':
            if self.qty:
                raise ValueError('open while held')
            self.qty, self.cost, self.first_qty = quantity, price, quantity
            self.opened, self.last, self.adds = now, now, 0
        elif action == 'add':
            if not self.qty:
                raise ValueError('add while flat')
            self.cost = (self.cost*self.qty + price*quantity)/(self.qty+quantity)
            self.qty += quantity
            self.adds += 1
            self.last = now
        elif action in ('reduce', 'close'):
            if not self.qty or quantity > self.qty * (1+1e-10):
                raise ValueError('oversized/flat exit')
            if action == 'close' and not math.isclose(quantity, self.qty, rel_tol=1e-9):
                raise ValueError('partial close')
            self.qty = max(0., self.qty-quantity)
            self.last = now
            if self.qty < 1e-12:
                self.qty, self.cost, self.first_qty, self.adds = 0., 0., 0., 0
        else:
            raise ValueError(action)


class SizeModel:
    def __init__(self, samples):
        self.distribution = {}
        for action in ('open', 'add', 'reduce'):
            values = np.asarray(samples[action], float)
            if not len(values) or not (np.isfinite(values) & (values > 0)).all():
                raise ValueError('invalid quantity samples: '+action)
            self.distribution[action] = dict(n=len(values), q25=float(np.quantile(values, .25)),
                                            median=float(np.median(values)), q75=float(np.quantile(values, .75)))

    def quantity(self, action, state, visible_price):
        if action == 'open':
            return self.distribution['open']['median']/visible_price
        if action == 'add':
            return self.distribution['add']['median']*state.first_qty
        if action == 'reduce':
            return min(self.distribution['reduce']['median'], 1.)*state.qty
        if action == 'close':
            return state.qty
        raise ValueError(action)


class OnlineDecoder:
    """Chronological budget, contemporaneous ranking, deterministic ties.

    No day-end Top-K; no 30m cooldown or cap on additions. One pending virtual
    operation per symbol is an execution restriction, explicitly logged.
    """
    def __init__(self, thresholds, budgets):
        self.thresholds, self.budgets = np.asarray(thresholds), dict(zip(ACTIONS, budgets))
        self.used = Counter()
        self.rejections = Counter()

    def choose(self, now, symbols, probabilities, held, pending):
        candidates = []
        for i, symbol in enumerate(symbols):
            valid = range(1, 4) if held[i] else (0,)
            eligible = [j for j in valid if probabilities[i, j] >= self.thresholds[j]]
            if not eligible:
                self.rejections['below_threshold'] += 1
                continue
            j = max(eligible, key=lambda j: (probabilities[i, j]/max(self.thresholds[j], 1e-12), -j))
            if symbol in pending:
                self.rejections['pending'] += 1
                continue
            candidates.append((i, j))
        candidates.sort(key=lambda ij: (-probabilities[ij], stable_key(symbols[ij[0]], ACTIONS[ij[1]], now)))
        chosen = []
        day = now//(1440*M)
        for i, j in candidates:
            key = (day, ACTIONS[j])
            if self.used[key] >= self.budgets[ACTIONS[j]]:
                self.rejections['daily_budget_'+ACTIONS[j]] += 1
                continue
            self.used[key] += 1
            chosen.append((i, ACTIONS[j]))
        return chosen


def causal_match(predictions, actual, tolerance=M):
    """One-to-one interval matching: actual cannot precede prediction issue.

    With a fixed execution delay, interval bounds are monotonic in issue time.
    Earliest eligible actual gives maximal cardinality within each group.
    """
    pp, aa = defaultdict(list), defaultdict(list)
    for p in predictions:
        pp[(p['symbol'], p['action'])].append(p)
    for a in actual:
        aa[(a['symbol'], a['action'])].append(a)
    errors, matched = [], []
    for key in pp:
        gold = sorted(aa[key], key=lambda e: e['time_ms'])
        cursor = 0
        for p in sorted(pp[key], key=lambda e: (e['time_ms'], e['issue_ms'])):
            low = max(p['issue_ms'], p['time_ms']-tolerance)
            high = p['time_ms']+tolerance
            while cursor < len(gold) and gold[cursor]['time_ms'] < low:
                cursor += 1
            if cursor < len(gold) and gold[cursor]['time_ms'] <= high:
                a = gold[cursor]
                errors.append((p['time_ms']-a['time_ms'])/M)
                matched.append((p, a))
                cursor += 1
    tp, fp, fn = len(matched), len(predictions)-len(matched), len(actual)-len(matched)
    return dict(tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0., f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                early_matches=sum(v < 0 for v in errors), late_matches=sum(v > 0 for v in errors),
                mean_signed_minutes=float(np.mean(errors)) if errors else None,
                tolerance_ms=tolerance, actual_cannot_precede_issue=True)


def path_card(predicted_qty, actual_qty, marks, scale):
    zhat, z = np.asarray(predicted_qty)*marks/scale, np.asarray(actual_qty)*marks/scale
    denom = np.abs(z).sum()
    union = ((zhat > 0) | (z > 0)).sum()
    return dict(minutes=len(z), normalized_absolute_error=float(np.abs(zhat-z).sum()/denom) if denom else None,
                mean_absolute_scaled_exposure_error=float(np.abs(zhat-z).mean()) if len(z) else None,
                held_overlap_iou=float(((zhat > 0) & (z > 0)).sum()/union) if union else 1.,
                predicted_peak_scaled_exposure=float(np.max(zhat)) if len(z) else 0.,
                actual_peak_scaled_exposure=float(np.max(z)) if len(z) else 0.,
                prediction_and_truth_state_intersection_filter=False)
