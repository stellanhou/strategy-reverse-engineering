"""Causal public-price inputs, fit-only clipping and budgeted opening alerts."""
from collections import Counter

import numpy as np

from behavior_hazard import M, stable_key


def timing_inputs(close, n_symbols, n_minutes):
    """Use already-lagged symbol-major closes, with complete valid windows."""
    close = np.asarray(close, dtype=float).reshape(n_symbols, n_minutes)
    result = np.full((n_symbols, n_minutes, 3), np.nan)
    for column, width in enumerate((2, 4, 15)):
        if n_minutes < width:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(close, width, axis=1)
        valid = (np.isfinite(windows) & (windows > 0)).all(axis=2)
        base = windows.max(axis=2) if column == 2 else windows[:, :, 0]
        values = np.full(valid.shape, np.nan)
        np.divide(windows[:, :, -1], base, out=values, where=valid)
        values[valid] -= 1
        result[:, width - 1:, column] = values
    return result.reshape(n_symbols * n_minutes, 3), ['ret1', 'ret3', 'pullback15']


class FitClipper:
    """Column-wise bounds learned only from the supplied natural fit rows."""

    def __init__(self, quantiles=(.005, .995)):
        if len(quantiles) != 2 or not 0 <= quantiles[0] <= quantiles[1] <= 1:
            raise ValueError('two ordered quantiles in [0, 1] are required')
        self.quantiles = quantiles

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or not x.size or not np.isfinite(x).all():
            raise ValueError('fit rows must be a nonempty finite matrix')
        self.lower_, self.upper_ = np.quantile(x, self.quantiles, axis=0)
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.lower_):
            raise ValueError('transform columns must match fitted columns')
        return np.clip(x, self.lower_, self.upper_)


def decode_refractory(times, scores, symbols, threshold, cap, start, stop, delay,
                      cooldown_minutes=30):
    """Issue at most one alert per minute, using only earlier own issues."""
    used = Counter()
    last_issue = {}
    alerts, pending = [], []
    minutes = np.flatnonzero((times >= start) & (times < stop))
    in_window = scores[:, minutes]
    candidates = minutes[np.any(np.isfinite(in_window) & (in_window >= threshold), axis=0)]
    for minute in candidates:
        now = int(times[minute])
        day = now // (1440 * M)
        if used[day] >= cap:
            continue
        current = scores[:, minute]
        available = [i for i in np.flatnonzero(np.isfinite(current) & (current >= threshold))
                     if i not in last_issue or now - last_issue[i] >= cooldown_minutes * M]
        if not available:
            continue
        coin = min(available, key=lambda i: (-current[i], stable_key(symbols[i], 'open', now)))
        used[day] += 1
        last_issue[coin] = now
        alert = dict(symbol=symbols[coin], action='open', issue_ms=now, time_ms=now + delay * M,
                     minute_index=int(minute), score=float(current[coin]))
        (alerts if alert['time_ms'] < stop else pending).append(alert)
    return alerts, pending


def calibrate_threshold(times, scores, symbols, start, stop, fit_rate, cap,
                        cooldown_minutes=30, delay=1):
    """Match the fit event rate using score-only calibration-window decoding."""
    minutes = np.flatnonzero((times >= start) & (times < stop))
    calibration = scores[:, minutes]
    finite = np.isfinite(calibration)
    valid = finite.any(axis=0)
    if not valid.any():
        raise ValueError('calibration window must contain finite scores')
    maxima = np.max(np.where(finite[:, valid], calibration[:, valid], -np.inf), axis=0)
    quantiles = np.unique([.9, .95, .97, .98, .99, .995, .997, .999, .9995, 1 - fit_rate])
    thresholds = np.unique(np.quantile(maxima, quantiles))
    target = float(fit_rate * len(maxima))
    trials = []
    for threshold in thresholds:
        alerts, pending = decode_refractory(times, scores, symbols, float(threshold), cap,
                                           start, stop, delay, cooldown_minutes)
        trials.append(dict(threshold=float(threshold), realized_alerts=len(alerts) + len(pending)))
    best = min(trials, key=lambda trial: (abs(trial['realized_alerts'] - target), -trial['threshold']))
    return dict(threshold=best['threshold'], target_alerts=target,
                realized_alerts=best['realized_alerts'], trials=trials)
