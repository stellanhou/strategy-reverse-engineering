"""Frozen event-level scoring for the reverse-engineering experiment."""
from collections import defaultdict


def match_events(predicted, actual, tolerance_ms=60000):
    """Maximum chronological one-to-one matches within symbol/direction/action.

    Sorting by timestamp and identifier fixes ties. The earliest feasible pair
    preserves maximum cardinality for the ordered, fixed-width time windows.
    """
    p, a = defaultdict(list), defaultdict(list)
    for row in predicted:
        p[(row['symbol'], row['direction'], row['action'])].append(row)
    for row in actual:
        a[(row['symbol'], row['direction'], row['action'])].append(row)
    matches = []
    for key in sorted(set(p) | set(a)):
        ps = sorted(p[key], key=lambda x: (x['time_ms'], x.get('event_id', '')))
        aa = sorted(a[key], key=lambda x: (x['time_ms'], x.get('event_id', '')))
        i = j = 0
        while i < len(ps) and j < len(aa):
            delta = ps[i]['time_ms'] - aa[j]['time_ms']
            if delta < -tolerance_ms:
                i += 1
            elif delta > tolerance_ms:
                j += 1
            else:
                matches.append({'prediction': ps[i], 'actual': aa[j], 'delta_ms': delta})
                i += 1
                j += 1
    tp, n_pred, n_actual = len(matches), len(predicted), len(actual)
    return {
        'tp': tp, 'fp': n_pred - tp, 'fn': n_actual - tp,
        'precision': tp / n_pred if n_pred else 0.0,
        'recall': tp / n_actual if n_actual else None,
        'f1': 2 * tp / (n_pred + n_actual) if n_pred + n_actual else None,
        'predictions': n_pred, 'actual_events': n_actual,
        'tolerance_ms': tolerance_ms,
    }


def score_by_action(predicted, actual, tolerance_ms=60000):
    return {action: match_events(
        [x for x in predicted if x['action'] == action],
        [x for x in actual if x['action'] == action], tolerance_ms)
        for action in ['open', 'add', 'reduce', 'close']}


def event_block_counts(predicted, actual, start_ms, end_ms, block_ms=86400000):
    """TP pairs share the actual event block; FP/FN retain their own block."""
    import numpy as np
    blocks = (end_ms - start_ms + block_ms - 1) // block_ms
    counts = np.zeros((blocks, 3), dtype=int)
    p, a = defaultdict(list), defaultdict(list)
    for i, row in enumerate(predicted):
        p[(row['symbol'], row['direction'], row['action'])].append((i, row))
    for i, row in enumerate(actual):
        a[(row['symbol'], row['direction'], row['action'])].append((i, row))
    matched_p, matched_a = set(), set()
    for key in sorted(set(p) | set(a)):
        ps = sorted(p[key], key=lambda v: (v[1]['time_ms'], v[1].get('event_id', '')))
        aa = sorted(a[key], key=lambda v: (v[1]['time_ms'], v[1].get('event_id', '')))
        i = j = 0
        while i < len(ps) and j < len(aa):
            delta = ps[i][1]['time_ms'] - aa[j][1]['time_ms']
            if delta < -60000:
                i += 1
            elif delta > 60000:
                j += 1
            else:
                counts[(aa[j][1]['time_ms'] - start_ms) // block_ms, 0] += 1
                matched_p.add(ps[i][0]); matched_a.add(aa[j][0])
                i += 1; j += 1
    for i, row in enumerate(predicted):
        if i not in matched_p:
            counts[(row['time_ms'] - start_ms) // block_ms, 1] += 1
    for i, row in enumerate(actual):
        if i not in matched_a:
            counts[(row['time_ms'] - start_ms) // block_ms, 2] += 1
    total = counts.sum(0)
    score = match_events(predicted, actual)
    assert tuple(total) == (score['tp'], score['fp'], score['fn'])
    return counts


def paired_f1_interval(candidate, baseline, actual, start_ms, end_ms):
    import numpy as np
    out = {}
    for days in [1, 3]:
        c = event_block_counts(candidate, actual, start_ms, end_ms, days * 86400000)
        b = event_block_counts(baseline, actual, start_ms, end_ms, days * 86400000)
        picked = np.random.default_rng(20261002).integers(len(c), size=(1000, len(c)))
        cc, bb = c[picked].sum(1), b[picked].sum(1)
        cd, bd = 2 * cc[:, 0] + cc[:, 1] + cc[:, 2], 2 * bb[:, 0] + bb[:, 1] + bb[:, 2]
        valid = (cd > 0) & (bd > 0)
        differences = 2 * cc[valid, 0] / cd[valid] - 2 * bb[valid, 0] / bd[valid]
        out[f'{days}_day_block_95_interval'] = [float(x) for x in np.quantile(differences, [.025, .975])] if valid.sum() >= 100 else None
        out[f'{days}_day_valid_replicates'] = int(valid.sum())
    return out
