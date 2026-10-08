"""Natural-frequency action versus no-action contrasts on the minute panel."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/results'
DAY = 86400000
SEED = 20261002


def contrast(values, positive, negative, times):
    p, n = values[positive], values[negative]
    result = {'positive_minutes': len(p), 'control_minutes': len(n)}
    if not len(p) or not len(n):
        return dict(result, status='insufficient_events_or_controls')
    result.update(positive_median=float(np.median(p)), control_median=float(np.median(n)),
                  mean_difference=float(p.mean() - n.mean()))
    for block_days in [1, 3]:
        blocks = times // (DAY * block_days)
        domain = blocks[positive | negative]
        blocks = blocks - domain.min()
        size = int(domain.max() - domain.min()) + 1
        ps = np.bincount(blocks[positive], weights=p, minlength=size)
        ns = np.bincount(blocks[negative], weights=n, minlength=size)
        pc = np.bincount(blocks[positive], minlength=size)
        nc = np.bincount(blocks[negative], minlength=size)
        rng = np.random.default_rng(SEED)
        picked = rng.integers(size, size=(1000, size))
        pcount, ncount = pc[picked].sum(1), nc[picked].sum(1)
        valid = (pcount > 0) & (ncount > 0)
        deltas = ps[picked[valid]].sum(1) / pcount[valid] - ns[picked[valid]].sum(1) / ncount[valid]
        result[f'{block_days}_day_block_95_interval'] = (
            [float(x) for x in np.quantile(deltas, [.025, .975])]
            if len(deltas) >= 100 else None)
        result[f'{block_days}_day_valid_replicates'] = int(valid.sum())
    return result


def purged_mask(data, symbols, start, end, cycles):
    mask = (data['time_ms'] >= start) & (data['time_ms'] < end)
    for c in cycles:
        if c['status'] != 'complete':
            continue
        lo, hi = c['start_ms'], c['end_ms']
        if any(lo < boundary <= hi for boundary in [start, end]):
            symbol_idx = symbols.index(c['symbol'])
            mask &= ~((data['symbol_idx'] == symbol_idx) &
                      (data['time_ms'] >= lo // 60000 * 60000) &
                      (data['time_ms'] <= hi // 60000 * 60000))
    return mask


def main():
    data = np.load(OUT / 'reverse_panel.npz', allow_pickle=False)
    meta = json.loads((OUT / 'reverse_panel_meta.json').read_text())
    cycles = json.loads((OUT / 'leader_cycles.json').read_text())
    protocol = json.loads((OUT / 'reverse_protocol.json').read_text())
    symbols = meta['symbols']
    names = meta['feature_names']
    tests = [(1, 'open', 'ret15', 1), (1, 'open', 'btc_relative60', 1),
             (2, 'add', 'mark_return', -1), (4, 'close', 'mark_return', 1)]
    eligible = data['label_mask'].astype(bool)
    times = data['time_ms']
    top_symbol = symbols[int(np.bincount(data['symbol_idx'][data['label'] > 0],
                                         minlength=len(symbols)).argmax())]
    from datetime import datetime
    to_ms = lambda value: int(datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp() * 1000)
    reports = []
    for label, action, feature, sign in tests:
        values = data['X'][:, names.index(feature)].astype(float) * sign
        short_open = (data['label'] == 1) & (data['label_direction'] == -1)
        opportunity = (data['open_negative_eligible'].astype(bool) | short_open) if action == 'open' else data['direction'] == -1
        base = eligible & opportunity & np.isfinite(values)
        positive = base & (data['label'] == label)
        if action == 'open':
            positive &= data['label_direction'] == -1
        negative = base & (data['label'] == 0)
        report = {'action': action, 'score_feature': feature, 'sign': sign,
                  'expected_direction': 'positive mean difference',
                  'full_exploratory_calendar': contrast(values, positive, negative, times)}
        no_top = data['symbol_idx'] != symbols.index(top_symbol)
        report['remove_largest_event_symbol'] = {
            'removed_symbol': top_symbol,
            **contrast(values, positive & no_top, negative & no_top, times)}
        folds = []
        for fold in protocol['folds']:
            start, end = to_ms(fold['train_end']), to_ms(fold['eval_end'])
            fm = purged_mask(data, symbols, start, end, cycles)
            folds.append({'name': fold['name'], **contrast(values, positive & fm, negative & fm, times)})
        report['purged_evaluation_folds'] = folds
        valid = [f for f in folds if f['positive_minutes'] and f['control_minutes']]
        report['positive_direction_folds'] = sum(f['mean_difference'] > 0 for f in valid)
        report['valid_folds'] = len(valid)
        reports.append(report)
    result = {
        'status': 'exploratory_action_vs_no_action',
        'controls': 'full naturally occurring known flat or short-held no-MARKET-action minutes with finite score feature; labels and masks come from panel',
        'interval_method': 'paired calendar-block resampling of positive/control sum and count; 1000 replicates; daily and 3-day sensitivity; zero-event resamples excluded and counted',
        'scope': '33 retrospectively selected symbols; past observations only',
        'tests': reports,
    }
    (OUT / 'reverse_controls.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps([{'action': r['action'], 'feature': r['score_feature'],
                       'consistent_folds': r['positive_direction_folds'],
                       'valid_folds': r['valid_folds']} for r in reports]))


if __name__ == '__main__':
    main()
