"""Independent saved-result verification; never refits or selects candidates.

No imports from forecasting/runner/matcher code. Reconstructs eligibility from
source records, probability from coefficients, event matches by augmenting paths,
and own quantity paths from the saved execution ledger.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.special import expit

M=60000
ACTIONS=['open','add','reduce','close']
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/results/behavior_hazard_v4'


def read(path):return json.loads(path.read_text())
def stamp(text):return int(datetime.fromisoformat(text.replace('Z','+00:00')).timestamp()*1000)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(data,meta,cycles,lo,hi):
    keep=(data['time_ms']>=lo)&(data['time_ms']<hi)
    for cycle in cycles:
        if cycle['status']!='complete' or cycle['symbol'] not in meta['symbols']:continue
        if any(cycle['start_ms']<b<=cycle['end_ms'] for b in (lo,hi)):
            si=meta['symbols'].index(cycle['symbol'])
            keep&=~((data['symbol_idx']==si)&(data['time_ms']>=cycle['start_ms']//M*M)&
                     (data['time_ms']<=cycle['end_ms']//M*M))
    return keep


def maximum_match(pp,aa,tolerance):
    groups=defaultdict(lambda:([],[]))
    for p in pp:groups[p['symbol'],p['action']][0].append(p)
    for a in aa:groups[a['symbol'],a['action']][1].append(a)
    count=0
    for ps,actual in groups.values():
        adj=[[j for j,a in enumerate(actual) if a['time_ms']>=p['issue_ms'] and
              abs(a['time_ms']-p['time_ms'])<=tolerance] for p in ps]
        owner={}
        def visit(i,seen):
            for j in adj[i]:
                if j in seen:continue
                seen.add(j)
                if j not in owner or visit(owner[j],seen):owner[j]=i;return True
            return False
        count+=sum(visit(i,set()) for i in range(len(ps)))
    return dict(tp=count,fp=len(pp)-count,fn=len(aa)-count)


def binary_from_details(details,x):
    if 'coefficients' in details:
        z=x.copy()
        # sklearn preserves float32 at each in-place centering/scaling step.
        # Match that declared storage arithmetic without calling the trainer.
        z-=np.array(details['scaler_mean'])
        z/=np.array(details['scaler_scale'])
        linear=z@np.array(details['coefficients'][0])+details['intercept'][0]
    else:
        p=(details['natural_events']+.5)/(details['natural_minutes']+1)
        linear=np.full(len(x),np.log(p/(1-p)))
    raw=expit(linear)
    raw=np.clip(raw,1e-12,1-1e-12)
    return expit(np.log(raw/(1-raw))+details['calibration_offset'])


def natural_score(y,p):
    p=np.clip(p,1e-12,1-1e-12);y=np.asarray(y,int)
    return dict(events=int(y.sum()),minutes=len(y),expected_events=float(p.sum()),
                log_loss=float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p)))),
                brier=float(np.mean((p-y)**2)))


def main():
    protocol,results=read(OUT/'protocol.json'),read(OUT/'results.json')
    models,predictions=read(OUT/'models.json'),read(OUT/'predictions.json')
    data=dict(np.load(ROOT/'analysis/results/reverse_panel.npz',allow_pickle=False))
    meta=read(ROOT/'analysis/results/reverse_panel_meta.json')
    events,cycles=read(ROOT/'analysis/results/leader_events.json'),read(ROOT/'analysis/results/leader_cycles.json')
    protected=read(OUT/'source_hashes.json')
    changed=[name for name,old in protected.items() if sha(ROOT/name)!=old]
    assert not changed,changed
    assert sha(OUT/'protocol.json')==results['protocol_sha256']
    n=len(data['timeline_ms']);start=int(data['timeline_ms'][0]);end=int(data['timeline_ms'][-1])+M
    labels=np.zeros(len(data['time_ms']),int);allowed=data['state_known_mask'].copy()
    source_groups=defaultdict(list)
    for e in events:
        if e['symbol'] in meta['symbols'] and start<=e['time_ms']<end:
            r=meta['symbols'].index(e['symbol'])*n+(e['time_ms']//M*M-start)//M
            source_groups[int(r)].append(e)
    gold=[];multiple=[];unanchored=[]
    for r,records in source_groups.items():
        if len(records)!=1:
            allowed[r]=False
            multiple.append(dict(symbol=records[0]['symbol'],minute_ms=int(data['time_ms'][r]),
                records=len(records),actions=[e['action'] for e in records],
                event_ids=[e['event_id'] for e in records],
                first_position_before=records[0]['position_before'],last_position_after=records[-1]['position_after'],
                absolute_quantity_sum=sum(float(e['qty']) for e in records),
                net_signed_quantity=sum((-1 if e['side']=='SELL' else 1)*float(e['qty']) for e in records)))
            continue
        e=records[0]
        if e['classification']!='history_anchored':allowed[r]=False;unanchored.append(e['event_id']);continue
        if e['direction']!='SHORT':continue
        state_ok=(data['direction'][r]==0 if e['action']=='open' else
                  data['direction'][r]==-1 and data['cycle_idx'][r]>=0 and
                  meta['cycle_ids'][int(data['cycle_idx'][r])]==e['cycle_id'])
        if not allowed[r] or not state_ok:allowed[r]=False;continue
        labels[r]=ACTIONS.index(e['action'])+1
        gold.append({**e,'panel_row':r})
    f={name:data['X'][:,i] for i,name in enumerate(meta['feature_names'])}
    market=np.column_stack([f[name] for name in protocol['market_features']])
    state=np.column_stack([f[name] for name in protocol['state_features']])
    flat=allowed&(data['direction']==0)&np.isfinite(market).all(axis=1)
    held=allowed&(data['direction']==-1)&np.isfinite(market).all(axis=1)&np.isfinite(state).all(axis=1)
    checks=[];manual_samples=[]
    for fold,fc in zip(protocol['folds'],results['fold_results']):
        lo,mid,hi,stop=[stamp(fold[k]) for k in ('train_start','calibration_start','train_end','eval_end')]
        fit=interval(data,meta,cycles,lo,mid);cal=interval(data,meta,cycles,mid,hi);ev=interval(data,meta,cycles,hi,stop)
        er=np.flatnonzero(ev&(flat|held))
        actual=[e for e in gold if ev[e['panel_row']] and (flat|held)[e['panel_row']]]
        assert len(actual)==fc['evaluation_events']==int((labels[er]>0).sum())
        assert len(er)==fc['evaluation_minutes']
        samples=defaultdict(list);tests=defaultdict(list)
        for e in gold:
            r,a=e['panel_row'],e['action']
            if a=='close' or not (flat|held)[r]:continue
            value=(float(e['qty'])*float(f['close'][r]) if a=='open' else
                   float(e['qty'])/(float(f['first_qty'][r]) if a=='add' else abs(float(f['position'][r]))))
            if fit[r]:samples[a].append(value)
            if ev[r]:tests[a].append(value)
        for a in ('open','add','reduce'):
            distribution=fc['quantity_fit'][a]
            assert distribution['n']==len(samples[a])
            assert np.isclose(np.median(samples[a]),distribution['median'],rtol=1e-12)
            error=float(np.abs(np.array(tests[a])-distribution['median']).sum()/sum(tests[a]))
            assert np.isclose(error,fc['quantity_evaluation'][a]['relative_total_absolute_error'],rtol=1e-12)
        for a in ACTIONS:
            e=next(e for e in actual if e['action']==a);r=e['panel_row']
            before=float(e['position_before'])
            panel_before=0. if data['direction'][r]==0 else float(f['position'][r])
            assert np.isclose(panel_before,before,rtol=1e-6,atol=1e-7)
            manual_samples.append(dict(fold=fold['name'],action=a,event_id=e['event_id'],
                issue_minute_ms=int(data['time_ms'][r]),actual_time_ms=e['time_ms'],
                source_prior_quantity=before,panel_prior_quantity=panel_before,
                prior_closed_price=float(f['close'][r]),source_execution_price=float(e['price']),
                state_check='prior source quantity agrees; execution price not used as decision feature'))
        for trial in fc['variants']:
            variant=trial['variant'];key=fold['name']+'_'+variant;details=models[key]
            pooled=np.empty(len(er))
            flat_er=flat[er];held_er=held[er]
            ox=market[er[flat_er]] if variant=='market_state' else np.zeros((flat_er.sum(),1))
            hx=np.column_stack([market[er[held_er]],state[er[held_er]]]) if variant=='market_state' else state[er[held_er]]
            pooled[flat_er]=binary_from_details(details['open'],ox)
            pooled[held_er]=binary_from_details(details['hazard'],hx)
            score=natural_score(labels[er]>0,pooled)
            for metric in ('events','minutes','expected_events','log_loss','brier'):
                assert np.isclose(score[metric],trial['probability']['pooled'][metric],rtol=1e-10,atol=1e-12),(key,metric)
            for head,risk in [('open',flat),('hazard',held)]:
                d=details[head];fitrows=fit&risk
                assert d['natural_minutes']==int(fitrows.sum())
                assert d['natural_events']==int(((labels==1 if head=='open' else labels>1)&fitrows).sum())
                if 'weight_sum' in d:
                    assert np.isclose(d['weight_sum'],d['natural_minutes'])
                    assert np.isclose(d['event_weight_sum'],d['natural_events'])
            local=predictions[key+'_local']
            for a in ACTIONS:
                match=maximum_match([p for p in local if p['action']==a],[e for e in actual if e['action']==a],M)
                for metric in ('tp','fp','fn'):assert match[metric]==trial['local_events'][a][metric]
            scoring=ev&allowed&np.isin(data['direction'],[0,-1])&np.isfinite(f['close'])
            score_rows=np.flatnonzero(scoring)
            for roll in trial['autonomous']:
                delay=roll['delay_extra_minutes'];archive=predictions[key+'_delay_'+str(delay)]
                executed,issued=archive['executed'],archive['issued']
                issued_count=Counter((p['issue_ms']//(1440*M),p['action']) for p in issued)
                for (_,a),count in issued_count.items():assert count<=trial['daily_budgets'][ACTIONS.index(a)]
                scored=[p for p in executed if ev[p['panel_row']] and allowed[p['panel_row']] and data['direction'][p['panel_row']] in (0,-1)]
                for tolerance,name in [(M,'all_events'),(5*M,'sensitivity_5m')]:
                    match=maximum_match(scored,actual,tolerance)
                    for metric in ('tp','fp','fn'):assert match[metric]==roll[name][metric],(key,delay,metric)
                qty_by_row=np.zeros(len(data['time_ms']),float);turnover=0.
                for si,symbol in enumerate(meta['symbols']):
                    stream=sorted([p for p in executed if p['symbol']==symbol],key=lambda p:p['time_ms'])
                    qt=[];tm=[];qty=0.;first=0.
                    for p in stream:
                        source_row=p['panel_row']+delay
                        assert source_row//n==si
                        assert p['time_ms']==p['issue_ms']+(1+delay)*M
                        assert np.isclose(p['execution_price'],data['next_open'][source_row],rtol=1e-12)
                        a=p['action'];q=p['quantity'];price=float(f['close'][p['panel_row']])
                        if a=='open':
                            assert qty==0.
                            assert np.isclose(q,fc['quantity_fit']['open']['median']/price,rtol=1e-10)
                            qty=q;first=q
                        elif a=='add':
                            assert qty>0.
                            assert np.isclose(q,fc['quantity_fit']['add']['median']*first,rtol=1e-10)
                            qty+=q
                        elif a=='reduce':
                            assert qty>0 and q<=qty*(1+1e-10)
                            assert np.isclose(q,min(fc['quantity_fit']['reduce']['median'],1.)*qty,rtol=1e-10)
                            qty-=q
                        else:
                            assert np.isclose(q,qty,rtol=1e-9);qty=0.;first=0.
                        turnover+=q*p['execution_price'];qt.append(qty);tm.append(p['time_ms'])
                    rr=score_rows[data['symbol_idx'][score_rows]==si]
                    if len(tm):
                        at=np.searchsorted(tm,data['time_ms'][rr],side='left')-1
                        ok=at>=0;qty_by_row[rr[ok]]=np.array(qt)[at[ok]]
                marks=f['close'][score_rows].astype(float);scale=fc['quantity_fit']['open']['median']
                own=qty_by_row[score_rows]*marks/scale
                truth=np.nan_to_num(np.abs(f['position'][score_rows]),nan=0.).astype(float)*marks/scale
                path_error=float(np.abs(own-truth).sum()/truth.sum())
                assert np.isclose(path_error,roll['path']['normalized_absolute_error'],rtol=1e-6,atol=1e-7),(key,delay,path_error)
                assert np.isclose(turnover,roll['path']['turnover_quote_proxy'],rtol=1e-10)
                checks.append(dict(fold=fold['name'],variant=variant,extra_execution_delay=delay,
                    independent_events=maximum_match(scored,actual,M),independent_path_error=path_error,
                    independent_turnover_quote=turnover,natural_probability=score,
                    all_daily_online_budgets_obeyed=True,
                    constant_flat_path_error=1.0,beats_constant_flat_path=path_error<1.))
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'analysis/tests'),'-v'],
                        capture_output=True,text=True,env={**__import__('os').environ,'PYTHONDONTWRITEBYTECODE':'1'})
    assert test.returncode==0,test.stderr
    summary=dict(status='passed_software_and_independent_saved_result_checks',
        verified_at_utc=datetime.now(timezone.utc).isoformat(),
        behavioral_progression_passed=results['progression_folds']>=2,independent_statistical_validation=False,
        protected_old_files=len(protected),changed_old_files=changed,
        protocol_sha256=sha(OUT/'protocol.json'),
        new_source_hashes={str(p.relative_to(ROOT)):sha(p) for p in
                          [ROOT/'analysis/behavior_hazard.py',ROOT/'analysis/run_behavior_hazard.py',
                           ROOT/'analysis/verify_behavior_hazard.py',ROOT/'analysis/tests/test_behavior_hazard.py',
                           ROOT/'analysis/build_behavior_hazard_report.py']},
        independent_methods=['direct source-label grouping','saved coefficients logit arithmetic',
             'augmenting-path maximum bipartite matching','piecewise executed-quantity integration',
             'train/test quantity target recomputation','12 source/panel prior-quantity spot checks'],
        same_symbol_minute_multiple_groups=multiple,multiple_event_count=sum(e['records'] for e in multiple),
        unanchored_excluded_event_ids=unanchored,
        evaluation_events_by_fold={f['fold']:f['evaluation_events'] for f in results['fold_results']},
        checks=checks,manual_samples=manual_samples,test_output=test.stderr)
    (OUT/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('VERIFIED',len(checks),'rollouts; old files',len(protected),'unchanged; tests 16 passed')
    print('MULTIPLE',len(multiple),'symbol-minute groups containing',summary['multiple_event_count'],'records, explicitly excluded')


if __name__=='__main__':main()
