"""Independent v5 coefficient, event, quantity, path and decoder verification.

No imports from model/runner/matcher implementations; no new fits/search.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,subprocess,sys,os
import numpy as np
from scipy.special import expit,softmax

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'analysis/results/entry_management_v5';M=60000
ACTIONS=['open','add','reduce','close']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stamp(s):return int(datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()*1000)


def binary(d,x):
    if 'coefficients' in d:
        z=x.copy();z-=np.array(d['scaler_mean']);z/=np.array(d['scaler_scale'])
        linear=z@np.array(d['coefficients'][0])+d['intercept'][0]
    else:
        q=(d['natural_events']+.5)/(d['natural_minutes']+1);linear=np.full(len(x),np.log(q/(1-q)))
    q=np.clip(expit(linear),1e-12,1-1e-12)
    return expit(np.log(q/(1-q))+d['calibration_offset'])


def multiclass(d,market,state):
    x=np.column_stack([market,state])if d['variant']=='market_state'else state[:,:5]
    h=binary(d['head'],x)
    z=x.copy();z-=np.array(d['action_scaler_mean']);z/=np.array(d['action_scaler_scale'])
    logits=z@np.array(d['action_coefficients']).T+np.array(d['action_intercept'])
    if logits.shape[1]==1:
        a=expit(logits[:,0]);conditional=np.column_stack([1-a,a])
    else:conditional=softmax(logits,axis=1)
    return conditional*h[:,None]


def aggregate(market,ns,n):
    cube=market.reshape(ns,n,14);a=np.full((n,10),np.nan);count=np.zeros(n,int)
    for i in range(n):
        available=[cube[j,i]for j in range(ns)if np.isfinite(cube[j,i]).all()]
        count[i]=len(available)
        if not available:continue
        x=np.asarray(available)
        a[i]=[np.mean(x[:,1]),np.max(x[:,1]),np.max(x[:,7]),np.max(x[:,10]),np.max(x[:,11]),
              np.mean(x[:,1]>0),np.max(x[:,3]),np.mean(x[:,4]),np.median(x[:,12]),len(available)/ns]
    return a,count


def matching(pp,actual,tol,episode=False):
    groups=defaultdict(lambda:[[],[]])
    for j,events in enumerate((pp,actual)):
        for e in events:groups[(e['episode_id']if episode else e['symbol'],e['action'])][j].append(e)
    tp=0
    for predictions,aa in groups.values():
        adj=[[j for j,a in enumerate(aa)if a['time_ms']>=p['issue_ms']and abs(a['time_ms']-p['time_ms'])<=tol]for p in predictions]
        owner={}
        def search(i,seen):
            for j in adj[i]:
                if j in seen:continue
                seen.add(j)
                if j not in owner or search(owner[j],seen):owner[j]=i;return True
            return False
        tp+=sum(search(i,set())for i in range(len(predictions)))
    return dict(tp=tp,fp=len(pp)-tp,fn=len(actual)-tp)


def path_stats(q,truth,marks,scale):
    own=np.asarray(q,float)*np.asarray(marks,float)/scale
    actual=np.asarray(truth,float)*np.asarray(marks,float)/scale
    return float(np.abs(own-actual).sum()),float(np.abs(actual).sum())


def flow(pp,actual,closes,symbols,n,start,stop,first,scale):
    bins=defaultdict(lambda:[0.,0.])
    for j,events in enumerate((pp,actual)):
        for e in events:
            if not start<=e['time_ms']<stop:continue
            minute=(e['time_ms']//M*M-first)//M
            if not 0<=minute<n:continue
            row=symbols.index(e['symbol'])*n+minute;mark=float(closes[row])
            if not np.isfinite(mark):continue
            key=(e.get('episode_id',e['symbol']),e['action'],e['time_ms']//(5*M))
            bins[key][j]+=float(e.get('quantity',e.get('qty')))*mark/scale
    num=sum(abs(x-y)for x,y in bins.values());den=sum(y for x,y in bins.values())
    return num,den


def event_quantity_path(events,times,initial=0.):
    events=sorted(events,key=lambda e:(e['time_ms'],e.get('source_index',0)))
    change=[float(e.get('quantity',e.get('qty')))*(1 if e['action']in('open','add')else -1)for e in events]
    cumulative=initial+np.cumsum(change)
    at=np.searchsorted([e['time_ms']for e in events],times,side='left')-1
    q=np.full(len(times),initial,float);ok=at>=0;q[ok]=cumulative[at[ok]]
    return np.maximum(q,0.)


def main():
    protocol,result,models,archive,audit=map(read,[OUT/'protocol.json',OUT/'results.json',OUT/'models.json',OUT/'predictions.json',OUT/'observation_audit.json'])
    protected=read(OUT/'source_hashes.json');changed=[name for name,digest in protected.items()if sha(ROOT/name)!=digest]
    assert not changed,changed
    assert sha(OUT/'protocol.json')==result['protocol_sha256']
    assert (OUT/'implementation_attempt_1/repair.json').is_file()
    data=dict(np.load(ROOT/'analysis/results/reverse_panel.npz',allow_pickle=False))
    meta=read(ROOT/'analysis/results/reverse_panel_meta.json');source=read(ROOT/'analysis/results/leader_events.json')
    extras=dict(np.load(ROOT/'analysis/results/observed_behavior_v3/features.npz',allow_pickle=False))
    scores=np.load(OUT/'scores.npz',allow_pickle=False)
    n=len(data['timeline_ms']);ns=len(meta['symbols']);times=data['timeline_ms'];first=int(times[0])
    f={name:data['X'][:,i]for i,name in enumerate(meta['feature_names'])}
    market=np.column_stack([data['X'][:,:8],extras['X_extra'][:,:6]])
    state=np.column_stack([data['X'][:,[meta['feature_names'].index(name)for name in protocol['state_features'][:5]]],extras['X_extra'][:,6:]])
    agg,available=aggregate(market,ns,n);ready=(available>=20)&np.isfinite(agg).all(axis=1)
    y=np.zeros(n,int);counts=np.zeros(n,int);primary=[];source_groups=defaultdict(list);ambiguous_minutes=set()
    labels=np.zeros(ns*n,int);eligible=data['state_known_mask'].copy()
    for e in source:
        if e['symbol']not in meta['symbols']or not first<=e['time_ms']<times[-1]+M:continue
        row=meta['symbols'].index(e['symbol'])*n+(e['time_ms']//M*M-first)//M
        ee={**e,'panel_row':int(row)};source_groups[int(row)].append(ee)
        if e['direction']=='SHORT'and e['classification']=='history_anchored':
            primary.append(ee)
            if e['action']=='open':y[row%n]=1;counts[row%n]+=1
        elif e['direction']=='SHORT'and e['action']=='open':ambiguous_minutes.add(row%n)
    score_ready=ready.copy()
    for i in ambiguous_minutes:score_ready[i]=False
    for row,ee in source_groups.items():
        if len(ee)!=1:eligible[row]=False;continue
        e=ee[0]
        if e['classification']!='history_anchored':eligible[row]=False;continue
        if e['direction']!='SHORT':continue
        if e['action']=='open':labels[row]=1
        elif data['direction'][row]==-1 and data['cycle_idx'][row]>=0 and meta['cycle_ids'][int(data['cycle_idx'][row])]==e['cycle_id']:
            labels[row]=ACTIONS.index(e['action'])+1
        else:eligible[row]=False
    held=eligible&(data['direction']==-1)&np.isfinite(market).all(axis=1)&np.isfinite(state).all(axis=1)
    assert len(primary)==audit['raw_primary_events']and int(counts.sum())==audit['raw_primary_openings']
    assert int(y.sum())==audit['opening_minutes']
    assert int((counts>1).sum())==len(audit['multi_coin_open_minutes'])==1
    assert sum(len(g['event_ids'])for g in audit['same_symbol_multiple_groups'])==8
    assert np.all(extras['anchor_time_ms'][held]<data['time_ms'][held])
    checks=[];source_samples=[];decoder_openings_checked=0;management_samples_checked=0
    for fold,fc in zip(protocol['folds'],result['folds']):
        lo,mid,hi,end=[stamp(fold[k])for k in ('train_start','calibration_start','train_end','eval_end')]
        evaluation=(times>=hi)&(times<end);risk_rows=np.flatnonzero(evaluation&score_ready)
        angle=(times%(1440*M))/(1440*M)*2*np.pi;clock=np.column_stack([np.sin(angle),np.cos(angle)])
        timing_all={}
        for tc in fc['timing']:
            variant=tc['variant'];head=models[fold['name']+'_'+variant+'_timing']['head']
            x=np.column_stack([clock,agg])if variant=='market_clock'else clock
            valid_rows=np.flatnonzero(ready);full=np.full(n,np.nan);full[valid_rows]=binary(head,x[valid_rows]);timing_all[variant]=full
            saved_rows=scores[fold['name']+'_'+variant+'_timing_rows'];saved_p=scores[fold['name']+'_'+variant+'_timing_p']
            assert np.array_equal(risk_rows,saved_rows)
            np.testing.assert_allclose(full[risk_rows],saved_p,rtol=1e-10,atol=1e-12)
            q=np.clip(full[risk_rows],1e-12,1-1e-12);target=y[risk_rows]
            ll=float(-(target*np.log(q)+(1-target)*np.log(1-q)).mean());brier=float(((q-target)**2).mean())
            assert np.isclose(ll,tc['probability']['log_loss'],rtol=1e-10)
            assert np.isclose(brier,tc['probability']['brier'],rtol=1e-10)
            assert np.isclose(q.sum(),tc['probability']['expected_events'],rtol=1e-10)
            aa=archive[fold['name']+'_'+variant+'_timing'];used=Counter();expected=[]
            for i in np.flatnonzero(evaluation&ready):
                if full[i]<tc['threshold']:continue
                day=int(times[i]//(1440*M))
                if used[day]>=tc['daily_cap']:continue
                used[day]+=1;expected.append(int(i))
            assert expected==[a['minute_index']for a in aa['alerts']]
            scored=[p for p in aa['alerts']if score_ready[p['minute_index']]]
            for tol,name in ((M,'primary'),(5*M,'sensitivity_5m')):
                match=matching(scored,aa['actual'],tol)
                for key in ('tp','fp','fn'):assert match[key]==tc[name][key]
            checks.append(dict(fold=fold['name'],component='timing_'+variant,log_loss=ll,brier=brier,public_only_alerts_verified=len(expected)))
        for probability in fc['local_management_probability']:
            variant=probability['variant'];rr=scores[fold['name']+'_'+variant+'_held_rows']
            expected=np.flatnonzero(np.tile(evaluation,ns)&held);assert np.array_equal(rr,expected)
            p=multiclass(models[fold['name']+'_'+variant+'_manager'],market[rr],state[rr])
            np.testing.assert_allclose(p,scores[fold['name']+'_'+variant+'_held_p'],rtol=1e-10,atol=1e-12)
            q=np.clip(p.sum(axis=1),1e-12,1-1e-12);target=labels[rr]>1
            ll=float(-(target*np.log(q)+(1-target)*np.log(1-q)).mean())
            assert np.isclose(ll,probability['any_operation']['log_loss'],rtol=1e-10)
        scale=fc['fit_size']['open']['median']
        for management in fc['management']:
            variant=management['variant'];delay=management['extra_delay'];key=fold['name']+'_'+variant+'_conditional_delay'+str(delay)
            ledger=archive[key];pp,actual=ledger['predictions'],ledger['actual'];num=0.;den=0.
            for ep,epcard in zip(management['episode_records'],management['episode_paths']):
                assert ep['end_ms']==min(end,ep['source_open_ms']+protocol['episode_horizon_minutes']*M)
                assert ep['first_decision_ms']>ep['source_open_ms']
                source_cycle=[e for e in source if e['cycle_id']==ep['episode_id']]
                index=np.flatnonzero((times>=ep['first_decision_ms'])&(times<ep['end_ms']))
                si=meta['symbols'].index(ep['symbol']);rr=si*n+index;ok=np.isfinite(f['close'][rr]);rr=rr[ok]
                own_events=[p for p in pp if p['episode_id']==ep['episode_id']]
                truth=event_quantity_path(source_cycle,data['time_ms'][rr]);own=event_quantity_path(own_events,data['time_ms'][rr],ep['true_first_qty'])
                a,b=path_stats(own,truth,f['close'][rr],scale);num+=a;den+=b
                assert np.isclose(a,epcard['absolute_exposure_error_sum'],rtol=1e-9,atol=1e-8)
                assert np.isclose(b,epcard['truth_exposure_sum'],rtol=1e-9,atol=1e-8)
                assert all(p['time_ms']<ep['end_ms']for p in own_events)
            assert np.isclose(num/den,management['path']['normalized_error'],rtol=1e-9)
            for tol,name in ((M,'primary'),(5*M,'sensitivity_5m')):
                match=matching(pp,actual,tol,True)
                for metric in ('tp','fp','fn'):assert match[metric]==management[name][metric]
            a,b=flow(pp,actual,f['close'],meta['symbols'],n,hi,end,first,scale)
            assert np.isclose(a,management['quantity_flow']['absolute_flow_error'],rtol=1e-10)
            assert np.isclose(b,management['quantity_flow']['actual_gross_flow'],rtol=1e-10)
            checks.append(dict(fold=fold['name'],component=variant+'_conditional',delay=delay,episodes=management['episodes'],independent_path_error=num/den,
                               independent_quantity_flow_error=a/b if b else None))
        for combo in fc['combinations']:
            delay=combo['extra_delay'];key=fold['name']+'_'+combo['id']+'_composed_delay'+str(delay);ledger=archive[key]
            pp,issued,actual=ledger['predictions'],ledger['issued'],ledger['actual']
            config=next(c for c in protocol['combinations']if c['id']==combo['id']);timing=next(t for t in fc['timing']if t['variant']==config['timing'])
            frequency=np.array([sum(e['symbol']==s and e['action']=='open'and lo<=e['time_ms']<mid for e in primary)for s in meta['symbols']],float)
            rows=np.flatnonzero(np.tile(evaluation,ns)&data['state_known_mask']&np.isfinite(f['close']))
            own=np.zeros(len(rows));truth=np.where(data['direction'][rows]==-1,np.abs(f['position'][rows]),0.)
            for si,symbol in enumerate(meta['symbols']):
                initial=combo['initial_positions'][si];initial_row=si*n+int(np.searchsorted(times,hi))
                expected_init=abs(float(f['position'][initial_row]))if data['state_known_mask'][initial_row]and data['direction'][initial_row]==-1 else 0.
                assert np.isclose(initial['quantity'],expected_init,rtol=1e-12)
                rr=rows[data['symbol_idx'][rows]==si];at=data['symbol_idx'][rows]==si
                own[at]=event_quantity_path([p for p in pp if p['symbol']==symbol],data['time_ms'][rr],initial['quantity'])
            a,b=path_stats(own,truth,f['close'][rows],scale)
            assert np.isclose(a/b,combo['path']['normalized_error'],rtol=1e-6,atol=1e-8)
            fa,fb=flow(pp,actual,f['close'],meta['symbols'],n,hi,end,first,scale)
            assert np.isclose(fa,combo['quantity_flow']['absolute_flow_error'],rtol=1e-10)
            for tol,name in ((M,'primary'),(5*M,'sensitivity_5m')):
                match=matching(pp,actual,tol)
                for metric in ('tp','fp','fn'):assert match[metric]==combo[name][metric]
            # Reconstruct own quantities/queues at every minute and reproduce
            # every opening decision, including times with no real opening.
            states=[dict(x)for x in combo['initial_positions']];pending={};used=Counter()
            issued_by_time=defaultdict(list);executed_by_time=defaultdict(list)
            for p in issued:issued_by_time[p['issue_ms']].append(p)
            for p in pp:executed_by_time[p['time_ms']].append(p)
            for i in np.flatnonzero(evaluation):
                now=int(times[i]);rr=np.arange(ns)*n+int(i);at_issue=issued_by_time[now]
                for p in at_issue:
                    si=meta['symbols'].index(p['symbol']);s=states[si];mark=float(f['close'][p['panel_row']])
                    if p['action']=='open':continue
                    assert s['quantity']>0 and si not in pending
                    expected_qty=(fc['fit_size']['add']['median']*s['first_qty']if p['action']=='add'else
                                  min(fc['fit_size']['reduce']['median'],1.)*s['quantity']if p['action']=='reduce'else s['quantity'])
                    assert np.isclose(p['quantity'],expected_qty,rtol=1e-9)
                    pending[si]=p;management_samples_checked+=1
                expected_coin=None
                if ready[i]and timing_all[config['timing']][i]>=timing['threshold']:
                    day=now//(1440*M)
                    if used[day]<timing['daily_cap']:
                        used[day]+=1
                        eligible=np.isfinite(market[rr]).all(axis=1)&np.isfinite(f['close'][rr])&(f['close'][rr]>0)
                        eligible&=np.array([s['quantity']==0 and si not in pending for si,s in enumerate(states)])
                        candidates=np.flatnonzero(eligible)
                        if len(candidates):
                            rank=frequency.copy()
                            if config['selector']=='market_rank':
                                rank[:]=-np.inf;rank[candidates]=binary(models[fold['name']+'_selector']['head'],market[rr[candidates]])
                            def stable(si):return hashlib.sha256(f"{meta['symbols'][si]}|open|{now}".encode()).hexdigest()
                            expected_coin=int(min(candidates,key=lambda si:(-rank[si],stable(si))))
                emitted=[p for p in at_issue if p['action']=='open']
                assert len(emitted)==(1 if expected_coin is not None else 0),(key,now)
                if emitted:
                    p=emitted[0];assert p['symbol']==meta['symbols'][expected_coin]
                    assert np.isclose(p['timing_probability'],timing_all[config['timing']][i],rtol=1e-10)
                    assert np.isclose(p['quantity'],scale/float(f['close'][p['panel_row']]),rtol=1e-10)
                    pending[expected_coin]=p;decoder_openings_checked+=1
                for p in executed_by_time[now]:
                    si=meta['symbols'].index(p['symbol']);s=states[si];q=p['quantity'];price=p['execution_price']
                    assert p['time_ms']==p['issue_ms']+(1+delay)*M
                    assert np.isclose(price,data['next_open'][p['panel_row']+delay],rtol=1e-12)
                    assert data['next_open_time_ms'][p['panel_row']+delay]==p['time_ms']
                    if p['action']=='open':assert s['quantity']==0;s['quantity']=q;s['first_qty']=q;s['cost']=price
                    elif p['action']=='add':s['cost']=(s['cost']*s['quantity']+price*q)/(s['quantity']+q);s['quantity']+=q
                    else:assert q<=s['quantity']*(1+1e-9);s['quantity']=max(0.,s['quantity']-q)
                for si,p in list(pending.items()):
                    if p['time_ms']<=now:del pending[si]
            turnover=sum(p['quantity']*p['execution_price']for p in pp)
            assert np.isclose(turnover,combo['path']['virtual_turnover_quote'],rtol=1e-12)
            checks.append(dict(fold=fold['name'],component=combo['id']+'_composed',delay=delay,independent_path_error=a/b,
                               independent_quantity_flow_error=fa/fb if fb else None,independent_turnover_quote=turnover))
        for action in ACTIONS:
            e=next(e for e in primary if hi<=e['time_ms']<end and e['action']==action);r=e['panel_row']
            expected=float(e['position_before']);observed=0. if data['direction'][r]==0 else float(f['position'][r])
            # Same-minute multi-event states are intentionally not equated to
            # each individual pre-event quantity; select a unique minute.
            if len(source_groups[r])==1:
                assert np.isclose(observed,expected,rtol=1e-6,atol=1e-6)
                source_samples.append(dict(event_id=e['event_id'],action=action,source_prior=expected,panel_prior=observed,minute_ms=int(data['time_ms'][r])))
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'analysis/tests'),'-v'],capture_output=True,text=True,
                        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    assert test.returncode==0,test.stderr
    verification=dict(status='passed_independent_software_and_result_checks',verified_at_utc=datetime.now(timezone.utc).isoformat(),
        independent_behavior_validation=False,progression_gates=result['gates'],protected_original_files=len(protected),changed_original_files=changed,
        protocol_sha256=sha(OUT/'protocol.json'),checks=checks,decoder_openings_checked=decoder_openings_checked,
        management_issue_quantities_checked=management_samples_checked,source_spot_checks=source_samples,
        public_readiness='finite strictly-past public market features >=20 symbols; independent of labels, true positions and future close',
        first_implementation_attempt_preserved=True,executed_head_attempts=42,accepted_fixed_heads=21,configuration_search_expansion=False,
        test_output=test.stderr,new_source_hashes={str(f.relative_to(ROOT)):sha(f)for f in
            [ROOT/'analysis/entry_management.py',ROOT/'analysis/run_entry_management.py',ROOT/'analysis/verify_entry_management.py',ROOT/'analysis/tests/test_entry_management.py']})
    (OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('VERIFIED',len(checks),'component/scenario checks; openings',decoder_openings_checked,'management quantities',management_samples_checked,'old files',len(protected),'unchanged')


if __name__=='__main__':main()
