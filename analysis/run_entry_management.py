"""Finite v5 decomposition experiment. Old files remain immutable."""
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path
import hashlib,json
import numpy as np

from behavior_hazard import M,causal_match,probability_card
from entry_management import (public_timing_matrix,EntryTiming,EntrySelector,Manager,ManagedPosition,
    EntryBudget,select_coin,size_for,state_from_past,quantity_path_card,cycle_truth)
from run_behavior_hazard import bootstrap_days

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/results/entry_management_v5'
ACTIONS=('open','add','reduce','close')

def read(path):return json.loads(path.read_text())
def write(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def stamp(s):return int(datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()*1000)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def load_view(root=ROOT):
    data=dict(np.load(root/'analysis/results/reverse_panel.npz',allow_pickle=False))
    meta=read(root/'analysis/results/reverse_panel_meta.json')
    events=read(root/'analysis/results/leader_events.json');cycles=read(root/'analysis/results/leader_cycles.json')
    extra=dict(np.load(root/'analysis/results/observed_behavior_v3/features.npz',allow_pickle=False))
    protocol=read(OUT/'protocol.json');n=len(data['timeline_ms']);ns=len(meta['symbols'])
    features={name:data['X'][:,i]for i,name in enumerate(meta['feature_names'])}
    features.update({str(name):extra['X_extra'][:,i]for i,name in enumerate(extra['feature_names'])})
    market=np.column_stack([features[name]for name in protocol['market_features']])
    state=np.column_stack([features[name]for name in protocol['state_features']])
    aggregate,count=public_timing_matrix(market,ns,n)
    timing_ready=(count>=20)&np.isfinite(aggregate).all(axis=1)
    timing_ok=timing_ready.copy()
    label=np.zeros(len(data['time_ms']),int);manager_ok=data['state_known_mask'].copy()
    groups=defaultdict(list);primary=[];inferred=[];ambiguous=[]
    for e in events:
        if e['symbol'] not in meta['symbols'] or not data['timeline_ms'][0]<=e['time_ms']<data['timeline_ms'][-1]+M:continue
        row=meta['symbols'].index(e['symbol'])*n+(e['time_ms']//M*M-int(data['timeline_ms'][0]))//M
        e={**e,'panel_row':int(row)};groups[int(row)].append(e)
        if e['direction']=='SHORT' and e['classification']=='history_anchored':primary.append(e)
        elif e['direction']=='SHORT' and e['action']=='open':inferred.append(e)
    for row,ee in groups.items():
        if len(ee)!=1:
            manager_ok[row]=False
            ambiguous.append(dict(panel_row=row,event_ids=[e['event_id']for e in ee],actions=[e['action']for e in ee],
                gross_quantity=sum(float(e['qty'])for e in ee),net_signed_quantity=sum(float(e['qty'])*(1 if e['side']=='BUY'else -1)for e in ee)))
            continue
        e=ee[0]
        if e['classification']!='history_anchored':manager_ok[row]=False;continue
        if e['direction']!='SHORT':continue
        if e['action']=='open':label[row]=1
        elif data['direction'][row]==-1 and data['cycle_idx'][row]>=0 and meta['cycle_ids'][int(data['cycle_idx'][row])]==e['cycle_id']:
            label[row]=ACTIONS.index(e['action'])+1
        else:manager_ok[row]=False
    held=manager_ok&(data['direction']==-1)&np.isfinite(market).all(axis=1)&np.isfinite(state).all(axis=1)
    opens=[e for e in primary if e['action']=='open']
    timing_y=np.zeros(n,int);open_count=np.zeros(n,int);rank_y=np.zeros(len(label),int)
    for e in opens:
        t=e['panel_row']%n;timing_y[t]=1;open_count[t]+=1;rank_y[e['panel_row']]=1
    for e in inferred:timing_ok[e['panel_row']%n]=False
    rank_ok=np.isfinite(market).all(axis=1)&np.tile(timing_y.astype(bool),ns)
    return dict(data=data,meta=meta,events=events,cycles=cycles,extra=extra['X_extra'],market=market,state=state,
        aggregate=aggregate,timing_ready=timing_ready,timing_ok=timing_ok,timing_y=timing_y,open_count=open_count,rank_y=rank_y,rank_ok=rank_ok,
        held=held,label=label,primary=primary,opens=opens,inferred=inferred,ambiguous=ambiguous,features=features)


def sizes_and_duration(view,fit,start,stop):
    f=view['features'];samples=defaultdict(list)
    for e in view['primary']:
        r=e['panel_row'];a=e['action']
        if not fit[r] or a=='close':continue
        if a=='open':value=float(e['qty'])*float(f['close'][r])
        elif not view['held'][r]:continue
        elif a=='add':value=float(e['qty'])/float(f['first_qty'][r])
        else:value=float(e['qty'])/abs(float(f['position'][r]))
        if np.isfinite(value):samples[a].append(value)
    size={a:dict(n=len(x),median=float(np.median(x)),q25=float(np.quantile(x,.25)),q75=float(np.quantile(x,.75)))for a,x in samples.items()}
    durations=[(c['end_ms']-c['start_ms'])/M for c in view['cycles']if c['status']=='complete'and c['direction']=='SHORT'
               and start<=c['start_ms']<c['end_ms']<stop]
    assert all(a in size for a in ('open','add','reduce'))and durations
    return size,float(np.median(durations))


def event_card(pp,aa,episode=False,tolerance=M):
    if episode:
        pp=[{**p,'symbol':p['episode_id']}for p in pp]
        aa=[{**e,'symbol':e['episode_id']}for e in aa]
    return causal_match(pp,aa,tolerance)


def flow_card(predicted,actual,view,scale,start,stop):
    n=len(view['data']['timeline_ms']);first=int(view['data']['timeline_ms'][0]);bins=defaultdict(lambda:[0.,0.])
    missing=Counter()
    for group,records in enumerate((predicted,actual)):
        for e in records:
            if not start<=e['time_ms']<stop:continue
            minute=(e['time_ms']//M*M-first)//M
            if not 0<=minute<n:missing[str(group)]+=1;continue
            r=view['meta']['symbols'].index(e['symbol'])*n+int(minute)
            mark=float(view['features']['close'][r])
            if not np.isfinite(mark):missing[str(group)]+=1;continue
            key=(e.get('episode_id',e['symbol']),e['action'],e['time_ms']//(5*M))
            bins[key][group]+=float(e.get('quantity',e.get('qty')))*mark/scale
    num=sum(abs(x-y)for x,y in bins.values());den=sum(y for x,y in bins.values())
    return dict(fixed_nonoverlap_bin_minutes=5,absolute_flow_error=num,actual_gross_flow=den,
        relative_gross_flow_error=num/den if den else None,missing_public_mark_records=dict(missing),
        separate_actions_preserved=True)


def timing_alerts(model,view,mask,threshold,cap):
    rows=np.flatnonzero(mask&view['timing_ready']);t=view['data']['timeline_ms'][rows]
    p=model.predict(t,view['aggregate'][rows]);budget=EntryBudget(threshold,cap);pp=[]
    for i,r in enumerate(rows):
        now=int(view['data']['timeline_ms'][r])
        if budget.accept(now,float(p[i])):pp.append(dict(symbol='ALL',action='open',issue_ms=now,time_ms=now+M,minute_index=int(r)))
    scoring=view['timing_ok'][rows]
    scored_pp=[e for e in pp if view['timing_ok'][e['minute_index']]]
    actual=[]
    for r in rows[(view['timing_y'][rows]>0)&scoring]:
        ee=[e for e in view['opens']if e['panel_row']%len(view['data']['timeline_ms'])==r]
        actual.append(dict(symbol='ALL',action='open',time_ms=min(e['time_ms']for e in ee),raw_openings=len(ee)))
    return dict(probability=probability_card(view['timing_y'][rows[scoring]],p[scoring]),primary=event_card(scored_pp,actual),
                sensitivity_5m=event_card(scored_pp,actual,tolerance=5*M),alerts=pp,actual=actual,unscored_label_ambiguity_alerts=len(pp)-len(scored_pp),
                raw_opening_events=sum(e['raw_openings']for e in actual),reject=dict(budget.reject),rows=rows[scoring],p=p[scoring])


def execute_due(pending,states,now,view,extra_delay,predictions,failed,end_by_state=None):
    n=len(view['data']['timeline_ms'])
    for key,p in list(pending.items()):
        if p['time_ms']>now:continue
        if end_by_state is not None and p['time_ms']>=end_by_state[key]:
            failed['administratively_censored_pending']+=1;del pending[key];continue
        row=p['panel_row']+extra_delay
        price=float(view['data']['next_open'][row])if row<len(view['data']['next_open'])and row//n==p['panel_row']//n else np.nan
        if np.isfinite(price)and price>0:
            states[key].apply(p['action'],p['quantity'],price,now)
            predictions.append({**p,'execution_price':price})
        else:failed['missing_execution_price']+=1
        del pending[key]


def conditional_management(view,manager,size,opens,start,stop,horizon,extra_delay=0):
    n=len(view['data']['timeline_ms']);by_cycle=defaultdict(list)
    for e in view['primary']+view['inferred']:by_cycle[e['cycle_id']].append(e)
    episodes=[];states=[];raw_actual=[]
    for e in opens:
        if not start<=e['time_ms']<stop:continue
        first_minute=(e['time_ms']//M+1)*M;end=min(stop,e['time_ms']+horizon*M)
        s=ManagedPosition();s.apply('open',float(e['qty']),float(e['price']),e['time_ms'])
        states.append(s);episode_id=e['cycle_id'];records=by_cycle[episode_id]
        episode=dict(episode_id=episode_id,symbol=e['symbol'],open_event_id=e['event_id'],
            source_open_ms=e['time_ms'],first_decision_ms=first_minute,end_ms=end,
            source_classification=e['classification'],true_first_qty=float(e['qty']),true_first_price=float(e['price']),
            right_censored_by_fold=end==stop,rows=[],own_qty=[],truth_qty=[],marks=[],record_ids=[r['event_id']for r in records])
        episodes.append(episode)
        raw_actual.extend({**r,'episode_id':episode_id}for r in records if r['action']!='open'and e['time_ms']<=r['time_ms']<end)
    pending={};pp=[];issued=[];failed=Counter();rejected=Counter()
    for minute in np.flatnonzero((view['data']['timeline_ms']>=start)&(view['data']['timeline_ms']<stop)):
        now=int(view['data']['timeline_ms'][minute]);ix=[];markets=[];contexts=[]
        for i,ep in enumerate(episodes):
            if not ep['first_decision_ms']<=now<ep['end_ms']:continue
            si=view['meta']['symbols'].index(ep['symbol']);r=si*n+int(minute)
            mark=float(view['features']['close'][r])
            if np.isfinite(mark):
                ep['rows'].append(r);ep['own_qty'].append(states[i].qty);ep['marks'].append(mark)
            if not states[i].qty:continue
            prev=float(view['features']['close'][r-1])if minute else np.nan
            context=states[i].context(now,mark,prev,float(view['features']['ret5'][r]))
            market=view['market'][r]
            if i in pending:rejected['pending']+=1;continue
            if not np.isfinite(manager.matrix(market[None,:],context[None,:])).all():rejected['missing_features']+=1;continue
            ix.append(i);markets.append(market);contexts.append(context)
        if ix:
            decisions=manager.actions(np.asarray(markets),np.asarray(contexts))
            for i,code in zip(ix,decisions):
                if not code:continue
                ep=episodes[i];si=view['meta']['symbols'].index(ep['symbol']);r=si*n+int(minute);a=ACTIONS[int(code)-1]
                p=dict(symbol=ep['symbol'],episode_id=ep['episode_id'],action=a,issue_ms=now,
                    time_ms=now+(1+extra_delay)*M,panel_row=r,quantity=size_for(a,states[i],float(view['features']['close'][r]),size))
                pending[i]=p;issued.append(p)
        # Equal-time executions never enter an earlier decision at that minute.
        execute_due(pending,states,now,view,extra_delay,pp,failed,[ep['end_ms']for ep in episodes])
    path_cards=[];episode_records=[]
    for ep,state in zip(episodes,states):
        times=view['data']['time_ms'][ep['rows']]
        truth=cycle_truth(by_cycle[ep['episode_id']],times)
        card=quantity_path_card(ep['own_qty'],truth,ep['marks'],size['open']['median'])
        selected=[p for p in pp if p['episode_id']==ep['episode_id']and p['time_ms']<ep['end_ms']]
        actual=[e for e in raw_actual if e['episode_id']==ep['episode_id']]
        card.update(residual_own_quantity=state.qty,observed_truth_quantity_at_censor=float(cycle_truth(by_cycle[ep['episode_id']],[ep['end_ms']])[0]),
                    administrative_horizon_minutes=horizon,right_censored_by_fold=ep['right_censored_by_fold'])
        path_cards.append(card)
        episode_records.append({k:v for k,v in ep.items()if k not in ('rows','own_qty','truth_qty','marks')})
    numerator=sum(c['absolute_exposure_error_sum']for c in path_cards);denominator=sum(c['truth_exposure_sum']for c in path_cards)
    matched=event_card(pp,raw_actual,True)
    return dict(episodes=len(episodes),primary=matched,sensitivity_5m=event_card(pp,raw_actual,True,5*M),
        events_by_action={a:event_card([p for p in pp if p['action']==a],[e for e in raw_actual if e['action']==a],True)for a in ACTIONS[1:]},
        path=dict(normalized_error=numerator/denominator if denominator else None,absolute_exposure_error_sum=numerator,
                  truth_exposure_sum=denominator,minutes=sum(c['minutes']for c in path_cards),
                  zero_truth_exposure_episodes=sum(c['truth_exposure_sum']==0 for c in path_cards),
                  residual_held_episodes=sum(c['residual_own_quantity']>0 for c in path_cards)),
        quantity_flow=flow_card(pp,raw_actual,view,size['open']['median'],start,stop),
        episode_paths=path_cards,episode_records=episode_records,predictions=pp,issued=issued,actual=raw_actual,
        rejection=dict(rejected),execution_failures=dict(failed),pending_at_end=len(pending),extra_delay=extra_delay,
        executable_entry=False,true_entry_supplied_once=True,no_true_close_stop=True)


def composed(view,timing,selector,manager,size,frequency,threshold,cap,start,stop,extra_delay=0):
    n=len(view['data']['timeline_ms']);ns=len(view['meta']['symbols']);first=int(np.searchsorted(view['data']['timeline_ms'],start))
    states=[state_from_past(si*n+first,start,view['data'],view['meta'],view['events'],view['extra'])for si in range(ns)]
    initial=[dict(symbol=s,quantity=p.qty,cost=p.cost,first_qty=p.first_qty,opened_ms=p.opened,last_action_ms=p.last,
                  add_count=p.adds,increase_price=p.increase_price,increased_ms=p.increased,worst_mark=None if not np.isfinite(p.worst)else p.worst)
             for s,p in zip(view['meta']['symbols'],states)]
    pending={};pp=[];issued=[];failed=Counter();rejected=Counter();budget=EntryBudget(threshold,cap)
    rows=[];qty=[];truth=[];marks=[]
    for minute in np.flatnonzero((view['data']['timeline_ms']>=start)&(view['data']['timeline_ms']<stop)):
        now=int(view['data']['timeline_ms'][minute]);rr=np.arange(ns)*n+int(minute)
        mark=view['features']['close'][rr]
        scoring=view['data']['state_known_mask'][rr]&np.isfinite(mark)
        rows.extend(rr[scoring].tolist());qty.extend(states[i].qty for i in np.flatnonzero(scoring))
        truth.extend(abs(float(view['features']['position'][r]))if view['data']['direction'][r]==-1 else 0. for r in rr[scoring])
        marks.extend(mark[scoring].tolist())
        active=[];contexts=[]
        for si,s in enumerate(states):
            if not s.qty:continue
            r=int(rr[si]);prev=float(view['features']['close'][r-1])if minute else np.nan
            state=s.context(now,float(mark[si]),prev,float(view['features']['ret5'][r]))
            if si in pending:continue
            if not np.isfinite(manager.matrix(view['market'][r:r+1],state[None,:])).all():continue
            active.append(si);contexts.append(state)
        if active:
            decisions=manager.actions(view['market'][rr[active]],np.asarray(contexts))
            for si,code in zip(active,decisions):
                if not code:continue
                a=ACTIONS[int(code)-1];r=int(rr[si]);q=size_for(a,states[si],float(mark[si]),size)
                p=dict(symbol=view['meta']['symbols'][si],action=a,issue_ms=now,time_ms=now+(1+extra_delay)*M,panel_row=r,quantity=q)
                pending[si]=p;issued.append(p)
        if view['timing_ready'][minute]:
            probability=float(timing.predict(np.array([now]),view['aggregate'][minute:minute+1])[0])
            if budget.accept(now,probability):
                eligible=np.isfinite(view['market'][rr]).all(axis=1)&np.isfinite(mark)&(mark>0)
                eligible&=np.array([s.qty==0 and si not in pending for si,s in enumerate(states)])
                scores=selector.score(view['market'][rr[eligible]])if selector is not None and eligible.any()else None
                rank=np.full(ns,-np.inf)
                if selector is None:rank=np.asarray(frequency,float)
                elif eligible.any():rank[eligible]=scores
                si=select_coin(view['meta']['symbols'],rank,now,eligible)
                if si is None:rejected['no_own_flat_public_coin']+=1
                else:
                    r=int(rr[si]);q=size_for('open',states[si],float(mark[si]),size)
                    p=dict(symbol=view['meta']['symbols'][si],action='open',issue_ms=now,time_ms=now+(1+extra_delay)*M,panel_row=r,
                           quantity=q,timing_probability=probability)
                    pending[si]=p;issued.append(p)
        execute_due(pending,states,now,view,extra_delay,pp,failed)
    actual=[e for e in view['primary']if start<=e['time_ms']<stop]
    # Matching covers all identifiable raw records, including multi-event minutes.
    card=quantity_path_card(qty,truth,marks,size['open']['median'])
    scored_rows=np.asarray(rows,int);qty_array=np.asarray(qty);truth_array=np.asarray(truth);mark_array=np.asarray(marks)
    card['by_symbol']={symbol:quantity_path_card(qty_array[(scored_rows//n)==si],truth_array[(scored_rows//n)==si],mark_array[(scored_rows//n)==si],size['open']['median']) for si,symbol in enumerate(view['meta']['symbols'])}
    turnover=sum(p['quantity']*p['execution_price']for p in pp)
    card.update(residual_symbols=sum(s.qty>0 for s in states),virtual_turnover_quote=turnover,
                fee_stress_quote={str(b)+'bps':turnover*b/10000 for b in (0,5,10)},account_return_not_computed=True)
    return dict(primary=event_card(pp,actual),sensitivity_5m=event_card(pp,actual,tolerance=5*M),
        events_by_action={a:event_card([p for p in pp if p['action']==a],[e for e in actual if e['action']==a])for a in ACTIONS},
        path=card,quantity_flow=flow_card(pp,actual,view,size['open']['median'],start,stop),initial_positions=initial,
        initial_held_symbols=sum(x['quantity']>0 for x in initial),predictions=pp,issued=issued,actual=actual,
        timing_rejection=dict(budget.reject),rejection=dict(rejected),execution_failures=dict(failed),pending_at_end=len(pending),extra_delay=extra_delay,
        executable_entry=True,true_state_after_initialization_used=False)


def strip_archive(value):
    keys=('predictions','issued','actual')
    archived={k:value.pop(k)for k in keys if k in value}
    return archived


def main():
    protocol=read(OUT/'protocol.json');protocol_sha=sha(OUT/'protocol.json');protected=read(OUT/'source_hashes.json')
    for name,digest in protected.items():assert sha(ROOT/name)==digest,name
    view=load_view();data=view['data'];times=data['timeline_ms'];n=len(times);ns=len(view['meta']['symbols'])
    write('observation_audit.json',dict(raw_primary_events=len(view['primary']),raw_primary_openings=len(view['opens']),
        opening_minutes=int(view['timing_y'].sum()),multi_coin_open_minutes=[dict(time_ms=int(times[i]),raw_openings=int(view['open_count'][i]))for i in np.flatnonzero(view['open_count']>1)],
        same_symbol_multiple_groups=view['ambiguous'],inferred_openings=view['inferred'],
        timing_minutes=int(view['timing_ok'].sum()),manager_eligible_minutes=int(view['held'].sum()),
        target='identifiable saved successful events; not hidden intention'))
    folds=[];models={};archive={};score_arrays={}
    for fold in protocol['folds']:
        lo,mid,hi,end=[stamp(fold[k])for k in ('train_start','calibration_start','train_end','eval_end')]
        tm={'fit':(times>=lo)&(times<mid),'cal':(times>=mid)&(times<hi),'eval':(times>=hi)&(times<end)}
        masks={part:np.tile(mask,ns)for part,mask in tm.items()}
        size,duration=sizes_and_duration(view,masks['fit'],lo,mid)
        rank_fit=masks['fit']&view['rank_ok'];rank_cal=masks['cal']&view['rank_ok']
        selector=EntrySelector().fit(view['market'],view['rank_y'],rank_fit,rank_cal)
        models[fold['name']+'_selector']=selector.details
        fit_open=[e for e in view['opens']if lo<=e['time_ms']<mid]
        frequency=np.array([sum(e['symbol']==s for e in fit_open)for s in view['meta']['symbols']],float)
        rank_diag=[]
        for part in ('cal','eval'):
            opening=[e for e in view['opens']if tm[part][e['panel_row']%n]]
            for selector_id in ('fit_frequency','market_rank'):
                top1=0;top3=0;excluded=0
                for e in opening:
                    rr=np.arange(ns)*n+e['panel_row']%n;eligible=np.isfinite(view['market'][rr]).all(axis=1)
                    target=view['meta']['symbols'].index(e['symbol'])
                    if not eligible[target]:excluded+=1;continue
                    scores=frequency if selector_id=='fit_frequency'else np.full(ns,-np.inf)
                    if selector_id=='market_rank':scores[eligible]=selector.score(view['market'][rr[eligible]])
                    order=sorted(np.flatnonzero(eligible),key=lambda i:(-scores[i],view['meta']['symbols'][i]))
                    top1+=target in order[:1];top3+=target in order[:3]
                rank_diag.append(dict(part=part,selector=selector_id,events=len(opening),excluded_missing_market=excluded,
                                      top1=top1,top3=top3,true_timing_oracle=True,executable_timing=False))
        fc=dict(fold=fold['name'],fit_size=size,fit_median_duration_minutes=duration,rank_diagnostic=rank_diag,timing=[],management=[],combinations=[])
        timing_models={};timing_parameters={}
        for variant in protocol['timing_variants']:
            model=EntryTiming(variant).fit(times,view['aggregate'],view['timing_y'],tm['fit']&view['timing_ok'],tm['cal']&view['timing_ok'])
            rows=np.flatnonzero(tm['cal']&view['timing_ok']);p=model.predict(times[rows],view['aggregate'][rows])
            rate=float(view['timing_y'][tm['fit']&view['timing_ok']].mean())
            threshold=float(np.quantile(p,1-rate));cap=int(np.ceil(2*int(view['timing_y'][tm['fit']&view['timing_ok']].sum())/((mid-lo)/(1440*M))))
            timing_models[variant]=model;timing_parameters[variant]=(threshold,cap)
            result=timing_alerts(model,view,tm['eval'],threshold,cap)
            score_arrays[fold['name']+'_'+variant+'_timing_rows']=result.pop('rows');score_arrays[fold['name']+'_'+variant+'_timing_p']=result.pop('p')
            archive[fold['name']+'_'+variant+'_timing']=dict(alerts=result.pop('alerts'),actual=result.pop('actual'))
            result.update(variant=variant,threshold=threshold,daily_cap=cap)
            fc['timing'].append(result);models[fold['name']+'_'+variant+'_timing']=model.details
        managers={}
        for variant in protocol['manager_variants']:
            manager=Manager(variant,duration).fit(view['market'],view['state'],view['label'],masks['fit']&view['held'],masks['cal']&view['held'])
            managers[variant]=manager;models[fold['name']+'_'+variant+'_manager']=manager.details
            if variant!='duration':
                rr=np.flatnonzero(masks['eval']&view['held']);p=manager.predict(view['market'][rr],view['state'][rr])
                score_arrays[fold['name']+'_'+variant+'_held_rows']=rr;score_arrays[fold['name']+'_'+variant+'_held_p']=p
                fc.setdefault('local_management_probability',[]).append(dict(variant=variant,any_operation=probability_card(view['label'][rr]>1,p.sum(axis=1)),
                    by_action={a:probability_card(view['label'][rr]==j+2,p[:,j])for j,a in enumerate(ACTIONS[1:])}))
            for delay in protocol['extra_execution_delay_minutes']:
                result=conditional_management(view,manager,size,view['opens'],hi,end,protocol['episode_horizon_minutes'],delay)
                archive[fold['name']+'_'+variant+'_conditional_delay'+str(delay)]=strip_archive(result)
                result['variant']=variant;fc['management'].append(result)
                print('CONDITIONAL',fold['name'],variant,delay,'path',result['path']['normalized_error'],'events',result['primary']['tp'],result['primary']['fp'],result['primary']['fn'],flush=True)
        # Uncertain right-edge opening is a separate observation sensitivity.
        tail=conditional_management(view,managers['market_state'],size,view['inferred'],hi,end,protocol['episode_horizon_minutes'],0)
        archive[fold['name']+'_inferred_right_edge']=strip_archive(tail);fc['inferred_right_edge_sensitivity']=tail
        for config in protocol['combinations']:
            threshold,cap=timing_parameters[config['timing']]
            for delay in protocol['extra_execution_delay_minutes']:
                result=composed(view,timing_models[config['timing']],selector if config['selector']=='market_rank'else None,
                    managers[config['manager']],size,frequency,threshold,cap,hi,end,delay)
                archive[fold['name']+'_'+config['id']+'_composed_delay'+str(delay)]=strip_archive(result)
                result.update(id=config['id']);fc['combinations'].append(result)
                print('COMPOSED',fold['name'],config['id'],delay,'path',result['path']['normalized_error'],'events',result['primary']['tp'],result['primary']['fp'],result['primary']['fn'],flush=True)
        clock,market=fc['timing'];tgain=1-market['probability']['log_loss']/clock['probability']['log_loss']
        clockrows=score_arrays[fold['name']+'_clock_timing_rows'];marketrows=score_arrays[fold['name']+'_market_clock_timing_rows'];assert np.array_equal(clockrows,marketrows)
        daydata={'time_ms':times};market['loss_comparison']=dict(relative_gain=tgain,uncertainty=bootstrap_days(daydata,marketrows,view['timing_y'][marketrows],score_arrays[fold['name']+'_market_clock_timing_p'],score_arrays[fold['name']+'_clock_timing_p']))
        conditional={x['variant']:x for x in fc['management']if x['extra_delay']==0}
        path=conditional['market_state']['path']['normalized_error'];references=[conditional[v]['path']['normalized_error']for v in ('duration','state_clock')]
        mgain=1-path/min(references)if min(references)>0 else 0.
        baseline=next(x for x in fc['combinations']if x['id']=='baseline'and x['extra_delay']==0)
        candidate=next(x for x in fc['combinations']if x['id']=='candidate'and x['extra_delay']==0)
        cgain=1-candidate['path']['normalized_error']/baseline['path']['normalized_error']if baseline['path']['normalized_error']else 0.
        p=protocol['progression'];eventpass=lambda cards,actions,minimum:all(cards[a]['precision']>=minimum and cards[a]['recall']>=minimum for a in actions)
        largest=Counter(e['symbol'] for e in view['primary']if hi<=e['time_ms']<end).most_common(1)[0][0]
        cn=candidate['path']['absolute_exposure_error_sum']-candidate['path']['by_symbol'][largest]['absolute_exposure_error_sum']
        bn=baseline['path']['absolute_exposure_error_sum']-baseline['path']['by_symbol'][largest]['absolute_exposure_error_sum']
        leave_gain=1-cn/bn if bn>0 else None
        timing_pass=tgain>=p['timing_relative_log_loss_gain']and p['timing_event_mass_ratio'][0]<=market['probability']['event_mass_ratio']<=p['timing_event_mass_ratio'][1] and market['primary']['precision']>=p['timing_precision_recall_min'] and market['primary']['recall']>=p['timing_precision_recall_min']
        management_pass=path is not None and path<=p['conditional_path_error_max']and mgain>=p['conditional_relative_path_gain_vs_both_refs']and eventpass(conditional['market_state']['events_by_action'],ACTIONS[1:],p['management_each_action_precision_recall_min']) and conditional['market_state']['episodes']>=p['min_episodes'] and sum(conditional['market_state']['primary'][k]for k in ('tp','fn'))>=p['min_management_events']
        combination_pass=candidate['path']['normalized_error']<=p['combined_path_error_max']and cgain>=p['combined_relative_path_gain']and eventpass(candidate['events_by_action'],ACTIONS,p['combined_each_action_precision_recall_min']) and leave_gain is not None and leave_gain>0
        fc['gates']=dict(timing=bool(timing_pass),management=bool(management_pass),combination=bool(combination_pass),all=bool(timing_pass and management_pass and combination_pass),timing_log_loss_gain=tgain,management_path_gain_vs_best_reference=mgain,combined_path_gain=cgain,largest_event_symbol=largest,combined_path_gain_without_largest=leave_gain)
        folds.append(fc)
    for name,digest in protected.items():assert sha(ROOT/name)==digest,name
    assert sha(OUT/'protocol.json')==protocol_sha
    result=dict(version=protocol['version'],status='completed_exploratory_decoupling',protocol_sha256=protocol_sha,
        old_protected_files_unchanged=len(protected),budget=protocol['budget'],folds=folds,
        gates={k:sum(f['gates'][k]for f in folds)for k in ('timing','management','combination','all')},
        implementation_execution=dict(accepted_learned_heads=21,total_executed_heads=42,repair_reruns=1,distinct_model_configurations_unchanged=True,first_attempt_preserved='implementation_attempt_1',reason='Separate public timing readiness from future label ambiguity scoring mask'),
        decision='No independent behavior validation; fixed study complete, no search expansion.',limits=protocol['limits'])
    write('results.json',result);write('models.json',models);write('predictions.json',archive)
    np.savez_compressed(OUT/'scores.npz',**score_arrays)
    print('COMPLETED',result['gates'],'original files unchanged',len(protected),flush=True)


if __name__=='__main__':main()
