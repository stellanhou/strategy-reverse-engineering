"""v5 executable public entry timing and own-state management components."""
from dataclasses import dataclass
from collections import Counter
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax

from behavior_hazard import BinaryHead, Position, M, stable_key


def clock_features(times):
    angle=(np.asarray(times)%(1440*M))/(1440*M)*2*np.pi
    return np.column_stack([np.sin(angle),np.cos(angle)])


def public_timing_matrix(market,n_symbols,n_minutes):
    """All available public coins, never trader/true-flat availability."""
    cube=np.asarray(market).reshape(n_symbols,n_minutes,-1).transpose(1,0,2)
    available=np.isfinite(cube).all(axis=2)
    rows=np.full((n_minutes,10),np.nan)
    for i in range(n_minutes):
        a=cube[i,available[i]]
        if not len(a):continue
        rows[i]=[np.mean(a[:,1]),np.max(a[:,1]),np.max(a[:,7]),np.max(a[:,10]),
                 np.max(a[:,11]),np.mean(a[:,1]>0),np.max(a[:,3]),np.mean(a[:,4]),
                 np.median(a[:,12]),len(a)/n_symbols]
    return rows,available.sum(axis=1)


class EntryTiming:
    def __init__(self,variant):self.variant=variant
    def matrix(self,times,aggregate):
        clock=clock_features(times)
        return np.column_stack([clock,aggregate]) if self.variant=='market_clock' else clock
    def fit(self,times,aggregate,labels,fit,cal):
        x=self.matrix(times,aggregate)
        self.head=BinaryHead().fit(x[fit],labels[fit])
        self.head.calibrate(x[cal],labels[cal])
        self.details=dict(variant=self.variant,head=self.head.details)
        return self
    def predict(self,times,aggregate):return self.head.predict(self.matrix(times,aggregate))


class EntrySelector:
    """Real entry times supervise fit/cal rows, never appear as an input."""
    def fit(self,market,labels,fit,cal):
        self.head=BinaryHead().fit(market[fit],labels[fit])
        self.head.calibrate(market[cal],labels[cal])
        self.details=dict(head=self.head.details,rank_probability_not_joint_event_probability=True)
        return self
    def score(self,market):return self.head.predict(market)


class Manager:
    def __init__(self,variant,duration):self.variant,self.duration=variant,float(duration)
    def matrix(self,market,state):
        if self.variant=='duration':return state[:,:1]
        return np.column_stack([market,state]) if self.variant=='market_state' else state[:,:5]
    def fit(self,market,state,labels,fit,cal):
        if self.variant=='duration':
            self.details=dict(variant='duration',duration_minutes=self.duration)
            return self
        x=self.matrix(market,state)
        self.head=BinaryHead().fit(x[fit],labels[fit]>1)
        self.head.calibrate(x[cal],labels[cal]>1)
        positive=fit&(labels>1)
        self.classes=np.unique(labels[positive])
        self.scaler=StandardScaler().fit(x[positive])
        self.model=LogisticRegression(C=.1,class_weight=None,solver='lbfgs',max_iter=1000,random_state=20261003)
        self.model.fit(self.scaler.transform(x[positive]),labels[positive])
        self.details=dict(variant=self.variant,head=self.head.details,action_classes=self.classes.tolist(),
            action_coefficients=self.model.coef_.tolist(),action_intercept=self.model.intercept_.tolist(),
            action_scaler_mean=self.scaler.mean_.tolist(),action_scaler_scale=self.scaler.scale_.tolist(),
            action_fit_events=int(positive.sum()))
        p=self.predict(market[cal],state[cal]).sum(axis=1)
        rate=float((labels[fit]>1).mean())
        self.threshold=float(np.quantile(p,1-rate))
        self.details['threshold']=self.threshold
        return self
    def predict(self,market,state):
        if self.variant=='duration':
            out=np.zeros((len(market),3));out[:,2]=(state[:,0]>=self.duration)
            return out
        x=self.matrix(market,state)
        h=self.head.predict(x)
        conditional=self.model.predict_proba(self.scaler.transform(x))
        out=np.zeros((len(x),3))
        for j,code in enumerate(self.classes):out[:,int(code)-2]=h*conditional[:,j]
        return out
    def actions(self,market,state):
        p=self.predict(market,state)
        threshold=.5 if self.variant=='duration' else self.threshold
        return np.where(p.sum(axis=1)>=threshold,np.argmax(p,axis=1)+2,0)


@dataclass
class ManagedPosition(Position):
    increase_price:float=0.
    increased:int=0
    worst:float=np.inf
    def apply(self,action,quantity,price,now):
        super().apply(action,quantity,price,now)
        if action=='open':self.worst=np.inf
        if action in ('open','add'):self.increase_price,self.increased=price,now
    def context(self,now,mark,previous_close,ret5):
        basic=self.features(now,mark)
        if not self.qty:return np.full(10,np.nan)
        current=basic[3]
        self.worst=min(self.worst,current)
        extra=[mark/self.increase_price-1,(now-self.increased)/M,
               -(mark-previous_close)/self.cost,-(mark-mark/(1+ret5))/self.cost,current-self.worst]
        return np.r_[basic,extra]


class EntryBudget:
    def __init__(self,threshold,limit):
        self.threshold,self.limit=threshold,int(limit);self.used=Counter();self.reject=Counter()
    def accept(self,now,probability):
        if probability<self.threshold:self.reject['below_threshold']+=1;return False
        day=now//(1440*M)
        if self.used[day]>=self.limit:self.reject['daily_cap']+=1;return False
        self.used[day]+=1;return True


def select_coin(symbols,scores,now,eligible):
    choices=np.flatnonzero(eligible)
    if not len(choices):return None
    return min(choices,key=lambda i:(-scores[i],stable_key(symbols[i],'open',now)))


def size_for(action,position,mark,size):
    if action=='open':return size['open']['median']/mark
    if action=='add':return size['add']['median']*position.first_qty
    if action=='reduce':return min(size['reduce']['median'],1.)*position.qty
    if action=='close':return position.qty
    raise ValueError(action)


def state_from_past(row,now,data,meta,events,extra):
    """One observable boundary initialization. No later truth reset."""
    names=meta['feature_names'];x=data['X'][row]
    s=ManagedPosition()
    if not data['state_known_mask'][row] or data['direction'][row]!=-1:return s
    get=lambda name:float(x[names.index(name)])
    s.qty=abs(get('position'));s.cost=get('avgcost');s.first_qty=get('first_qty')
    s.adds=int(get('add_count'));s.opened=now-int(get('age_minutes'))*M
    s.last=now-int(get('minutes_last_action'))*M
    symbol=meta['symbols'][int(data['symbol_idx'][row])]
    prior=[e for e in events if e['symbol']==symbol and e['time_ms']<now and e['classification']=='history_anchored'
           and e['direction']=='SHORT' and e['action'] in ('open','add')]
    if not prior:raise ValueError('held boundary has no past increase')
    increase=max(prior,key=lambda e:e['time_ms']);s.increase_price=float(increase['price']);s.increased=increase['time_ms']
    if np.isfinite(extra[row,10]):s.worst=get('mark_return')-float(extra[row,10])
    return s


def quantity_path_card(predicted,truth,marks,scale):
    predicted,truth,marks=map(lambda a:np.asarray(a,float),(predicted,truth,marks))
    own,actual=predicted*marks/scale,truth*marks/scale
    numerator=float(np.abs(own-actual).sum());denominator=float(np.abs(actual).sum())
    union=int(((own>0)|(actual>0)).sum())
    return dict(minutes=len(own),absolute_exposure_error_sum=numerator,truth_exposure_sum=denominator,
        normalized_error=numerator/denominator if denominator else None,
        mean_absolute_scaled_error=numerator/len(own) if len(own) else None,
        held_intersection=int(((own>0)&(actual>0)).sum()),held_union=union,
        held_iou=float(((own>0)&(actual>0)).sum()/union) if union else 1.,
        predicted_peak=float(own.max()) if len(own) else 0.,actual_peak=float(actual.max()) if len(actual) else 0.)


def cycle_truth(events,times):
    """Raw operation quantity path, strict past; not a future close mask."""
    ee=sorted(events,key=lambda e:(e['time_ms'],e['source_index']))
    delta=[float(e['qty'])*(1 if e['action'] in ('open','add') else -1) for e in ee]
    cumulative=np.cumsum(delta)
    at=np.searchsorted([e['time_ms']for e in ee],times,side='left')-1
    q=np.zeros(len(times));valid=at>=0;q[valid]=cumulative[at[valid]]
    if np.any(q < -1e-6):raise ValueError('negative reconstructed SHORT episode')
    return np.maximum(q,0.)
