"""Information-boundary, administrative-censor and raw-event preservation tests."""
import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.special import softmax,expit
from behavior_hazard import M,MARKET,POSITION
from entry_management import (public_timing_matrix,ManagedPosition,Manager,EntryBudget,cycle_truth,quantity_path_card)
from run_entry_management import conditional_management,composed,event_card


class FakeManager:
    variant='duration'
    def matrix(self,market,state):return state[:,:1]
    def actions(self,market,state):return np.where(state[:,0]>=3,4,0)


class FakeTiming:
    def predict(self,times,aggregate):return (np.asarray(times)==2*M).astype(float)


class FakeSelector:
    def score(self,market):return market[:,0]


def fixture(same_minute=False,two_coins=False):
    ns,n=2,16;times=np.arange(n)*M
    names=list(MARKET)+['close','avgcost','position','first_qty']+list(POSITION)
    x=np.zeros((ns*n,len(names)));x[:,names.index('close')]=10.
    features={name:x[:,i]for i,name in enumerate(names)}
    features['ret5'][:]=.01
    market=np.zeros((ns*n,14));market[:n,0]=1.
    data=dict(X=x,timeline_ms=times,time_ms=np.tile(times,ns),symbol_idx=np.repeat(np.arange(ns),n),
              direction=np.zeros(ns*n,int),state_known_mask=np.ones(ns*n,bool),next_open=np.full(ns*n,11.))
    opened=2*M+10000;closed=opened+10000 if same_minute else 7*M+10000
    def event(symbol,action,now,index,cycle):
        return dict(event_id=f'e{index}',source_index=index,symbol=symbol,action=action,direction='SHORT',
                    time_ms=now,qty='2',price='10',classification='history_anchored',cycle_id=cycle,
                    panel_row=(0 if symbol=='A'else n)+int(now//M))
    events=[event('A','open',opened,1,'a'),event('A','close',closed,2,'a')]
    if two_coins:events.extend([event('B','open',opened+1000,3,'b'),event('B','close',closed+1000,4,'b')])
    for si,symbol in enumerate(['A','B']):
        for row in range(si*n,(si+1)*n):
            t=int(data['time_ms'][row]);q=cycle_truth([e for e in events if e['symbol']==symbol],[t])[0]
            if q:
                data['direction'][row]=-1;features['position'][row]=-q;features['avgcost'][row]=10;features['first_qty'][row]=2
    size={a:dict(median=v)for a,v in [('open',20.),('add',1.),('reduce',.5)]}
    view=dict(data=data,meta=dict(symbols=['A','B'],feature_names=names),market=market,
              aggregate=np.zeros((n,10)),timing_ready=np.ones(n,bool),timing_ok=np.ones(n,bool),features=features,extra=np.zeros((ns*n,11)),
              events=events,primary=events,inferred=[],opens=[e for e in events if e['action']=='open'])
    return view,size


class TestDecoupling(unittest.TestCase):
    def test_public_aggregation_has_no_future_or_true_position_argument(self):
        rng=np.random.default_rng(1);market=rng.normal(size=(21*20,14));a,_=public_timing_matrix(market,21,20)
        changed=market.copy().reshape(21,20,14);changed[:,12:,:]=1000.
        b,_=public_timing_matrix(changed.reshape(-1,14),21,20)
        np.testing.assert_array_equal(a[:12],b[:12])

    def test_real_close_does_not_end_or_correct_conditional_policy(self):
        view,size=fixture();a=conditional_management(view,FakeManager(),size,view['opens'],0,16*M,8)
        view['primary'][1]['time_ms']=14*M+10000
        b=conditional_management(view,FakeManager(),size,view['opens'],0,16*M,8)
        self.assertEqual(a['issued'],b['issued'])
        self.assertEqual(a['episode_records'][0]['end_ms'],view['opens'][0]['time_ms']+8*M)
        self.assertNotEqual(a['path']['normalized_error'],b['path']['normalized_error'])

    def test_same_minute_open_close_preserved_as_unobservable_management_event(self):
        view,size=fixture(True);r=conditional_management(view,FakeManager(),size,view['opens'],0,16*M,8)
        self.assertEqual(r['episodes'],1)
        self.assertEqual(r['primary']['fn'],1)
        self.assertEqual(r['path']['zero_truth_exposure_episodes'],1)
        self.assertGreater(r['path']['absolute_exposure_error_sum'],0)

    def test_two_coin_openings_score_missing_second_coin(self):
        view,size=fixture(two_coins=True)
        r=composed(view,FakeTiming(),FakeSelector(),FakeManager(),size,[1,1],.5,2,0,16*M)
        c=r['events_by_action']['open']
        self.assertEqual(c['tp'],1);self.assertEqual(c['fn'],1)
        self.assertEqual(len([e for e in r['actual']if e['action']=='open']),2)

    def test_future_label_ambiguity_cannot_gate_composed_opening(self):
        view,size=fixture();a=composed(view,FakeTiming(),FakeSelector(),FakeManager(),size,[1,1],.5,2,0,16*M)
        view['timing_ok'][2]=False
        b=composed(view,FakeTiming(),FakeSelector(),FakeManager(),size,[1,1],.5,2,0,16*M)
        self.assertEqual(a['issued'],b['issued'])

    def test_future_truth_positions_do_not_change_composed_decisions(self):
        view,size=fixture();a=composed(view,FakeTiming(),FakeSelector(),FakeManager(),size,[1,1],.5,2,0,16*M)
        n=16
        for si in range(2):
            rows=np.arange(si*n+3,(si+1)*n)
            view['data']['direction'][rows]=-1
            for key in ['position','avgcost','first_qty']+list(POSITION):view['features'][key][rows]=77.
            view['extra'][rows]=999.
        b=composed(view,FakeTiming(),FakeSelector(),FakeManager(),size,[1,1],.5,2,0,16*M)
        self.assertEqual(a['issued'],b['issued']);self.assertEqual(a['predictions'],b['predictions'])

    def test_future_execution_price_cannot_size_or_filter_earlier_open(self):
        view,size=fixture();view['data']['next_open'][2]=np.nan
        r=composed(view,FakeTiming(),FakeSelector(),FakeManager(),size,[1,1],.5,2,0,16*M)
        self.assertEqual(r['issued'][0]['quantity'],2.)
        self.assertEqual(r['issued'][0]['issue_ms'],2*M)
        self.assertEqual(r['execution_failures']['missing_execution_price'],1)

    def test_administrative_censor_prevents_after_window_execution_credit(self):
        view,size=fixture();r=conditional_management(view,FakeManager(),size,view['opens'],0,16*M,4,1)
        end=r['episode_records'][0]['end_ms']
        self.assertTrue(all(p['time_ms']<end for p in r['predictions']))
        self.assertEqual(r['execution_failures'].get('administratively_censored_pending'),1)

    def test_boundary_initialization_used_once(self):
        view,size=fixture();row=0;view['data']['direction'][row]=-1
        for name,value in [('position',-2),('avgcost',10),('first_qty',2),('age_minutes',2),('minutes_last_action',2),('mark_return',0)]:view['features'][name][row]=value
        view['events'].append(dict(symbol='A',time_ms=-2*M,classification='history_anchored',direction='SHORT',action='open',price='10'))
        r=composed(view,FakeTiming(),FakeSelector(),FakeManager(),size,[1,1],.5,2,0,16*M)
        self.assertEqual(r['initial_held_symbols'],1)
        self.assertFalse(r['true_state_after_initialization_used'])

    def test_own_increase_anchor_and_worst_mark(self):
        s=ManagedPosition();s.apply('open',2,10,0)
        c=s.context(M,12,11,.01);self.assertAlmostEqual(c[5],.2);self.assertAlmostEqual(c[9],0.)
        c=s.context(2*M,11,12,.01);self.assertAlmostEqual(c[9],.1)
        s.apply('add',2,14,2*M);c=s.context(3*M,15,11,.01)
        self.assertAlmostEqual(c[5],15/14-1);self.assertAlmostEqual(c[6],1.)

    def test_event_matching_forbids_already_observed_event(self):
        p=dict(symbol='A',action='close',issue_ms=3*M,time_ms=4*M)
        a=dict(symbol='A',action='close',time_ms=2*M)
        self.assertEqual(event_card([p],[a],tolerance=5*M)['tp'],0)

    def test_timing_budget_never_uses_end_of_day_ranking(self):
        b=EntryBudget(.2,1)
        self.assertTrue(b.accept(0,.3));self.assertFalse(b.accept(M,.9))
        self.assertTrue(b.accept(1440*M,.3))

    def test_saved_manager_coefficients_reconstruct_competing_probabilities(self):
        rng=np.random.default_rng(3);n=1800;market=rng.normal(size=(n,14)).astype(np.float32);state=rng.normal(size=(n,10)).astype(np.float32)
        labels=np.zeros(n,int);labels[np.arange(20,1600,40)]=np.resize([2,3,4],40)
        fit=np.arange(n)<1300;cal=~fit
        manager=Manager('market_state',10).fit(market,state,labels,fit,cal);d=manager.details
        x=np.column_stack([market[cal],state[cal]]);z=x.copy();z-=np.array(d['action_scaler_mean']);z/=np.array(d['action_scaler_scale'])
        conditional=softmax(z@np.array(d['action_coefficients']).T+np.array(d['action_intercept']),axis=1)
        h=d['head'];z=x.copy();z-=np.array(h['scaler_mean']);z/=np.array(h['scaler_scale'])
        raw=expit(z@np.array(h['coefficients'][0])+h['intercept'][0]);raw=np.clip(raw,1e-12,1-1e-12)
        hazard=expit(np.log(raw/(1-raw))+h['calibration_offset'])
        np.testing.assert_allclose(manager.predict(market[cal],state[cal]),conditional*hazard[:,None],rtol=1e-10)


if __name__=='__main__':unittest.main()
