"""Independent invariants and adversarial information-boundary tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from scipy.special import expit, logit
from behavior_hazard import (M, MARKET, POSITION, Forecast, Position, SizeModel, OnlineDecoder,
                             corrected_sample, offset_calibration, probability_card, causal_match, path_card)
from run_behavior_hazard import autonomous


def small_panel():
    names=list(MARKET)+['close','avgcost','position','first_qty']+list(POSITION)
    n=12
    x=np.zeros((n,len(names)),dtype=float)
    x[:,names.index('close')]=10.
    return (dict(X=x, timeline_ms=np.arange(n)*M, next_open=np.full(n,11.),
                 time_ms=np.arange(n)*M, symbol_idx=np.zeros(n,int), direction=np.zeros(n,int)),
            dict(symbols=['A'],feature_names=names))


class ScriptedForecast:
    def predict(self, market, state, held):
        out=np.zeros((len(market),4))
        out[~held,0]=.9
        for i in np.flatnonzero(held):
            out[i,1 if state[i,2]==0 else 3]=.9
        return out


class TestProbability(unittest.TestCase):
    def test_inverse_sampling_reproduces_natural_counts(self):
        y=np.r_[np.ones(7),np.zeros(997)]
        rows,w=corrected_sample(y,17)
        self.assertEqual(len(rows),24)
        self.assertAlmostEqual(w[y[rows]==1].sum(),7)
        self.assertAlmostEqual(w[y[rows]==0].sum(),997)
        self.assertAlmostEqual(w.sum(),1004)

    def test_intercept_calibration_has_one_degree_and_preserves_ranking(self):
        p=np.linspace(.1,.9,2000);y=np.zeros(2000,int);y[[1,100,999]]=1
        offset=offset_calibration(p,y)
        calibrated=expit(logit(p)+offset)
        self.assertTrue(np.all(np.diff(calibrated)>0))
        self.assertAlmostEqual(calibrated.mean(),3.5/2001,places=12)

    def test_metrics_reference_arithmetic(self):
        c=probability_card([0,1],[.25,.75])
        self.assertAlmostEqual(c['log_loss'],-np.log(.75))
        self.assertAlmostEqual(c['brier'],.0625)
        self.assertEqual(c['expected_events'],1.)

    def test_competing_probabilities_and_illegal_actions(self):
        rng=np.random.default_rng(4);n=2000
        market=rng.normal(size=(n,8));state=rng.normal(size=(n,5))
        flat=np.arange(n)<1000;held=~flat
        labels=np.zeros(n,int);labels[[5,105,205]]=1
        labels[[1005,1105,1205,1305,1405,1505]]=np.tile([2,3,4],2)
        f=Forecast('market_state').fit(market,state,labels,flat,held)
        f.calibrate(market,state,labels,flat,held)
        p=f.predict(market,state,held)
        self.assertTrue(np.all((p>=0)&(p<=1)))
        self.assertTrue(np.all(p.sum(axis=1)<=1+1e-12))
        self.assertTrue(np.all(p[flat,1:]==0))
        self.assertTrue(np.all(p[held,0]==0))


class TestLedger(unittest.TestCase):
    def test_hand_computed_replay(self):
        s=Position();s.apply('open',100,10,0);s.apply('add',50,14,M)
        self.assertEqual(s.qty,150)
        self.assertAlmostEqual(s.cost,34/3)
        s.apply('reduce',37.5,12,2*M)
        self.assertEqual(s.qty,112.5)
        self.assertAlmostEqual(s.cost,34/3)
        s.apply('close',112.5,9,3*M)
        self.assertEqual(s.qty,0)

    def test_invalid_actions_raise(self):
        s=Position()
        for a in ('add','reduce','close'):
            with self.assertRaises(ValueError):s.apply(a,1,10,0)
        s.apply('open',2,10,0)
        with self.assertRaises(ValueError):s.apply('open',1,10,1)
        with self.assertRaises(ValueError):s.apply('reduce',3,10,1)
        with self.assertRaises(ValueError):s.apply('close',1,10,1)

    def test_sizing_uses_visible_price_and_own_quantity(self):
        size=SizeModel(dict(open=[100,200,300],add=[.5,1,2],reduce=[.2,.4,.8]))
        s=Position(qty=50,first_qty=20)
        self.assertEqual(size.quantity('open',s,10),20)
        self.assertEqual(size.quantity('add',s,10),20)
        self.assertEqual(size.quantity('reduce',s,10),20)
        self.assertEqual(size.quantity('close',s,10),50)

    def test_path_penalizes_false_positions_on_truth_flat_minutes(self):
        c=path_card([1,1,1],[0,1,0],np.ones(3),1)
        self.assertEqual(c['normalized_absolute_error'],2)
        self.assertAlmostEqual(c['held_overlap_iou'],1/3)


class TestDecoder(unittest.TestCase):
    def test_ties_obey_online_daily_budget(self):
        d=OnlineDecoder([.5]*4,[2]*4)
        syms=['A','B','C'];p=np.tile([.5,0,0,0],(3,1))
        a=d.choose(0,syms,p,np.zeros(3,bool),set())
        self.assertEqual(len(a),2)
        self.assertEqual(d.choose(M,syms,p,np.zeros(3,bool),set()),[])
        self.assertEqual(len(d.choose(1440*M,syms,p,np.zeros(3,bool),set())),2)
        d2=OnlineDecoder([.5]*4,[2]*4)
        self.assertEqual(a,d2.choose(0,syms,p,np.zeros(3,bool),set()))

    def test_pending_is_explicit_not_hidden_behavior_cooldown(self):
        d=OnlineDecoder([.1]*4,[10]*4)
        self.assertEqual(d.choose(0,['A'],np.array([[.9,0,0,0]]),[False],{'A'}),[])
        self.assertEqual(d.rejections['pending'],1)
        self.assertEqual(len(d.choose(M,['A'],np.array([[.9,0,0,0]]),[False],set())),1)

    def test_matching_rejects_past_event_even_with_wide_tolerance(self):
        p=dict(symbol='A',action='add',issue_ms=2*M,time_ms=3*M)
        a=dict(symbol='A',action='add',time_ms=M)
        self.assertEqual(causal_match([p],[a],5*M)['tp'],0)

    def test_greedy_matches_independent_bipartite_reference(self):
        rng=np.random.default_rng(8)
        for _ in range(100):
            pt=np.sort(rng.integers(0,20,12))*M
            at=np.sort(rng.integers(0,25,10))*M+100
            pp=[dict(symbol='A',action='add',issue_ms=int(t),time_ms=int(t+M)) for t in pt]
            aa=[dict(symbol='A',action='add',time_ms=int(t)) for t in at]
            owner={}
            def dfs(i,seen):
                for j,a in enumerate(aa):
                    if j in seen or a['time_ms']<pp[i]['issue_ms'] or abs(a['time_ms']-pp[i]['time_ms'])>2*M:continue
                    seen.add(j)
                    if j not in owner or dfs(owner[j],seen):owner[j]=i;return True
                return False
            reference=sum(dfs(i,set()) for i in range(len(pp)))
            self.assertEqual(causal_match(pp,aa,2*M)['tp'],reference)


class TestInformationBoundary(unittest.TestCase):
    def run_small(self,data,meta,drop=None):
        size=SizeModel(dict(open=[100],add=[1],reduce=[.5]))
        return autonomous(data,meta,ScriptedForecast(),size,[.5]*4,[100]*4,0,12*M,
                          np.ones(12,bool),injected_drop_action=drop)

    def test_truth_state_mutation_cannot_change_policy(self):
        d,m=small_panel();original=self.run_small(d,m)
        d['direction'][:]=-1
        for name in ['position','avgcost','first_qty']+list(POSITION):
            d['X'][:,m['feature_names'].index(name)]=np.arange(12)+20
        changed=self.run_small(d,m)
        self.assertEqual(original['issued'],changed['issued'])
        self.assertEqual(original['predictions'],changed['predictions'])

    def test_future_market_mutation_cannot_change_earlier_issues(self):
        d,m=small_panel();original=self.run_small(d,m)
        d['X'][7:,0]=500.;d['next_open'][7:]=100.
        changed=self.run_small(d,m)
        before=lambda r:[e for e in r['issued'] if e['issue_ms']<7*M]
        self.assertEqual(before(original),before(changed))

    def test_unavailable_future_execution_does_not_filter_signal(self):
        d,m=small_panel();d['next_open'][0]=np.nan
        changed=self.run_small(d,m)
        self.assertEqual(changed['issued'][0]['issue_ms'],0)
        self.assertEqual(changed['issued'][0]['quantity'],10)
        self.assertEqual(changed['execution_failures']['missing_execution_price'],1)

    def test_state_changes_only_after_execution_and_drop_is_visible(self):
        d,m=small_panel();r=self.run_small(d,m)
        self.assertEqual(r['issued'][0]['action'],'open')
        self.assertEqual(r['issued'][1]['issue_ms'],2*M)
        self.assertEqual(r['issued'][1]['action'],'add')
        dropped=self.run_small(d,m,'add')
        self.assertEqual(dropped['execution_failures']['injected_drop_add'],1)
        self.assertNotEqual(r['issued'],dropped['issued'])


if __name__=='__main__':unittest.main()
