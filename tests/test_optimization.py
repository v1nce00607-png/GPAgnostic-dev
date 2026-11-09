"""Isolated regression tests requiring no proprietary CAD or dataset."""
from types import SimpleNamespace
import numpy as np
from grasp_planner.geometry_config import GeometryConfig
from grasp_planner.cleaning import relabel_dominance
from grasp_planner.geometry_planner import select
from grasp_planner.models import check


def test_baseline_strict_competitor_and_counterfactual_dominance():
    cfg=GeometryConfig()
    own=np.array([32,32,3,22,32,32]);other=np.array([1,0,1,2,1,1]);occ=np.array([34]*6);free=np.array([0,0,0,0,120,0]);
    baseline=relabel_dominance(own,other,occ,free,cfg)
    assert list(baseline)==[False,True,False,False,False,False]
    cfg.relabel_max_competitor_fraction=.10
    opt=relabel_dominance(own,other,occ,free,cfg)
    assert list(opt)==[True,True,False,True,False,True]


class SimpleGripper:
    width_limits=(0.005,0.1)
    def joint(self,w):return w
    def contact_geometry(self,q):return None,np.eye(4),None
    def parts(self,M,q,pad_tolerance=0):return [M]


class SimpleScene:
    def target_clear(self,parts):
        M=parts[0]
        return check(M[0,3]>5,reason='mock-only')
    def clear(self,parts):
        M=parts[0]
        return check(M[1,3]>5,reason='mock-only')


def test_gate_aware_shortlist_prefers_verified_joint_pass():
    cfg=GeometryConfig(axial_samples=5,candidates_per_object=10,candidate_screen_multiplier=2)
    pool=[]
    for i in range(25):
        T=np.eye(4);T[:2,3]=[10 if i>=12 else i*.02, 10 if i>=12 else i*.02]
        # Distinct position and not deduplicated; scoring intentionally low
        # for the geometrically feasible candidates.
        T[2,3]=i*.01
        pool.append(dict(bin=i%5,width=.03,fit=[1.,1.],T=T,score=float(25-i)))
    g=SimpleGripper();scene=SimpleScene()
    legacy,la=select(pool,g,GeometryConfig(axial_samples=5,candidates_per_object=10),scene)
    chosen,audit=select(pool,g,cfg,scene)
    assert len(legacy)==len(chosen)==10
    assert len(audit.get('selection_strategy'))>0
    assert audit['screened']==20
    assert audit['screened_joint_pass']>0
    assert any(c['_prechecked_environment']['status']=='PASS' and c['_prechecked_target']['status']=='PASS' for c in chosen)
    for c in chosen:
        assert '_prechecked_environment' in c and '_prechecked_target' in c


def test_default_screening_matches_historical_shortlist():
    cfg=GeometryConfig(axial_samples=5,candidates_per_object=10)
    pool=[]
    for i in range(25):
        T=np.eye(4);T[0,3]=.02*i
        pool.append(dict(bin=i%5,width=.03,fit=[1.,1.],T=T,score=float(i)))
    a,_=select(pool,SimpleGripper(),cfg)
    b,_=select(pool,SimpleGripper(),cfg,SimpleScene())
    assert [c['score'] for c in a]==[c['score'] for c in b]
