"""OPT6 contact-goal optimizer: strict same-pose bilateral + collision gates."""
from types import SimpleNamespace
import numpy as np
import pytest
import trimesh
from grasp_planner.models import check
from grasp_planner.ur_hardware import Part
from grasp_planner.geometry_config import GeometryConfig
from grasp_planner.contact_goal_opt import optimize_contact_goal,original_pad_anchor_residuals


@pytest.fixture(autouse=True)
def naive_proximity(monkeypatch):
    # The test environment has no OEM Rtree assets; use exact triangle math.
    monkeypatch.setattr(trimesh.proximity,'closest_point',trimesh.proximity.closest_point_naive)


class TwoJaw:
    spec=SimpleNamespace(opening_extra_m=.012)
    width_limits=(.006,.05)
    def joint(self,width):return float(width)
    def width(self,q):return float(q)
    def contact_geometry(self,q):return q,np.eye(4),None
    def parts(self,M,q):
        out=[]
        for side in (-1,1):
            m=trimesh.creation.box(extents=[.002,.01,.012])
            m.apply_translation([side*(q/2+.001),0,0])
            n=('right' if side>0 else 'left')+'_fingertip_link'
            out.append(Part(n,m,m,np.asarray(M,dtype=float).copy(),True))
        return out
    def self_collision(self,q):return check(True)


class ExampleScene:
    def __init__(self,target_ok=True,env_ok=True):
        self.obj=SimpleNamespace(mesh=trimesh.creation.box(extents=[.02,.014,.014]))
        self.target_model='test_exact_surface'
        self.target_ok=target_ok;self.env_ok=env_ok
        self.seen=[]
    def target_clear(self,parts,contacts=None):
        self.seen.append(('target',parts[0].T.copy(),parts[0].raw.centroid.copy()))
        return check(self.target_ok)
    def clear(self,parts):
        self.seen.append(('env',parts[0].T.copy(),parts[0].raw.centroid.copy()))
        return check(self.env_ok)


def config(**extra):
    return GeometryConfig(opt6_goal_optimizer_enabled=True,
        opt6_goal_multistart=1,opt6_goal_max_nfev_per_seed=45,
        opt6_goal_translation_limit_m=.008,
        adaptive_contact_tolerance_m=.0002,**extra)


def candidate(shift=.006):
    T=np.eye(4);T[0,3]=shift
    return {'width':.02,'T':T,'contacts':np.array([[.01,0,0],[-.01,0,0]])}


def test_multiaxis_fixes_six_mm_misregistration_without_relaxing_gates():
    s=ExampleScene();g=TwoJaw();c=candidate()
    # A 6 mm offset is too far for the former one-dimensional 0 mm mount-X
    # budget; both true OEM pad surfaces are initially misaligned.
    Mbad=c['T'];before=np.linalg.norm(original_pad_anchor_residuals(g,Mbad,.02,c['contacts']).reshape(2,3),axis=1)
    assert max(before)>.004
    solution,audit=optimize_contact_goal(g,s,c,config())
    assert solution is not None,audit
    M,q,qo=solution
    distances=np.linalg.norm(original_pad_anchor_residuals(g,M,q,c['contacts']).reshape(2,3),axis=1)
    assert max(distances)<=.0002
    assert abs(M[0,3])<.0005
    assert qo>q
    assert audit['status']=='PASS'
    assert audit['contact_goal_target_body']['status']=='PASS'
    assert audit['contact_goal_environment']['status']=='PASS'


@pytest.mark.parametrize('target_ok,env_ok',[(False,True),(True,False),(False,False)])
def test_near_contacts_never_override_target_or_environment_failure(target_ok,env_ok):
    solution,audit=optimize_contact_goal(TwoJaw(),ExampleScene(target_ok,env_ok),candidate(),config())
    assert solution is None
    assert audit['status']=='UNVERIFIED'
    assert audit.get('near_contact_count',0)>0
    assert audit['reason']=='near_contact_rejected_by_static_collision'


def test_cannot_exceed_mount_translation_limit():
    sol,audit=optimize_contact_goal(TwoJaw(),ExampleScene(),candidate(shift=.03),config())
    assert sol is None
    assert audit['status']=='UNVERIFIED'


def test_no_single_jaw_can_fake_bilateral_contact():
    class OneJaw(TwoJaw):
        def parts(self,M,q):return super().parts(M,q)[1:]
    with pytest.raises(ValueError,match='Both original OEM'):
        original_pad_anchor_residuals(OneJaw(),np.eye(4),.02,candidate()['contacts'])
    sol,audit=optimize_contact_goal(OneJaw(),ExampleScene(),candidate(),config())
    assert sol is None
    assert audit['status']=='UNVERIFIED'


def test_opt_in_config_and_limits():
    assert not GeometryConfig().opt6_goal_optimizer_enabled
    for kv in ({'opt6_goal_translation_limit_m':.06},
               {'opt6_goal_multistart':15},
               {'opt6_goal_min_closure_m':0}):
        with pytest.raises(ValueError):GeometryConfig(**kv)


def test_outer_pad_faces_cannot_fake_a_valid_inward_grasp():
    g=TwoJaw()
    # Both anchors lie on *outer* pad faces at this q, while true inward
    # pinch faces are 2 mm away. Original-mesh proximity alone would say 0.
    anchors=np.array([[.012,0,0],[-.012,0,0]])
    old=[]
    for part in g.parts(np.eye(4),.02):
        ix=0 if part.side>0 else 1
        _,dist,_=trimesh.proximity.closest_point(part.raw,anchors[ix:ix+1])
        old.append(float(dist[0]))
    assert max(old)<1e-9
    actual=np.linalg.norm(original_pad_anchor_residuals(
        g,np.eye(4),.02,anchors).reshape(2,3),axis=1)
    assert min(actual)>.0019
