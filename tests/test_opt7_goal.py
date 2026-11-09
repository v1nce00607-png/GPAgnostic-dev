"""OPT7: collision-aware contact-goal stage (penetration + clearance in the objective)."""
from types import SimpleNamespace
import numpy as np
import pytest
import trimesh
from grasp_planner.models import check
from grasp_planner.ur_hardware import Part
from grasp_planner.geometry_config import GeometryConfig
from grasp_planner.contact_goal_opt import optimize_contact_goal,original_pad_anchor_residuals,_rolled_mount

@pytest.fixture(autouse=True)
def naive_proximity(monkeypatch):
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
            m=trimesh.creation.box(extents=[.002,.01,.012]);m.apply_translation([side*(q/2+.001),0,0])
            out.append(Part(('right' if side>0 else 'left')+'_fingertip_link',m,m,np.asarray(M,dtype=float).copy(),True))
        return out
    def self_collision(self,q):return check(True)

class MockScene:
    """Analytic scene. depth(M)=max(0,Mz-free_z) metres for the pad parts; the real
    verdict allows a shallow pad contact (<= 0.5 mm) like Scene._allowed_pad_contact."""
    def __init__(self,free_z=-.004,env_gain=None):
        self.obj=SimpleNamespace(mesh=trimesh.creation.box(extents=[.02,.014,.014]))
        self.target_model='mock';self.free_z=free_z;self.env_gain=env_gain
        self.cfg=GeometryConfig();self.verified=[]
    def depth(self,M):return max(0.,float(M[2,3])-self.free_z)
    def target_depths(self,parts):
        d=self.depth(parts[0].T)
        return [dict(link='left_fingertip_link',pad=True,depth=d,witnesses=[(np.array([.01,0,0.]),d)])] if d>0 else []
    def target_clear(self,parts,contacts=None):
        d=self.depth(parts[0].T);self.verified.append(('target',parts[0].T.copy()))
        return check(d<=.0005,link='left_fingertip_link',cause='mock_intersection',penetration_depth_m=d)
    def clr(self,M):
        return .02 if self.env_gain is None else .0002+self.env_gain*max(0.,float(M[1,3]))
    def clear(self,parts):
        c=self.clr(parts[0].T);self.verified.append(('env',parts[0].T.copy()))
        return check(c>=GeometryConfig().environment_clearance_m,clearance_lower_bound_m=c,link='right_fingertip_link',cause='mock_occupancy')

def cfg(**kw):
    return GeometryConfig(opt6_goal_optimizer_enabled=True,opt6_goal_multistart=1,opt6_goal_max_nfev_per_seed=45,
        opt6_goal_translation_limit_m=.008,adaptive_contact_tolerance_m=.0002,**kw)

def candidate():
    return {'width':.02,'T':np.eye(4),'contacts':np.array([[.01,0,0],[-.01,0,0]])}

def test_without_the_stage_nominal_contact_is_rejected_by_static_collision():
    sol,audit=optimize_contact_goal(TwoJaw(),MockScene(),candidate(),cfg())
    assert sol is None and audit['reason']=='near_contact_rejected_by_static_collision'
    assert 'collision_aware_stage' not in audit and GeometryConfig().opt7_goal_collision_aware is False

def test_stage_moves_the_mount_to_a_collision_free_contact_pose_and_gates_still_decide():
    s=MockScene();g=TwoJaw()
    sol,audit=optimize_contact_goal(g,s,candidate(),cfg(opt7_goal_collision_aware=True,opt7_goal_collision_max_evals=150))
    assert sol is not None,audit
    M,q,qo=sol
    assert audit['status']=='PASS' and audit['search_stage']=='collision_aware'
    assert M[2,3]<=-.0035 and s.depth(M)<=.0005                     # moved out of the penetration
    gaps=np.linalg.norm(original_pad_anchor_residuals(g,M,q,np.asarray(candidate()['contacts'])).reshape(2,3),axis=1)
    assert max(gaps)<=.0002+1e-9                                    # both pad contacts kept
    st=audit['collision_aware_stage'];assert st['evaluations']<=st['budget']
    # The returned pose is exactly the one the unmodified gates were run on.
    assert any(k=='target' and np.allclose(T,M) for k,T in s.verified)
    assert any(k=='env' and np.allclose(T,M) for k,T in s.verified)

def test_stage_uses_clearance_deficit_not_only_penetration():
    s=MockScene(free_z=1.,env_gain=.8)                              # no penetration; env needs y>=~+3.5 mm
    sol,audit=optimize_contact_goal(TwoJaw(),s,candidate(),cfg(opt7_goal_collision_aware=True,opt7_goal_collision_max_evals=200))
    assert sol is not None,audit
    assert s.clr(sol[0])>=GeometryConfig().environment_clearance_m and sol[0][1,3]>.002

def test_no_free_pose_inside_bounds_fails_closed_and_never_passes():
    s=MockScene(free_z=-.05)                                        # needs 50 mm shift; limit is 8 mm
    sol,audit=optimize_contact_goal(TwoJaw(),s,candidate(),cfg(opt7_goal_collision_aware=True))
    assert sol is None and audit['reason']=='near_contact_rejected_by_static_collision'
    assert audit['collision_aware_stage']['best']['penetration_beyond_tolerance_m']>.03

def test_evaluation_budget_is_respected():
    s=MockScene()
    sol,audit=optimize_contact_goal(TwoJaw(),s,candidate(),cfg(opt7_goal_collision_aware=True,opt7_goal_collision_max_evals=20))
    st=audit['collision_aware_stage']
    assert st['evaluations']<=20 and st['budget']==20

def test_roll_about_the_jaw_axis_keeps_axis_midpoint_and_pad_contact():
    anchors=np.array([[.01,0,0],[-.01,0,0]]);M=np.eye(4);M[:3,3]=[.001,.002,-.001]
    R=_rolled_mount(M,anchors,np.array([1.,0,0]),12.)
    assert np.allclose(R[:3,0],M[:3,0])                              # closing direction unchanged
    assert np.allclose((R@np.r_[anchors.mean(0),1.])[:3]-(M@np.r_[anchors.mean(0),1.])[:3],0,atol=1.5e-2)
    g=TwoJaw();d0=original_pad_anchor_residuals(g,M,.02,anchors);d1=original_pad_anchor_residuals(g,R,.02,anchors)
    assert np.linalg.norm(d1.reshape(2,3),axis=1).max()<=np.linalg.norm(d0.reshape(2,3),axis=1).max()+.002
    assert np.allclose(_rolled_mount(M,anchors,np.array([1.,0,0]),0.),M)

def test_new_config_fields_are_bounded():
    with pytest.raises(ValueError):GeometryConfig(opt7_goal_roll_limit_deg=45.)
    with pytest.raises(ValueError):GeometryConfig(opt7_goal_collision_max_evals=5)
    assert GeometryConfig(opt6_goal_attempts_per_object=100).opt6_goal_attempts_per_object==100

def test_hopeless_start_is_skipped_cheaply():
    s=MockScene(free_z=-.05)                                        # start penetrates 50 mm > 6 mm skip depth
    sol,audit=optimize_contact_goal(TwoJaw(),s,candidate(),cfg(opt7_goal_collision_aware=True))
    st=audit['collision_aware_stage']
    assert sol is None and st['skipped_starts']==1 and st['evaluations']<=2     # one probe, no optimization
    s2=MockScene(free_z=-.004)                                      # 4 mm: within the skip depth -> optimized
    sol2,audit2=optimize_contact_goal(TwoJaw(),s2,candidate(),cfg(opt7_goal_collision_aware=True))
    assert sol2 is not None and audit2['collision_aware_stage']['skipped_starts']==0

class DeficitScene(MockScene):
    """No penetration problem, but a large environment clearance deficit at the start."""
    def __init__(self,clr):
        super().__init__(free_z=1.0);self._clr=clr
    def clr(self,M):return self._clr

def test_start_with_large_clearance_deficit_is_filtered_before_optimization():
    c=cfg(opt7_goal_collision_aware=True,opt7_goal_collision_skip_deficit_m=.003)
    s=DeficitScene(clr=-.01)                                         # deficit = margin - (-10 mm) ~ 12 mm > 3 mm
    sol,audit=optimize_contact_goal(TwoJaw(),s,candidate(),c)
    st=audit['collision_aware_stage']
    assert sol is None and st['filtered_by']=='clearance_deficit' and st['optimized_starts']==0 and st['evaluations']<=2
    assert st['start_deficit_m']>.003

def test_start_probe_values_are_logged_and_good_starts_are_optimized():
    sol,audit=optimize_contact_goal(TwoJaw(),MockScene(),candidate(),cfg(opt7_goal_collision_aware=True))
    st=audit['collision_aware_stage']
    assert sol is not None and st['optimized_starts']==1 and st['filtered_by'] is None
    assert st['start_penetration_m']>.003                           # the probe saw the ~4 mm penetration

def test_budget_flag_disables_the_stage_without_touching_the_contact_only_result():
    sol,audit=optimize_contact_goal(TwoJaw(),MockScene(),candidate(),cfg(opt7_goal_collision_aware=True),allow_collision_stage=False)
    assert sol is None and audit['collision_aware_stage']['skipped'].startswith('per-object budget')
    assert audit['reason']=='near_contact_rejected_by_static_collision'

def test_planner_spends_the_budget_on_the_best_ranked_candidates(monkeypatch,tmp_path):
    import trimesh as tm
    from grasp_planner import geometry_planner as gp
    from grasp_planner.ur_hardware import HardwareSpec
    from grasp_planner.models import check
    calls=[]
    def fake(g,scene,c,cfg,allow_collision_stage=True):
        calls.append(allow_collision_stage)
        return None,dict(status='UNVERIFIED',reason='near_contact_rejected_by_static_collision',distance_evaluations=1,
            best_worst_gap_m=0.,collision_aware_stage=dict(optimized_starts=1 if allow_collision_stage else 0))
    monkeypatch.setattr(gp,'optimize_contact_goal',fake)
    box=tm.creation.box(extents=[.024,.012,.060]);box.apply_translation([0,0,.03])
    obj=SimpleNamespace(id=1,name='Box_001',kind='Box',mesh=box,audit=dict(watertight=True))
    cap=SimpleNamespace(objects=[obj],by_id={0:{'camera_from_world':np.eye(4).tolist()}})
    rec=SimpleNamespace(plane=np.array([0,0,1,0.]),points=np.zeros((1,3))+10,labels=np.zeros(1,int))
    c=GeometryConfig(candidates_per_object=10,axial_samples=5,yaw_count=4,tilt_degrees=[0.],opt6_goal_optimizer_enabled=True,
        opt6_goal_attempts_per_object=10,opt7_goal_collision_aware=True,opt7_goal_collision_budget_per_object=3)
    gp.plan_model(cap,rec,{'Box_001':check(True)},HardwareSpec(model='2fg7',name='OnRobot_2FG7'),c,tmp_path,progress=lambda m:None)
    assert len(calls)>=4 and calls[:3]==[True,True,True] and not any(calls[3:])     # first 3 get the stage, the rest do not
