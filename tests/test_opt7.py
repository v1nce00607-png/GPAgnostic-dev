"""OPT7: penetration depth, RRT-Connect approach, 180-degree/jaw-shift diagnostics.

Synthetic scenes only; the 318-frame RGB-D dataset is not bundled. Real OEM
CAD (packaged assets) is used where a test concerns gripper geometry.
"""
from types import SimpleNamespace
import numpy as np
import pytest
import trimesh
from grasp_planner.models import check
from grasp_planner.ur_hardware import CADGripper,HardwareSpec,Part
from grasp_planner.geometry_config import GeometryConfig
from grasp_planner.geometry_collision import Scene,witness_summary
from grasp_planner.approach_rrt import rrt_connect
from grasp_planner.geometry_planner import approach
from grasp_planner.opt7_diagnostics import flip180_mount,colliding_links,opt7_candidate_diagnostics

# ---------------------------------------------------------------- helpers
def stub_scene(box_extents,box_center,cfg=None):
    box=trimesh.creation.box(extents=box_extents);box.apply_translation(box_center)
    obj=SimpleNamespace(id=1,name='box',mesh=box)
    rec=SimpleNamespace(plane=np.array([0,0,1,5.]),points=np.zeros((1,3))+10,labels=np.zeros(1,int))
    return Scene(SimpleNamespace(objects=[obj]),rec,obj,cfg or GeometryConfig(),())

@pytest.fixture(scope='module')
def g2fg7():return CADGripper(HardwareSpec(model='2fg7',name='OnRobot_2FG7'))

class Wall:
    """Axis-aligned slab in mount-translation space with a gap (x>gap_x)."""
    def __init__(self,gap_x=.09,z=(.22,.26)):self.gap_x=gap_x;self.z=z;self.calls=0
    def blocked(self,p):
        return (p[0]<self.gap_x) and (self.z[0]<=p[2]<=self.z[1])
    def segment_free(self,a,b):
        self.calls+=1
        for t in np.linspace(0,1,max(2,int(np.linalg.norm(b-a)/.001)+1)):
            if self.blocked(a+(b-a)*t):return False
        return True

# ---------------------------------------------------------------- penetration depth
def test_witness_summary_picks_deepest_and_is_robust():
    w=[SimpleNamespace(depth=.001,point=np.array([0,0,0.])),SimpleNamespace(depth=-.004,point=np.array([1,2,3.])),
       SimpleNamespace(depth=np.nan,point=np.array([9,9,9.])),SimpleNamespace(depth=.01,point=np.array([np.nan,0,0.]))]
    r=witness_summary(w)
    assert r['penetration_depth_m']==pytest.approx(.004) and r['penetration_witness_world_m']==[1,2,3]
    assert witness_summary(None)['penetration_depth_m'] is None and witness_summary([])['penetration_witness_count']==0

def test_target_clear_reports_penetration_without_changing_verdict():
    sc=stub_scene([.02,.02,.02],[0,0,0])
    cube=trimesh.creation.box(extents=[.02,.02,.02]);T=np.eye(4);T[0,3]=.017   # 3 mm overlap in X
    r=sc.target_clear([Part('right_fingertip_link',cube,cube,T,False)])
    assert r['status']=='FAIL' and r['cause']=='conservative_target_volume_intersection'
    assert r['penetration_depth_m']==pytest.approx(.003,abs=.0006)
    T[0,3]=.025   # separated by 5 mm: PASS and no depth keys
    ok=sc.target_clear([Part('right_fingertip_link',cube,cube,T,False)])
    assert ok['status']=='PASS' and 'penetration_depth_m' not in ok

def test_penetration_matches_geometry_on_real_2fg7(g2fg7):
    q=g2fg7.joint(.03);_,tcp,_=g2fg7.contact_geometry(q)
    for ext,expected in [(.036,.003),(.050,.010)]:
        sc=stub_scene([ext,.012,.020],tcp[:3,3]+[0,0,-.004])
        r=sc.target_clear(g2fg7.parts(np.eye(4),q))
        assert r['status']=='FAIL'
        assert r['penetration_depth_m']==pytest.approx(expected,abs=.0015)

# ---------------------------------------------------------------- RRT-Connect (pure)
def test_rrt_direct_edge_needs_no_tree():
    out=rrt_connect(np.zeros(3),np.array([0,0,.3]),lambda a,b:True,lambda r:r.uniform(-1,1,3))
    assert out['reason']=='direct_edge' and len(out['path'])==2

def test_rrt_finds_path_around_wall_and_every_edge_is_free():
    wall=Wall();start=np.array([0,0,.2]);goal=np.array([0,0,.3])
    assert not wall.segment_free(start,goal)
    sample=lambda r:np.array([r.uniform(-.15,.15),r.uniform(-.15,.15),r.uniform(.05,.31)])
    out=rrt_connect(start,goal,wall.segment_free,sample,step_m=.02,max_iterations=800,
                    max_edge_checks=6000,rng=np.random.default_rng(3))
    assert out['path'] is not None,out
    p=out['path'];assert np.allclose(p[0],start) and np.allclose(p[-1],goal)
    assert all(wall.segment_free(a,b) for a,b in zip(p[:-1],p[1:]))
    assert max(q[0] for q in p)>wall.gap_x      # really went around the wall

def test_rrt_is_deterministic_for_a_seed():
    def run():
        wall=Wall();sample=lambda r:np.array([r.uniform(-.15,.15),r.uniform(-.15,.15),r.uniform(.05,.31)])
        return rrt_connect(np.array([0,0,.2]),np.array([0,0,.3]),wall.segment_free,sample,
                           max_iterations=800,max_edge_checks=6000,rng=np.random.default_rng(11))
    a,b=run(),run()
    assert a['edge_checks']==b['edge_checks'] and len(a['path'])==len(b['path'])
    assert all(np.allclose(x,y) for x,y in zip(a['path'],b['path']))

def test_rrt_fails_closed_when_fully_blocked_and_respects_budget():
    wall=Wall(gap_x=1e9)           # no gap anywhere
    sample=lambda r:np.array([r.uniform(-.15,.15),r.uniform(-.15,.15),r.uniform(.05,.31)])
    out=rrt_connect(np.array([0,0,.2]),np.array([0,0,.3]),wall.segment_free,sample,
                    max_iterations=60,max_edge_checks=150,rng=np.random.default_rng(0))
    assert out['path'] is None and out['edge_checks']<=150+5
    assert out['reason'] in ('iteration_budget','edge_check_budget')

def test_rrt_rejects_colliding_endpoints():
    out=rrt_connect(np.zeros(3),np.ones(3),lambda a,b:False,lambda r:r.uniform(-1,1,3))
    assert out['path'] is None and out['reason']=='start_in_collision'

# ---------------------------------------------------------------- approach() integration
class MockGripper:
    def contact_geometry(self,q):return q,np.eye(4),None
    def parts(self,M,q,tol=0.):return []

class MockScene:
    def __init__(self,wall):self.wall=wall
    def clear(self,parts,target=False,contacts=None):return check(True)
    def translation(self,g,A,B,q,contacts=None):
        return check(self.wall.segment_free(A[:3,3],B[:3,3]))

def run_approach(rrt):
    cfg=GeometryConfig(opt7_rrt_enabled=rrt,opt7_rrt_lateral_extent_m=.15,opt7_rrt_back_extent_m=.25,
                       opt7_rrt_max_iterations=1500,opt7_rrt_max_edge_checks=8000)
    M=np.eye(4);M[2,3]=.3
    return approach(MockScene(Wall()),MockGripper(),M,.03,cfg,np.eye(4),np.eye(4),contacts=None),Wall()

def test_approach_without_rrt_fails_behind_wall_control():
    r,_=run_approach(False)
    assert r['status']=='FAIL' and r['planner']=='deterministic_corridor' and r['rrt_audit'] is None

def test_approach_with_rrt_passes_and_samples_avoid_wall():
    r,wall=run_approach(True)
    assert r['status']=='PASS' and r['planner']=='rrt_connect',r['rrt_audit']
    assert all(e['status']=='PASS' for e in r['edge_checks'])
    assert r['rrt_audit']['status']=='PASS' and r['rrt_audit']['waypoints']>=3
    pts=np.array([s['position_world_m'] for s in r['samples']])
    assert np.allclose(pts[0],[0,0,.2]) and np.allclose(pts[-1],[0,0,.3],atol=1e-9)
    assert not any(wall.blocked(p) for p in pts)
    assert r['executable'] is False and 'IK not checked' in r['scope']

def test_rrt_is_not_used_when_a_deterministic_corridor_passes():
    wall=Wall(gap_x=-1.)           # no obstruction at all
    cfg=GeometryConfig(opt7_rrt_enabled=True);M=np.eye(4);M[2,3]=.3
    r=approach(MockScene(wall),MockGripper(),M,.03,cfg,np.eye(4),np.eye(4))
    assert r['status']=='PASS' and r['planner']=='deterministic_corridor'

def test_rrt_never_rescues_a_failed_goal_pose():
    class BadGoal(MockScene):
        def clear(self,parts,target=False,contacts=None):return check(False,cause='observed_occupancy')
    cfg=GeometryConfig(opt7_rrt_enabled=True);M=np.eye(4);M[2,3]=.3
    r=approach(BadGoal(Wall()),MockGripper(),M,.03,cfg,np.eye(4),np.eye(4))
    assert r['status']=='FAIL' and r['rrt_audit'] is None

# ---------------------------------------------------------------- 180-degree twin and jaw shift
def test_flip180_is_an_involution_and_keeps_tcp_axis(g2fg7):
    q=g2fg7.joint(.03);_,tcp,_=g2fg7.contact_geometry(q)
    M=np.eye(4);M[:3,3]=[.1,-.05,.2]
    F=flip180_mount(M,tcp)
    assert np.allclose(flip180_mount(F,tcp),M,atol=1e-12)
    W,WF=M@tcp,F@tcp
    assert np.allclose(W[:3,3],WF[:3,3]) and np.allclose(W[:3,2],WF[:3,2]) and np.allclose(W[:3,0],-WF[:3,0])

@pytest.mark.parametrize('model,name',[('2fg7','OnRobot_2FG7'),('rg2','OnRobot_RG2'),('rg6','OnRobot_RG6')])
def test_oem_collision_geometry_is_not_flip_symmetric(model,name):
    # Measured property: the flipped twin is a genuinely different collision
    # problem, so it is not redundant and must not be skipped as a duplicate.
    from scipy.spatial import cKDTree
    g=CADGripper(HardwareSpec(model=model,name=name));q=g.joint(float(np.mean(g.width_limits)))
    _,tcp,_=g.contact_geometry(q)
    cloud=lambda M:np.vstack([p.visual_mesh().vertices for p in g.parts(M,q)])
    A=cloud(np.eye(4));B=cloud(flip180_mount(np.eye(4),tcp))
    hausdorff=max(cKDTree(B).query(A)[0].max(),cKDTree(A).query(B)[0].max())
    assert hausdorff>.005

def test_first_failing_link_is_check_order_not_the_only_colliding_side(g2fg7):
    q=g2fg7.joint(.03);_,tcp,_=g2fg7.contact_geometry(q)
    sc=stub_scene([.036,.012,.020],tcp[:3,3]+[0,0,-.004]);parts=g2fg7.parts(np.eye(4),q)
    r=sc.target_clear(parts);cl=colliding_links(sc,parts)
    assert r['status']=='FAIL' and cl['sides']==['left','right']
    assert r['link']==cl['first_in_iteration_order']      # verdict link = first hit only

def test_diagnostics_are_report_only_and_do_not_mutate_inputs(g2fg7):
    q=g2fg7.joint(.03);qo=g2fg7.joint(min(g2fg7.width_limits[1],.03+.012));_,tcp,_=g2fg7.contact_geometry(q)
    cfg=GeometryConfig(opt7_diagnostics_enabled=True,opt7_jaw_shift_mm=[-3.,3.])
    sc=stub_scene([.036,.012,.020],tcp[:3,3]+[0,0,-.004]);M=np.eye(4);M0=M.copy()
    contacts=np.array([[.018,0,.15],[-.018,0,.15]]);c0=contacts.copy()
    d=opt7_candidate_diagnostics(g2fg7,sc,cfg,M,q,qo,contacts,0.,tcp)
    assert d['report_only'] is True and d['grasp_valid'] is False
    assert [e['shift_mm'] for e in d['jaw_axis_open_shift_scan']['entries']]==[-3.,3.]
    assert d['flip180']['original']['target']['status']=='FAIL'
    assert np.array_equal(M,M0) and np.array_equal(contacts,c0)

def test_opt7_flags_default_off_and_validated():
    cfg=GeometryConfig()
    assert not cfg.opt7_rrt_enabled and not cfg.opt7_diagnostics_enabled
    with pytest.raises(ValueError):GeometryConfig(opt7_jaw_shift_mm=[12.])

# ---------------------------------------------------------------- planner smoke (synthetic)
def test_plan_model_smoke_stores_opt7_diagnostics_and_serializes(tmp_path,g2fg7):
    import json
    from grasp_planner.geometry_planner import plan_model
    box=trimesh.creation.box(extents=[.024,.012,.060]);box.apply_translation([0,0,.03])
    obj=SimpleNamespace(id=1,name='Box_001',kind='Box',mesh=box,audit=dict(watertight=True))
    cap=SimpleNamespace(objects=[obj],by_id={0:{'camera_from_world':np.eye(4).tolist()}})
    rec=SimpleNamespace(plane=np.array([0,0,1,0.]),points=np.zeros((1,3))+10,labels=np.zeros(1,int))
    cfg=GeometryConfig(candidates_per_object=10,axial_samples=5,yaw_count=4,tilt_degrees=[0.],
        opt7_diagnostics_enabled=True,opt7_diagnostic_attempts_per_object=3,opt7_jaw_shift_mm=[-2.,2.],
        opt7_rrt_enabled=True,opt7_rrt_max_iterations=20,opt7_rrt_max_edge_checks=60)
    rows=plan_model(cap,rec,{'Box_001':check(True)},HardwareSpec(model='2fg7',name='OnRobot_2FG7'),
                    cfg,tmp_path,progress=lambda m:None)
    assert rows and all(r['status'] in ('PASS','FAIL','UNVERIFIED') for r in rows)
    with_diag=[r for r in rows if r['opt7_diagnostics'] is not None]
    assert 0<len(with_diag)<=3
    assert all(r['opt7_diagnostics']['report_only'] and not r['opt7_diagnostics']['grasp_valid'] for r in with_diag)
    # A diagnostic can never change how many gates a candidate passes.
    json.dumps(rows,default=lambda o:o.tolist() if hasattr(o,'tolist') else str(o))
    assert (tmp_path/'OnRobot_2FG7_candidates.json').is_file()

def test_diagnostics_do_not_change_any_gate_status(tmp_path):
    from grasp_planner.geometry_planner import plan_model
    box=trimesh.creation.box(extents=[.024,.012,.060]);box.apply_translation([0,0,.03])
    def run(sub,diag):
        obj=SimpleNamespace(id=1,name='Box_001',kind='Box',mesh=box.copy(),audit=dict(watertight=True))
        cap=SimpleNamespace(objects=[obj],by_id={0:{'camera_from_world':np.eye(4).tolist()}})
        rec=SimpleNamespace(plane=np.array([0,0,1,0.]),points=np.zeros((1,3))+10,labels=np.zeros(1,int))
        cfg=GeometryConfig(candidates_per_object=10,axial_samples=5,yaw_count=4,tilt_degrees=[0.],
            opt7_diagnostics_enabled=diag,opt7_diagnostic_attempts_per_object=10)
        out=tmp_path/sub;out.mkdir()
        rows=plan_model(cap,rec,{'Box_001':check(True)},HardwareSpec(model='2fg7',name='OnRobot_2FG7'),cfg,out,progress=lambda m:None)
        return [(r['id'],r['status'],{k:v['status'] for k,v in r['checks'].items()}) for r in rows]
    assert run('off',False)==run('on',True)

def test_failed_goal_search_keeps_the_nominal_candidate_so_more_attempts_never_lower_pass_counts(tmp_path):
    from grasp_planner.geometry_planner import plan_model
    box=trimesh.creation.box(extents=[.024,.012,.060]);box.apply_translation([0,0,.03])
    def run(sub,attempts):
        obj=SimpleNamespace(id=1,name='Box_001',kind='Box',mesh=box.copy(),audit=dict(watertight=True))
        cap=SimpleNamespace(objects=[obj],by_id={0:{'camera_from_world':np.eye(4).tolist()}})
        rec=SimpleNamespace(plane=np.array([0,0,1,0.]),points=np.zeros((1,3))+10,labels=np.zeros(1,int))
        cfg=GeometryConfig(candidates_per_object=10,axial_samples=5,yaw_count=4,tilt_degrees=[0.],
            opt6_goal_optimizer_enabled=attempts>0,opt6_goal_attempts_per_object=max(attempts,1))
        out=tmp_path/sub;out.mkdir()
        return plan_model(cap,rec,{'Box_001':check(True)},HardwareSpec(model='2fg7',name='OnRobot_2FG7'),cfg,out,progress=lambda m:None)
    base=run('base',0);full=run('full',10)
    assert [r['id'] for r in base]==[r['id'] for r in full]
    for b,f in zip(base,full):
        goal=f.get('opt6_contact_goal_optimization')
        if goal is not None and goal['status']!='PASS':        # search failed -> must behave like the un-searched nominal pose
            assert f['checks']['contact_geometry']['status']==b['checks']['contact_geometry']['status']
            assert f['contact_basis']=='nominal_CAD_ray_geometric_pad_contact_not_verified'
        if goal is not None and goal['status']=='PASS':
            assert f['contact_basis']=='opt6_bilateral_OEM_pad_contact_verified'
    searched=[f for f in full if f.get('opt6_contact_goal_optimization')]
    failed=[f for f in searched if f['opt6_contact_goal_optimization']['status']!='PASS']
    assert searched and failed,'scene must exercise the fallback (searched candidates whose search failed)'
    assert all('contact_basis' in r for r in full)

def test_rank_key_is_unchanged_without_pose_rank_and_diverse_when_present():
    from grasp_planner.geometry_planner import rank_key,GATES
    def row(status,rank=None,verified=False,score=0.):
        checks={g:{'status':'PASS'} for g in GATES}
        r=dict(status=status,checks=checks,score=score,approach_precheck_corridor_status='PASS',nonpad_target_surface_clearance_m=.01,
               contact_basis='opt6_bilateral_OEM_pad_contact_verified' if verified else 'nominal_CAD_ray_geometric_pad_contact_not_verified')
        if rank is not None:r['opt7_pose_rank']=rank
        return r
    # no pose rank attached: the verified flag must NOT influence the order (old behaviour)
    a,b=row('PASS',verified=False,score=.9),row('PASS',verified=True,score=.1)
    assert rank_key(a)>rank_key(b)
    # with pose rank: distinct poses first (rank 0), verified before nominal, duplicates last
    dup=row('PASS',rank=1,verified=True,score=.99);nom=row('PASS',rank=0,verified=False,score=.9);ver=row('PASS',rank=0,verified=True,score=.1)
    order=sorted([dup,nom,ver],key=rank_key,reverse=True)
    assert order==[ver,nom,dup]
    assert rank_key(row('FAIL',rank=0,verified=True))<rank_key(dup)       # FAIL never outranks PASS
