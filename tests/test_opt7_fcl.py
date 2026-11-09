"""OPT7 step 1: persistent FCL geometry must give IDENTICAL verdicts, only faster."""
import time
from types import SimpleNamespace
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation
from grasp_planner.geometry_config import GeometryConfig
from grasp_planner.geometry_collision import Scene
from grasp_planner.ur_hardware import CADGripper,HardwareSpec
from grasp_planner.fcl_persistent import PosedMesh,PersistentCollisionManager,closest_distance
import trimesh.collision as tc

def depth_close(a,b):
    """EPA penetration depth is exact-equal (1e-9) below 5 mm; deeper than that it is only a diagnostic
    (far beyond every 0.5 mm gate threshold) and FCL's EPA is frame-dependent, so allow 15 %."""
    a,b=abs(a),abs(b)
    return abs(a-b)<1e-9 if min(a,b)<=.005 else abs(a-b)<=.15*max(a,b)

def rand_pose(rng,spread=.03):
    T=np.eye(4);T[:3,:3]=Rotation.random(random_state=int(rng.integers(1e9))).as_matrix();T[:3,3]=rng.uniform(-spread,spread,3);return T

def l_shape():
    a=trimesh.creation.box(extents=[.04,.01,.04]);b=trimesh.creation.box(extents=[.01,.04,.04]);b.apply_translation([-.015,.015,0])
    return trimesh.boolean.union([a,b],engine='manifold')

def test_posedmesh_matches_the_transformed_mesh():
    rng=np.random.default_rng(1);m=l_shape();T=rand_pose(rng,.05);ref=m.copy();ref.apply_transform(T);p=PosedMesh(m,T)
    assert np.allclose(p.vertices,ref.vertices,atol=1e-12) and np.allclose(p.bounds,ref.bounds,atol=1e-12)
    pts=rng.uniform(-.05,.05,(400,3))+T[:3,3];assert np.array_equal(p.contains(pts),ref.contains(pts))
    assert np.allclose(closest_distance(p,pts[:50]),closest_distance(ref,pts[:50]),atol=1e-12)
    assert np.allclose(p.to_mesh().vertices,ref.vertices,atol=1e-12)

@pytest.mark.parametrize('shape',['convex','nonconvex'])
def test_manager_results_equal_the_legacy_path_on_random_poses(shape):
    rng=np.random.default_rng(7)
    target=trimesh.creation.box(extents=[.02,.02,.05]) if shape=='convex' else l_shape()
    probe=trimesh.creation.icosphere(subdivisions=2,radius=.012) if shape=='convex' else l_shape()
    legacy=tc.CollisionManager();new=PersistentCollisionManager()
    for mgr in (legacy,new):mgr.add_object('t',target)
    hits=0
    for _ in range(120):
        T=rand_pose(rng,.035);moved=probe.copy();moved.apply_transform(T)
        h0,c0=legacy.in_collision_single(moved,return_data=True);h1,c1=new.in_collision_single(PosedMesh(probe,T),return_data=True)
        assert h0==h1 and len(c0)==len(c1);hits+=h0
        if c0:
            assert depth_close(max(abs(c.depth) for c in c0),max(abs(c.depth) for c in c1))
            assert np.allclose(sorted(abs(c.depth) for c in c0),sorted(abs(c.depth) for c in c1),atol=1e-9)
        d0=legacy.min_distance_single(moved);d1=new.min_distance_single(PosedMesh(probe,T));assert abs(d0-d1)<1e-9
    assert 10<hits<110,'test poses must include both colliding and free cases'

def test_fcl_geometry_is_built_once_per_static_mesh(monkeypatch):
    calls=[]
    real_convex,real_bvh=tc.mesh_to_convex,tc.mesh_to_BVH
    monkeypatch.setattr(tc,'mesh_to_convex',lambda m:(calls.append('convex'),real_convex(m))[1])
    monkeypatch.setattr(tc,'mesh_to_BVH',lambda m:(calls.append('bvh'),real_bvh(m))[1])
    mgr=PersistentCollisionManager();mgr.add_object('t',trimesh.creation.box(extents=[.02]*3));calls.clear()
    probe=l_shape();rng=np.random.default_rng(3)
    for _ in range(25):mgr.in_collision_single(PosedMesh(probe,rand_pose(rng)))
    assert calls.count('bvh')==1                       # not 25
    calls.clear();hull=probe.convex_hull
    for _ in range(25):mgr.min_distance_single(PosedMesh(hull,rand_pose(rng)))
    assert calls.count('convex')==1
    legacy=tc.CollisionManager();legacy.add_object('t',trimesh.creation.box(extents=[.02]*3));calls.clear()
    for _ in range(5):
        m=probe.copy();m.apply_transform(rand_pose(rng));legacy.in_collision_single(m)
    assert calls.count('bvh')==5                       # the old path rebuilds every time

def stub_scene(box_extents,box_center,persistent,points=None,policy='convex_strict'):
    box=trimesh.creation.box(extents=box_extents);box.apply_translation(box_center)
    obj=SimpleNamespace(id=1,name='b',mesh=box)
    rng=np.random.default_rng(5);pts=np.zeros((1,3))+10 if points is None else points
    rec=SimpleNamespace(plane=np.array([0,0,1,.02]),points=pts,labels=np.zeros(len(pts),int))
    cfg=GeometryConfig(fcl_persistent_objects=persistent,target_contact_policy=policy)
    return Scene(SimpleNamespace(objects=[obj]),rec,obj,cfg,())

def same(a,b):
    assert a['status']==b['status'] and a.get('link')==b.get('link') and a.get('cause')==b.get('cause')
    if a.get('penetration_depth_m') is not None or b.get('penetration_depth_m') is not None:
        assert depth_close(a['penetration_depth_m'],b['penetration_depth_m']),(a,b)
    if a.get('clearance_lower_bound_m') is not None or b.get('clearance_lower_bound_m') is not None:
        assert abs(a['clearance_lower_bound_m']-b['clearance_lower_bound_m'])<1e-9,(a,b)

@pytest.mark.parametrize('model,name',[('rg2','OnRobot_RG2'),('rg6','OnRobot_RG6'),('2fg7','OnRobot_2FG7')])
def test_scene_verdicts_are_identical_for_every_gate_on_real_grippers(model,name):
    g=CADGripper(HardwareSpec(model=model,name=name));q=g.joint(float(np.mean(g.width_limits))*.8);_,tcp,_=g.contact_geometry(q)
    rng=np.random.default_rng(11);pts=tcp[:3,3]+rng.uniform(-.1,.1,(8000,3));pts[:,2]=np.abs(pts[:,2])
    ctr=tcp[:3,3]+[0,0,-.01];ext=[g.width(q)+.0004,.012,.05]
    old=stub_scene(ext,ctr,False,pts,'local_pad_witness');new=stub_scene(ext,ctr,True,pts,'local_pad_witness')
    anchors=np.array([ctr+[ext[0]/2,0,0],ctr-[ext[0]/2,0,0]]);seen=set()
    for k in range(60):
        M=np.eye(4)
        if k>0:M[:3,3]=rng.uniform(-.012,.012,3);M[:3,:3]=Rotation.from_euler('xyz',rng.uniform(-.2,.2,3)).as_matrix()
        pa,pb=g.parts(M,q,.0005),g.parts(M,q,.0005)
        for contacts in (None,anchors):
            a=old.target_clear(pa,contacts=contacts);b=new.target_clear(pb,contacts=contacts);same(a,b);seen.add((a['status'],a.get('cause')))
        same(old.clear(pa),new.clear(pb))
        da=old.target_depths(pa);db=new.target_depths(pb)
        assert [(r['link'],r['pad']) for r in da]==[(r['link'],r['pad']) for r in db]
        assert all(depth_close(x['depth'] or 0,y['depth'] or 0) for x,y in zip(da,db))
    assert ('PASS',None) in seen and any(s[0]=='FAIL' for s in seen),'poses must exercise both outcomes'

def test_default_is_persistent_and_legacy_switch_still_works():
    assert GeometryConfig().fcl_persistent_objects is True
    assert stub_scene([.02]*3,[0,0,0],False).persistent_fcl is False

def test_persistent_path_is_faster_on_real_geometry():
    g=CADGripper(HardwareSpec(model='rg2',name='OnRobot_RG2'));q=g.joint(.03);_,tcp,_=g.contact_geometry(q)
    ctr=tcp[:3,3]+[0,0,-.01];rng=np.random.default_rng(2);pts=tcp[:3,3]+rng.uniform(-.1,.1,(20000,3));pts[:,2]=np.abs(pts[:,2])
    res={}
    for flag in (False,True):
        sc=stub_scene([.02,.014,.06],ctr,flag,pts);sc.target_clear(g.parts(np.eye(4),q));sc.clear(g.parts(np.eye(4),q))   # warm up
        t=time.perf_counter()
        for k in range(15):
            M=np.eye(4);M[0,3]=.0005*k;parts=g.parts(M,q);sc.target_clear(parts);sc.clear(parts)
        res[flag]=(time.perf_counter()-t)/15
    print(f'per pose: legacy {res[False]*1e3:.1f} ms -> persistent {res[True]*1e3:.1f} ms (x{res[False]/res[True]:.1f})')
    assert res[True]<res[False]/1.5
