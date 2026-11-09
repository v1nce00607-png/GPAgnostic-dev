"""OPT7: watertight reconstruction of NON-watertight OEM gripper links for static collision."""
from types import SimpleNamespace
import numpy as np
import pytest
import trimesh
from grasp_planner.geometry_config import GeometryConfig
from grasp_planner.geometry_collision import Scene
from grasp_planner.ur_hardware import CADGripper,HardwareSpec,Part
from grasp_planner import cad_repair

TIPS=['right_finger_tip_link','left_finger_tip_link']

@pytest.fixture(scope='module')
def rg2():
    g=CADGripper(HardwareSpec(model='rg2',name='OnRobot_RG2'));g.use_repaired_links(.0003,40000,None,links=TIPS);return g

def stub_scene(box_extents,box_center,points=None,cfg=None):
    box=trimesh.creation.box(extents=box_extents);box.apply_translation(box_center)
    obj=SimpleNamespace(id=1,name='b',mesh=box)
    pts=np.zeros((1,3))+10 if points is None else np.asarray(points,dtype=float)
    rec=SimpleNamespace(plane=np.array([0,0,1,5.]),points=pts,labels=np.zeros(len(pts),int))
    return Scene(SimpleNamespace(objects=[obj]),rec,obj,cfg or GeometryConfig(),())

def test_default_collision_mesh_is_unchanged_behaviour():
    box=trimesh.creation.box(extents=[.02,.02,.02]);hull=box.convex_hull
    open_raw=trimesh.Trimesh(box.vertices,box.faces[1:],process=False)       # non-watertight
    T=np.eye(4);T[0,3]=.1
    p_open=Part('x',hull,open_raw,T);p_wt=Part('y',hull,box,T)
    assert np.allclose(p_open.collision_mesh().vertices,p_open.mesh().vertices)       # hull cover
    assert np.allclose(p_wt.collision_mesh().vertices,p_wt.visual_mesh().vertices)    # exact original
    assert p_open.exact_solid() is None and p_wt.exact_solid() is box

def test_real_rg2_fingertips_become_watertight_solids_and_other_links_are_untouched(rg2):
    assert set(rg2.solids)==set(TIPS)
    for n in TIPS:
        a=rg2.solid_audit[n];s=rg2.solids[n]
        assert a['repaired'] and s.is_watertight and s.is_winding_consistent
        assert a['repaired_to_original_surface_max_m']<.001
        assert a['hull_volume_m3']>1.2*a['volume_m3']                                   # the cover was much larger
    parts={p.name:p for p in rg2.parts(np.eye(4),rg2.joint(.03))}
    assert parts[TIPS[0]].solid is not None and parts['body_link'].solid is None
    assert 'right_finger_tip_link' in rg2.evidence()['collision_model_repair']

def test_hull_overhang_collides_but_the_exact_pad_does_not(rg2):
    q=rg2.joint(.03);part=next(p for p in rg2.parts(np.eye(4),q) if p.name==TIPS[0])
    hull=part.mesh();exact=part.collision_mesh();assert exact.is_watertight
    pts,_=trimesh.sample.sample_surface(hull,20000,seed=3)
    d=trimesh.proximity.closest_point(exact,pts)[1];out=~exact.contains(pts)
    far=pts[out&(d>.0025)];assert len(far)>0,'hull must overhang the exact pad by >2.5 mm somewhere'
    sc=stub_scene([.001,.001,.001],far[0])                                         # 1 mm target inside the hull overhang
    hull_part=Part(part.name,part.local,part.raw,part.T,part.pad)                  # no solid, raw non-watertight -> hull
    assert hull_part.exact_solid() is None
    assert sc.target_clear([hull_part])['status']=='FAIL'                          # old behaviour: false collision
    assert sc.target_clear([part])['status']=='PASS'                               # exact solid: free

def test_exact_solid_still_collides_where_the_pad_really_is(rg2):
    q=rg2.joint(.03);part=next(p for p in rg2.parts(np.eye(4),q) if p.name==TIPS[0])
    inner=part.collision_mesh();c=inner.vertices[np.argmin(np.linalg.norm(inner.vertices-inner.centroid,axis=1))]
    sc=stub_scene([.004,.004,.004],c)
    r=sc.target_clear([part]);assert r['status']=='FAIL' and r['penetration_depth_m']>.0005

def test_environment_refinement_uses_the_solid_for_non_watertight_links():
    # L-shaped solid (concave) with a missing face on the raw surface -> raw not watertight.
    a=trimesh.creation.box(extents=[.04,.01,.04]);b=trimesh.creation.box(extents=[.01,.04,.04]);b.apply_translation([-.015,.015,0])
    solid=trimesh.boolean.union([a,b],engine='manifold');hull=solid.convex_hull
    raw=trimesh.Trimesh(solid.vertices,solid.faces[1:],process=False);assert not raw.is_watertight
    notch=np.array([[.012,.012,0.]])                                               # inside the hull, outside the L (voxel 3 mm)
    cfg=GeometryConfig(sensor_uncertainty_m=0.,voxel_m=.001,environment_clearance_m=.001)
    sc=stub_scene([.001,.001,.001],[1,1,1],points=notch,cfg=cfg)
    no_solid=Part('base_link',hull,raw,np.eye(4),False);with_solid=Part('base_link',hull,raw,np.eye(4),False,solid=solid)
    assert sc.clear([no_solid])['status']=='FAIL'                                  # hull cover rejects
    assert sc.clear([with_solid])['status']=='PASS'                                # exact solid sees the notch is free

def test_unrepairable_link_falls_back_to_convex_cover(monkeypatch):
    def boom(*a,**k):raise ValueError('leak: shell has holes')
    monkeypatch.setattr(cad_repair,'gripper_link_solid',boom)
    g=CADGripper(HardwareSpec(model='rg2',name='OnRobot_RG2'));audit=g.use_repaired_links(.0003,40000,None,links=TIPS)
    assert not g.solids and all(audit[n]['repaired'] is False and 'holes' in audit[n]['reason'] for n in TIPS)

def test_gripper_solid_disk_cache_roundtrip(tmp_path):
    g=CADGripper(HardwareSpec(model='rg2',name='OnRobot_RG2'));raw=g.links[TIPS[0]][0]
    cad_repair._MEMORY.clear()
    s1,a1=cad_repair.gripper_link_solid('rg2',TIPS[0],raw,.0003,40000,str(tmp_path))
    assert len(list(tmp_path.glob('gripper_rg2_*.npz')))==1
    cad_repair._MEMORY.clear()
    s2,a2=cad_repair.gripper_link_solid('rg2',TIPS[0],raw,.0003,40000,str(tmp_path))
    assert np.allclose(s1.vertices,s2.vertices,atol=1e-9) and a1['volume_m3']==a2['volume_m3']

def test_config_defaults_and_bounds():
    assert GeometryConfig().gripper_collision_model=='oem_as_is'
    assert GeometryConfig(gripper_collision_model='repaired_solid').gripper_repair_pitch_m==.0003
    with pytest.raises(ValueError):GeometryConfig(gripper_collision_model='nonsense')
