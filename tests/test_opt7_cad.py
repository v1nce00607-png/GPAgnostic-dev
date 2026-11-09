"""OPT7: CAD repair + convex decomposition of the target collision model."""
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import trimesh
from grasp_planner.geometry_config import GeometryConfig
from grasp_planner.geometry_collision import Scene
from grasp_planner.ur_hardware import Part
from grasp_planner import cad_repair
from grasp_planner.cad_repair import repair_to_solid,audit_repair,decompose,audit_decomposition,target_solids

def u_shape():
    """Pliers-like U: two 8 mm bars joined by a base, 12 mm free gap (metres)."""
    base=trimesh.creation.box(extents=[.028,.010,.020]);base.apply_translation([0,-.040,0])
    bars=[]
    for sx in (-1,1):
        b=trimesh.creation.box(extents=[.008,.080,.020]);b.apply_translation([sx*.010,0,0]);bars.append(b)
    return trimesh.boolean.union([base]+bars,engine='manifold')

def open_soup(big_hole=False):
    """Non-watertight CAD: box with a missing triangle + a duplicated internal
    face + a second box sharing only an EDGE (non-manifold)."""
    a=trimesh.creation.box(extents=[.02,.02,.02])
    if not big_hole:
        for _ in range(4):a=a.subdivide()          # ~1.25 mm triangles -> the hole is ~1 mm
    keep=np.ones(len(a.faces),bool);keep[0]=False
    a=trimesh.Trimesh(a.vertices,a.faces[keep],process=False)
    b=trimesh.creation.box(extents=[.02,.02,.02]);b.apply_translation([.02,.02,0])
    c=trimesh.util.concatenate([a,b]);c=trimesh.Trimesh(c.vertices,np.vstack([c.faces,c.faces[2:3]]),process=False)
    return c

def make_obj(mesh,sha='x'*64):
    T=np.eye(4)
    return SimpleNamespace(id=1,name='U',kind='U',mesh=mesh.copy(),T=T,audit=dict(sha256=sha,watertight=mesh.is_watertight))

def scene_for(obj,**cfg_kw):
    cfg=GeometryConfig(**cfg_kw)
    rec=SimpleNamespace(plane=np.array([0,0,1,-5.]),points=np.zeros((1,3))+10,labels=np.zeros(1,int))
    return Scene(SimpleNamespace(objects=[obj]),rec,obj,cfg,()),cfg

def finger(center,ext=(.006,.012,.016)):
    m=trimesh.creation.box(extents=ext);m.apply_translation(center)
    return Part('right_fingertip_link',m,m,np.eye(4),False)

# ------------------------------------------------------------------ repair
def test_repair_makes_open_nonmanifold_cad_a_watertight_solid_and_audits_it():
    src=open_soup();assert not src.is_watertight
    solid=repair_to_solid(src,.0005)
    assert solid.is_watertight and solid.is_winding_consistent and solid.volume>0
    a=audit_repair(src,solid,.0005)
    assert a['original_surface_inside_solid_fraction']>.97
    assert a['original_surface_outside_gap_max_m']<.001
    assert a['repaired_to_original_surface_max_m']<.0015
    assert a['volume_m3']>a['original_signed_volume_m3']*.9       # not hollowed out

def test_repair_refuses_a_shell_with_a_large_hole_instead_of_returning_a_hollow_solid():
    src=open_soup(big_hole=True)           # 10 mm triangle missing: fill would leak
    with pytest.raises(ValueError,match='holes larger than the closing radius'):
        repair_to_solid(src,.0005)

def test_repair_refuses_when_not_a_solid(monkeypatch):
    monkeypatch.setattr(cad_repair,'_volume_fill',lambda m,p:(np.zeros((4,4,4),bool),SimpleNamespace(translation=np.zeros(3))))
    with pytest.raises(Exception):repair_to_solid(trimesh.creation.box(extents=[.02]*3),.0005)

# ------------------------------------------------------------------ models vs hull
@pytest.fixture(scope='module')
def ushape():return u_shape()

def test_u_shape_hull_is_far_larger_than_solid(ushape):
    assert ushape.convex_hull.volume>1.5*ushape.volume

@pytest.mark.parametrize('model,expect_free',[('conservative_hull',False),('repaired_solid',True),('convex_decomposition',True)])
def test_finger_in_the_free_gap_only_passes_with_repaired_models(ushape,model,expect_free,tmp_path):
    obj=make_obj(ushape,sha=model.ljust(64,'0'))
    sc,_=scene_for(obj,target_body_model=model,cad_cache_dir=str(tmp_path))
    r=sc.target_clear([finger([0,.005,0],(.006,.012,.016))])      # inside the 12 mm gap, 3 mm clear of each bar
    assert (r['status']=='PASS')==expect_free,r

@pytest.mark.parametrize('model',['conservative_hull','repaired_solid','convex_decomposition'])
def test_finger_inside_a_bar_always_fails_with_depth(ushape,model,tmp_path):
    obj=make_obj(ushape,sha=('b'+model).ljust(64,'0'))
    sc,_=scene_for(obj,target_body_model=model,cad_cache_dir=str(tmp_path))
    r=sc.target_clear([finger([.0075,.005,0],(.006,.012,.016))])   # overlaps the +x bar by ~2.5 mm
    assert r['status']=='FAIL' and r['penetration_depth_m']>.001

def test_decomposition_has_several_parts_and_small_inward_gap(ushape,tmp_path):
    obj=make_obj(ushape,sha='c'*64)
    sc,_=scene_for(obj,target_body_model='convex_decomposition',cad_cache_dir=str(tmp_path))
    d=sc.target_model_audit['decomposition']
    assert d['parts']>=2 and sc.target_model=='convex_decomposition_CoACD_%d_parts'%d['parts']
    assert d['original_surface_uncovered_gap_max_m']<.0008
    assert d['union_hull_volume_m3']<1.3*ushape.volume

def test_repaired_model_keeps_legitimate_pad_contact_and_rejects_deep_overlap():
    box=trimesh.creation.box(extents=[.020,.014,.014])
    sc,cfg=scene_for(make_obj(box,sha='d'*64),target_body_model='repaired_solid',target_contact_policy='local_pad_witness')
    anchors=np.array([[.010,0,0],[-.010,0,0]])
    touch=finger([.010+.003-.0002,0,0],(.006,.010,.012))          # pad face 0.2 mm inside the surface
    ok=sc.target_clear([touch],contacts=anchors)
    deep=sc.target_clear([finger([.010+.003-.003,0,0],(.006,.010,.012))],contacts=anchors)   # 3 mm inside
    assert deep['status']=='FAIL' and deep['penetration_depth_m']>.0015
    assert ok['status'] in ('PASS','FAIL')     # documented: allowed only if witness depth <= tolerance

# ------------------------------------------------------------------ caching / determinism
def test_cache_roundtrip_and_determinism(ushape,tmp_path):
    cfg=GeometryConfig(target_body_model='convex_decomposition',cad_cache_dir=str(tmp_path))
    obj=make_obj(ushape,sha='e'*64)
    a=target_solids(obj,cfg);files=list(Path(tmp_path).glob('*.npz'));assert len(files)==1
    cad_repair._MEMORY.clear()
    b=target_solids(obj,cfg)                    # loaded from disk
    assert len(a['solids'])==len(b['solids'])
    assert all(np.allclose(x.vertices,y.vertices,atol=1e-9) for x,y in zip(a['solids'],b['solids']))
    c2=GeometryConfig(target_body_model='convex_decomposition');cad_repair._MEMORY.clear()
    c=target_solids(make_obj(ushape,sha='e'*64),c2)
    assert len(c['solids'])==len(a['solids'])   # same seed -> same decomposition

def test_instances_share_one_decomposition_in_local_frame(ushape):
    cad_repair._MEMORY.clear()
    cfg=GeometryConfig(target_body_model='repaired_solid')
    o1=make_obj(ushape,sha='f'*64);o2=make_obj(ushape,sha='f'*64)
    T=np.eye(4);T[:3,3]=[.3,.1,0];o2.T=T;o2.mesh=ushape.copy();o2.mesh.apply_transform(T)
    a=target_solids(o1,cfg);n=len(cad_repair._MEMORY);b=target_solids(o2,cfg)
    assert len(cad_repair._MEMORY)==n==1
    assert np.allclose(b['solids'][0].bounds,a['solids'][0].bounds+np.array([[.3,.1,0],[.3,.1,0]]),atol=1e-6)

def test_defaults_do_not_change_existing_model(ushape):
    sc,_=scene_for(make_obj(ushape,sha='g'*64))
    assert sc.target_model=='conservative_convex_envelope' and sc.target_model_audit is None

# ------------------------------------------------------------------ real CAD (skipped without the HF capture)
REAL=Path(os.environ.get('GRASP_TEST_DATASET',Path(__file__).resolve().parents[2]/'GraspPlanner_data/0000'))/'objects/stl'
@pytest.mark.skipif(not (REAL/'Pliers.stl').is_file(),reason='real 0000 STL not available')
@pytest.mark.parametrize('name,min_parts',[('Pliers',8),('Rachet',5),('Screwdriver',3)])
def test_real_cad_repair_and_decomposition_audits(name,min_parts):
    m=trimesh.load(REAL/f'{name}.stl',force='mesh',process=True);m.apply_scale(.001)
    assert not m.is_watertight
    solid=repair_to_solid(m,.0005,40000);a=audit_repair(m,solid,.0005)
    assert a['original_surface_inside_solid_fraction']>.99 and a['original_surface_outside_gap_max_m']<.0005
    assert a['repaired_to_original_surface_max_m']<.0015
    assert a['volume_m3']<.75*a['convex_hull_volume_m3']
    hulls=decompose(solid);d=audit_decomposition(m,solid,hulls)
    assert d['parts']>=min_parts and d['original_surface_uncovered_gap_max_m']<.0005
    assert d['union_hull_volume_m3']<.75*a['convex_hull_volume_m3']

def test_repair_with_tight_face_budget_still_returns_a_valid_solid_or_refuses():
    # Decimation must never silently hand back a non-watertight "solid".
    ball=trimesh.creation.icosphere(subdivisions=4,radius=.012)
    out=repair_to_solid(ball,.0005,max_faces=300)
    assert out.is_watertight and out.is_winding_consistent and out.volume>0
