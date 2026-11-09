from types import SimpleNamespace
import numpy as np,pytest,trimesh
from fastapi.testclient import TestClient
from grasp_planner.models import check,verdict,pose
from grasp_planner.geometry_config import GeometryConfig
from grasp_planner.geometry_planner import GATES
from grasp_planner.ur_hardware import CADGripper,HardwareSpec,Part
from grasp_planner.geometry_collision import Scene,swept
from grasp_planner.cleaning import preprocess
from grasp_planner.trajectory import timed_path

@pytest.mark.parametrize('model',['2fg7','rg2','rg6'])
def test_manufacturer_fk_contact_and_limits(model):
    g=CADGripper(HardwareSpec(model=model))
    for w in np.linspace(.01,g.width_limits[1]-.001,5):
        q=g.joint(w);actual,_,_=g.contact_geometry(q)
        assert abs(actual-w)<1e-7
        assert g.self_collision(q)['status']=='PASS'
    with pytest.raises(ValueError):g.joint(g.width_limits[1]+.001)

@pytest.mark.parametrize('model',['2fg7','rg2','rg6'])
def test_convex_fallback_covers_original(model):
    g=CADGripper(HardwareSpec(model=model))
    for raw,hull in g.links.values():
        v=raw.vertices[::max(1,len(raw.vertices)//300)]
        assert np.max(v@hull.face_normals.T-np.sum(hull.triangles_center*hull.face_normals,axis=1))<1e-7

@pytest.mark.parametrize('s',['FAIL','UNVERIFIED','HELD',None,''])
def test_no_unchecked_pass(s):
    checks={k:check(True) for k in GATES};checks['environment']={'status':s}
    assert verdict(checks,GATES)!='PASS'

def test_missing_gate_is_unverified():
    assert verdict({'environment':check(True)},GATES)=='UNVERIFIED'

def test_config_rejects_dynamics_and_invalid_geometry():
    with pytest.raises(ValueError):GeometryConfig(mass=1)
    with pytest.raises(ValueError):GeometryConfig(tilt_degrees=[90])
    with pytest.raises(ValueError):HardwareSpec(model='imaginary')
    with pytest.raises(ValueError):HardwareSpec(contact_inset_m=-1)

def scene(points):
    cfg=GeometryConfig(voxel_m=.002,sensor_uncertainty_m=.001)
    target=trimesh.creation.box(extents=[.02,.02,.02]);target.apply_translation([.3,0,.1])
    obj=SimpleNamespace(id=1,name='test',mesh=target)
    rec=SimpleNamespace(points=np.array(points).reshape(-1,3),labels=np.zeros(len(points),int),plane=np.array([0,0,1,0]))
    return Scene(SimpleNamespace(objects=[obj]),rec,obj,cfg)

def part(center):
    m=trimesh.creation.box(extents=[.01,.01,.01]);return Part('control',m,m,pose(np.eye(3),center))

def test_observed_collision_and_free_control():
    s=scene([[0,0,.1]])
    assert s.clear([part([0,0,.1])])['status']=='FAIL'
    assert s.clear([part([.1,0,.1])])['status']=='PASS'
    assert s.clear([part([.1,0,0])])['status']=='FAIL'

def test_swept_obstacle_between_free_endpoints():
    s=scene([[0,0,.1]]);a=part([-.05,0,.1]);b=part([.05,0,.1])
    assert s.clear([a])['status']==s.clear([b])['status']=='PASS'
    assert s.clear([swept(a,b)])['status']=='FAIL'

def test_solid_containment_is_collision():
    assert scene([]).target_clear([part([.3,0,.1])])['status']=='FAIL'

def test_mask_border_excluded_and_depth_jump():
    depth=np.ones((20,20));mask=np.zeros((20,20),int);mask[5:15,5:15]=1
    good,lab,_=preprocess(depth,mask,GeometryConfig())
    assert not good[5,8] and good[8,8] and lab[8,8]==1
    depth[:,10:]=2;good,_,_=preprocess(depth,mask,GeometryConfig())
    assert not good[2,9] and good[2,2]

def test_quintic_limits():
    rows=timed_path([[0,0,0],[.1,.2,.1]],np.eye(3),.08,.3,.005)
    assert max(np.linalg.norm(r['velocity_world_m_s']) for r in rows)<=.080001
    assert max(np.linalg.norm(r['acceleration_world_m_s2']) for r in rows)<=.300001
    assert np.allclose(rows[-1]['position_world_m'],[.1,.2,.1])

def test_api_rejects_unknown_and_missing_dataset(tmp_path,monkeypatch):
    from grasp_planner.api import app
    monkeypatch.setenv('GRASP_DATA_ROOT',str(tmp_path));c=TestClient(app)
    assert c.get('/health').status_code==200
    assert c.post('/v3/grippers/validate',json={'model':'fake'}).status_code==422
    assert c.post('/v3/plan',json={'grippers':[{'model':'2fg7'}]}).status_code==404
    assert c.get('/v3/jobs/not-a-uuid').status_code==400

@pytest.mark.parametrize('model',['rg2','rg6'])
def test_curved_sweep_encloses_intermediate_fk(model):
 g=CADGripper(HardwareSpec(model=model));qa=.7;qb=.74;eps=(qb-qa)**2/8
 covers=[swept(a,b,eps).mesh() for a,b in zip(g.parts(np.eye(4),qa),g.parts(np.eye(4),qb))]
 for q in np.linspace(qa,qb,7):
  for p,m in zip(g.parts(np.eye(4),q),covers):
   v=p.mesh().vertices;offset=np.sum(m.triangles_center*m.face_normals,axis=1);assert np.max(v@m.face_normals.T-offset)<1e-7

def test_zip_path_traversal_rejected(tmp_path):
 import zipfile,importlib.util
 from pathlib import Path
 path=Path(__file__).resolve().parents[1]/'scripts/prepare_data.py';spec=importlib.util.spec_from_file_location('prepare',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);archive=tmp_path/'bad.zip'
 with zipfile.ZipFile(archive,'w') as z:z.writestr('../outside.txt','bad')
 with pytest.raises(ValueError):mod.safe_extract(archive,tmp_path/'out')
