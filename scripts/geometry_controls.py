"""Known free/blocked geometric controls using all three OEM CAD grippers."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np,trimesh
from grasp_planner.models import pose,save,check,verdict
from grasp_planner.geometry_config import GeometryConfig
from grasp_planner.geometry_collision import Scene
from grasp_planner.geometry_planner import approach
from grasp_planner.ur_hardware import HardwareSpec,CADGripper

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);cfg=GeometryConfig()
    mesh=trimesh.creation.box(extents=[.02,.02,.02]);mesh.apply_translation([0,0,.3])
    o=SimpleNamespace(id=1,name='control_cube',mesh=mesh);cap=SimpleNamespace(objects=[o])
    result=[]
    for model in ['2fg7','rg2','rg6']:
        g=CADGripper(HardwareSpec(model=model,name='OnRobot_'+model.upper()));q=g.joint(.02);qo=g.joint(.032)
        T=pose(np.diag([1,-1,-1]),[0,0,.309]);_,B,_=g.contact_geometry(q);M=T@np.linalg.inv(B)
        for blocked in [False,True]:
            pts=np.array([M[:3,3]+M[:3,2]*.04]) if blocked else np.empty((0,3))
            rec=SimpleNamespace(points=pts,labels=np.zeros(len(pts),int),plane=np.array([0,0,1,0]))
            scene=Scene(cap,rec,o,cfg);a=approach(scene,g,M,qo,cfg,np.eye(4),B)
            checks=dict(environment=scene.clear(g.parts(M,q)),target_body=scene.target_clear(g.parts(M,q,cfg.target_penetration_tolerance_m)),closure=scene.closure(g,M,qo,q),approach={k:v for k,v in a.items() if k!='samples'},self_collision=g.self_collision(q))
            row=dict(id=model+('_blocked' if blocked else '_free'),hardware=model,expected='FAIL' if blocked else 'PASS',status=verdict(checks,list(checks)),checks=checks,world_from_mount=M,joint=q,open_joint=qo,trajectory=a,synthetic=True,scope='Geometry control only, not a real dataset grasp or physical success')
            save(out/(row['id']+'.json'),row);result.append(row);print(row['id'],row['status'],flush=True)
    save(out/'summary.json',[{k:r[k] for k in ['id','expected','status','synthetic','scope']} for r in result])
    assert all(r['status']==r['expected'] for r in result), 'Control mismatch: inspect individual gates'
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/controls');run(p.parse_args().output)
