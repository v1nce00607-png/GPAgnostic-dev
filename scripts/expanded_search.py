"""Repeat wider contact search and extreme bbox deletion ablation. Partial checks only."""
from pathlib import Path
from collections import Counter
import argparse,json,numpy as np,trimesh
from grasp_planner.models import Capture,save
from grasp_planner.pipeline import load_reconstruction
from grasp_planner.geometry_config import GeometryConfig
from grasp_planner.geometry_planner import generate,select
from grasp_planner.ur_hardware import CADGripper,HardwareSpec
from grasp_planner.geometry_collision import Scene
p=argparse.ArgumentParser();p.add_argument('--dataset',default='data/0000');p.add_argument('--results',default='results');a=p.parse_args();out=Path(a.results);cap=Capture(a.dataset);rec=load_reconstruction(out);cfg=GeometryConfig.model_validate_json((out/'effective_config.json').read_text());cfg.candidates_per_object=1000;planes=[trimesh.load(p,force='mesh') for p in sorted(out.glob('obstacle_plane_*.ply'))];result=[]
for o in cap.objects:
 pool,_=generate(o,rec.plane[:3],cfg)
 for model in ['2fg7','rg2','rg6']:
  g=CADGripper(HardwareSpec(model=model,name='OnRobot_'+model.upper()));selected,_=select(pool,g,cfg);scene=Scene(cap,rec,o,cfg,planes);erased=Scene(cap,rec,o,cfg,planes,bbox_erase=True);counts=Counter();wins=[]
  for i,c in enumerate(selected):
   q=g.joint(c['width']);_,B,_=g.contact_geometry(q);M=c['T']@np.linalg.inv(B);tg=scene.target_clear(g.parts(M,q,cfg.target_penetration_tolerance_m));counts['tested']+=1
   if tg['status']!='PASS':continue
   counts['target_clear']+=1;ee=scene.clear(g.parts(M,q));counts['environment_clear']+=ee['status']=='PASS';eb=erased.clear(g.parts(M,q));counts['bbox_environment_clear_unverified']+=eb['status']=='PASS'
   if ee['status']=='PASS':wins.append(dict(index=i,candidate=c))
  result.append(dict(object=o.name,hardware=model,counts=dict(counts),wins=wins));print(o.name,model,dict(counts),flush=True);save(out/'expanded_search_probe.json',result)
