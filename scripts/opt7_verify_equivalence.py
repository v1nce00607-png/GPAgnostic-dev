"""Re-evaluate the STORED final pose of every candidate with the current collision backend and compare
with the verdicts stored by the original run (target_body, environment): status, failing link, cause,
clearance and penetration depth. Read-only; proves a backend change did not alter results.

Usage: python scripts/opt7_verify_equivalence.py --dataset .../0000 --runs run_A run_B ... [--grippers g1.yaml ...]
"""
import argparse,json,time,collections as C
from pathlib import Path
import numpy as np,trimesh,yaml
from grasp_planner.models import Capture
from grasp_planner.geometry_config import GeometryConfig
from grasp_planner.ur_hardware import CADGripper,HardwareSpec
from grasp_planner.geometry_collision import Scene
from grasp_planner.pipeline import load_reconstruction

def depth_close(a,b):
    a,b=abs(a),abs(b);return abs(a-b)<1e-9 if min(a,b)<=.005 else abs(a-b)<=.15*max(a,b)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--runs',nargs='+',required=True)
    ap.add_argument('--limit',type=int,default=0);ap.add_argument('--dump',default='');a=ap.parse_args()
    runs=[Path(r).resolve() for r in a.runs];base=runs[0]
    cfg=GeometryConfig.model_validate(json.loads((base/'effective_config.json').read_text()))
    cap=Capture(a.dataset);rec=load_reconstruction(base);by={o.name:o for o in cap.objects}
    planes=[trimesh.load(p,force='mesh') for p in sorted(base.glob('obstacle_plane_*.ply'))]
    grips=json.loads((base/'grippers.json').read_text()) if (base/'grippers.json').exists() else {}
    tally=C.Counter();bad=[];allbad=[];t_new=0.;n=0;scenes={};print('loaded; starting',flush=True)
    for run in runs:
        rc=GeometryConfig.model_validate(json.loads((run/'effective_config.json').read_text()))
        if rc!=cfg:raise SystemExit('runs differ in configuration')
        specs={n:HardwareSpec.model_validate(s) for n,s in json.loads((run/'grippers.json').read_text()).items()}
        gs={n:CADGripper(s) for n,s in specs.items()}
        if cfg.gripper_collision_model=='repaired_solid':
            for g in gs.values():g.use_repaired_links(cfg.gripper_repair_pitch_m,cfg.gripper_collision_max_faces,cfg.cad_cache_dir)
        rows=json.loads((run/'all_candidates.json').read_text())
        for x in (rows[:a.limit] if a.limit else rows):
            if x.get('pose_phase','contact_goal')!='contact_goal':tally['skipped_non_goal_phase']+=1;continue
            sc=scenes.setdefault(x['object_instance'],Scene(cap,rec,by[x['object_instance']],cfg,planes))
            assert sc.persistent_fcl,'verification must run the persistent path'
            g=gs[x['hardware']];M=np.array(x['world_from_mount']);q=x['joint'];anchors=np.array(x['contacts_world_m'])
            ps=0. if cfg.target_body_model=='original_surface_ablation' else cfg.target_penetration_tolerance_m
            t=time.perf_counter()
            tb=sc.target_clear(g.parts(M,q,ps),contacts=anchors if cfg.target_contact_policy=='local_pad_witness' else None)
            en=sc.clear(g.parts(M,q));t_new+=time.perf_counter()-t;n+=1
            for gate,new in (('target_body',tb),('environment',en)):
                old=x['checks'][gate]
                verdict_same=old['status']==new['status'] and old.get('link')==new.get('link') and old.get('cause')==new.get('cause')
                if verdict_same and (old.get('clearance_lower_bound_m') is not None or new.get('clearance_lower_bound_m') is not None):
                    verdict_same=abs(old['clearance_lower_bound_m']-new['clearance_lower_bound_m'])<1e-9
                depth_same=True
                if old.get('penetration_depth_m') is not None and new.get('penetration_depth_m') is not None:
                    depth_same=abs(old['penetration_depth_m']-new['penetration_depth_m'])<1e-9
                kind='identical' if (verdict_same and depth_same) else ('depth_only_differs' if verdict_same else 'VERDICT_DIFFERENT')
                tally[(gate,kind)]+=1
                if kind!='identical':
                    allbad.append(dict(id=x['id'],gate=gate,kind=kind,old={k:old.get(k) for k in ('status','link','cause','clearance_lower_bound_m','penetration_depth_m')},new={k:new.get(k) for k in ('status','link','cause','clearance_lower_bound_m','penetration_depth_m')}))
            if n%400==0:print(f'  {n} checked | {dict((f"{k[0]}:{k[1]}",v) for k,v in tally.items() if isinstance(k,tuple))}',flush=True)
    depth_only=[d for d in allbad if d['kind']=='depth_only_differs']
    dd=sorted(abs(d['old']['penetration_depth_m']-d['new']['penetration_depth_m'])*1e3 for d in depth_only if d['old']['penetration_depth_m'] is not None and d['new']['penetration_depth_m'] is not None)
    shallow=[d for d in depth_only if d['old']['penetration_depth_m'] is not None and min(d['old']['penetration_depth_m'],d['new']['penetration_depth_m'])<=.0005]
    if a.dump:Path(a.dump).write_text(json.dumps(allbad,indent=1,default=str))
    print(json.dumps(dict(candidates_checked=n,seconds_new_backend=round(t_new,1),ms_per_candidate=round(1000*t_new/max(n,1),1),
        results={f'{k[0]}:{k[1]}':v for k,v in tally.items() if isinstance(k,tuple)},
        verdict_different=[d for d in allbad if d['kind']=='VERDICT_DIFFERENT'][:10],
        depth_only_differences=dict(count=len(depth_only),diff_mm_median=round(float(np.median(dd)),4) if dd else None,diff_mm_max=round(float(max(dd)),3) if dd else None,
            involving_depth_at_or_below_the_0_5mm_gate_limit=len(shallow))),indent=1,default=str))
if __name__=='__main__':main()
