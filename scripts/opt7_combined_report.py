"""Merge per-gripper runs (same dataset/config) into ONE run folder and render the
standard OPT4-style report (report.html, figures, 3D scene, replays, top-10 files).

Nothing is re-planned: candidates are taken verbatim from each run's all_candidates.json.
`finish()` only (a) re-ranks, (b) computes diagnostic trajectories for top-10/representatives
that lack one (never touches the ten gates), (c) writes summary/CSV/trajectories;
`render()` draws the figures and HTML exactly as the original pipeline does.

Usage:
  python scripts/opt7_combined_report.py --dataset .../0000 \
     --runs run_N2_2fg7 run_N2_rg2 run_N2_rg6 \
     --grippers configs/grippers/2fg7.yaml configs/grippers/rg2.yaml configs/grippers/rg6.yaml \
     --output run_N2_combined
"""
import argparse,json,shutil,time
from datetime import datetime,timezone
from pathlib import Path
import yaml
from grasp_planner.models import Capture,save,digest
from grasp_planner.geometry_config import GeometryConfig
from grasp_planner.ur_hardware import HardwareSpec
from grasp_planner.pipeline import load_reconstruction,finish
from grasp_planner.geometry_report import render

SHARED_FILES=['input_audit.json','reconstruction.npz','reconstruction_report.json','frame_metrics.json','filter_audit.json',
              'filter_evidence.npz','normal_audit.json','object_alignment.json','obstacle_planes.json','target_normals.npz']

def attach_pose_rank(rows,tol=.003):
    """Post-hoc: group PASS candidates into physically distinct poses (same gripper/object,
    TCP within 3 mm, approach axis within ~11 deg) and store opt7_pose_rank (0 = best member).
    Only used by rank_key to build a DIVERSE top-10; statuses and gates are never touched."""
    import numpy as np
    from grasp_planner.geometry_planner import rank_key
    clusters=[]
    for r in sorted([x for x in rows if x['status']=='PASS'],key=rank_key,reverse=True):
        T=np.array(r['world_from_tcp']);p=T[:3,3];z=T[:3,2]
        for c in clusters:
            if c['key']==(r['hardware'],r['object_instance']) and np.linalg.norm(c['p']-p)<tol and c['z']@z>.98:
                r['opt7_pose_rank']=len(c['n']);c['n'].append(r);break
        else:
            r['opt7_pose_rank']=0;clusters.append(dict(key=(r['hardware'],r['object_instance']),p=p,z=z,n=[r]))
    for r in rows:r.setdefault('opt7_pose_rank',0)
    return len(clusters)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',required=True);ap.add_argument('--runs',nargs='+',required=True)
    ap.add_argument('--grippers',nargs='+',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--ignore-config-keys',nargs='*',default=[],help='keys allowed to differ between the per-gripper runs (e.g. opt7_goal_collision_aware)')
    a=ap.parse_args();t0=time.perf_counter()
    runs=[Path(r).resolve() for r in a.runs];out=Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):raise SystemExit(f'Output must be empty/new: {out}')
    out.mkdir(parents=True,exist_ok=True)
    specs=[HardwareSpec.model_validate(yaml.safe_load(Path(p).read_text())) for p in a.grippers]
    if len(specs)!=len(runs):raise SystemExit('One run per gripper, in the same order')
    # All runs must share the exact effective configuration (except nothing: they come from one yaml).
    cfgs=[json.loads((r/'effective_config.json').read_text()) for r in runs]
    strip=lambda c:{k:v for k,v in c.items() if k not in a.ignore_config_keys}
    if any(strip(c)!=strip(cfgs[0]) for c in cfgs[1:]):raise SystemExit('Runs were produced with different effective configurations')
    differing={k:[c.get(k) for c in cfgs] for k in a.ignore_config_keys if len({json.dumps(c.get(k)) for c in cfgs})>1}
    cfg=GeometryConfig.model_validate(cfgs[0])
    audits=[json.loads((r/'input_audit.json').read_text()) for r in runs]
    if any(x!=audits[0] for x in audits[1:]):raise SystemExit('Runs used different input captures')
    base=runs[0]
    for f in SHARED_FILES:shutil.copy2(base/f,out/f)
    shutil.copytree(base/'raw',out/'raw')
    for p in base.glob('obstacle_plane_*.ply'):shutil.copy2(p,out/p.name)
    rows=[]
    for r,spec in zip(runs,specs):
        part=json.loads((r/'all_candidates.json').read_text())
        if {x['hardware'] for x in part}!={spec.name}:raise SystemExit(f'{r.name} does not contain only {spec.name}')
        rows+=part
        for suffix in ['generation','hardware','timing']:
            src=r/f'{spec.name}_{suffix}.json'
            if src.exists():shutil.copy2(src,out/src.name)
    attach_pose_rank(rows)
    cap=Capture(a.dataset);rec=load_reconstruction(out)
    summary=finish(cap,rec,rows,specs,cfg,out)
    save(out/'run_provenance.json',dict(finished_utc=datetime.now(timezone.utc).isoformat(),
        merged_from=[r.name for r in runs],combine_seconds=time.perf_counter()-t0,
        note='per-gripper runs merged verbatim; candidates were not re-planned',config_differences_per_run=differing,
        hardware_sha256={Path(p).name:digest(p) for p in a.grippers}))
    render(out,cap,rec,rows,specs,cfg)
    summary=json.loads((out/'summary.json').read_text())
    summary.update(opt7_merged_runs=[r.name for r in runs],opt7_features=dict(
        target_body_model=cfg.target_body_model,environment_clearance_m=cfg.environment_clearance_m,
        opt6_goal_optimizer=cfg.opt6_goal_optimizer_enabled,opt6_goal_attempts_per_object=cfg.opt6_goal_attempts_per_object,
        opt7_goal_collision_aware=cfg.opt7_goal_collision_aware,opt7_rrt_enabled=cfg.opt7_rrt_enabled))
    save(out/'summary.json',summary)
    print(json.dumps(dict(output=str(out),candidates=summary['candidates'],status_counts=summary['status_counts']),indent=1))

if __name__=='__main__':main()
