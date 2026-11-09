from pathlib import Path
import json,time,csv,os,platform,importlib.metadata
from datetime import datetime,timezone
import numpy as np,psutil
from .models import save,digest,Capture
from .reconstruction import Reconstruction,reconstruct,alignment
from .geometry_config import GeometryConfig
from .ur_hardware import HardwareSpec
from .geometry_planner import plan_model,generate,rank_key,GATES
from .cleaning import filter_scene,obstacle_planes,estimate_normals
import yaml
def load_reconstruction(out):
    out = Path(out)
    z = np.load(out / "reconstruction.npz")
    return Reconstruction(
        **{k: z[k] for k in z.files},
        report=json.loads((out / "reconstruction_report.json").read_text()),
        frames=json.loads((out / "frame_metrics.json").read_text()),
    )


def runtime_hardware():
    cpu = (
        next(
            (
                s.split(":", 1)[1].strip()
                for s in Path("/proc/cpuinfo").read_text().splitlines()
                if s.startswith("model name")
            ),
            "unknown",
        )
        if Path("/proc/cpuinfo").exists()
        else platform.processor()
    )
    versions = {}
    for n in [
        "numpy",
        "scipy",
        "Pillow",
        "trimesh",
        "python-fcl",
        "scikit-learn", "embreex",
        "matplotlib",
        "plotly",
        "pydantic",
        "fastapi",
        "pytest",
    ]:
        try:
            versions[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            versions[n] = None
    return {
        "cpu_model": cpu,
        "logical_cpus": psutil.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "available_cpu_affinity": (
            len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        ),
        "host_memory_bytes": psutil.virtual_memory().total,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": versions,
        "cgroup_limits": {
            n: Path("/sys/fs/cgroup", n).read_text().strip()
            for n in ["memory.max", "cpu.max"]
            if Path("/sys/fs/cgroup", n).exists()
        },
        "compute_backend": "CPU; no GPU used",
        "sensor": "RealSense D455 per capture metadata",
        "physical_robot_or_gripper_tested": False,
        "threads": {
            n: os.environ.get(n) for n in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]
        },
    }


def store_diagnostic_checks(row, approach_result, closure_result, cfg):
    """Rejected candidates may have visual diagnostic paths; never count them as validation."""
    result={k:v for k,v in approach_result.items()
            if k not in ('samples','waypoints_world_m','world_from_mount_at_goal')}
    if cfg.opt5_enabled:
        row['diagnostic_checks']={'approach':result,'closure':closure_result}
    else:
        row['checks']['approach']=result
        row['checks']['closure']=closure_result


def finish(cap,rec,rows,specs,cfg,out):
    out=Path(out);rows.sort(key=rank_key,reverse=True)
    top=rows[:cfg.top_k]
    reps=[next((r for r in rows if r['hardware']==h.name and r['object_instance']==o.name),None) for h in specs for o in cap.objects]
    reps=[r for r in reps if r]
    from .geometry_collision import Scene
    from .geometry_planner import approach
    from .ur_hardware import CADGripper
    from .models import verdict
    import trimesh
    planes=[trimesh.load(p,force='mesh') for p in sorted(out.glob('obstacle_plane_*.ply'))]
    gg={h.name:CADGripper(h) for h in specs};oo={o.name:o for o in cap.objects};C=np.asarray(cap.by_id[cfg.reference_frame]['camera_from_world'])
    for r in {r['id']:r for r in top+reps}.values():
        if r['trajectory']:continue
        g=gg[r['hardware']];scene=Scene(cap,rec,oo[r['object_instance']],cfg,planes);M=np.asarray(r['world_from_mount']);q=r['joint'];qo=r['open_joint'];_,B,_=g.contact_geometry(q)
        a=approach(scene,g,M,qo,cfg,C,B);a['diagnostic_for_rejected_grasp']=True;r['trajectory']=a
        # OPT5: a diagnostic trajectory may be visualized but must NEVER
        # overwrite the ten scientific validation gates for rejected grasps.
        diagnostic_closure=scene.closure(g,M,qo,q)
        store_diagnostic_checks(r,a,diagnostic_closure,cfg)
        duration=max(.05,1.875*(g.width(qo)-g.width(q))/g.spec.closing_speed_m_s);ss=[]
        for u in np.linspace(0,1,51):
            v=10*u**3-15*u**4+6*u**5;w=(1-v)*g.width(qo)+v*g.width(q);j=g.joint(w);_,B,_=g.contact_geometry(j);ss.append(dict(time_s=a['duration_s']+u*duration,aperture_m=w,joint=j,camera_from_tcp=C@M@B,camera_from_mount=C@M))
        r['closure_trajectory']=dict(status=diagnostic_closure['status'],samples=ss,executable=False,diagnostic_for_rejected_grasp=True);r['status']=verdict(r['checks'],GATES)
    rows.sort(key=rank_key,reverse=True);top=rows[:cfg.top_k]
    save(out/'all_candidates.json',rows);save(out/'top10_grasps.json',top);save(out/'representatives.json',reps)
    for h in specs:
        save(out/f'top10_{h.name}.json',[r for r in rows if r['hardware']==h.name][:cfg.top_k])
        (out/(h.name+'_candidates.json')).unlink(missing_ok=True)
    with (out/'all_candidates.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','hardware','object','status','score',*GATES])
        for r in rows:w.writerow([r['id'],r['hardware'],r['object_instance'],r['status'],r['score'],*[r['checks'][k]['status'] for k in GATES]])
    td=out/'trajectories';td.mkdir(exist_ok=True)
    for r in {r['id']:r for r in top+reps+[r for r in rows if r['status']=='PASS']}.values():
        if not r['trajectory']:continue
        save(td/(r['id']+'.json'),dict(candidate_id=r['id'],candidate_status=r['status'],approach=r['trajectory'],closure=r['closure_trajectory']))
        with (td/(r['id']+'.csv')).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['time_s','camera_x_m','camera_y_m','camera_z_m','qx','qy','qz','qw','speed_m_s','accel_m_s2'])
            for s in r['trajectory']['samples']:w.writerow([s['time_s'],*s['position_camera_m'],*s['quaternion_camera_xyzw'],np.linalg.norm(s['velocity_camera_m_s']),np.linalg.norm(s['acceleration_camera_m_s2'])])
    summary=dict(version=('3.0.5-geometry-opt5' if cfg.opt5_enabled else '3.0.4-geometry-opt4'),dataset_id=cfg.dataset_id,reference_camera=cfg.reference_frame,frames_processed=rec.report['frame_count'],scene_points=len(rec.points),candidates=len(rows),hardware_count=len(specs),object_instance_count=len(cap.objects),status_counts={s:sum(r['status']==s for r in rows) for s in ['PASS','FAIL','UNVERIFIED']},gate_counts={k:{s:sum(r['checks'][k]['status']==s for r in rows) for s in ['PASS','FAIL','UNVERIFIED']} for k in GATES},per_hardware={h.name:dict(candidates=sum(r['hardware']==h.name for r in rows),geometry_pass=sum(r['hardware']==h.name and r['status']=='PASS' for r in rows),environment_pass=sum(r['hardware']==h.name and r['checks']['environment']['status']=='PASS' for r in rows)) for h in specs},dynamics_executed=False,physical_robot_tested=False,executable_plans=0,top10_meaning='Ranked geometric candidates; rejected candidates retain FAIL.',validation_scope=('EXPERIMENTAL inferred scene after ROI deletion; any geometric PASS is CONDITIONAL, not certified clearance in original observed scene' if cfg.halo_policy=='experimental_roi' else 'Filtered observed scene and conservative CAD envelopes; no dynamics, unknown-volume certification, robot-arm IK or execution.'),halo_policy=cfg.halo_policy,opt5_enabled=cfg.opt5_enabled,diagnostic_trajectories_excluded_from_validation=cfg.opt5_enabled)
    save(out/'summary.json',summary);save(out/'hardware.json',runtime_hardware());save(out/'grippers.json',{h.name:h.model_dump() for h in specs});save(out/'effective_config.json',cfg.model_dump())
    return summary

def run(dataset,config,grippers,output,progress=print):
    out=Path(output).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Output must be empty/new')
    out.mkdir(parents=True,exist_ok=True);t=time.perf_counter()
    cfg=GeometryConfig.model_validate(yaml.safe_load(Path(config).read_text()))
    specs=[HardwareSpec.model_validate(yaml.safe_load(Path(p).read_text())) for p in grippers]
    if len({s.name for s in specs})!=len(specs):raise ValueError('Duplicate hardware names')
    cap=Capture(dataset)
    if cfg.reference_frame not in cap.by_id:raise ValueError('Unknown reference camera')
    if cfg.adaptive_voxel:cfg.voxel_m=float(np.clip(min(min(o.mesh.extents) for o in cap.objects)/12,.0015,.003))
    save(out/'input_audit.json',cap.audit());raw=out/'raw';raw.mkdir()
    rec=reconstruct(cap,cfg,raw,progress);rec=filter_scene(cap,rec,cfg,out,progress)
    al=alignment(cap,rec,cfg);save(out/'object_alignment.json',al)
    planes=obstacle_planes(rec,cfg,out);estimate_normals(cap,rec,cfg,out)
    pools={o.name:generate(o,rec.plane[:3],cfg) for o in cap.objects}
    rows=[]
    for spec in specs:rows+=plan_model(cap,rec,al,spec,cfg,out,planes,progress,pools)
    result=finish(cap,rec,rows,specs,cfg,out)
    save(out/'run_provenance.json',dict(finished_utc=datetime.now(timezone.utc).isoformat(),pipeline_seconds=time.perf_counter()-t,config_sha256=digest(config),hardware_sha256={Path(p).name:digest(p) for p in grippers}))
    from .geometry_report import render
    render(out,cap,rec,rows,specs,cfg)
    return result
