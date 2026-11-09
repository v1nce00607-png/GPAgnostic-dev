"""Geometry-only candidate generation and camera-frame approach/closure planning."""
from pathlib import Path
import time,numpy as np
from scipy.spatial.transform import Rotation
from .models import unit,pose,transform,save,check,verdict
from .ur_hardware import CADGripper
from .geometry_collision import Scene
from .trajectory import timed_path
from .adaptive_contact import solve_adaptive_closure
from .static_pair import find_joint_static_pregrasp
from .contact_goal_opt import optimize_contact_goal
from .approach_rrt import rrt_connect
from .opt7_diagnostics import opt7_candidate_diagnostics
GATES=['data_alignment','hardware_geometry','kinematics','contact_geometry','self_collision','target_body','environment','approach','closure','filter_policy']

def generate(obj,up,cfg):
    mesh=obj.mesh;com=mesh.centroid;v=mesh.vertices-com;tangent=v-(v@up)[:,None]*up;_,_,vh=np.linalg.svd(tangent,full_matrices=False);axis=unit(vh[0]-up*(vh[0]@up));z=-up
    ss=np.linspace(*np.quantile(v@axis,[.02,.98]),cfg.axial_samples);hh=np.quantile(v@up,cfg.candidate_height_quantiles);definitions=[];origins=[];directions=[]
    for bi,s in enumerate(ss):
        for h in hh:
            for yaw in np.linspace(0,2*np.pi,cfg.yaw_count,endpoint=False):
                x=unit(np.cos(yaw)*axis+np.sin(yaw)*np.cross(up,axis));R=np.column_stack([x,np.cross(z,x),z]);c=com+s*axis+h*up;definitions.append((bi,c,R));origins.extend([c+.4*x,c-.4*x]);directions.extend([-x,x])
    origins=np.array(origins);directions=np.array(directions);hits=np.full_like(origins,np.nan);faces=np.full(len(origins),-1,int)
    for a in range(0,len(origins),64):
        f,r,p=mesh.ray.intersects_id(origins[a:a+64],directions[a:a+64],multiple_hits=False,return_locations=True);hits[a+r]=p;faces[a+r]=f
    allc=[]
    for i,(bi,c,R) in enumerate(definitions):
        p=hits[2*i:2*i+2];f=faces[2*i:2*i+2]
        if not np.isfinite(p).all():continue
        n=mesh.face_normals[f];width=float((p[0]-p[1])@R[:,0]);fit=np.array([n[0]@R[:,0],-n[1]@R[:,0]])
        if width<=0:continue
        for angle in sorted(set([0.]+[a for t in cfg.tilt_degrees for a in [-t,t]])):
            RR=R@Rotation.from_euler('x',angle,degrees=True).as_matrix();T=pose(RR,p.mean(0));score=float(fit.mean()-.15*np.linalg.norm(T[:3,3]-com)/max(mesh.extents)-.001*abs(angle))
            allc.append(dict(T=T,width=width,contacts=p,normals=n,fit=fit,score=score,bin=bi,tilt_deg=angle,faces=f))
    return allc,dict(grid_contact_pairs=len(definitions),complete_tilt_candidates=len(allc),generator='generic PCA axial/height/yaw grid, exact CAD contact rays, approach tilt; no object-specific rule')

def candidate_target_check(scene,parts,cfg,contacts):
    if cfg.target_contact_policy=='local_pad_witness':
        return scene.target_clear(parts,contacts=contacts)
    return scene.target_clear(parts)

def nonpad_target_surface_clearance(scene,parts):
    """Selection heuristic only: CAD surface separation of non-pad OEM links.

    Does not establish solid occupancy for an open target and must NEVER
    replace `Scene.target_clear` or declare a PASS.
    """
    vals=[]
    for p in parts:
        if p.pad:continue
        mesh=p.collision_posed() if getattr(scene,'persistent_fcl',False) else p.collision_mesh()
        vals.append(float(scene.target_original.min_distance_single(mesh)))
    return min(vals) if vals else float('inf')

def select(pool,g,cfg,scene=None,progress=None):
    """Diverse shortlist; optionally screen a wider pool with actual gates.

    No PASS value is inferred from a heuristic: prechecks call Scene's original
    target and environment validators and their outcomes are reused verbatim.
    The final candidate budget is unchanged, so the comparison is auditable.
    """
    eligible=[c for c in pool if g.width_limits[0]<=c['width']<=g.width_limits[1] and min(c['fit'])>=np.cos(np.deg2rad(cfg.max_contact_angle_deg))]
    bins={b:[] for b in range(cfg.axial_samples)}
    for c in sorted(eligible,key=lambda c:c['score'],reverse=True):
        arr=bins[c['bin']]
        if any(np.linalg.norm(c['T'][:3,3]-x['T'][:3,3])<.004 and c['T'][:3,0]@x['T'][:3,0]>.98 and c['T'][:3,2]@x['T'][:3,2]>.98 for x in arr):continue
        arr.append(c)
    selected=[]
    screen_limit=cfg.candidates_per_object*(cfg.candidate_screen_multiplier if scene is not None else 1)
    while any(bins.values()) and len(selected)<screen_limit:
        for b in bins:
            if bins[b]:selected.append(bins[b].pop(0))
            if len(selected)>=screen_limit:break
    audit=dict(kinematic_or_contact_pruned=len(pool)-len(eligible),eligible_before_diversity=len(eligible),screened=len(selected),selection_strategy='historical_diversity',evaluated=min(len(selected),cfg.candidates_per_object))
    if scene is None or cfg.candidate_screen_multiplier==1:
        return selected[:cfg.candidates_per_object],audit
    screened=[]
    def evaluate(c):
        q=g.joint(c['width'])
        _,tcp,_=g.contact_geometry(q)
        M=c['T']@np.linalg.inv(tcp)
        # These are the same tests as in plan_model. Preserve exact results.
        parts=g.parts(M,q)
        # Apply EXACTLY the same target-pad policy as final validation.
        pad_shift=(0. if cfg.target_body_model=='original_surface_ablation'
                   else cfg.target_penetration_tolerance_m)
        target=candidate_target_check(scene,g.parts(M,q,pad_shift),cfg,c.get('contacts'))
        env=scene.clear(parts)
        candidate=dict(c)
        candidate['_prechecked_target']=target
        candidate['_prechecked_environment']=env
        body_clearance=None
        if cfg.target_body_clearance_ranking and env['status']=='PASS':
            body_clearance=nonpad_target_surface_clearance(scene,parts)
        candidate['_nonpad_target_surface_clearance_m']=body_clearance
        # Evaluate actual fully-open approach corridor ONLY for joint-static
        # survivors. This is an early ranking heuristic, not an approach PASS.
        corridor=None
        if (cfg.approach_precheck_for_joint_pass and target['status']=='PASS'
                and env['status']=='PASS'):
            qo=g.joint(min(g.width_limits[1],c['width']+g.spec.opening_extra_m))
            start=M.copy();start[:3,3]-=M[:3,2]*cfg.approach_distance_m
            corridor=scene.translation(g,start,M,qo,contacts=(c['contacts']
                if cfg.target_contact_policy=='local_pad_witness' else None))
        candidate['_prechecked_corridor']=corridor
        clr=env.get('clearance_lower_bound_m',-1)
        clr=float(clr) if clr is not None and np.isfinite(clr) else -1.
        status=(int(target['status']=='PASS')+int(env['status']=='PASS'),
                int(corridor is not None and corridor['status']=='PASS'),
                int(env['status']=='PASS'),int(target['status']=='PASS'),
                min(.025,body_clearance) if body_clearance is not None and np.isfinite(body_clearance) else -1.,
                min(.025,clr) if env['status']=='PASS' else -1.)
        return status,candidate

    # CAD collision screening may be expensive; report actual progress rather
    # than appearing stuck at the high-level "proposals" log line.
    screened=[]
    screen_t0=time.perf_counter()
    if progress:progress(f'pre-screen start: {len(selected)} candidate poses')
    for k,item in enumerate(selected,1):
        screened.append(evaluate(item))
        if progress and (k==1 or k%25==0 or k==len(selected)):
            progress(f'pre-screen {k}/{len(selected)} | {time.perf_counter()-screen_t0:.1f}s')
    # OPT5: local rotation around the actual pad-contact midpoint; the X
    # closing axis and both CAD contact anchors remain fixed. No translation
    # or synthetic contact points are introduced. Every proposal is screened
    # by the same exact target + occupancy checks as the original candidate.
    refined=0
    if cfg.opt5_enabled and cfg.opt5_refine_seeds_per_group:
        env_only=[x for x in screened if x[1]['_prechecked_environment']['status']=='PASS'
                  and x[1]['_prechecked_target']['status']!='PASS']
        target_only=[x for x in screened if x[1]['_prechecked_target']['status']=='PASS'
                  and x[1]['_prechecked_environment']['status']!='PASS']
        both=[x for x in screened if x[1]['_prechecked_target']['status']=='PASS'
              and x[1]['_prechecked_environment']['status']=='PASS']
        seeds=[];seen=set()
        for group in [both,env_only,target_only]:
            for _,item in sorted(group,key=lambda x:x[1]['score'],reverse=True)[:cfg.opt5_refine_seeds_per_group]:
                key=id(item.get('_opt5_origin',item.get('T')))
                if key not in seen:seeds.append(item);seen.add(key)
        if progress:progress(f'roll-refinement start: {len(seeds)} seeds')
        for seed_idx,seed in enumerate(seeds,1):
            for angle in cfg.opt5_refine_roll_deg:
                if abs(angle)<1.e-8:continue
                c=dict(seed)
                c['T']=seed['T'] @ pose(Rotation.from_euler('x',angle,degrees=True).as_matrix(),[0,0,0])
                c['tilt_deg']=float(seed['tilt_deg'])+float(angle)
                c['score']=float(seed['score'])-.00025*abs(angle)
                c['opt5_refinement_deg']=float(angle)
                # Do not reuse any prechecked status from the original pose.
                for k in ['_prechecked_target','_prechecked_environment','_prechecked_corridor',
                          '_nonpad_target_surface_clearance_m']:
                    c.pop(k,None)
                screened.append(evaluate(c));refined+=1
            if progress and (seed_idx==1 or seed_idx%3==0 or seed_idx==len(seeds)):
                progress(f'roll-refinement {seed_idx}/{len(seeds)} seeds | {refined} poses | {time.perf_counter()-screen_t0:.1f}s')
    audit['opt5_extra_screened']=refined
    screened.sort(key=lambda item:(item[0],item[1]['score']),reverse=True)
    # Preserve coverage of the axial range while prioritizing joint feasibility.
    limit_per_bin=max(2,int(np.ceil(cfg.candidates_per_object/cfg.axial_samples))*2)
    counts={b:0 for b in bins}
    chosen=[];deferred=[]
    for _,c in screened:
        if counts[c['bin']]<limit_per_bin and len(chosen)<cfg.candidates_per_object:
            chosen.append(c);counts[c['bin']]+=1
        else:deferred.append(c)
    if len(chosen)<cfg.candidates_per_object:
        chosen+=deferred[:cfg.candidates_per_object-len(chosen)]
    audit.update(selection_strategy='gate_aware_with_axial_diversity',evaluated=len(chosen),screened_target_pass=sum(t['_prechecked_target']['status']=='PASS' for _,t in screened),screened_environment_pass=sum(t['_prechecked_environment']['status']=='PASS' for _,t in screened),screened_joint_pass=sum(t['_prechecked_target']['status']=='PASS' and t['_prechecked_environment']['status']=='PASS' for _,t in screened),screened_preapproach_pass=sum(t['_prechecked_corridor'] is not None and t['_prechecked_corridor']['status']=='PASS' for _,t in screened),screened_nonpad_surface_clearance_evaluated=sum(t['_nonpad_target_surface_clearance_m'] is not None for _,t in screened))
    return chosen,audit

def rank_key(r):
    checks=r['checks'];ispass=lambda k:checks[k]['status']=='PASS'
    clear=r.get('nonpad_target_surface_clearance_m')
    clear=float(clear) if clear is not None and np.isfinite(clear) else -1.
    # OPT7 (only when a post-hoc pose rank was attached by scripts/opt7_combined_report.py):
    # best representative of every physically distinct pose first, verified pad contact first.
    diverse='opt7_pose_rank' in r
    return (r['status']=='PASS',-int(r.get('opt7_pose_rank',0)),
            diverse and r.get('contact_basis')=='opt6_bilateral_OEM_pad_contact_verified',
            ispass('target_body') and ispass('environment'),
            ispass('approach'),sum(ispass(k) for k in GATES),
            ispass('environment'),r.get('approach_precheck_corridor_status')=='PASS',
            min(.025,clear),r['score'])

def _rrt_approach(scene,g,M,q,cfg,contacts,start_poses):
    """RRT-Connect over mount translation, fixed orientation, open aperture."""
    R=M[:3,:3];goal=M[:3,3].copy()
    seen=[];starts=[]
    for S in start_poses:
        t=np.asarray(S[:3,3],dtype=float)
        if not any(np.linalg.norm(t-u)<1e-9 for u in seen):seen.append(t);starts.append(t)
    def pose_at(t):
        T=M.copy();T[:3,3]=t;return T
    def edge_free(a,b):
        return scene.translation(g,pose_at(a),pose_at(b),q,contacts=contacts)['status']=='PASS'
    def sample(rng):
        local=np.array([rng.uniform(-cfg.opt7_rrt_lateral_extent_m,cfg.opt7_rrt_lateral_extent_m),
                        rng.uniform(-cfg.opt7_rrt_lateral_extent_m,cfg.opt7_rrt_lateral_extent_m),
                        rng.uniform(-cfg.opt7_rrt_back_extent_m,.01)])
        return goal+R@local
    rng=np.random.default_rng(cfg.seed);attempts=[]
    for k,start in enumerate(starts):
        out=rrt_connect(start,goal,edge_free,sample,step_m=cfg.opt7_rrt_step_m,
            max_iterations=cfg.opt7_rrt_max_iterations,max_edge_checks=cfg.opt7_rrt_max_edge_checks,
            shortcut_passes=cfg.opt7_rrt_shortcut_passes,rng=rng)
        attempts.append(dict(start_index=k,reason=out['reason'],iterations=out['iterations'],edge_checks=out['edge_checks']))
        if out['path'] is not None:
            return [pose_at(t) for t in out['path']],dict(status='PASS',attempts=attempts,waypoints=len(out['path']))
    return None,dict(status='FAIL',attempts=attempts,waypoints=0)

def approach(scene,g,M,q,cfg,C,tcp_local,contacts=None):
    tcp_local=g.contact_geometry(q)[1]  # actual open-jaw contact center
    start=M.copy();start[:3,3]-=M[:3,2]*cfg.approach_distance_m
    candidates=[[start,M]]
    # Deterministic waypoint corridors. These NEVER bypass the continuous
    # swept collision checker and retain a fixed final gripper pose.
    laterals=cfg.opt5_approach_lateral_m if cfg.opt5_enabled else [.06]
    for lateral in laterals:
        for axis in [0,1]:
            for sign in [-1,1]:
                a=start.copy();a[:3,3]+=M[:3,axis]*sign*lateral
                b=M.copy();b[:3,3]-=M[:3,2]*.03
                b[:3,3]+=M[:3,axis]*sign*lateral
                candidates.append([a,b,M])
    if cfg.opt5_enabled:
        for sign0 in [-1,1]:
            for sign1 in [-1,1]:
                a=start.copy();a[:3,3]+=M[:3,0]*sign0*.025+M[:3,1]*sign1*.025
                b=M.copy();b[:3,3]-=M[:3,2]*.03
                b[:3,3]+=M[:3,0]*sign0*.025+M[:3,1]*sign1*.025
                candidates.append([a,b,M])
    checks=[];chosen=candidates[0];planner='deterministic_corridor';rrt_audit=None
    goal_check=scene.clear(g.parts(M,q),target=True,contacts=contacts)
    if goal_check['status']!='PASS':
        checks=[goal_check]
    else:
        found=False
        for path in candidates:
            cc=[]
            for a,b in zip(path[:-1],path[1:]):
                edge=scene.translation(g,a,b,q,contacts=contacts)
                cc.append(edge)
                if edge['status']!='PASS':break
            if not checks:checks=cc
            if len(cc)==len(path)-1 and all(c['status']=='PASS' for c in cc):
                chosen=path;checks=cc;found=True;break
        if not found and cfg.opt7_rrt_enabled:
            # OPT7 fallback: only after every deterministic corridor failed.
            rrt_path,rrt_audit=_rrt_approach(scene,g,M,q,cfg,contacts,[c[0] for c in candidates])
            if rrt_path is not None:
                chosen=rrt_path;planner='rrt_connect'
                # Re-validate each final edge so the stored evidence is the
                # exact same checker output as for every other planner.
                checks=[scene.translation(g,a,b,q,contacts=contacts) for a,b in zip(chosen[:-1],chosen[1:])]
    positions=[(A@tcp_local)[:3,3] for A in chosen];R=(M@tcp_local)[:3,:3];samples=timed_path(positions,R,cfg.max_tcp_speed_m_s,cfg.max_tcp_accel_m_s2,cfg.path_sample_m);quat=Rotation.from_matrix(C[:3,:3]@R).as_quat().tolist()
    for s in samples:
        s['position_camera_m']=transform([s['position_world_m']],C)[0];s['quaternion_camera_xyzw']=quat
        for a,u in [('velocity','m_s'),('acceleration','m_s2')]:s[f'{a}_camera_{u}']=C[:3,:3]@s[f'{a}_world_{u}']
    return dict(status=verdict({'e'+str(i):c for i,c in enumerate(checks)},['e'+str(i) for i in range(len(checks))]),method='continuous convex sweeps, fixed orientation',planner=planner,rrt_audit=rrt_audit,edge_checks=checks,samples=samples,duration_s=samples[-1]['time_s'],waypoints_world_m=positions,opening_joint=q,world_from_mount_at_goal=M,executable=False,scope='gripper geometry in the observed scene; robot arm and IK not checked')

def plan_model(cap,rec,alignment,spec,cfg,out,planes=(),progress=print,pools=None):
    g=CADGripper(spec)
    if cfg.gripper_collision_model=='repaired_solid':
        progress(f'{spec.name}: repairing non-watertight OEM links for exact static collision')
        g.use_repaired_links(cfg.gripper_repair_pitch_m,cfg.gripper_collision_max_faces,cfg.cad_cache_dir)
    C=np.array(cap.by_id[cfg.reference_frame]['camera_from_world']);rows=[];generation={};start=time.perf_counter()
    save(Path(out)/(spec.name+'_hardware.json'),g.evidence())
    for o in cap.objects:
        if cfg.target_instances and o.name not in cfg.target_instances:continue
        progress(f'{spec.name} / {o.name}: proposals')
        pool,audit=pools[o.name] if pools else generate(o,rec.plane[:3],cfg)
        scene=Scene(cap,rec,o,cfg,planes)
        progress(f'{spec.name} / {o.name}: scene ready; screening candidates')
        selected,pruning=select(pool,g,cfg,scene,
            progress=lambda message: progress(f'{spec.name} / {o.name}: {message}'))
        generation[o.name]={**audit,**pruning}
        progress(f'{spec.name} / {o.name}: selected {len(selected)} candidates; adaptive search begins')
        adaptive_t0=time.perf_counter();b_used=0
        for i,c in enumerate(selected):
            if i==0 or (i+1)%8==0 or (i+1)==len(selected):
                progress(f'{spec.name} / {o.name}: candidate {i+1}/{len(selected)} | adaptive time {time.perf_counter()-adaptive_t0:.1f}s')
            q=g.joint(c['width']);qo=g.joint(min(g.width_limits[1],c['width']+spec.opening_extra_m));_,tcp,_=g.contact_geometry(q);M=c['T']@np.linalg.inv(tcp)
            adaptive=None
            solution=None  # shared result from 1-D / OPT6 goal solvers
            staging=None
            pose_phase='contact_goal'
            # Spend adaptive searches on the most promising pre-screened
            # environmental candidates. Exact CAD contact must be bilateral.
            attempt=(cfg.adaptive_closure_enabled and i<cfg.adaptive_attempts_per_object)
            if attempt:
                solution,adaptive=solve_adaptive_closure(g,scene,c,cfg)
                if solution is not None:M,q,qo=solution
            # OPT6: on failures of the one-axis first-contact solver, jointly
            # optimize 3-D OEM mount translation and legal jaw aperture.
            # Static collision checks run at the SAME contact-goal pose,
            # independently of every previously cached/open-stage result.
            goal_optimization=None
            if (cfg.opt6_goal_optimizer_enabled
                    and i<cfg.opt6_goal_attempts_per_object
                    and (not attempt or solution is None)):
                goal_solution,goal_optimization=optimize_contact_goal(g,scene,c,cfg,
                    allow_collision_stage=(cfg.opt7_goal_collision_budget_per_object is None
                                           or b_used<cfg.opt7_goal_collision_budget_per_object))
                if (goal_optimization.get('collision_aware_stage') or {}).get('optimized_starts',0)>0:b_used+=1
                worst=goal_optimization.get('best_worst_gap_m')
                worst_text=(f'{1000*worst:.3f} mm' if worst is not None else 'n/a')
                progress(f'{spec.name} / {o.name}: OPT6 goal {i+1}/{len(selected)} | '
                    f'{goal_optimization["status"]} | CAD gap {worst_text} | '
                    f'{goal_optimization.get("reason")} | '
                    f'distance eval {goal_optimization["distance_evaluations"]}')
                if goal_solution is not None:
                    M,q,qo=goal_solution
                    adaptive=goal_optimization
                    solution=goal_solution
            # A distinct open/pregrasp staging search is useful when target
            # contact fails but environment is already clear. This is NOT a
            # grasp: force contact_geometry FAIL to prevent a false full PASS.
            # For the static goal the final target/env checks still run on the
            # actual OEM geometry, not on an OBB or cached earlier pose.
            if (cfg.opt5_static_pair_enabled and i<cfg.opt5_static_pair_attempts_per_object
                    and solution is None):
                baseline_target=c.get('_prechecked_target')
                baseline_env=c.get('_prechecked_environment')
                if baseline_target is None or baseline_env is None:
                    p0=g.parts(M,q)
                    baseline_target=candidate_target_check(scene,p0,cfg,c['contacts'])
                    baseline_env=scene.clear(p0)
                if baseline_target['status']=='FAIL' and baseline_env['status']=='PASS':
                    stage,staging=find_joint_static_pregrasp(g,scene,c,cfg,
                        max_checks=cfg.opt5_static_pair_max_checks)
                    staging['original_contact_goal_target_body']=baseline_target['status']
                    staging['original_contact_goal_environment']=baseline_env['status']
                    if stage is not None:
                        M,q,qo=stage
                        pose_phase='open_pregrasp_staging_no_contact'
            # Cached gates belong only to ORIGINAL nominal q/M, not an
            # adaptive solution or a relocated staging configuration.
            original_nominal=(not attempt and goal_optimization is None
                              and pose_phase=='contact_goal')
            target=(c.get('_prechecked_target') if original_nominal else None)
            env=(c.get('_prechecked_environment') if original_nominal else None)
            contact_valid=min(c['fit'])>=np.cos(np.deg2rad(cfg.max_contact_angle_deg))
            if pose_phase!='contact_goal':
                contact_valid=False
            elif attempt or (goal_optimization is not None and solution is not None):
                # A SUCCESSFUL search verified bilateral OEM pad contact at the final pose.
                # OPT7: a FAILED goal search no longer disqualifies the candidate; it keeps
                # its unchanged nominal pose and the same geometric contact criterion as an
                # un-searched candidate (otherwise a larger search budget lowered PASS counts).
                contact_valid=(contact_valid and adaptive is not None
                               and adaptive['status']=='PASS')
            elif cfg.adaptive_closure_enabled:
                # Budget-excluded candidates keep their static diagnostics,
                # but may NOT claim a CAD-guided two-sided grasp.
                contact_valid=None
            # New original-triangle-surface model does not shift OEM pads by
            # an artificial tolerance; shallow contact witnesses are checked
            # directly against the mesh and documented separately.
            pad_shift=(0. if cfg.target_body_model=='original_surface_ablation'
                       else cfg.target_penetration_tolerance_m)
            contact_basis=('opt6_bilateral_OEM_pad_contact_verified' if (goal_optimization is not None and solution is not None)
                           else ('adaptive_CAD_closure' if attempt and solution is not None else 'nominal_CAD_ray_geometric_pad_contact_not_verified'))
            checks=dict(data_alignment=alignment[o.name],hardware_geometry=check(g.fit_error<1e-5,source='manufacturer URDF and CAD',aperture_fit_error_m=g.fit_error),kinematics=check(g.joint_limits[0]<=q<=g.joint_limits[1] and g.joint_limits[0]<=qo<=g.joint_limits[1],joint=q,open_joint=qo),contact_geometry=check(contact_valid,normal_alignment=c['fit'],criterion=('OPEN staging only: no target contact, not a grasp' if pose_phase!='contact_goal' else ('CAD-guided two-sided first-contact, not force closure' if attempt else ('UNVERIFIED: adaptive search budget excluded' if cfg.adaptive_closure_enabled else 'geometric opposing contacts, not force closure'))),adaptive_closure=adaptive),self_collision=g.self_collision(q),target_body=target or candidate_target_check(scene,g.parts(M,q,pad_shift),cfg,c['contacts']),environment=env or scene.clear(g.parts(M,q)),filter_policy=check(True if cfg.bbox_policy=='evidence' else None,policy=cfg.bbox_policy,halo_policy=cfg.halo_policy,scene_ablation=(cfg.halo_policy=='experimental_roi'),interpretation=('CONDITIONAL_ON_ASSUMED_HALO_REMOVAL' if cfg.halo_policy=='experimental_roi' else 'OBSERVED_SCENE')))
            # Run closure and approach only if the final configuration is geometrically feasible.
            if all(checks[k]['status']=='PASS' for k in ['contact_geometry','self_collision','target_body','environment']):
                checks['closure']=scene.closure(g,M,qo,q,contacts=c['contacts'] if cfg.target_contact_policy=='local_pad_witness' else None);_,tcp_open,_=g.contact_geometry(qo);traj=approach(scene,g,M,qo,cfg,C,tcp_open,contacts=c['contacts'] if cfg.target_contact_policy=='local_pad_witness' else None);checks['approach']={k:v for k,v in traj.items() if k not in ['samples','waypoints_world_m','world_from_mount_at_goal']}
            else:
                checks['closure']=check(None,reason='Final configuration rejected; closure not validated');checks['approach']=check(None,reason='Final configuration rejected; path not validated');traj=None
            status=verdict(checks,GATES)
            opt7_diag=None
            if cfg.opt7_diagnostics_enabled and i<cfg.opt7_diagnostic_attempts_per_object and pose_phase=='contact_goal':
                _,diag_tcp,_=g.contact_geometry(q)
                opt7_diag=opt7_candidate_diagnostics(g,scene,cfg,M,q,qo,c['contacts'],pad_shift,diag_tcp)
            cl=[]
            if traj:
                duration=max(.05,1.875*(g.width(qo)-g.width(q))/spec.closing_speed_m_s)
                for u in np.linspace(0,1,51):
                    v=10*u**3-15*u**4+6*u**5;w=(1-v)*g.width(qo)+v*g.width(q);j=g.joint(w);_,B,_=g.contact_geometry(j);cl.append(dict(time_s=traj['duration_s']+u*duration,aperture_m=w,joint=j,camera_from_tcp=C@M@B,camera_from_mount=C@M))
            # Recompute actual TCP because both adaptive contact and staging
            # can change q/M relative to the originally proposed CAD rays.
            _,actual_tcp,_=g.contact_geometry(q)
            actual_world_from_tcp=M@actual_tcp
            rows.append(dict(pose_phase=pose_phase,contact_basis=contact_basis,opt7_diagnostics=opt7_diag,static_pregrasp_search=staging,
                opt6_contact_goal_optimization=goal_optimization,
                id=f'{spec.name}__{o.name}__{i:03d}',hardware=spec.name,hardware_model=spec.model,object_instance=o.name,object_type=o.kind,status=status,validation_scope=('GEOMETRY_ONLY_OPEN_CAD_SURFACE_AND_EXPERIMENTAL_HALO_ABLATIONS' if cfg.target_body_model=='original_surface_ablation' and cfg.halo_policy=='experimental_roi' else ('GEOMETRY_ONLY_EXPERIMENTAL_HALO_REMOVAL' if cfg.halo_policy=='experimental_roi' else 'GEOMETRY_ONLY_OBSERVED_SCENE_ANNOTATED_TARGET_MODEL')),required_checks=GATES,checks=checks,score=c['score'],score_definition='contact-normal alignment and centroid; OPT5 optional collision-screened pad-axis roll refinement; no physical score',nonpad_target_surface_clearance_m=c.get('_nonpad_target_surface_clearance_m'),approach_precheck_corridor_status=(c.get('_prechecked_corridor') or {}).get('status','UNVERIFIED'),world_from_tcp=actual_world_from_tcp,camera_from_tcp=C@actual_world_from_tcp,world_from_mount=M,camera_from_mount=C@M,joint=q,open_joint=qo,aperture_m=g.width(q),adaptive_CAD_closure=adaptive,certified_solid_target_collision=(bool(o.mesh.is_watertight) and cfg.halo_policy!='experimental_roi'),contacts_world_m=c['contacts'],contact_normals_world=c['normals'],ray_face_ids=c['faces'],tilt_degrees=c['tilt_deg'],opt5_refinement_deg=c.get('opt5_refinement_deg',0.),trajectory=traj,closure_trajectory=dict(status=checks['closure']['status'],samples=cl,executable=False),robot_reachability='UNVERIFIED',unknown_environment_volume='UNVERIFIED',original_mesh_audit=o.audit,target_collision_model=scene.target_model,target_model_audit=scene.target_model_audit,dynamics_evaluated=False,executable=False))
        progress(f'{spec.name} / {o.name}: {len(selected)} tested, geometry PASS {sum(r["status"]=="PASS" for r in rows if r["object_instance"]==o.name and r["hardware"]==spec.name)}, environment PASS {sum(r["checks"]["environment"]["status"]=="PASS" for r in rows if r["object_instance"]==o.name and r["hardware"]==spec.name)}')
        save(Path(out)/(spec.name+'_candidates.json'),rows)
    save(Path(out)/(spec.name+'_generation.json'),generation);save(Path(out)/(spec.name+'_timing.json'),dict(seconds=time.perf_counter()-start,candidates=len(rows)));return rows
