"""Bounded OEM CAD contact-goal search (OPT6 experiment).

The two original ray/CAD contact anchors are fixed in world coordinates.
Only the OEM gripper mount translation and its permitted jaw joint move.
Neither obstacles nor collision tolerances are edited. A geometric PASS
requires BOTH actual OEM fingertip anchors, original target-CAD clearance,
non-target environment clearance, and self-collision clearance at one pose.
A goal can still FAIL approach/closure; these remain distinct gates.
"""
import time
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .adaptive_contact import local_contact_obb


def original_pad_anchor_residuals(g, M, q, anchors):
    """World-space nearest-triangle residual, separately for both OEM pads.

    Return a 6-vector (right XYZ, left XYZ). Unlike a mere aperture estimate,
    these signed coordinate residuals also constrain lateral/approach offsets.
    The closest point is evaluated on the ORIGINAL collision triangles.
    """
    anchors=np.asarray(anchors,dtype=float)
    if anchors.shape!=(2,3) or not np.isfinite(anchors).all():
        raise ValueError('Expected two finite original-CAD contact anchors')
    residuals={}
    for part in g.parts(M,q):
        if not part.pad or not part.side:
            continue
        side='right' if part.side>0 else 'left'
        ix=0 if part.side>0 else 1
        R=np.asarray(part.T[:3,:3],dtype=float)
        anchor_local=(anchors[ix]-part.T[:3,3])@R
        # Only the jaw-INWARD facing OEM triangles can establish a pinch.
        # Using the entire fingertip allows its OUTER face to touch one CAD
        # anchor while the actual inward pad is still millimetres away.
        inward_world=-float(part.side)*np.asarray(M[:3,0],dtype=float)
        inward_local=inward_world@R
        key=tuple(np.round(inward_local,3))
        cache=getattr(part.raw,'_opt6_inward_triangle_cache',None)
        if cache is None:
            cache={}
            part.raw._opt6_inward_triangle_cache=cache
        if key not in cache:
            faces=np.flatnonzero(part.raw.face_normals@inward_local>=.5)
            if len(faces)==0:
                raise ValueError('OEM fingertip has no inward-facing triangles')
            cache[key]=trimesh.Trimesh(vertices=part.raw.vertices,
                faces=part.raw.faces[faces],process=False)
        closest,dist,_=trimesh.proximity.closest_point(cache[key],anchor_local[None,:])
        if not np.isfinite(closest).all() or not np.isfinite(dist).all():
            raise ValueError('Non-finite original OEM pad distance')
        # Put residual back into world coordinates, including its direction.
        residual=(anchor_local-closest[0])@R.T
        if side not in residuals or np.linalg.norm(residual)<np.linalg.norm(residuals[side]):
            residuals[side]=residual
    if set(residuals)!={'right','left'}:
        raise ValueError('Both original OEM fingertip pads required')
    return np.concatenate([residuals['right'],residuals['left']])


def _rolled_mount(M,anchors,axis,roll_deg):
    """Rotate the mount about the jaw axis through the contact midpoint (OPT7).

    The closing direction is unchanged, so both pad faces still face the anchors."""
    if abs(roll_deg)<1e-12:return M
    R=Rotation.from_rotvec(np.deg2rad(roll_deg)*np.asarray(axis,dtype=float)).as_matrix()
    P=np.asarray(anchors,dtype=float).mean(0);T=np.eye(4);T[:3,:3]=R;T[:3,3]=P-R@P
    return T@M


class _Budget(Exception):
    pass


def _translated_mount(reference,axes,xyz):
    M=reference.copy()
    M[:3,3]+=axes@np.asarray(xyz,dtype=float)
    return M


def optimize_contact_goal(g,scene,c,cfg,allow_collision_stage=True):
    """Return ((mount, contact_q, open_q), audit) or (None, audit).

    Bounded multi-axis nonlinear least squares is a CONTACT-CANDIDATE
    generator, not a collision checker. All terminal candidates pass through
    unmodified Scene.target_clear / Scene.clear and OEM self_collision.
    """
    audit=dict(status='UNVERIFIED',method='bounded_3d_mount_and_OEM_aperture',
        reason='no_valid_original_CAD_contact_goal',distance_evaluations=0,
        static_evaluations=0,search_solutions=0,
        contact_tolerance_m=float(cfg.adaptive_contact_tolerance_m),
        target_collision_model=getattr(scene,'target_model','unknown'))
    try:
        anchors=np.asarray(c['contacts'],dtype=float)
        if anchors.shape!=(2,3) or not np.isfinite(anchors).all():
            return None,audit|dict(reason='invalid_original_CAD_anchors')
        nearest,anchor_dist,_=trimesh.proximity.closest_point(scene.obj.mesh,anchors)
        if not np.isfinite(anchor_dist).all() or max(anchor_dist)>cfg.target_contact_surface_distance_m:
            return None,audit|dict(reason='anchors_not_on_original_CAD_surface',
                anchor_error_m=[float(x) for x in anchor_dist])
        nominal=float(c['width'])
        if not g.width_limits[0]<=nominal<=g.width_limits[1]:
            return None,audit|dict(reason='nominal_outside_OEM_aperture')
        qnom=g.joint(nominal)
        _,tcp,_=g.contact_geometry(qnom)
        nominal_mount=np.asarray(c['T'])@np.linalg.inv(tcp)
        axes=np.asarray(c['T'][:3,:3],dtype=float)
        translation_limit=float(cfg.opt6_goal_translation_limit_m)
        # Only OEM-joint-feasible openings, not inflated surrogate jaws.
        lower=max(float(g.width_limits[0]),nominal-cfg.opt6_goal_extra_close_m)
        upper=float(g.width_limits[1])
        if upper-lower<=1e-7:
            return None,audit|dict(reason='no_OEM_joint_search_range')
        try:
            obb=local_contact_obb(scene.obj.mesh,c['T'],cfg.adaptive_obb_half_window_m,
                contacts=anchors,contact_patch_radius_m=cfg.target_contact_patch_radius_m)
            audit['contact_local_obb']=obb
        except ValueError as exc:
            audit['obb_warning']=str(exc)
        bounds_lo=np.array([-translation_limit]*3+[lower],dtype=float)
        bounds_hi=np.array([translation_limit]*3+[upper],dtype=float)
        width0=float(np.clip(nominal,lower+1e-7,upper-1e-7))
        seed_offsets=[(0.,0.,0.),(0.,0.,-.004),(0.,-.003,0.),(0.,.003,0.),(-.003,0.,0.)]
        seeds=seed_offsets[:cfg.opt6_goal_multistart]
        trial_points=[]
        best_miss=float('inf');best_bad=None;seen=set()

        def eval_residual(x):
            audit['distance_evaluations']+=1
            M=_translated_mount(nominal_mount,axes,x[:3])
            q=g.joint(float(x[3]))
            d=original_pad_anchor_residuals(g,M,q,anchors)
            # Weak pose regularization prevents unnecessary large motions
            # without overriding near-contact accuracy.
            return np.r_[d,.03*np.asarray(x[:3]),.015*(x[3]-nominal)]

        for seed_xyz in seeds:
            if audit['distance_evaluations']>=cfg.opt6_goal_max_distance_evaluations:
                break
            x0=np.array([*seed_xyz,width0],dtype=float)
            x0[:3]=np.clip(x0[:3],bounds_lo[:3]+1e-7,bounds_hi[:3]-1e-7)
            # Try numerical optimization with a bounded number of iterations.
            result=least_squares(eval_residual,x0,bounds=(bounds_lo,bounds_hi),
                max_nfev=cfg.opt6_goal_max_nfev_per_seed,xtol=1e-7,ftol=1e-7,
                gtol=1e-7,method='trf')
            # A 4-D optimizer may find contact near but not exactly at an
            # iteration endpoint. Retest nominal and the returned optimum.
            trial_points.extend([x0,result.x])
            audit['search_solutions']+=1
        state={'bad':None}

        def build_items(points,extra):
            items=[]
            for x in points:
                x=np.asarray(x,dtype=float);key=tuple(np.round(x,7))
                if key in seen:continue
                seen.add(key)
                M=_translated_mount(nominal_mount,axes,x[:3]);q=g.joint(float(x[3]))
                gaps=np.linalg.norm(original_pad_anchor_residuals(g,M,q,anchors).reshape(2,3),axis=1)
                items.append((float(max(gaps)),x,M,q,gaps,dict(extra)))
            return items

        def verify(items):
            """Unmodified static gates at the precise SAME pose, smallest CAD gap first."""
            for worst,x,M,q,gaps,extra in sorted(items,key=lambda t:t[0]):
                if worst>cfg.adaptive_contact_tolerance_m:
                    continue
                audit['static_evaluations']+=1
                parts=g.parts(M,q)
                target=scene.target_clear(parts,contacts=anchors)
                if target['status']!='PASS':
                    state['bad']=dict(gate='target_body',link=target.get('link'),cause=target.get('cause'))
                    continue
                environment=scene.clear(parts)
                if environment['status']!='PASS':
                    state['bad']=dict(gate='environment',link=environment.get('link'),cause=environment.get('cause'))
                    continue
                self_col=g.self_collision(q)
                if self_col['status']!='PASS':
                    state['bad']=dict(gate='self_collision',pairs=self_col.get('pairs'))
                    continue
                # Use OEM-available opening and require genuine non-zero closure.
                w_contact=float(g.width(q))
                w_open=min(float(g.width_limits[1]),max(
                    w_contact+cfg.opt6_goal_min_closure_m,
                    w_contact+float(g.spec.opening_extra_m)))
                if w_open<=w_contact+cfg.opt6_goal_min_closure_m-1e-8:
                    state['bad']=dict(gate='opening',reason='no_legal_open_closure_span')
                    continue
                qo=g.joint(float(w_open))
                # The opening is not presumed clear. Keep its status diagnostic;
                # the full trajectory's own swept test decides closure PASS.
                open_target=scene.target_clear(g.parts(M,qo),contacts=None)
                open_env=scene.clear(g.parts(M,qo))
                record=audit|dict(status='PASS',reason='multiaxis_bilateral_original_CAD_contact',
                    source='original_CAD_triangles_and_OEM_fingertips',
                    mount_translation_grasp_xyz_m=[float(y) for y in x[:3]],
                    contact_width_m=w_contact,opening_width_m=float(w_open),
                    closure_aperture_change_m=float(w_open-w_contact),
                    per_pad_CAD_gap_m={'right':float(gaps[0]),'left':float(gaps[1])},
                    best_worst_gap_m=worst,opening_target_body=open_target,
                    opening_environment=open_env,contact_goal_target_body=target,
                    contact_goal_environment=environment,
                    scope='CONDITIONAL original open-CAD surface and inherited occupancy assumptions',**extra)
                return (M,q,qo),record
            return None

        def collision_stage(starts):
            """OPT7: bounded search that trades pose for lower penetration/clearance deficit.

            Objective = 2*contact residuals + penetration (beyond the allowed pad
            tolerance) + environment-clearance deficit. It only PROPOSES poses;
            verify() decides. Variables (mm,mm,mm,mm,deg): mount XYZ, aperture offset, roll.
            """
            t0=time.perf_counter()
            info=dict(starts=len(starts),evaluations=0,budget=int(cfg.opt7_goal_collision_max_evals),
                      best=None,stopped_by_budget=False)
            tol=cfg.target_penetration_tolerance_m
            margin=cfg.environment_clearance_m+cfg.opt7_goal_env_safety_m
            axis=np.asarray(axes[:,0],dtype=float);axis=axis/np.linalg.norm(axis)
            roll_lim=float(cfg.opt7_goal_roll_limit_deg)
            lo_u=np.array([-translation_limit*1e3]*3+[(lower-width0)*1e3,-roll_lim])
            hi_u=np.array([translation_limit*1e3]*3+[(upper-width0)*1e3,roll_lim])
            trace=[]
            def pose(u):
                M=_rolled_mount(_translated_mount(nominal_mount,axes,np.asarray(u[:3])/1e3),anchors,axis,float(u[4]))
                return M,float(np.clip(width0+u[3]/1e3,lower+1e-7,upper-1e-7))
            def objective(u):
                if info['evaluations']>=info['budget']:
                    info['stopped_by_budget']=True;raise _Budget()
                info['evaluations']+=1
                M,w=pose(u);q=g.joint(w)
                contact=original_pad_anchor_residuals(g,M,q,anchors)
                parts=g.parts(M,q)
                pen=0.
                # Same rule as Scene._allowed_pad_contact: a pad may touch only
                # shallowly (<= tol) and only within the patch radius of an anchor;
                # every other witness counts with its full depth.
                by_name={x.name:x for x in parts}
                for row in scene.target_depths(parts):
                    link_pen=0. if row['witnesses'] else (.001 if row['depth'] is None else float(row['depth']))
                    region=scene.pad_region(by_name[row['link']]) if cfg.target_contact_patch_mode=='pad_face_rect' else None
                    for pt,d in row['witnesses']:
                        near=(region(pt) if region is not None
                              else float(np.min(np.linalg.norm(anchors-pt,axis=1)))<=cfg.target_contact_patch_radius_m)
                        link_pen=max(link_pen,max(0.,d-tol) if (row['pad'] and near) else d)
                    pen=max(pen,link_pen)
                # Floor: when the continuous penalty vanishes, the real gate (which also
                # checks original-CAD intersections) can still reject -> keep a signal.
                if pen<=1e-9 and scene.target_clear(parts,contacts=anchors)['status']!='PASS':
                    pen=.0005
                clr=scene.clear(parts).get('clearance_lower_bound_m')
                deficit=0. if clr is None else max(0.,margin-float(clr))
                r=np.r_[2.*contact,cfg.opt7_goal_penetration_weight*pen,
                        cfg.opt7_goal_clearance_weight*deficit,
                        .03*np.asarray(u[:3])/1e3,.015*u[3]/1e3]
                trace.append((float(r@r),np.array(u,dtype=float),float(pen),float(deficit),float(np.max(np.abs(contact)))))
                return r
            per_start=max(2,int(cfg.opt7_goal_collision_max_evals)//max(1,len(starts))//(len(lo_u)+1))
            ends=[]
            info['skipped_starts']=0;info['optimized_starts']=0;info['start_penetration_m']=None;info['start_deficit_m']=None;info['filtered_by']=None;probes=[]
            for x in starts:
                u0=np.r_[np.asarray(x[:3])*1e3,(float(x[3])-width0)*1e3,0.]
                u0=np.clip(u0,lo_u+1e-6,hi_u-1e-6)
                try:
                    objective(u0)                      # cheap feasibility probe of the start
                    probes.append((trace[-1][2],trace[-1][3]))
                    if trace[-1][2]>cfg.opt7_goal_collision_skip_depth_m:
                        info['skipped_starts']+=1;info['filtered_by']=info['filtered_by'] or 'penetration';continue
                    if cfg.opt7_goal_collision_skip_deficit_m is not None and trace[-1][3]>cfg.opt7_goal_collision_skip_deficit_m:
                        info['skipped_starts']+=1;info['filtered_by']=info['filtered_by'] or 'clearance_deficit';continue
                    info['optimized_starts']+=1
                    res=least_squares(objective,u0,bounds=(lo_u,hi_u),max_nfev=per_start,diff_step=.3,
                        x_scale=np.array([1.,1.,1.,1.,2.]),xtol=1e-6,ftol=1e-6,gtol=1e-6,method='trf')
                    ends.append(res.x)
                except _Budget:
                    break
            # Candidates = optimizer end points plus the best-scoring evaluated points.
            pts=list(ends)+[t[1] for t in sorted(trace,key=lambda t:t[0])[:4]]
            items=[]
            for u in pts:
                M,w=pose(u);q=g.joint(w)
                gaps=np.linalg.norm(original_pad_anchor_residuals(g,M,q,anchors).reshape(2,3),axis=1)
                xyz=np.asarray(u[:3])/1e3
                items.append((float(max(gaps)),np.r_[xyz,w],M,q,gaps,
                    dict(mount_roll_deg=float(u[4]),search_stage='collision_aware')))
            info['seconds']=time.perf_counter()-t0
            if probes:info['start_penetration_m']=float(min(p[0] for p in probes));info['start_deficit_m']=float(min(p[1] for p in probes))
            if trace:
                b=min(trace,key=lambda t:t[0]);info['best']=dict(objective=b[0],penetration_beyond_tolerance_m=b[2],
                    clearance_deficit_m=b[3],worst_contact_residual_m=b[4])
            return items,info

        scored=build_items(trial_points,dict(mount_roll_deg=0.,search_stage='contact_only'))
        best_miss=min([t[0] for t in scored],default=float('inf'))
        found=verify(scored)
        if found is not None:
            return found
        if cfg.opt7_goal_collision_aware and scored and not allow_collision_stage:
            audit['collision_aware_stage']=dict(skipped='per-object budget used up',optimized_starts=0)
        elif cfg.opt7_goal_collision_aware and scored:
            starts=[t[1] for t in sorted(scored,key=lambda t:t[0])[:3]]
            stage_items,stage_info=collision_stage(starts)
            audit['collision_aware_stage']=stage_info
            found=verify(stage_items)
            if found is not None:
                return found
        audit.update(best_worst_gap_m=best_miss if np.isfinite(best_miss) else None,
                     near_contact_count=sum(t[0]<=cfg.adaptive_contact_tolerance_m for t in scored),
                     last_static_failure=state['bad'],
                     reason=('near_contact_rejected_by_static_collision' if state['bad'] else
                             'no_bilateral_contact_within_bounded_multiaxis_search'))
        return None,audit
    except (ValueError,RuntimeError,IndexError,TypeError) as exc:
        return None,audit|dict(reason='contact_goal_optimization_error',detail=str(exc))
