"""CAD-guided adaptive closure around a local oriented bounding box.

A local OBB sets ONLY the opening; OEM fingertip triangles against original
CAD anchors determine the first bilateral near-contact. No penetration
allowance or environment PASS is inferred from the box.
"""
import numpy as np
import trimesh
from scipy.optimize import minimize_scalar
from .models import check


def local_contact_obb(mesh, world_from_grasp, half_window_m, contacts=None, contact_patch_radius_m=None):
    """Local, jaw-axis-aligned envelope near the proposed CAD contacts.

    Coordinates use the candidate's world_from_grasp rotation: X is closure,
    Y/Z are the bounded local cross-section. All input vertices are original
    CAD vertices (no ray or convex-hull extrapolation).
    """
    v=np.asarray(mesh.vertices,dtype=float)
    T=np.asarray(world_from_grasp,dtype=float)
    local=(v-T[:3,3])@T[:3,:3]
    yz=np.max(np.abs(local[:,1:]),axis=1)
    # When exact ray/CAD contacts are present, use a *contact-local* OBB,
    # not the full +/-12 mm cross-section: that region may contain plier
    # arms which the OEM tips will never touch. The full band is still
    # reported as a diagnostic; it must not silently force a clipped opening.
    broad=local[yz<=half_window_m]
    inside=broad
    mode='broad_YZ_band'
    if contacts is not None:
        c=np.asarray(contacts,dtype=float)
        if c.shape!=(2,3) or not np.isfinite(c).all():
            raise ValueError('Two CAD contacts required for local OBB')
        local_contacts=(c-T[:3,3])@T[:3,:3]
        radius=float(contact_patch_radius_m or min(half_window_m,.003))
        if radius<=0 or radius>half_window_m:
            raise ValueError('Invalid contact patch radius')
        # Restrict the transverse footprint, keeping the original CAD
        # contact points as exact witnesses even on a coarse CAD mesh.
        around=np.min(np.linalg.norm(local[:,None,1:]-local_contacts[None,:,1:],axis=2),axis=1)
        inside=np.vstack([local[around<=radius],local_contacts])
        mode='contact_patch_obb'
    elif len(inside)<6:
        count=min(len(local),max(6,int(.02*len(local))))
        inside=local[np.argsort(yz)[:count]]
    if len(inside)==0:raise ValueError('Target CAD has no vertices')
    return dict(x_min_m=float(inside[:,0].min()),x_max_m=float(inside[:,0].max()),
                width_m=float(np.ptp(inside[:,0])),vertex_count=int(len(inside)),
                broad_band_width_m=float(np.ptp(broad[:,0])) if len(broad) else None,
                selection=mode,half_window_m=float(half_window_m),
                coordinate_frame='candidate_grasp_xyz')


def fingertip_anchor_distances(g,world_from_mount,q,contacts):
    """Minimum unsigned CAD-pad triangle distance to EACH designated anchor.

    Strictly compare right pad to anchor[0] and left to anchor[1]. A nearby
    opposite pad must not pass on behalf of a missing side contact.
    """
    c=np.asarray(contacts,dtype=float)
    if c.shape!=(2,3) or not np.isfinite(c).all():raise ValueError('Two finite CAD anchors required')
    results={}
    for part in g.parts(world_from_mount,q):
        if not part.pad or part.side==0:continue
        side='right' if part.side>0 else 'left'
        index=0 if part.side>0 else 1
        # An OEM part is rigidly transformed from its ORIGINAL raw triangle
        # mesh. Query the unchanged local mesh instead of copying and
        # transforming the entire mesh for every q / offset / candidate.
        # This is mathematically the SAME Euclidean distance under rigid FK.
        # Crucially, trimesh caches the spatial triangle index on part.raw,
        # which is shared across every gripper.parts() invocation.
        R=np.asarray(part.T[:3,:3],dtype=float)
        point_local=(c[index]-part.T[:3,3]) @ R
        _,dist,_=trimesh.proximity.closest_point(
            part.raw,np.asarray([point_local],dtype=float))
        dist=float(dist[0])
        if not np.isfinite(dist):raise ValueError('Non-finite CAD-pad distance')
        results[side]=min(dist,results.get(side,np.inf))
    if set(results)!={'left','right'}:raise ValueError('Both OEM fingertip sides are required')
    return results


def solve_adaptive_closure(g,scene,c,cfg):
    """Search first bilateral CAD-surface contact from a locally OBB-sized opening.

    q is evaluated from the manufacturer aperture map, not by translating STL
    meshes. The OEM mount pose is held fixed throughout each trajectory.
    For asymmetrical contacts, bounded center corrections are screened.

    Fails closed if no two-sided triangle contact or the final static gates fail.
    Does NOT claim force closure, stable grasp, robot IK, or open-shell solidity.
    """
    result=dict(status='UNVERIFIED',reason='no_bilateral_CAD_contact',
                source='original_CAD_triangles_and_OEM_fingertips',
                contact_tolerance_m=cfg.adaptive_contact_tolerance_m)
    try:
        obb=local_contact_obb(scene.obj.mesh,c['T'],cfg.adaptive_obb_half_window_m,
            contacts=c['contacts'],contact_patch_radius_m=cfg.target_contact_patch_radius_m)
        result['local_obb']=obb
        nominal=float(c['width'])
        # A requested opening clearance is a planning preference, not a
        # reason to exceed OEM hard limits. Reject only when the ACTUAL local
        # OBB cannot fit with the minimum 0.2 mm staging clearance.
        hardware_max=float(g.width_limits[1])
        minimum_stage_clearance=.0002
        required_min=obb['width_m']+2*minimum_stage_clearance
        if required_min>hardware_max+1e-9:
            return None,result|dict(reason='local_contact_OBB_exceeds_OEM_aperture',
                minimum_required_opening_m=float(required_min),
                OEM_max_opening_m=hardware_max)
        desired=max(nominal+g.spec.opening_extra_m,
                    obb['width_m']+2*cfg.adaptive_obb_side_clearance_m)
        opening=min(hardware_max,desired)
        result['requested_opening_width_m']=float(desired)
        result['OBB_available_side_margin_m']=float((opening-obb['width_m'])/2)
        w_low=max(g.width_limits[0],nominal-cfg.adaptive_contact_undertravel_m)
        if opening<=w_low:return None,result|dict(reason='No OEM aperture range for contact search')
        q0=g.joint(nominal)
        _,tcp,_=g.contact_geometry(q0)
        reference=c['T']@np.linalg.inv(tcp)
        # Descending opening gives a first-contact search. For robust reuse
        # choose the earliest feasible state rather than deepest penetration.
        widths=np.linspace(opening,w_low,cfg.adaptive_contact_search_steps)
        attempts=0
        for offset in cfg.adaptive_mount_x_offsets_m:
            M=reference.copy()
            M[:3,3]+=c['T'][:3,0]*offset
            # Fast path: the original candidate was already computed from CAD
            # contact rays. Verify its actual OEM pad triangles before doing
            # an expensive 1-D contact search across the whole jaw stroke.
            nominal_gap=fingertip_anchor_distances(g,M,q0,c['contacts'])
            attempts+=1
            if max(nominal_gap.values())<=cfg.adaptive_contact_tolerance_m:
                target=scene.target_clear(g.parts(M,q0),contacts=c['contacts'])
                env=scene.clear(g.parts(M,q0))
                if (target['status']=='PASS' and env['status']=='PASS'
                        and g.self_collision(q0)['status']=='PASS'):
                    return (M,q0,g.joint(float(opening))),dict(
                        status='PASS',reason='bilateral_original_CAD_contact',
                        opening_width_m=float(opening),contact_width_m=float(nominal),
                        closure_aperture_change_m=float(opening-nominal),
                        per_pad_CAD_gap_m=nominal_gap,local_obb=obb,
                        mount_x_offset_m=float(offset),search_evaluations=attempts,
                        contact_tolerance_m=cfg.adaptive_contact_tolerance_m,
                        scope='surface_contacts_only; open_target_solid_UNVERIFIED')
            # OEM jaw movement over the bounded search cannot resolve a
            # gross misregistration of an anchor. Skip hopeless mount offsets.
            if max(nominal_gap.values())>(
                    cfg.adaptive_contact_undertravel_m
                    + cfg.adaptive_contact_tolerance_m + .003):
                continue
            sample=[]
            for width in widths:
                q=g.joint(float(width));attempts+=1
                gaps=fingertip_anchor_distances(g,M,q,c['contacts'])
                sample.append((float(width),gaps))
            best_idx=int(np.argmin([max(d.values()) for _,d in sample]))
            # Coarse samples may skip a sub-millimetre contact window. Refine
            # the closest interval numerically with the *original* pad triangles.
            lo=float(widths[min(len(widths)-1,best_idx+1)])
            hi=float(widths[max(0,best_idx-1)])
            if hi>lo+1.e-8:
                def objective(width):
                    qq=g.joint(float(width))
                    d=fingertip_anchor_distances(g,M,qq,c['contacts'])
                    return max(d.values())
                opt=minimize_scalar(objective,bounds=(lo,hi),method='bounded',
                        options={'xatol':1.e-5,'maxiter':32})
                if opt.success:
                    attempts+=opt.nfev
                    sample.append((float(opt.x),fingertip_anchor_distances(
                        g,M,g.joint(float(opt.x)),c['contacts'])))
            # First feasible contact while closing, not maximal penetration.
            for width,gaps in sorted(sample,key=lambda w:w[0],reverse=True):
                q=g.joint(float(width))
                if max(gaps.values())>cfg.adaptive_contact_tolerance_m:
                    continue
                # Pair contact is necessary but not sufficient. No target-
                # or environment-body overlap may be converted into PASS.
                target=scene.target_clear(g.parts(M,q),contacts=c['contacts'])
                if target['status']!='PASS':continue
                env=scene.clear(g.parts(M,q))
                if env['status']!='PASS':continue
                if g.self_collision(q)['status']!='PASS':continue
                qo=g.joint(float(opening))
                return (M,q,qo),dict(status='PASS',reason='bilateral_original_CAD_contact',
                    opening_width_m=float(opening),contact_width_m=float(width),
                    closure_aperture_change_m=float(opening-width),
                    per_pad_CAD_gap_m=gaps,local_obb=obb,
                    mount_x_offset_m=float(offset),search_evaluations=attempts,
                    contact_tolerance_m=cfg.adaptive_contact_tolerance_m,
                    scope='surface_contacts_only; open_target_solid_UNVERIFIED')
        return None,result|dict(opening_width_m=float(opening),
                contact_width_m=None,search_evaluations=attempts,
                reason='no_two_sided_near_contact_with_static_clearance')
    except (ValueError,RuntimeError,IndexError) as exc:
        return None,result|dict(reason='adaptive_geometry_error',detail=str(exc))
