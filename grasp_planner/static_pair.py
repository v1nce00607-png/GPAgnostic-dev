"""Fail-closed search for a collision-free *OPEN* pregrasp staging pose.

This is NOT a target contact or a grasp certificate. Every returned staging
pose is checked against the same target, environment and self-collision gates.
The result is never used to mark contact_geometry or closure PASS.
"""
import numpy as np
from .models import check


def _pregrasp_offsets():
    # Small local offsets only; never move the object or erase obstacles.
    return ((0., 0.), (-.002, 0.), (-.004, 0.), (-.002, -.0015),
            (-.002, .0015), (-.004, -.0015), (-.004, .0015))


def find_joint_static_pregrasp(g, scene, candidate, cfg, max_checks=15):
    """Return (M,q,qo), audit for a certified *static* two-gate open state.

    The opening is bounded by the exact OEM aperture limits. A target-body
    PASS is never inferred from a successful environment PASS or the OBB.
    The returned end pose is explicitly non-contact and may NOT be promoted
    to a grasp. The object's complete original triangles stay in Scene.
    """
    audit=dict(status='UNVERIFIED',role='open_pregrasp_only',
               reason='no_joint_static_clear_pose',evaluations=0,
               collision_model=scene.target_model)
    nominal=float(candidate['width'])
    wmax=float(g.width_limits[1])
    if nominal>wmax+1e-9 or nominal<g.width_limits[0]-1e-9:
        return None,audit|dict(reason='invalid_candidate_aperture')
    qnom=g.joint(nominal)
    _,tcp,_=g.contact_geometry(qnom)
    mount=candidate['T']@np.linalg.inv(tcp)
    # OPEN-stage clearance first: try the maximum available OEM aperture,
    # then smaller openings if nearby scene obstacles forbid wide jaws.
    # Contact-goal search remains a separate, strictly checked stage.
    widths=sorted(set(min(wmax,nominal+d) for d in (.001,.0025,.0045,.007,wmax-nominal)),reverse=True)
    tried=set()
    for w in widths:
        if w<nominal+1e-8:continue
        q=g.joint(w)
        if g.self_collision(q)['status']!='PASS':continue
        for depth,side in _pregrasp_offsets():
            if audit['evaluations']>=max_checks:
                return None,audit|dict(reason='static_search_budget_exhausted')
            M=mount.copy()
            M[:3,3]+=candidate['T'][:3,2]*depth+candidate['T'][:3,1]*side
            key=(round(w,8),depth,side)
            if key in tried:continue
            tried.add(key)
            parts=g.parts(M,q)
            audit['evaluations']+=1
            # Deliberately forbid even localized fingertip penetration at a
            # *non-contact* pregrasp pose. Target PASS must be unconditional.
            target=scene.target_clear(parts,contacts=None)
            if target['status']!='PASS':
                audit['last_target_failure']=dict(link=target.get('link'),cause=target.get('cause'))
                continue
            env=scene.clear(parts)
            if env['status']!='PASS':
                audit['last_environment_failure']=dict(link=env.get('link'),cause=env.get('cause'))
                continue
            # Preserve exact same OEM base transform during subsequent closure.
            return (M,q,q),audit|dict(status='PASS',reason='joint_static_pregrasp_clear',
                    aperture_m=float(w),mount_retreat_m=float(depth),
                    mount_lateral_m=float(side),target_body=target,environment=env,
                    note='OPEN stage only; no CAD contact, closure or approach validated')
    return None,audit
