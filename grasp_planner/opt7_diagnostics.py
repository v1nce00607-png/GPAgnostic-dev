"""OPT7 report-only diagnostics: 180-degree twin and jaw-axis shift scan.

Nothing here can create, upgrade or reorder a PASS: results are stored under
``opt7_diagnostics`` and the candidate's ten gates are decided elsewhere.
"""
import numpy as np
from scipy.spatial.transform import Rotation

def flip180_mount(M,tcp_local):
    """Mount pose after turning the grasp 180 degrees about its approach (TCP z)
    axis through the TCP origin. Both contact anchors swap sides."""
    W=M@tcp_local
    Rz=np.eye(4);Rz[:3,:3]=Rotation.from_euler('z',180,degrees=True).as_matrix()
    return W@Rz@np.linalg.inv(tcp_local)

def _side(link):
    if not link:return None
    n=str(link)
    return 'right' if n.startswith('right') else 'left' if n.startswith('left') else 'base'

def _brief(r):
    keep=('status','cause','link','clearance_lower_bound_m','penetration_depth_m')
    return {k:r[k] for k in keep if k in r}

def colliding_links(scene,parts):
    """Report-only: EVERY link that intersects the target model, with depth.

    ``Scene.target_clear`` stops at the first failing link in iteration order,
    so its reported link (often 'right_*') may only reflect check order.
    """
    rows=[dict(link=r['link'],side=_side(r['link']),pad=r['pad'],penetration_depth_m=r['depth'],
               penetration_witness_count=len(r['witnesses'])) for r in scene.target_depths(parts)]
    return dict(links=rows,sides=sorted({r['side'] for r in rows}),count=len(rows),
                first_in_iteration_order=rows[0]['link'] if rows else None)

def opt7_candidate_diagnostics(g,scene,cfg,M,q,qo,contacts,pad_shift,tcp_local):
    contacts=None if contacts is None else np.asarray(contacts,dtype=float)
    use_contacts=contacts if cfg.target_contact_policy=='local_pad_witness' else None
    out=dict(report_only=True,grasp_valid=False)
    # 1) 180-degree twin at the SAME aperture and swapped anchors.
    Mf=flip180_mount(M,tcp_local)
    swapped=None if use_contacts is None else use_contacts[::-1].copy()
    tf=scene.target_clear(g.parts(Mf,q,pad_shift),contacts=swapped)
    ef=scene.clear(g.parts(Mf,q))
    t0=scene.target_clear(g.parts(M,q,pad_shift),contacts=use_contacts)
    e0=scene.clear(g.parts(M,q))
    out['flip180']=dict(original=dict(target=_brief(t0),environment=_brief(e0),
            target_colliding_links=colliding_links(scene,g.parts(M,q,pad_shift))),
        flipped_all_links=colliding_links(scene,g.parts(Mf,q,pad_shift)),
        flipped=dict(target=_brief(tf),environment=_brief(ef)),
        same_target_status=tf['status']==t0['status'],same_environment_status=ef['status']==e0['status'],
        target_fail_side_swapped=(t0['status']=='FAIL' and tf['status']=='FAIL'
            and _side(t0.get('link')) in ('left','right') and _side(t0.get('link'))!=_side(tf.get('link'))))
    # 2) Open-aperture asymmetric staging: shift mount along the jaw (X) axis.
    # At the closed contact aperture a shift would push one pad into the target
    # by construction, so only the OPEN state is informative here.
    scan=[]
    for d_mm in cfg.opt7_jaw_shift_mm:
        Ms=M.copy();Ms[:3,3]+=M[:3,0]*(d_mm/1000.)
        t=scene.target_clear(g.parts(Ms,qo,0.))
        e=scene.clear(g.parts(Ms,qo))
        scan.append(dict(shift_mm=float(d_mm),target=_brief(t),environment=_brief(e),
                         both_pass=t['status']=='PASS' and e['status']=='PASS'))
    out['jaw_axis_open_shift_scan']=dict(aperture='open',entries=scan,
        any_both_pass=any(x['both_pass'] for x in scan),
        meaning='static OPEN-state collision clearance only; not a grasp, no contact')
    return out

