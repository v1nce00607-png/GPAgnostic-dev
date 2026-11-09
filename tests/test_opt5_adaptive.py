"""OPT5 adaptive closure, conditional open CAD and no-false-PASS regressions."""
from types import SimpleNamespace
import numpy as np
import trimesh
import pytest
from grasp_planner.models import check
from grasp_planner.geometry_config import GeometryConfig
from grasp_planner.ur_hardware import Part
from grasp_planner.adaptive_contact import (
    local_contact_obb,fingertip_anchor_distances,solve_adaptive_closure)
from grasp_planner.geometry_collision import Scene


def test_local_box_is_bounded_cross_section_not_entire_target():
    shaft=trimesh.creation.box(extents=[.02,.01,.01]);shaft.apply_translation([0,0,0])
    knob=trimesh.creation.box(extents=[.1,.01,.01]);knob.apply_translation([0,.1,0])
    combined=trimesh.util.concatenate([shaft,knob])
    bb=local_contact_obb(combined,np.eye(4),.012)
    assert bb['width_m']==pytest.approx(.02)
    assert bb['vertex_count']==8
    assert bb['half_window_m']==.012


def test_adaptive_config_is_opt_in_and_parameter_bounded():
    assert not GeometryConfig().adaptive_closure_enabled
    assert GeometryConfig().target_body_model=='conservative_hull'
    for kwargs in ({'adaptive_contact_tolerance_m':.1},
                   {'adaptive_mount_x_offsets_m':[.009]},
                   {'target_body_model':'make_everything_pass'}):
        with pytest.raises(ValueError):GeometryConfig(**kwargs)


def _pads(q):
    # Fingertip inward faces at +/- q/2, each 2 mm thick.
    left=trimesh.creation.box(extents=[.002,.006,.01]);left.apply_translation([-(q/2+.001),0,0])
    right=trimesh.creation.box(extents=[.002,.006,.01]);right.apply_translation([q/2+.001,0,0])
    return [Part('right_finger_tip_link',right,right,np.eye(4),True),
            Part('left_finger_tip_link',left,left,np.eye(4),True)]


class ToyGripper:
    spec=SimpleNamespace(opening_extra_m=.012)
    width_limits=(.005,.05)
    def joint(self,w):return float(w)
    def width(self,q):return float(q)
    def contact_geometry(self,q):return q,np.eye(4),None
    def parts(self,M,q,pad_tolerance=0):
        ps=_pads(q)
        for p in ps:p.T=M
        return ps
    def self_collision(self,q):return check(True)


def test_bilateral_contact_from_CAD_and_real_finger_kinematics(monkeypatch):
    # Naive trimesh nearest triangle implementation permits CAD regression
    # without the optional Rtree backend on a development-only CI machine.
    monkeypatch.setattr(trimesh.proximity,'closest_point',trimesh.proximity.closest_point_naive)
    cad=trimesh.creation.box(extents=[.02,.018,.015])
    contacts=np.array([[.01,0,0],[-.01,0,0]])
    scene=SimpleNamespace(obj=SimpleNamespace(mesh=cad),target_clear=lambda p,contacts:check(True),clear=lambda p:check(True))
    cfg=GeometryConfig(adaptive_closure_enabled=True,adaptive_contact_search_steps=17,
        adaptive_mount_x_offsets_m=[0.],adaptive_contact_tolerance_m=.0002)
    c=dict(T=np.eye(4),width=.02,contacts=contacts)
    result,audit=solve_adaptive_closure(ToyGripper(),scene,c,cfg)
    assert result is not None and audit['status']=='PASS'
    M,q,qo=result
    assert gwidth(q:=float(q))==pytest.approx(.02,abs=.0002)
    assert qo>q
    assert audit['closure_aperture_change_m']>0
    assert audit['per_pad_CAD_gap_m']['left']<.0002
    assert audit['per_pad_CAD_gap_m']['right']<.0002


def gwidth(q):return q


def test_incomplete_pad_side_cannot_be_falsely_accepted(monkeypatch):
    monkeypatch.setattr(trimesh.proximity,'closest_point',trimesh.proximity.closest_point_naive)
    class OneFinger(ToyGripper):
        def parts(self,M,q,pad_tolerance=0):return super().parts(M,q,pad_tolerance)[:1]
    with pytest.raises(ValueError):fingertip_anchor_distances(OneFinger(),np.eye(4),.02,
                                                   np.array([[.01,0,0],[-.01,0,0]]))


def test_target_surface_ablation_does_not_hide_original_nonpad_collision():
    """An FCL original-triangle hit on the BODY always fails (even if open CAD)."""
    obj=SimpleNamespace(mesh=SimpleNamespace(is_watertight=False,bounds=np.array([[-.02]*3,[.02]*3])))
    m=trimesh.creation.box(extents=[.005]*3)
    class Collision:
        def __init__(self,hit):self.hit=hit
        def in_collision_single(self,*a,**kw):
            return (self.hit,[]) if kw.get('return_data') else self.hit
    scene=Scene.__new__(Scene)
    scene.cfg=GeometryConfig(target_body_model='original_surface_ablation')
    scene.obj=obj;scene.target_original=Collision(True);scene.target=Collision(True)
    p=Part('base_link',m,m,np.eye(4),False)
    r=scene.target_clear([p])
    assert r['status']=='FAIL'
    assert r['cause']=='original_target_CAD_triangle_intersection'


def test_target_surface_ablation_only_conditional_when_hull_conflicts():
    obj=SimpleNamespace(mesh=SimpleNamespace(is_watertight=False,bounds=np.array([[-.02]*3,[.02]*3])))
    m=trimesh.creation.box(extents=[.005]*3)
    class Collision:
        def __init__(self,hit):self.hit=hit
        def in_collision_single(self,*a,**kw):
            return (self.hit,[]) if kw.get('return_data') else self.hit
    scene=Scene.__new__(Scene)
    scene.cfg=GeometryConfig(target_body_model='original_surface_ablation')
    scene.obj=obj;scene.target_original=Collision(False);scene.target=Collision(True)
    p=Part('base_link',m,m,np.eye(4),False)
    r=scene.target_clear([p])
    assert r['status']=='PASS'
    assert r['solid_interior']=='UNVERIFIED_NONWATERTIGHT'
    assert r['conditional_surface_model'] is True
    assert r['convex_envelope_overlap_not_proven_solid']==['base_link']
