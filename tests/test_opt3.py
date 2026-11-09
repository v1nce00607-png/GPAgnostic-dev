"""OPT3 isolated, geometry-safe regression tests (no manufacturer assets)."""
from types import SimpleNamespace
import numpy as np
import pytest
from grasp_planner.cleaning import floating_geometry_candidates, floating_depth_confirmed
from grasp_planner.geometry_collision import local_contact_witnesses
from grasp_planner.geometry_config import GeometryConfig
from grasp_planner.geometry_planner import candidate_target_check


def test_floating_rejects_shape_only_and_keeps_plane_or_labeled_points():
    cfg=GeometryConfig(floating_policy='depth_confirmed',floating_min_height_m=.012)
    label=np.array([0,0,0,0,3]);height=np.array([.030,.010,.030,.030,.040])
    sor=np.array([True]*5);radius=np.array([True,True,False,True,True]);dbnoise=np.array([False,True,False,True,True])
    assert floating_geometry_candidates(label,height,sor,radius,dbnoise,cfg).tolist()==[True,False,False,True,False]


def test_free_ray_contradiction_required_and_masks_veto():
    cfg=GeometryConfig(floating_policy='depth_confirmed')
    free=np.array([9,9,5,8,20,15]);occupied=np.array([1,1,1,2,1,1]);support=np.array([0,4,0,0,0,0])
    assert floating_depth_confirmed(free,occupied,support,cfg).tolist()==[True,False,False,False,True,True]


def test_contact_witness_requires_shallow_and_local_evidence():
    anchors=np.array([[0.,0.,0.],[.03,0.,0.]])
    a=SimpleNamespace(point=np.array([.001,0.,0.]),depth=.0002)
    b=SimpleNamespace(point=np.array([.031,0.,0.]),depth=.0003)
    assert local_contact_witnesses([a,b],anchors,.0005,.0025)
    assert not local_contact_witnesses([],anchors,.0005,.0025)
    assert not local_contact_witnesses([SimpleNamespace(point=[.007,0.,0.],depth=.0002)],anchors,.0005,.0025)
    assert not local_contact_witnesses([SimpleNamespace(point=[0.,0.,0.],depth=.001)],anchors,.0005,.0025)
    assert not local_contact_witnesses([SimpleNamespace(point=[np.nan,0.,0.],depth=.0001)],anchors,.0005,.0025)


def test_contact_gate_is_opt_in():
    class S:
        def target_clear(self,parts,contacts=None):
            return {'status':'PASS','contacts_forwarded':contacts is not None}
    s=S();c=np.zeros((2,3));p=[]
    assert candidate_target_check(s,p,GeometryConfig(),c)['contacts_forwarded'] is False
    assert candidate_target_check(s,p,GeometryConfig(target_contact_policy='local_pad_witness'),c)['contacts_forwarded'] is True


def test_config_rejects_unsafe_tolerances():
    with pytest.raises(ValueError):GeometryConfig(target_contact_patch_radius_m=.02)
    with pytest.raises(ValueError):GeometryConfig(target_contact_surface_distance_m=.01)
    with pytest.raises(ValueError):GeometryConfig(floating_min_free_views=2)
    with pytest.raises(ValueError):GeometryConfig(target_penetration_tolerance_m=.005)


def test_link_body_is_never_exempted(monkeypatch):
    import trimesh
    from grasp_planner.geometry_collision import Scene
    cfg=GeometryConfig(target_contact_policy='local_pad_witness')
    anchors=np.array([[0.,0.,0.],[.03,0.,0.]])
    manager=SimpleNamespace(in_collision_single=lambda mesh,return_data=True:(True,[
        SimpleNamespace(point=np.array([.001,0.,0.]),depth=.0001)]))
    s=object.__new__(Scene)
    s.cfg=cfg;s.obj=SimpleNamespace(mesh=trimesh.creation.box(extents=[1,1,1]))
    s.target=manager;s.target_original=manager
    monkeypatch.setattr(trimesh.proximity,'closest_point',lambda mesh,points:(points,np.zeros(len(points)),np.zeros(len(points),int)))
    assert not s._allowed_pad_contact(SimpleNamespace(pad=False),object(),anchors)
    assert s._allowed_pad_contact(SimpleNamespace(pad=True),object(),anchors)


def test_original_CAD_deep_witness_vetoes_tip_exception(monkeypatch):
    import trimesh
    from grasp_planner.geometry_collision import Scene
    cfg=GeometryConfig(target_contact_policy='local_pad_witness')
    anchors=np.array([[0.,0.,0.],[.03,0.,0.]])
    shallow=SimpleNamespace(in_collision_single=lambda mesh,return_data=True:(True,[
        SimpleNamespace(point=np.array([.001,0.,0.]),depth=.0001)]))
    deep=SimpleNamespace(in_collision_single=lambda mesh,return_data=True:(True,[
        SimpleNamespace(point=np.array([.001,0.,0.]),depth=.002)]))
    s=object.__new__(Scene)
    s.cfg=cfg;s.obj=SimpleNamespace(mesh=trimesh.creation.box(extents=[1,1,1]))
    s.target=shallow;s.target_original=deep
    monkeypatch.setattr(trimesh.proximity,'closest_point',lambda mesh,points:(points,np.zeros(len(points)),np.zeros(len(points),int)))
    assert not s._allowed_pad_contact(SimpleNamespace(pad=True),object(),anchors)
