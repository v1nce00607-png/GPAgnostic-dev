"""Ablation safeguards: the policy can remove uncertain background, never known CAD or the support plane."""
import numpy as np
import trimesh
import pytest
from types import SimpleNamespace
from grasp_planner.cleaning import halo_roi_candidates,halo_delete_candidates
from grasp_planner.geometry_config import GeometryConfig

def obj(name,x):
    return SimpleNamespace(name=name,mesh=trimesh.creation.box(extents=[.04,.04,.04],transform=trimesh.transformations.translation_matrix([x,0.,.05])))

def test_halo_roi_protects_labeled_background_and_table_and_other_cad():
    cfg=GeometryConfig(halo_policy='experimental_roi',halo_objects=['Rachet_001'],halo_margin_m=.015,
                       halo_plane_guard_m=.008,halo_ceiling_margin_m=.035)
    a=obj('Rachet_001',0.);b=obj('Pliers_001',.06)
    pts=np.array([[0.,0.,.065], [0.,0.,.065], [0.,0.,.005], [.06,0,.06], [.04,0,.060], [.0,0.,.12], [.2,0.,.055]])
    labels=np.array([0,3,0,0,0,0,0])
    roi,owners,protected=halo_roi_candidates(pts,labels,np.array([0.,0.,1.,0.]),[a,b],cfg)
    assert roi.tolist()==[True,False,False,False,False,False,False]
    assert owners[0]==0

def test_all_objects_halo_without_screen_vertical():
    cfg=GeometryConfig(halo_policy='experimental_roi',halo_objects=None,halo_margin_m=.005)
    objs=[obj('Rachet_001',0.),obj('Pliers_001',.2)]
    pts=np.array([[0.,0.,.065],[.2,0.,.065]])
    labels=np.array([0,0])
    roi,owners,_=halo_roi_candidates(pts,labels,np.array([0,0,1.,0.]),objs,cfg)
    assert roi.tolist()==[True,True]
    assert owners.tolist()==[0,1]

def test_optimistic_policy_never_removes_semantic_foreground():
    c=np.array([True,True,False]);labels=np.array([0,1,0]);ones=np.ones(3,dtype=bool);zero=np.zeros(3,dtype=bool)
    assert halo_delete_candidates(c,labels,ones,zero,zero,'experimental_roi').tolist()==[True,False,False]
    assert halo_delete_candidates(c,labels,zero,zero,zero,'isolated_only').tolist()==[False,False,False]
    assert halo_delete_candidates(c,labels,ones,zero,zero,'isolated_only').tolist()==[True,False,False]
    assert not halo_delete_candidates(c,labels,ones,zero,zero,'off').any()

def test_roi_config_constrained():
    for kw in [{'halo_margin_m':1.}, {'halo_plane_guard_m':0.}, {'halo_policy':'erase_everything'}, {'halo_objects':['x','x']}]:
        with pytest.raises(ValueError):GeometryConfig(**kw)
