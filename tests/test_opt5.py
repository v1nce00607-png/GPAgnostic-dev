"""Independent OPT5 regression tests (no proprietary OEM assets required)."""
from types import SimpleNamespace
import copy
import numpy as np
import pytest
import trimesh
from grasp_planner.geometry_config import GeometryConfig
from grasp_planner.geometry_planner import select
from grasp_planner.geometry_collision import Scene
from grasp_planner.models import check
from grasp_planner.pipeline import store_diagnostic_checks
from scripts.opt5_cad_audit import boundary_edges


def test_diagnostic_does_not_mutate_scientific_gates():
    cfg=GeometryConfig(opt5_enabled=True)
    row={'checks':{'target_body':{'status':'FAIL'},'approach':{'status':'UNVERIFIED'},'closure':{'status':'UNVERIFIED'}}}
    old=copy.deepcopy(row['checks'])
    store_diagnostic_checks(row,{'status':'PASS','samples':[1],'diagnostic_for_rejected_grasp':True}, {'status':'PASS'},cfg)
    assert row['checks']==old
    assert row['diagnostic_checks']['approach']['status']=='PASS'
    assert 'samples' not in row['diagnostic_checks']['approach']
    assert row['diagnostic_checks']['closure']['status']=='PASS'


def test_2fg7_translation_closure_is_subdivided_without_pass_shortcut(monkeypatch):
    from grasp_planner import geometry_collision as col
    monkeypatch.setattr(col,'swept',lambda a,b,curvature_bound_m=0:object())
    class G:
        spec=SimpleNamespace(model='2fg7')
        def width(self,q):return q
        def parts(self,M,q,pad_tolerance=0.):return [q]
        def self_collision(self,q):return check(True)
    class S:
        cfg=GeometryConfig(opt5_enabled=True,opt5_closure_step_m=.002)
        def clear(self,p):return check(True)
        def target_clear(self,p,contacts=None):return check(True)
    s=S();r=Scene.closure(s,G(),np.eye(4),0.,.012)
    assert r['status']=='PASS'
    assert r['segments']==6
    s.cfg=GeometryConfig(opt5_enabled=False)
    assert Scene.closure(s,G(),np.eye(4),0.,.012)['segments']==1
    s.cfg=GeometryConfig(opt5_enabled=True,opt5_closure_step_m=.002)
    s.target_clear=lambda p,contacts=None:check(False,cause='deep_body_collision')
    assert Scene.closure(s,G(),np.eye(4),0.,.012)['status']=='FAIL'


def test_joint_gate_refinement_finds_actual_collision_checked_orientation():
    cfg=GeometryConfig(opt5_enabled=True,candidate_screen_multiplier=2,candidates_per_object=10,
        axial_samples=5,approach_precheck_for_joint_pass=False,target_body_clearance_ranking=False,
        opt5_refine_seeds_per_group=1,opt5_refine_roll_deg=[-25.,25.])
    class G:
        width_limits=(.001,.10)
        spec=SimpleNamespace(opening_extra_m=.012)
        def joint(self,width):return width
        def contact_geometry(self,q):return q,np.eye(4),None
        def parts(self,M,q,pad_tolerance=0):return [M]
    class S:
        def target_clear(self,parts,contacts=None):
            # True only at +/-25 degrees, not at the original 0-degree pose.
            return check(parts[0][2,2]<.95)
        def clear(self,parts):return check(True,clearance_lower_bound_m=.01)
    src=[{'T':np.eye(4),'width':.04,'contacts':np.zeros((2,3)),
          'fit':np.array([1.,1.]),'score':.9,'bin':0,'tilt_deg':0.}]
    chosen,audit=select(src,G(),cfg,S())
    assert audit['opt5_extra_screened']==2
    assert audit['screened_joint_pass']==2
    assert chosen[0]['_prechecked_target']['status']=='PASS'
    assert chosen[0]['_prechecked_environment']['status']=='PASS'
    assert abs(chosen[0]['opt5_refinement_deg'])==25.


def test_mesh_boundary_edge_diagnostic_detects_missing_face():
    m=trimesh.creation.box()
    assert boundary_edges(m)==0
    o=trimesh.Trimesh(vertices=m.vertices,faces=m.faces[:-1],process=False)
    assert boundary_edges(o)==3


def test_opt5_config_has_bounded_parameters():
    assert GeometryConfig(opt5_enabled=True).opt5_closure_step_m==.002
    for params in [{'opt5_refine_roll_deg':[90.]},
                   {'opt5_approach_lateral_m':[.5]},
                   {'opt5_closure_step_m':.05}]:
        with pytest.raises(ValueError):GeometryConfig(**params)


def test_multiple_urdf_collision_nodes_are_not_silently_dropped(tmp_path,monkeypatch):
    import grasp_planner.ur_hardware as hw
    root=tmp_path/'onrobot_2fg7'
    (root/'meshes').mkdir(parents=True)
    (root/'srdf').mkdir()
    first=trimesh.creation.box(extents=[.01,.01,.01])
    second=trimesh.creation.box(extents=[.01,.01,.01])
    first.export(root/'meshes'/'a.STL');second.export(root/'meshes'/'b.STL')
    (root/'model.urdf').write_text('''<robot name="test"><link name="base_link">
      <collision><origin xyz="0 0 0"/><geometry><mesh filename="a.STL"/></geometry></collision>
      <collision><origin xyz="0.1 0 0"/><geometry><mesh filename="b.STL"/></geometry></collision>
    </link></robot>''')
    (root/'srdf'/'onrobot_2fg7.srdf').write_text('<robot name="test"/>')
    monkeypatch.setattr(hw,'ASSETS',tmp_path)
    hw.resources.cache_clear()
    try:
        links,_,_=hw.resources('2fg7')
        m=links['base_link'][0]
        assert len(m.faces)==len(first.faces)+len(second.faces)
        assert m.extents[0]>.105
    finally:
        hw.resources.cache_clear()
