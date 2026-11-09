"""Regression: missing OEM VISUAL DAE never masks missing COLLISION geometry."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
import trimesh
from scripts import opt5_cad_audit as audit


@pytest.fixture
def dummy_oem(tmp_path, monkeypatch):
    root = tmp_path / 'onrobot_2fg7'
    (root / 'meshes').mkdir(parents=True)
    box = trimesh.creation.box(extents=[.01, .02, .03])
    box.export(root / 'meshes' / 'base_link.STL')
    class Gripper:
        joint_limits = (0.0, .1)
        def __init__(self, spec):
            self.spec = spec
        def fk(self, q):
            return {'base_link':np.eye(4)}
    monkeypatch.setattr(audit, 'ASSETS', tmp_path)
    monkeypatch.setattr(audit, 'CADGripper', Gripper)
    return root


def test_missing_visual_dae_does_not_block_valid_collision_audit(dummy_oem, tmp_path):
    (dummy_oem / 'model.urdf').write_text('''<robot name="test">
      <link name="base_link">
        <visual><geometry><mesh filename="package://onrobot_2fg7/meshes/base_link.dae" /></geometry></visual>
        <collision><geometry><mesh filename="package://onrobot_2fg7/meshes/base_link.STL" /></geometry></collision>
      </link></robot>''', encoding='utf-8')
    out=tmp_path/'report'
    r=audit.audit_model('2fg7',out,html=True)
    assert r['collision_status']=='COMPLETE'
    assert r['collision_loaded']==1
    assert r['visual_status']=='INCOMPLETE'
    assert r['visual_loaded']==0
    assert len(r['visual_sources_missing'])==1
    assert r['visual_sources_missing'][0]['uri'].endswith('base_link.dae')
    assert (out/'OPT5_2FG7_visual_collision.html').is_file()
    assert json.loads((out/'OPT5_2FG7_CAD_audit.json').read_text())['visual_status']=='INCOMPLETE'


def test_missing_collision_mesh_remains_fatal(dummy_oem, tmp_path):
    (dummy_oem/'model.urdf').write_text('''<robot name="test">
      <link name="base_link">
        <visual><geometry><mesh filename="package://onrobot_2fg7/meshes/base_link.dae" /></geometry></visual>
        <collision><geometry><mesh filename="package://onrobot_2fg7/meshes/MISSING.STL" /></geometry></collision>
      </link></robot>''', encoding='utf-8')
    with pytest.raises(FileNotFoundError,match='URDF mesh missing'):
        audit.audit_model('2fg7',tmp_path/'bad',html=False)


def test_complete_visual_and_collision_report_as_complete(dummy_oem,tmp_path):
    (dummy_oem/'meshes'/'base_link_vis.STL').write_bytes((dummy_oem/'meshes'/'base_link.STL').read_bytes())
    (dummy_oem/'model.urdf').write_text('''<robot name="test">
      <link name="base_link">
        <visual><geometry><mesh filename="base_link_vis.STL" /></geometry></visual>
        <collision><geometry><mesh filename="base_link.STL" /></geometry></collision>
      </link></robot>''', encoding='utf-8')
    r=audit.audit_model('2fg7',tmp_path/'complete',html=False)
    assert r['visual_status']=='COMPLETE' and r['visual_loaded']==1
    assert r['collision_status']=='COMPLETE' and r['collision_loaded']==1
    assert r['visual_sources_missing']==[]
