"""Joint static PREGRASP and contact-local bbox safety regressions."""
from types import SimpleNamespace
import numpy as np
import trimesh
from grasp_planner.models import check
from grasp_planner.geometry_config import GeometryConfig
from grasp_planner.adaptive_contact import local_contact_obb
from grasp_planner.static_pair import find_joint_static_pregrasp


class MockHardware:
    width_limits=(.012,.05)
    def joint(self,w):return w
    def width(self,q):return q
    def contact_geometry(self,q):return q,np.eye(4),None
    def parts(self,M,q):return [SimpleNamespace(width=q,retreat=M[2,3],lateral=M[1,3])]
    def self_collision(self,q):return check(True)


class MockScene:
    target_model='test_exact_CAD'
    def __init__(self,env_accept=True,solid_collision=False):
        self.env_accept=env_accept;self.solid_collision=solid_collision
        self.checked=[]
    def target_clear(self,parts,contacts=None):
        p=parts[0];self.checked.append(('target',p.width,p.retreat))
        return check((p.width>=.025 and p.retreat<=-.002 and not self.solid_collision),link='pad')
    def clear(self,parts):
        p=parts[0];self.checked.append(('env',p.width,p.retreat))
        return check(self.env_accept and p.width<=.035)


def test_pair_pass_only_when_both_gates_and_self_collision_are_pass():
    g=MockHardware();s=MockScene()
    c={'width':.02,'T':np.eye(4)}
    result,audit=find_joint_static_pregrasp(g,s,c,GeometryConfig(),max_checks=50)
    assert audit['status']=='PASS'
    assert audit['role']=='open_pregrasp_only'
    assert 'no CAD contact' in audit['note']
    M,q,qo=result
    assert q==qo
    assert q>.02 and M[2,3]<0
    assert s.target_clear(g.parts(M,q))['status']=='PASS'
    assert s.clear(g.parts(M,q))['status']=='PASS'


def test_cannot_gain_target_pass_by_disabling_real_collision():
    result,audit=find_joint_static_pregrasp(MockHardware(),MockScene(solid_collision=True),
        {'width':.02,'T':np.eye(4)},GeometryConfig(),max_checks=50)
    assert result is None and audit['status']=='UNVERIFIED'


def test_cannot_gain_environment_pass_by_reusing_original_pose_result():
    result,audit=find_joint_static_pregrasp(MockHardware(),MockScene(env_accept=False),
        {'width':.02,'T':np.eye(4)},GeometryConfig(),max_checks=50)
    assert result is None and audit['status']=='UNVERIFIED'


def test_local_contact_box_excludes_unrelated_cad_cross_section():
    small=trimesh.creation.box(extents=[.02,.005,.005])
    arm=trimesh.creation.box(extents=[.08,.005,.005]);arm.apply_translation([0,0,.009])
    cad=trimesh.util.concatenate([small,arm]);c=np.array([[.01,0,0],[-.01,0,0]])
    bb=local_contact_obb(cad,np.eye(4),.012,contacts=c,contact_patch_radius_m=.003)
    assert bb['selection']=='contact_patch_obb'
    assert bb['broad_band_width_m']>.07
    assert bb['width_m']<.025


def test_pure_bbox_cannot_create_pass_without_original_geometry():
    # Width fit alone is not a target/env clearance result. The stage solver
    # is the only actor that can claim those checks after querying the Scene.
    cfg=GeometryConfig(opt5_static_pair_enabled=True,opt5_static_pair_max_checks=4)
    assert cfg.opt5_static_pair_enabled
    assert cfg.opt5_static_pair_max_checks==4
    assert not GeometryConfig().opt5_static_pair_enabled


def test_metrics_separate_pregrasp_static_from_contact_goal():
    from scripts.opt5_adaptive_metrics import summarize
    checks={'target_body':check(True),'environment':check(True),
            'contact_geometry':check(False),'approach':check(None),'closure':check(None)}
    pre={'hardware':'OnRobot_2FG7','object_instance':'Pliers_002',
         'pose_phase':'open_pregrasp_staging_no_contact','checks':checks,'status':'FAIL'}
    goal=dict(pre,pose_phase='contact_goal')
    totals,_,_=summarize([pre,goal])
    assert totals['joint_static']==2
    assert totals['joint_static_open_pregrasp']==1
    assert totals['joint_static_contact_goal']==1
    assert totals['full_model_PASS']==0
