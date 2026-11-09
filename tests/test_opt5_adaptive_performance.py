"""Regressions for exact CAD distance, memory reuse and verbose progress.

Uses small synthetic meshes; full OnRobot asset integration is Windows-only.
"""
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from types import SimpleNamespace
from grasp_planner.ur_hardware import Part
from grasp_planner.adaptive_contact import fingertip_anchor_distances
from grasp_planner.geometry_planner import select
from grasp_planner.geometry_config import GeometryConfig
from grasp_planner.models import check


def test_anchor_distances_equivalent_to_transformed_world_mesh(monkeypatch):
    monkeypatch.setattr(trimesh.proximity,'closest_point',trimesh.proximity.closest_point_naive)
    m=trimesh.creation.icosphere(subdivisions=1,radius=0.003)
    m2=trimesh.creation.box(extents=[.006,.004,.008])
    class OEM:
        def __init__(self):
            self.origins=(m,m2)
            self.calls=0
        def parts(self, mount, q, pad_tolerance=0.):
            self.calls+=1
            R=Rotation.from_euler('xyz',[.12,-.22,q]).as_matrix()
            a=np.eye(4);a[:3,:3]=R;a[:3,3]=[.015+q,.008,-.012]
            b=np.eye(4);b[:3,:3]=R;b[:3,3]=[-.014-q,-.006,.009]
            return [Part('right_fingertip_link',m,m,a,True),
                    Part('left_fingertip_link',m2,m2,b,True)]
    g=OEM()
    anchors=np.array([[.016,.011,-.012],[-.015,-.007,.007]])
    for q in [0.,.02,.05,.08]:
        fast=fingertip_anchor_distances(g,np.eye(4),q,anchors)
        parts=g.parts(np.eye(4),q)
        for index,side in enumerate(('right','left')):
            _,dist,_=trimesh.proximity.closest_point_naive(
                parts[index].visual_mesh(),anchors[index:index+1])
            assert abs(fast[side]-dist[0])<1.e-12
    assert g.origins[0] is m and g.origins[1] is m2


def test_select_can_report_long_screen_stage_without_changing_results():
    cfg=GeometryConfig(opt5_enabled=False,candidate_screen_multiplier=2,
      candidates_per_object=10,axial_samples=5,
      approach_precheck_for_joint_pass=False,target_body_clearance_ranking=False)
    class G:
        width_limits=(.01,.09)
        def joint(self,width):return width
        def contact_geometry(self,q):return q,np.eye(4),None
        def parts(self,M,q,pad_tolerance=0.):return [M]
    class S:
        def target_clear(self,p,contacts=None):return check(True)
        def clear(self,p):return check(True,clearance_lower_bound_m=.01)
    pool=[]
    for i in range(12):
        t=np.eye(4);t[0,3]=i*.01
        pool.append(dict(T=t,width=.05,contacts=np.zeros((2,3)),
           fit=np.array([1.,1.]),score=1.-i*.01,bin=i%5,tilt_deg=0.))
    received=[]
    a,aa=select(pool,G(),cfg,S(),progress=received.append)
    b,bb=select(pool,G(),cfg,S())
    assert [x['score'] for x in a]==[x['score'] for x in b]
    assert aa==bb
    assert any('pre-screen start' in s for s in received)
    assert any('pre-screen 12/12' in s for s in received)
