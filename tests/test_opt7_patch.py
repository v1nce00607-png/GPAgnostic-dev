"""OPT7: shallow pad contact allowed over the WHOLE pad face (inner-face rectangle + margin)."""
from types import SimpleNamespace
import numpy as np
import pytest
import trimesh
from grasp_planner.geometry_config import GeometryConfig
from grasp_planner.geometry_collision import Scene
from grasp_planner.ur_hardware import CADGripper,HardwareSpec

@pytest.mark.parametrize('model,name,hy,hz',[('rg2','OnRobot_RG2',.0101,.0149),('rg6','OnRobot_RG6',.0125,.0185),('2fg7','OnRobot_2FG7',.0147,.0177)])
def test_pad_footprint_is_measured_from_the_oem_inner_face(model,name,hy,hz):
    g=CADGripper(HardwareSpec(model=model,name=name));fp=g.pad_footprints()
    assert len(fp)==2
    for f in fp.values():
        assert abs(f['hy']-hy)<.0015 and abs(f['hz']-hz)<.0025
        assert abs(np.linalg.norm(f['ey'])-1)<1e-9 and abs(f['ey']@f['ez'])<1e-6

def test_on_pad_face_follows_the_pose_and_the_margin():
    g=CADGripper(HardwareSpec(model='rg2',name='OnRobot_RG2'));q=g.joint(.05)
    M=np.eye(4);M[:3,3]=[.3,-.2,.5]
    for p in g.parts(M,q):
        if not p.pad:continue
        f=p.footprint;centre=p.T@np.r_[f['ey']*f['u0']+f['ez']*f['v0'],1.]       # a point of the face plane (normal offset is irrelevant)
        assert p.on_pad_face(centre[:3],0.)
        inside=centre[:3]+p.T[:3,:3]@(f['ey']*(f['hy']-.0002))
        edge=centre[:3]+p.T[:3,:3]@(f['ey']*(f['hy']+.0003))
        far=centre[:3]+p.T[:3,:3]@(f['ez']*(f['hz']+.003))
        assert p.on_pad_face(inside,0.) and not p.on_pad_face(edge,0.) and p.on_pad_face(edge,.0005) and not p.on_pad_face(far,.0005)

def test_fixed_mode_is_the_default():
    assert GeometryConfig().target_contact_patch_mode=='fixed'

def run(mode,lateral,extra,height=.008):
    g=CADGripper(HardwareSpec(model='rg2',name='OnRobot_RG2'));q=g.joint(float(np.mean(g.width_limits))*.8);w,tcp,_=g.contact_geometry(q)
    cfg=GeometryConfig(target_contact_policy='local_pad_witness',target_contact_patch_mode=mode)
    ctr=tcp[:3,3]+[0,0,-height/2];ext=[w+extra,lateral,height]
    box=trimesh.creation.box(extents=ext);box.apply_translation(ctr);obj=SimpleNamespace(id=1,name='b',mesh=box)
    rec=SimpleNamespace(plane=np.array([0,0,1,.02]),points=np.zeros((1,3))+10,labels=np.zeros(1,int))
    sc=Scene(SimpleNamespace(objects=[obj]),rec,obj,cfg,());anchors=np.array([ctr+[ext[0]/2,0,0],ctr-[ext[0]/2,0,0]])
    return sc.target_clear(g.parts(np.eye(4),q,.0005),contacts=anchors)

def test_shallow_overlap_over_the_pad_face_passes_only_in_rect_mode():
    # 0.2 mm overlap over +-8 mm laterally: beyond the 2.5 mm anchor disc but on the pad face.
    assert run('fixed',.016,.0014)['status']=='FAIL'
    assert run('pad_face_rect',.016,.0014)['status']=='PASS'
    assert run('pad_face_rect',.016,.0014,.025)['status']=='PASS'      # also along the finger length

def test_depth_rule_is_unchanged():
    assert run('pad_face_rect',.016,.004,.025)['status']=='FAIL'


def test_swept_closure_part_keeps_the_pad_face_rule():
    from grasp_planner.geometry_collision import swept
    g=CADGripper(HardwareSpec(model='rg2',name='OnRobot_RG2'));qa,qb=g.joint(.06),g.joint(.04)
    for a,b in zip(g.parts(np.eye(4),qa),g.parts(np.eye(4),qb)):
        if not a.pad:continue
        sw=swept(a,b,.0003);f=b.footprint;c=(b.T@np.r_[f['ey']*f['u0']+f['ez']*f['v0'],1.])[:3]
        assert sw.on_pad_face(c,.0005)
        assert not sw.on_pad_face(c+b.T[:3,:3]@(f['ez']*(f['hz']+.005)),.0005)
