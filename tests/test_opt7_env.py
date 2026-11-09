"""OPT7: environment collision needs a minimum number of DIFFERENT occupied voxels (ignores isolated floating points)."""
from types import SimpleNamespace
import numpy as np
import pytest
import trimesh
from grasp_planner.geometry_config import GeometryConfig
from grasp_planner.geometry_collision import Scene
from grasp_planner.ur_hardware import Part

VOX=.003
def link(exact):
    box=trimesh.creation.box(extents=[.02,.02,.02]);hull=box.convex_hull
    raw=box if exact else trimesh.Trimesh(box.vertices,box.faces[1:],process=False)      # watertight -> exact path, open -> hull path
    return Part('base_link',hull,raw,np.eye(4),False)

def scene(points,min_vox=1,**kw):
    tgt=trimesh.creation.box(extents=[.01]*3);tgt.apply_translation([1,1,1])
    obj=SimpleNamespace(id=1,name='t',mesh=tgt);pts=np.asarray(points,float).reshape(-1,3)
    rec=SimpleNamespace(plane=np.array([0,0,1,1.]),points=pts,labels=np.zeros(len(pts),int))
    cfg=GeometryConfig(voxel_m=VOX,environment_min_occupied_voxels=min_vox,**kw)
    return Scene(SimpleNamespace(objects=[obj]),rec,obj,cfg,())

NEAR=lambda k:[0.015,k*0.0035,0.]          # 5 mm outside the face: inside the inflated volume (voxel half-diagonal + 4 mm + margin)

@pytest.mark.parametrize('exact',[False,True])
def test_isolated_point_rejects_by_default_but_is_ignored_with_a_minimum(exact):
    pts=[NEAR(0)]
    r1=scene(pts,1).clear([link(exact)]);assert r1['status']=='FAIL' and r1['cause']=='observed_occupancy'
    assert 'occupied_voxels_in_collision' not in r1                       # default path untouched
    r3=scene(pts,3).clear([link(exact)]);assert r3['status']=='PASS'
    assert r3['ignored_isolated_voxels']==1 and r3['min_occupied_voxels_required']==3
    assert r3['clearance_lower_bound_m']>scene(pts,3).cfg.environment_clearance_m or np.isfinite(r3['clearance_lower_bound_m'])

@pytest.mark.parametrize('exact',[False,True])
def test_a_cluster_of_different_voxels_still_rejects(exact):
    pts=[NEAR(k) for k in range(5)]                                        # 5 different voxels
    assert scene(pts,5).clear([link(exact)])['status']=='FAIL'
    r=scene(pts,3).clear([link(exact)]);assert r['status']=='FAIL' and r['occupied_voxels_in_collision']==5 and r['min_occupied_voxels_required']==3
    assert scene(pts,6).clear([link(exact)])['status']=='PASS'

def test_many_points_in_one_voxel_count_once():
    base=np.array([0.015,0.0005,0.0005]);pts=base+np.random.default_rng(0).uniform(0,.0004,(100,3))   # same 3 mm cell
    assert scene(pts,1).clear([link(False)])['status']=='FAIL'
    r=scene(pts,2).clear([link(False)]);assert r['status']=='PASS' and r['ignored_isolated_voxels']==1

def test_far_points_are_not_counted():
    pts=[NEAR(0)]+[[0.2,k*0.0035,0.] for k in range(40)]
    r=scene(pts,2).clear([link(False)]);assert r['status']=='PASS' and r['ignored_isolated_voxels']==1

def test_hull_and_exact_agree_and_counts_diagnostic_matches():
    pts=[NEAR(k) for k in range(5)]                        # all five inside the inflated volume for both models
    for min_vox in (1,3,5,6,7):
        a=scene(pts,min_vox).clear([link(False)])['status'];b=scene(pts,min_vox).clear([link(True)])['status'];assert a==b
    assert scene(pts,1).occupied_voxel_counts([link(False)])=={'base_link':5}
    assert scene(pts,1).occupied_voxel_counts([link(True)])=={'base_link':5}
    assert scene([],1).occupied_voxel_counts([link(False)])=={}
    # a longer row reaches past the inflated volume: the hull cover (conservative) never counts FEWER than the exact solid
    row=[NEAR(k) for k in range(7)]
    assert scene(row,1).occupied_voxel_counts([link(False)])['base_link']>=scene(row,1).occupied_voxel_counts([link(True)])['base_link']

def test_support_plane_is_not_affected():
    p=link(False);T=np.eye(4);T[2,3]=-1.0                                  # link below the plane z=-1
    obj=scene([NEAR(0)],5);part=Part('base_link',p.local,p.raw,T,False)
    r=obj.clear([part]);assert r['status']=='FAIL' and r['cause']=='support_plane'

def test_dense_cloud_across_chunk_boundary():
    rng=np.random.default_rng(1);pts=np.c_[rng.uniform(.012,.024,4000),rng.uniform(-.03,.03,4000),rng.uniform(-.03,.03,4000)]
    assert scene(pts,50).clear([link(False)])['status']=='FAIL'
    assert scene(pts,500).clear([link(False)])['status']=='PASS'                 # ~270 voxels inside < 500 allowed maximum

def test_defaults_and_bounds():
    assert GeometryConfig().environment_min_occupied_voxels==1
    with pytest.raises(ValueError):GeometryConfig(environment_min_occupied_voxels=0)
    assert GeometryConfig().candidate_height_quantiles==[.2,.4,.6,.78,.9,.98]

# ------------------------------------------------------------------ target shell rule
def shell_scene(points,labels,shell):
    tgt=trimesh.creation.box(extents=[.04,.04,.04])                         # target centred at the origin, surface at +-20 mm
    obj=SimpleNamespace(id=1,name='t',mesh=tgt);pts=np.asarray(points,float).reshape(-1,3)
    rec=SimpleNamespace(plane=np.array([0,0,1,1.]),points=pts,labels=np.asarray(labels,int))
    cfg=GeometryConfig(voxel_m=VOX,environment_ignore_target_shell_m=shell)
    return Scene(SimpleNamespace(objects=[obj]),rec,obj,cfg,())

def test_background_points_in_the_target_shell_are_removed_others_stay():
    pts=[[0.0245,0,0],[0.0300,0,0],[0.05,0,0],[0.0245,0,0]]                  # 4.5, 10, 30 mm from the surface; last one is labelled as another object
    lab=[0,0,0,2]
    off=shell_scene(pts,lab,0.);assert len(off.points)==4 and off.target_shell_removed==0
    on=shell_scene(pts,lab,0.006)
    assert on.target_shell_removed==1 and len(on.points)==3                  # only the 4.5 mm background point
    assert {int(l) for l in on.labels}=={0,2} and (on.labels==2).sum()==1    # other objects' points are never removed
    assert shell_scene(pts,lab,0.012).target_shell_removed==2

def test_shell_changes_the_environment_verdict_only_for_points_near_the_target():
    base=link(False);T=np.eye(4);T[0,3]=0.030                              # 20 mm cube link x in [20,40] mm: touches the target surface at x=20 mm
    p=Part('base_link',base.local,base.raw,T,False)
    near=[[0.0245,0.0,0.0]]                                                 # inside the link, 4.5 mm from the target surface
    assert shell_scene(near,[0],0.).clear([p])['status']=='FAIL'
    assert shell_scene(near,[0],0.006).clear([p])['status']=='PASS'
    far=[[0.0385,0.0,0.0]]                                                  # inside the link, 18.5 mm from the target surface
    assert shell_scene(far,[0],0.006).clear([p])['status']=='FAIL'          # still an obstacle: not part of the target shell
    mixed=near+far
    assert shell_scene(mixed,[0,0],0.006).clear([p])['status']=='FAIL'      # one real obstacle remains

def test_shell_default_off_and_bounded():
    assert GeometryConfig().environment_ignore_target_shell_m==0.
    with pytest.raises(ValueError):GeometryConfig(environment_ignore_target_shell_m=.05)

# ------------------------------------------------------------------ table band rule
def band_scene(points,labels,band,plane=(0,0,1,0.)):
    tgt=trimesh.creation.box(extents=[.01]*3);tgt.apply_translation([1,1,1])
    obj=SimpleNamespace(id=1,name='t',mesh=tgt);pts=np.asarray(points,float).reshape(-1,3)
    rec=SimpleNamespace(plane=np.array(plane,float),points=pts,labels=np.asarray(labels,int))
    return Scene(SimpleNamespace(objects=[obj]),rec,obj,GeometryConfig(voxel_m=VOX,environment_ignore_table_band_m=band),())

def test_table_band_removes_only_background_points_near_the_plane():
    pts=[[0,0,0.004],[0,0,-0.003],[0,0,0.012],[0,0,0.004]];lab=[0,0,0,2]   # heights 4, -3, 12 mm; last is another object's point
    s=band_scene(pts,lab,0.008)
    assert s.table_band_removed==2 and len(s.points)==2 and sorted(s.labels.tolist())==[0,2]
    assert band_scene(pts,lab,0.).table_band_removed==0
    assert band_scene(pts,lab,0.015).table_band_removed==3                 # 12 mm point now inside the band too

def test_table_band_lets_a_link_resting_on_table_noise_pass_but_not_a_real_obstacle():
    base=link(False);T=np.eye(4);T[2,3]=0.0105                              # 20 mm cube link whose bottom is 0.5 mm above the plane z=0 (margin 2 mm -> support check)
    p=Part('base_link',base.local,base.raw,T,False)
    noise=[[0.0,0.0,0.003]]                                                 # table noise point 3 mm above the plane, under the link
    assert band_scene(noise,[0],0.).clear([p])['status'] in ('FAIL',)       # support plane or occupancy: the 0.5 mm clearance is below the 2 mm margin
    T2=np.eye(4);T2[2,3]=0.0125;p2=Part('base_link',base.local,base.raw,T2,False)   # bottom 2.5 mm above the plane: support check passes
    assert band_scene(noise,[0],0.).clear([p2])['status']=='FAIL'           # the noise point blocks it (inflated volume)
    assert band_scene(noise,[0],0.008).clear([p2])['status']=='PASS'        # noise band ignored
    real=[[0.0,0.0,0.020]]                                                  # 20 mm above the plane, inside the link volume: a real obstacle (outside the noise band)
    assert band_scene(real,[0],0.008).clear([p2])['status']=='FAIL'

def test_table_band_default_off_and_bounded():
    assert GeometryConfig().environment_ignore_table_band_m==0.
    with pytest.raises(ValueError):GeometryConfig(environment_ignore_table_band_m=.05)
