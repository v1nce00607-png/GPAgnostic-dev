from pathlib import Path
import json,numpy as np,pytest
from grasp_planner.models import verdict,valid_transform
from grasp_planner.geometry_planner import GATES
ROOT=Path(__file__).resolve().parents[1]/'results'
@pytest.fixture(scope='module')
def rows():
 if not (ROOT/'all_candidates.json').exists():pytest.skip('No shipped results')
 return json.loads((ROOT/'all_candidates.json').read_text())
def test_real_capture_counts(rows):
 s=json.loads((ROOT/'summary.json').read_text());assert s['frames_processed']==318;assert s['candidates']==len(rows)==2400;assert len({r['hardware'] for r in rows})==3
 assert s['status_counts']=={v:sum(r['status']==v for r in rows) for v in ['PASS','FAIL','UNVERIFIED']}
def test_no_status_promotion(rows):
 for r in rows:assert r['status']==verdict(r['checks'],GATES) and not r['executable'] and not r['dynamics_evaluated'] and r['robot_reachability']=='UNVERIFIED'
def test_rigid_transforms(rows):
 assert all(valid_transform(r['camera_from_tcp']) and valid_transform(r['world_from_mount']) for r in rows)
def test_top10_keeps_real_status(rows):
 top=json.loads((ROOT/'top10_grasps.json').read_text());lookup={r['id']:r for r in rows};assert len(top)==len({r['id'] for r in top})==10;assert all(r==lookup[r['id']] for r in top)
def test_filter_accounting():
 z=np.load(ROOT/'filter_evidence.npz');r=np.load(ROOT/'reconstruction.npz');a=json.loads((ROOT/'filter_audit.json').read_text());assert len(z['raw_points'])==a['raw_points'];assert z['keep'].sum()==len(r['points'])==a['retained_points'];assert len(z['keep'])-z['keep'].sum()==a['removed_points'];assert np.allclose(r['points'],z['raw_points'][z['keep']]);assert np.all(z['keep'][z['raw_labels']>0]);assert a['retained_background_inside_bbox']>0
 assert a['bbox_policy']=='evidence'
def test_real_trajectory_limits(rows):
 aa=[r for r in rows if r['trajectory']];assert len(aa)>0
 for r in aa:
  a=r['trajectory'];ss=a['samples'];assert np.all(np.diff([s['time_s'] for s in ss])>0);assert a['status']==r['checks']['approach']['status'];assert not a['executable'];assert max(np.linalg.norm(s['velocity_camera_m_s']) for s in ss)<=.080001;assert max(np.linalg.norm(s['acceleration_camera_m_s2']) for s in ss)<=.300001
  assert all(valid_transform(s['camera_from_mount']) and valid_transform(s['camera_from_tcp']) for s in r['closure_trajectory']['samples'])
def test_controls_separate():
 r=json.loads((ROOT/'controls/summary.json').read_text());assert len(r)==6;assert all(x['expected']==x['status'] and x['synthetic'] for x in r)
def test_input_provenance():
 r=json.loads((ROOT/'run_provenance.json').read_text())['source_integrity'];assert r['all_available_verified'];assert r['verified_files_this_resume']==976;assert r['full_source_verification_previous_run']['verified_files']==1613
