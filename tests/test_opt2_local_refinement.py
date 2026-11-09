import sys
from pathlib import Path
import numpy as np
from grasp_planner.models import check
from grasp_planner.geometry_planner import GATES
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from opt2_local_refinement import offsets, contact_key, final_gate_status

def test_offsets_bounded_symmetric_and_include_zero():
    x=offsets(.25,.5)
    assert len(x)==33
    assert any(np.allclose(v,0) for v in x)
    assert all(np.linalg.norm(v)<=.000500001 for v in x)
    assert all(any(np.allclose(-v,y) for y in x) for v in x)

def test_swapped_contacts_are_same_contact_group():
    a=dict(hardware='RG2',object_instance='Pliers_001',contacts_world_m=[[0,0,0],[.003,0,0]])
    b=dict(a,contacts_world_m=list(reversed(a['contacts_world_m'])))
    assert contact_key(a)==contact_key(b)

def test_missing_or_failed_gate_never_full_pass():
    checks={key: check(True) for key in GATES}
    assert final_gate_status(checks)=='PASS'
    del checks['target_body']
    assert final_gate_status(checks)=='UNVERIFIED'
    checks['target_body']=check(False)
    assert final_gate_status(checks)=='FAIL'
