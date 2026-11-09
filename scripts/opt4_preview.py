"""Quick, read-only preview of OPT4 object-space ROI on saved OPT3 points.

The witness count is a SINGLE-FIRST-WITNESS diagnostic, NOT a rerun of FCL or
proof that the candidate becomes collision free after suppression.
"""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np,yaml
from scipy.spatial import cKDTree
from grasp_planner.models import Capture
from grasp_planner.geometry_config import GeometryConfig
from grasp_planner.cleaning import halo_roi_candidates,halo_delete_candidates

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--opt3-results',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    cfg=GeometryConfig.model_validate(yaml.safe_load(a.config.read_text(encoding='utf-8')))
    cap=Capture(a.dataset)
    with np.load(a.opt3_results/'reconstruction.npz') as z:
        p=z['points'];l=z['labels'];plane=z['plane']
    roi,owners,protected=halo_roi_candidates(p,l,plane,cap.objects,cfg)
    # OPT3 reconstruction has already been filtered. SOR/radius/DBSCAN are
    # unavailable in that indexed array, so preview experimental ROI exactly;
    # isolated_only must wait for actual OPT4 filtering evidence arrays.
    points_removed=roi if cfg.halo_policy=='experimental_roi' else np.zeros(len(p),bool)
    counts={o.name:int(np.sum(points_removed&(owners==i))) for i,o in enumerate(cap.objects)}
    rows=json.loads((a.opt3_results/'all_candidates.json').read_text(encoding='utf-8'))
    suspects=[]
    for r in rows:
        env=r['checks']['environment']
        if env.get('status')=='FAIL' and env.get('cause')=='observed_occupancy' and env.get('witness_label')==0:
            suspects.append((r,env['witness_world_m']))
    matches=Counter();total_by=Counter();witnesses=[]
    if suspects:
        tree=cKDTree(p)
        d,ix=tree.query(np.asarray([w for _,w in suspects]),workers=1)
        for (row,_),di,pi in zip(suspects,d,ix):
            key=row['object_instance'];total_by[key]+=1
            if di<=1e-6 and points_removed[pi]:
                matches[key]+=1
                witnesses.append(dict(candidate=row['id'],target=key,witness_position=row['checks']['environment']['witness_world_m']))
    report={
        'policy':cfg.halo_policy,
        'note':'PREDICTION only: snapshot of previously saved OPT3 reconstruction; no new collision tests were run',
        'points_opt3':int(len(p)),
        'experimental_roi_removed_if_enabled':int(np.sum(roi)),
        'by_object':counts,
        'first_environment_fail_witnesses_within_roi':dict(matches),
        'first_environment_fail_witnesses_total':dict(total_by),
        'warning':'Deleting an initial collision witness does not establish a PASS: further collision witnesses may exist.',
        'sample_witnesses':witnesses[:20],
    }
    a.output.parent.mkdir(exist_ok=True,parents=True)
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('OPT4 PREVIEW (NOT VALIDATION)',flush=True)
    print('Halo candidate voxels:',report['experimental_roi_removed_if_enabled'],flush=True)
    print('By object:',counts,flush=True)
    print('Affected initial rejection witnesses:',dict(matches),flush=True)
    print('Saved:',a.output,flush=True)

if __name__=='__main__':main()
