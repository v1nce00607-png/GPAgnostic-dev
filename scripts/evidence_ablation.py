"""Read-only audit of recorded occupancy witnesses against multiview evidence.

This is NOT a collision resimulation and can never upgrade a grasp to PASS.
It asks which stored witnesses merit deeper image/CAD inspection.
"""
import argparse,csv,json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True,help='Existing all_candidates.json')
    p.add_argument('--reconstruction',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=json.loads(a.results.read_text(encoding='utf-8'))
    audit=json.loads(a.audit.read_text(encoding='utf-8'))
    objects=list(audit['relabelled_by_object'])
    object_col={name:i for i,name in enumerate(objects)}
    with np.load(a.reconstruction) as z:
        points=z['points'].copy();labels=z['labels'].copy()
    with np.load(a.evidence) as z:
        raw=z['raw_points'].copy();keep=z['keep'].copy()
        subset=z['indices'].copy();support=z['support'].copy()
        free=z['free'].copy();occupied=z['occupied'].copy()
    kept=np.flatnonzero(keep)
    if len(kept)!=len(points) or not np.allclose(raw[kept],points,atol=1e-9):
        raise ValueError('Evidence and reconstruction do not describe the same filtered point cloud')
    mapping=np.full(len(keep),-1,dtype=np.int64)
    mapping[subset]=np.arange(len(subset))
    tree=cKDTree(points)
    counts=Counter();records=[]
    for r in rows:
        env=r.get('checks',{}).get('environment',{})
        if env.get('cause')!='observed_occupancy':continue
        counts['occupancy_failures']+=1
        point=env.get('witness_world_m')
        if point is None:counts['missing_point']+=1;continue
        distance,filtered_index=tree.query(point)
        if distance>1e-7:
            counts['unmatched_witness']+=1;continue
        filtered_index=int(filtered_index)
        raw_index=int(kept[filtered_index]);j=int(mapping[raw_index])
        lab=int(labels[filtered_index]);counts['matched']+=1
        name=r.get('object_instance','')
        data=dict(candidate_id=r.get('id'),object_instance=name,hardware=r.get('hardware'),witness_label=lab,witness_raw_index=raw_index,observed_distance_m=distance,own_votes='',competitor_votes='',occupied_views='',free_views='',dominant_evidence=False,near_surface_unverified=True)
        if lab==0:counts['background_witness']+=1
        if j>=0 and name in object_col:
            col=object_col[name]
            own=int(support[j,col]);competitor=int(max(np.delete(support[j],col))) if support.shape[1]>1 else 0
            occ=int(occupied[j]);free_views=int(free[j])
            data.update(own_votes=own,competitor_votes=competitor,occupied_views=occ,free_views=free_views)
            strong=lab==0 and own>=3 and own/max(occ,1)>=.6 and competitor>0 and competitor<=.1*own and own>=5*competitor and not (free_views>=5 and free_views>3*occ)
            data['dominant_evidence']=strong
            if strong:counts['strong_mask_conflict_witness_occurrences']+=1
        records.append(data)
    with (a.output/'witness_audit.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['candidate_id','object_instance','hardware','witness_label','witness_raw_index','observed_distance_m','own_votes','competitor_votes','occupied_views','free_views','dominant_evidence','near_surface_unverified']);w.writeheader();w.writerows(records)
    unique={r['witness_raw_index'] for r in records if r['dominant_evidence']}
    unique_candidates={r['candidate_id'] for r in records if r['dominant_evidence']}
    lines=[
        '# Grasp Planner – semantikai ellenpróba (csak olvasás)',
        '',
        f"Vizsgált jelöltek: **{len(rows)}**",
        f"Rögzített `observed_occupancy` elutasítások: **{counts['occupancy_failures']}**",
        f"Illesztett ütközési tanúpontok: **{counts['matched']}**",
        f"0 címkéjű tanúk: **{counts['background_witness']}**",
        f"Erős saját maszkdominanciát és gyenge konkurenciát jelző tanúk előfordulásai: **{counts['strong_mask_conflict_witness_occurrences']}**",
        f"Különálló ilyen tanúpontok: **{len(unique)}**",
        f"Érintett jelöltek: **{len(unique_candidates)}**",
        '',
        '**Értelmezés:** Ezek diagnosztikai jelölések, nem elfogadott megfogások. '
        'A tanúpont a legelső rögzített ütközési okot mutatja. '
        'Az eredeti CAD-től való távolság / teljes ütközésvizsgálat itt nem számolható újra. '
        'A tanú újracímkézése önmagában nem jelenti, hogy egy jelölt Environment vagy Full PASS lesz.',
        '', 'CSV: `witness_audit.csv`.',''
    ]
    (a.output/'semantic_counterfactual.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))

if __name__=='__main__':main()
