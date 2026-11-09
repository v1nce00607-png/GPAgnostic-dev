"""Generate transparent OPT1-vs-OPT3 metrics and contact-model provenance.

Does not alter results. All counts come from saved JSON; no inferred PASS.
"""
import argparse
from pathlib import Path
import json
from collections import Counter

GATES=('target_body','environment','approach','closure')

def load(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data,list):raise ValueError(f'Candidate file must be a list: {path}')
    return data

def metric(rows):
    ans={x:sum(r.get('checks',{}).get(x,{}).get('status')=='PASS' for r in rows) for x in GATES}
    ans['candidates']=len(rows)
    ans['full_pass']=sum(r.get('status')=='PASS' for r in rows)
    ans['joint_static_pass']=sum(all(r.get('checks',{}).get(x,{}).get('status')=='PASS' for x in ('target_body','environment')) for r in rows)
    ans['accepted_contact_rows']=sum(bool(r.get('checks',{}).get('target_body',{}).get('allowed_local_fingertip_contacts')) for r in rows)
    ans['accepted_open_CAD_contact_rows']=sum(r.get('checks',{}).get('target_body',{}).get('local_contact_assumption')=='open_CAD_surface_not_solid_ground_truth' for r in rows)
    return ans

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--opt1',type=Path,required=True)
    p.add_argument('--opt3',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    original=load(args.opt1);current=load(args.opt3)
    m1=metric(original);m3=metric(current)
    text=['# OPT3 verification — observed data and original CAD', '',
          f'- OPT1: `{args.opt1}`',f'- OPT3: `{args.opt3}`','',
          '| Metric | OPT1 | OPT3 | Difference |','|---|---:|---:|---:|']
    for k in ('candidates','target_body','environment','joint_static_pass','approach','closure','full_pass','accepted_contact_rows','accepted_open_CAD_contact_rows'):
        text.append(f'| {k} | {m1[k]} | {m3[k]} | {m3[k]-m1[k]:+d} |')
    f=args.opt3.parent/'filter_audit.json'
    if f.exists():
        audit=json.loads(f.read_text(encoding='utf-8'))
        text+=['','## Floating-point filter audit', '']
        for k in ('floating_policy','floating_probe_candidates','floating_morphology_candidates',
                  'floating_extra_evidence_points','floating_depth_confirmed_removed',
                  'retained_points','removed_points','relabelled_points'):
            text.append(f'- {k}: {audit.get(k,"n/a")}')
    text+=['','## Per gripper and object','', '| Gripper | Object | OPT1 env | OPT3 env | OPT3 joint | OPT3 full |', '|---|---|---:|---:|---:|---:|']
    key=lambda r:(r.get('hardware','?'),r.get('object_instance','?'))
    groups=sorted(set(map(key,original))|set(map(key,current)))
    for g in groups:
        a=[r for r in original if key(r)==g];b=[r for r in current if key(r)==g]
        text.append(f'| {g[0]} | {g[1]} | {metric(a)["environment"]} | {metric(b)["environment"]} | {metric(b)["joint_static_pass"]} | {metric(b)["full_pass"]} |')
    text+=['','## Scientific boundaries','',
        '- The gripper and object CAD are evaluated in the annotated observed scene, not on a real robot.',
        '- Local fingertip-witness acceptance on non-watertight CAD is a **model-dependent assumption**, not ground-truth solid certification.',
        '- Only the original 10 validation checks can contribute to a Full PASS.',
        '- Candidate IDs contain ranking indices and do not identify matched physical poses across runs.',
        '- Floating-point removal requires repeated camera depth-ray contradictions; individual removed samples are saved in filter_evidence.npz.',
        '- Robot-arm inverse kinematics, dynamics and unseen environment are not evaluated.','']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(text),encoding='utf-8')
    print(f'OPT3 metrics written: {args.output}',flush=True)

if __name__=='__main__':main()
