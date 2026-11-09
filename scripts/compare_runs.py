"""Compare two complete grasp experiments (aggregate metrics, not candidate IDs)."""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path

GATES=('data_alignment','hardware_geometry','kinematics','contact_geometry','self_collision','target_body','environment','approach','closure','filter_policy')

def summarize(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[(r['hardware'],r['object_instance'])].append(r)
    def c(rs):
        return dict(candidates=len(rs),full_pass=sum(r['status']=='PASS' for r in rs),environment_pass=sum(r.get('checks',{}).get('environment',{}).get('status')=='PASS' for r in rs),target_body_pass=sum(r.get('checks',{}).get('target_body',{}).get('status')=='PASS' for r in rs),joint_static_pass=sum(r.get('checks',{}).get('environment',{}).get('status')=='PASS' and r.get('checks',{}).get('target_body',{}).get('status')=='PASS' for r in rs),approach_pass=sum(r.get('checks',{}).get('approach',{}).get('status')=='PASS' for r in rs),closure_pass=sum(r.get('checks',{}).get('closure',{}).get('status')=='PASS' for r in rs))
    return c(rows),{k:c(v) for k,v in groups.items()}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--optimized',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    baseline=json.loads(a.baseline.read_text(encoding='utf8'))
    optimized=json.loads(a.optimized.read_text(encoding='utf8'))
    b,bs=summarize(baseline);o,os=summarize(optimized)
    keys=('candidates','environment_pass','target_body_pass','joint_static_pass','approach_pass','closure_pass','full_pass')
    lines=['# Grasp Planner – baseline vs OPT1','',f"- Baseline: `{a.baseline}`",f"- OPT1: `{a.optimized}`",'','| Metric | Baseline | OPT1 | Change |','|---|---:|---:|---:|']
    for k in keys:lines.append(f'| {k} | {b[k]} | {o[k]} | {o[k]-b[k]:+d} |')
    lines+=['','## Per gripper / target','', '| Gripper | Object | Candidates base/opt | Environment PASS base/opt | Joint static PASS base/opt | Full PASS base/opt |','|---|---|---:|---:|---:|---:|']
    for k in sorted(set(bs)|set(os)):
        old=bs.get(k,dict.fromkeys(keys,0));new=os.get(k,dict.fromkeys(keys,0))
        lines.append(f'| {k[0]} | {k[1]} | {old["candidates"]}/{new["candidates"]} | {old["environment_pass"]}/{new["environment_pass"]} | {old["joint_static_pass"]}/{new["joint_static_pass"]} | {old["full_pass"]}/{new["full_pass"]} |')
    lines+=['','## Methodological limits','',
        '- Candidate IDs contain ranking indices, which change between runs: **do not** compare IDs as matched physical grasp poses.',
        '- These are model-relative geometric validation counts, not evidence of real robot execution or dynamics.',
        '- A higher PASS count alone cannot certify false-positive rate; inspect collision witnesses and replay images.',
        '- Both experiments must use the same capture, gripper CAD, calibration, units, and validation gate definitions.',
        '- If target-model provenance differs (exact watertight vs convex fallback), explicitly stratify by model.', '']
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text('\n'.join(lines),encoding='utf8')
    print('\n'.join(lines[:12]))
if __name__=='__main__':main()
