"""Compare OPT4 and OPT5 scientific gate statuses; exclude diagnostic-only paths."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,json

def metrics(rows):
    groups=defaultdict(lambda:Counter())
    totals=Counter()
    for r in rows:
        checks=r['checks'];tt=checks['target_body']['status']=='PASS';ee=checks['environment']['status']=='PASS'
        aa=checks['approach']['status']=='PASS';cc=checks['closure']['status']=='PASS'
        delta={'candidates':1,'target_body':int(tt),'environment':int(ee),'joint_static':int(tt and ee),
               'validated_approach':int(tt and ee and aa),'validated_closure':int(tt and ee and cc),
               'validated_both_trajectories':int(tt and ee and aa and cc),
               'full_pass':int(r['status']=='PASS'),
               'reported_approach_gate':int(aa),'reported_closure_gate':int(cc),
               'opt5_refined_retained':int(bool(r.get('opt5_refinement_deg',0))),
               'diagnostic_approach':int((r.get('diagnostic_checks') or {}).get('approach',{}).get('status')=='PASS')}
        for k,v in delta.items():totals[k]+=v;groups[(r['hardware'],r['object_instance'])][k]+=v
    return totals,groups

def main():
    p=argparse.ArgumentParser();p.add_argument('--opt4',required=True);p.add_argument('--opt5',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    before=json.loads(Path(a.opt4).read_text(encoding='utf-8'))
    after=json.loads(Path(a.opt5).read_text(encoding='utf-8'))
    b,bg=metrics(before);n,ng=metrics(after)
    keys=['candidates','target_body','environment','joint_static','validated_approach','validated_closure','validated_both_trajectories','full_pass','reported_approach_gate','reported_closure_gate','opt5_refined_retained','diagnostic_approach']
    t=['# OPT5 - audit of scientific gates / OPT4 comparison','',
       '> OPT4 and OPT5 use the same experimental ROI background deletion. PASS on these scenes is conditional, not verified physical grasp.','',
       '> Diagnostic approach and closure results for rejected grasps are NOT counted in scientific gates for OPT5.','',
       '| Metric | OPT4 | OPT5 | Delta |','|---|---:|---:|---:|']
    for k in keys:t.append(f'| {k} | {b[k]} | {n[k]} | {n[k]-b[k]:+d} |')
    t+=['','## Per gripper and object','','| Gripper | Target | OPT4 env | OPT5 env | OPT5 static | OPT5 approach valid | OPT5 closure valid | OPT5 full |','|---|---|---:|---:|---:|---:|---:|---:|']
    for key in sorted(set(bg)|set(ng)):
        x=bg[key];y=ng[key]
        t.append('| '+ ' | '.join([*key,str(x['environment']),str(y['environment']),str(y['joint_static']),str(y['validated_approach']),str(y['validated_closure']),str(y['full_pass'])])+' |')
    out=Path(a.output);out.write_text('\n'.join(t)+'\n',encoding='utf-8')
    print('\n'.join(t[:20]));print('Written:',out)

if __name__=='__main__':main()
