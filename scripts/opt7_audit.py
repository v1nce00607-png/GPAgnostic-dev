"""OPT7 post-hoc audit of any all_candidates.json (no re-planning, read-only).

Reports: gate counts, environment penetration/clearance, target penetration depth
(when the run stored it), 180-degree twin consistency, TCP/contact-midpoint offset
along the jaw axis, and which side the first failing link is on.
Writes OPT7_audit.md and OPT7_audit.json into --output. Never edits the input.
"""
import argparse,json,collections as C
from pathlib import Path
import numpy as np

def st(c,k):return c['checks'][k]['status']

def twins(cands,pos_tol=1.5e-3):
    T=[np.array(c['world_from_tcp']) for c in cands];by=C.defaultdict(list)
    for i,c in enumerate(cands):by[(c['hardware'],c['object_instance'])].append(i)
    out={}
    for ids in by.values():
        for a in ids:
            for b in ids:
                if a!=b and np.linalg.norm(T[a][:3,3]-T[b][:3,3])<pos_tol and T[a][:3,2]@T[b][:3,2]>.999 and T[a][:3,0]@T[b][:3,0]<-.999:
                    out[a]=b;break
    return out

def audit(cands):
    r={'candidates':len(cands)}
    r['gates']={k:dict(C.Counter(st(c,k) for c in cands)) for k in cands[0]['checks']}
    both=C.Counter((st(c,'target_body'),st(c,'environment')) for c in cands)
    r['target_environment_joint']={f'{a}/{b}':n for (a,b),n in both.items()}
    # environment: signed clearance (negative = penetration into observed occupancy)
    pen=[-c['checks']['environment']['clearance_lower_bound_m']*1e3 for c in cands
         if st(c,'environment')=='FAIL' and c['checks']['environment'].get('clearance_lower_bound_m') is not None]
    r['environment_fail']=dict(n=len(pen),true_penetration_gt0mm=int(sum(p>0 for p in pen)),
        only_inside_clearance_margin=int(sum(p<=0 for p in pen)),
        penetration_mm_pct_10_50_90_100=[round(float(x),3) for x in np.percentile(pen,[10,50,90,100])] if pen else None)
    td=[c['checks']['target_body']['penetration_depth_m']*1e3 for c in cands
        if st(c,'target_body')=='FAIL' and c['checks']['target_body'].get('penetration_depth_m') is not None]
    r['target_body_penetration_mm']=(dict(n=len(td),pct_10_50_90_100=[round(float(x),3) for x in np.percentile(td,[10,50,90,100])],
        under_1mm=int(sum(x<1 for x in td)),under_2mm=int(sum(x<2 for x in td)))
        if td else 'NOT STORED in this run (needs OPT7 geometry_collision.witness_summary)')
    # near misses for environment margin
    nm=[c for c in cands if st(c,'target_body')=='PASS' and st(c,'environment')=='FAIL'
        and c['checks']['environment'].get('clearance_lower_bound_m') is not None]
    r['env_margin_sensitivity_target_pass_only']={f'{t*1e3:.1f}mm':int(sum(c['checks']['environment']['clearance_lower_bound_m']>=t for c in nm)) for t in (.002,.0015,.001,.0005,0.)}
    # first failing link side
    side=C.Counter()
    for c in cands:
        t=c['checks']['target_body']
        if t['status']=='FAIL' and t.get('link'):side[t['link'].split('_')[0] if t['link'].split('_')[0] in('right','left') else 'base']+=1
    r['target_first_failing_link_side']=dict(side)
    # 180-degree twins
    tw=twins(cands);pairs=[(a,b) for a,b in tw.items() if a<b]
    same_t=sum(st(cands[a],'target_body')==st(cands[b],'target_body') for a,b in pairs)
    same_e=sum(st(cands[a],'environment')==st(cands[b],'environment') for a,b in pairs)
    swap=C.Counter()
    for a,b in pairs:
        ta,tb=cands[a]['checks']['target_body'],cands[b]['checks']['target_body']
        if ta['status']==tb['status']=='FAIL' and ta.get('link') and tb.get('link'):
            sa,sb=ta['link'].split('_')[0],tb['link'].split('_')[0];swap['same_side_named' if sa==sb else 'swapped_side']+=1
    r['twin_180']=dict(candidates_with_twin=len(tw),pairs=len(pairs),identical_target_status=int(same_t),
        identical_environment_status=int(same_e),failing_link_side_in_pairs=dict(swap),
        flip_rescues_target=int(sum(st(cands[a],'target_body')=='FAIL' and st(cands[b],'target_body')=='PASS' for a,b in tw.items())))
    # contact midpoint vs TCP on the jaw axis
    off=[]
    for c in cands:
        T=np.array(c['world_from_tcp']);P=np.array(c['contacts_world_m'])
        mid=(np.linalg.inv(T)@np.r_[P.mean(0),1.])[:3];off.append(float(abs(mid[0])*1e3))
    r['tcp_to_contact_midpoint_jaw_axis_mm']=dict(max=round(max(off),4),median=round(float(np.median(off)),4))
    # OPT7 in-planner diagnostics, if present
    dg=[c['opt7_diagnostics'] for c in cands if c.get('opt7_diagnostics')]
    if dg:
        fl=[d['flip180'] for d in dg]
        both_side=sum(1 for d in fl if len(d['original']['target_colliding_links']['sides'])==2)
        r['opt7_diagnostics']=dict(candidates=len(dg),
            flip_same_target_status=int(sum(d['same_target_status'] for d in fl)),
            flip_same_environment_status=int(sum(d['same_environment_status'] for d in fl)),
            target_fail_with_both_sides_colliding=both_side,
            jaw_shift_any_open_both_pass=int(sum(d['jaw_axis_open_shift_scan']['any_both_pass'] for d in dg)))
    return r

def markdown(r,source):
    L=[f'# OPT7 post-hoc audit',f'Source: `{source}` — read-only analysis; no PASS is created or changed.','']
    L+=['## Gate counts','|gate|PASS|FAIL|UNVERIFIED|','|---|--:|--:|--:|']
    for k,v in r['gates'].items():L.append(f"|{k}|{v.get('PASS',0)}|{v.get('FAIL',0)}|{v.get('UNVERIFIED',0)}|")
    L+=['','## target_body x environment (joint)']+[f'- {k}: {n}' for k,n in sorted(r['target_environment_joint'].items())]
    e=r['environment_fail'];L+=['',f"## Environment failures ({e['n']})",f"- real penetration >0 mm: {e['true_penetration_gt0mm']}; only inside the clearance margin: {e['only_inside_clearance_margin']}",f"- penetration mm p10/p50/p90/max: {e['penetration_mm_pct_10_50_90_100']}"]
    L+=['','## Target-body penetration depth',f"- {r['target_body_penetration_mm']}"]
    L+=['','## Environment margin sensitivity (target PASS only; rescued = clearance >= threshold)']+[f'- {k}: {n}' for k,n in r['env_margin_sensitivity_target_pass_only'].items()]
    L+=['','## First failing link side (target_body)',f"- {r['target_first_failing_link_side']}"]
    t=r['twin_180'];L+=['','## 180-degree twins',f"- candidates with exact twin: {t['candidates_with_twin']} ({t['pairs']} pairs)",f"- identical target status in pairs: {t['identical_target_status']}/{t['pairs']}; environment: {t['identical_environment_status']}/{t['pairs']}",f"- failing link side in pairs: {t['failing_link_side_in_pairs']}",f"- flip would rescue a target FAIL: {t['flip_rescues_target']}"]
    m=r['tcp_to_contact_midpoint_jaw_axis_mm'];L+=['','## TCP vs contact midpoint on the jaw axis',f"- max {m['max']} mm, median {m['median']} mm (0 = symmetric closing by construction)"]
    if 'opt7_diagnostics' in r:L+=['','## OPT7 in-planner diagnostics',f"- {r['opt7_diagnostics']}"]
    return '\n'.join(L)+'\n'

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidates',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();cands=json.loads(a.candidates.read_text(encoding='utf-8'))
    r=audit(cands);a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'OPT7_audit.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    (a.output/'OPT7_audit.md').write_text(markdown(r,a.candidates),encoding='utf-8');print(markdown(r,a.candidates))
if __name__=='__main__':main()
