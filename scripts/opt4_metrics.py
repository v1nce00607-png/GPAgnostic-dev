"""Report OPT3/OPT4 PASS values with prominent experimental-scene caveat."""
import argparse,json
from collections import Counter
from pathlib import Path

def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def metric(rows):
    counts={k:sum(r.get('checks',{}).get(k,{}).get('status')=='PASS' for r in rows) for k in ('target_body','environment','approach','closure')}
    counts['candidates']=len(rows)
    counts['joint_static_pass']=sum(r['checks']['target_body']['status']=='PASS' and r['checks']['environment']['status']=='PASS' for r in rows)
    counts['full_pass']=sum(r.get('status')=='PASS' for r in rows)
    return counts

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--opt3',type=Path,required=True);ap.add_argument('--opt4',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a3=load(a.opt3);a4=load(a.opt4);m3=metric(a3);m4=metric(a4)
    audit=load(a.opt4.parent/'filter_audit.json')
    lines=['# OPT4 – experimentális ROI-tisztítás / OPT3 összehasonlítás','',
           '> **FONTOS:** Az OPT4 aggressive ROI módja feltételezett háttérpontokat töröl. A további PASS-ok a megváltoztatott jelenetmodellre vonatkoznak, NEM bizonyítják az eredeti megfigyelt jelenet ütközésmentességét.','',
           f'- OPT3: `{a.opt3}`',f'- OPT4: `{a.opt4}`','',
           '| Mutató | OPT3 | OPT4 | Különbség |','|---|---:|---:|---:|']
    for k in ('candidates','target_body','environment','joint_static_pass','approach','closure','full_pass'):
        lines.append(f'| {k} | {m3[k]} | {m4[k]} | {m4[k]-m3[k]:+d} |')
    lines+=['','## Törlés audit','']
    for k in ['halo_policy','halo_roi_candidates','halo_extra_removed','halo_deleted_by_object','floating_depth_confirmed_removed','retained_points','removed_points','relabelled_points']:
        lines.append(f'- **{k}**: {audit.get(k)}')
    lines+=['','## Objektumok és megfogók','', '| Megfogó | Objektum | OPT3 env | OPT4 env | OPT4 közös statikus | OPT4 teljes |','|---|---|---:|---:|---:|---:|']
    groups=sorted(set((r['hardware'],r['object_instance']) for r in a3+a4))
    for hardware,object_name in groups:
        x=[r for r in a3 if r['hardware']==hardware and r['object_instance']==object_name]
        y=[r for r in a4 if r['hardware']==hardware and r['object_instance']==object_name]
        mx=metric(x);my=metric(y)
        lines.append(f'| {hardware} | {object_name} | {mx["environment"]} | {my["environment"]} | {my["joint_static_pass"]} | {my["full_pass"]} |')
    lines+=['','## Értelmezés','',
            '- A teljes geometriai PASS a tíz ellenőrzési kapu státuszát jelzi **az OPT4 kísérleti jelenetben**.','- A törölt pontokat visszakereshetően tartalmazza a filter_evidence.npz / halo_deleted bitmask.',
            '- A jelöltek eltérhetnek a két futás között; az indexelt azonosítóik nem párosíthatók fizikai pózként.',
            '- Az újraosztályozott jelenetben kapott javulás nem helyettesíti a tényleges mélységképes ellenőrzést, a CAD javítását és az eredeti jeleneten történő regressziót.',
            '- Valós robot, IK, dinamika és ismeretlen tér nincs ellenőrizve.','']
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text('\n'.join(lines),encoding='utf-8')
    print('OPT4 metrics saved:',a.output)

if __name__=='__main__':main()
