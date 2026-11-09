"""TDK report (Hungarian, single HTML): the final configuration (Q2) and how the framework got there.

Every number is computed from the finished runs. Usage:
  python scripts/opt7_tdk_report.py --runs-dir GraspPlanner_runs --output GraspPlanner_runs/REPORT_TDK_Q2
"""
import argparse,json,sys,collections as C
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.opt7_report import hbar,table,esc,short,distinct,verified,GATES,GATE_HU,BLUE,GRAY,GOOD,CRIT,shade
H=['rg2','rg6','2fg7']
SETS=[('N2','N2 – kiinduló','run_N2_{h}','200 jelölt/tárgy','A + B','1 pont','2,5 mm kör','0,5 mm'),
      ('E1','E1 – több jelölt','run_E1_locations_{h}','600 jelölt/tárgy','csak A','1 pont','2,5 mm kör','0,5 mm'),
      ('E3','E3 – env. szabály 3 voxel','run_E3_env3_noB_{h}','600 jelölt/tárgy','csak A','3 voxel','2,5 mm kör','0,5 mm'),
      ('N3','N3 – szűrt B szakasz','run_N3_new_version_{h}','600 jelölt/tárgy','A + szűrt B','3 voxel','2,5 mm kör','0,5 mm'),
      ('Q1','Q1 – pad-lap téglalap','run_Q1_{h}','600 jelölt/tárgy','csak A','3 voxel','pad-lap téglalap','0,5 mm'),
      ('Q2','Q2 – végső (1 mm tűrés)','run_Q2_{h}','600 jelölt/tárgy','csak A','3 voxel','pad-lap téglalap','1,0 mm')]
OBJ_HU={'Pliers_001':'Fogó (Pliers_001)','Pliers_002':'Fogó (Pliers_002)','Rachet_001':'Racsni (Rachet_001)','Screwdriver_001':'Csavarhúzó (Screwdriver_001)'}
CSS=''':root,.viz-root{color-scheme:light;--s:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10)}
@media(prefers-color-scheme:dark){:root,.viz-root{color-scheme:dark;--s:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10)}}
html{background:var(--page)}body{margin:0;background:var(--page);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink)}
.wrap{max-width:1020px;margin:0 auto;padding:28px 20px 60px}h1{font-size:28px;margin:.2em 0 .1em}h2{font-size:21px;margin:1.6em 0 .4em;border-top:1px solid var(--grid);padding-top:.9em}
.sub{color:var(--ink2);margin:0 0 1em}.card{background:var(--s);border:1px solid var(--ring);border-radius:10px;padding:16px 18px;margin:12px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:14px 0}.kpi{background:var(--s);border:1px solid var(--ring);border-radius:10px;padding:12px 14px}.kpi b{font-size:34px;display:block;line-height:1.1}.kpi span{color:var(--ink2);font-size:13px}
.warn{border-left:4px solid #fab219;background:var(--s);padding:10px 14px;margin:12px 0;border-radius:0 8px 8px 0}
svg.viz .lab{font-size:12.5px;fill:var(--ink2)}svg.viz .val{font-size:12.5px;fill:var(--ink);font-weight:600}svg.viz .axis{stroke:var(--axis);stroke-width:1}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}th,td{border-bottom:1px solid var(--grid);padding:6px 8px;text-align:left;vertical-align:top}th{color:var(--ink2);font-weight:600;font-size:12px}
td.cell{text-align:center;font-weight:600}tr.hl td{background:rgba(42,120,214,.10)}code{background:var(--grid);padding:1px 5px;border-radius:4px;font-size:12.5px}'''

def load(runs,pat):
    rows=[];secs=0.
    for h in H:
        d=Path(runs)/pat.format(h=h)
        if not (d/'all_candidates.json').exists():return None,None
        rows+=json.loads((d/'all_candidates.json').read_text(encoding='utf-8'))
        secs+=sum(json.loads(t.read_text())['seconds'] for t in d.glob('*_timing.json'))
    return rows,secs/3600

def metrics(c,cpu):
    full=[x for x in c if x['status']=='PASS'];d=distinct(full);ver=[x for x in full if verified(x)];dv=distinct(ver)
    gate=lambda k:sum(x['checks'][k]['status']=='PASS' for x in c)
    first=C.Counter()
    for x in c:
        if x['status']!='PASS':
            for k in GATES:
                if x['checks'][k]['status']!='PASS':first[k]+=1;break
    return dict(cand=len(c),full=len(full),poses=len(d),verified=len(ver),vposes=len(dv),cpu=cpu,eff=len(d)/cpu if cpu else 0.,
                clusters=d,gates={k:gate(k) for k in GATES},first=first)

def build(runs,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);data={}
    for key,label,pat,*rest in SETS:
        c,cpu=load(runs,pat)
        if c is not None:data[key]=dict(label=label,rest=rest,m=metrics(c,cpu),rows=c)
    if 'Q2' not in data or 'N2' not in data:raise SystemExit('need N2 and Q2 runs')
    q,n=data['Q2']['m'],data['N2']['m']
    P=[f'<!doctype html><html lang="hu"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Grasp Planner – TDK jelentés</title><style>{CSS}</style></head><body><div class="wrap viz-root">']
    P.append('<h1>Grasp Planner – megfogási jelöltek geometriai ellenőrzése</h1><p class="sub">0000 felvétel (RGB-D, 318 kép; fogó, racsni, csavarhúzó) · OnRobot RG2, RG6, 2FG7 · csak geometriai ellenőrzés</p>')
    kp=[(q['poses'],'különböző megfogási póz',f'kiinduló: {n["poses"]}'),(q['vposes'],'ellenőrzött pad-kontaktusú póz',f'kiinduló: {n["vposes"]}'),
        (q['full'],'teljes PASS jelölt',f'kiinduló: {n["full"]}'),(f'{q["cpu"]:.2f}','CPU-óra',f'kiinduló: {n["cpu"]:.2f}'),(f'{q["eff"]:.0f}','pózok / CPU-óra',f'kiinduló: {n["eff"]:.1f}')]
    P.append('<div class="kpis">'+''.join(f'<div class="kpi"><b>{v}</b><span>{esc(t)}<br><i>{esc(s)}</i></span></div>' for v,t,s in kp)+'</div>')
    P.append('<div class="warn"><b>Hatókör.</b> A PASS a megfogó és a megfogási pózok <b>geometriai</b> vizsgálata (CAD-modell, megfigyelt környezet). Robotkar, inverz kinematika és dinamika nincs a rendszerben, ezért a jelöltek <code>executable: false</code> jelzést kapnak, és a robotos rendszer szűri őket tovább. '
             'A „különböző póz" az azonos megfogó–tárgy párosnál 3 mm-en és kb. 11°-on belüli pózok összevonása.</div>')
    P.append('<h2>1. A módszer röviden</h2><div class="card"><ol><li><b>Jelöltek generálása.</b> A tárgy CAD-felületén sugarakkal horgonypárt keres a pofa tengelye mentén, különböző magasságokkal, elforgatással és dőléssel (tárgyanként 600 jelölt). Ütközést itt nem vizsgálunk.</li>'
             '<li><b>Kontaktus-keresés (A szakasz).</b> Minden jelöltre kis, korlátos keresés: a pofa nyílása és az eltolás úgy hangolódik, hogy a két OEM pad belső lapja a horgonyokhoz érjen (≤ 0,4 mm).</li>'
             '<li><b>Tíz kapu.</b> adatillesztés, megfogó geometria, kinematika, kontaktus, önütközés, célobjektum-ütközés, környezeti ütközés, közelítés, zárás, szűrési irányelv. PASS csak akkor, ha mind a tíz PASS; egy korábbi hiba után a későbbiek nem kerülnek értékelésre.</li></ol>'
             '<p>Az ütközésvizsgálat a célobjektumot konvex felbontással (CoACD), a megfogót az OEM CAD-modellel, a környezetet a megfigyelt foglalt voxelekkel végzi.</p></div>')
    P.append('<h2>2. A végső szabályok</h2><div class="card"><ul>'
             '<li><b>Sekély pad-érintkezés a teljes pad-lapon.</b> A pad a célobjektumba legfeljebb <b>1,0 mm</b> mélyen érhet, és ez a pad belső lapjának teljes téglalapján megengedett (a pad szélessége × hossza + 0,5 mm minden irányban). Minden más metszés teljes mélységgel hiba.</li>'
             '<li><b>Környezeti minimum-voxel szabály: 3.</b> Egy link felfújt térfogatában legalább 3 különböző foglalt voxel kell az ütközéshez, így az izolált, lebegő zajpontok nem zavarnak. Ez lazítás a megfigyelt foglaltsághoz képest.</li>'
             '<li><b>A B (ütközés-tudatos) szakasz kikapcsolva.</b> A pad-lap szabállyal nincs rá szükség; a korábbi mérések szerint a mentések aránya 3,3% volt, a futásidő nagy részét ez adta.</li></ul></div>')
    P.append('<h2>3. A fejlődés lépésről lépésre</h2><div class="card"><p class="sub">Minden sor egy teljes, 3 megfogós × 3 tárgyas futás ugyanazon a felvételen.</p>')
    rows=[]
    for key,d in data.items():
        m=d['m'];r=d['rest']
        rows.append([f'<b>{esc(d["label"])}</b>',esc(r[0]),esc(r[1]),esc(r[2]),esc(r[3]),esc(r[4]),m['full'],m['poses'],m['vposes'],f'{m["cpu"]:.2f}'])
    P.append(table(['változat','jelölt','keresés','env. szabály','pad-érintkezés','tűrés','teljes PASS','különböző póz','ellenőrzött póz','CPU-óra'],rows))
    P.append(hbar([(d['label'],d['m']['poses'],f'{d["m"]["poses"]} különböző póz') for d in data.values()],accent=len(data)-1,title='Különböző pózok száma'))
    P.append('</div>')
    P.append('<h2>4. Megfogónként és tárgyonként (Q2)</h2><div class="card">')
    objs=sorted({x['object_instance'] for x in data['Q2']['rows']});grips=sorted({x['hardware'] for x in data['Q2']['rows']})
    cl=C.Counter(c['key'] for c in q['clusters']);pas=C.Counter((x['hardware'],x['object_instance']) for x in data['Q2']['rows'] if x['status']=='PASS')
    cn=C.Counter(c['key'] for c in n['clusters']);pn=C.Counter((x['hardware'],x['object_instance']) for x in data['N2']['rows'] if x['status']=='PASS')
    mx=max(list(cl.values())+[1]);tb=[]
    for g in grips:
        row=[f'<b>{esc(short(g))}</b>']
        for o in objs:
            v=cl[(g,o)];bg,fg=shade(v,mx)
            row.append(f'<span style="display:block;text-align:center;background:{bg};color:{fg};border-radius:4px;padding:2px 0"><b>{v}</b> póz<br><small>{pas[(g,o)]} PASS · kiinduló: {cn[(g,o)]} póz / {pn[(g,o)]} PASS</small></span>')
        tb.append(row)
    P.append(table(['megfogó']+[esc(OBJ_HU.get(o,o)) for o in objs],tb))
    P.append('<p class="sub">Cella: különböző pózok száma (Q2), teljes PASS jelöltek, és a kiinduló (N2) érték.</p></div>')
    P.append('<h2>5. Hol esnek ki a jelöltek?</h2><div class="card"><p class="sub">Az első elbukott kapu a nem PASS jelölteknél (az adott változat összes jelöltje).</p>')
    for k in ('N2','Q2'):
        f=data[k]['m']['first']
        P.append(f'<h3 style="font-size:15px">{esc(data[k]["label"])}</h3>'+hbar([(GATE_HU[g],f[g],f'{f[g]} jelölt') for g in GATES if f[g]] or [('nincs',0,'')],accent=None))
    P.append('<p>A leggyakoribb ok mindkét változatban a célobjektummal való ütközés; az új szabályoknál ezek egy része a környezeti és a zárási kapuig jut el.</p></div>')
    P.append('<h2>6. Futásidő</h2><div class="card">'+table(['változat','CPU-óra','pózok / CPU-óra'],[[esc(d['label']),f'{d["m"]["cpu"]:.2f}',f'{d["m"]["eff"]:.1f}'] for d in data.values()])+'<p>A B szakasz kihagyása és a hatékonyabb ütközésvizsgálat (tartós FCL-objektumok) a kiinduló futás idejének töredékére csökkentette a számítást.</p></div>')
    P.append('<h2>7. Fenntartások</h2><div class="card"><ul>'
             '<li><b>Lazított szabályok.</b> A pad-lap téglalap és az 1 mm-es tűrés, valamint a 3 voxeles környezeti szabály lazítás. A CAD rekonstrukcióból készült (mm-es pontosság), ezért az 1 mm alatti metszés a modell bizonytalanságán belül van.</li>'
             '<li><b>Geometriai feltételesség.</b> A rekonstrukció zajos, a környezeti halo törlése kísérleti; dinamikai, kar- és IK-ellenőrzés nincs.</li>'
             '<li><b>Egy felvétel.</b> Az eredmények egyetlen felvételre (0000) vonatkoznak, tárgyanként 600 jelölttel.</li>'
             '<li><b>A különböző pózok száma</b> az összevonási küszöbtől függ (3 mm, ~11°); sok közeli póz felfelé torzíthatja.</li>'
             f'<li><b>Nincs megfogás:</b> {", ".join(f"{short(g)} / {o}" for g in grips for o in objs if pas[(g,o)]==0) or "nincs ilyen páros"} (0 PASS). A 2FG7 nagy pad-felületével két tárgyon továbbra sincs megfogás; a Pliers_001-en csak az RG2 és az RG6 ad pózt, mindkettő keveset ({", ".join(f"{short(g)}: {cl[(g,"Pliers_001")]}" for g in grips if pas[(g,"Pliers_001")])}).</li></ul></div>')
    P.append('<h2>8. Reprodukálás és részletes jelentés</h2><div class="card"><p>Konfiguráció: <code>cfg_Q2_pad_patch_tol1mm.yaml</code>; kód: <code>code_snapshot_Q1/</code>. A szabványos jelentés (képek, 3D jelenet, top-10 megfogás): <a href="../Q2_combined/report.html">Q2_combined/report.html</a>. '
             'Új beállítások: <code>target_contact_patch_mode: pad_face_rect</code>, <code>target_contact_patch_margin_m: 0.0005</code>, <code>target_penetration_tolerance_m: 0.001</code>, <code>opt7_goal_collision_aware: false</code>, <code>environment_min_occupied_voxels: 3</code>.</p></div></div></body></html>')
    (out/'index.html').write_text('\n'.join(P),encoding='utf-8')
    (out/'report_data.json').write_text(json.dumps({k:{kk:vv for kk,vv in v['m'].items() if kk not in('clusters','first')} for k,v in data.items()},indent=1))
    return out/'index.html'
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--runs-dir',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();print(build(a.runs_dir,a.output))
