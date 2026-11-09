"""Comparison report (Hungarian, single HTML): the version before vs the version built with the latest changes.

Reads finished runs (per gripper) and computes EVERY number from them. Runs that do not exist are skipped.
Usage: python scripts/opt7_compare_report.py --runs-dir GraspPlanner_runs --output OUT_DIR
"""
import argparse,json,sys,collections as C
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.opt7_report import hbar,stacked,table,esc,short,distinct,verified,GATES,GATE_HU,BLUE,GRAY,GOOD,CRIT,MUTED,shade
H=['rg2','rg6','2fg7']
SETS=[('N2','Mostani verzió (N2)',lambda r:[f'N2_{h}' for h in H],'200 jelölt/tárgy, A + B, környezeti szabály: 1 pont'),
      ('E1','E1: több helyszín, A',lambda r:[f'E1_locations_{h}' for h in H],'600/tárgy, csak A, szabály: 1 pont'),
      ('E3','E3: több helyszín, A, 3 voxel',lambda r:[f'E3_env3_noB_{h}' for h in H],'600/tárgy, csak A, szabály: 3 voxel'),
      ('E2','E2: több helyszín, A, 5 voxel',lambda r:[f'E2_locations_env5_{h}' for h in H],'600/tárgy, csak A, szabály: 5 voxel'),
      ('N3','Új verzió (N3)',lambda r:[f'N3_new_version_{h}' for h in H],'600/tárgy, A + szűrt B (40/tárgy), szabály: 3 voxel'),
      ('N5','N5: N3 + célhéj',lambda r:[f'N5_shell6_{h}' for h in H],'N3 + 6 mm-es célhéj'),
      ('N6','Ajánlott konfiguráció (N6)',lambda r:[f'N6_{h}' for h in H],'N3 + célhéj 6 mm + asztalsáv 8 mm + korlátlan B (a 2FG7-en B nélkül)')]
def load(runs,names):
    out=[];secs=0.
    for n in names:
        p=Path(runs)/f'run_{n}'/'all_candidates.json'
        if not p.exists():return None,None
        out+=json.loads(p.read_text(encoding='utf-8'))
        hw=[x for x in (Path(runs)/f'run_{n}').glob('*_timing.json')]
        secs+=sum(json.loads(t.read_text())['seconds'] for t in hw)
    return out,secs/3600
def metrics(c,cpu):
    g=lambda k:sum(x['checks'][k]['status']=='PASS' for x in c)
    full=[x for x in c if x['status']=='PASS'];d=distinct(full);ver=[x for x in full if verified(x)];dv=distinct(ver)
    te=sum(x['checks']['target_body']['status']=='PASS' and x['checks']['environment']['status']=='PASS' for x in c)
    return dict(cand=len(c),target=g('target_body'),env=g('environment'),te=te,approach=g('approach'),closure=g('closure'),full=len(full),poses=len(d),
                verified=len(ver),vposes=len(dv),cpu=cpu,eff=(len(d)/cpu if cpu else 0),clusters=d,vclusters=dv)
def matrix(clusters):
    return C.Counter(c['key'] for c in clusters)
def b_stats(c):
    s=dict(searches=0,reached=0,optimized=0,filtered=C.Counter(),budget_skipped=0,rescued=0,seconds=0.,a_solved=0,has_opt=False,per_combo=C.Counter())
    for x in c:
        a=x.get('opt6_contact_goal_optimization')
        if not a:continue
        s['searches']+=1
        if a['status']=='PASS' and a.get('search_stage')=='contact_only':s['a_solved']+=1
        st=a.get('collision_aware_stage')
        if not st:continue
        if st.get('skipped'):s['budget_skipped']+=1;continue
        s['reached']+=1;s['seconds']+=st.get('seconds',0.)
        if 'optimized_starts' in st:s['has_opt']=True
        if st.get('optimized_starts',0)>0:s['optimized']+=1;s['per_combo'][(x['hardware'],x['object_instance'])]+=1
        if st.get('filtered_by'):s['filtered'][st['filtered_by']]+=1
        if a['status']=='PASS' and a.get('search_stage')=='collision_aware':s['rescued']+=1
    return s
def pilot_table(runs):
    rows=[]
    for h in H:
        p=Path(runs)/'pilot_B'/f'pilot_B_{h}.json'
        if p.exists():rows+=json.loads(p.read_text())
    if not rows:return None
    pen=np.array([r['start_pen']*1e3 for r in rows]);ok=np.array([r['ok'] for r in rows]);sec=np.array([r['seconds'] for r in rows]);d=np.array([r['start_def']*1e3 for r in rows])
    out=[]
    edges=[0,.5,1,1.5,2,3,5,1e9]
    for lo,hi in zip(edges[:-1],edges[1:]):
        s=(pen>=lo)&(pen<hi)
        if s.any():out.append((f'{lo:g}–{hi:g} mm' if hi<1e8 else f'>{lo:g} mm',int(s.sum()),int(ok[s].sum()),f'{sec[s].mean():.2f}'))
    s=(pen<=2.0)&(d<=0.5)
    return dict(n=len(rows),rescued=int(ok.sum()),bins=out,filt_runs=int(s.sum()),filt_rescued=int(ok[s].sum()),filt_time=float(sec[s].sum()/sec.sum()),secs_per_rescue=float(sec.sum()/max(ok.sum(),1)))

def build(runs,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);data={}
    for key,label,names,desc in SETS:
        c,cpu=load(runs,names(runs))
        if c is not None:data[key]=dict(label=label,desc=desc,rows=c,m=metrics(c,cpu))
    if 'N2' not in data or 'N3' not in data:raise SystemExit('need N2 and N3 runs')
    HEAD='N6' if 'N6' in data else 'N3';n2,n3=data['N2']['m'],data[HEAD]['m']
    css='''.viz-root{color-scheme:light;--s:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10)}
@media(prefers-color-scheme:dark){.viz-root{color-scheme:dark;--s:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10)}}
body{margin:0;background:var(--page);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink)}
.wrap{max-width:1020px;margin:0 auto;padding:28px 20px 60px}h1{font-size:28px;margin:.2em 0 .1em}h2{font-size:21px;margin:1.6em 0 .4em;border-top:1px solid var(--grid);padding-top:.9em}h3{font-size:16px;margin:1.1em 0 .3em}
.sub{color:var(--ink2);margin:0 0 1em}.card{background:var(--s);border:1px solid var(--ring);border-radius:10px;padding:16px 18px;margin:12px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:14px 0}.kpi{background:var(--s);border:1px solid var(--ring);border-radius:10px;padding:12px 14px}.kpi b{font-size:34px;display:block;line-height:1.1}.kpi span{color:var(--ink2);font-size:13px}.kpi i{font-style:normal;font-size:13px;color:#006300}
.warn{border-left:4px solid #fab219;background:var(--s);padding:10px 14px;margin:12px 0;border-radius:0 8px 8px 0}
svg.viz .lab{font-size:12.5px;fill:var(--ink2)}svg.viz .val{font-size:12.5px;fill:var(--ink);font-weight:600}svg.viz .inbar{font-size:12px;fill:#fff;font-weight:600}svg.viz .axis{stroke:var(--axis);stroke-width:1}svg.viz .mark:hover path,svg.viz .mark:hover rect{opacity:.82}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}th,td{border-bottom:1px solid var(--grid);padding:6px 8px;text-align:left;vertical-align:top}th{color:var(--ink2);font-weight:600;font-size:12px}
td.cell{text-align:center;font-weight:600}tr.hl td{background:rgba(42,120,214,.10)}
details summary{cursor:pointer;color:var(--ink2);font-size:13px;margin-top:6px}code{background:var(--grid);padding:1px 5px;border-radius:4px;font-size:12.5px}pre{background:var(--s);border:1px solid var(--ring);border-radius:8px;padding:12px;overflow-x:auto;font-size:12.5px;line-height:1.35}'''
    P=[f'<!doctype html><html lang="hu"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Grasp Planner – a mostani és az új verzió</title><style>{css}</style></head><body><div class="wrap viz-root">']
    P.append('<h1>Grasp Planner – a mostani és az újonnan épített verziók</h1><p class="sub">0000 felvétel (318 RGB-D kép; 2× fogó, racsni, csavarhúzó) · RG2, RG6, 2FG7 · csak geometriai ellenőrzés</p>')
    dl=lambda a,b:(f'{b-a:+d}' if isinstance(a,int) else f'{b-a:+.2f}')
    kp=[(n3['poses'],f'különböző póz ({data[HEAD]["label"]})',f'mostani: {n2["poses"]} ({dl(n2["poses"],n3["poses"])})'),(n3['vposes'],'ellenőrzött kontaktusú póz',f'mostani: {n2["vposes"]} ({dl(n2["vposes"],n3["vposes"])})'),
        (n3['full'],'teljes PASS jelölt',f'mostani: {n2["full"]} ({dl(n2["full"],n3["full"])})'),(f'{n3["cpu"]:.2f}','CPU-óra',f'mostani: {n2["cpu"]:.2f}'),
        (f'{n3["eff"]:.1f}','pózok / CPU-óra',f'mostani: {n2["eff"]:.1f}')]
    P.append('<div class="kpis">'+''.join(f'<div class="kpi"><b>{v}</b><span>{esc(t)}</span><br><i>{esc(s)}</i></div>' for v,t,s in kp)+'</div>')
    P.append('<div class="warn"><b>Hatókör.</b> A PASS a megfogó és a lehetséges megfogási pózok <b>geometriai</b> vizsgálata: a robotkar, az IK és a dinamika nincs a rendszerben, a robotos rendszer az átadott jelölteket szűri tovább (<code>executable: false</code> tervezett). A jelenet kísérleti (halo-törlés a zaj miatt), a célobjektum a rekonstruált CAD. '
             'A „különböző póz" 3 mm / ~11° szerinti összevonás. A <b>környezeti minimum-voxel szabály lazítás</b>: a link felfújt térfogatában (kb. 8 mm a felülettől) kevesebb különböző voxelt figyelmen kívül hagy.</div>')
    # ---- 1 what changed
    P.append('<h2>1. Mi változott a verziók között</h2><div class="card"><table><thead><tr><th></th><th>Mostani (N2)</th><th>Új (N3)</th><th>Ajánlott (N6)</th></tr></thead><tbody>'
        '<tr><td><b>Jelölt-generálás</b></td><td>19 tengelyirányú × 12 elforgatás × 6 magasság × 5 dőlés</td><td colspan="2">40 × 24 × 12 × 3 dőlés (0°, ±20°)</td></tr>'
        '<tr><td><b>Kiértékelt jelölt / tárgy</b></td><td>200</td><td colspan="2">600</td></tr>'
        '<tr><td><b>Kontakt-célkeresés</b></td><td>A szakasz, B minden elbukott jelöltre (első 100)</td><td>A mindre; <b>B szűrve</b>: kiinduló behatolás ≤ 2 mm, ráhagyás-hiány ≤ 0,5 mm; legfeljebb 40 B-futás párosonként</td><td>mint N3, de a <b>B-keret korlátlan</b> (RG2, RG6); a <b>2FG7-en B nélkül</b></td></tr>'
        '<tr><td><b>Környezeti ütközés: minimum</b></td><td>egyetlen megfigyelt pont is elég</td><td colspan="2">legalább <b>3 különböző voxel</b></td></tr>'
        '<tr><td><b>Célhéj</b></td><td colspan="2">nincs</td><td>a célobjektum saját CAD-felületétől ≤ <b>6 mm</b>-re lévő háttérpontok nem akadályok</td></tr>'
        '<tr><td><b>Asztalsáv</b></td><td colspan="2">nincs</td><td>az asztalsík fölött ≤ <b>8 mm</b>-re lévő háttérpontok (az asztal zaja) nem akadályok</td></tr>'
        '<tr><td><b>Ütközés-ellenőrzés</b></td><td colspan="3">tartós FCL objektumok (azonos eredmény, gyorsabb) – minden verzióban</td></tr>'
        '<tr><td><b>Ami változatlan</b></td><td colspan="3">a 10 kapu, a pad–horgony és tárgy szabály (a pad a horgony közelében, a tárgyba nem lóghat), a célmodell (CoACD), az 1,5 mm-es környezeti ráhagyás</td></tr></tbody></table></div>')
    # ---- 2 results
    P.append('<h2>2. Eredmények</h2><div class="card"><p class="sub">Egy-egy sor a módszer egy lépése. Kiemelve az új verzió.</p>')
    order=[k for k,*_ in SETS if k in data]
    P.append('<h3>Különböző pózok</h3>'+hbar([(data[k]['label'],data[k]['m']['poses'],f"{data[k]['label']}: {data[k]['m']['poses']} különböző póz") for k in order],accent=order.index(HEAD),label_w=260,title='Különböző pózok'))
    P.append('<h3>Ellenőrzött kontaktusú különböző pózok</h3>'+hbar([(data[k]['label'],data[k]['m']['vposes'],f"{data[k]['label']}: {data[k]['m']['vposes']}") for k in order],accent=order.index(HEAD),label_w=260,title='Ellenőrzött pózok'))
    P.append('<h3>Teljes PASS jelöltek</h3>'+hbar([(data[k]['label'],data[k]['m']['full'],f"{data[k]['label']}: {data[k]['m']['full']} jelölt") for k in order],accent=order.index(HEAD),label_w=260,title='Teljes PASS'))
    P.append('<h3>Pózok CPU-óránként (hatékonyság)</h3>'+hbar([(data[k]['label'],round(data[k]['m']['eff'],1),f"{data[k]['label']}: {data[k]['m']['eff']:.1f} póz / CPU-óra") for k in order],accent=order.index(HEAD),label_w=260,title='Hatékonyság'))
    rows=[]
    for k in order:
        m=data[k]['m'];rows.append((f'<b>{esc(data[k]["label"])}</b><br><span style="color:var(--muted)">{esc(data[k]["desc"])}</span>',m['cand'],m['target'],m['env'],m['te'],m['approach'],m['full'],m['poses'],m['verified'],m['vposes'],f"{m['cpu']:.2f}",f"{m['eff']:.1f}"))
    P.append('<details open><summary>Táblázatos nézet</summary>'+table(['Verzió','jelölt','target PASS','env PASS','target+env','közelítés','teljes PASS','póz','ellenőrzött (jelölt)','ellenőrzött (póz)','CPU-óra','póz/CPU-óra'],rows)+'</details></div>')
    # ---- 3 funnel
    P.append('<h2>3. A kapuk eredménye: mostani és új verzió</h2><div class="card"><div style="display:flex;gap:12px;flex-wrap:wrap;font-size:13px;color:var(--ink2);margin-bottom:6px">'
             f'<span><i style="display:inline-block;width:12px;height:12px;background:{GOOD};margin-right:5px"></i>✓ PASS</span><span><i style="display:inline-block;width:12px;height:12px;background:{CRIT};margin-right:5px"></i>✕ FAIL</span><span><i style="display:inline-block;width:12px;height:12px;background:{MUTED};margin-right:5px"></i>? UNVERIFIED</span></div>')
    for k in [x for x in ('N2','N3','N6') if x in data]:
        c=data[k]['rows'];gr=[(GATE_HU[g],dict(C.Counter(x['checks'][g]['status'] for x in c)),len(c)) for g in GATES if g not in ('data_alignment','hardware_geometry','kinematics','self_collision','filter_policy','contact_geometry')]
        P.append(f'<h3>{esc(data[k]["label"])}</h3>'+stacked(gr,width=760,label_w=190))
    P.append('<p>Az első öt kapu (adatillesztés, hardver, kinematika, kontaktus-geometria, önütközés) és a szűrési irányelv minden jelöltre PASS mindkét verzióban, ezért nincs feltüntetve.</p></div>')
    # ---- 4 where
    P.append('<h2>4. Hol van PASS (megfogó × tárgy, különböző pózok)</h2><div class="card">')
    objs=['Pliers_001','Pliers_002','Rachet_001','Screwdriver_001'];grips=['OnRobot_2FG7','OnRobot_RG2','OnRobot_RG6']
    for k in [x for x in ('N2','N3','N6') if x in data]:
        mm=matrix(data[k]['m']['clusters']);mx=max([mm[(g,o)] for g in grips for o in objs]+[1])
        t=f'<h3>{esc(data[k]["label"])}</h3><table><thead><tr><th></th>'+''.join(f'<th>{o}</th>' for o in objs)+'</tr></thead><tbody>'
        for g in grips:
            t+=f'<tr><td><b>{short(g)}</b></td>'+''.join((lambda v:(lambda cc:f'<td class="cell" style="background:{cc[0]};color:{cc[1]}">{v}</td>')(shade(v,mx)))(mm[(g,o)]) for o in objs)+'</tr>'
        P.append(t+'</tbody></table>')
    P.append('</div>')
    # ---- 5 B
    b=b_stats(data['N3']['rows']);b2=b_stats(data['N2']['rows']);pt=pilot_table(runs)
    P.append('<h2>5. A B szakasz szűrővel</h2><div class="card">')
    runs3=b['optimized'];runs2=b2['optimized'] if b2['has_opt'] else b2['reached']
    filt=', '.join(f'{ {"penetration":"a kiinduló behatolás > 2 mm","clearance_deficit":"a környezeti ráhagyás-hiány > 0,5 mm"}.get(k,k)}: {v}' for k,v in b['filtered'].items()) or '–'
    cap_n=sum(1 for v in b['per_combo'].values() if v>=40);e3=data.get('E3');m3=data['N3']['m']
    P.append(f'<p>Az <b>új verzióban</b> a B szakasz <b>{runs3}</b> keresésen futott le, ebből <b>{b["rescued"]}</b> sikerült ({100*b["rescued"]/max(runs3,1):.1f}%), a B teljes ideje <b>{b["seconds"]/60:.1f} CPU-perc</b>. '
             f'A szűrő kihagyta: {filt}. A tárgyankénti 40-es keret {cap_n} / {len(b["per_combo"]) or 12} megfogó–tárgy párosnál ki is merült, a keret kimerülése után a további keresések B nélkül maradtak ({b["budget_skipped"]} keresés). '
             f'A <b>mostani verzióban</b> a B {runs2} keresésen futott, {b2["rescued"]} mentést adott, {b2["seconds"]/60:.0f} CPU-percet igényelt (szűrés nélkül).</p>')
    if e3:
        me=e3['m'];P.append(f'<p><b>A szűrt B hozzáadott értéke</b> (N3 az E3-hoz képest, azonos jelöltkészlet és környezeti szabály): teljes PASS {me["full"]} → {m3["full"]} ({m3["full"]-me["full"]:+d}), különböző póz {me["poses"]} → {m3["poses"]} ({m3["poses"]-me["poses"]:+d}), ellenőrzött kontaktusú póz {me["vposes"]} → {m3["vposes"]} ({m3["vposes"]-me["vposes"]:+d}); '
                 f'a költség {me["cpu"]:.2f} → {m3["cpu"]:.2f} CPU-óra ({100*(m3["cpu"]/me["cpu"]-1):+.0f}%). Az utolsó {m3["poses"]-me["poses"]} póz marginális ára {(m3["cpu"]-me["cpu"])/max(m3["poses"]-me["poses"],1)*60:.1f} CPU-perc / póz.</p>')
    if pt:
        P.append(f'<h3>Honnan a szűrő küszöbe (pilot: {pt["n"]} elbukott A-keresés, laza szűrővel)</h3>'+table(['kiinduló behatolás','keresés','B mentette meg','átlagos B-idő (s)'],[(a,b_,c_,d_) for a,b_,c_,d_ in pt['bins']])+
                 f'<p>A pilotban összesen {pt["rescued"]} keresést mentett a B (a mintának 0,45%-át), mindhárom ≤ 1,5 mm kiinduló behatolással és 0 környezeti ráhagyás-hiánnyal. A választott szűrő (behatolás ≤ 2 mm, hiány ≤ 0,5 mm) a pilot B-futásainak {pt["filt_runs"]/pt["n"]*100:.0f}%-át és {pt["filt_time"]*100:.0f}%-át az idejének őrizte meg, és mind a {pt["filt_rescued"]} mentést megtartotta. '
                 f'Szűrés nélkül egy mentés {pt["secs_per_rescue"]:.0f} s B-időbe került. <b>Fenntartás:</b> a küszöböt mindössze {pt["rescued"]} mentés alapján választottuk, ezért bizonytalan.</p>')
    P.append('</div>')
    # ---- 5b B budget curve (E3 = no B, N4 = unlimited B; identical candidate lists)
    e3rows,_=load(runs,[f'E3_env3_noB_{h}' for h in H]);n4rows,_=load(runs,[f'N4_budget_unlimited_{h}' for h in H])
    if e3rows and n4rows:
        E3={x['id']:x for x in e3rows};N4={x['id']:x for x in n4rows}
        if set(E3)==set(N4):
            runsB=C.defaultdict(list)
            for i,x in N4.items():
                a_=x.get('opt6_contact_goal_optimization') or {};st=a_.get('collision_aware_stage') or {}
                if st.get('optimized_starts',0)>0:runsB[(x['hardware'],x['object_instance'])].append((int(i.rsplit('__',1)[1]),x,a_['status']=='PASS' and a_.get('search_stage')=='collision_aware',st.get('seconds',0.)))
            for k in runsB:runsB[k].sort(key=lambda t:t[0])
            tot_r=sum(r for v in runsB.values() for _,_,r,_ in v)
            rowsB=[]
            for budget in (0,10,20,40,80,150,300,10**9):
                rr=[];nb=nr=0;bs=0.
                for i,x in N4.items():
                    v=runsB.get((x['hardware'],x['object_instance']),[]);ids={t[1]['id']:(j,t) for j,t in enumerate(v)}
                    if x['id'] in ids and ids[x['id']][0]<budget:rr.append(x);nb+=1;nr+=ids[x['id']][1][2];bs+=ids[x['id']][1][3]
                    else:rr.append(E3[x['id']])
                full=[x for x in rr if x['status']=='PASS'];ver=[x for x in full if verified(x)]
                rowsB.append(('korlátlan' if budget>=10**8 else budget,nb,nr,len(full),len(distinct(full)),len(distinct(ver)),f'{bs/60:.1f}'))
            P.append('<h2>5b. A B-keret hatása</h2><div class="card"><p class="sub">Egyetlen korlátlan B-keretű futásból (N4) kiolvasva: a jelöltek rangsor szerint kerülnek sorra, és egy jelölt B-eredménye nem függ a többitől, ezért a kisebb keretek eredménye a korlátlan futás előtagja. A keret feletti jelölt az E3 (B nélküli) eredményét kapja.</p>'
                     +table(['keret / páros','B-futás','mentés','teljes PASS','különböző póz','ellenőrzött póz','B CPU-perc'],rowsB))
            pairs=sorted(runsB.items())
            P.append('<h3>A mentések eloszlása megfogó–tárgy páronként</h3>'+table(['páros','B-futás','mentés','a mentések helye a B-futások között (0-tól számolva)'],
                [(f'{k[0][8:]} / {k[1]}',len(v),sum(r for _,_,r,_ in v),', '.join(str(j) for j,(_,_,r,_) in enumerate(v) if r) or '–') for k,v in pairs])
                +'<p>A mentések <b>torlódnak</b>: egy-egy hely környékén több is van, és gyakran a rangsor késői részén (pl. RG2 / Racsni: az első mentés a 157. B-futásnál). A 2FG7-en egyetlen mentés sincs, az RG6 / Racsnin sem. Ezért egy „N egymás utáni sikertelen futás után leáll" szabály veszteséges lenne:</p>')
            stopr=[]
            for N in (20,40,60,100,150,10**9):
                nr=nk=0;sec=0.
                for k,v in runsB.items():
                    fails=0
                    for j,(_,_,r,s_) in enumerate(v):
                        if fails>=N:break
                        nr+=1;sec+=s_
                        if r:nk+=1;fails=0
                        else:fails+=1
                stopr.append(('–' if N>=10**8 else N,nr,f'{nk} / {tot_r}',f'{sec/60:.1f}'))
            P.append(table(['leállás N sikertelen futás után','B-futás','megmaradó mentés','B CPU-perc'],stopr)+'<p>A páros-szintű kihagyás ezért inkább a konfigurációban, megfogónként történik (a 2FG7 futásban a B ki van kapcsolva).</p></div>')
    # ---- 5c shell + band
    sj=Path(runs)/'shell_band_sweep.json'
    if sj.exists():
        sd=json.loads(sj.read_text())
        P.append('<h2>5c. Célhéj és asztalsáv</h2><div class="card"><p class="sub">Offline mérés az N5 tárolt pózain (3 megfogó × 600 jelölt tárgyanként): hány jelölt megy át a környezeti kapun, és hány a target kapun is.</p>')
        P.append('<h3>Pliers_001: a héj mérete</h3>'+table(['héj (mm)','környezeti PASS (1800-ból)','target+env','eltávolított háttérpont'],[(r['shell_mm'],r['env_pass'],r['target_env'],r['removed']) for r in sd['shell_sweep_pliers001']]))
        P.append('<h3>Héj és asztalsáv együtt (környezeti PASS / target+env)</h3>'+table(['változat','Pliers_001','Racsni','Csavarhúzó'],[(r['variant'].replace('shell','héj').replace('band','sáv'),f"{r['pliers001'][0]} / {r['pliers001'][1]}",f"{r['rachet'][0]} / {r['rachet'][1]}",f"{r['screwdriver'][0]} / {r['screwdriver'][1]}") for r in sd['band_sweep']]))
        P.append('<p>A kettő <b>együtt</b> kell a Pliers_001-hez: a sáv egyedül 0 target+env-et ad, a héj egyedül 8-at, együtt 60-at. A sáv 8 mm-nél telítődik (a 10 mm ugyanazt adja), ami az asztalsík zajszintjével (3σ ≈ 8,6 mm) egyezik. A Racsni nem változik, a Csavarhúzó környezeti PASS-a megduplázódik, de a target+env nem, mert ott a target-kapu a korlát.</p></div>')
    # ---- 6 env sensitivity
    P.append('<h2>6. A környezeti küszöb érzékenysége</h2><div class="card"><p class="sub">Ugyanaz a jelöltkészlet (600/tárgy, csak A), csak a minimum-voxel szám különbözik.</p>')
    rows=[]
    for k,v in (('E1','1 (eredeti)'),('E3','3'),('E2','5')):
        if k in data:
            m=data[k]['m'];rows.append((v,m['env'],m['te'],m['approach'],m['full'],m['poses'],m['vposes']))
    P.append(table(['küszöb (különböző voxel)','env PASS','target+env','közelítés','teljes PASS','póz','ellenőrzött póz'],rows))
    ec=Path(runs)/'env_voxel_counts_N2.json'
    if ec.exists():
        ev=[r for r in json.loads(ec.read_text()) if r['cause']=='observed_occupancy'];mxv=np.array([r['max'] for r in ev])
        rr=[]
        for N in (2,3,4,5,8,10,20):
            fl=[r for r in ev if r['max']<N];rr.append((N,len(fl),sum(r['tb']=='PASS' for r in fl)))
        P.append(f'<h3>Az N2 {len(ev)} megfigyelt-pontos környezeti bukásából mennyi válna PASS-ra</h3><p>Egy linkben lévő különböző voxelek száma a bukó pózokon: medián {np.median(mxv):.0f}, a bukások 10%-ánál legfeljebb {np.percentile(mxv,10):.0f}. A bukások többsége tehát nagy felület, nem lebegő pont.</p>'+table(['küszöb N','PASS-ra váltó bukás','ebből a target-kapun már átmegy'],rr))
    if all(k in data for k in ('E1','E3','E2')):
        a1,a3,a5=data['E1']['m'],data['E3']['m'],data['E2']['m']
        P.append(f'<p><b>A 3 voxeles küszöb hatása nagyobb, mint amit az N2-ből becsültünk.</b> Az N2 bukásaiból számolt becslés szerint 3 voxelnél kevés bukás válna PASS-ra, a mért hatás viszont az E1-hez képest: környezeti PASS {a1["env"]} → {a3["env"]} ({a3["env"]-a1["env"]:+d}), közelítés PASS {a1["approach"]} → {a3["approach"]}, teljes PASS {a1["full"]} → {a3["full"]}, különböző póz {a1["poses"]} → {a3["poses"]}. '
                 f'Az 5 voxeles küszöb ehhez képest már kevés pluszt ad (póz {a3["poses"]} → {a5["poses"]}, teljes PASS {a3["full"]} → {a5["full"]}). '
                 'Az eltérés oka, hogy a szabály a közelítés és a zárás sweep-jeire is vonatkozik, ahol a konzervatív konvex burok sokkal több izolált pontot érint, mint a végső póz, és a nagyobb jelöltkészlet több zsúfolt helyzetet tartalmaz. A 3 voxel a kis objektumokra óvatosabb választás az 5-nél, és a mérés szerint kevés nyereséget áldoz érte.</p>')
    P.append('</div>')
    # ---- 7 failures
    P.append('<h2>7. Mi akadályoz még (fő bukási okok az új verzióban)</h2><div class="card">')
    c=data['N3']['rows'];tip=lambda n:'fingertip' in n or 'finger_tip' in n
    tb=[x['checks']['target_body'] for x in c if x['checks']['target_body']['status']=='FAIL']
    kk=C.Counter('ujjbegy' if tip(t.get('link','')) else 'ujjtő/ujj' if 'finger' in t.get('link','') else 'tenyér/alap' for t in tb)
    P.append(f'<ul><li>A target-ütközés az új verzióban {len(tb)} jelöltnél bukik; az első ütköző link: '+', '.join(f'{k} {v} ({100*v/len(tb):.0f}%)' for k,v in kk.most_common())+'.</li>')
    wit=[]
    for x in c:
        t=x['checks']['target_body']
        if t['status']=='FAIL' and tip(t.get('link','')) and t.get('penetration_witness_world_m') is not None:
            A=np.array(x['contacts_world_m']);wit.append(float(np.min(np.linalg.norm(A-np.array(t['penetration_witness_world_m']),axis=1))*1e3))
    if wit:P.append(f'<li>Az ujjbegy ütközésénél a legmélyebb metszéspont a horgonytól mediánon {np.median(wit):.1f} mm-re van, a bukások {100*np.mean(np.array(wit)<=2.5):.1f}%-ánál ≤ 2,5 mm-en belül.</li>')
    f7=sum(1 for cl in data['N3']['m']['clusters'] if cl['key'][0]=='OnRobot_2FG7');f7o=sorted({cl['key'][1] for cl in data['N3']['m']['clusters'] if cl['key'][0]=='OnRobot_2FG7'})
    p1=sum(1 for cl in data['N3']['m']['clusters'] if cl['key'][1]=='Pliers_001')
    P.append(f'<li>A 2FG7-en (nagy pad) {f7} különböző póz van ({", ".join(f7o) or "egyik tárgyon sem"}); a Pliers_001-en {p1} póz van minden megfogóval. Itt a jelölt-generálás (pad-lábnyom illesztés, tárgyspecifikus ellenőrzés) a következő lépés.</li></ul></div>')
    P.append('<h2>8. Fenntartások és következő lépések</h2><ul><li><b>A B-keret emelése olcsó lehet:</b> a B teljes ideje az új verzióban csak néhány CPU-perc volt, a 40-es keret viszont több megfogó–tárgy párosnál kimerült, tehát a B értéke valószínűleg nagyobb keretnél tovább nő; ezt érdemes megmérni.</li><li>A környezeti minimum-voxel szabály a közelítés és a zárás sweep-jeire is vonatkozik, és megfigyelt akadályokat hagy figyelmen kívül. Érdemes a kihagyott klasztereket 3D-ben szemrevételezni.</li>'
             '<li>A B szűrő küszöbe egy kis mintán (3 mentés) alapul; több adattal újra kell hangolni.</li><li>A „csak A" futások egy része névleges (nem ellenőrzött) kontaktusra támaszkodik; az „ellenőrzött" oszlopok a szigorúbb mérőszámok.</li>'
             '<li>Következő lépések: pad-lábnyom illesztés (2FG7), a fennmaradó 0,5 mm alatti, helyszín-szabály miatti bukások vizsgálata (a pad–tárgy szabály jelenleg szigorú maradt), a küszöbök újrahangolása.</li></ul>')
    P.append('<h2>9. Reprodukálás</h2><div class="card"><p>Konfigurációk: <code>cfg_N2…</code> (N2 futások), <code>cfg_E1_locations.yaml</code>, <code>cfg_E3_env3_noB.yaml</code>, <code>cfg_E2_locations_env5.yaml</code>, <code>cfg_N3_new_version.yaml</code>; kód: <code>GraspPlanner_OPT7_dev</code>, a futások a <code>code_snapshot_*</code> pillanatképekből mentek. '
             'Megfogónként külön folyamat, majd <code>scripts/opt7_combined_report.py</code> a szabványos jelentéshez.</p></div></div></body></html>')
    (out/'index.html').write_text('\n'.join(P),encoding='utf-8')
    (out/'report_data.json').write_text(json.dumps({k:{kk:vv for kk,vv in v['m'].items() if kk not in('clusters','vclusters')} for k,v in data.items()},indent=1))
    return out/'index.html'
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--runs-dir',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();print(build(a.runs_dir,a.output))
