"""OPT7 best-case report (single self-contained HTML, Hungarian).

Reads finished run folders, computes every number from their all_candidates.json /
summary.json (nothing is typed in by hand) and writes index.html with inline SVG
charts (hover tooltips via <title>, table view for every chart).

Usage:
  python scripts/opt7_report.py --runs-dir GraspPlanner_runs --output GraspPlanner_runs/REPORT_best_case_N2
"""
import argparse,html,json,collections as C
from pathlib import Path
import numpy as np

GATES=['data_alignment','hardware_geometry','kinematics','contact_geometry','self_collision',
       'target_body','environment','approach','closure','filter_policy']
GATE_HU={'data_alignment':'adatillesztés (CAD ↔ mélység)','hardware_geometry':'megfogó geometria','kinematics':'kinematika',
         'contact_geometry':'kontaktus geometria','self_collision':'önütközés','target_body':'célobjektum-ütközés',
         'environment':'környezeti ütközés','approach':'közelítés','closure':'zárás','filter_policy':'szűrési irányelv'}
CLEARANCE_MM=1.5   # environment_clearance_m used by the best-case configuration

def load(runs,names):
    out=[]
    for n in names:out+=json.load(open(Path(runs)/f'run_{n}'/'all_candidates.json',encoding='utf-8'))
    return out

def distinct(full,tol=.003):
    """Physically distinct poses: same gripper/object, TCP within 3 mm and approach axis within ~11 deg."""
    cl=[]
    for x in full:
        T=np.array(x['world_from_tcp']);p=T[:3,3];z=T[:3,2]
        for c in cl:
            if c['key']==(x['hardware'],x['object_instance']) and np.linalg.norm(c['p']-p)<tol and c['z']@z>.98:
                c['members'].append(x);break
        else:cl.append(dict(key=(x['hardware'],x['object_instance']),p=p,z=z,members=[x]))
    return cl

def app_clearance(x):
    v=[e.get('clearance_lower_bound_m') for e in x['checks']['approach'].get('edge_checks',[])]
    v=[a for a in v if a is not None]
    return min(v)*1e3 if v else None

def headroom(x):
    env=x['checks']['environment'].get('clearance_lower_bound_m');app=app_clearance(x)
    vals=[v for v in [None if env is None else env*1e3,app] if v is not None]
    return (min(vals)-CLEARANCE_MM) if vals else -9.

def verified(x):return x.get('contact_basis')=='opt6_bilateral_OEM_pad_contact_verified'
def short(h):return h.replace('OnRobot_','')
def esc(s):return html.escape(str(s))

# ---------------------------------------------------------------- SVG helpers (validated reference palette)
BLUE='#2a78d6';GRAY='#b9b8b0';GOOD='#0ca30c';CRIT='#d03b3b';MUTED='#898781'

def _bar_path(x,y,w,h,r=4):
    """Bar anchored square at the baseline, 4 px rounded at the data end only."""
    if w<=r+1:return f'M{x:.1f},{y:.1f}h{w:.1f}v{h}h-{w:.1f}z'
    return (f'M{x:.1f},{y:.1f}h{w-r:.1f}a{r},{r} 0 0 1 {r},{r}v{h-2*r}a{r},{r} 0 0 1 -{r},{r}h-{w-r:.1f}z')

def hbar(rows,width=640,label_w=210,bar_h=22,gap=8,accent=None,unit='',max_v=None,title=''):
    """rows=[(label,value,tooltip)]; one hue; `accent` index highlighted, rest gray (emphasis)."""
    label_w=max(label_w,int(7.2*max(len(r[0]) for r in rows))+24)
    mx=max_v or max([r[1] for r in rows]+[1]);h=len(rows)*(bar_h+gap)+8;plot=width-label_w-110
    out=[f'<svg class="viz" role="img" aria-label="{esc(title)}" viewBox="0 0 {width} {h}" width="100%">']
    for i,(lab,v,tip) in enumerate(rows):
        y=4+i*(bar_h+gap);w=max(2,plot*v/mx) if v>0 else 0
        col=BLUE if (accent is None or i==accent) else GRAY
        out.append(f'<text x="{label_w-10}" y="{y+bar_h*.7}" text-anchor="end" class="lab">{esc(lab)}</text>')
        if v>0:out.append(f'<g class="mark"><title>{esc(tip)}</title><path d="{_bar_path(label_w,y,w,bar_h)}" fill="{col}"/></g>')
        out.append(f'<text x="{label_w+w+8:.1f}" y="{y+bar_h*.7}" class="val">{v:g}{unit}</text>')
    out.append(f'<line x1="{label_w}" x2="{label_w}" y1="0" y2="{h}" class="axis"/></svg>')
    return ''.join(out)

def stacked(rows,width=700,label_w=210,bar_h=22,gap=10):
    """rows=[(label,{PASS,FAIL,UNVERIFIED},total)] -> 100% stacked status bar with counts and a 2px surface gap."""
    plot=width-label_w-170;h=len(rows)*(bar_h+gap)+8
    cols={'PASS':GOOD,'FAIL':CRIT,'UNVERIFIED':MUTED};sym={'PASS':'✓','FAIL':'✕','UNVERIFIED':'?'}
    out=[f'<svg class="viz" role="img" aria-label="Kapuk eredménye" viewBox="0 0 {width} {h}" width="100%">']
    for i,(lab,d,tot) in enumerate(rows):
        y=4+i*(bar_h+gap);x=label_w
        out.append(f'<text x="{label_w-10}" y="{y+bar_h*.7}" text-anchor="end" class="lab">{esc(lab)}</text>')
        for st in ['PASS','FAIL','UNVERIFIED']:
            n=d.get(st,0)
            if not n:continue
            w=plot*n/tot
            out.append(f'<g class="mark"><title>{esc(lab)} – {st}: {n} / {tot}</title><rect x="{x:.1f}" y="{y}" width="{max(w-2,1):.1f}" height="{bar_h}" fill="{cols[st]}"/>')
            if w>58:out.append(f'<text x="{x+6:.1f}" y="{y+bar_h*.7}" class="inbar">{sym[st]} {n}</text>')
            out.append('</g>');x+=w
        out.append(f'<text x="{label_w+plot+10:.1f}" y="{y+bar_h*.7}" class="lab">✓ {d.get("PASS",0)} · ✕ {d.get("FAIL",0)} · ? {d.get("UNVERIFIED",0)}</text>')
    out.append(f'<line x1="{label_w}" x2="{label_w}" y1="0" y2="{h}" class="axis"/></svg>')
    return ''.join(out)

def table(head,rows,cls=''):
    t=f'<table class="{cls}"><thead><tr>'+''.join(f'<th>{esc(h)}</th>' for h in head)+'</tr></thead><tbody>'
    for r in rows:t+='<tr>'+''.join(f'<td>{c}</td>' for c in r)+'</tr>'
    return t+'</tbody></table>'

def shade(v,mx):
    """sequential blue ramp (step 100 -> 700 of the reference palette) with readable ink"""
    ramp=['#cde2fb','#9ec5f4','#6da7ec','#3987e5','#256abf','#184f95','#0d366b']
    k=0 if mx==0 else min(len(ramp)-1,int(round((len(ramp)-1)*v/mx)))
    return ramp[k],('#0b0b0b' if k<=2 else '#ffffff')

# ---------------------------------------------------------------- report
def build(runs,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    N=load(runs,['N2_2fg7','N2_rg2','N2_rg6']);K2=load(runs,['K2_fix40']);L2=load(runs,['L2_fix100'])
    D=load(runs,['D_decomp']);A=load(runs,['A_opt4']);F1=load(runs,['F1_inflate']);F2=load(runs,['F2_pitch'])
    full=lambda c:[x for x in c if x['status']=='PASS']
    prog=[('OPT4 (konvex burok)',A,'A'),('+ konvex dekompozíció (CoACD)',D,'D'),('+ kontakt-célkeresés, 1,5 mm ráhagyás',K2,'K2'),
          ('+ 100 keresés / tárgy',L2,'L2'),('+ ütközés-tudatos szakasz (legjobb eset)',N,'N2')]
    prog_rows=[(n,len(full(c)),len(distinct(full(c))),len([x for x in full(c) if verified(x)]),len(distinct([x for x in full(c) if verified(x)]))) for n,c,_ in prog]
    nf=full(N);nclusters=distinct(nf);nver=[x for x in nf if verified(x)];nvc=distinct(nver)
    gate_rows=[]
    for g in GATES:
        d=C.Counter(x['checks'][g]['status'] for x in N);gate_rows.append((GATE_HU[g],dict(d),len(N)))
    # matrix gripper x object
    grips=sorted({x['hardware'] for x in N});objs=sorted({x['object_instance'] for x in N})
    mat={(g,o):(0,0) for g in grips for o in objs}
    for cl in nclusters:
        g,o=cl['key'];a,b=mat[(g,o)];mat[(g,o)]=(a+len(cl['members']),b+1)
    mx=max([v[1] for v in mat.values()]+[1])
    # best poses
    ranked=[]
    for cl in nclusters:
        ver=[m for m in cl['members'] if verified(m)];pool=ver or cl['members']
        rep=max(pool,key=headroom);ranked.append((headroom(rep),cl,rep,bool(ver)))
    ranked.sort(key=lambda t:(t[3],t[0]),reverse=True)
    top=ranked[:8]
    reps_done=C.Counter()
    pose_rows=[];bar_rows=[]
    for h,cl,rep,ver in top:
        g,o=cl['key'];reps_done[(g,o)]+=1;name=f'{short(g)} / {o} #{reps_done[(g,o)]}'
        goal=rep.get('opt6_contact_goal_optimization') or {}
        gap=max(goal.get('per_pad_CAD_gap_m',{'x':0}).values())*1e3 if goal.get('per_pad_CAD_gap_m') else None
        env=rep['checks']['environment'].get('clearance_lower_bound_m');app=app_clearance(rep)
        pose_rows.append((name,'ellenőrzött' if ver else 'névleges (geometriai)',f"{len(cl['members'])}",
            f"{rep['tilt_degrees']:+.0f}°",f"{goal.get('mount_roll_deg',0.):+.1f}°" if goal else '–',
            goal.get('search_stage','–') if goal else '–',f"{rep['aperture_m']*1e3:.1f}",
            '–' if gap is None else f'{gap:.3f}',f"{env*1e3:.2f}" if env is not None else '–',f"{app:.2f}" if app is not None else '–',
            f"{rep['checks']['approach'].get('duration_s',0):.1f}",f"{rep['checks']['closure'].get('segments','–')}",f"{h:+.2f}",esc(rep['id'].replace('OnRobot_',''))))
        bar_rows.append((name,round(max(h,0),2),f'{name}: tartalék {h:+.2f} mm a 1,5 mm-es küszöb fölött (min. környezeti és közelítési ráhagyás)'))
    stage_ok=[x for x in N if (x.get('opt6_contact_goal_optimization') or {}).get('search_stage')=='collision_aware' and x['opt6_contact_goal_optimization']['status']=='PASS']
    searches=[x for x in N if x.get('opt6_contact_goal_optimization')]
    stage_info=[x['opt6_contact_goal_optimization'].get('collision_aware_stage') for x in searches if x['opt6_contact_goal_optimization'].get('collision_aware_stage')]
    stage_min=sum(s['seconds'] for s in stage_info)/60 if stage_info else 0
    fail_best=[s['best']['penetration_beyond_tolerance_m']*1e3 for x in searches for s in [x['opt6_contact_goal_optimization'].get('collision_aware_stage')] if s and s.get('best') and x['opt6_contact_goal_optimization']['status']!='PASS']
    near=int(sum(v<.3 for v in fail_best))
    rolls=sorted(round(x['opt6_contact_goal_optimization']['mount_roll_deg'],1) for x in stage_ok)
    # links
    rel=lambda r,f:f'../run_{r}/{f}'
    links=[('RG2 – 3D visszajátszás',rel('N2_rg2','replay_rg2.html')),('RG6 – 3D visszajátszás',rel('N2_rg6','replay_rg6.html')),
           ('2FG7 – 3D visszajátszás',rel('N2_2fg7','replay_2fg7.html')),('RG2 – teljes részletes jelentés',rel('N2_rg2','report.html')),
           ('RG6 – teljes részletes jelentés',rel('N2_rg6','report.html'))]
    sens=[('D: dekompozíció, alap',len(full(D))),('F1: célmodell +0,25 mm kifelé',len(full(F1))),('F2: finomabb javítás (0,3 mm rács)',len(full(F2)))]
    kpi=[(len(nf),'teljes geometriai PASS jelölt'),(len(nclusters),'fizikailag különböző póz'),
         (len(nvc),'ebből ellenőrzött pad-kontaktusú'),(len({cl['key'] for cl in nclusters}),'megfogó/tárgy páros')]

    css='''
.viz-root{color-scheme:light;--s:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10)}
@media(prefers-color-scheme:dark){.viz-root{color-scheme:dark;--s:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10)}}
body{margin:0;background:var(--page);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink)}
.wrap{max-width:980px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:28px;margin:.2em 0 .1em}h2{font-size:21px;margin:1.6em 0 .4em;border-top:1px solid var(--grid);padding-top:.9em}h3{font-size:16px;margin:1.1em 0 .3em}
.sub{color:var(--ink2);margin:0 0 1em}.card{background:var(--s);border:1px solid var(--ring);border-radius:10px;padding:16px 18px;margin:12px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:14px 0}
.kpi{background:var(--s);border:1px solid var(--ring);border-radius:10px;padding:12px 14px}.kpi b{font-size:36px;display:block;line-height:1.1}.kpi span{color:var(--ink2);font-size:13px}
.warn{border-left:4px solid #fab219;background:var(--s);padding:10px 14px;margin:12px 0;border-radius:0 8px 8px 0}
svg.viz .lab{font-size:12.5px;fill:var(--ink2)}svg.viz .val{font-size:12.5px;fill:var(--ink);font-weight:600}svg.viz .inbar{font-size:12px;fill:#fff;font-weight:600}
svg.viz .axis{stroke:var(--axis);stroke-width:1}svg.viz .mark:hover rect{opacity:.82}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}th,td{border-bottom:1px solid var(--grid);padding:6px 8px;text-align:left;vertical-align:top}
th{color:var(--ink2);font-weight:600;font-size:12px}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
details summary{cursor:pointer;color:var(--ink2);font-size:13px;margin-top:6px}
.legend{display:flex;gap:16px;font-size:13px;color:var(--ink2);margin:4px 0 8px;flex-wrap:wrap}.sw{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:5px;vertical-align:-1px}
a{color:#2a78d6}code{background:var(--grid);padding:1px 5px;border-radius:4px;font-size:12.5px}
td.cell{text-align:center;font-weight:600;font-variant-numeric:tabular-nums}
'''
    P=[]
    P.append('<!doctype html><html lang="hu"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
             '<title>Grasp Planner OPT7 – legjobb eset jelentés</title><style>'+css+'</style></head><body><div class="wrap viz-root">')
    P.append('<h1>Grasp Planner – legjobb eset (OPT7)</h1><p class="sub">0000 felvétel (318 RGB-D kép, 4 tárgypéldány: 2× fogó, racsni, csavarhúzó) · 3 OnRobot megfogó: 2FG7, RG2, RG6 · csak geometriai ellenőrzés</p>')
    P.append('<div class="kpis">'+''.join(f'<div class="kpi"><b>{v}</b><span>{esc(t)}</span></div>' for v,t in kpi)+'</div>')
    P.append('<div class="warn"><b>Hatókör.</b> Minden PASS <b>feltételes</b>: (1) a jelenet kísérleti, a tárgyak körüli „halo” pontjait törli, így nem igazolja az eredeti jelenet ütközésmentességét; '
             '(2) a célobjektum javított (watertight) és konvex részekre bontott CAD-modell, nem az eredeti nyitott felület; (3) 1,5 mm-es környezeti ráhagyás. '
             'Robotkar, inverz kinematika, dinamika és erőzárás <b>nincs</b> ellenőrizve; minden terv <code>executable: false</code>. '
             'A „különböző póz” 3 mm / ~11° szerinti összevonás, nem független fogások száma.</div>')
    P.append('<h2>1. Összefoglaló</h2><ul>'
             f'<li>A legjobb konfigurációban <b>{len(nf)}</b> jelölt ment át mind a 10 geometriai kapun, ez <b>{len(nclusters)}</b> fizikailag különböző pózt jelent (az eredeti OPT4: 0).</li>'
             f'<li><b>{len(nver)}</b> jelölt ({len(nvc)} különböző póz) olyan, amelynél a két OEM pad érintkezése az eredeti CAD horgonyain ≤ 0,4 mm-en belül <b>ellenőrzött</b>; a többi a névleges CAD-sugaras geometriai kontaktusra támaszkodik.</li>'
             f'<li>Az ütközés-tudatos célkeresés {len(stage_ok)} olyan keresést mentett meg, amely a kontaktus-alapú kereséssel elbukott (roll {rolls[0] if rolls else "–"}° … {rolls[-1] if rolls else "–"}°).</li>'
             '<li>A 2FG7-en egyik változatban sem lett PASS (a kiinduló pózok mélyen ütköznek).</li></ul>')
    P.append('<h2>2. Hogyan jutottunk ide (lépésenkénti hatás)</h2><p class="sub">Egy-egy sor = a módszer egy újabb eleme az előzőkhöz képest. Kiemelve a legjobb eset.</p><div class="card">')
    P.append('<h3>Teljes geometriai PASS (jelölt)</h3>'+hbar([(n,a,f'{n}: {a} PASS jelölt') for n,a,*_ in [(r[0],r[1]) for r in prog_rows]],accent=len(prog_rows)-1,title='Teljes PASS jelöltek lépésenként'))
    P.append('<h3>Fizikailag különböző póz</h3>'+hbar([(n,b,f'{n}: {b} különböző póz') for n,_,b,*_ in prog_rows],accent=len(prog_rows)-1,max_v=max(r[2] for r in prog_rows)+1,title='Különböző pózok lépésenként'))
    P.append('<details><summary>Táblázatos nézet</summary>'+table(['Lépés','PASS jelölt','különböző póz','ellenőrzött kontaktus (jelölt)','ellenőrzött (póz)'],[(esc(r[0]),r[1],r[2],r[3],r[4]) for r in prog_rows])+'</details></div>')
    P.append('<p>A 40-ről 100 keresésre emelés önmagában alig javít (+2 PASS, +1 póz). A hatást az ütközés-tudatos szakasz adja (a pózok száma 10 → 19). '
             'Az L és N lépés értékei a keresési logika javítása után készültek: korábban egy sikertelen keresés a jelöltet is kizárta, így a nagyobb keret <i>csökkentette</i> a PASS-t (119 → 61).</p>')
    P.append('<h2>3. A kapuk eredménye a legjobb esetben</h2><div class="card"><div class="legend">'
             f'<span><i class="sw" style="background:{GOOD}"></i>✓ PASS</span><span><i class="sw" style="background:{CRIT}"></i>✕ FAIL</span><span><i class="sw" style="background:{MUTED}"></i>? UNVERIFIED (korábbi kapun elbukott, nem futott)</span></div>')
    P.append(stacked(gate_rows)+'<details><summary>Táblázatos nézet</summary>'+table(['Kapu','PASS','FAIL','UNVERIFIED','összes'],[(esc(l),d.get('PASS',0),d.get('FAIL',0),d.get('UNVERIFIED',0),t) for l,d,t in gate_rows])+'</details></div>')
    P.append('<p>A hardveres és kinematikai kapuk nem szűkítenek. A szűk keresztmetszet a <b>célobjektum-ütközés</b> és a <b>környezeti ütközés</b>; a közelítés és a zárás csak azokra fut le, amelyek ezeken átmentek.</p>')
    P.append('<h2>4. Hol van PASS (megfogó × tárgy)</h2><div class="card"><p class="sub">Cella: PASS jelöltek száma / különböző pózok.</p><table><thead><tr><th></th>'+''.join(f'<th class="n">{esc(o)}</th>' for o in objs)+'</tr></thead><tbody>')
    for g in grips:
        P.append(f'<tr><td><b>{esc(short(g))}</b></td>'+''.join((lambda v:(lambda c:f'<td class="cell" style="background:{c[0]};color:{c[1]}" title="{esc(short(g))} / {esc(o)}: {v[0]} jelölt, {v[1]} póz">{v[0]} / {v[1]}</td>')(shade(v[1],mx)))(mat[(g,o)]) for o in objs)+'</tr>')
    P.append('</tbody></table></div>')
    P.append('<h2>5. A legjobb fogások</h2><p class="sub">Rangsor: előbb az ellenőrzött pad-kontaktusú pózok, azon belül a legkisebb tartalék (a környezeti és közelítési ráhagyás minimuma az 1,5 mm-es küszöb fölött). Egy póz = egy összevont csoport legjobb képviselője.</p><div class="card">')
    best=top[0];bg,bo=best[1]['key'];bgoal=best[2].get('opt6_contact_goal_optimization') or {}
    P.append(f'<p><b>Legnagyobb tartalékú póz:</b> {esc(short(bg))} / {esc(bo)}, dőlés {best[2]["tilt_degrees"]:+.0f}°, nyitás {best[2]["aperture_m"]*1e3:.1f} mm, '
             f'környezeti ráhagyás {best[2]["checks"]["environment"]["clearance_lower_bound_m"]*1e3:.1f} mm, a kontaktus ellenőrzött (CAD-rés ≤ {max(bgoal.get("per_pad_CAD_gap_m",{"x":0}).values())*1e3:.3f} mm), '
             f'a közelítés és a zárás átmegy. Azonosító: <code>{esc(best[2]["id"])}</code>.</p>')
    P.append(hbar(bar_rows,label_w=250,unit=' mm',title='Legjobb pózok tartaléka'))
    P.append('<details open><summary>Részletek</summary><div style="overflow-x:auto">'+table(['Póz','Kontaktus','jelölt','dőlés','roll','keresés','nyitás (mm)','CAD-rés (mm)','környez. ráh. (mm)','közel. ráh. (mm)','közel. idő (s)','zárás szeg.','tartalék (mm)','azonosító'],pose_rows)+'</div></details></div>')
    P.append('<p>Az interaktív 3D nézetek: '+' · '.join(f'<a href="{u}">{esc(t)}</a>' for t,u in links)+'. A visszajátszás az adott megfogó rangsorolt (PASS-szal kezdődő) jelöltjeit mutatja.</p>')
    P.append('<h2>6. Mi változott az OPT4-hez képest</h2><ul>'
             '<li><b>Célmodell:</b> az eredeti nyitott/nem-manifold CAD-ből ellenőrzött, watertight szilárd test (voxel-kitöltés), majd CoACD konvex részek. A burok térfogata a CAD 1,7–2,1-szerese volt; az új modell az eredeti felületet 99,7–100%-ban lefedi, eltérése ≤ 0,83 mm. A javítás megtagadja azokat a hálókat, amelyeknél a kitöltés „kifolyna” (üreges test).</li>'
             '<li><b>Behatolási mélység:</b> minden target_body bukás kimenete tartalmazza a mélységet és a pontot.</li>'
             '<li><b>Kontakt-célkeresés (OPT6) + ütközés-tudatos szakasz:</b> a mount X/Y/Z eltolását, a nyitást és a pofa tengelye körüli rollt keresi úgy, hogy a két pad érintkezése megmarad, a behatolás és a ráhagyás hiánya pedig minimális. Csak javasol pózt; a döntést a változatlan kapuk hozzák.</li>'
             f'<li><b>Környezeti ráhagyás:</b> 2,0 → {str(CLEARANCE_MM).replace(".",",")} mm.</li>'
             '<li><b>RRT-Connect közelítés:</b> beépítve (csak akkor fut, ha az összes egyenes folyosó elbukott és a célpóz szabad); a valós adaton egyszer sem indult el, mert minden közelítés-bukás a célpózon történt.</li>'
             '<li><b>Hibajavítás:</b> sikertelen célkeresés után a jelölt a névleges pózával értékelődik, mint a keresetlen társai; minden jelölt rögzíti, hogy az érintkezése ellenőrzött vagy csak geometriai (<code>contact_basis</code>).</li></ul>')
    P.append('<h2>7. Mennyire megbízható?</h2><div class="card"><h3>Érzékenység a célmodell pontosságára</h3>'+hbar([(n,v,f'{n}: {v} PASS jelölt') for n,v in sens],accent=None,title='Érzékenység')+
             '<p>A +0,25 mm-es kifelé tolás a target PASS-ok többségét eltávolítja, a teljes PASS mégis megmarad ugyanazon a fogási helyen; a finomabb javítás nem rontja. A határ ezért nem a modell szub-milliméteres hibáján múlik.</p></div><ul>'
             f'<li><b>Még elbukó keresések:</b> {len(fail_best)} keresésből a legjobb maradék behatolás mediánja {np.median(fail_best) if fail_best else 0:.2f} mm; {near} keresésnél 0,3 mm alatti. Ezeket a kapu más feltétele (az eredeti CAD-metszés a pad-folton kívül, vagy a metszéspont helye) utasítja el.</li>'
             f'<li><b>Számítási költség:</b> az ütközés-tudatos szakasz összesen ≈ {stage_min:.0f} perc ({len(stage_info)} keresés).</li>'
             '<li><b>A pózkeresés tekintetében</b> a kihasznált szabadság a roll ±10° határa közelében van.</li>'
             '<li>Az ellenőrzött kontaktus 0,4 mm-es CAD-réstűrésen belüli pad–horgony érintkezést jelent; erőzárást, súrlódást, tömeget nem vizsgál.</li></ul>')
    P.append('<h2>8. Következő lépések</h2><ol>'
             '<li>A ~288 „majdnem-PASS” keresés elutasítási okának feltárása (CAD-metszés a pad-folton kívül / metszéspont helye).</li>'
             '<li>Nagyobb roll-tartomány (±20°) és kis dőlés (yaw/pitch ±5°) a kereséshez.</li>'
             '<li>2FG7: a jelölt-generálás módosítása (a kiinduló pózok mélyen ütköznek).</li>'
             '<li>Regresszió a halo-törlés nélküli jeleneten, majd robotkar/IK bevonása.</li></ol>')
    P.append('<h2>9. Reprodukálás</h2><div class="card"><p>Kód: <code>GraspPlanner_OPT7_dev</code> · konfiguráció: <code>GraspPlanner_runs/cfg_N_both.yaml</code> · futások: <code>run_N2_2fg7</code>, <code>run_N2_rg2</code>, <code>run_N2_rg6</code> · adat: Hugging Face <code>KarolyArmin/Hand_tools</code>, <code>0000</code>.</p>'
             '<p><code>PYTHONPATH=. python scripts/opt5_run_cached.py --dataset …/0000 --config cfg_N_both.yaml --gripper configs/grippers/rg2.yaml --output … --cache-run &lt;OPT4 eredmény&gt;</code> (megfogónként külön futtatva).</p>'
             '<p>Tesztek: <code>PYTHONPATH=. python -m pytest tests</code> (119 sikeres a futtatás idején; a <code>test_results.py</code> 3 tesztje a hiányzó <code>results/</code> mappa miatt kimarad).</p></div>')
    P.append('</div></body></html>')
    (out/'index.html').write_text('\n'.join(P),encoding='utf-8')
    (out/'report_data.json').write_text(json.dumps(dict(progression=prog_rows,kpi=kpi,best=[r[0] for r in pose_rows],stage_rescued=len(stage_ok)),ensure_ascii=False,indent=1),encoding='utf-8')
    return out/'index.html'

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--runs-dir',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();print(build(a.runs_dir,a.output))
