"""Scientific plots, offline camera-frame 3D models and kinematic replays."""
from pathlib import Path
import json,html
import numpy as np,trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from .models import transform,pose,save
from .ur_hardware import CADGripper
from .geometry_planner import GATES,rank_key
COLORS=['#85929e','#e46c42','#1aa79e','#8765bc','#dab442']

def render(out,cap,rec,rows,specs,cfg):
 out=Path(out);figdir=out/'figures';figdir.mkdir(exist_ok=True);modeldir=out/'models_3d';modeldir.mkdir(exist_ok=True)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 raw=np.load(out/'raw/reconstruction.npz');ev=np.load(out/'filter_evidence.npz');audit=json.loads((out/'filter_audit.json').read_text());summary=json.loads((out/'summary.json').read_text());C=np.asarray(cap.by_id[cfg.reference_frame]['camera_from_world'])
 up=rec.plane[:3];x=np.array([1.,0,0]);x-=up*(x@up);x/=np.linalg.norm(x);D=pose(np.stack([x,np.cross(up,x),up]),[0,0,rec.plane[3]])
 gs={h.name:CADGripper(h) for h in specs};ov=transform(np.vstack([o.mesh.vertices for o in cap.objects]),D);lo=ov.min(0)-.06;hi=ov.max(0)+.08;lo[2]=-.02;hi[2]=.36;gallery=[]
 def export(fig,name,caption):
  for ext in ['png','svg']:fig.savefig(figdir/(name+'.'+ext),dpi=190,bbox_inches='tight',pad_inches=.25)
  plt.close(fig);gallery.append((name,caption))
 def axes(ax,title=''):
  ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='asztal X [m]',ylabel='asztal Y [m]',zlabel='magasság [m]',title=title);ax.set_box_aspect(hi-lo);ax.view_init(32,-58);ax.tick_params(labelsize=8)
 def cloud(ax,p,col,limit=13000):
  v=transform(p,D);valid=np.flatnonzero(np.all((v>=lo)&(v<=hi),axis=1));ix=valid[np.linspace(0,len(valid)-1,min(limit,len(valid))).astype(int)];ax.scatter(*v[ix].T,c=np.asarray(col)[ix],s=.8,alpha=.65,rasterized=True)
 def poly(ax,m,color,alpha=.9,display_transform=D):
  v=transform(m.vertices,display_transform);ax.add_collection3d(Poly3DCollection(v[m.faces],facecolors=color,linewidths=0,alpha=alpha,shade=True,lightsource=LightSource(315,35),rasterized=True))
 trimesh.PointCloud(rec.points,colors=rec.colors).export(out/'scene_filtered_world.ply');removed=~ev['keep'];trimesh.PointCloud(ev['raw_points'][removed],colors=raw['colors'][removed]).export(out/'scene_removed_uncertain_world.ply')
 rgb,depth,mask=cap.frame(cfg.reference_frame);fig,aa=plt.subplots(1,3,figsize=(14,4));aa[0].imshow(rgb);aa[1].imshow(depth,cmap='viridis',vmin=.2,vmax=1);aa[2].imshow(mask,cmap='tab10')
 for ax,title in zip(aa,['RGB / 000000','Mélység [m], 0.2–1.0','Automatikus instance maszk']):ax.set_title(title);ax.axis('off')
 export(fig,'01_input','Valódi referencia-képkocka a 0000 adatkészletből.')
 fig=plt.figure(figsize=(14,6))
 for i,(p,c,title) in enumerate([(raw['points'],raw['colors']/255,f"Nyers: {len(raw['points']):,} pont"),(rec.points,rec.colors/255,f'Szűrt: {len(rec.points):,} pont')]):
  ax=fig.add_subplot(1,2,i+1,projection='3d');cloud(ax,p,c);axes(ax,title)
 export(fig,'02_reconstruction','318 képkocka metrikus voxel-fúziója. Azonos ábrázolási régió, determinisztikus ritkítás a megjelenítéshez.')
 fig,aa=plt.subplots(1,3,figsize=(14,4));names=['SOR','Radius','DBSCAN','Eltávolított','Átcímkézett'];aa[0].bar(names,[audit[k] for k in ['sor_flagged','radius_flagged','dbscan_noise_or_small_cluster','removed_points','relabelled_points']],color=['#8fa6b5']*3+['#d57058','#329e84']);aa[0].tick_params(axis='x',rotation=25);aa[0].set_title('Szűrési audit (átfedő halmazok)');md=ev['mean_knn_distance']*1000;aa[1].hist(md,bins=60,range=(0,np.quantile(md,.995)),color='#5987ad');aa[1].set(xlabel='átlagos kNN távolság [mm]',ylabel='pontok');aa[2].hist(raw['counts'],bins=np.arange(1,32),color='#439a91');aa[2].axvline(cfg.occupancy_min_views,color='red',ls='--');aa[2].set(xlabel='képkockák száma / voxel (1–30)',ylabel='voxelek');fig.tight_layout();export(fig,'03_filtering','Az eltávolított bizonytalan pontok is megmaradnak az evidence NPZ-ben és a külön PLY-ben.')
 planes=json.loads((out/'obstacle_planes.json').read_text());fig=plt.figure(figsize=(13,5));ax=fig.add_subplot(121,projection='3d');cloud(ax,rec.points,[COLORS[i%len(COLORS)] for i in rec.labels]);axes(ax,'Szemantika és véges akadályfelületek')
 for p in sorted(out.glob('obstacle_plane_*.ply')):poly(ax,trimesh.load(p,force='mesh'),'#d79537',.16)
 ax=fig.add_subplot(122);ax.bar(range(len(planes)),[p['rms_m']*1000 for p in planes],color='#d79537');ax.set(xlabel='síkprizma index',ylabel='RMS [mm]',title='RANSAC felületillesztés');export(fig,'04_planes','Az asztal és a véges akadálysíkok megmaradnak az ütközési modellben.')
 fig=plt.figure(figsize=(15,5))
 for i,(name,g) in enumerate(gs.items()):
  ax=fig.add_subplot(1,3,i+1,projection='3d');parts=g.parts(np.eye(4),g.joint(.025));vv=np.vstack([p.mesh().vertices for p in parts])
  for p in parts:poly(ax,p.display_mesh(),'#277fa3' if p.pad else '#a4b0bb',1,np.eye(4))
  ax.auto_scale_xyz(*vv.T);ax.set_box_aspect(np.ptp(vv,axis=0));ax.set(title=name+' / 25 mm',xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]');ax.view_init(25,-65);ax.tick_params(labelsize=8)
 export(fig,'05_hardware','Gyártói CAD/URDF és stock ujjak; megjelenítéshez egyszerűsített hálók.')
 fig,aa=plt.subplots(1,2,figsize=(12,4))
 for name,g in gs.items():
  qq=np.linspace(*g.joint_limits,35);ww=np.array([g.width(q) for q in qq])*1000;zz=[g.contact_geometry(q)[1][2,3]*1000 for q in qq];aa[0].plot(np.linspace(0,1,len(qq)),ww,label=name);aa[1].plot(ww,zz,label=name)
 for ax in aa:ax.legend(fontsize=8);ax.grid(alpha=.2)
 aa[0].set(xlabel='normalizált hajtóízület',ylabel='CAD nyílás [mm]');aa[1].set(xlabel='CAD nyílás [mm]',ylabel='talp–kontaktközép [mm]');export(fig,'06_kinematics','Az RG kontaktközéppontja zárás közben is mozog; a talppózt kell rögzíteni.')
 fig,ax=plt.subplots(figsize=(12,5));left=np.zeros(len(GATES))
 for status,col in [('PASS','#299976'),('FAIL','#d66454'),('UNVERIFIED','#b6bdc4')]:
  values=[summary['gate_counts'][k][status] for k in GATES];ax.barh(GATES,values,left=left,color=col,label=status);left+=values
 ax.invert_yaxis();ax.legend();ax.set(xlabel='jelöltek száma',title=f"0000: {len(rows)} jelölt / {summary['status_counts']['PASS']} teljes geometriai PASS");export(fig,'07_gates','Egyetlen hibás vagy hiányzó kötelező kapu mellett sem keletkezhet teljes PASS.')
 fig,aa=plt.subplots(1,2,figsize=(13,4))
 for ax,gate,title in zip(aa,['environment',None],['Environment PASS','Teljes geometriai PASS']):
  values=np.array([[sum(r['hardware']==h.name and r['object_instance']==o.name and (r['checks'][gate]['status'] if gate else r['status'])=='PASS' for r in rows) for o in cap.objects] for h in specs]);ax.imshow(values,cmap='YlGnBu',vmin=0,vmax=max(1,values.max()));ax.set_xticks(range(len(cap.objects)),[o.name for o in cap.objects],rotation=20);ax.set_yticks(range(len(specs)),[h.name for h in specs]);ax.set_title(title)
  for (i,j),v in np.ndenumerate(values):ax.text(j,i,str(v),ha='center',va='center')
 fig.tight_layout();export(fig,'08_comparison','Az environment PASS egy részvizsgálat, nem sikeres megfogás.')
 normals=np.load(out/'target_normals.npz');fig=plt.figure(figsize=(12,5));ax=fig.add_subplot(121,projection='3d');ix=np.linspace(0,len(normals['points'])-1,min(400,len(normals['points']))).astype(int);p=transform(normals['points'][ix],D);n=normals['normals'][ix]@D[:3,:3].T;ax.quiver(*p.T,*n.T,length=.008,linewidth=.5);axes(ax,'Orientált PCA normálok');ax=fig.add_subplot(122);ax.hist(normals['local_rms_m']*1000,bins=40,color='#6a87ae');ax.set(xlabel='lokális sík RMS [mm]',ylabel='pontok',title='MLS diagnosztika');export(fig,'09_normals','MLS csak diagnosztika; a collision pontok nem mozdulnak el.')
 top=sorted(rows,key=rank_key,reverse=True)[:10];obj_display={o.name:o.mesh.simplify_quadric_decimation(face_count=4000) for o in cap.objects}
 for i,r in enumerate(top,1):
  fig=plt.figure(figsize=(12,6));ax=fig.add_subplot(121,projection='3d');cloud(ax,rec.points,rec.colors/255,10000);axes(ax,f"#{i:02d} {r['hardware']} / {r['object_instance']}\n{r['status']}");scene=trimesh.Scene()
  for o in cap.objects:
   m=obj_display[o.name].copy();m.apply_transform(C);scene.add_geometry(m,node_name=o.name)
  for p in gs[r['hardware']].parts(np.array(r['world_from_mount']),r['joint']):
   m=p.display_mesh();poly(ax,m,'#2594b1' if p.pad else '#a6b3c0',.85);m.apply_transform(C);scene.add_geometry(m,node_name=p.name)
  pp=transform(np.array(r['contacts_world_m']),D);ax.scatter(*pp.T,c='#22c689',s=25)
  witness=r['checks']['environment'].get('witness_world_m')
  if witness is not None:ax.scatter(*transform([witness],D)[0],c='#e53c38',marker='x',s=70)
  if r['trajectory']:
   p=transform(np.array([s['position_world_m'] for s in r['trajectory']['samples']]),D);ax.plot(*p.T,color='#d3455f',lw=2)
  ax=fig.add_subplot(122);ax.axis('off');lines=[f"{k}: {r['checks'][k]['status']}" for k in GATES]+['',f"Nyílás: {r['aperture_m']*1000:.2f} mm",'Dinamika: kihagyva','Robotkar / ismeretlen tér: UNVERIFIED','Végrehajtható: nem'];ax.text(.02,.98,'\n'.join(lines),va='top',fontfamily='monospace',fontsize=10);export(fig,f'grasp_{i:02d}',r['id']+' — rangsorolt jelölt, eredeti státusszal.');scene.export(modeldir/f'grasp_{i:02d}.glb')
 (modeldir/'README.txt').write_text('Camera coordinates, metres. Display-only decimated meshes, not collision geometry. Numerical checks use original CAD and documented conservative covers.')
 trajectories=[r for r in rows if r['trajectory']];fig,aa=plt.subplots(1,3,figsize=(14,4))
 for r in trajectories[:12]:
  ss=r['trajectory']['samples'];t=[s['time_s'] for s in ss];aa[0].plot(t,[s['position_camera_m'][2] for s in ss],label=r['hardware']+' '+r['trajectory']['status']);aa[1].plot(t,[np.linalg.norm(s['velocity_camera_m_s']) for s in ss]);aa[2].plot(t,[np.linalg.norm(s['acceleration_camera_m_s2']) for s in ss])
 for ax,y in zip(aa,['kamera Z [m]','TCP sebesség [m/s]','TCP gyorsulás [m/s²]']):ax.set(xlabel='idő [s]',ylabel=y);ax.grid(alpha=.2)
 aa[0].legend(fontsize=6);export(fig,'10_trajectories','A FAIL pályák diagnosztikai megközelítések; nem végrehajthatók.')
 # Offline 3D scene and all top candidates.
 (out/'plotly.min.js').write_text(get_plotlyjs(),encoding='utf-8');ii=np.linspace(0,len(rec.points)-1,14000).astype(int);p=transform(rec.points[ii],C);base=[go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='markers',marker=dict(size=1,color=['rgb(%d,%d,%d)'%tuple(c) for c in rec.colors[ii]],opacity=.4),name='RGB-D')]
 def meshtrace(m,T,name,color,visible=True):
  v=transform(m.vertices,T);f=m.faces;return go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],name=name,color=color,opacity=.9,visible=visible)
 for o in cap.objects:base.append(meshtrace(obj_display[o.name],C,o.name,COLORS[o.id%len(COLORS)]))
 traces=list(base);groups=[]
 for i,r in enumerate(top):
  group=[]
  for part in gs[r['hardware']].parts(np.array(r['world_from_mount']),r['joint']):group.append(len(traces));traces.append(meshtrace(part.display_mesh(),C,part.name,'#2c91b0' if part.pad else '#aab9c7',i==0))
  if r['trajectory']:
   p=np.array([s['position_camera_m'] for s in r['trajectory']['samples']]);group.append(len(traces));traces.append(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='lines',line=dict(color='#d74456',width=5),name='Approach '+r['trajectory']['status'],visible=i==0))
  groups.append(group)
 buttons=[dict(label=f"{i+1}. {r['hardware']} / {r['object_instance']} [{r['status']}]",method='update',args=[{'visible':[j<len(base) or j in groups[i] for j in range(len(traces))]},{'title':r['id']+' — '+r['status']+' — executable=false'}]) for i,r in enumerate(top)]
 f=go.Figure(traces);f.update_layout(title='0000 / top 10 / kamera-koordináták [m]',height=850,scene=dict(aspectmode='data',xaxis_title='Camera X [m]',yaxis_title='Camera Y [m]',zaxis_title='Camera Z [m]'),updatemenus=[dict(buttons=buttons,x=0,y=1.1)]);f.write_html(out/'scene_explorer.html',include_plotlyjs='plotly.min.js')
 replays=[]
 for h in specs:
  r=next((r for r in sorted(rows,key=rank_key,reverse=True) if r['hardware']==h.name and r['trajectory']),None)
  if r is None:continue
  g=gs[h.name];M=np.asarray(r['world_from_mount']);qo=r['open_joint'];_,B,_=g.contact_geometry(qo);local={}
  for part in g.parts(np.eye(4),qo):
   m=part.raw.copy();m.merge_vertices(digits_vertex=5)
   if len(m.faces)>400:m=m.simplify_quadric_decimation(face_count=400)
   local[part.name]=m
  states=[];ss=r['trajectory']['samples']
  for j in np.linspace(0,len(ss)-1,11).astype(int):
   A=M.copy();A[:3,3]=np.asarray(ss[j]['position_world_m'])-M[:3,:3]@B[:3,3];states.append((ss[j]['time_s'],A,qo,'approach'))
  states.extend((s['time_s'],M,s['joint'],'closure') for s in r['closure_trajectory']['samples'][::5]);tt=list(base);moving=list(range(len(tt),len(tt)+len(local)));frames=[]
  for j,(time,A,q,phase) in enumerate(states):
   fk=g.fk(q);updates=[]
   for name,m in local.items():
    v=transform(m.vertices,C@A@fk[name]);updates.append(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2]))
    if j==0:tt.append(meshtrace(m,C@A@fk[name],name,'#d8625a'))
   frames.append(go.Frame(name=str(j),data=updates,traces=moving,layout=dict(title=f'{h.name} / {phase} / t={time:.2f}s / {r["status"]} / executable=false')))
  f=go.Figure(tt,frames=frames);f.update_layout(title=h.name+' — diagnostic FAIL trajectory',height=850,scene=dict(aspectmode='data'),updatemenus=[dict(type='buttons',buttons=[dict(label='Lejátszás',method='animate',args=[None,dict(frame=dict(duration=150,redraw=True),transition=dict(duration=0),fromcurrent=True)]),dict(label='Szünet',method='animate',args=[[None],dict(mode='immediate',frame=dict(duration=0,redraw=False))])])],sliders=[dict(steps=[dict(label=str(j),method='animate',args=[[str(j)],dict(mode='immediate',frame=dict(duration=0,redraw=True),transition=dict(duration=0))]) for j in range(len(frames))])]);filename='replay_'+h.model+'.html';f.write_html(out/filename,include_plotlyjs='plotly.min.js');replays.append((filename,h.name))
 table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [i+1,r['hardware'],r['object_instance'],r['status'],r['checks']['environment']['status'],round(r['aperture_m']*1000,2)])+'</tr>' for i,r in enumerate(top));images=''.join(f'<figure><a href="figures/{n}.svg"><img src="figures/{n}.png"></a><figcaption>{html.escape(c)}</figcaption></figure>' for n,c in gallery)
 body=f'''<!doctype html><html lang="hu"><meta charset="utf-8"><title>Grasp Planner V3.0 / 0000</title><style>body{{font:16px system-ui;color:#183145;background:#f5f7fa;max-width:1180px;margin:40px auto;padding:20px}}h1{{font-size:36px}}.card,figure{{background:white;border-radius:12px;padding:24px;margin:24px 0}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #d7dfe6;text-align:left}}.warn{{border-left:5px solid #d08730}}a{{color:#087aa1}}</style><h1>Grasp Planner V3.0</h1><p>2FG7 / RG2 / RG6 · Valódi 0000 RGB-D teszt · Geometriai tervezés</p><div class="card"><b>{summary['frames_processed']} képkocka · {len(rows)} jelölt · {summary['status_counts']['PASS']} teljes geometriai PASS · {summary['gate_counts']['environment']['PASS']} environment PASS</b><p>{len(raw['points']):,} nyers → {len(rec.points):,} szűrt pont.</p><a href="scene_explorer.html">Interaktív 3D top 10</a> · <a href="all_candidates.csv">CSV eredmények</a> · <a href="summary.json">JSON összefoglaló</a></div><div class="card warn"><b>A státuszok hatóköre</b><p>A PASS csak a szűrt megfigyelt jelenetre és a dokumentált CAD-modellekre vonatkozó geometriai vizsgálat. Nem fizikai siker. Dinamika nincs értékelve; a robotkar-IK és az ismeretlen tér UNVERIFIED; minden terv executable=false.</p><p>A top 10 rangsorolt jelölt, nem garantált sikerlista. A nem zárt cél-CAD-ok konvex burkolata konzervatív, ezért valós megfogást is elutasíthat. A nulla teljes PASS nem bizonyítja a megfogás fizikai lehetetlenségét.</p></div><div class="card"><h2>Top 10</h2><table><tr><th>#</th><th>Megfogó</th><th>Tárgy</th><th>Összesített</th><th>Környezet</th><th>Nyílás mm</th></tr>{table}</table></div>{images}<div class="card"><h2>3D pályavisszajátszás</h2><p>A piros geometria elutasított diagnosztikai pálya. A lejátszás kinematikai megjelenítés, nem dinamikai szimuláció.</p>{''.join(f'<p><a href="{f}">{n} megközelítés és zárás</a></p>' for f,n in replays)}</div></html>'''
 (out/'report.html').write_text(body,encoding='utf-8')
 if (out/'expanded_search_probe.json').exists() and (out/'controls/summary.json').exists():supplement(out)

def supplement(out):
 out=Path(out);data=json.loads((out/'expanded_search_probe.json').read_text());x=np.arange(len(data));names=[r['object']+' / '+r['hardware'] for r in data];fig,aa=plt.subplots(1,2,figsize=(15,5))
 for key,label,col,dx in [('tested','Vizsgált jelölt','#97a7b7',-.15),('target_clear','Célgeometria tiszta','#499cb1',.15)]:aa[0].bar(x+dx,[r['counts'].get(key,0) for r in data],.3,label=label,color=col)
 for key,label,col,dx in [('environment_clear','Normál környezet','#2d9983',-.15),('bbox_environment_clear_unverified','BBox törlés — NEM elfogadott PASS','#d9953c',.15)]:aa[1].bar(x+dx,[r['counts'].get(key,0) for r in data],.3,label=label,color=col)
 for ax in aa:ax.set_xticks(x,names,rotation=70,ha='right',fontsize=8);ax.legend(fontsize=8);ax.set_ylabel('jelöltek száma')
 aa[0].set_title('Kibővített keresés: 6320 jelölt (átfedő kísérlet)');aa[1].set_title('Környezeti eredmény a célgeometriát teljesítő jelölteken');fig.tight_layout()
 for ext in ['png','svg']:fig.savefig(out/'figures'/('11_expanded_search.'+ext),dpi=190,bbox_inches='tight')
 plt.close(fig)
 control=json.loads((out/'controls/summary.json').read_text());fig,ax=plt.subplots(figsize=(10,3));ax.barh([r['id'] for r in control],[1]*len(control),color=['#2e9f81' if r['status']==r['expected'] else '#d66351' for r in control]);ax.set_xlim(0,1.2);ax.set_xticks([]);ax.set_title('Szintetikus geometriai kontrollok — helyes döntés, nem fizikai siker')
 for i,r in enumerate(control):ax.text(.05,i,f"Várt: {r['expected']} / kapott: {r['status']}",va='center',color='white')
 for ext in ['png','svg']:fig.savefig(out/'figures'/('12_controls.'+ext),dpi=190,bbox_inches='tight')
 plt.close(fig)
 addition='<div class="card"><h2>Kibővített keresés és kontrollok</h2><p>A kibővített kísérlet a 2026-10-29-i tényleges futás megőrzött mérési eredménye. A fő kereséssel átfed; nem 6320 további független megfogás. A részkapuk nem igazolnak teljes PASS-t.</p><img src="figures/11_expanded_search.png"><img src="figures/12_controls.png"><p>A szintetikus kontrollok külön kísérletek, nem a 0000 adaton elért grasp-sikerek.</p></div>'
 p=out/'report.html';p.write_text(p.read_text(encoding='utf-8').replace('</html>',addition+'</html>'),encoding='utf-8')
