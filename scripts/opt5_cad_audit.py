"""Audit OnRobot 2FG7 manufacturer URDF visual/collision meshes in-place.

Read-only: never edits source STL, URDF or collision thresholds. Compare the
actual visual mesh and the mesh used by this planner. An open surface is not
necessarily a physical hole; report topology without making that inference.
"""
from pathlib import Path
import argparse, hashlib, json, xml.etree.ElementTree as ET
import numpy as np
import trimesh
from grasp_planner.models import native
from grasp_planner.ur_hardware import ASSETS, CADGripper, HardwareSpec, origin


def _load_mesh(root, node):
    filename=node.get('filename')
    path=root/'meshes'/Path(filename).name
    if not path.is_file():
        raise FileNotFoundError(f'URDF mesh missing: {filename} -> {path}')
    m=trimesh.load(path,force='mesh',process=True)
    scale=node.get('scale')
    if scale:
        v=np.fromstring(scale,sep=' ')
        if v.shape!=(3,) or not np.isfinite(v).all() or np.any(v<=0):
            raise ValueError(f'Invalid URDF mesh scale {filename}: {scale}')
        m.apply_scale(v)
    return m,path


def boundary_edges(mesh):
    # Unique triangle edges incident to only 1 face indicate open boundaries.
    if not len(mesh.faces):return 0
    counts=np.bincount(mesh.edges_unique_inverse,minlength=len(mesh.edges_unique))
    return int(np.count_nonzero(counts==1))


def audit_model(model='2fg7',out='opt5_cad_audit',html=True):
    root=ASSETS/('onrobot_'+model)
    source=root/'model.urdf'
    if not source.is_file():raise FileNotFoundError(f'Actual manufacturer URDF not found: {source}')
    doc=ET.parse(source).getroot()
    gripper=CADGripper(HardwareSpec(model=model,name='OnRobot_2FG7' if model=='2fg7' else 'OnRobot_'+model.upper()))
    mid=sum(gripper.joint_limits)*.5
    poses=gripper.fk(mid)
    rows=[];assemblies={'visual':[],'collision':[]};warnings=[]
    visual_sources_missing=[];visual_sources_unreadable=[];visual_loaded=0;collision_loaded=0
    for link in doc.findall('link'):
        name=link.get('name')
        item={'link':name,'visual_count':len(link.findall('visual')),
              'collision_count':len(link.findall('collision')),'geometries':[]}
        for kind in ('visual','collision'):
            chunks=[]
            for section in link.findall(kind):
                mesh_node=section.find('geometry/mesh')
                if mesh_node is None:
                    if kind=='collision':
                        raise ValueError(f'{name}: non-mesh collision geometry cannot be audited by this mesh-only loader')
                    item['geometries'].append({'kind':kind,'unsupported_geometry':'non_mesh'})
                    warnings.append(f'{name}: non-mesh {kind} geometry, separate review needed')
                    continue
                filename=mesh_node.get('filename')
                if not filename:
                    raise ValueError(f'{name}: {kind} mesh has no filename in OEM URDF')
                expected=root/'meshes'/Path(filename).name
                if kind=='visual' and not expected.is_file():
                    # OEM packages sometimes ship collision STL meshes but not
                    # the DAE visual meshes referenced by their URDF. This is
                    # an INCOMPLETE VISUAL AUDIT, never proof that the physical
                    # model has an opening. The collision meshes remain strict.
                    entry={'link':name,'uri':filename,'expected':str(expected)}
                    visual_sources_missing.append(entry)
                    item['geometries'].append({'kind':kind,'status':'MISSING_VISUAL_FILE',**entry})
                    warnings.append(f'{name}: visual mesh MISSING ({filename}); visual model incomplete, collision geometry still checked')
                    continue
                try:
                    mesh,path=_load_mesh(root,mesh_node)
                    mesh.apply_transform(origin(section.find('origin')))
                except Exception as exc:
                    if kind=='collision':
                        raise  # Never permit a broken/missing collision asset.
                    entry={'link':name,'uri':filename,'expected':str(expected),
                           'error':f'{type(exc).__name__}: {exc}'}
                    visual_sources_unreadable.append(entry)
                    item['geometries'].append({'kind':kind,'status':'UNREADABLE_VISUAL_FILE',**entry})
                    warnings.append(f'{name}: visual mesh COULD NOT LOAD ({filename}); {entry["error"]}; collision still checked')
                    continue
                if kind=='visual':visual_loaded+=1
                else:collision_loaded+=1
                sha=hashlib.sha256(path.read_bytes()).hexdigest()
                details={'kind':kind,'mesh':path.name,'sha256':sha,'faces':len(mesh.faces),
                         'vertices':len(mesh.vertices),'watertight':bool(mesh.is_watertight),
                         'boundary_edges':boundary_edges(mesh),
                         'components':len(mesh.split(only_watertight=False)),
                         'bounds_link_m':mesh.bounds.tolist()}
                item['geometries'].append(details)
                if not mesh.is_watertight:
                    warnings.append(f'{name} {kind} {path.name}: open mesh; {details["boundary_edges"]} boundary edges (not necessarily a physical hole)')
                if name in poses:
                    m=mesh.copy();m.apply_transform(poses[name]);chunks.append(m)
            if chunks:
                assemblies[kind].append((name,trimesh.util.concatenate(chunks)))
        if item['collision_count']>1:
            warnings.append(f'{name}: {item["collision_count"]} collision tags; OPT5 combines all of them; review link assembly')
        if item['visual_count'] and not item['collision_count']:
            warnings.append(f'{name}: visual mesh present but no collision geometry; this link is not represented in planner collisions')
        if item['collision_count'] and not item['visual_count']:
            warnings.append(f'{name}: collision present but no visual geometry; 3D viewer may appear incomplete')
        rows.append(item)
    visual_declared=sum(item['visual_count'] for item in rows)
    visual_status=('COMPLETE' if visual_loaded==visual_declared and visual_declared else
                   'INCOMPLETE' if visual_declared else 'NOT_DECLARED')
    result={'model':model,'urdf':str(source),'urdf_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'mid_joint':mid,'links':rows,'warnings':warnings,
            'visual_status':visual_status,'visual_declared':visual_declared,
            'visual_loaded':visual_loaded,'visual_sources_missing':visual_sources_missing,
            'visual_sources_unreadable':visual_sources_unreadable,
            'collision_loaded':collision_loaded,'collision_status':'COMPLETE',
            'interpretation':'Topology of available source meshes only. Missing/unreadable OEM visual meshes are NOT assessed for holes. Planner uses collision meshes; missing/unreadable collision geometry is fatal.'}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'OPT5_2FG7_CAD_audit.json').write_text(json.dumps(native(result),indent=2),encoding='utf-8')
    if html:
        import plotly.graph_objects as go
        fig=go.Figure();kind_trace=[]
        for kind in ('visual','collision'):
            for name,mesh in assemblies[kind]:
                # Never hide/remove geometry silently: simplify display only.
                rendered=mesh
                if len(mesh.faces)>12000:
                    try:rendered=mesh.simplify_quadric_decimation(face_count=12000)
                    except Exception:rendered=mesh
                xyz=rendered.vertices;faces=rendered.faces
                fig.add_trace(go.Mesh3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],
                    name=f'{kind}: {name}',legendgroup=kind,showlegend=True,opacity=.85 if kind=='visual' else .42,
                    color='#4c92cd' if kind=='visual' else '#ef8a62',visible=(kind=='collision'),
                    flatshading=True,hoverinfo='name'))
                kind_trace.append(kind)
        choices=[('Collision only',{'collision'})]
        if assemblies['visual']:
            choices += [('Visual only',{'visual'}),('Overlay',{'collision','visual'})]
        title='OPT5 2FG7 — original visual / collision URDF meshes (joint midpoint)'
        note=('INCOMPLETE OEM VISUAL MODEL: missing or unreadable DAE files. '
              'Showing COLLISION meshes, not evidence of a physical hole.')
        if visual_status!='COMPLETE': title+=' | VISUAL INCOMPLETE'
        fig.update_layout(title=title,
            scene=dict(aspectmode='data'),width=1300,height=850,margin=dict(l=0,r=0,b=0,t=90),
            annotations=([dict(text=note,showarrow=False,xref='paper',yref='paper',
                               x=.5,y=1.01,font=dict(color='#ac2d20',size=12))]
                         if visual_status!='COMPLETE' else []),
            updatemenus=[dict(buttons=[dict(label=label,method='update',args=[{'visible':[kind in shown for kind in kind_trace]}])
                for label,shown in choices],x=.01,y=1.07)])
        fig.write_html(str(out/'OPT5_2FG7_visual_collision.html'),include_plotlyjs=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default='2fg7',choices=['2fg7','rg2','rg6']);p.add_argument('--output',required=True)
    a=p.parse_args();d=audit_model(a.model,a.output)
    print(f'2FG7 CAD audit: {len(d["links"])} URDF links, {len(d["warnings"])} review findings')
    print(f'  Collision: {d["collision_status"]} ({d["collision_loaded"]} meshes); visual: {d["visual_status"]} ({d["visual_loaded"]}/{d["visual_declared"]} meshes)')
    for w in d['warnings']:print('  REVIEW:',w)

if __name__=='__main__':main()
