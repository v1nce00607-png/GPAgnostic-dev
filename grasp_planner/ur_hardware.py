"""Manufacturer URDF forward kinematics. Geometry-only; no force or mass assumptions."""
from pathlib import Path
from functools import lru_cache
from typing import Literal
import json,xml.etree.ElementTree as ET
import numpy as np,trimesh
from scipy.spatial.transform import Rotation
from scipy.optimize import brentq
from pydantic import Field
from .models import Strict,pose,transform,check,digest
from .fcl_persistent import PosedMesh
ASSETS=Path(__file__).parent/'assets'

class HardwareSpec(Strict):
    model: Literal['2fg7','rg2','rg6']='2fg7'
    name: str = Field('OnRobot_2FG7',pattern=r'^[A-Za-z0-9_-]+$')
    contact_inset_m: float = Field(.002,ge=.002,le=.008)
    opening_extra_m: float = Field(.012,gt=0,le=.04)
    closing_speed_m_s: float = Field(.02,gt=0,le=.03)
    # Stock inward silicone 2FG7, stock inward standard RG fingertips.

def origin(e):
    if e is None:return np.eye(4)
    return pose(Rotation.from_euler('xyz',np.fromstring(e.get('rpy','0 0 0'),sep=' ')).as_matrix(),np.fromstring(e.get('xyz','0 0 0'),sep=' '))

class Part:
    def __init__(self,name,hull,raw,T,pad=False,solid=None,footprint=None):
        # solid: OPT7 audited watertight reconstruction of a non-watertight OEM link (same local frame).
        self.name=name;self.local=hull;self.raw=raw;self.T=T;self.pad=pad;self.solid=solid;self.footprint=footprint
        self.side=1 if name.startswith('right') else (-1 if name.startswith('left') else 0)
    def on_pad_face(self,point,margin):
        """True if a world point lies within the inner pad-face rectangle (+margin on every side) in the link frame."""
        ends=getattr(self,'sweep_ends',None)
        if ends is not None:return any(e.on_pad_face(point,margin+self.sweep_slack) for e in ends)
        f=self.footprint
        if f is None:return False
        x=(np.linalg.inv(self.T)@np.r_[np.asarray(point,dtype=float),1.])[:3]
        return abs(x@f['ey']-f['u0'])<=f['hy']+margin+1e-9 and abs(x@f['ez']-f['v0'])<=f['hz']+margin+1e-9
    def mesh(self):
        m=self.local.copy();m.apply_transform(self.T);return m
    def exact_solid(self):
        """Watertight local-frame mesh usable for exact narrow-phase tests, or None (convex cover only)."""
        if self.solid is not None:return self.solid
        return self.raw if self.raw.is_watertight else None
    def collision_mesh(self):
        """Exact transformed solid when one exists, otherwise the convex cover (unchanged behaviour)."""
        ex=self.exact_solid()
        if ex is None:return self.mesh()
        m=ex.copy();m.apply_transform(self.T);return m
    def collision_posed(self):
        """Same geometry as collision_mesh() but un-copied (local mesh + pose); see fcl_persistent."""
        ex=self.exact_solid()
        return PosedMesh(ex if ex is not None else self.local,self.T)
    def visual_mesh(self):
        m=self.raw.copy();m.apply_transform(self.T);return m
    def display_mesh(self):
        if not hasattr(self.raw,'_display'):
            m=self.raw.copy();m.merge_vertices(digits_vertex=5)
            if len(m.faces)>5000:m=m.simplify_quadric_decimation(face_count=5000)
            self.raw._display=m
        m=self.raw._display.copy();m.apply_transform(self.T);return m

@lru_cache(maxsize=3)
def resources(model):
    root=ASSETS/('onrobot_'+model);xml=ET.parse(root/'model.urdf').getroot();links={}
    for l in xml.findall('link'):
        collisions=l.findall('collision')
        if not collisions:continue
        meshes=[]
        for c in collisions:
            node=c.find('geometry/mesh')
            if node is None:raise ValueError('Non-mesh geometry in manufacturer asset')
            path=root/'meshes'/Path(node.get('filename')).name
            m=trimesh.load(path,force='mesh',process=True)
            scale=node.get('scale')
            if scale:
                v=np.fromstring(scale,sep=' ')
                if v.shape!=(3,) or np.any(v<=0) or not np.isfinite(v).all():
                    raise ValueError(f'Invalid OEM mesh scale: {scale}')
                m.apply_scale(v)
            m.apply_transform(origin(c.find('origin')))
            meshes.append(m)
        # URDF permits >1 collision block per link. The previous loader
        # silently used the first, potentially omitting solid parts ('holes').
        m=trimesh.util.concatenate(meshes) if len(meshes)>1 else meshes[0]
        links[l.get('name')]=(m,m.convex_hull)
    joints=[]
    for j in xml.findall('joint'):
        a=j.find('axis');mi=j.find('mimic');lim=j.find('limit')
        joints.append(dict(name=j.get('name'),kind=j.get('type'),parent=j.find('parent').get('link'),child=j.find('child').get('link'),T=origin(j.find('origin')),axis=np.fromstring(a.get('xyz'),sep=' ') if a is not None else np.zeros(3),mimic=mi.attrib if mi is not None else None,limit=lim.attrib if lim is not None else {}))
    srdf=ET.parse(root/'srdf'/('onrobot_'+model+'.srdf')).getroot()
    disabled={frozenset([x.get('link1'),x.get('link2')]) for x in srdf.findall('disable_collisions')}
    disabled|={frozenset([j['parent'],j['child']]) for j in joints}
    return links,joints,disabled

class CADGripper:
    def __init__(self,spec):
        self.spec=spec;self.links,self.joints,self.disabled=resources(spec.model)
        self.driver='finger_stroke' if spec.model=='2fg7' else 'finger_joint'
        j=next(j for j in self.joints if j['name']==self.driver);self.joint_limits=(float(j['limit']['lower']),float(j['limit']['upper']))
        # Fit exact sinusoidal/linear aperture relationship from manufacturer FK.
        xs=np.linspace(*self.joint_limits,9);gaps=np.array([self.contact_geometry(t)[0] for t in xs])
        A=np.column_stack([xs,np.ones(len(xs))]) if spec.model=='2fg7' else np.column_stack([np.sin(xs),np.cos(xs),np.ones(len(xs))])
        self.coeff=np.linalg.lstsq(A,gaps,rcond=None)[0];self.fit_error=float(max(abs(A@self.coeff-gaps)))
        self.width_limits=(max(.001,float(min(gaps))),float(max(gaps)))
        if self.fit_error>1e-5:raise ValueError('Aperture fit is not exact enough')
        self._self={};self.solids={};self.solid_audit={}
    def fk(self,q):
        values={self.driver:float(q),'grip_stroke':0.};T={'base_link':np.eye(4)};pending=list(self.joints)
        while pending:
            advanced=False
            for j in pending[:]:
                if j['parent'] not in T:continue
                mi=j['mimic'];v=(float(mi.get('multiplier',1))*values[mi['joint']]+float(mi.get('offset',0))) if mi else values.get(j['name'],0.)
                values[j['name']]=v;M=np.eye(4)
                if j['kind']=='prismatic':M[:3,3]=j['axis']*v
                elif j['kind'] in ['continuous','revolute']:M[:3,:3]=Rotation.from_rotvec(j['axis']*v).as_matrix()
                T[j['child']]=T[j['parent']]@j['T']@M;pending.remove(j);advanced=True
            if not advanced:raise ValueError('Unresolved URDF graph')
        return T
    def contact_geometry(self,q):
        fk=self.fk(q);tipnames=[n for n in self.links if ('fingertip' in n or 'finger_tip' in n)]
        tips=[]
        for n in tipnames:
            m=self.links[n][0];p=transform(m.vertices,fk[n]);tips.append((n,p))
        right=next(p for n,p in tips if n.startswith('right'));left=next(p for n,p in tips if n.startswith('left'))
        # The OEM base uses X jaw closing, Z tool approach for all three products.
        z=min(right[:,2].max(),left[:,2].max())-self.spec.contact_inset_m
        y=(right[:,1].mean()+left[:,1].mean())/2
        surfaces=[]
        for n,p in tips:
            mesh=self.links[n][0].copy();mesh.apply_transform(fk[n]);sgn=1 if p[:,0].mean()>0 else -1
            hits,_,_=mesh.ray.intersects_location([[sgn*.4,y,z]],[[-sgn,0,0]],multiple_hits=True)
            if not len(hits):raise ValueError(f'No fingertip contact ray: {self.spec.model} q={q}, {n}')
            surfaces.append(hits[np.argmin(sgn*hits[:,0])])
        surfaces=np.array(surfaces);width=float(surfaces[0,0]-surfaces[1,0]);c=surfaces.mean(0)
        return width,pose(np.eye(3),c),surfaces
    def pad_footprints(self):
        """Per fingertip link: the flat inner OEM pad face as a rectangle in the LINK frame.

        Returns {link: dict(ey,ez,u0,v0,hy,hz)}: unit tangent axes (images of the grasp Y = lateral and Z = along
        the finger), rectangle centre (u0,v0) and half extents, measured at the mid-range joint (rigid in the link).
        """
        if getattr(self,'_footprints',None) is None:
            fk=self.fk(self.joint(sum(self.width_limits)/2));out={}
            for n in self.links:
                if not('fingertip' in n or 'finger_tip' in n):continue
                v=self.links[n][0].vertices;vg=transform(v,fk[n]);inner=(1 if vg[:,0].mean()>0 else -1)*vg[:,0]
                face=v[inner<inner.min()+.0004];R=fk[n][:3,:3];ey=R.T@[0,1,0.];ez=R.T@[0,0,1.]
                u=face@ey;w=face@ez
                out[n]=dict(ey=ey,ez=ez,u0=float(u.max()+u.min())/2,v0=float(w.max()+w.min())/2,hy=float(u.max()-u.min())/2,hz=float(w.max()-w.min())/2)
            self._footprints=out
        return self._footprints
    def width(self,q):
        if self.spec.model=='2fg7':return float(self.coeff@[q,1.])
        return float(self.coeff@[np.sin(q),np.cos(q),1.])
    def joint(self,width):
        if not self.width_limits[0]-1e-8<=width<=self.width_limits[1]+1e-8:raise ValueError('Width outside stock geometry')
        lo,hi=self.joint_limits
        if abs(self.width(lo)-width)<1e-8:return lo
        if abs(self.width(hi)-width)<1e-8:return hi
        return float(brentq(lambda q:self.width(q)-width,lo,hi,xtol=1e-12))
    def parts(self,M,q,pad_tolerance=0.):
        fk=self.fk(q);out=[]
        for n,(raw,hull) in self.links.items():
            P=M@fk[n];pad='fingertip' in n or 'finger_tip' in n
            if pad and pad_tolerance:
                # Contact tolerance in the jaw-closing direction only.
                side=np.sign((fk[n]@np.r_[raw.centroid,1.])[0]);P=P.copy();P[:3,3]+=M[:3,0]*side*pad_tolerance
            out.append(Part(n,hull,raw,P,pad,solid=self.solids.get(n),footprint=self.pad_footprints().get(n) if pad else None))
        return out
    def use_repaired_links(self,pitch_m=.0003,max_faces=40000,cache_dir=None,links=None):
        """OPT7: replace the convex-cover collision of every NON-watertight OEM link by an audited
        watertight reconstruction (voxel fill, same as the target model). Watertight links keep
        their original mesh. A link that cannot be repaired keeps its convex cover (recorded)."""
        from .cad_repair import repair_to_solid,audit_repair,gripper_link_solid
        for n,(raw,hull) in self.links.items():
            if raw.is_watertight or (links is not None and n not in links):continue
            try:
                sol,audit=gripper_link_solid(self.spec.model,n,raw,pitch_m,max_faces,cache_dir)
                audit['hull_volume_m3']=float(hull.volume);self.solids[n]=sol;self.solid_audit[n]=audit
            except ValueError as exc:
                self.solid_audit[n]=dict(repaired=False,reason=str(exc)[:200])
        return self.solid_audit
    def self_collision(self,q):
        key=round(float(q),8)
        if key in self._self:return self._self[key]
        manager=trimesh.collision.CollisionManager()
        for p in self.parts(np.eye(4),q):
            if p.side:manager.add_object(p.name,p.visual_mesh())
        _,pairs=manager.in_collision_internal(return_names=True)
        side=lambda n:1 if n.startswith('right') else (-1 if n.startswith('left') else 0)
        bad=sorted([sorted(pair) for pair in pairs if side(tuple(pair)[0])*side(tuple(pair)[1])==-1 and frozenset(pair) not in self.disabled])
        r=check(not bad,pairs=bad,method='original CAD opposite-jaw collision; stock internal mechanism follows OEM joint limits',stock_internal_overlap_pairs=[sorted(p) for p in pairs if sorted(p) not in bad]);self._self[key]=r;return r
    def evidence(self):
        root=ASSETS/('onrobot_'+self.spec.model)
        return dict(manufacturer='OnRobot',model=self.spec.model.upper(),specification=self.spec.model_dump(),source=json.loads((ASSETS/'source.json').read_text()),aperture_limits_m=self.width_limits,joint_limits=self.joint_limits,joint_units='m' if self.spec.model=='2fg7' else 'rad',kinematic_fit_error_m=self.fit_error,meshes={p.name:digest(p) for p in (root/'meshes').glob('*.STL')},collision_model=('OPT7: non-watertight links use an audited watertight reconstruction for static collision, watertight links their original mesh; sweeps and unrepaired links use convex covers; original triangles for self collision' if self.solids else 'all URDF per-link collision components, per-link convex covers; original triangles for self collision'),collision_model_repair=self.solid_audit,dynamics_evaluated=False,robot_arm_included=False)
