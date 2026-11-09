"""Conservative convex CAD / occupancy / finite-plane collision checks."""
import numpy as np,trimesh
from scipy.spatial import cKDTree
from .models import check,pose,transform
from .ur_hardware import Part
from .fcl_persistent import PosedMesh,PersistentCollisionManager,closest_distance

def swept(a,b,curvature_bound_m=0.):
    # Convex hull of endpoint covers; optional interval bound encloses curved FK motion.
    v=np.vstack([a.mesh().vertices,b.mesh().vertices])
    if curvature_bound_m>0:
        v=np.vstack([v+curvature_bound_m*np.array([x,y,z]) for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    hull=trimesh.convex.convex_hull(v)
    out=Part(a.name,hull,hull,np.eye(4),a.pad)
    # A swept pad keeps its end poses so the pad-face rectangle test can still be applied (plus the motion bound).
    out.sweep_ends=(a,b);out.sweep_slack=float(curvature_bound_m)
    return out

def local_contact_witnesses(witnesses,contact_points,depth_limit_m,radius_m,region=None):
    """Validate FCL contact witnesses against explicitly proposed CAD contacts.

    Surface intersections on a non-watertight target are not, by themselves,
    proof of solid penetration or non-penetration. This only certifies that an
    observed *collision witness* is shallow and localized at the allowed pad.
    """
    anchors=np.asarray(contact_points,dtype=float)
    if anchors.shape!=(2,3) or not np.isfinite(anchors).all():return False
    if not witnesses:return False
    for w in witnesses:
        point=np.asarray(getattr(w,'point',np.nan),dtype=float)
        depth=float(getattr(w,'depth',np.inf))
        if point.shape!=(3,) or not np.isfinite(point).all():return False
        if not np.isfinite(depth) or abs(depth)>depth_limit_m+1e-8:return False
        if region is not None:
            if not region(point):return False
        elif min(np.linalg.norm(anchors-point,axis=1))>radius_m+1e-8:return False
    return True

def witness_summary(witnesses):
    """OPT7 diagnostic only: deepest FCL witness. Never changes a verdict."""
    best=None;n=0
    for w in (witnesses or []):
        try:
            depth=float(getattr(w,'depth',np.nan));point=np.asarray(getattr(w,'point',np.nan),dtype=float)
        except (TypeError,ValueError):continue
        if not np.isfinite(depth) or point.shape!=(3,) or not np.isfinite(point).all():continue
        n+=1
        if best is None or abs(depth)>abs(best[0]):best=(depth,point)
    if best is None:return dict(penetration_depth_m=None,penetration_witness_world_m=None,penetration_witness_count=0)
    return dict(penetration_depth_m=abs(best[0]),penetration_witness_world_m=[float(x) for x in best[1]],penetration_witness_count=n)

class Scene:
    def __init__(self,cap,rec,obj,cfg,planes=(),bbox_erase=False):
        self.cfg=cfg;self.plane=rec.plane;self.obj=obj
        keep=rec.labels!=obj.id
        if bbox_erase:
            from .cleaning import bbox_mask
            keep&=~bbox_mask(rec.points,obj.mesh.bounds,cfg.bbox_scale)
        self.target_shell_removed=0
        if getattr(cfg,'environment_ignore_target_shell_m',0.)>0:
            r=float(cfg.environment_ignore_target_shell_m);P=rec.points;lo,hi=obj.mesh.bounds
            cand=np.flatnonzero(keep&(rec.labels==0)&np.all((P>=lo-r)&(P<=hi+r),axis=1))
            if len(cand):
                near=trimesh.proximity.closest_point(obj.mesh,P[cand])[1]<=r
                keep=keep.copy();keep[cand[near]]=False;self.target_shell_removed=int(near.sum())
        self.table_band_removed=0
        if getattr(cfg,'environment_ignore_table_band_m',0.)>0:
            hh=rec.points@np.asarray(rec.plane[:3],float)+float(rec.plane[3])
            band=keep&(rec.labels==0)&(hh<=float(cfg.environment_ignore_table_band_m))
            self.table_band_removed=int(band.sum());keep=keep&~band
        self.points=rec.points[keep];self.labels=rec.labels[keep];self.centers=(np.floor(self.points/cfg.voxel_m)+.5)*cfg.voxel_m;self.tree=cKDTree(self.points)
        self.persistent_fcl=bool(getattr(cfg,'fcl_persistent_objects',True))
        self.other=PersistentCollisionManager()
        for o in cap.objects:
            if o.id!=obj.id:
                # Preserve concavities when the original CAD defines a closed
                # solid; never infer solids from open or non-manifold surfaces.
                other_mesh=o.mesh if cfg.prefer_exact_watertight_meshes and o.mesh.is_watertight else o.mesh.convex_hull
                self.other.add_object(o.name,other_mesh)
        for i,m in enumerate(planes):self.other.add_object('obstacle_plane_'+str(i),m)
        self.target=PersistentCollisionManager()
        self.target_model_audit=None
        if cfg.target_body_model in ('repaired_solid','convex_decomposition'):
            # OPT7: audited watertight reconstruction (and optional convex parts)
            # of the open CAD. Conditional on that reconstruction; the ORIGINAL
            # mesh stays the reference for pad-contact anchors.
            from .cad_repair import target_solids
            model=target_solids(obj,cfg)
            self.target_solids=model['solids'];self.target_model_audit=model['audit']
            self.target_hull=(self.target_solids[0] if len(self.target_solids)==1
                else trimesh.convex.convex_hull(np.vstack([x.vertices for x in self.target_solids])))
            self.target_model=('repaired_solid_CAD_voxel' if cfg.target_body_model=='repaired_solid'
                else f'convex_decomposition_CoACD_{len(self.target_solids)}_parts')
            for k,part in enumerate(self.target_solids):self.target.add_object(f'{obj.name}#{k}',part)
        else:
            self.target_hull=(obj.mesh if cfg.prefer_exact_watertight_meshes and obj.mesh.is_watertight else obj.mesh.convex_hull)
            self.target_solids=[self.target_hull]
            self.target_model=('exact_watertight_CAD' if self.target_hull is obj.mesh else 'conservative_convex_envelope')
            self.target.add_object(obj.name,self.target_hull)
        self.target_original=PersistentCollisionManager()
        self.target_original.add_object(obj.name,obj.mesh)

    def _cmesh(self,p):
        """Collision geometry of a part: un-copied posed mesh (persistent FCL) or the legacy copied mesh."""
        return p.collision_posed() if self.persistent_fcl else p.collision_mesh()

    def target_depths(self,parts):
        """Per colliding link: dict(link,pad,depth(max),witnesses=[(point,depth)]).

        Diagnostic/objective helper only (never a verdict): Scene.target_clear stops at
        the first failing link; this lists every colliding link with ALL FCL witnesses.
        """
        ablated=(self.cfg.target_body_model=='original_surface_ablation' and not self.obj.mesh.is_watertight)
        manager=self.target_original if ablated else self.target
        bounds=self.obj.mesh.bounds if ablated else self.target_hull.bounds
        rows=[]
        for p in parts:
            m=p.visual_mesh() if ablated else self._cmesh(p)
            if np.any(m.bounds[0]>bounds[1]) or np.any(m.bounds[1]<bounds[0]):continue
            hit,w=manager.in_collision_single(m,return_data=True)
            if not hit:continue
            pts=[]
            for x in (w or []):
                d=float(getattr(x,'depth',np.nan));pt=np.asarray(getattr(x,'point',np.nan),dtype=float)
                if np.isfinite(d) and pt.shape==(3,) and np.isfinite(pt).all():pts.append((pt,abs(d)))
            rows.append(dict(link=p.name,pad=bool(p.pad),depth=witness_summary(w)['penetration_depth_m'],witnesses=pts))
        return rows

    def _inside_target(self,m):
        # Same test as before for one solid; any part for a decomposition.
        for solid in self.target_solids:
            if solid.contains(m.vertices).any() or m.contains(solid.vertices[:1]).any():return True
        return False

    def pad_region(self,p):
        """Predicate for 'witness is inside the allowed shallow-contact region of pad p', or None (anchor-disc rule)."""
        if self.cfg.target_contact_patch_mode!='pad_face_rect':return None
        if p.footprint is None and getattr(p,'sweep_ends',None) is None:return None
        margin=self.cfg.target_contact_patch_margin_m
        return lambda pt:p.on_pad_face(pt,margin)

    def _allowed_pad_contact(self,p,m,contacts):
        cfg=self.cfg
        if (cfg.target_contact_policy!='local_pad_witness' or not p.pad
                or contacts is None):return False
        anchors=np.asarray(contacts,dtype=float)
        if anchors.shape!=(2,3) or not np.isfinite(anchors).all():return False
        # The proposed contacts must themselves lie on the ORIGINAL target
        # triangles. A convex hull face is not an admissible contact surface.
        try:
            _,distance,_=trimesh.proximity.closest_point(self.obj.mesh,anchors)
        except (ValueError,RuntimeError):return False
        if not np.isfinite(distance).all() or np.max(distance)>cfg.target_contact_surface_distance_m:
            return False
        # The same CAD contacts must also be reachable by the *actual OEM
        # fingertip surface*; target-near contact alone is not sufficient.
        try:
            tip_dist=closest_distance(m,anchors)
        except (ValueError,RuntimeError):return False
        if not np.isfinite(tip_dist).all() or float(np.min(tip_dist))>(
                cfg.target_contact_surface_distance_m+cfg.target_penetration_tolerance_m):
            return False
        try:
            collides,witnesses=self.target.in_collision_single(m,return_data=True)
        except (ValueError,RuntimeError):return False
        if not collides:return False
        region=self.pad_region(p)
        if not local_contact_witnesses(witnesses,anchors,
            cfg.target_penetration_tolerance_m,cfg.target_contact_patch_radius_m,region):return False
        # Reject any original CAD intersection outside the same shallow patch.
        try:
            original_hit,original_witness=self.target_original.in_collision_single(m,return_data=True)
        except (ValueError,RuntimeError):return False
        if original_hit and not local_contact_witnesses(original_witness,anchors,
            cfg.target_penetration_tolerance_m,cfg.target_contact_patch_radius_m,region):return False
        return True

    def target_clear(self,parts,contacts=None):
        """Target-body validation with explicitly separated open-shell ablation.

        The conservative default is unchanged. The opt-in original-surface
        branch checks OEM triangles against ORIGINAL object triangles; it
        cannot infer unseen solid occupancy in a nonwatertight object and
        therefore labels every accepted pose as conditional rather than a
        watertight solid collision certificate.
        """
        # Preserve normal watertight target handling without approximation.
        surface_ablated=(self.cfg.target_body_model=='original_surface_ablation'
                         and not self.obj.mesh.is_watertight)
        if not surface_ablated:
            accepted=[]
            for p in parts:
                m=self._cmesh(p)
                if np.any(m.bounds[0]>self.target_hull.bounds[1]) or np.any(m.bounds[1]<self.target_hull.bounds[0]):continue
                hit,hit_witnesses=self.target.in_collision_single(m,return_data=True)
                if hit:
                    if self._allowed_pad_contact(p,m,contacts):
                        accepted.append(p.name)
                        continue
                    return check(False,link=p.name,cause='conservative_target_volume_intersection',
                                 **witness_summary(hit_witnesses))
                if self._inside_target(m):
                    return check(False,link=p.name,cause='solid_containment')
            return check(True,scope=self.target_model,
                         allowed_local_fingertip_contacts=accepted,
                         local_contact_assumption=('open_CAD_surface_not_solid_ground_truth'
                            if accepted and not self.obj.mesh.is_watertight else None))

        # Experimental ORIGINAL TRIANGLE-SURFACE model for nonwatertight CAD.
        # The convex hull is a diagnostic broadphase, not an unquestioned solid.
        hull_overlap=[];accepted=[]
        anchors=np.asarray(contacts,dtype=float) if contacts is not None else None
        for part in parts:
            # Raw OEM triangles matter: never use the per-link convex cover
            # as the narrowphase geometry in this ablation.
            mesh=part.visual_mesh()
            if np.any(mesh.bounds[0]>self.obj.mesh.bounds[1]) or np.any(mesh.bounds[1]<self.obj.mesh.bounds[0]):
                continue
            original_hit,witness=self.target_original.in_collision_single(mesh,return_data=True)
            if original_hit:
                if (part.pad and anchors is not None and anchors.shape==(2,3)
                        and np.isfinite(anchors).all()):
                    index=0 if part.side>0 else 1
                    try:
                        _,dist,_=trimesh.proximity.closest_point(mesh,anchors[index:index+1])
                        _,cad_dist,_=trimesh.proximity.closest_point(self.obj.mesh,anchors[index:index+1])
                        close=float(dist[0])<=self.cfg.target_contact_surface_distance_m+self.cfg.target_penetration_tolerance_m
                        anchor_on_CAD=float(cad_dist[0])<=self.cfg.target_contact_surface_distance_m
                    except (ValueError,RuntimeError):
                        close=False;anchor_on_CAD=False
                    if (close and anchor_on_CAD and local_contact_witnesses(
                            witness,np.repeat(anchors[index:index+1],2,axis=0),
                            self.cfg.target_penetration_tolerance_m,
                            self.cfg.target_contact_patch_radius_m,self.pad_region(part))):
                        accepted.append(part.name)
                        continue
                return check(False,link=part.name,cause='original_target_CAD_triangle_intersection',
                             model='open_CAD_triangle_surface_ablation',**witness_summary(witness))
            if self.target.in_collision_single(mesh):
                hull_overlap.append(part.name)
        return check(True,scope='original_open_CAD_triangle_surface_ONLY',
                     solid_interior='UNVERIFIED_NONWATERTIGHT',
                     conditional_surface_model=True,
                     convex_envelope_overlap_not_proven_solid=hull_overlap,
                     permitted_local_fingertip_contacts=accepted,
                     warning='NOT a solid collision certificate: omitted interior possible')
    def _link_suspects(self,p,ids,normals,offset,support,margin):
        """Occupied points inside the inflated link volume: (indices,clearance bounds, smallest clearance of the rest).
        Same hull test and exact-solid refinement as the original single-point rule."""
        S=[];Dc=[];rest=np.inf
        for start in range(0,len(ids),512):
            ix=np.asarray(ids[start:start+512]);d=np.max(self.centers[ix]@normals.T-offset-support,axis=1)
            inside=d<margin
            if (~inside).any():rest=min(rest,float(d[~inside].min()))
            if not inside.any():continue
            sus=ix[inside];dd=d[inside];exact=p.exact_solid()
            if exact is not None and exact is not p.local:
                local=(self.centers[sus]-p.T[:3,3])@p.T[:3,:3]
                _,dist,_=trimesh.proximity.closest_point(exact,local);ins=exact.contains(local);dist[ins]=-dist[ins]
                bound=dist-(np.sqrt(3)*self.cfg.voxel_m/2+self.cfg.sensor_uncertainty_m)
                keep=bound<margin
                if (~keep).any():rest=min(rest,float(bound[~keep].min()))
                sus=sus[keep];dd=bound[keep]
            if len(sus):S.append(sus);Dc.append(dd)
        if not S:return np.empty(0,int),np.empty(0),rest
        return np.concatenate(S),np.concatenate(Dc),rest

    def _unique_voxels(self,idx):
        if len(idx)==0:return 0
        return len(np.unique(np.floor(self.points[idx]/self.cfg.voxel_m).astype(np.int64),axis=0))

    def occupied_voxel_counts(self,parts):
        """Diagnostic: {link: number of different occupied voxels inside its inflated volume} (links with >0 only)."""
        margin=self.cfg.environment_clearance_m;out={}
        for p in parts:
            mesh=PosedMesh(p.local,p.T) if self.persistent_fcl else p.mesh();lo,hi=mesh.bounds;center=(lo+hi)/2
            radius=np.linalg.norm((hi-lo)/2)+np.sqrt(3)*self.cfg.voxel_m+self.cfg.sensor_uncertainty_m+margin
            ids=self.tree.query_ball_point(center,radius)
            normals=p.local.face_normals@p.T[:3,:3].T;offset=np.sum(p.local.triangles_center*p.local.face_normals,axis=1)+normals@p.T[:3,3]
            support=.5*self.cfg.voxel_m*np.abs(normals).sum(1)+self.cfg.sensor_uncertainty_m
            pl=np.unique(np.round(np.c_[normals,offset,support],10),axis=0)
            S,_,_=self._link_suspects(p,ids,pl[:,:3],pl[:,3],pl[:,4],margin)
            n=self._unique_voxels(S)
            if n:out[p.name]=n
        return out

    def clear(self,parts,target=False,contacts=None):
        margin=self.cfg.environment_clearance_m;minimum=np.inf;min_vox=self.cfg.environment_min_occupied_voxels;ignored=0;ignored_min=np.inf
        for p in parts:
            mesh=PosedMesh(p.local,p.T) if self.persistent_fcl else p.mesh();dp=float(np.min(mesh.vertices@self.plane[:3]+self.plane[3]));minimum=min(minimum,dp)
            if dp<margin:return check(False,cause='support_plane',link=p.name,clearance_lower_bound_m=dp)
            lo,hi=mesh.bounds;center=(lo+hi)/2;radius=np.linalg.norm((hi-lo)/2)+np.sqrt(3)*self.cfg.voxel_m+self.cfg.sensor_uncertainty_m+margin
            ids=self.tree.query_ball_point(center,radius)
            normals=p.local.face_normals@p.T[:3,:3].T;offset=np.sum(p.local.triangles_center*p.local.face_normals,axis=1)+normals@p.T[:3,3]
            support=.5*self.cfg.voxel_m*np.abs(normals).sum(1)+self.cfg.sensor_uncertainty_m
            # Remove coplanar duplicate facets; no geometric simplification.
            planes=np.unique(np.round(np.c_[normals,offset,support],10),axis=0);normals=planes[:,:3];offset=planes[:,3];support=planes[:,4]
            if min_vox>1:
                S,Dc,rest=self._link_suspects(p,ids,normals,offset,support,margin);minimum=min(minimum,rest)
                n=self._unique_voxels(S)
                if n>=min_vox:
                    k=int(S[int(Dc.argmin())])
                    return check(False,cause='observed_occupancy',link=p.name,clearance_lower_bound_m=float(Dc.min()),witness_world_m=self.points[k],witness_label=int(self.labels[k]),
                                 occupied_voxels_in_collision=n,min_occupied_voxels_required=min_vox,narrowphase='exact watertight CAD or conservative convex fallback')
                if n:ignored+=n;ignored_min=min(ignored_min,float(Dc.min()))
            else:
                for start in range(0,len(ids),512):
                    ix=ids[start:start+512];d=np.max(self.centers[ix]@normals.T-offset-support,axis=1);dc=float(d.min());minimum=min(minimum,dc)
                    if dc<margin:
                        # A hull is only the broad phase for a closed original CAD link.
                        # Exact distance + solid containment refines empty concavities.
                        suspects=np.asarray(ix)[d<margin]
                        exact=p.exact_solid()
                        if exact is not None and exact is not p.local:
                            local=(self.centers[suspects]-p.T[:3,3])@p.T[:3,:3]
                            _,dist,_=trimesh.proximity.closest_point(exact,local)
                            inside=exact.contains(local)
                            dist[inside]=-dist[inside]
                            bound=dist-(np.sqrt(3)*self.cfg.voxel_m/2+self.cfg.sensor_uncertainty_m)
                            if np.all(bound>=margin):continue
                            k=suspects[int(bound.argmin())];dc=float(bound.min())
                        else:k=ix[int(d.argmin())]
                        return check(False,cause='observed_occupancy',link=p.name,clearance_lower_bound_m=dc,witness_world_m=self.points[k],witness_label=int(self.labels[k]),narrowphase='exact watertight CAD or conservative convex fallback')
            dc=float(self.other.min_distance_single(mesh));minimum=min(minimum,dc)
            if dc<margin:return check(False,cause='other_CAD_or_finite_plane',link=p.name,clearance_lower_bound_m=dc)
            if target:
                r=self.target_clear([p],contacts=contacts)
                if r['status']!='PASS':return r
        extra=dict(ignored_isolated_voxels=ignored,ignored_min_clearance_m=(None if not np.isfinite(ignored_min) else float(ignored_min)),min_occupied_voxels_required=min_vox) if min_vox>1 else {}
        return check(True,clearance_lower_bound_m=minimum,scope='observed non-target occupancy, conservative other CAD, finite obstacle planes, infinite table halfspace',**extra)
    def translation(self,g,A,B,q,contacts=None):
        return self.clear([swept(a,b) for a,b in zip(g.parts(A,q),g.parts(B,q))],target=True,contacts=contacts)
    def closure(self,g,M,qa,qb,contacts=None):
        if g.spec.model=='2fg7':
            # Linear prismatic jaw motion: segment at a spatial bound rather
            # than spanning all 12 mm with one large swept convex cover.
            if self.cfg.opt5_enabled:
                count=max(1,int(np.ceil(abs(g.width(qa)-g.width(qb))/self.cfg.opt5_closure_step_m)))
                nodes=np.linspace(qa,qb,count+1)
            else:nodes=np.array([qa,qb])
            curvature=0.
        else:
            nodes=np.linspace(qa,qb,max(2,int(np.ceil(abs(qa-qb)/self.cfg.closure_step_rad))+1))
            # All OEM RG translations are sums of at most four rotating offsets.
            # Sum of their lengths < 1 m; ||x''(q)|| <= 1 m/rad^2.
            curvature=float(np.max(np.diff(nodes)**2)/8.)
        for a,b in zip(nodes[:-1],nodes[1:]):
            env=[swept(p,r,curvature) for p,r in zip(g.parts(M,a),g.parts(M,b))]
            r=self.clear(env)
            if r['status']!='PASS':return check(False,environment=r,segments=len(nodes)-1,arc_bound_m=curvature)
            tol=(0. if self.cfg.target_body_model=='original_surface_ablation'
                 else self.cfg.target_penetration_tolerance_m)
            target=[swept(p,r,curvature) for p,r in zip(g.parts(M,a,tol),g.parts(M,b,tol))]
            r=self.target_clear(target,contacts=contacts)
            if r['status']!='PASS':return check(False,target=r,segments=len(nodes)-1,arc_bound_m=curvature)
            for q in [a,b]:
                r=g.self_collision(q)
                if r['status']!='PASS':return check(False,self_collision=r,segments=len(nodes)-1)
        return check(True,segments=len(nodes)-1,arc_bound_m=curvature,method='continuous endpoint convex sweeps with analytic curvature enclosure; fixed gripper base',pad_tolerance_m=self.cfg.target_penetration_tolerance_m)
