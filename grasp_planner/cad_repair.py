"""OPT7: explicit CAD repair + convex decomposition of the TARGET collision model.

Why: the OPT4 target model is ONE convex hull per object (non-watertight CAD).
It is ~2x the CAD volume and fills the gap between pliers handles, so a gripper
finger in a physically free gap is rejected. Here the open/non-manifold CAD is
converted to a watertight solid (voxel fill; audited against the ORIGINAL CAD)
and, optionally, split into a few convex parts (CoACD).

Nothing is assumed silently: every model carries an audit dict with watertight
state, original-surface coverage, repaired->original deviation and volumes.
The repaired solid is a *reconstruction* of a solid the open CAD only implies;
results using it are labelled conditional (see Scene.target_model).
"""
import hashlib,json
from pathlib import Path
import numpy as np,trimesh
from scipy import ndimage

_MEMORY={}

def _volume_fill(mesh,pitch):
    v=mesh.voxelized(pitch);surf=v.matrix.copy()
    # Close sub-pitch gaps in the shell (STL seams), fill, keep the shell itself.
    closed=ndimage.binary_closing(np.pad(surf,2),structure=ndimage.generate_binary_structure(3,2),iterations=1)
    solid=ndimage.binary_fill_holes(closed)[2:-2,2:-2,2:-2]|surf
    return solid,v

def repair_to_solid(mesh,pitch_m=.0005,max_faces=40000,min_volume_ratio=.85,max_voxels=8e6,simplifier='quadric',simplify_tol_m=.0002):
    """Watertight solid from an open/non-manifold surface. Raises if it cannot."""
    from skimage.measure import marching_cubes
    # Large parts: coarsen the grid so the voxel count (and memory) stays bounded.
    pitch_m=max(float(pitch_m),float((np.prod(mesh.extents)/max_voxels)**(1/3)))
    solid,v=_volume_fill(mesh,pitch_m)
    vol=ndimage.gaussian_filter(np.pad(solid,1).astype(float),.7)
    verts,faces,_,_=marching_cubes(vol,.5)
    out=trimesh.Trimesh((verts-1)*pitch_m+v.translation,faces,process=True);out.fix_normals()
    if len(out.faces)>max_faces and simplifier=='manifold':
        out=_manifold_simplify(out,max_faces,simplify_tol_m)
    elif len(out.faces)>max_faces:
        full=out;out=None
        # Decimation can break watertightness on dense meshes: retry with a larger
        # budget and finally keep the undecimated (still valid) solid.
        for budget in (int(max_faces),int(max_faces*1.5),int(max_faces*2.5)):
            if budget>=len(full.faces):break
            cand=full.simplify_quadric_decimation(face_count=budget);cand.fix_normals()
            if cand.is_watertight and cand.is_winding_consistent and cand.volume>0:out=cand;break
        out=full if out is None else out
    out.metadata['pitch_m']=pitch_m
    if not (out.is_watertight and out.is_winding_consistent and out.volume>0):
        raise ValueError('CAD repair failed: result is not a watertight consistently wound solid')
    # Leak guard: a hole larger than the closing radius lets the fill escape and
    # leaves a HOLLOW shell. The original signed volume (valid for consistently
    # wound surfaces up to the holes) must then be clearly larger than the result.
    if mesh.is_winding_consistent and mesh.volume>0 and out.volume<min_volume_ratio*mesh.volume:
        raise ValueError(f'CAD repair rejected: filled volume {out.volume:.3e} m3 < {min_volume_ratio:.2f} x original signed volume '
                         f'{mesh.volume:.3e} m3 (shell has holes larger than the closing radius; interior not filled)')
    return out

def _sample(mesh,n,seed):
    pts,_=trimesh.sample.sample_surface(mesh,n,seed=seed);return pts

def _manifold_simplify(mesh,max_faces,tol_m):
    """Edge-collapse simplification that keeps the mesh a valid closed 2-manifold (manifold3d).

    Unlike quadric decimation it never breaks watertightness; tolerance is escalated
    until the face budget is met. Deviation is bounded by ~tolerance and is audited later
    against the ORIGINAL mesh."""
    import manifold3d
    mf=manifold3d.Manifold(manifold3d.Mesh(vert_properties=np.asarray(mesh.vertices,np.float32),
                                           tri_verts=np.asarray(mesh.faces,np.uint32)))
    best=None;tol=float(tol_m)
    for _ in range(5):
        mm=mf.simplify(tol).to_mesh()
        cand=trimesh.Trimesh(np.asarray(mm.vert_properties,dtype=float)[:,:3],np.asarray(mm.tri_verts),process=False)
        if cand.is_watertight and cand.is_winding_consistent and cand.volume>0:
            best=cand
            if len(cand.faces)<=max_faces:break
        tol*=1.5
    if best is None:raise ValueError('manifold simplification produced no valid closed solid')
    return best

def audit_repair(original,solid,pitch_m,samples=20000,seed=0):
    pts=_sample(original,samples,seed);inside=solid.contains(pts);out=pts[~inside]
    gap=trimesh.proximity.closest_point(solid,out)[1] if len(out) else np.zeros(1)
    back=trimesh.proximity.closest_point(original,_sample(solid,samples,seed+1))[1]
    return dict(watertight=bool(solid.is_watertight),winding_consistent=bool(solid.is_winding_consistent),
        faces=int(len(solid.faces)),pitch_m=float(pitch_m),
        volume_m3=float(solid.volume),original_signed_volume_m3=float(original.volume),
        convex_hull_volume_m3=float(original.convex_hull.volume),
        original_surface_inside_solid_fraction=float(inside.mean()),
        original_surface_outside_gap_max_m=float(gap.max()),
        repaired_to_original_surface_max_m=float(back.max()),
        repaired_to_original_surface_p99_m=float(np.percentile(back,99)))

def _inflate(hull,m):
    if m<=0:return hull
    off=np.array([[x,y,z] for x in(-m,m) for y in(-m,m) for z in(-m,m)])
    return trimesh.convex.convex_hull((hull.vertices[:,None,:]+off[None]).reshape(-1,3))

def decompose(solid,threshold=.03,inflate_m=0.,seed=0,max_faces=20000):
    """Convex parts covering the solid (CoACD). Parts are NOT guaranteed to
    cover the solid exactly; audit_decomposition measures the inward gap."""
    import coacd
    coacd.set_log_level('error')
    src=solid.simplify_quadric_decimation(face_count=int(max_faces)) if len(solid.faces)>max_faces else solid
    parts=coacd.run_coacd(coacd.Mesh(np.asarray(src.vertices,float),np.asarray(src.faces,int)),
        threshold=float(threshold),max_convex_hull=-1,preprocess_mode='off',seed=int(seed))
    hulls=[_inflate(trimesh.Trimesh(v,f),inflate_m) for v,f in parts]
    if not hulls:raise ValueError('Convex decomposition returned no parts')
    return hulls

def audit_decomposition(original,solid,hulls,samples=20000,seed=0):
    pts=_sample(original,samples,seed);cov=np.zeros(len(pts),bool)
    for h in hulls:cov|=h.contains(pts)
    out=pts[~cov]
    gap=np.min([trimesh.proximity.closest_point(h,out)[1] for h in hulls],axis=0) if len(out) else np.zeros(1)
    return dict(parts=len(hulls),union_hull_volume_m3=float(sum(h.volume for h in hulls)),
        original_surface_in_union_fraction=float(cov.mean()),
        original_surface_uncovered_gap_max_m=float(gap.max()),
        original_surface_uncovered_gap_p99_m=float(np.percentile(gap,99)))

def _key(sha,cfg):
    return hashlib.sha256(json.dumps([sha,cfg.cad_repair_pitch_m,cfg.cad_repair_max_faces,
        cfg.cad_decomp_threshold,cfg.cad_decomp_inflate_m,'v1'],sort_keys=True).encode()).hexdigest()[:20]

def target_solids(obj,cfg):
    """World-frame target collision solids for ``cfg.target_body_model``.

    Computed once per CAD (local frame, keyed by STL sha256) and shared between
    instances (Pliers_001/002) and grippers; optional disk cache.
    Returns dict(kind, solids=[Trimesh], audit=dict).
    """
    want=cfg.target_body_model
    key=(_key(obj.audit['sha256'],cfg),want)
    if key not in _MEMORY:
        disk=Path(cfg.cad_cache_dir).expanduser()/f'{key[0]}_{want}.npz' if cfg.cad_cache_dir else None
        local=obj.mesh.copy();local.apply_transform(np.linalg.inv(obj.T))
        if disk is not None and disk.is_file():
            z=np.load(disk,allow_pickle=False);audit=json.loads(str(z['audit']))
            n=int(z['n']);solids=[trimesh.Trimesh(z[f'v{i}'],z[f'f{i}'],process=False) for i in range(n)]
        else:
            solid=repair_to_solid(local,cfg.cad_repair_pitch_m,cfg.cad_repair_max_faces)
            audit=dict(repair=audit_repair(local,solid,cfg.cad_repair_pitch_m),source_sha256=obj.audit['sha256'])
            if want=='convex_decomposition':
                hulls=decompose(solid,cfg.cad_decomp_threshold,cfg.cad_decomp_inflate_m)
                audit['decomposition']=audit_decomposition(local,solid,hulls);solids=hulls
            else:solids=[solid]
            if disk is not None:
                disk.parent.mkdir(parents=True,exist_ok=True)
                arrays={'audit':np.array(json.dumps(audit)),'n':np.array(len(solids))}
                for i,s in enumerate(solids):arrays[f'v{i}']=np.asarray(s.vertices);arrays[f'f{i}']=np.asarray(s.faces)
                np.savez_compressed(disk,**arrays)
        _MEMORY[key]=(solids,audit)
    solids,audit=_MEMORY[key];world=[]
    for s in solids:
        w=s.copy();w.apply_transform(obj.T);world.append(w)
    return dict(kind=want,solids=world,audit=audit)


def gripper_link_solid(model,link,raw,pitch_m,max_faces,cache_dir=None):
    """Watertight solid + audit for one OEM gripper link (memory + optional disk cache)."""
    sha=hashlib.sha256(np.ascontiguousarray(raw.vertices).tobytes()+np.ascontiguousarray(raw.faces).tobytes()+
        json.dumps([pitch_m,max_faces,'g2-manifold']).encode()).hexdigest()[:20]
    key=('gripper',model,link,sha)
    if key not in _MEMORY:
        disk=Path(cache_dir).expanduser()/f'gripper_{model}_{link}_{sha}.npz' if cache_dir else None
        if disk is not None and disk.is_file():
            z=np.load(disk,allow_pickle=False);sol=trimesh.Trimesh(z['v'],z['f'],process=False);audit=json.loads(str(z['audit']))
        else:
            sol=repair_to_solid(raw,pitch_m,max_faces,simplifier='manifold')
            pitch=float(sol.metadata.get('pitch_m',pitch_m))
            audit=dict(repaired=True,**audit_repair(raw,sol,pitch,samples=6000),effective_pitch_m=pitch)
            if disk is not None:
                disk.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(disk,v=np.asarray(sol.vertices),f=np.asarray(sol.faces),audit=np.array(json.dumps(audit)))
        _MEMORY[key]=(sol,audit)
    return _MEMORY[key]
