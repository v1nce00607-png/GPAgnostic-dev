"""Audited statistical filtering, multi-view semantics and finite obstacle planes."""
from pathlib import Path
import time
import numpy as np,trimesh
from scipy.spatial import cKDTree,ConvexHull
from scipy.ndimage import maximum_filter,minimum_filter,binary_erosion
from sklearn.cluster import DBSCAN
from .models import transform,save,unit
from .reconstruction import Reconstruction

def preprocess(depth,mask,cfg):
    valid=depth>0;hi=maximum_filter(depth,size=3);lo=minimum_filter(np.where(valid,depth,np.inf),size=3)
    edge=(hi-lo)>cfg.depth_jump_m
    labels=mask.copy()
    if cfg.mask_erosion_px:
        for label in np.unique(mask):
            if label==0:continue
            core=binary_erosion(mask==label,iterations=cfg.mask_erosion_px)
            labels[(mask==label)&~core]=0
    return (~edge)&valid&(mask==labels),labels,{'depth_jump_rejected_pixels':int(sum(edge.ravel()&valid.ravel())),'mask_border_unlabelled_pixels':int(sum((mask!=labels).ravel()))}

def relabel_dominance(own,competitor,occupied,free,cfg):
    """Return mask of independently supported, non-conflicted target assignments.

    Depth-ray contradiction prevents a reclassification even with strong masks.
    Competitor tolerance is opt-in; the historical policy is zero competitors.
    """
    clear_competition=(competitor==0)
    if cfg.relabel_max_competitor_fraction>0:
        clear_competition|=(competitor>0)&(competitor<=own*cfg.relabel_max_competitor_fraction)&(own>=cfg.relabel_min_dominance_ratio*competitor)
    contradictory=(free>=5)&(free>3*occupied)
    # Historical strict policy is preserved bit-for-bit at fraction=0.
    safe_depth=(~contradictory if cfg.relabel_max_competitor_fraction>0 else np.ones_like(own,dtype=bool))
    return (own>=cfg.relabel_min_views)&(own/np.maximum(occupied,1)>=cfg.relabel_vote_fraction)&clear_competition&safe_depth

def floating_geometry_candidates(labels,height,sor,radius,dbnoise,cfg):
    """3D morphology only: a candidate is NEVER deleted by this function.

    In particular, the sign of image vertical is irrelevant; `height` is a
    signed metric distance to the reconstructed support plane.
    """
    flags=np.asarray(sor,dtype=np.int8)+np.asarray(radius,dtype=np.int8)+np.asarray(dbnoise,dtype=np.int8)
    return (np.asarray(labels)==0)&(np.asarray(height)>=cfg.floating_min_height_m)&(flags>=2)


def floating_evidence_probe(labels,height,cfg):
    """Points requiring extra depth rays, not removal candidates by fiat."""
    return (np.asarray(labels)==0)&(np.asarray(height)>=cfg.floating_min_height_m)


def floating_depth_confirmed(free,occupied,mask_support,cfg):
    """Only multi-view geometric contradictions authorize floating removal.

    Mask support is an additional veto (not proof of occupied space). Requiring
    both statistical isolation and independent free-space rays controls the
    risk of erasing real small objects or partially observed obstacles.
    """
    free=np.asarray(free)
    occupied=np.asarray(occupied)
    mask_support=np.asarray(mask_support)
    return ((free>=cfg.floating_min_free_views)
            &(free>cfg.floating_free_to_occupied_ratio*np.maximum(occupied,1))
            &(mask_support<cfg.relabel_min_views))

def halo_roi_candidates(points,labels,plane,objects,cfg):
    """Candidate pixels for OBJECT-RELATIVE 3D HALO ABLATION, not certified emptiness.

    The object's projected footprint is evaluated in the reconstructed support
    plane rather than screen coordinates. All physically labeled samples,
    the table slab, and other known CAD AABBs are protected from deletion.
    A label-0 point in this region is only an *assumed* false obstacle in the
    experimental policy, not a sensor-proven false obstacle.
    """
    points=np.asarray(points,dtype=float)
    labels=np.asarray(labels)
    n=np.asarray(plane[:3],dtype=float)
    n=n/np.linalg.norm(n)
    tangent=np.eye(3)[np.argmin(np.abs(n))]
    tangent=tangent-tangent.dot(n)*n
    tangent=tangent/np.linalg.norm(tangent)
    bitangent=np.cross(n,tangent)
    basis=np.column_stack((tangent,bitangent))
    uv=points@basis
    height=points@n+float(plane[3])
    eligible=(labels==0)&(height>cfg.halo_plane_guard_m)
    out=np.zeros(len(points),bool)
    owner=np.full(len(points),-1,dtype=np.int16)
    protected=np.zeros(len(points),bool)
    boxes=[]
    for o in objects:
        bounds=np.asarray(o.mesh.bounds,dtype=float)
        corners=np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]],dtype=float)
        # bounds is (min,max) per axis, so use each column for AABB corners.
        boxes.append(corners)
    # Resolve overlapping object halos by nearest unexpanded CAD AABB.
    # Iteration order must not change which physical target owns a point.
    nearest=np.full(len(points),np.inf,dtype=float)
    for idx,o in enumerate(objects):
        if cfg.halo_objects is not None and o.name not in cfg.halo_objects:
            continue
        corners=boxes[idx]
        obj_uv=corners@basis
        top=float(np.max(corners@n+plane[3]))
        lo=obj_uv.min(0)-cfg.halo_margin_m
        hi=obj_uv.max(0)+cfg.halo_margin_m
        region=(eligible&(height<top+cfg.halo_ceiling_margin_m)
                &np.all((uv>=lo)&(uv<=hi),axis=1))
        bounds=np.asarray(o.mesh.bounds,dtype=float)
        dist=np.linalg.norm(np.maximum(np.maximum(bounds[0]-points,points-bounds[1]),0.),axis=1)
        best=region&(dist<nearest)
        owner[best]=idx
        nearest[best]=dist[best]
        out|=region
    # Protect the physical AABB of a DIFFERENT labelled CAD object.
    # Known CAD remains a separate collision obstacle even when its surrounding
    # unlabeled RGB-D samples are removed.
    for j,o in enumerate(objects):
        box=np.asarray(o.mesh.bounds,dtype=float)
        lo=box[0]-cfg.halo_other_object_guard_m
        hi=box[1]+cfg.halo_other_object_guard_m
        guard=np.all((points>=lo)&(points<=hi),axis=1)
        protected|=guard&(owner!=j)
    out&=~protected
    owner[~out]=-1
    return out,owner,protected


def halo_delete_candidates(candidate,labels,sor,radius,dbnoise,policy):
    """Return ablation removal set. 'experimental_roi' is deliberately permissive."""
    c=np.asarray(candidate,bool)&(np.asarray(labels)==0)
    if policy=='off':return np.zeros_like(c)
    if policy=='isolated_only':
        flags=np.asarray(sor,bool).astype(int)+np.asarray(radius,bool)+np.asarray(dbnoise,bool)
        return c&(flags>=1)
    if policy=='experimental_roi':return c
    raise ValueError(f'Unknown halo_policy: {policy}')

def bbox_mask(p,bounds,scale):
    lo,hi=bounds;return np.all(np.abs(p-(lo+hi)/2)<=(hi-lo)*scale/2,axis=1)

def filter_scene(cap,rec,cfg,out,progress=print):
    start=time.perf_counter();p=rec.points;n=len(p);labels=rec.labels.copy();tree=cKDTree(p)
    ds,nn=tree.query(p,k=min(cfg.sor_k+1,n),workers=1);md=ds[:,1:].mean(1);sor=md>md.mean()+cfg.sor_std*md.std()
    counts=tree.query_ball_point(p,cfg.radius_m,return_length=True,workers=1)-1;radius=counts<cfg.radius_min_neighbors
    height=p@rec.plane[:3]+rec.plane[3];bg=labels==0
    ids=np.flatnonzero(bg&(height>.008));clusters=np.full(n,-2,np.int32)
    if len(ids):clusters[ids]=DBSCAN(eps=cfg.dbscan_eps_m,min_samples=cfg.dbscan_min_samples,n_jobs=1).fit_predict(p[ids])
    sizes=np.bincount(clusters[ids][clusters[ids]>=0]);dbnoise=clusters==-1
    if len(sizes):dbnoise[ids]|=(clusters[ids]>=0)&(sizes[np.maximum(clusters[ids],0)]<cfg.dbscan_min_cluster)
    near=np.zeros(n,bool);dist=np.full((n,len(cap.objects)),np.inf,np.float32)
    for j,o in enumerate(cap.objects):
        box=bbox_mask(p,o.mesh.bounds,cfg.bbox_scale);near|=box;ix=np.flatnonzero(bg&box&(height>cfg.sensor_uncertainty_m))
        d=cKDTree(o.mesh.vertices).query(p[ix],workers=1)[0];ix=ix[d<cfg.relabel_band_m+.003]
        for a in range(0,len(ix),1000):
            jj=ix[a:a+1000];dist[jj,j]=trimesh.proximity.closest_point(o.mesh,p[jj])[1]
    near_surface=bg&(dist.min(1)<=cfg.relabel_band_m);low=bg&(rec.counts<cfg.filter_min_views)
    floating_candidate=floating_geometry_candidates(labels,height,sor,radius,dbnoise,cfg)
    floating_probe=floating_evidence_probe(labels,height,cfg)
    # The historical evidence pool misses high-count background points.
    # All geometrically elevated background is now probed with independent
    # calibrated free-space rays; elevation alone NEVER authorizes removal.
    extra_floating=floating_probe if cfg.floating_policy=='depth_confirmed' else np.zeros(n,bool)
    test=np.flatnonzero(near_surface|low|extra_floating);pp=p[test];support=np.zeros((len(test),len(cap.objects)),np.int16);occupied=np.zeros(len(test),np.int16);free=np.zeros(len(test),np.int16)
    frames=np.linspace(0,len(cap.poses)-1,min(cfg.evidence_frames,len(cap.poses))).astype(int)
    for c,i in enumerate(frames):
        po=cap.poses[i];_,depth,mask=cap.frame(po['frame_id']);pc=transform(pp,po['camera_from_world']);z=pc[:,2];uv=pc[:,:2]/np.maximum(z[:,None],1e-9);x=np.rint(uv[:,0]*cap.K[0,0]+cap.K[0,2]).astype(int);y=np.rint(uv[:,1]*cap.K[1,1]+cap.K[1,2]).astype(int)
        ok=(z>0)&(x>=0)&(x<depth.shape[1])&(y>=0)&(y<depth.shape[0]);ix=np.flatnonzero(ok);dd=depth[y[ix],x[ix]];v=dd>0;ix=ix[v];dd=dd[v]
        free[ix]+=z[ix]<dd-cfg.depth_agreement_m;same=np.abs(z[ix]-dd)<=cfg.depth_agreement_m;occupied[ix[same]]+=1;ix=ix[same]
        for k,o in enumerate(cap.objects):
            dil=maximum_filter((mask==o.id).astype(np.uint8),size=5)>0;support[ix,k]+=dil[y[ix],x[ix]]
        if c%16==0:progress(f'Filtering evidence {c+1}/{len(frames)}')
    repaired=np.zeros(n,bool)
    minority_conflict_relabelled={}
    for k,o in enumerate(cap.objects):
        own=support[:,k];competitor=np.max(np.delete(support,k,axis=1),axis=1) if len(cap.objects)>1 else np.zeros(len(test))
        supported=relabel_dominance(own,competitor,occupied,free,cfg)
        ok=near_surface[test]&(dist[test,k]<=cfg.relabel_band_m)&supported
        minority_conflict_relabelled[o.name]=int(np.sum(ok&(competitor>0)))
        labels[test[ok]]=o.id;repaired[test[ok]]=True
    # Neither a statistical outlier nor DBSCAN's -1 means proven empty space.
    # Require independent depth-ray contradiction before removing an observation.
    contradiction=(free>=5)&(free>3*occupied)
    statistical=(sor|radius|dbnoise)
    remove=np.zeros(n,bool);remove[test]=low[test]&contradiction&~repaired[test]
    # Every low-view deletion has multi-view evidence; flags explain morphology.
    consensus=bg&(rec.counts<cfg.occupancy_min_views)&~repaired
    morphology=bg&(rec.counts<cfg.occupancy_min_views+2)&((sor.astype(int)+radius+dbnoise)>=2)&~repaired
    remove|=consensus|morphology
    floating_deleted=np.zeros(n,bool)
    if cfg.floating_policy=='depth_confirmed':
        # Crucial: never erase solely because a point looks like a floating
        # speckle in Plotly. Confirm clear depth rays in the original frames.
        full_candidate=floating_probe[test]
        max_support=support.max(axis=1) if support.shape[1] else np.zeros(len(test),int)
        certified=full_candidate&floating_depth_confirmed(free,occupied,max_support,cfg)&~repaired[test]
        floating_deleted[test]=certified&~remove[test]
        remove[test[certified]]=True
    # OPT4 object-relative halo ablation. It is off by default, and never
    # modifies labelled target/other-object samples or table proximity.
    halo_roi=np.zeros(n,bool)
    halo_owner=np.full(n,-1,np.int16)
    halo_protected=np.zeros(n,bool)
    halo_deleted=np.zeros(n,bool)
    halo_deleted_by_object={o.name:0 for o in cap.objects}
    if cfg.halo_policy!='off':
        halo_roi,halo_owner,halo_protected=halo_roi_candidates(p,labels,rec.plane,cap.objects,cfg)
        hdel=halo_delete_candidates(halo_roi,labels,sor,radius,dbnoise,cfg.halo_policy)
        halo_deleted=hdel&~remove
        remove|=hdel
        for j,o in enumerate(cap.objects):
            halo_deleted_by_object[o.name]=int(np.sum(halo_deleted&(halo_owner==j)))
    naive=near&bg
    if cfg.bbox_policy=='erase_unverified':remove|=naive&~repaired
    keep=~remove
    audit=dict(halo_policy=cfg.halo_policy,halo_objects=cfg.halo_objects,halo_margin_m=cfg.halo_margin_m,halo_ceiling_margin_m=cfg.halo_ceiling_margin_m,halo_plane_guard_m=cfg.halo_plane_guard_m,halo_roi_candidates=int(np.sum(halo_roi)),halo_other_object_protected=int(np.sum(halo_protected&halo_roi)),halo_extra_removed=int(np.sum(halo_deleted)),halo_deleted_by_object=halo_deleted_by_object,halo_experimental_assumption=(cfg.halo_policy=='experimental_roi'),floating_policy=cfg.floating_policy,floating_morphology_candidates=int(floating_candidate.sum()),floating_probe_candidates=int(floating_probe.sum()),floating_extra_evidence_points=int(np.sum(extra_floating&~near_surface&~low)),floating_depth_confirmed_removed=int(floating_deleted.sum()),floating_min_free_views=cfg.floating_min_free_views,floating_min_height_m=cfg.floating_min_height_m,raw_points=n,retained_points=int(keep.sum()),removed_points=int(remove.sum()),relabelled_points=int(repaired.sum()),relabelled_by_object={o.name:int(sum(repaired&(labels==o.id))) for o in cap.objects},relabelled_minority_competitor_by_object=minority_conflict_relabelled,relabel_max_competitor_fraction=cfg.relabel_max_competitor_fraction,relabel_min_dominance_ratio=cfg.relabel_min_dominance_ratio,sor_flagged=int(sor.sum()),radius_flagged=int(radius.sum()),dbscan_noise_or_small_cluster=int(dbnoise.sum()),dbscan_clusters=len(sizes),removed_statistically_flagged=int(sum(remove&statistical)),removed_without_statistical_flag=int(sum(remove&~statistical)),low_view_points=int(low.sum()),naive_bbox_deleted=int(naive.sum()),retained_background_inside_bbox=int(sum(near&keep&(labels==0))),bbox_scale=cfg.bbox_scale,bbox_policy=cfg.bbox_policy,evidence_frame_ids=[cap.poses[i]['frame_id'] for i in frames],seconds=time.perf_counter()-start,consensus_rejected=int(consensus.sum()),morphology_rejected=int(morphology.sum()),policy='Minimum frame consensus plus audited SOR/radius/DBSCAN and depth-ray contradiction. Removed samples retained as uncertainty evidence.',limitations=['Filtering is a reconstruction assumption, not ground truth','Finite obstacle planes and table remain collision obstacles','Blanket bbox erasure can never yield an accepted PASS','The 3D free-space test is reconstruction-dependent and may still have calibration errors','OPT4 halo ablation is an assumed scene cleanup, not an independently certified empty volume'])
    out=Path(out);save(out/'filter_audit.json',audit)
    np.savez_compressed(out/'filter_evidence.npz',raw_points=p,raw_labels=rec.labels,new_labels=labels,keep=keep,repaired=repaired,floating_candidate=floating_candidate,floating_probe=floating_probe,floating_deleted=floating_deleted,halo_roi=halo_roi,halo_owner=halo_owner,halo_deleted=halo_deleted,halo_protected=halo_protected,near_bbox=near,sor=sor,radius=radius,dbscan=clusters,mean_knn_distance=md,indices=test,support=support,free=free,occupied=occupied)
    report={**rec.report,'points':int(keep.sum()),'label_counts':{str(i):int(sum(labels[keep]==i)) for i in range(max(o.id for o in cap.objects)+1)},'filter':audit}
    r=Reconstruction(p[keep],rec.colors[keep],labels[keep],rec.confidence[keep],rec.counts[keep],rec.plane,report,rec.frames);r.save(out/'reconstruction.npz');save(out/'reconstruction_report.json',report);save(out/'frame_metrics.json',rec.frames)
    return r

def obstacle_planes(rec,cfg,out):
    p=rec.points;h=p@rec.plane[:3]+rec.plane[3];ix=np.flatnonzero((rec.labels==0)&(rec.counts>=3)&(h>.008));rng=np.random.default_rng(cfg.seed);meshes=[];reports=[]
    for k in range(cfg.max_obstacle_planes):
        if len(ix)<cfg.plane_min_points:break
        sample=p[ix[rng.choice(len(ix),min(12000,len(ix)),replace=False)]];best=None;bestcount=0
        for _ in range(180):
            a,b,c=sample[rng.choice(len(sample),3,replace=False)];n=np.cross(b-a,c-a)
            if np.linalg.norm(n)<1e-10:continue
            n=unit(n);d=-n@a;count=int(sum(abs(sample@n+d)<cfg.plane_threshold_m))
            if count>bestcount:bestcount=count;best=(n,d)
        n,d=best;hit=abs(p[ix]@n+d)<cfg.plane_threshold_m;ids=ix[hit];ix=ix[~hit]
        if len(ids)<cfg.plane_min_points:continue
        labels=DBSCAN(eps=.014,min_samples=6,n_jobs=1).fit_predict(p[ids])
        for label in sorted(set(labels)-{-1}):
            v=p[ids[labels==label]]
            if len(v)<cfg.plane_min_points:continue
            center=v.mean(0);_,_,vh=np.linalg.svd(v-center,full_matrices=False);normal=vh[-1];axes=vh[:2];xy=(v-center)@axes.T
            try:boundary=xy[ConvexHull(xy).vertices]
            except Exception:continue
            projected=boundary@axes+center;thickness=cfg.sensor_uncertainty_m+cfg.plane_threshold_m
            mesh=trimesh.convex.convex_hull(np.vstack([projected+normal*thickness,projected-normal*thickness]));name=f'obstacle_plane_{len(meshes):02d}';mesh.export(Path(out)/(name+'.ply'));meshes.append(mesh)
            reports.append(dict(name=name,points=len(v),normal=normal,center_m=center,rms_m=float(np.sqrt(np.mean(((v-center)@normal)**2))),half_thickness_m=thickness,area_m2=float(ConvexHull(xy).volume),role='additional finite collision obstacle; supporting voxels retained'))
    save(Path(out)/'obstacle_planes.json',reports);return meshes

def estimate_normals(cap,rec,cfg,out):
    ids=np.flatnonzero(rec.labels>0);p=rec.points[ids];tree=cKDTree(p);_,ii=tree.query(p,k=min(24,len(p)),workers=1);v=p[ii]-p[ii].mean(1)[:,None,:];cov=np.einsum('nki,nkj->nij',v,v);val,vec=np.linalg.eigh(cov);normals=vec[:,:,0];camera=np.median([np.array(po['world_from_camera'])[:3,3] for po in cap.poses],axis=0);normals[np.sum(normals*(camera-p),axis=1)<0]*=-1
    # MLS first-order plane projection exported only for inspection, never moves collision samples.
    centers=p[ii].mean(1);projected=p-normals*np.sum((p-centers)*normals,axis=1)[:,None]
    np.savez_compressed(Path(out)/'target_normals.npz',points=p,normals=normals,mls_projection=projected,indices=ids,local_rms_m=np.sqrt(np.maximum(val[:,0],0)/ii.shape[1]))
    save(Path(out)/'normal_audit.json',dict(points=len(p),neighbors=ii.shape[1],orientation='towards median camera position',contact_normals='original CAD ray-hit face normals; no noisy point-normal grasp scoring',MLS_applied_to_collision=False))
