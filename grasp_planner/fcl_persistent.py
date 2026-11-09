"""OPT7 step 1: persistent FCL geometry (no per-query mesh copy / BVH rebuild).

Profiling showed ~70 % of a collision query was spent re-creating the FCL object from a
freshly transformed mesh copy (trimesh builds a BVH or Convex every call, and a fresh mesh
also loses its cached ray/rtree structures). Here the link's LOCAL mesh keeps its FCL geometry
and acceleration structures; only the pose is passed per query. The geometry and the FCL calls
are identical, so verdicts are unchanged (verified by tests/test_opt7_fcl.py and
scripts/opt7_verify_equivalence.py against the stored results).
"""
import numpy as np,trimesh
from trimesh.collision import CollisionManager

class PosedMesh:
    """A static local-frame mesh and its world pose. Quacks like the transformed mesh for the
    few attributes the collision code needs (vertices, bounds, contains) without copying."""
    __slots__=('local','T','_wv')
    def __init__(self,local,T):
        self.local=local;self.T=np.asarray(T,dtype=float);self._wv=None
        local._opt7_static=True          # the local mesh is never mutated -> safe to cache on it
    @property
    def vertices(self):
        if self._wv is None:self._wv=self.local.vertices@self.T[:3,:3].T+self.T[:3,3]
        return self._wv
    @property
    def bounds(self):
        v=self.vertices;return np.array([v.min(0),v.max(0)])
    def to_local(self,points):
        return (np.asarray(points,dtype=float)-self.T[:3,3])@self.T[:3,:3]
    def contains(self,points):
        return self.local.contains(self.to_local(points))
    def to_mesh(self):
        m=self.local.copy();m.apply_transform(self.T);return m

def closest_distance(m,points):
    """Distance from points to the mesh surface (same value for a posed or a transformed mesh)."""
    if isinstance(m,PosedMesh):return trimesh.proximity.closest_point(m.local,m.to_local(points))[1]
    return trimesh.proximity.closest_point(m,points)[1]

class PersistentCollisionManager(CollisionManager):
    """trimesh CollisionManager that accepts PosedMesh and reuses the FCL geometry of static meshes."""
    def _get_fcl_obj(self,mesh):
        if getattr(mesh,'_opt7_static',False):
            g=getattr(mesh,'_opt7_fcl',None)
            if g is None:
                g=super()._get_fcl_obj(mesh);mesh._opt7_fcl=g
            return g
        return super()._get_fcl_obj(mesh)
    def in_collision_single(self,mesh,transform=None,return_names=False,return_data=False):
        if isinstance(mesh,PosedMesh):
            if transform is not None:raise ValueError('PosedMesh already carries its transform')
            return super().in_collision_single(mesh.local,mesh.T,return_names,return_data)
        return super().in_collision_single(mesh,transform,return_names,return_data)
    def min_distance_single(self,mesh,transform=None,return_name=False,return_data=False):
        if isinstance(mesh,PosedMesh):
            if transform is not None:raise ValueError('PosedMesh already carries its transform')
            return super().min_distance_single(mesh.local,mesh.T,return_name,return_data)
        return super().min_distance_single(mesh,transform,return_name,return_data)
