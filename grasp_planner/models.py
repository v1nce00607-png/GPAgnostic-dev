from __future__ import annotations
import hashlib, json, os, tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml, trimesh


def native(v):
    if isinstance(v, np.ndarray):
        return native(v.tolist())
    if isinstance(v, np.generic):
        return native(v.item())
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, dict):
        return {str(k): native(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [native(x) for x in v]
    return v


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload=json.dumps(native(data), indent=2, allow_nan=False)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:f.write(payload)
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(4 << 20):
            h.update(b)
    return h.hexdigest()


def transform(p, T):
    T = np.asarray(T)
    return np.asarray(p) @ T[:3, :3].T + T[:3, 3]


def pose(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def valid_transform(T):
    T = np.asarray(T, dtype=float)
    return bool(
        T.shape == (4, 4)
        and np.isfinite(T).all()
        and np.allclose(T[3], [0, 0, 0, 1], atol=1e-6)
        and np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
        and abs(np.linalg.det(T[:3, :3]) - 1) < 1e-6
    )


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError("Invalid direction")
    return v / n


def check(ok=None, **kw):
    return {
        "status": "UNVERIFIED" if ok is None else ("PASS" if bool(ok) else "FAIL"),
        **kw,
    }


REQUIRED = ['data_alignment','hardware_geometry','kinematics','contact_geometry','self_collision','target_body','environment','approach','closure','filter_policy']


def verdict(checks, required=REQUIRED):
    s = [checks.get(k, {}).get("status", "UNVERIFIED") for k in required]
    if "FAIL" in s:
        return "FAIL"
    return "PASS" if s and all(x == "PASS" for x in s) else "UNVERIFIED"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


@dataclass
class Object:
    id: int
    name: str
    kind: str
    mesh: trimesh.Trimesh
    T: np.ndarray
    audit: dict


class Capture:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.camera = json.loads((self.root / "camera.json").read_text())
        self.annotations = json.loads(
            (self.root / "sam6d_scene/annotations.json").read_text()
        )
        self.poses = sorted(
            json.loads(
                (
                    self.root
                    / "reconstruction/metric_alignment/camera_poses_metric.json"
                ).read_text()
            )["poses"],
            key=lambda p: p["frame_id"],
        )
        self.by_id = {p["frame_id"]: p for p in self.poses}
        self.K = np.array(self.camera["cam_K"]).reshape(3, 3)
        u = self.camera["depth_unit"]
        if u not in ["millimetre", "metre"]:
            raise ValueError("Unsupported depth unit")
        self.depth_to_m = self.camera["depth_scale"] * (
            0.001 if u == "millimetre" else 1
        )
        if (
            self.depth_to_m <= 0
            or not np.isfinite(self.K).all()
            or min(self.K[0, 0], self.K[1, 1]) <= 0
        ):
            raise ValueError("Invalid calibration")
        for p in self.poses:
            W = np.array(p["world_from_camera"])
            C = np.array(p["camera_from_world"])
            if (
                not valid_transform(W)
                or not valid_transform(C)
                or not np.allclose(W @ C, np.eye(4), atol=1e-6)
            ):
                raise ValueError("Invalid camera transform")
        self.objects = []
        scale = self.annotations["model_scale_to_metre"]
        if scale <= 0:
            raise ValueError("Invalid model scale")
        for o in self.annotations["objects"]:
            path = self.root / "objects/stl" / f"{o['object_name']}.stl"
            m = trimesh.load(path, force="mesh", process=True)
            m.apply_scale(scale)
            T = np.array(o["world_from_object"])
            if not valid_transform(T) or not np.isfinite(m.vertices).all():
                raise ValueError("Invalid CAD/pose")
            counts = np.bincount(m.edges_unique_inverse)
            audit = {
                "sha256": digest(path),
                "scale_to_m": scale,
                "vertices": len(m.vertices),
                "faces": len(m.faces),
                "watertight": bool(m.is_watertight),
                "winding_consistent": bool(m.is_winding_consistent),
                "boundary_edges": int(sum(counts == 1)),
                "nonmanifold_edges": int(sum(counts > 2)),
                "geometry_repaired": False,
                "extents_m": m.extents.tolist(),
            }
            m.apply_transform(T)
            self.objects.append(
                Object(
                    o["object_id"], o["instance_name"], o["object_name"], m, T, audit
                )
            )

    def frame(self, i):
        rgb = np.asarray(Image.open(self.root / f"images/{i:06d}.png").convert("RGB"))
        d = np.asarray(Image.open(self.root / f"depth_maps/depth_{i:06d}.png"))
        mask = np.asarray(Image.open(self.root / f"sam6d_scene/masks/{i:06d}.png"))
        shape = (self.camera["image_height"], self.camera["image_width"])
        if (
            d.dtype != np.uint16
            or d.shape != shape
            or mask.shape != shape
            or rgb.shape[:2] != shape
        ):
            raise ValueError("Unaligned/nonmetric input")
        return rgb, d.astype(float) * self.depth_to_m, mask

    def audit(self):
        return {
            "frames": len(self.poses),
            "objects": {o.name: o.audit for o in self.objects},
            "K": self.K,
            "depth_to_m": self.depth_to_m,
            "sensor": self.camera.get("device_name"),
            "axes": "OpenCV X right, Y down, Z forward",
        }


