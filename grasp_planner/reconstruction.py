"""Calibrated metric RGB-D voxel fusion; supplied poses, no hidden registration."""

from dataclasses import dataclass
import time, numpy as np, trimesh
from scipy.spatial import cKDTree
from .models import transform, unit, save, check


@dataclass
class Reconstruction:
    points: np.ndarray
    colors: np.ndarray
    labels: np.ndarray
    confidence: np.ndarray
    counts: np.ndarray
    plane: np.ndarray
    report: dict
    frames: list

    def save(self, path):
        np.savez_compressed(
            path,
            points=self.points,
            colors=self.colors,
            labels=self.labels,
            confidence=self.confidence,
            counts=self.counts,
            plane=self.plane,
        )


def backproject(depth, K, stride=1):
    y, x = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[::stride, ::stride]
    return np.stack(
        [(x - K[0, 2]) * z / K[0, 0], (y - K[1, 2]) * z / K[1, 1], z], axis=-1
    )


def reduce(points, colors, labels, voxel, nlabels):
    _, inv = np.unique(
        np.floor(points / voxel).astype(np.int32), axis=0, return_inverse=True
    )
    n = np.bincount(inv)
    p = np.stack([np.bincount(inv, weights=points[:, i]) / n for i in range(3)], 1)
    c = np.stack([np.bincount(inv, weights=colors[:, i]) / n for i in range(3)], 1)
    votes = np.zeros((len(n), nlabels), np.int32)
    np.add.at(votes, (inv, labels), 1)
    return p, c, n, votes


def fit_plane(points, cameras, seed=0, threshold=0.005):
    rng = np.random.default_rng(seed)
    p = points[rng.choice(len(points), min(len(points), 20000), replace=False)]
    best = np.zeros(len(p), bool)
    for _ in range(350):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-10:
            continue
        n = unit(n)
        inside = np.abs((p - a) @ n) < threshold
        if inside.sum() > best.sum():
            best = inside
    if best.sum() < 100:
        raise ValueError("No supported plane")
    center = p[best].mean(0)
    _, _, vh = np.linalg.svd(p[best] - center, full_matrices=False)
    n = vh[-1]
    if n @ (np.median(cameras, axis=0) - center) < 0:
        n = -n
    d = -n @ center
    return np.r_[n, d], {
        "inliers": int(best.sum()),
        "sample_count": len(p),
        "rms_m": float(np.sqrt(np.mean((p[best] @ n + d) ** 2))),
        "gravity_inferred_from_plane": True,
    }


def reconstruct(cap, cfg, out, progress=print):
    t = time.perf_counter()
    lo = np.min([o.mesh.bounds[0] for o in cap.objects], axis=0) - cfg.roi_margin_m
    hi = np.max([o.mesh.bounds[1] for o in cap.objects], axis=0) + cfg.roi_margin_m
    nl = max(o.id for o in cap.objects) + 1
    chunks = []
    frames = []
    for count, po in enumerate(cap.poses[:: cfg.frame_stride]):
        i = po["frame_id"]
        rgb, d, mask = cap.frame(i)
        pixel_good = d > 0
        preprocessing = {}
        if hasattr(cfg,"depth_jump_m"):
            from .cleaning import preprocess
            pixel_good, mask, preprocessing = preprocess(d,mask,cfg)
        pc = backproject(d, cap.K, cfg.pixel_stride).reshape(-1, 3)
        pw = transform(pc, po["world_from_camera"])
        lab = mask[:: cfg.pixel_stride, :: cfg.pixel_stride].ravel().astype(int)
        col = rgb[:: cfg.pixel_stride, :: cfg.pixel_stride].reshape(-1, 3)
        if np.any(lab >= nl):
            raise ValueError("Unknown label")
        good = (
            pixel_good[::cfg.pixel_stride,::cfg.pixel_stride].ravel()
            & (pc[:, 2] > cfg.depth_min_m)
            & (pc[:, 2] < cfg.depth_max_m)
            & np.all((pw >= lo) & (pw <= hi), axis=1)
        )
        p, c, n, v = reduce(pw[good], col[good], lab[good], cfg.voxel_m, nl)
        chunks.append((p, c, v.argmax(1)))
        frames.append(
            {
                "frame_id": i,
                **preprocessing,
                "depth_valid_fraction": float(
                    np.mean((d > cfg.depth_min_m) & (d < cfg.depth_max_m))
                ),
                "roi_samples": int(good.sum()),
                "roi_voxels": len(p),
                **{
                    f"instance_{o.id}_samples": int(sum(lab[good] == o.id))
                    for o in cap.objects
                },
            }
        )
        if count % 40 == 0:
            progress(f"RGB-D {count+1}/{len(cap.poses[::cfg.frame_stride])}")
    p, c, n, v = reduce(
        np.concatenate([x[0] for x in chunks]),
        np.concatenate([x[1] for x in chunks]),
        np.concatenate([x[2] for x in chunks]),
        cfg.voxel_m,
        nl,
    )
    del chunks
    labels = v.argmax(1)
    confidence = v.max(1) / n
    labels[
        (confidence < cfg.semantic_confidence) | (n < cfg.min_voxel_observations)
    ] = 0
    cameras = np.array([np.array(x["world_from_camera"])[:3, 3] for x in cap.poses])
    plane, fit = fit_plane(p[labels == 0], cameras, cfg.seed)
    cloud = trimesh.load(cap.root / "reconstruction/mvs_filtered_metric.ply")
    ref = np.asarray(cloud.vertices)
    ref = ref[np.all((ref >= lo) & (ref <= hi), axis=1)]
    rng = np.random.default_rng(cfg.seed)
    sample = p[rng.choice(len(p), min(len(p), 50000), replace=False)]
    dist = cKDTree(ref).query(sample, workers=1)[0]
    report = {
        "method": "metric RGB-D voxel fusion with supplied camera poses",
        "frame_count": len(frames),
        "pixel_stride": cfg.pixel_stride,
        "points": len(p),
        "voxel_m": cfg.voxel_m,
        "roi_bounds_m": [lo, hi],
        "label_counts": {str(i): int(sum(labels == i)) for i in range(nl)},
        "plane": plane,
        "plane_fit": fit,
        "views_per_voxel_median": float(np.median(n)),
        "mvs_comparison": {
            "samples": len(dist),
            "median_m": float(np.median(dist)),
            "p95_m": float(np.quantile(dist, 0.95)),
            "meaning": "RGBD-to-MVS consistency, not independent ground truth",
        },
        "seconds": time.perf_counter() - t,
    }
    result = Reconstruction(
        p, c.astype(np.uint8), labels, confidence, n, plane, report, frames
    )
    result.save(out / "reconstruction.npz")
    save(out / "reconstruction_report.json", report)
    save(out / "frame_metrics.json", frames)
    return result


def alignment(cap, rec, cfg):
    result = {}
    for o in cap.objects:
        p = rec.points[rec.labels == o.id]
        if not len(p):
            result[o.name] = check(False, reason="No observed target points")
            continue
        ids = np.linspace(0, len(p) - 1, min(len(p), 3000)).astype(int)
        _, d, _ = trimesh.proximity.closest_point(o.mesh, p[ids])
        med = float(np.median(d))
        q = float(np.quantile(d, 0.95))
        result[o.name] = check(
            med <= cfg.alignment_median_limit_m and q <= cfg.alignment_p95_limit_m,
            observed_points=len(p),
            tested_points=len(ids),
            median_m=med,
            p95_m=q,
            meaning="automatic CAD/depth consistency, not ground-truth accuracy",
        )
    return result
