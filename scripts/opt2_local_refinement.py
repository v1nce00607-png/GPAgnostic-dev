"""Revalidate local grasp-pose offsets against the original geometric gates.

Designed for an *existing* OPT1 run: NO RGB-D reconstruction, NO gate bypass,
NO modification of input results. For open CAD, raw triangle-surface collision
is only a diagnostic, never an alternative PASS criterion.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from itertools import product

import numpy as np
import trimesh
import yaml

from grasp_planner.models import Capture, check, verdict
from grasp_planner.geometry_config import GeometryConfig
from grasp_planner.pipeline import load_reconstruction
from grasp_planner.geometry_collision import Scene
from grasp_planner.geometry_planner import approach, GATES
from grasp_planner.ur_hardware import CADGripper, HardwareSpec


def offsets(step_mm: float, radius_mm: float):
    """Axis/diagonal small translations, ordered by increasing magnitude."""
    if step_mm <= 0 or radius_mm < 0:
        raise ValueError('Step must be positive and radius nonnegative')
    n = int(np.floor(radius_mm / step_mm + 1e-9))
    values = np.arange(-n, n + 1, dtype=float) * step_mm * .001
    delta = [np.asarray(q, dtype=float) for q in product(values, repeat=3)
             if np.linalg.norm(q) <= radius_mm * .001 + 1e-12]
    return sorted(delta, key=lambda x: (round(np.linalg.norm(x), 12), tuple(np.round(x, 12))))


def contact_key(row, decimals=4):
    """Orientation-invariant fingerprint of the same physical CAD ray contacts.

    Default 0.1mm quantization; suitable only for uniqueness reporting, never
    a collision validation or assertion that grasps are physically equivalent.
    """
    pts = [tuple(np.round(np.asarray(p, dtype=float), decimals))
           for p in row['contacts_world_m']]
    return row['hardware'], row['object_instance'], tuple(sorted(pts))


def final_gate_status(checks):
    """Use the project's exact 10-gate logic; never infer a success."""
    return verdict(checks, GATES)


def sha256(path):
    d = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            d.update(b)
    return d.hexdigest()


def fmt(v):
    return '—' if v is None else str(v)


def surface_probe(scene, gripper, M, q):
    """Open CAD surface intersection flag; NEVER a solid-clearance verdict."""
    manager = trimesh.collision.CollisionManager()
    manager.add_object(scene.obj.name, scene.obj.mesh)
    hits = []
    for part in gripper.parts(M, q, scene.cfg.target_penetration_tolerance_m):
        mesh = part.visual_mesh() if part.raw.is_watertight else part.mesh()
        if manager.in_collision_single(mesh):
            hits.append(part.name)
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--results', type=Path, required=True,
                    help='Existing OPT1 output directory containing reconstruction.npz and all_candidates.json')
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--gripper', type=Path, required=True, action='append')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--step-mm', type=float, default=.25)
    ap.add_argument('--radius-mm', type=float, default=.5)
    ap.add_argument('--max-candidates', type=int, default=16)
    args = ap.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        ap.error('Output directory must be new/empty; original experiment is read-only.')
    if args.max_candidates < 1 or args.radius_mm > 2.0:
        ap.error('max-candidates >= 1 and radius-mm <= 2 are required')
    shifts = offsets(args.step_mm, args.radius_mm)
    if not shifts:
        ap.error('No offsets')
    args.output.mkdir(parents=True, exist_ok=True)

    cfg = GeometryConfig.model_validate(yaml.safe_load(args.config.read_text(encoding='utf-8')))
    cap = Capture(args.dataset)
    rec = load_reconstruction(args.results)
    src = args.results / 'all_candidates.json'
    rows = json.loads(src.read_text(encoding='utf-8'))
    baseline = [r for r in rows if r['checks']['environment']['status'] == 'PASS']
    baseline.sort(key=lambda r: (-r['score'], r['id']))
    subset = baseline[:args.max_candidates]
    specs = [HardwareSpec.model_validate(yaml.safe_load(x.read_text(encoding='utf-8')))
             for x in args.gripper]
    grippers = {s.name: CADGripper(s) for s in specs}
    objects = {o.name: o for o in cap.objects}
    planes = [trimesh.load(x, force='mesh')
              for x in sorted(args.results.glob('obstacle_plane_*.ply'))]
    camera_transform = np.asarray(cap.by_id[cfg.reference_frame]['camera_from_world'])
    scenes = {}
    results = []
    best_by_status = Counter()

    print(f'OPT2: {len(rows)} saved candidates, {len(baseline)} environment PASS, '
          f'{len(subset)} selected, {len(shifts)} offsets each.', flush=True)
    for i, r in enumerate(subset, 1):
        target_name = r['object_instance']
        if target_name not in scenes:
            scenes[target_name] = Scene(cap, rec, objects[target_name], cfg, planes)
        scene = scenes[target_name]
        g = grippers[r['hardware']]
        M = np.asarray(r['world_from_mount'], dtype=float)
        q = float(r['joint'])
        qo = float(r['open_joint'])
        _, tcp, finger_contacts = g.contact_geometry(q)
        world_contacts = np.asarray(r['contacts_world_m'], dtype=float)
        original_hull = scene.target_clear(g.parts(M, q, cfg.target_penetration_tolerance_m))
        original_surface_hits = surface_probe(scene, g, M, q)
        records = []
        for local_delta in shifts:
            candidate_M = M.copy()
            # LOCAL frame translation. No rotation, no joint change.
            candidate_M[:3, 3] += M[:3, :3] @ local_delta
            test_contact_world = finger_contacts @ candidate_M[:3, :3].T + candidate_M[:3, 3]
            # Both gripper contact patches must still lie close to their intended CAD contacts.
            # Robust to left/right swap; the assignment is not taken for granted.
            pairing = min(max(np.linalg.norm(test_contact_world[j] - world_contacts[p[j]])
                              for j in (0, 1)) for p in ((0, 1), (1, 0)))
            if pairing > .000501:
                continue
            # Extra proximity check to the original CAD surface for both tip surface points.
            _, distances, _ = trimesh.proximity.closest_point(scene.obj.mesh, test_contact_world)
            if float(np.max(distances)) > .000501:
                continue
            tar = scene.target_clear(g.parts(candidate_M, q, cfg.target_penetration_tolerance_m))
            env = scene.clear(g.parts(candidate_M, q))
            checks = {k: dict(v) for k, v in r['checks'].items()}
            checks['target_body'] = tar
            checks['environment'] = env
            checks['contact_geometry'] = check(True,
                criterion='original CAD opposing contacts + both OEM fingertip contact points within 0.501mm',
                position_deviation_m=float(pairing),
                max_cad_surface_distance_m=float(np.max(distances)),
                original_alignment=r['checks']['contact_geometry'].get('normal_alignment'))
            traj_status = None
            if (tar['status'] == 'PASS' and env['status'] == 'PASS'
                    and checks['self_collision']['status'] == 'PASS'):
                closure = scene.closure(g, candidate_M, qo, q)
                traj = approach(scene, g, candidate_M, qo, cfg, camera_transform, tcp)
                checks['closure'] = closure
                checks['approach'] = {k: v for k, v in traj.items()
                                      if k not in ('samples', 'waypoints_world_m', 'world_from_mount_at_goal')}
                traj_status = traj['status']
            else:
                checks['approach'] = check(None, reason='A static gate rejected this local pose')
                checks['closure'] = check(None, reason='A static gate rejected this local pose')
            verdict_status = final_gate_status(checks)
            best_by_status[(tar['status'], env['status'], verdict_status)] += 1
            records.append(dict(shift_local_mm=(local_delta * 1000).round(5).tolist(),
                                contact_position_deviation_mm=round(pairing*1000, 6),
                                original_cad_surface_distance_mm=round(float(np.max(distances))*1000, 6),
                                target_body=tar, environment=env,
                                approach=checks['approach']['status'],
                                closure=checks['closure']['status'],
                                status=verdict_status,
                                world_from_mount=candidate_M.tolist()))
        def ranking(z):
            return (z['status'] == 'PASS',
                    z['target_body']['status'] == 'PASS' and z['environment']['status'] == 'PASS',
                    z['target_body']['status'] == 'PASS',
                    z['environment']['status'] == 'PASS',
                    -np.linalg.norm(z['shift_local_mm']))
        records.sort(key=ranking, reverse=True)
        results.append(dict(original_candidate_id=r['id'],
            original_contact_pair_key=str(contact_key(r)),
            original_target_body=original_hull,
            original_surface_intersecting_parts=original_surface_hits,
            target_cad_watertight=bool(scene.obj.mesh.is_watertight),
            target_collision_model=scene.target_model,
            tried_offsets=len(records),
            best=records[0] if records else None,
            full_pass_variants=sum(s['status'] == 'PASS' for s in records),
            joint_static_pass_variants=sum(
                s['target_body']['status'] == s['environment']['status'] == 'PASS'
                for s in records)))
        print(f'[{i}/{len(subset)}] {r["id"]}: tried {len(records)}, '
              f'joint static PASS {results[-1]["joint_static_pass_variants"]}, '
              f'Full PASS {results[-1]["full_pass_variants"]}; '
              f'raw surface intersections {original_surface_hits}', flush=True)

    provenance = dict(dataset=str(args.dataset), opt1_results=str(args.results),
        config=str(args.config), sha256_opt1_candidates=sha256(src),
        sha256_config=sha256(args.config),
        assumptions=['Original scene, sensor uncertainties, CAD and conservative target collision gate unchanged',
                     'Open CAD triangle-surface intersection is diagnostic ONLY; cannot certify solid emptiness',
                     'No full PASS from unverified final or trajectory gates',
                     'Local translations limited to 0.5mm paired contact deviation and CAD surface distance',
                     'Real robot execution, arm IK, unseen environment and dynamics not evaluated'],
        step_mm=args.step_mm, radius_mm=args.radius_mm,
        selected_environment_pass=len(subset), total_saved_environment_pass=len(baseline),
        unique_contact_groups=len({contact_key(r) for r in subset}),
        search_outcomes={str(k): n for k, n in best_by_status.items()})
    payload = dict(provenance=provenance, candidates=results)
    (args.output / 'opt2_local_refinement.json').write_text(
        json.dumps(payload, indent=2), encoding='utf-8')
    lines = ['# OPT2 – offline local CAD revalidation', '',
             f'- OPT1 saved environment PASS: **{len(baseline)}**',
             f'- Analyzed candidates: **{len(subset)}**',
             f'- Distinct gripper/object/contact-pair groups: **{provenance["unique_contact_groups"]}**',
             f'- Tested translations per selected candidate: **{len(shifts)}**',
             f'- Candidates with at least one joint static PASS: **{sum(x["joint_static_pass_variants"]>0 for x in results)}**',
             f'- Candidates with at least one fully revalidated PASS: **{sum(x["full_pass_variants"]>0 for x in results)}**', '',
             '| Candidate | Raw CAD intersection at original pose | Joint static PASS offsets | Full PASS offsets |',
             '|---|---|---:|---:|']
    for r in results:
        hits = ', '.join(r['original_surface_intersecting_parts']) or '(none detected)'
        lines.append(f'| `{r["original_candidate_id"]}` | {hits} | '
                     f'{r["joint_static_pass_variants"]} | {r["full_pass_variants"]} |')
    lines.extend(['', '## Validation notes', '',
        '- Every claimed local PASS is checked by the original `Scene.target_clear`, `Scene.clear`, `Scene.closure` and `approach` path gates.',
        '- Raw CAD surface collision is a diagnostic **only**, because the input target mesh is non-watertight.',
        '- Small pose changes are subject to paired fingertip contact coincidence and original CAD surface proximity limits.',
        '- Count by distinct contact groups; repeated tilt/mirror candidates are not independent grasp locations.',
        '- The original results and baseline are not altered.',
        '- No dynamics, robot-arm inverse kinematics, or unknown-space validation.'])
    (args.output / 'OPT2_local_refinement.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Wrote: {args.output / "OPT2_local_refinement.md"}', flush=True)


if __name__ == '__main__':
    main()
