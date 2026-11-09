"""Run full OPT4 validation, reusing IDENTICAL OPT3 raw RGB-D voxel fusion.

Only reconstruction is cached: OPT4 still re-filters the original raw points,
creates obstacle meshes, generates proposals, checks ALL collision gates, and
writes full diagnostics. Fails closed on calibration/config mismatches.
"""
import argparse,json,shutil
from pathlib import Path
from grasp_planner.models import native
from grasp_planner.reconstruction import Reconstruction
from grasp_planner import pipeline
import numpy as np

RECON_KEYS=(
 'seed','reference_frame','pixel_stride','frame_stride','voxel_m',
 'adaptive_voxel','roi_margin_m','depth_min_m','depth_max_m',
 'depth_jump_m','mask_erosion_px','semantic_confidence',
 'min_voxel_observations')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--gripper',action='append',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--cache-run',type=Path,required=True)
    a=p.parse_args();cached=a.cache_run
    old_cfg=json.loads((cached/'effective_config.json').read_text(encoding='utf-8'))
    old_audit=json.loads((cached/'input_audit.json').read_text(encoding='utf-8'))
    cache_raw=cached/'raw'
    for name in ('reconstruction.npz','reconstruction_report.json','frame_metrics.json'):
        if not (cache_raw/name).is_file():raise FileNotFoundError(cache_raw/name)
    def cached_reconstruct(cap,cfg,out,progress=print):
        diffs={k:(old_cfg.get(k),getattr(cfg,k)) for k in RECON_KEYS
               if old_cfg.get(k)!=getattr(cfg,k)}
        if diffs:raise RuntimeError(f'Reconstruction settings have changed; raw fusion cache is INVALID: {diffs}')
        now=native(cap.audit())
        if now!=old_audit:
            raise RuntimeError('Input capture audit mismatched OPT3, refusing reuse of raw reconstruction')
        progress(f'Using matching OPT3 raw RGB-D reconstruction cache: {cache_raw}')
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        for name in ('reconstruction.npz','reconstruction_report.json','frame_metrics.json'):
            shutil.copy2(cache_raw/name,out/name)
        with np.load(out/'reconstruction.npz') as z:
            arrays={k:z[k] for k in z.files}
        return Reconstruction(**arrays,
            report=json.loads((out/'reconstruction_report.json').read_text(encoding='utf-8')),
            frames=json.loads((out/'frame_metrics.json').read_text(encoding='utf-8')))
    pipeline.reconstruct=cached_reconstruct
    summary=pipeline.run(a.dataset,a.config,a.gripper,a.output)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
