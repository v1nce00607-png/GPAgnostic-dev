#!/usr/bin/env bash
# Gyors próbafutás: 1 tárgy (Rachet_001), 60 jelölt, RG2. Általában néhány perc.
# A scriptet a source_code mappából kell indítani (minden útvonal relatív).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
OUT="${1:-out_smoke_rg2}"
python scripts/opt5_run_cached.py --dataset ../data/0000 --config configs/planner_Q2_smoke.yaml \
    --gripper configs/grippers/rg2.yaml --output "$OUT" --cache-run ../data/reconstruction_cache
python - "$OUT" <<'PY'
import json,sys,collections
rows=json.load(open(sys.argv[1]+'/all_candidates.json'))
print('jelöltek:',len(rows),dict(collections.Counter(r['status'] for r in rows)))
PY
