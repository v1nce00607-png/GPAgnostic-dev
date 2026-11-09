#!/usr/bin/env bash
# A teljes Q2 futás (3 megfogó, 4 tárgy, 600 jelölt/tárgy) + szabványos összevont jelentés.
# A source_code mappából indítandó; kb. 25 perc 3 párhuzamos folyamattal, ~10 GB memória.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
for g in rg2 rg6 2fg7; do
  python scripts/opt5_run_cached.py --dataset ../data/0000 --config configs/planner_Q2_final.yaml \
      --gripper configs/grippers/$g.yaml --output out_Q2_$g --cache-run ../data/reconstruction_cache > log_Q2_$g.txt 2>&1 &
  sleep 20
done
wait
python scripts/opt7_combined_report.py --dataset ../data/0000 --runs out_Q2_2fg7 out_Q2_rg2 out_Q2_rg6 \
    --grippers configs/grippers/2fg7.yaml configs/grippers/rg2.yaml configs/grippers/rg6.yaml --output out_Q2_combined
echo "Kész: out_Q2_combined/report.html"
