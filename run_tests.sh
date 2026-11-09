#!/usr/bin/env bash
# Egység- és integrációs tesztek (a test_results.py kihagyva: régi eredménymappát igényel).
set -euo pipefail
cd "$(dirname "$0")"
PYTHONPATH=. python -m pytest tests -q -p no:cacheprovider --deselect tests/test_opt7_cad.py --ignore tests/test_results.py
