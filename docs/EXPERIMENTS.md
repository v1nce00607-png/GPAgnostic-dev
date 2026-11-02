# Experiment registry

| Version | Scope | Environment PASS | Target-body PASS | Joint static PASS | Full geometric PASS |
|---|---|---:|---:|---:|---:|
| Baseline | 2,400 candidates, capture `0000` | 4 | 113 | 0 | 0 |
| OPT1 | 2,400 candidates, same reference capture | 16 | 229 | 0 | 0 |
| OPT2 | 16 OPT1 environment-PASS candidates, 7 local offsets each | — | — | 0 | 0 |
| OPT3 | 2,400 candidates, same reference capture | 16 | 245 | 0 | 0 |
| OPT4 | Experimental ROI occupancy removal; comparison pending | — | — | — | — |

**Do not compare OPT2 subset counts to 2,400-candidate full runs.**

## Interpretation
- OPT1 increased Environment PASS from 4 to 16, but no jointly valid static candidates were found.
- OPT3 increased Target-body PASS from 229 to 245; 16 rows were accepted with a model-dependent non-watertight-CAD fingertip contact rule. Environment PASS remained 16 and Full PASS remained 0.
- OPT4 is an experimental occupancy-ablation branch; removing unlabeled data can improve reported clearance without proving the removed locations are empty.

## Required run metadata
Store git commit/tag, configuration SHA-256, dataset/camera revision, CAD source revision, dependency versions, random seed, scene policy, thresholds, result SHA-256, date and interpretation. Each `experiments/<version>/` README explains changes. For private raw data, retain paths locally and publish only non-sensitive hashes/relative identifiers.
