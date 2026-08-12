# GT Calibration v1

This directory is the local, untracked build of the frozen 76-unit geometry
calibration plan in `configs/eval/gt_calibration_v1.json`.

## Build state

| Dataset | Tier | Units | Prepared | Existing predictions evaluated | State |
|---|---:|---:|---:|---:|---|
| DA3 ScanNet++ | G0 | 20 | 20 | 20 | Complete |
| ETH3D indoor training | G0 | 7 | 7 | 2 | Complete inputs/GT; five await reconstruction |
| Tanks and Temples Meetingroom | G0 | 1 | 1 | 1 | Representative prediction evaluated in official-crop scope |
| 7-Scenes | G1 | 7 | 7 | 1 | `chess` representative evaluated; six await reconstruction |
| Redwood synthetic | G2 | 2 | 2 | 1 | `livingroom` representative evaluated; one awaits reconstruction |
| DTU | O0 | 15 | 15 | 1 | `scan24` representative evaluated; 14 await reconstruction |
| OmniObject3D requested slots | O0 | 24 | 0 | 0 | OpenXLab login and AK/SK required |
| **Total** | | **76** | **52** | **26** | **24 declared blockers; 26 prepared units await prediction** |

`prepared` means the requested input split, camera metadata, reference geometry,
source provenance and strict manifest are available. It does not mean that a
GenRecon reconstruction has already been run. The current prepared set contains
415 conditioning and 415 heldout images; ETH3D `pipes` has 7+7 because the
public registered sequence contains only 14 views. T&T Meetingroom contributes
8+8 undistorted views in the official laser-GT frame.

## Validation

- `validation.json`: `pass`; 76 manifests, 58 reference files, 339,797,392
  reference vertices, 78,818,470 faces, 830 decoded RGB views, 464 decoded
  depth maps, 26 prediction files totaling 12,059,314,964 bytes, and 147 strict
  JSON files.
- `source_validation.json`: `pass`; 33 source archives/videos and 39,113,567,386
  bytes checked with ZIP/7z CRC or video sample decoding. T&T image/video MD5
  values match the official GCS metadata; the imported 11-scan ZIP, scanner
  positions, alignment Sim(3), and fixed-pose intrinsics artifact also validate.
- `sources/tanks-and-temples/Meetingroom_individual_scans.previous-quota-response.html`
  preserves the old 2,009-byte quota response only as resolved download history.
  It is not a blocker and is not treated as GT.

## Commands

```bash
PYTHONPATH=/tmp/pycolmap-wheel .venv/bin/python \
  tools/calibrate_tnt_meetingroom.py
.venv/bin/python tools/prepare_gt_calibration_datasets.py all
.venv/bin/python tools/validate_gt_calibration_sources.py
.venv/bin/python tools/evaluate_gt_suite.py batch \
  --registry data/gt-calibration-v1/registry.json \
  --output-root reports/generated/gt-calibration-v1/evaluations \
  --num-samples 200000 --max-gt-samples 1000000 --workers -1
.venv/bin/python tools/evaluate_gt_suite.py summarize \
  --output-root reports/generated/gt-calibration-v1/evaluations
```

The four new representative predictions are built and registered separately:

```bash
.venv/bin/python tools/prepare_gt_representative_genrecon.py prepare
.venv/bin/python tools/prepare_gt_representative_genrecon.py preflight
.venv/bin/python tools/run_foundation_genrecon_batch.py all \
  --input-root data/gt-calibration-v1/genrecon-inputs-v1 \
  --output-root outputs/gt-calibration-v1/representative-genrecon-v1 \
  --report-root reports/generated/gt-calibration-v1/representative-genrecon-v1 \
  --track-name GT-pose-foundation-pseudo-geometry --fail-fast
.venv/bin/python tools/prepare_gt_representative_genrecon.py package
.venv/bin/python tools/prepare_gt_representative_genrecon.py validate
.venv/bin/python tools/prepare_gt_calibration_datasets.py register-predictions
```

The geometry evaluator uses `raw-global` scope, seed 42, 200,000
area-weighted prediction samples, up to 1,000,000 reference points, absolute
2/5/10 cm F-scores, bbox-normalized 0.5%/1%/2% scores, and strict JSON. GT
provenance tiers, protocol signatures, scene/instance tracks, and prediction
generation tracks remain separate. T&T uses the distinct
`official-crop-global-reference` scope; the other 25 results use `raw-global`.
Point-cloud GT has no fabricated normal score; the field is `null`.

## Boundaries

- ScanNet++ and ETH3D calibration results are not strict zero-shot evidence for
  GenRecon; they are calibration data with known training-contamination risk.
- 7-Scenes references are clean-depth/KinectFusion-derived (`G1`), Redwood is
  synthetic exact point geometry (`G2`), and DTU/Omni are instance scans
  (`O0`). Their means must not be merged into the `G0` scene table.
- ScanNet++ heldout images are disjoint from the eight conditioning images but
  belong to the same supplied global COLMAP model, so they are not
  geometry-independent heldout evidence.
- T&T heldout images are disjoint from the eight conditioning images, but all
  371 official frames contribute to fixed-pose intrinsics calibration and the
  official global trajectory. They are not intrinsics- or geometry-independent
  heldout evidence.
- The four representative additions use VGGT-1B pseudo geometry and official
  conditioning cameras for Sim(3) alignment. They belong to the
  `GT-pose-foundation-pseudo-geometry` prediction track, not an estimated-pose
  end-to-end track. VGGT-1B is CC-BY-NC-4.0.
- DTU `scan24` has a high F@10 cm value but visibly contains large green/white
  background planes that occlude the building. The geometry score does not
  override this hallucination finding.
- Dataset licenses and terms remain attached to every source and derivative.
