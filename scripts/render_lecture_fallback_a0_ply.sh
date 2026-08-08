#!/usr/bin/env bash
set -euo pipefail
root="outputs/eth3d/lecture_room/fallback_instance_experiment"
for variant in current random_01 random_02 random_03 adaptive_3; do
  out="$root/a0_${variant}/fidelity"
  .venv/bin/python tools/evaluate_view_fidelity.py \
    --stage ply \
    --ply "$root/a0_${variant}/mesh.ply" \
    --glb outputs/eth3d/lecture_room/final/scene.glb \
    --cameras data/eth3d/lecture_room/colmap/cameras.txt \
    --images data/eth3d/lecture_room/colmap/images.txt \
    --points data/eth3d/lecture_room/colmap/points3D.txt \
    --images-root data/eth3d/lecture_room/rgb \
    --input-cameras-json "$root/a0_${variant}/cameras.json" \
    --output "$out" \
    --scene-label "lecture_room_a0_${variant}" \
    --width 960 \
    --no-lpips
done
