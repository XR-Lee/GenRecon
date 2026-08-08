#!/usr/bin/env bash
set -euo pipefail

root="outputs/eth3d/lecture_room/fallback_instance_experiment"
configs="$root/configs"
for variant in random_01 random_02 random_03 adaptive_3; do
  out="$root/a0_${variant}"
  mkdir -p "$out"
  .venv/bin/python tools/profile_command.py \
    --log "$out/reconstruct.log" \
    --result "$out/reconstruct_profile.json" \
    --label "lecture-room-a0-${variant}" \
    --sample-interval 1 \
    --progress-interval 30 \
    --expect-file "$out/mesh.ply" \
    -- \
    .venv/bin/python reconstruct_scene.py \
      --path data/eth3d/lecture_room \
      --mode Iphone \
      --colmap_subdir colmap \
      --output_path "$out" \
      --ss_ckpt weights/ss/checkpoints/sparse_structure.pt \
      --shape_ckpt weights/shape/checkpoints/shape_slat.pt \
      --tex_ckpt weights/texture/checkpoints/texture_slat.pt \
      --pipeline 512 \
      --pipeline_config data/hf-cache/converted/dinov3-vitl16-timm-qkvb/pipeline.json \
      --num_imgs_per_scene 16 \
      --seed 42 \
      --save_imgs \
      --min_overlap_factor 4 \
      --chunk_size_factor 1.04 \
      --min_points_per_chunk 30 \
      --max_reproj_error 2 \
      --min_track_len 3 \
      --occ_threshold -1 \
      --proj_batch_voxels 256 \
      --joint_decode_max_chunks_per_group 5 \
      --joint_decode_max_inflated_voxels 30000 \
      --chunk_layout_json "$configs/${variant}.json" \
      --fixed_cond2d_scene_views "$configs/fixed_cond2d_scene_views.json" \
      --overlap_diagnostics "$out/overlap_diagnostics.json" \
      --overlap_diagnostics_roi_box -2 2.25 -2.15 -1 -0.91 -0.84 \
      --overlap_diagnostics_roi_box -2 2.25 -1.45 -1.05 -1.75 -0.92
done
