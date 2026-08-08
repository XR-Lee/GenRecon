#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

SCENE_ID=${SCENE_ID:-286b55a2bf}
SCENE_ROOT=${SCENE_ROOT:-data/da3-adapted/$SCENE_ID}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/$SCENE_ID/reconstruction}
PROFILE_ROOT=${PROFILE_ROOT:-reports/generated/$SCENE_ID}
PROJ_BATCH_VOXELS=${PROJ_BATCH_VOXELS:-256}
PIPELINE_CONFIG=${PIPELINE_CONFIG:-configs/pipelines/original.json}
JOINT_DECODE_ARGS=()
if [[ -n "${JOINT_DECODE_MAX_CHUNKS_PER_GROUP:-}" ]]; then
  JOINT_DECODE_ARGS+=(--joint_decode_max_chunks_per_group "$JOINT_DECODE_MAX_CHUNKS_PER_GROUP")
fi
if [[ -n "${JOINT_DECODE_MAX_INFLATED_VOXELS:-}" ]]; then
  JOINT_DECODE_ARGS+=(--joint_decode_max_inflated_voxels "$JOINT_DECODE_MAX_INFLATED_VOXELS")
fi

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.6}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6}
export HF_HOME=${HF_HOME:-$ROOT_DIR/data/hf-cache}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

mkdir -p "$OUTPUT_ROOT" "$PROFILE_ROOT"

.venv/bin/python tools/profile_command.py \
  --cwd "$ROOT_DIR" \
  --label "genrecon-$SCENE_ID" \
  --log "$PROFILE_ROOT/reconstruct.log" \
  --result "$PROFILE_ROOT/reconstruct_profile.json" \
  --expect-file "$OUTPUT_ROOT/mesh.ply" \
  --expect-file "$OUTPUT_ROOT/to_glb_inputs.pt" \
  --expect-file "$OUTPUT_ROOT/chunk_inputs.pt" \
  -- \
  .venv/bin/python reconstruct_scene.py \
    --mode Scannet_iphone \
    --path "$SCENE_ROOT" \
    --output_path "$OUTPUT_ROOT" \
    --ss_ckpt weights/ss/checkpoints/sparse_structure.pt \
    --shape_ckpt weights/shape/checkpoints/shape_slat.pt \
    --tex_ckpt weights/texture/checkpoints/texture_slat.pt \
    --ss_config configs/gen/ss_flow_img/genrecon.json \
    --shape_config configs/gen/slat_flow_img2shape/genrecon_512.json \
    --tex_config configs/gen/slat_flow_imgshape2tex/genrecon_512.json \
    --pipeline_config "$PIPELINE_CONFIG" \
    --num_imgs_per_scene 8 \
    --center_crop \
    --seed 42 \
    --chunk_size_factor 1.11 \
    --min_overlap_factor 4 \
    --proj_batch_voxels "$PROJ_BATCH_VOXELS" \
    "${JOINT_DECODE_ARGS[@]}" \
    --save_imgs

.venv/bin/python tools/evaluate_mesh.py \
  "$OUTPUT_ROOT/mesh.ply" \
  "$SCENE_ROOT/scans/mesh_aligned_0.05.ply" \
  --num-samples 200000 \
  --seed 42 \
  --output "$PROFILE_ROOT/mesh_metrics_unclipped.json"

.venv/bin/python tools/profile_command.py \
  --cwd "$ROOT_DIR" \
  --label "genrecon-glb-$SCENE_ID" \
  --log "$PROFILE_ROOT/glb.log" \
  --result "$PROFILE_ROOT/glb_profile.json" \
  --expect-file "$OUTPUT_ROOT/scene.glb" \
  -- \
  .venv/bin/python chunked_to_glb.py \
    --inputs "$OUTPUT_ROOT/to_glb_inputs.pt" \
    --chunk_inputs "$OUTPUT_ROOT/chunk_inputs.pt" \
    --output_dir "$OUTPUT_ROOT"
