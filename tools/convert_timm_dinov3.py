#!/usr/bin/env python3
"""Convert the public timm DINOv3 ViT-L qkv-bias weights to Transformers.

The timm checkpoint contains the same inference parameters as the Meta
Transformers ViT-L model, except for the mask token. GenRecon never masks image
patches, so the generated mask token is fixed to zero and explicitly recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel


MODEL_CONFIG = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_register_tokens": 4,
    "image_size": 224,
    "patch_size": 16,
    "query_bias": True,
    "key_bias": False,
    "value_bias": True,
    "rope_theta": 100.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_state_dict(source: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {
        "embeddings.cls_token": source["cls_token"],
        "embeddings.register_tokens": source["reg_token"],
        "embeddings.patch_embeddings.weight": source["patch_embed.proj.weight"],
        "embeddings.patch_embeddings.bias": source["patch_embed.proj.bias"],
        "norm.weight": source["norm.weight"],
        "norm.bias": source["norm.bias"],
    }

    for index in range(MODEL_CONFIG["num_hidden_layers"]):
        src = f"blocks.{index}"
        dst = f"layer.{index}"
        q_weight, k_weight, v_weight = source[f"{src}.attn.qkv.weight"].chunk(3, dim=0)
        converted.update(
            {
                f"{dst}.norm1.weight": source[f"{src}.norm1.weight"],
                f"{dst}.norm1.bias": source[f"{src}.norm1.bias"],
                f"{dst}.attention.q_proj.weight": q_weight,
                f"{dst}.attention.q_proj.bias": source[f"{src}.attn.q_bias"],
                f"{dst}.attention.k_proj.weight": k_weight,
                f"{dst}.attention.v_proj.weight": v_weight,
                f"{dst}.attention.v_proj.bias": source[f"{src}.attn.v_bias"],
                f"{dst}.attention.o_proj.weight": source[f"{src}.attn.proj.weight"],
                f"{dst}.attention.o_proj.bias": source[f"{src}.attn.proj.bias"],
                f"{dst}.layer_scale1.lambda1": source[f"{src}.gamma_1"],
                f"{dst}.norm2.weight": source[f"{src}.norm2.weight"],
                f"{dst}.norm2.bias": source[f"{src}.norm2.bias"],
                f"{dst}.mlp.up_proj.weight": source[f"{src}.mlp.fc1.weight"],
                f"{dst}.mlp.up_proj.bias": source[f"{src}.mlp.fc1.bias"],
                f"{dst}.mlp.down_proj.weight": source[f"{src}.mlp.fc2.weight"],
                f"{dst}.mlp.down_proj.bias": source[f"{src}.mlp.fc2.bias"],
                f"{dst}.layer_scale2.lambda1": source[f"{src}.gamma_2"],
            }
        )
    return converted


def convert_checkpoint(
    source_path: Path,
    output_dir: Path,
    *,
    source_repo: str,
    source_revision: str,
    base_pipeline_config: Path,
) -> dict[str, Any]:
    if not source_path.is_file():
        raise FileNotFoundError(f"Source safetensors file not found: {source_path}")
    if not base_pipeline_config.is_file():
        raise FileNotFoundError(f"Base pipeline config not found: {base_pipeline_config}")

    source = load_file(source_path, device="cpu")
    converted = convert_state_dict(source)
    config = DINOv3ViTConfig(**MODEL_CONFIG)
    model = DINOv3ViTModel(config)
    converted["embeddings.mask_token"] = torch.zeros_like(model.embeddings.mask_token)
    incompatible = model.load_state_dict(converted, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Converted checkpoint is incompatible: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval().save_pretrained(output_dir, safe_serialization=True)

    pipeline = json.loads(base_pipeline_config.read_text(encoding="utf-8"))
    pipeline["args"]["image_cond_model"]["args"]["model_name"] = str(output_dir.resolve())
    pipeline_path = output_dir / "pipeline.json"
    pipeline_path.write_text(json.dumps(pipeline, indent=2) + "\n", encoding="utf-8")

    output_weights = output_dir / "model.safetensors"
    report = {
        "schema": "genrecon.timm-dinov3-conversion",
        "schema_version": 1,
        "source_repo": source_repo,
        "source_revision": source_revision,
        "source_path": str(source_path.resolve()),
        "source_sha256": _sha256(source_path),
        "source_tensor_count": len(source),
        "source_parameter_count": sum(tensor.numel() for tensor in source.values()),
        "output_path": str(output_dir.resolve()),
        "output_sha256": _sha256(output_weights),
        "output_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "mask_token": {
            "source_present": False,
            "output_initialization": "all zeros",
            "used_by_genrecon": False,
            "reason": "GenRecon calls embeddings with bool_masked_pos=None.",
        },
        "pipeline_config": str(pipeline_path.resolve()),
    }
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--source-repo",
        default="timm/vit_large_patch16_dinov3_qkvb.lvd1689m",
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--base-pipeline-config",
        type=Path,
        default=Path("configs/pipelines/original.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = convert_checkpoint(
        args.source,
        args.output,
        source_repo=args.source_repo,
        source_revision=args.source_revision,
        base_pipeline_config=args.base_pipeline_config,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
