#!/usr/bin/env python3
"""Numerically compare a timm DINOv3 checkpoint with its Transformers conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import DINOv3ViTModel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timm_genrecon_features(
    model: torch.nn.Module, image: torch.Tensor
) -> torch.Tensor:
    hidden_states = model.patch_embed(image)
    hidden_states, position_embeddings = model._pos_embed(hidden_states)
    hidden_states = model.norm_pre(hidden_states)
    for block in model.blocks:
        hidden_states = block(hidden_states, rope=position_embeddings)
    return F.layer_norm(hidden_states, hidden_states.shape[-1:])


def _transformers_genrecon_features(
    model: DINOv3ViTModel, image: torch.Tensor
) -> torch.Tensor:
    hidden_states = model.embeddings(image, bool_masked_pos=None)
    position_embeddings = model.rope_embeddings(image)
    for layer in model.layer:
        hidden_states = layer(
            hidden_states,
            position_embeddings=position_embeddings,
        )
    return F.layer_norm(hidden_states, hidden_states.shape[-1:])


def verify(args: argparse.Namespace) -> dict[str, object]:
    source_path = args.source.resolve()
    converted_dir = args.converted_dir.resolve()
    output_weights = converted_dir / "model.safetensors"
    if not source_path.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source_path}")
    if not output_weights.is_file():
        raise FileNotFoundError(f"Converted checkpoint not found: {output_weights}")

    device = torch.device(args.device)
    source_state = load_file(source_path, device="cpu")
    source_model = timm.create_model(args.timm_model, pretrained=False)
    incompatible = source_model.load_state_dict(source_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Source checkpoint is incompatible with the timm model: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    converted_model = DINOv3ViTModel.from_pretrained(
        converted_dir,
        local_files_only=True,
    )
    source_model.eval().to(device)
    converted_model.eval().to(device)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    image = torch.randn(
        1,
        3,
        args.image_size,
        args.image_size,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    with torch.no_grad():
        source_features = _timm_genrecon_features(source_model, image)
        converted_features = _transformers_genrecon_features(converted_model, image)

    source_features = source_features.float().cpu()
    converted_features = converted_features.float().cpu()
    difference = source_features - converted_features
    cosine = F.cosine_similarity(
        source_features.double().flatten(),
        converted_features.double().flatten(),
        dim=0,
    )
    report: dict[str, object] = {
        "schema": "genrecon.dinov3-conversion-verification",
        "schema_version": 1,
        "source_checkpoint": str(source_path),
        "source_sha256": _sha256(source_path),
        "converted_checkpoint": str(output_weights),
        "converted_sha256": _sha256(output_weights),
        "timm_model": args.timm_model,
        "device": str(device),
        "seed": args.seed,
        "input_shape": list(image.shape),
        "output_shape": list(source_features.shape),
        "comparison_path": "pre-final-norm tokens followed by GenRecon functional layer_norm",
        "max_absolute_error": difference.abs().max().item(),
        "mean_absolute_error": difference.abs().mean().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "cosine_similarity": cosine.item(),
        "allclose_atol_2e-5_rtol_2e-5": torch.allclose(
            source_features,
            converted_features,
            atol=2e-5,
            rtol=2e-5,
        ),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("converted_dir", type=Path)
    parser.add_argument(
        "--timm-model",
        default="vit_large_patch16_dinov3_qkvb",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    report = verify(build_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
