#!/usr/bin/env python3
"""Export frame-exact source/GenRecon comparison videos.

Only foundation views with explicit predicted cameras are rendered. The videos
use fixed-rate inspection playback; source timestamps remain in the frame
manifest and side-by-side footer.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import imageio_ffmpeg
except ImportError:  # pragma: no cover - the project environment provides it
    imageio_ffmpeg = None

try:
    from tools.evaluate_view_fidelity import read_colmap_images
except ModuleNotFoundError:
    from evaluate_view_fidelity import read_colmap_images

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOUNDATION_ROOT = ROOT / "data" / "internet-zero-shot" / "foundation-sfm-v1"
DEFAULT_PREPRODUCTS_ROOT = ROOT / "data" / "internet-zero-shot" / "sfm-preproducts-v1"
DEFAULT_GENRECON_ROOT = ROOT / "outputs" / "internet-zero-shot" / "foundation-genrecon-v1"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "internet-zero-shot" / "foundation-video-comparisons-v1"
DEFAULT_REPORT_ROOT = (
    ROOT / "reports" / "generated" / "internet-zero-shot" / "foundation-video-comparisons-v1"
)
SCHEMA_VERSION = 1
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {value} in {path}")
        ),
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_to(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def ffmpeg_executable(explicit: str | None = None) -> str:
    if explicit:
        path = shutil.which(explicit) or explicit
        if not Path(path).is_file():
            raise FileNotFoundError(f"FFmpeg does not exist: {path}")
        return str(path)
    if imageio_ffmpeg is not None:
        return imageio_ffmpeg.get_ffmpeg_exe()
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError("FFmpeg is unavailable")
    return path


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if FONT_PATH.is_file():
        return ImageFont.truetype(str(FONT_PATH), size=size)
    return ImageFont.load_default()


def fitting_font(draw: ImageDraw.ImageDraw, text: str, maximum_width: int) -> ImageFont.ImageFont:
    for size in range(16, 9, -1):
        candidate = font(size)
        bounds = draw.textbbox((0, 0), text, font=candidate)
        if bounds[2] - bounds[0] <= maximum_width:
            return candidate
    return font(9)


def even_height(height: int) -> int:
    return height + height % 2


def padded_rgb(image: Image.Image, width: int, height: int) -> Image.Image:
    rgb = image.convert("RGB")
    if rgb.size != (width, height):
        rgb = rgb.resize((width, height), Image.Resampling.LANCZOS)
    output_height = even_height(height)
    if output_height == height:
        return rgb
    output = Image.new("RGB", (width, output_height), (0, 0, 0))
    output.paste(rgb, (0, 0))
    return output


def padded_mask(image: Image.Image, width: int, height: int) -> Image.Image:
    mask = image.convert("L")
    if mask.size != (width, height):
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
    output_height = even_height(height)
    if output_height == height:
        return mask
    output = Image.new("L", (width, output_height), 0)
    output.paste(mask, (0, 0))
    return output


def compose_comparison_frame(
    original: Image.Image,
    reconstruction: Image.Image,
    *,
    candidate_id: str,
    frame_name: str,
    source_timestamp_s: float,
    role: str,
    coverage: float,
    frame_index: int,
    frame_count: int,
) -> Image.Image:
    if original.size != reconstruction.size:
        raise ValueError(f"Panel dimensions differ: {original.size} != {reconstruction.size}")
    panel_width, panel_height = original.size
    header_height, footer_height = 48, 48
    canvas = Image.new(
        "RGB", (panel_width * 2, panel_height + header_height + footer_height), (18, 20, 22)
    )
    canvas.paste(original.convert("RGB"), (0, header_height))
    canvas.paste(reconstruction.convert("RGB"), (panel_width, header_height))
    draw = ImageDraw.Draw(canvas)
    heading_font = font(21)
    draw.text((16, 13), "ORIGINAL SELECTED FRAME", fill=(245, 247, 248), font=heading_font)
    draw.text(
        (panel_width + 16, 13),
        "GENRECON PBR RENDER",
        fill=(245, 247, 248),
        font=heading_font,
    )
    draw.line((panel_width, 0, panel_width, canvas.height), fill=(225, 229, 232), width=2)
    footer = (
        f"{candidate_id}  |  {frame_index + 1:02d}/{frame_count:02d}  |  {frame_name}  |  "
        f"source t={source_timestamp_s:.3f}s  |  {role}  |  coverage={coverage:.1%}"
    )
    detail_font = fitting_font(draw, footer, canvas.width - 32)
    draw.text((16, header_height + panel_height + 14), footer, fill=(224, 228, 231), font=detail_font)
    return canvas


def conditioning_roles(cameras_json: Path) -> dict[str, str]:
    document = load_json(cameras_json)
    scene = {
        Path(item["img_path"]).stem
        for item in document.get("scene", [])
        if item.get("img_path")
    }
    chunks = {
        Path(item["cond2d_view"]["img_path"]).stem
        for item in document.get("chunks", [])
        if item.get("cond2d_view", {}).get("img_path")
    }
    roles: dict[str, str] = {}
    for name in scene | chunks:
        if name in scene:
            roles[name] = "GenRecon scene conditioning"
        else:
            roles[name] = "GenRecon chunk conditioning"
    return roles


def selected_source_records(
    foundation_manifest: dict[str, Any], frames_manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    selected = foundation_manifest["selection"]["selected"]
    dynamic = foundation_manifest["selection"].get("selected_dynamic_fraction", [])
    if dynamic and len(dynamic) != len(selected):
        raise ValueError("selected_dynamic_fraction does not match selected views")
    frame_lookup = {Path(item["name"]).stem: item for item in frames_manifest["frames"]}
    records = []
    for index, name in enumerate(selected):
        stem = Path(name).stem
        if stem not in frame_lookup:
            raise ValueError(f"Selected frame {name} is absent from frames.json")
        source = frame_lookup[stem]
        records.append(
            {
                "selection_index": index,
                "stem": stem,
                "source_name": source["name"],
                "foundation_name": f"{stem}.png",
                "source_timestamp_s": float(source["timestamp_s"]),
                "source_index": int(source["index"]),
                "dynamic_fraction": float(dynamic[index]) if dynamic else None,
            }
        )
    records.sort(key=lambda item: (item["source_timestamp_s"], item["source_index"]))
    return records


def find_render_view(render_root: Path, stem: str) -> Path:
    matches = list(render_root.glob(f"views/*/{stem}"))
    if len(matches) != 1:
        raise ValueError(f"Expected one rendered view for {stem}, found {len(matches)}")
    return matches[0]


def render_cache_complete(render_root: Path, frame_stems: Sequence[str], width: int) -> bool:
    summary_path = render_root / "render_glb.json"
    if not summary_path.is_file():
        return False
    try:
        summary = load_json(summary_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if summary.get("render_width") != width or len(summary.get("views", [])) != len(frame_stems):
        return False
    for stem in frame_stems:
        try:
            view = find_render_view(render_root, stem)
        except ValueError:
            return False
        if not all((view / name).is_file() for name in ("glb_render.png", "glb_mask.png", "camera.json")):
            return False
    return True


def run_render(
    *,
    candidate: dict[str, Any],
    foundation_directory: Path,
    reconstruction_directory: Path,
    render_root: Path,
    report_directory: Path,
    frame_stems: Sequence[str],
    width: int,
    force: bool,
    colmap_subdir: str = "colmap_vggt",
) -> tuple[list[str], float]:
    if force and render_root.exists():
        shutil.rmtree(render_root)
    if render_cache_complete(render_root, frame_stems, width):
        return [], 0.0
    if render_root.exists():
        shutil.rmtree(render_root)
    render_root.mkdir(parents=True, exist_ok=True)
    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "tools" / "evaluate_view_fidelity.py"),
        "--stage",
        "glb",
        "--ply",
        str(reconstruction_directory / "mesh.ply"),
        "--glb",
        str(reconstruction_directory / "scene.glb"),
        "--cameras",
        str(foundation_directory / colmap_subdir / "cameras.txt"),
        "--images",
        str(foundation_directory / colmap_subdir / "images.txt"),
        "--points",
        str(foundation_directory / colmap_subdir / "points3D.txt"),
        "--images-root",
        str(foundation_directory / "rgb"),
        "--input-cameras-json",
        str(reconstruction_directory / "cameras.json"),
        "--output",
        str(render_root),
        "--scene-label",
        candidate["candidate_id"],
        "--width",
        str(width),
        "--no-lpips",
    ]
    report_directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    environment = os.environ.copy()
    environment.setdefault("EGL_PLATFORM", "surfaceless")
    with (report_directory / "render.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    duration = time.monotonic() - started
    if process.returncode != 0:
        raise RuntimeError(
            f"GLB render failed for {candidate['candidate_id']}; see {report_directory / 'render.log'}"
        )
    if not render_cache_complete(render_root, frame_stems, width):
        raise RuntimeError(f"Incomplete GLB render cache for {candidate['candidate_id']}")
    return command, duration


def clean_frame_directories(candidate_output: Path) -> None:
    frames_root = candidate_output / "frames"
    if frames_root.exists():
        shutil.rmtree(frames_root)
    for name in ("original", "reconstruction", "reconstruction_mask", "comparison"):
        (frames_root / name).mkdir(parents=True, exist_ok=True)


def compose_candidate_frames(
    *,
    candidate_id: str,
    records: list[dict[str, Any]],
    preproducts_directory: Path,
    reconstruction_directory: Path,
    render_root: Path,
    candidate_output: Path,
    geometry_only_role: str = "Foundation geometry only",
) -> tuple[list[dict[str, Any]], tuple[int, int], tuple[int, int]]:
    clean_frame_directories(candidate_output)
    roles = conditioning_roles(reconstruction_directory / "cameras.json")
    frame_records: list[dict[str, Any]] = []
    panel_size: tuple[int, int] | None = None
    comparison_size: tuple[int, int] | None = None
    for frame_index, record in enumerate(records):
        view = find_render_view(render_root, record["stem"])
        render_path = view / "glb_render.png"
        mask_path = view / "glb_mask.png"
        camera_path = view / "camera.json"
        source_path = preproducts_directory / "rgb" / record["source_name"]
        if not source_path.is_file():
            raise FileNotFoundError(f"Original extracted frame is missing: {source_path}")
        with Image.open(render_path) as opened:
            render_rgba = opened.convert("RGBA")
        render_width, render_height = render_rgba.size
        with Image.open(mask_path) as opened:
            mask = padded_mask(opened, render_width, render_height)
        alpha = padded_mask(render_rgba.getchannel("A"), render_width, render_height)
        if not np.array_equal(np.asarray(mask), np.asarray(alpha)):
            raise ValueError(f"Render alpha/mask mismatch for {candidate_id}/{record['stem']}")
        with Image.open(source_path) as opened:
            original = padded_rgb(opened, render_width, render_height)
        reconstruction = padded_rgb(render_rgba, render_width, render_height)
        role = roles.get(record["stem"], geometry_only_role)
        mask_array = np.asarray(mask) > 127
        render_array = np.asarray(reconstruction)
        coverage = float(mask_array.mean())
        inside = render_array[mask_array]
        nonblack = float(np.any(inside > 2, axis=1).mean()) if inside.size else 0.0
        comparison = compose_comparison_frame(
            original,
            reconstruction,
            candidate_id=candidate_id,
            frame_name=record["source_name"],
            source_timestamp_s=record["source_timestamp_s"],
            role=role,
            coverage=coverage,
            frame_index=frame_index,
            frame_count=len(records),
        )
        basename = f"{frame_index:06d}"
        original_output = candidate_output / "frames" / "original" / f"{basename}.jpg"
        reconstruction_output = (
            candidate_output / "frames" / "reconstruction" / f"{basename}.png"
        )
        mask_output = candidate_output / "frames" / "reconstruction_mask" / f"{basename}.png"
        comparison_output = candidate_output / "frames" / "comparison" / f"{basename}.jpg"
        original.save(original_output, quality=95, subsampling=0)
        reconstruction.save(reconstruction_output, compress_level=4)
        mask.save(mask_output, compress_level=4)
        comparison.save(comparison_output, quality=95, subsampling=0)
        camera = load_json(camera_path)
        frame_records.append(
            {
                **record,
                "frame_index": frame_index,
                "conditioning_role": role,
                "render_group": camera["group"],
                "coverage": coverage,
                "inside_nonblack_fraction": nonblack,
                "panel_width": original.width,
                "panel_height": original.height,
                "comparison_width": comparison.width,
                "comparison_height": comparison.height,
                "source_original_path": str(source_path.resolve()),
                "source_original_sha256": sha256_file(source_path),
                "camera_json": relative_to(camera_path, candidate_output),
                "original_frame": relative_to(original_output, candidate_output),
                "reconstruction_frame": relative_to(reconstruction_output, candidate_output),
                "reconstruction_mask": relative_to(mask_output, candidate_output),
                "comparison_frame": relative_to(comparison_output, candidate_output),
            }
        )
        current_panel = original.size
        current_comparison = comparison.size
        if panel_size is not None and panel_size != current_panel:
            raise ValueError(f"Inconsistent panel sizes for {candidate_id}")
        if comparison_size is not None and comparison_size != current_comparison:
            raise ValueError(f"Inconsistent comparison sizes for {candidate_id}")
        panel_size = current_panel
        comparison_size = current_comparison
    if panel_size is None or comparison_size is None:
        raise ValueError(f"No frames composed for {candidate_id}")
    return frame_records, panel_size, comparison_size


def encode_video(
    *,
    ffmpeg: str,
    source_pattern: Path,
    destination: Path,
    fps: float,
    report_log: Path,
) -> tuple[list[str], float]:
    temporary = destination.with_name(f".{destination.stem}.tmp.mp4")
    temporary.unlink(missing_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-framerate",
        f"{fps:g}",
        "-start_number",
        "0",
        "-i",
        str(source_pattern),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-threads",
        "4",
        str(temporary),
    ]
    started = time.monotonic()
    with report_log.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    duration = time.monotonic() - started
    if process.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError(f"Video encoding failed; see {report_log}")
    temporary.replace(destination)
    return command, duration


def probe_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    reported_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded_count = 0
    minimum_std = float("inf")
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded_count += 1
        minimum_std = min(minimum_std, float(frame.std()))
    capture.release()
    if decoded_count == 0:
        minimum_std = 0.0
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "codec": "H.264/libx264, yuv420p",
        "reported_frame_count": reported_count,
        "decoded_frame_count": decoded_count,
        "width": width,
        "height": height,
        "fps": fps,
        "duration_s": decoded_count / fps if fps > 0 else None,
        "minimum_decoded_rgb_std": minimum_std,
    }


def write_contact_sheet(paths: Sequence[Path], destination: Path) -> None:
    canvas = Image.new("RGB", (1600, 1000), (238, 241, 242))
    columns = 4
    rows = max(1, (len(paths) + columns - 1) // columns)
    cell_width, cell_height = 400, 1000 // rows
    for index, path in enumerate(paths):
        with Image.open(path) as opened:
            preview = ImageOps.contain(
                opened.convert("RGB"), (cell_width - 10, cell_height - 10), Image.Resampling.LANCZOS
            )
        left = (index % columns) * cell_width + (cell_width - preview.width) // 2
        top = (index // columns) * cell_height + (cell_height - preview.height) // 2
        canvas.paste(preview, (left, top))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=90, subsampling=0)


def make_contact(candidate_output: Path, frame_count: int) -> None:
    paths = [
        candidate_output / "frames" / "comparison" / f"{index:06d}.jpg"
        for index in range(frame_count)
    ]
    if not paths or not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"Comparison frames are incomplete under {candidate_output}")
    sample_indices = np.linspace(0, frame_count - 1, min(frame_count, 16)).round().astype(int)
    sampled_paths = [paths[index] for index in dict.fromkeys(sample_indices.tolist())]
    write_contact_sheet(sampled_paths, candidate_output / "overview.jpg")
    contacts = candidate_output / "contacts"
    if contacts.exists():
        shutil.rmtree(contacts)
    for page_index, start in enumerate(range(0, frame_count, 20)):
        write_contact_sheet(paths[start : start + 20], contacts / f"contact_{page_index:03d}.jpg")
    shutil.copy2(paths[0], candidate_output / "poster.jpg")


def candidate_is_complete(
    manifest_path: Path,
    *,
    width: int,
    fps: float,
    scene_glb_sha256: str,
) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = load_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if manifest.get("status") != "complete":
        return False
    if manifest.get("protocol", {}).get("panel_width") != width:
        return False
    if abs(float(manifest.get("protocol", {}).get("playback_fps", -1)) - fps) > 1e-6:
        return False
    if manifest.get("input", {}).get("scene_glb_sha256") != scene_glb_sha256:
        return False
    for video in manifest.get("videos", {}).values():
        path = Path(video.get("path", ""))
        if not path.is_file() or path.stat().st_size != video.get("size_bytes"):
            return False
        if sha256_file(path) != video.get("sha256"):
            return False
    return True


def run_candidate(
    *,
    candidate: dict[str, Any],
    foundation_root: Path,
    preproducts_root: Path,
    output_root: Path,
    report_root: Path,
    width: int,
    fps: float,
    ffmpeg: str,
    force: bool,
    track_name: str = "foundation-sfm",
    colmap_subdir: str = "colmap_vggt",
) -> dict[str, Any]:
    candidate_id = candidate["candidate_id"]
    foundation_directory = foundation_root / "candidates" / candidate_id
    preproducts_directory = preproducts_root / "candidates" / candidate_id
    reconstruction_directory = Path(candidate["reconstruction_directory"])
    candidate_output = output_root / "candidates" / candidate_id
    report_directory = report_root / "candidates" / candidate_id
    manifest_path = candidate_output / "manifest.json"
    scene_glb_hash = candidate["glb"]["sha256"]
    if not force and candidate_is_complete(
        manifest_path, width=width, fps=fps, scene_glb_sha256=scene_glb_hash
    ):
        print(f"[video] {candidate_id}: already complete")
        return load_json(manifest_path)
    for required in (
        foundation_directory / "manifest.json",
        foundation_directory / colmap_subdir / "images.txt",
        preproducts_directory / "frames.json",
        reconstruction_directory / "mesh.ply",
        reconstruction_directory / "scene.glb",
        reconstruction_directory / "cameras.json",
    ):
        if not required.is_file():
            raise FileNotFoundError(f"Required comparison input is missing: {required}")
    candidate_output.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    foundation_manifest = load_json(foundation_directory / "manifest.json")
    frames_manifest = load_json(preproducts_directory / "frames.json")
    records = selected_source_records(foundation_manifest, frames_manifest)
    colmap_images = read_colmap_images(foundation_directory / colmap_subdir / "images.txt")
    colmap_stems = {Path(name).stem for name in colmap_images}
    selected_stems = {item["stem"] for item in records}
    if colmap_stems != selected_stems:
        raise ValueError(
            f"Scene selection/COLMAP image mismatch for {candidate_id}: "
            f"selected-only={sorted(selected_stems - colmap_stems)}, "
            f"colmap-only={sorted(colmap_stems - selected_stems)}"
        )
    render_root = candidate_output / "render"
    render_command, render_duration = run_render(
        candidate=candidate,
        foundation_directory=foundation_directory,
        reconstruction_directory=reconstruction_directory,
        render_root=render_root,
        report_directory=report_directory,
        frame_stems=[item["stem"] for item in records],
        width=width,
        force=force,
        colmap_subdir=colmap_subdir,
    )
    geometry_only_role = (
        "Native SfM geometry only" if track_name == "native-sfm" else "Foundation geometry only"
    )
    frame_records, panel_size, comparison_size = compose_candidate_frames(
        candidate_id=candidate_id,
        records=records,
        preproducts_directory=preproducts_directory,
        reconstruction_directory=reconstruction_directory,
        render_root=render_root,
        candidate_output=candidate_output,
        geometry_only_role=geometry_only_role,
    )
    video_specs = {
        "original": candidate_output / "original.mp4",
        "reconstruction": candidate_output / "reconstruction.mp4",
        "side_by_side": candidate_output / "side_by_side.mp4",
    }
    patterns = {
        "original": candidate_output / "frames" / "original" / "%06d.jpg",
        "reconstruction": candidate_output / "frames" / "reconstruction" / "%06d.png",
        "side_by_side": candidate_output / "frames" / "comparison" / "%06d.jpg",
    }
    encode_commands: dict[str, list[str]] = {}
    encode_durations: dict[str, float] = {}
    videos: dict[str, dict[str, Any]] = {}
    for kind, destination in video_specs.items():
        command, duration = encode_video(
            ffmpeg=ffmpeg,
            source_pattern=patterns[kind],
            destination=destination,
            fps=fps,
            report_log=report_directory / f"encode_{kind}.log",
        )
        encode_commands[kind] = command
        encode_durations[kind] = duration
        videos[kind] = probe_video(destination)
    make_contact(candidate_output, len(frame_records))
    manifest = {
        "schema": f"genrecon.{track_name}-video-comparison-candidate",
        "track": track_name,
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "status": "complete",
        "candidate_id": candidate_id,
        "title": candidate["title"],
        "input_grade": candidate.get("input_grade", candidate.get("foundation_grade")),
        "foundation_grade": candidate.get("input_grade", candidate.get("foundation_grade")),
        "visual_disposition": candidate["visual_disposition"],
        "scope": (
            "Exact source frames with explicit native COLMAP cameras; fixed-rate inspection playback."
            if track_name == "native-sfm"
            else "Exact source frames with explicit foundation-predicted cameras; fixed-rate inspection "
            "playback, not every frame from the raw source video."
        ),
        "input": {
            "scene_directory": str(foundation_directory.resolve()),
            "foundation_directory": str(foundation_directory.resolve()),
            "colmap_subdir": colmap_subdir,
            "preproducts_directory": str(preproducts_directory.resolve()),
            "reconstruction_directory": str(reconstruction_directory.resolve()),
            "scene_glb": str((reconstruction_directory / "scene.glb").resolve()),
            "scene_glb_sha256": scene_glb_hash,
            "source_video": frames_manifest["source_video"],
            "source_video_sha256": frames_manifest["source_sha256"],
            "source_shot_start_s": frames_manifest["selected_start_s"],
            "source_shot_end_s": frames_manifest["selected_end_s"],
            "source_extraction_fps": frames_manifest["fps"],
        },
        "protocol": {
            "panel_width": width,
            "panel_height": panel_size[1],
            "comparison_width": comparison_size[0],
            "comparison_height": comparison_size[1],
            "playback_fps": fps,
            "playback_duration_s": len(frame_records) / fps,
            "frame_order": "ascending source timestamp",
            "renderer": "Open3D offscreen, defaultUnlit baked PBR albedo",
            "coordinate_map": "GenRecon glTF (x,z,-y) to COLMAP world",
            "video_codec": "H.264/libx264, CRF 18, yuv420p, no audio",
            "missing_surface_color": "black",
        },
        "counts": {
            "frames": len(frame_records),
            "scene_conditioning": sum(
                item["conditioning_role"] == "GenRecon scene conditioning"
                for item in frame_records
            ),
            "chunk_conditioning_only": sum(
                item["conditioning_role"] == "GenRecon chunk conditioning"
                for item in frame_records
            ),
            "geometry_only": sum(
                item["conditioning_role"] == geometry_only_role for item in frame_records
            ),
            "foundation_geometry_only": sum(
                item["conditioning_role"] == geometry_only_role for item in frame_records
            ),
        },
        "coverage": {
            "minimum": min(item["coverage"] for item in frame_records),
            "median": float(np.median([item["coverage"] for item in frame_records])),
            "maximum": max(item["coverage"] for item in frame_records),
        },
        "render": {
            "command": render_command,
            "duration_s": render_duration,
            "summary": relative_to(render_root / "render_glb.json", candidate_output),
        },
        "encoding": {
            "commands": encode_commands,
            "durations_s": encode_durations,
        },
        "videos": videos,
        "overview": str((candidate_output / "overview.jpg").resolve()),
        "poster": str((candidate_output / "poster.jpg").resolve()),
        "frames": frame_records,
        "limitations": [
            "No camera interpolation is used for raw-video frames without predicted poses.",
            (
                "Native SfM geometry uses multi-view observations from the registered sequence, so non-conditioning frames are not independent geometry holdout evidence."
                if track_name == "native-sfm"
                else "Foundation geometry was estimated from all listed views, so foundation-only frames are not independent geometry holdout evidence."
            ),
            "Black pixels in the reconstruction panel denote missing rendered surface/background.",
            "A nonblank comparison video does not establish geometric or photometric accuracy.",
        ],
    }
    write_json(manifest_path, manifest)
    print(
        f"[video] {candidate_id}: frames={len(frame_records)} "
        f"coverage={manifest['coverage']['median']:.3f} "
        f"duration={len(frame_records) / fps:.1f}s"
    )
    return manifest


def select_candidates(index: dict[str, Any], requested: Sequence[str]) -> list[dict[str, Any]]:
    candidates = [item for item in index["candidates"] if item.get("complete") is True]
    if requested:
        wanted = set(requested)
        available = {item["candidate_id"] for item in candidates}
        missing = sorted(wanted - available)
        if missing:
            raise ValueError(f"Unknown or incomplete candidates: {missing}")
        candidates = [item for item in candidates if item["candidate_id"] in wanted]
    return sorted(candidates, key=lambda item: item["candidate_id"])


def run_all(args: argparse.Namespace) -> None:
    index_path = args.genrecon_root / "index.json"
    index = load_json(index_path)
    candidates = select_candidates(index, args.candidate)
    ffmpeg = ffmpeg_executable(args.ffmpeg)
    version = subprocess.run(
        [ffmpeg, "-version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    write_json(
        args.output_root / "config.json",
        {
            "schema": f"genrecon.{args.track_name}-video-comparison-config",
            "track": args.track_name,
            "schema_version": SCHEMA_VERSION,
            "created_utc": utc_now(),
            "candidate_ids": [item["candidate_id"] for item in candidates],
            "scene_root": str(args.foundation_root.resolve()),
            "foundation_root": str(args.foundation_root.resolve()),
            "colmap_subdir": args.colmap_subdir,
            "preproducts_root": str(args.preproducts_root.resolve()),
            "genrecon_root": str(args.genrecon_root.resolve()),
            "genrecon_index_sha256": sha256_file(index_path),
            "output_root": str(args.output_root.resolve()),
            "report_root": str(args.report_root.resolve()),
            "panel_width": args.width,
            "playback_fps": args.fps,
            "ffmpeg": ffmpeg,
            "ffmpeg_version": version,
            "renderer_tool": str((ROOT / "tools" / "evaluate_view_fidelity.py").resolve()),
            "renderer_tool_sha256": sha256_file(ROOT / "tools" / "evaluate_view_fidelity.py"),
            "export_tool_sha256": sha256_file(Path(__file__)),
        },
    )
    for index_number, candidate in enumerate(candidates, start=1):
        print(f"[video] candidate {index_number}/{len(candidates)}: {candidate['candidate_id']}")
        run_candidate(
            candidate=candidate,
            foundation_root=args.foundation_root,
            preproducts_root=args.preproducts_root,
            output_root=args.output_root,
            report_root=args.report_root,
            width=args.width,
            fps=args.fps,
            ffmpeg=ffmpeg,
            force=args.force,
            track_name=args.track_name,
            colmap_subdir=args.colmap_subdir,
        )


def candidate_summary(manifest: dict[str, Any], output_root: Path) -> dict[str, Any]:
    videos = manifest["videos"]
    return {
        "candidate_id": manifest["candidate_id"],
        "title": manifest["title"],
        "input_grade": manifest.get("input_grade", manifest["foundation_grade"]),
        "foundation_grade": manifest["foundation_grade"],
        "visual_disposition": manifest["visual_disposition"],
        "frame_count": manifest["counts"]["frames"],
        "scene_conditioning": manifest["counts"]["scene_conditioning"],
        "foundation_geometry_only": manifest["counts"]["foundation_geometry_only"],
        "source_timestamp_start_s": manifest["frames"][0]["source_timestamp_s"],
        "source_timestamp_end_s": manifest["frames"][-1]["source_timestamp_s"],
        "playback_duration_s": manifest["protocol"]["playback_duration_s"],
        "coverage_min": manifest["coverage"]["minimum"],
        "coverage_median": manifest["coverage"]["median"],
        "coverage_max": manifest["coverage"]["maximum"],
        "continuous_source_video": os.path.relpath(
            manifest["input"]["source_video"], output_root.resolve()
        ),
        "original_video": relative_to(Path(videos["original"]["path"]), output_root),
        "reconstruction_video": relative_to(Path(videos["reconstruction"]["path"]), output_root),
        "side_by_side_video": relative_to(Path(videos["side_by_side"]["path"]), output_root),
        "poster": relative_to(Path(manifest["poster"]), output_root),
        "overview": relative_to(Path(manifest["overview"]), output_root),
        "manifest": relative_to(
            output_root / "candidates" / manifest["candidate_id"] / "manifest.json", output_root
        ),
        "video_bytes": sum(item["size_bytes"] for item in videos.values()),
        "original_sha256": videos["original"]["sha256"],
        "reconstruction_sha256": videos["reconstruction"]["sha256"],
        "side_by_side_sha256": videos["side_by_side"]["sha256"],
    }


def write_html(
    output_root: Path, rows: list[dict[str, Any]], track_name: str = "foundation-sfm"
) -> None:
    dispositions = sorted({row["visual_disposition"] for row in rows})
    options = ['<option value="all">All dispositions</option>'] + [
        f'<option value="{html.escape(value)}">{html.escape(value)}</option>'
        for value in dispositions
    ]
    cards = []
    for row in rows:
        cards.append(
            f"""<article class="candidate" data-disposition="{html.escape(row['visual_disposition'])}">
<header><div><h2>{html.escape(row['candidate_id'])}</h2><p>{html.escape(row['title'])}</p></div><span class="grade">{html.escape(row['foundation_grade'])}</span></header>
<video controls preload="metadata" poster="{html.escape(row['poster'])}" src="{html.escape(row['side_by_side_video'])}"></video>
<dl><div><dt>Frames</dt><dd>{row['frame_count']}</dd></div><div><dt>Playback</dt><dd>{row['playback_duration_s']:.1f}s</dd></div><div><dt>Median coverage</dt><dd>{row['coverage_median']:.1%}</dd></div><div><dt>Disposition</dt><dd>{html.escape(row['visual_disposition'])}</dd></div></dl>
<nav><a href="{html.escape(row['continuous_source_video'])}">Continuous source</a><a href="{html.escape(row['original_video'])}">Original selected MP4</a><a href="{html.escape(row['reconstruction_video'])}">Reconstruction MP4</a><a href="{html.escape(row['side_by_side_video'])}">Side-by-side MP4</a><a href="{html.escape(row['overview'])}">Frames</a><a href="{html.escape(row['manifest'])}">Manifest</a></nav>
</article>"""
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GenRecon Frame Comparisons</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#f2f4f5;color:#202428;font-family:Inter,ui-sans-serif,system-ui,sans-serif;letter-spacing:0}} .top{{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid #cdd3d7;padding:18px 24px}} .top-inner{{max-width:1500px;margin:auto;display:flex;align-items:end;justify-content:space-between;gap:20px}} h1{{font-size:25px;line-height:1.1;margin:0 0 5px}} .top p{{margin:0;color:#616a70;font-size:14px}} select{{height:38px;border:1px solid #aeb7bd;background:#fff;color:#202428;padding:0 34px 0 10px;border-radius:4px;font:inherit}} main{{max-width:1500px;margin:0 auto;padding:22px 24px 40px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}} .candidate{{background:#fff;border:1px solid #cfd5d9;border-radius:6px;overflow:hidden}} .candidate header{{display:flex;align-items:start;justify-content:space-between;gap:12px;padding:14px 16px}} h2{{font-size:16px;line-height:1.2;margin:0}} .candidate header p{{font-size:13px;color:#626b72;margin:4px 0 0}} .grade{{font:700 13px ui-monospace,monospace;background:#263238;color:#fff;padding:5px 7px;border-radius:3px}} video{{display:block;width:100%;background:#0f1113;max-height:620px}} dl{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));margin:0;border-top:1px solid #e0e4e7;border-bottom:1px solid #e0e4e7}} dl div{{padding:10px 12px;border-right:1px solid #e0e4e7;min-width:0}} dl div:last-child{{border:0}} dt{{font-size:11px;color:#697279;margin-bottom:4px}} dd{{font-size:13px;font-weight:650;margin:0;overflow-wrap:anywhere}} nav{{display:flex;flex-wrap:wrap;gap:8px 15px;padding:12px 16px}} a{{color:#086b78;font-size:13px;font-weight:650;text-decoration:none}} a:hover{{text-decoration:underline}}
@media(max-width:800px){{.top{{padding:14px 12px}} .top-inner{{align-items:start;flex-direction:column;gap:10px}} main{{grid-template-columns:1fr;padding:12px;gap:12px}} dl{{grid-template-columns:repeat(2,minmax(0,1fr))}} dl div:nth-child(2){{border-right:0}} dl div:nth-child(-n+2){{border-bottom:1px solid #e0e4e7}} h1{{font-size:21px}}}}
</style></head><body><header class="top"><div class="top-inner"><div><h1>GenRecon frame-exact comparisons</h1><p>{len(rows)} candidates / {html.escape(track_name)} cameras / fixed-rate inspection playback</p></div><select id="filter" aria-label="Filter by disposition">{''.join(options)}</select></div></header><main>{''.join(cards)}</main>
<script>const f=document.getElementById('filter');f.addEventListener('change',()=>document.querySelectorAll('.candidate').forEach(c=>c.hidden=f.value!=='all'&&c.dataset.disposition!==f.value));</script></body></html>"""
    (output_root / "index.html").write_text(document, encoding="utf-8")


def capture_screenshots(output_root: Path) -> None:
    browser = next(
        (
            path
            for executable in ("google-chrome", "chromium", "chromium-browser")
            if (path := shutil.which(executable)) is not None
        ),
        None,
    )
    if browser is None:
        raise RuntimeError("No headless Chrome/Chromium executable found")
    page_url = (output_root / "index.html").resolve().as_uri()
    for filename, width, height in (
        ("index_desktop.png", 1440, 1200),
        ("index_mobile.png", 390, 844),
    ):
        subprocess.run(
            [
                browser,
                "--headless",
                "--no-sandbox",
                "--disable-gpu",
                "--allow-file-access-from-files",
                "--hide-scrollbars",
                f"--window-size={width},{height}",
                "--virtual-time-budget=8000",
                f"--screenshot={output_root / filename}",
                page_url,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    index = load_json(args.genrecon_root / "index.json")
    candidates = select_candidates(index, args.candidate)
    rows = []
    for candidate in candidates:
        manifest_path = args.output_root / "candidates" / candidate["candidate_id"] / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Comparison manifest is missing: {manifest_path}")
        rows.append(candidate_summary(load_json(manifest_path), args.output_root))
    rows.sort(key=lambda item: item["candidate_id"])
    for row in rows:
        make_contact(
            args.output_root / "candidates" / row["candidate_id"], row["frame_count"]
        )
    summary = {
        "schema": f"genrecon.{args.track_name}-video-comparison-index",
        "track": args.track_name,
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "scope": (
            "Exact native COLMAP registered views."
            if args.track_name == "native-sfm"
            else "Exact foundation-camera views only; no interpolation to unposed raw-video frames."
        ),
        "summary": {
            "candidate_count": len(rows),
            "frame_count": sum(row["frame_count"] for row in rows),
            "video_count": len(rows) * 3,
            "video_bytes": sum(row["video_bytes"] for row in rows),
            "playback_duration_s": sum(row["playback_duration_s"] for row in rows),
            "grades": dict(sorted(Counter(row["input_grade"] for row in rows).items())),
            "visual_dispositions": dict(
                sorted(Counter(row["visual_disposition"] for row in rows).items())
            ),
        },
        "candidates": rows,
    }
    write_json(args.output_root / "index.json", summary)
    fields = list(rows[0]) if rows else []
    with (args.output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_html(args.output_root, rows, args.track_name)
    readme = f"""# {args.track_name} GenRecon Video Comparisons

- Candidates: {len(rows)}
- Exact posed frames: {summary['summary']['frame_count']}
- Videos: {summary['summary']['video_count']} (`original.mp4`, `reconstruction.mp4`, `side_by_side.mp4` per candidate)
- Playback: fixed {args.fps:g} fps, no audio

Open `index.html` for the video review grid and links to each continuous source
file. `original.mp4` contains the exact frames with explicit
{('native COLMAP' if args.track_name == 'native-sfm' else 'foundation-predicted')} cameras.
`index.json`, `summary.csv`, and `validation.json` provide machine-readable paths
and checks.

{('Every registered native COLMAP frame is included.' if args.track_name == 'native-sfm' else 'Only frames with explicit foundation-predicted cameras are included; no pose interpolation is used for remaining raw-video frames.')}
The frame manifest preserves each source-video timestamp.

Detailed protocol and interpretation:
`../../../reports/{('NATIVE_SFM_GENRECON_REPORT_zh.md' if args.track_name == 'native-sfm' else 'FOUNDATION_GENRECON_VIDEO_COMPARISON_REPORT_zh.md')}`.

```text
candidates/<candidate_id>/
  original.mp4
  reconstruction.mp4
  side_by_side.mp4
  poster.jpg
  overview.jpg
  manifest.json
  render/
  contacts/
  frames/original/
  frames/reconstruction/
  frames/reconstruction_mask/
  frames/comparison/
```
"""
    (args.output_root / "README.md").write_text(readme, encoding="utf-8")
    if not args.no_screenshots:
        capture_screenshots(args.output_root)
    return summary


def local_html_references(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return re.findall(r'(?:src|href)="([^"#]+)"', text)


def validate(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    index_path = args.output_root / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Comparison index is missing: {index_path}")
    index = load_json(index_path)
    candidate_results = []
    total_frames = 0
    total_videos = 0
    total_video_bytes = 0
    for row in index["candidates"]:
        candidate_id = row["candidate_id"]
        candidate_root = args.output_root / "candidates" / candidate_id
        manifest_path = candidate_root / "manifest.json"
        candidate_errors: list[str] = []
        try:
            manifest = load_json(manifest_path)
        except Exception as exc:
            errors.append(f"{candidate_id}: invalid manifest: {exc}")
            continue
        frames = manifest["frames"]
        expected_count = manifest["counts"]["frames"]
        total_frames += expected_count
        if len(frames) != expected_count:
            candidate_errors.append(f"manifest frame count {len(frames)} != {expected_count}")
        timestamps = [item["source_timestamp_s"] for item in frames]
        if timestamps != sorted(timestamps):
            candidate_errors.append("source timestamps are not ascending")
        for frame_index, frame in enumerate(frames):
            if frame["frame_index"] != frame_index:
                candidate_errors.append(f"non-contiguous frame index at {frame_index}")
            for key in (
                "original_frame",
                "reconstruction_frame",
                "reconstruction_mask",
                "comparison_frame",
                "camera_json",
            ):
                path = candidate_root / frame[key]
                if not path.is_file() or path.stat().st_size == 0:
                    candidate_errors.append(f"missing {key}: {path}")
            try:
                with Image.open(candidate_root / frame["original_frame"]) as original_image:
                    original = np.asarray(original_image.convert("RGB"))
                with Image.open(candidate_root / frame["reconstruction_frame"]) as render_image:
                    render = np.asarray(render_image.convert("RGB"))
                with Image.open(candidate_root / frame["reconstruction_mask"]) as mask_image:
                    mask = np.asarray(mask_image.convert("L"))
                with Image.open(candidate_root / frame["comparison_frame"]) as comparison_image:
                    comparison_size = comparison_image.size
                if original.shape != render.shape or original.shape[:2] != mask.shape:
                    candidate_errors.append(f"frame dimensions differ at {frame_index}")
                if set(np.unique(mask)) - {0, 255}:
                    candidate_errors.append(f"mask is not binary at {frame_index}")
                measured_coverage = float((mask > 127).mean())
                if abs(measured_coverage - frame["coverage"]) > 1e-12:
                    candidate_errors.append(f"coverage mismatch at {frame_index}")
                expected_comparison = (frame["comparison_width"], frame["comparison_height"])
                if comparison_size != expected_comparison:
                    candidate_errors.append(f"comparison dimensions differ at {frame_index}")

                source_path = Path(frame["source_original_path"])
                if not source_path.is_file():
                    candidate_errors.append(f"source original is missing at {frame_index}")
                elif sha256_file(source_path) != frame["source_original_sha256"]:
                    candidate_errors.append(f"source original SHA256 mismatch at {frame_index}")
                cache_view = (candidate_root / frame["camera_json"]).parent
                with Image.open(cache_view / "glb_render.png") as cached_render_image:
                    cached_render = cached_render_image.convert("RGBA")
                raw_width, raw_height = cached_render.size
                with Image.open(source_path) as source_image:
                    expected_original = np.asarray(
                        padded_rgb(source_image, raw_width, raw_height), dtype=np.int16
                    )
                original_mae = float(
                    np.abs(original.astype(np.int16) - expected_original).mean()
                )
                if original_mae > 3.0:
                    candidate_errors.append(
                        f"original frame/source pixels differ at {frame_index}: MAE={original_mae:.3f}"
                    )
                expected_render = np.asarray(
                    padded_rgb(cached_render, raw_width, raw_height)
                )
                if not np.array_equal(render, expected_render):
                    candidate_errors.append(f"reconstruction/cache pixels differ at {frame_index}")
                expected_mask = np.asarray(
                    padded_mask(cached_render.getchannel("A"), raw_width, raw_height)
                )
                if not np.array_equal(mask, expected_mask):
                    candidate_errors.append(f"reconstruction mask/cache alpha differ at {frame_index}")
            except Exception as exc:
                candidate_errors.append(f"could not validate frame {frame_index}: {exc}")
        for kind, video in manifest["videos"].items():
            total_videos += 1
            path = Path(video["path"])
            if not path.is_file():
                candidate_errors.append(f"missing {kind} video")
                continue
            total_video_bytes += path.stat().st_size
            if path.stat().st_size != video["size_bytes"]:
                candidate_errors.append(f"{kind} size mismatch")
            if sha256_file(path) != video["sha256"]:
                candidate_errors.append(f"{kind} SHA256 mismatch")
            try:
                probe = probe_video(path)
                if probe["decoded_frame_count"] != expected_count:
                    candidate_errors.append(
                        f"{kind} decoded {probe['decoded_frame_count']} frames, expected {expected_count}"
                    )
                if abs(probe["fps"] - manifest["protocol"]["playback_fps"]) > 1e-3:
                    candidate_errors.append(f"{kind} FPS mismatch")
                if kind != "reconstruction" and probe["minimum_decoded_rgb_std"] <= 0.25:
                    candidate_errors.append(f"{kind} contains a blank decoded frame")
            except Exception as exc:
                candidate_errors.append(f"could not decode {kind}: {exc}")
        if candidate_errors:
            errors.extend(f"{candidate_id}: {message}" for message in candidate_errors)
        candidate_results.append(
            {
                "candidate_id": candidate_id,
                "frame_count": expected_count,
                "video_count": len(manifest["videos"]),
                "errors": candidate_errors,
            }
        )
    references = local_html_references(args.output_root / "index.html")
    bad_references = []
    for reference in references:
        if reference.startswith(("http://", "https://", "data:")):
            continue
        if not (args.output_root / reference).is_file():
            bad_references.append(reference)
    if bad_references:
        errors.extend(f"invalid HTML reference: {reference}" for reference in bad_references)
    screenshot_dimensions = {}
    for name, expected in (("index_desktop.png", (1440, 1200)), ("index_mobile.png", (390, 844))):
        path = args.output_root / name
        if args.no_screenshots:
            continue
        if not path.is_file():
            errors.append(f"missing screenshot: {name}")
            continue
        with Image.open(path) as image:
            screenshot_dimensions[name] = list(image.size)
            if image.size != expected:
                errors.append(f"screenshot dimensions differ: {name} {image.size} != {expected}")
    validation_path = args.output_root / "validation.json"
    result = {
        "schema": f"genrecon.{args.track_name}-video-comparison-validation",
        "track": args.track_name,
        "schema_version": SCHEMA_VERSION,
        "validated_utc": utc_now(),
        "result": "pass" if not errors else "fail",
        "counts": {
            "candidate_count": len(candidate_results),
            "frame_count": total_frames,
            "video_count": total_videos,
            "video_bytes": total_video_bytes,
            "html_local_references": len(references),
            "strict_json_files": 0,
        },
        "checks": {
            "all_frame_payloads_valid": not any(item["errors"] for item in candidate_results),
            "all_original_frames_match_recorded_sources": not any(
                "source original" in error or "original frame/source" in error for error in errors
            ),
            "all_reconstruction_frames_match_render_cache": not any(
                "reconstruction/cache" in error or "mask/cache" in error for error in errors
            ),
            "all_video_sha256_match": not any("SHA256" in error for error in errors),
            "all_videos_decode_to_expected_frame_count": not any("decoded" in error for error in errors),
            "source_and_comparison_videos_have_nonblank_frames": not any(
                "blank decoded" in error for error in errors
            ),
            "html_references_valid": not bad_references,
            "screenshots_valid": not any("screenshot" in error for error in errors),
            "strict_json": False,
        },
        "screenshots": screenshot_dimensions,
        "candidates": candidate_results,
        "errors": errors,
    }
    write_json(validation_path, result)
    strict_paths = sorted(set(args.output_root.rglob("*.json")) | set(args.report_root.rglob("*.json")))
    strict_errors = []
    for path in strict_paths:
        try:
            load_json(path)
        except Exception as exc:
            strict_errors.append(f"non-strict JSON {path}: {exc}")
    result["counts"]["strict_json_files"] = len(strict_paths)
    result["checks"]["strict_json"] = not strict_errors
    result["errors"].extend(strict_errors)
    result["result"] = "pass" if not result["errors"] else "fail"
    write_json(validation_path, result)
    print(f"[video-validation] {result['result']} {result['counts']}")
    if result["errors"]:
        for error in result["errors"]:
            print(f"[video-validation] ERROR: {error}", file=sys.stderr)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "summarize", "validate", "all"))
    parser.add_argument(
        "--foundation-root", "--scene-root", dest="foundation_root", type=Path, default=DEFAULT_FOUNDATION_ROOT
    )
    parser.add_argument("--track-name", default="foundation-sfm")
    parser.add_argument("--colmap-subdir", default="colmap_vggt")
    parser.add_argument("--preproducts-root", type=Path, default=DEFAULT_PREPRODUCTS_ROOT)
    parser.add_argument("--genrecon-root", type=Path, default=DEFAULT_GENRECON_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-screenshots", action="store_true")
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.width < 256 or args.width % 2:
        raise SystemExit("--width must be an even integer of at least 256")
    if not np.isfinite(args.fps) or args.fps <= 0 or args.fps > 60:
        raise SystemExit("--fps must be finite and in (0, 60]")
    if args.stage in {"run", "all"}:
        for path in (
            args.foundation_root / "index.json",
            args.preproducts_root / "index.json",
            args.genrecon_root / "index.json",
        ):
            if not path.is_file():
                raise SystemExit(f"Required input index does not exist: {path}")


def main() -> None:
    args = build_parser().parse_args()
    validate_arguments(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_root / ".video_export.lock"
    try:
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.stage in {"run", "all"}:
                run_all(args)
            if args.stage in {"summarize", "all"}:
                summarize(args)
            if args.stage in {"validate", "all"}:
                result = validate(args)
                if result["result"] != "pass":
                    raise SystemExit(1)
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
