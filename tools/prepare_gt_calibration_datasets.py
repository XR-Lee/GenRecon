#!/usr/bin/env python3
"""Build the frozen mixed-GT calibration registry and lightweight input packages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from zipfile import ZipFile

import cv2
import numpy as np
import py7zr
import trimesh
from PIL import Image

try:
    import OpenEXR
except ModuleNotFoundError:
    OpenEXR = None

try:
    from tools.prepare_da3_scannetpp import (
        _camera_center,
        read_cameras_binary,
        read_image_headers_binary,
    )
except ModuleNotFoundError:
    from prepare_da3_scannetpp import (
        _camera_center,
        read_cameras_binary,
        read_image_headers_binary,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs" / "eval" / "gt_calibration_v1.json"
DEFAULT_OUTPUT = ROOT / "data" / "gt-calibration-v1"
SCHEMA = "genrecon.gt-calibration-unit"
REGISTRY_SCHEMA = "genrecon.gt-calibration-registry"
PLAN_SCHEMA = "genrecon.gt-calibration-plan"
_HASH_CACHE: dict[tuple[str, int, int], str] = {}
SEVEN_INTRINSICS = {"fx": 585.0, "fy": 585.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}
REDWOOD_INTRINSICS = {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5, "width": 640, "height": 480}
OMNI_DATASET_REPO = "OpenXDLab/OmniObject3D-New"
OMNI_ADAPTER_VERSION = 3
OMNI_OFFICIAL_CODE_REVISION = "d1e05fa62759089298c25bcccf83c94d718375c4"
OMNI_SCAN_TO_RENDER_AXIS = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)
OMNI_MAX_SILHOUETTE_EDGE_ERROR_PX = 8.0
OMNI_MIN_SILHOUETTE_BBOX_IOU = 0.94
OMNI_MIN_RAYCAST_MASK_IOU = 0.95
OMNI_MAX_CAMERA_ORTHOGONALITY_FOR_REPAIR = 5e-4
OMNI_MAX_CAMERA_ROTATION_REPAIR = 2e-4
OMNI_RENDER_TARGET_MAX_ABS = 0.99
OMNI_RECORDED_SCALE_MAX_TARGET_RATIO = 10.0
OMNI_RENDER_RESOLUTION = 800
OMNI_DEPTH_BACKGROUND_SENTINEL = float(np.finfo(np.float16).max)


class CalibrationBuildError(RuntimeError):
    pass


class OmniAlignmentError(CalibrationBuildError):
    pass


class OmniDependencyError(CalibrationBuildError):
    pass


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            CalibrationBuildError(f"Non-finite JSON constant {value} in {path}")
        ),
    )


def write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key in _HASH_CACHE:
        return _HASH_CACHE[key]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    value = digest.hexdigest()
    _HASH_CACHE[key] = value
    return value


def file_prefix(path: Path, size: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(size)


def relpath(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve())


def uniform_indices(count: int, requested: int, *, excluded: set[int] | None = None) -> list[int]:
    if count <= 0 or requested <= 0:
        return []
    excluded = excluded or set()
    eligible = [index for index in range(count) if index not in excluded]
    if len(eligible) <= requested:
        return eligible
    positions = np.linspace(0, len(eligible) - 1, requested)
    selected = [eligible[int(round(value))] for value in positions]
    return list(dict.fromkeys(selected))


def safe_zip_names(archive: ZipFile) -> list[str]:
    names = archive.namelist()
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise CalibrationBuildError(f"Unsafe ZIP path {name!r}")
    return names


def safe_tar_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise CalibrationBuildError(f"Unsafe TAR path {member.name!r}")
        if member.issym() or member.islnk() or member.isdev():
            raise CalibrationBuildError(f"Unsupported TAR member {member.name!r}")
    return members


def farthest_point_indices(
    points: np.ndarray,
    requested: int,
    *,
    excluded: set[int] | None = None,
) -> list[int]:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise CalibrationBuildError("Farthest-point input must be finite [N,3] coordinates")
    excluded = excluded or set()
    eligible = np.asarray(
        [index for index in range(len(values)) if index not in excluded],
        dtype=np.int64,
    )
    if requested <= 0 or not len(eligible):
        return []
    requested = min(int(requested), len(eligible))
    centered = values[eligible] - values[eligible].mean(axis=0)
    first = int(eligible[int(np.argmax(np.sum(centered * centered, axis=1)))])
    selected = [first]
    minimum_distance = np.sum((values[eligible] - values[first]) ** 2, axis=1)
    while len(selected) < requested:
        minimum_distance[np.isin(eligible, selected)] = -np.inf
        next_index = int(eligible[int(np.argmax(minimum_distance))])
        selected.append(next_index)
        distance = np.sum((values[eligible] - values[next_index]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
    return selected


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_sha256(document: Any) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalized_tar_name(member: tarfile.TarInfo) -> str:
    return member.name[2:] if member.name.startswith("./") else member.name


def _unique_tar_member(
    members: list[tarfile.TarInfo], expected_name: str, archive_path: Path
) -> tarfile.TarInfo:
    matches = [
        member
        for member in members
        if member.isfile() and _normalized_tar_name(member) == expected_name
    ]
    if len(matches) != 1:
        raise CalibrationBuildError(
            f"Expected one {expected_name!r} in {archive_path}, found {len(matches)}"
        )
    return matches[0]


def _read_tar_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    handle = archive.extractfile(member)
    if handle is None:
        raise CalibrationBuildError(f"Unable to read TAR member {member.name!r}")
    payload = handle.read()
    if len(payload) != member.size:
        raise CalibrationBuildError(f"Short read for TAR member {member.name!r}")
    return payload


def _read_tar_members_in_archive_order(
    archive: tarfile.TarFile, members: Iterable[tarfile.TarInfo]
) -> dict[str, bytes]:
    output = {}
    for member in sorted(members, key=lambda item: item.offset_data):
        name = _normalized_tar_name(member)
        if name in output:
            raise CalibrationBuildError(f"Duplicate TAR member request {name!r}")
        output[name] = _read_tar_member(archive, member)
    return output


def _omni_category(object_id: str) -> str:
    match = re.fullmatch(r"(.+)_([0-9]{3})", object_id)
    if not match:
        raise CalibrationBuildError(f"Invalid OmniObject3D object ID {object_id!r}")
    return match.group(1)


def _omni_blender_c2w_to_opencv_with_audit(
    value: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise CalibrationBuildError("Invalid OmniObject3D Blender camera pose")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise CalibrationBuildError("Invalid OmniObject3D homogeneous camera row")
    pose = pose.copy()
    raw_rotation = pose[:3, :3].copy()
    raw_determinant = float(np.linalg.det(raw_rotation))
    raw_orthogonality_error = float(
        np.max(np.abs(raw_rotation.T @ raw_rotation - np.eye(3)))
    )
    correction = np.zeros((3, 3), dtype=np.float64)
    method = "unchanged-within-tolerance"
    if raw_determinant <= 0.0:
        raise CalibrationBuildError("OmniObject3D camera rotation is not right-handed")
    if raw_orthogonality_error > 1e-5:
        left, _, right = np.linalg.svd(raw_rotation)
        repaired = left @ right
        if np.linalg.det(repaired) < 0.0:
            left[:, -1] *= -1.0
            repaired = left @ right
        correction = repaired - raw_rotation
        if (
            raw_orthogonality_error > OMNI_MAX_CAMERA_ORTHOGONALITY_FOR_REPAIR
            or abs(raw_determinant - 1.0)
            > OMNI_MAX_CAMERA_ORTHOGONALITY_FOR_REPAIR
            or float(np.max(np.abs(correction)))
            > OMNI_MAX_CAMERA_ROTATION_REPAIR
        ):
            raise CalibrationBuildError(
                "OmniObject3D camera rotation exceeds bounded SO(3) repair limits"
            )
        pose[:3, :3] = repaired
        method = "nearest-SO3-SVD"
    pose[:3, 1:3] *= -1.0
    rotation = pose[:3, :3]
    if (
        np.linalg.det(rotation) <= 0.0
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
    ):
        raise CalibrationBuildError("OmniObject3D camera rotation is not SO(3)")
    audit = {
        "method": method,
        "raw_determinant": raw_determinant,
        "raw_orthogonality_max_abs": raw_orthogonality_error,
        "max_abs_rotation_correction": float(np.max(np.abs(correction))),
        "frobenius_rotation_correction": float(np.linalg.norm(correction)),
    }
    return pose, audit


def _omni_blender_c2w_to_opencv(value: Any) -> np.ndarray:
    pose, _ = _omni_blender_c2w_to_opencv_with_audit(value)
    return pose


def _omni_select_render_scale(
    mesh: trimesh.Trimesh, recorded_scale: Any
) -> tuple[float, dict[str, Any]]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    scale = float(recorded_scale)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not len(vertices)
        or not np.isfinite(vertices).all()
        or not np.isfinite(scale)
        or scale <= 0.0
    ):
        raise CalibrationBuildError("Invalid OmniObject3D render scale input")
    source_radius = float(np.max(np.abs(vertices)))
    recorded_output_radius = source_radius * scale
    maximum_plausible_radius = (
        OMNI_RENDER_TARGET_MAX_ABS * OMNI_RECORDED_SCALE_MAX_TARGET_RATIO
    )
    method = "recorded-full-render-scale"
    selected_scale = scale
    if recorded_output_radius > maximum_plausible_radius:
        selected_scale = OMNI_RENDER_TARGET_MAX_ABS / source_radius
        method = "recomputed-published-renderer-scale-after-recorded-scale-sanity-failure"
    return selected_scale, {
        "method": method,
        "recorded_scale": scale,
        "recorded_output_max_abs_coordinate": recorded_output_radius,
        "recorded_scale_max_plausible_output_coordinate": maximum_plausible_radius,
        "published_render_target_max_abs_coordinate": OMNI_RENDER_TARGET_MAX_ABS,
        "selected_scale": selected_scale,
        "selected_output_max_abs_coordinate": source_radius * selected_scale,
        "recorded_scale_used": method == "recorded-full-render-scale",
    }


def _omni_scan_to_render_mesh(
    mesh: trimesh.Trimesh, official_scale: Any
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    scale = float(official_scale)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not len(vertices)
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or not np.isfinite(scale)
        or scale <= 0.0
    ):
        raise CalibrationBuildError("Invalid OmniObject3D scan mesh or render scale")
    transformed = vertices * scale @ OMNI_SCAN_TO_RENDER_AXIS
    output = trimesh.Trimesh(
        vertices=transformed,
        faces=faces,
        process=False,
        maintain_order=True,
    )
    source_radius = float(np.max(np.abs(vertices)))
    output_radius = float(np.max(np.abs(transformed)))
    return output, {
        "method": "audited-full-render-scale-and-Blender-OBJ-import-axis",
        "source_max_abs_coordinate": source_radius,
        "selected_uniform_scale": scale,
        "output_max_abs_coordinate": output_radius,
        "right_multiply_axis": OMNI_SCAN_TO_RENDER_AXIS.tolist(),
        "source_bounds": np.stack((vertices.min(axis=0), vertices.max(axis=0))).tolist(),
        "output_bounds": np.stack((transformed.min(axis=0), transformed.max(axis=0))).tolist(),
        "coordinate_units": "normalized-object",
    }


def _load_omni_scan_mesh(payload: bytes) -> trimesh.Trimesh:
    loaded = trimesh.load(io.BytesIO(payload), file_type="obj", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geometry
            for geometry in loaded.dump(concatenate=False)
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if not meshes:
            raise CalibrationBuildError("OmniObject3D OBJ contains no triangle mesh")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
        raise CalibrationBuildError("OmniObject3D OBJ is not a nonempty triangle mesh")
    loaded.merge_vertices(merge_tex=True, merge_norm=True)
    return loaded


def _omni_depth_mask(
    depth_payload: bytes,
) -> tuple[np.ndarray, tuple[int, int], list[int], dict[str, Any]]:
    if OpenEXR is None:
        raise OmniDependencyError(
            "OpenEXR is required for OmniObject3D full-render depth masks; "
            "install openexr==3.4.14"
        )
    try:
        with tempfile.NamedTemporaryFile(suffix=".exr") as temporary:
            temporary.write(depth_payload)
            temporary.flush()
            channels = OpenEXR.File(temporary.name).channels()
    except Exception as exc:
        raise CalibrationBuildError(
            f"Invalid OmniObject3D depth EXR: {exc}"
        ) from exc
    if len(channels) != 1:
        raise CalibrationBuildError(
            f"OmniObject3D depth EXR must contain one packed channel, got "
            f"{sorted(channels)}"
        )
    channel_name, channel = next(iter(channels.items()))
    packed = np.asarray(channel.pixels)
    if (
        packed.ndim != 3
        or packed.shape[2] != 3
        or packed.dtype.kind != "f"
        or not np.isfinite(packed).all()
        or not np.array_equal(packed[..., 0], packed[..., 1])
        or not np.array_equal(packed[..., 0], packed[..., 2])
    ):
        raise CalibrationBuildError(
            "OmniObject3D depth EXR is not a finite three-channel replicated depth pass"
        )
    depth = packed[..., 0]
    if not np.any(depth == OMNI_DEPTH_BACKGROUND_SENTINEL):
        raise CalibrationBuildError(
            "OmniObject3D depth EXR is missing the official half-float background sentinel"
        )
    if np.any(depth > OMNI_DEPTH_BACKGROUND_SENTINEL):
        raise CalibrationBuildError(
            "OmniObject3D depth EXR exceeds the official half-float range"
        )
    background_sentinel = OMNI_DEPTH_BACKGROUND_SENTINEL
    mask = depth < background_sentinel
    rows, columns = np.nonzero(mask)
    if (
        not len(rows)
        or np.all(mask)
        or background_sentinel <= 0.0
        or float(depth[mask].min()) <= 0.0
    ):
        raise CalibrationBuildError(
            "OmniObject3D depth EXR has an invalid foreground/background split"
        )
    bbox = [
        int(columns.min()),
        int(rows.min()),
        int(columns.max()),
        int(rows.max()),
    ]
    metadata = {
        "packed_channel": channel_name,
        "dtype": str(packed.dtype),
        "background_sentinel": background_sentinel,
        "foreground_min_depth": float(depth[mask].min()),
        "foreground_max_depth": float(depth[mask].max()),
        "foreground_pixels": int(mask.sum()),
    }
    return mask, (packed.shape[1], packed.shape[0]), bbox, metadata


def _omni_rgb_on_white(
    rgb_payload: bytes, mask: np.ndarray, image_size: tuple[int, int]
) -> bytes:
    try:
        with Image.open(io.BytesIO(rgb_payload)) as opened:
            rgb = opened.convert("RGB")
    except (OSError, ValueError) as exc:
        raise CalibrationBuildError(
            f"Invalid OmniObject3D RGB render: {exc}"
        ) from exc
    if rgb.size != image_size or mask.shape != (image_size[1], image_size[0]):
        raise CalibrationBuildError("OmniObject3D RGB and depth dimensions differ")
    rgb_array = np.asarray(rgb, dtype=np.uint8)
    composited = np.full_like(rgb_array, 255)
    composited[mask] = rgb_array[mask]
    output = io.BytesIO()
    Image.fromarray(composited, mode="RGB").save(
        output, format="PNG", compress_level=3
    )
    return output.getvalue()


def _omni_projection_bbox(
    vertices: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    world_to_camera = np.linalg.inv(camera_to_world)
    camera = vertices @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    valid = camera[:, 2] > 1e-8
    camera = camera[valid]
    if not len(camera):
        raise CalibrationBuildError("OmniObject3D scan is behind a declared camera")
    width, height = image_size
    u = intrinsic[0, 0] * camera[:, 0] / camera[:, 2] + intrinsic[0, 2]
    v = intrinsic[1, 1] * camera[:, 1] / camera[:, 2] + intrinsic[1, 2]
    inside = (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
    if not np.any(inside):
        raise CalibrationBuildError("OmniObject3D scan does not project into the image")
    return np.asarray(
        [u[inside].min(), v[inside].min(), u[inside].max(), v[inside].max()],
        dtype=np.float64,
    )


def _omni_raycast_scene(mesh: trimesh.Trimesh) -> Any:
    import open3d as o3d

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    return scene


def _omni_raycast_mask(
    scene: Any,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    import open3d as o3d

    width, height = image_size
    world_to_camera = np.linalg.inv(camera_to_world)
    rays = scene.create_rays_pinhole(
        o3d.core.Tensor(intrinsic.astype(np.float32)),
        o3d.core.Tensor(world_to_camera.astype(np.float32)),
        width_px=width,
        height_px=height,
    )
    mask = np.isfinite(scene.cast_rays(rays)["t_hit"].numpy())
    if mask.shape != (height, width) or not np.any(mask):
        raise CalibrationBuildError(
            "OmniObject3D visibility-aware raycast produced an invalid mask"
        )
    return mask


def _mask_bbox(mask: np.ndarray) -> np.ndarray:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise CalibrationBuildError("OmniObject3D mask has no foreground pixels")
    return np.asarray(
        [columns.min(), rows.min(), columns.max(), rows.max()], dtype=np.float64
    )


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape or first.dtype != np.bool_ or second.dtype != np.bool_:
        raise CalibrationBuildError("OmniObject3D masks are incompatible")
    union = np.logical_or(first, second)
    return float(np.logical_and(first, second).sum() / max(int(union.sum()), 1))


def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    minimum = np.maximum(first[:2], second[:2])
    maximum = np.minimum(first[2:], second[2:])
    intersection = float(np.prod(np.maximum(maximum - minimum, 0.0)))
    first_area = float(np.prod(np.maximum(first[2:] - first[:2], 0.0)))
    second_area = float(np.prod(np.maximum(second[2:] - second[:2], 0.0)))
    return intersection / max(first_area + second_area - intersection, 1e-12)


def _omni_intrinsic(camera_angle_x: float, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    angle = float(camera_angle_x)
    if not np.isfinite(angle) or not 0.0 < angle < np.pi:
        raise CalibrationBuildError("Invalid OmniObject3D horizontal field of view")
    focal = 0.5 * width / np.tan(0.5 * angle)
    return np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _write_binary_mesh_ply(mesh: trimesh.Trimesh, destination: Path) -> None:
    payload = trimesh.exchange.ply.export_ply(
        mesh,
        encoding="binary",
        vertex_normal=False,
        include_attributes=False,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)


def safe_7z_extract(path: Path, output: Path) -> None:
    with py7zr.SevenZipFile(path) as archive:
        names = archive.getnames()
        for name in names:
            member = Path(name)
            if member.is_absolute() or ".." in member.parts:
                raise CalibrationBuildError(f"Unsafe 7z path {name!r} in {path}")
        archive.extractall(path=output)


def _write_source_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [
        {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    ]


def _manifest_path(output: Path, unit_id: str) -> Path:
    return output / "units" / unit_id / "manifest.json"


def _write_manifest(output: Path, document: dict[str, Any]) -> dict[str, Any]:
    path = _manifest_path(output, document["unit_id"])
    write_json(path, document)
    return {
        "unit_id": document["unit_id"],
        "physical_scene_group": document["physical_scene_group"],
        "dataset": document["dataset"],
        "track": document["track"],
        "gt_tier": document["gt_tier"],
        "status": document["status"],
        "manifest": relpath(path, output),
        "prediction_mesh": document.get("prediction_mesh"),
        "blocker": document.get("blocker"),
    }


def _base_manifest(
    *,
    unit_id: str,
    dataset: str,
    track: str,
    gt_tier: str,
    capture_kind: str,
    license_status: str,
    status: str = "prepared",
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "unit_id": unit_id,
        "physical_scene_group": f"{dataset}:{unit_id}",
        "dataset": dataset,
        "track": track,
        "gt_tier": gt_tier,
        "capture_kind": capture_kind,
        "license_status": license_status,
        "status": status,
    }


def _parse_colmap_image_names(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    if len(lines) % 2:
        raise CalibrationBuildError(f"COLMAP images file has an odd number of records: {path}")
    return [line.split()[9] for line in lines[::2]]


def prepare_scannetpp(plan: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    dataset = plan["datasets"]["da3-scannetpp"]
    rows = []
    for scene_id in dataset["unit_ids"]:
        adapted = ROOT / "data" / "da3-adapted" / scene_id
        selection_path = adapted / "selection.json"
        source_scene = ROOT / "data" / "da3-bench" / "extracted" / "scannetpp" / scene_id
        merge_root = source_scene / "merge_dslr_iphone"
        source_model = merge_root / "colmap" / "sparse_render_rgb"
        reference = source_scene / "scans" / "mesh_aligned_0.05.ply"
        prediction = ROOT / "outputs" / scene_id / "reconstruction" / "mesh.ply"
        required = [
            selection_path,
            reference,
            adapted / "iphone" / "colmap" / "cameras.txt",
            adapted / "iphone" / "colmap" / "images.txt",
            source_model / "cameras.bin",
            source_model / "images.bin",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        status = "prepared" if not missing else "missing-local-source"
        unit_id = f"scannetpp-{scene_id}"
        unit_dir = output / "units" / unit_id
        manifest_path = _manifest_path(output, unit_id)
        selection = load_json(selection_path) if selection_path.is_file() else {}
        camera_records: list[dict[str, Any]] = []
        conditioning_paths: list[str] = []
        heldout_paths: list[str] = []
        conditioning_depths: list[str] = []
        heldout_depths: list[str] = []
        source_view_files: list[Path] = []
        camera_path = unit_dir / "cameras.json"
        if status == "prepared":
            cameras = read_cameras_binary(source_model / "cameras.bin")
            headers = sorted(
                (
                    item
                    for item in read_image_headers_binary(source_model / "images.bin")
                    if item.name.startswith("iphone/")
                ),
                key=lambda item: item.name,
            )
            by_name = {item.name: item for item in headers}
            conditioning_names = [item["original_name"] for item in selection["views"]]
            missing_names = sorted(set(conditioning_names).difference(by_name))
            if missing_names:
                raise CalibrationBuildError(
                    f"ScanNet++ selected views missing from source model {scene_id}: {missing_names}"
                )
            remaining = [item for item in headers if item.name not in set(conditioning_names)]
            heldout_headers = [remaining[index] for index in uniform_indices(len(remaining), 8)]
            role_headers = {
                "conditioning": [by_name[name] for name in conditioning_names],
                "heldout": heldout_headers,
            }
            for role, selected_headers in role_headers.items():
                for order, header in enumerate(selected_headers):
                    source_rgb = merge_root / "images" / header.name
                    source_depth = merge_root / "render_depth" / f"{Path(header.name).stem}.png"
                    if not source_rgb.is_file() or not source_depth.is_file():
                        raise CalibrationBuildError(
                            f"Missing ScanNet++ RGB/depth pair: {source_rgb}, {source_depth}"
                        )
                    rgb = unit_dir / "rgb" / role / f"{order:03d}{source_rgb.suffix.lower()}"
                    depth = unit_dir / "depth" / role / f"{order:03d}.png"
                    rgb.parent.mkdir(parents=True, exist_ok=True)
                    depth.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_rgb, rgb)
                    shutil.copy2(source_depth, depth)
                    camera = cameras[header.camera_id]
                    record = {
                        "role": role,
                        "order": order,
                        "source_image": header.name,
                        "rgb": relpath(rgb, unit_dir),
                        "depth": relpath(depth, unit_dir),
                        "qvec_wxyz": list(header.qvec),
                        "tvec_world_to_camera": list(header.tvec),
                        "camera_center_world_m": _camera_center(header.qvec, header.tvec),
                        "camera_id": header.camera_id,
                        "intrinsics": {
                            "model": camera.model_name,
                            "width": camera.width,
                            "height": camera.height,
                            "params": list(camera.params),
                        },
                    }
                    camera_records.append(record)
                    target_views = conditioning_paths if role == "conditioning" else heldout_paths
                    target_depths = conditioning_depths if role == "conditioning" else heldout_depths
                    target_views.append(relpath(rgb, manifest_path.parent))
                    target_depths.append(relpath(depth, manifest_path.parent))
                    source_view_files.extend([source_rgb, source_depth])
            write_json(
                camera_path,
                {
                    "schema": "genrecon.gt-camera-split",
                    "schema_version": 1,
                    "pose_convention": "world-to-camera-colmap-qvec-tvec",
                    "conditioning": [item for item in camera_records if item["role"] == "conditioning"],
                    "heldout": [item for item in camera_records if item["role"] == "heldout"],
                },
            )
        manifest = _base_manifest(
            unit_id=unit_id,
            dataset="da3-scannetpp",
            track=dataset["track"],
            gt_tier=dataset["gt_tier"],
            capture_kind=dataset["capture_kind"],
            license_status=dataset["license_status"],
            status=status,
        )
        manifest.update(
            {
                "source": {
                    "scene_id": scene_id,
                    "selection": str(selection_path.resolve()),
                    "records": _write_source_records([*required, *source_view_files]) if status == "prepared" else [],
                },
                "input": {
                    "conditioning_views": conditioning_paths,
                    "heldout_views": heldout_paths,
                    "conditioning_depths": conditioning_depths,
                    "heldout_depths": heldout_depths,
                    "cameras": relpath(camera_path, manifest_path.parent) if camera_records else None,
                    "camera_model": relpath(adapted / "iphone" / "colmap" / "cameras.txt", manifest_path.parent),
                    "images_model": relpath(adapted / "iphone" / "colmap" / "images.txt", manifest_path.parent),
                    "pose_source": "provided-registered-iphone-colmap",
                    "split_policy": "freeze-existing-8-conditioning; uniformly-select-8-disjoint-registered-iphone-heldout",
                },
                "reference": {
                    "kind": "mesh",
                    "paths": [relpath(reference, manifest_path.parent)],
                    "roi": "paper-like-unclipped",
                    "scope": "raw-global",
                    "source_type": "submillimeter-laser-scan-mesh",
                },
                "evaluation": {
                    "alignment": "provided-metric-world-frame",
                    "limitations": [
                        "GenRecon scanner observation envelope is unavailable; raw and optional GT-AABB protocols remain separate.",
                        "Heldout images are disjoint from conditioning views but participate in the provided global COLMAP model; they are not geometry-independent.",
                    ],
                },
                "prediction_mesh": relpath(prediction, output) if prediction.is_file() else None,
                "blocker": missing or None,
            }
        )
        rows.append(_write_manifest(output, manifest))
    return rows


def _eth3d_source_scene(scene: str, source_root: Path) -> Path:
    existing = ROOT / "data" / "eth3d" / "raw" / scene
    if existing.is_dir():
        return existing
    return source_root / "eth3d" / "extracted" / scene


def prepare_eth3d(plan: dict[str, Any], output: Path, source_root: Path) -> list[dict[str, Any]]:
    dataset = plan["datasets"]["eth3d-indoor-training"]
    extracted_root = source_root / "eth3d" / "extracted"
    extracted_root.mkdir(parents=True, exist_ok=True)
    for scene in dataset["unit_ids"]:
        source_scene = _eth3d_source_scene(scene, source_root)
        if source_scene.is_dir():
            continue
        for suffix in ("dslr_undistorted", "dslr_scan_eval"):
            archive = source_root / "eth3d" / f"{scene}_{suffix}.7z"
            if not archive.is_file():
                raise CalibrationBuildError(f"Missing ETH3D archive {archive}")
            safe_7z_extract(archive, extracted_root)

    prediction_map = {
        "delivery_area": ROOT / "outputs" / "eth3d" / "delivery_area" / "model_scale" / "mesh.ply",
        "pipes": ROOT / "outputs" / "eth3d" / "pipes" / "high_precision" / "mesh.ply",
    }
    rows = []
    for scene in dataset["unit_ids"]:
        source_scene = _eth3d_source_scene(scene, source_root)
        images_root = source_scene / "images"
        colmap = source_scene / "dslr_calibration_undistorted"
        scan_dir = source_scene / "dslr_scan_eval"
        images_txt = colmap / "images.txt"
        names = _parse_colmap_image_names(images_txt)
        selected = uniform_indices(len(names), min(16, len(names)))
        input_positions = selected[::2][:8]
        heldout_positions = [value for value in selected if value not in input_positions][:8]
        scans = sorted(scan_dir.glob("scan*.ply"))
        alignment = scan_dir / "scan_alignment.mlp"
        manifest = _base_manifest(
            unit_id=f"eth3d-{scene}",
            dataset="eth3d-indoor-training",
            track=dataset["track"],
            gt_tier=dataset["gt_tier"],
            capture_kind=dataset["capture_kind"],
            license_status=dataset["license_status"],
        )
        manifest_path = _manifest_path(output, manifest["unit_id"])
        archives = [
            source_root / "eth3d" / f"{scene}_{suffix}.7z"
            for suffix in ("dslr_undistorted", "dslr_scan_eval")
            if (source_root / "eth3d" / f"{scene}_{suffix}.7z").is_file()
        ]
        local_source_records = [
            colmap / "cameras.txt",
            images_txt,
            colmap / "points3D.txt",
            alignment,
            *scans,
        ]
        manifest.update(
            {
                "source": {
                    "scene_id": scene,
                    "records": _write_source_records(archives or local_source_records),
                },
                "input": {
                    "conditioning_views": [names[index] for index in input_positions],
                    "heldout_views": [names[index] for index in heldout_positions],
                    "images_root": relpath(images_root, manifest_path.parent),
                    "camera_model": relpath(colmap / "cameras.txt", manifest_path.parent),
                    "images_model": relpath(images_txt, manifest_path.parent),
                    "points_model": relpath(colmap / "points3D.txt", manifest_path.parent),
                    "pose_source": "provided-eth3d-dslr-calibration",
                },
                "reference": {
                    "kind": "pointcloud",
                    "paths": [relpath(path, manifest_path.parent) for path in scans],
                    "alignment_mlp": relpath(alignment, manifest_path.parent),
                    "roi": "full-public-scan-eval-cloud",
                    "source_type": "independent-laser-scan-evaluation-cloud",
                },
                "evaluation": {
                    "alignment": "official-meshlab-scan-alignment",
                    "limitations": ["Common suite point-cloud scores are not the official ETH3D depth-map evaluator."],
                },
                "prediction_mesh": relpath(prediction_map[scene], output) if scene in prediction_map and prediction_map[scene].is_file() else None,
            }
        )
        rows.append(_write_manifest(output, manifest))
    return rows


def _sequence_number(value: str) -> int:
    match = re.search(r"\d+", value)
    if not match:
        raise CalibrationBuildError(f"Invalid 7-Scenes sequence identifier {value!r}")
    return int(match.group())


def _nested_zip(outer: ZipFile, scene: str, sequence: int) -> ZipFile:
    name = f"{scene}/seq-{sequence:02d}.zip"
    try:
        payload = outer.read(name)
    except KeyError as exc:
        raise CalibrationBuildError(f"Missing nested sequence {name}") from exc
    nested = ZipFile(io.BytesIO(payload))
    safe_zip_names(nested)
    return nested


def _sequence_frame_count(nested: ZipFile) -> int:
    indices = {
        int(match.group(1))
        for name in nested.namelist()
        if (match := re.search(r"frame-(\d+)\.color\.png$", name))
    }
    if not indices:
        raise CalibrationBuildError("7-Scenes nested sequence contains no RGB frames")
    return max(indices) + 1


def _seven_frame_bytes(nested: ZipFile, sequence: int, frame: int, kind: str) -> bytes:
    suffix = {"rgb": "color.png", "depth": "depth.png", "pose": "pose.txt"}[kind]
    return nested.read(f"seq-{sequence:02d}/frame-{frame:06d}.{suffix}")


def _parse_pose(payload: bytes) -> np.ndarray:
    values = np.fromstring(payload.decode("ascii"), sep=" ", dtype=np.float64)
    if values.size != 16:
        raise CalibrationBuildError(f"Expected a 4x4 pose, found {values.size} values")
    matrix = values.reshape(4, 4)
    if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5):
        raise CalibrationBuildError("Invalid camera-to-world pose")
    return matrix


def _backproject_depth(depth: np.ndarray, pose_c2w: np.ndarray, intrinsics: dict[str, float], pixel_stride: int) -> np.ndarray:
    rows = np.arange(0, depth.shape[0], pixel_stride)
    cols = np.arange(0, depth.shape[1], pixel_stride)
    uu, vv = np.meshgrid(cols, rows)
    z = depth[vv, uu].astype(np.float64) / 1000.0
    valid = (z > 0.05) & (z < 20.0) & (depth[vv, uu] != 65535)
    z = z[valid]
    x = (uu[valid] - intrinsics["cx"]) * z / intrinsics["fx"]
    y = (vv[valid] - intrinsics["cy"]) * z / intrinsics["fy"]
    camera = np.column_stack((x, y, z))
    return (camera @ pose_c2w[:3, :3].T + pose_c2w[:3, 3]).astype(np.float32)


def _voxel_downsample(points: np.ndarray, voxel_m: float, max_points: int, seed: int) -> np.ndarray:
    finite = points[np.isfinite(points).all(axis=1)]
    keys = np.floor(finite / voxel_m).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    result = finite[np.sort(indices)]
    if len(result) > max_points:
        rng = np.random.default_rng(seed)
        result = result[np.sort(rng.choice(len(result), max_points, replace=False))]
    return result


def _extract_seven_views(
    nested: ZipFile,
    sequence: int,
    indices: list[int],
    destination: Path,
    role: str,
) -> list[dict[str, Any]]:
    records = []
    for order, frame in enumerate(indices):
        stem = f"{role}_{order:03d}"
        rgb = destination / "rgb" / f"{stem}.png"
        depth = destination / "depth" / f"{stem}.png"
        rgb.parent.mkdir(parents=True, exist_ok=True)
        depth.parent.mkdir(parents=True, exist_ok=True)
        rgb.write_bytes(_seven_frame_bytes(nested, sequence, frame, "rgb"))
        depth.write_bytes(_seven_frame_bytes(nested, sequence, frame, "depth"))
        pose = _parse_pose(_seven_frame_bytes(nested, sequence, frame, "pose"))
        records.append(
            {
                "role": role,
                "order": order,
                "sequence": sequence,
                "frame": frame,
                "rgb": relpath(rgb, destination),
                "depth": relpath(depth, destination),
                "camera_to_world": pose.tolist(),
                "intrinsics": SEVEN_INTRINSICS,
            }
        )
    return records


def prepare_seven_scenes(plan: dict[str, Any], output: Path, source_root: Path, *, force: bool) -> list[dict[str, Any]]:
    dataset = plan["datasets"]["seven-scenes"]
    rows = []
    tsdf_archive = source_root / "seven-scenes" / "tsdf.zip"
    for scene in dataset["unit_ids"]:
        archive_path = source_root / "seven-scenes" / f"{scene}.zip"
        unit_id = f"seven-scenes-{scene}"
        unit_dir = output / "units" / unit_id
        reference_path = unit_dir / "reference" / "fusion_reference.ply"
        camera_path = unit_dir / "cameras.json"
        if force and unit_dir.exists():
            shutil.rmtree(unit_dir)
        with ZipFile(archive_path) as outer:
            safe_zip_names(outer)
            train = outer.read(f"{scene}/TrainSplit.txt").decode("ascii").splitlines()
            test = outer.read(f"{scene}/TestSplit.txt").decode("ascii").splitlines()
            train_sequences = [_sequence_number(item) for item in train if item.strip()]
            test_sequences = [_sequence_number(item) for item in test if item.strip()]
            with _nested_zip(outer, scene, train_sequences[0]) as nested:
                count = _sequence_frame_count(nested)
                input_indices = uniform_indices(count, 8)
                input_records = _extract_seven_views(nested, train_sequences[0], input_indices, unit_dir, "conditioning")
            with _nested_zip(outer, scene, test_sequences[0]) as nested:
                count = _sequence_frame_count(nested)
                heldout_indices = uniform_indices(count, 8)
                heldout_records = _extract_seven_views(nested, test_sequences[0], heldout_indices, unit_dir, "heldout")

            sampled_frames = 0
            if not reference_path.is_file():
                batches = []
                for sequence in sorted(set(train_sequences + test_sequences)):
                    with _nested_zip(outer, scene, sequence) as nested:
                        count = _sequence_frame_count(nested)
                        for frame in range(0, count, 50):
                            depth = np.asarray(Image.open(io.BytesIO(_seven_frame_bytes(nested, sequence, frame, "depth"))))
                            pose = _parse_pose(_seven_frame_bytes(nested, sequence, frame, "pose"))
                            batches.append(_backproject_depth(depth, pose, SEVEN_INTRINSICS, pixel_stride=8))
                            sampled_frames += 1
                points = _voxel_downsample(np.concatenate(batches, axis=0), 0.01, 2_000_000, 42)
                reference_path.parent.mkdir(parents=True, exist_ok=True)
                trimesh.PointCloud(points).export(reference_path)
            else:
                for sequence in sorted(set(train_sequences + test_sequences)):
                    with _nested_zip(outer, scene, sequence) as nested:
                        count = _sequence_frame_count(nested)
                        sampled_frames += len(range(0, count, 50))
        write_json(
            camera_path,
            {
                "schema": "genrecon.gt-camera-split",
                "schema_version": 1,
                "pose_convention": "camera-to-world",
                "conditioning": input_records,
                "heldout": heldout_records,
            },
        )
        manifest = _base_manifest(
            unit_id=unit_id,
            dataset="seven-scenes",
            track=dataset["track"],
            gt_tier=dataset["gt_tier"],
            capture_kind=dataset["capture_kind"],
            license_status=dataset["license_status"],
        )
        manifest_path = _manifest_path(output, unit_id)
        manifest.update(
            {
                "source": {"scene_id": scene, "records": _write_source_records([archive_path, tsdf_archive])},
                "input": {
                    "conditioning_views": [record["rgb"] for record in input_records],
                    "heldout_views": [record["rgb"] for record in heldout_records],
                    "conditioning_depths": [record["depth"] for record in input_records],
                    "heldout_depths": [record["depth"] for record in heldout_records],
                    "cameras": relpath(camera_path, manifest_path.parent),
                    "pose_source": "official-kinectfusion-camera-to-world",
                    "intrinsics_status": "official-default-depth-intrinsics-rgb-uncalibrated",
                },
                "reference": {
                    "kind": "pointcloud",
                    "paths": [relpath(reference_path, manifest_path.parent)],
                    "roi": "all-official-train-and-test-sequences-sampled",
                    "source_type": "clean-depth-fusion-reference",
                    "generation": {"frame_stride": 50, "pixel_stride": 8, "voxel_m": 0.01, "max_points": 2_000_000, "sampled_frames": sampled_frames},
                },
                "evaluation": {
                    "alignment": "official-kinectfusion-world-frame",
                    "limitations": ["Reference depth and poses are KinectFusion-derived rather than an independent laser scan.", "Raw RGB and depth cameras are not calibrated; official default depth intrinsics are used."],
                },
                "prediction_mesh": None,
            }
        )
        rows.append(_write_manifest(output, manifest))
    return rows


def _read_redwood_poses(path: Path) -> list[np.ndarray]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 5:
        raise CalibrationBuildError(f"Redwood trajectory must contain 5 lines per frame: {path}")
    poses = []
    for start in range(0, len(lines), 5):
        header = [int(value) for value in lines[start].split()]
        matrix = np.asarray([[float(value) for value in row.split()] for row in lines[start + 1 : start + 5]])
        if len(header) != 3 or matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise CalibrationBuildError(f"Invalid Redwood pose record at line {start + 1}")
        poses.append(matrix)
    return poses


def _extract_flat_zip_frames(archive_path: Path, indices: list[int], destination: Path, suffix: str) -> list[Path]:
    paths = []
    with ZipFile(archive_path) as archive:
        names = safe_zip_names(archive)
        by_index = {int(Path(name).stem): name for name in names if Path(name).suffix.lower() == suffix}
        for order, index in enumerate(indices):
            if index not in by_index:
                raise CalibrationBuildError(f"Frame {index} is missing from {archive_path}")
            output = destination / f"{order:03d}{suffix}"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(archive.read(by_index[index]))
            paths.append(output)
    return paths


def prepare_redwood(plan: dict[str, Any], output: Path, source_root: Path, *, force: bool) -> list[dict[str, Any]]:
    dataset = plan["datasets"]["redwood-reconstruction"]
    rows = []
    for scene in dataset["unit_ids"]:
        unit_id = f"redwood-{scene}"
        unit_dir = output / "units" / unit_id
        if force and unit_dir.exists():
            shutil.rmtree(unit_dir)
        primary = int(dataset["primary_trajectory"])
        secondary = int(dataset["secondary_trajectory"])
        primary_pose_path = source_root / "redwood" / f"{scene}{primary}-traj.txt"
        secondary_pose_path = source_root / "redwood" / f"{scene}{secondary}-traj.txt"
        primary_poses = _read_redwood_poses(primary_pose_path)
        secondary_poses = _read_redwood_poses(secondary_pose_path)
        input_indices = uniform_indices(len(primary_poses), 8)
        heldout_indices = uniform_indices(len(secondary_poses), 8)
        input_rgb = _extract_flat_zip_frames(source_root / "redwood" / f"{scene}{primary}-color.zip", input_indices, unit_dir / "rgb" / "conditioning", ".jpg")
        input_depth = _extract_flat_zip_frames(source_root / "redwood" / f"{scene}{primary}-depth-clean.zip", input_indices, unit_dir / "depth" / "conditioning", ".png")
        heldout_rgb = _extract_flat_zip_frames(source_root / "redwood" / f"{scene}{secondary}-color.zip", heldout_indices, unit_dir / "rgb" / "heldout", ".jpg")
        heldout_depth = _extract_flat_zip_frames(source_root / "redwood" / f"{scene}{secondary}-depth-clean.zip", heldout_indices, unit_dir / "depth" / "heldout", ".png")
        reference_path = unit_dir / "reference" / f"{scene}.ply"
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(source_root / "redwood" / f"{scene}.ply.zip") as archive:
            safe_zip_names(archive)
            reference_path.write_bytes(archive.read(f"{scene}.ply"))
        camera_path = unit_dir / "cameras.json"
        conditioning = [
            {"order": order, "source_frame": index, "rgb": relpath(input_rgb[order], unit_dir), "depth": relpath(input_depth[order], unit_dir), "camera_to_world": primary_poses[index].tolist(), "intrinsics": REDWOOD_INTRINSICS}
            for order, index in enumerate(input_indices)
        ]
        heldout = [
            {"order": order, "source_frame": index, "rgb": relpath(heldout_rgb[order], unit_dir), "depth": relpath(heldout_depth[order], unit_dir), "camera_to_world": secondary_poses[index].tolist(), "intrinsics": REDWOOD_INTRINSICS}
            for order, index in enumerate(heldout_indices)
        ]
        write_json(camera_path, {"schema": "genrecon.gt-camera-split", "schema_version": 1, "pose_convention": "camera-to-world", "conditioning": conditioning, "heldout": heldout})
        archives = [source_root / "redwood" / f"{scene}.ply.zip"] + [source_root / "redwood" / f"{scene}{trajectory}-{kind}" for trajectory in (primary, secondary) for kind in ("color.zip", "depth-clean.zip", "traj.txt")]
        manifest = _base_manifest(unit_id=unit_id, dataset="redwood-reconstruction", track=dataset["track"], gt_tier=dataset["gt_tier"], capture_kind=dataset["capture_kind"], license_status=dataset["license_status"])
        manifest_path = _manifest_path(output, unit_id)
        manifest.update(
            {
                "source": {"scene_id": scene, "records": _write_source_records(archives)},
                "input": {
                    "conditioning_views": [item["rgb"] for item in conditioning],
                    "heldout_views": [item["rgb"] for item in heldout],
                    "conditioning_depths": [item["depth"] for item in conditioning],
                    "heldout_depths": [item["depth"] for item in heldout],
                    "cameras": relpath(camera_path, manifest_path.parent),
                    "pose_source": "official-synthetic-trajectory",
                },
                "reference": {"kind": "pointcloud", "paths": [relpath(reference_path, manifest_path.parent)], "roi": "full-exact-synthetic-surface", "source_type": "synthetic-exact-pointcloud"},
                "evaluation": {"alignment": "provided-synthetic-world-frame", "limitations": ["Synthetic imagery and exact geometry do not measure real-sensor domain robustness."]},
                "prediction_mesh": None,
            }
        )
        rows.append(_write_manifest(output, manifest))
    return rows


def _parse_dtu_camera(payload: bytes) -> dict[str, Any]:
    lines = [line.strip() for line in payload.decode("ascii").splitlines() if line.strip()]
    try:
        extrinsic_index = lines.index("extrinsic")
        intrinsic_index = lines.index("intrinsic")
    except ValueError as exc:
        raise CalibrationBuildError("Invalid DTU camera file") from exc
    w2c = np.asarray([[float(value) for value in row.split()] for row in lines[extrinsic_index + 1 : extrinsic_index + 5]], dtype=np.float64)
    intrinsics = np.asarray([[float(value) for value in row.split()] for row in lines[intrinsic_index + 1 : intrinsic_index + 4]], dtype=np.float64)
    if w2c.shape != (4, 4) or intrinsics.shape != (3, 3):
        raise CalibrationBuildError("Invalid DTU camera matrix shape")
    if not np.isfinite(w2c).all() or not np.isfinite(intrinsics).all():
        raise CalibrationBuildError("DTU camera contains non-finite values")
    if not np.allclose(w2c[3], [0, 0, 0, 1], atol=1e-8):
        raise CalibrationBuildError("Invalid DTU homogeneous extrinsic row")
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise CalibrationBuildError("DTU camera has a non-positive focal length")
    w2c[:3, 3] *= 0.001
    return {"world_to_camera": w2c.tolist(), "intrinsics": intrinsics.tolist(), "source_units": "millimeters", "output_units": "meters"}


def _scaled_pointcloud_from_zip(archive: ZipFile, name: str, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(name) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)
    loaded = trimesh.load(str(destination), process=False)
    if not isinstance(loaded, (trimesh.PointCloud, trimesh.Trimesh)):
        raise CalibrationBuildError(f"DTU reference is not point-like: {name}")
    points = np.asarray(loaded.vertices, dtype=np.float64) * 0.001
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise CalibrationBuildError(f"DTU reference has invalid vertices: {name}")
    if not np.isfinite(points).all():
        raise CalibrationBuildError(f"DTU reference has non-finite vertices: {name}")
    colors = getattr(loaded.visual, "vertex_colors", None) if isinstance(loaded, trimesh.Trimesh) else getattr(loaded, "colors", None)
    cloud = trimesh.PointCloud(points, colors=colors if colors is not None and len(colors) == len(points) else None)
    cloud.export(destination)
    return {"points": len(points), "bounds_m": np.stack((points.min(axis=0), points.max(axis=0))).tolist()}


def prepare_dtu(plan: dict[str, Any], output: Path, source_root: Path, *, force: bool) -> list[dict[str, Any]]:
    dataset = plan["datasets"]["dtu-mvs"]
    input_archive_path = source_root / "dtu" / "dtu-mvsnet-test.zip"
    reference_archive_path = source_root / "dtu" / "Points.zip"
    rows = []
    with ZipFile(input_archive_path) as inputs, ZipFile(reference_archive_path) as references:
        safe_zip_names(inputs)
        safe_zip_names(references)
        for scan in dataset["unit_ids"]:
            unit_id = f"dtu-{scan}"
            unit_dir = output / "units" / unit_id
            if force and unit_dir.exists():
                shutil.rmtree(unit_dir)
            all_indices = list(range(49))
            selected = uniform_indices(len(all_indices), 16)
            input_indices = selected[::2][:8]
            heldout_indices = [index for index in selected if index not in input_indices][:8]
            camera_records = {"conditioning": [], "heldout": []}
            for role, indices in (("conditioning", input_indices), ("heldout", heldout_indices)):
                for order, index in enumerate(indices):
                    image_name = f"dtu/{scan}/images/{index:08d}.jpg"
                    camera_name = f"dtu/{scan}/cams/{index:08d}_cam.txt"
                    image_path = unit_dir / "rgb" / role / f"{order:03d}.jpg"
                    camera_file = unit_dir / "camera_files" / role / f"{order:03d}.txt"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    camera_file.parent.mkdir(parents=True, exist_ok=True)
                    image_path.write_bytes(inputs.read(image_name))
                    payload = inputs.read(camera_name)
                    camera_file.write_bytes(payload)
                    camera_records[role].append({"order": order, "source_view": index, "rgb": relpath(image_path, unit_dir), "camera_file": relpath(camera_file, unit_dir), **_parse_dtu_camera(payload)})
            reference_path = unit_dir / "reference" / f"{scan}.ply"
            number = int(scan.removeprefix("scan"))
            reference_stats = _scaled_pointcloud_from_zip(references, f"Points/stl/stl{number:03d}_total.ply", reference_path)
            camera_path = unit_dir / "cameras.json"
            write_json(camera_path, {"schema": "genrecon.gt-camera-split", "schema_version": 1, "pose_convention": "world-to-camera", **camera_records})
            manifest = _base_manifest(unit_id=unit_id, dataset="dtu-mvs", track=dataset["track"], gt_tier=dataset["gt_tier"], capture_kind=dataset["capture_kind"], license_status=dataset["license_status"])
            manifest_path = _manifest_path(output, unit_id)
            manifest.update(
                {
                    "source": {"scan_id": scan, "input_mirror": dataset["input_mirror"], "records": _write_source_records([input_archive_path, reference_archive_path])},
                    "input": {"conditioning_views": [item["rgb"] for item in camera_records["conditioning"]], "heldout_views": [item["rgb"] for item in camera_records["heldout"]], "cameras": relpath(camera_path, manifest_path.parent), "pose_source": "preprocessed-official-DTU-camera-matrices", "scale_to_meters": 0.001},
                    "reference": {"kind": "pointcloud", "paths": [relpath(reference_path, manifest_path.parent)], "roi": "full-structured-light-reference", "source_type": "structured-light-pointcloud", "stats": reference_stats},
                    "evaluation": {"alignment": "provided-DTU-calibrated-frame-scaled-to-meters", "limitations": ["Input images are the standard MVSNet-preprocessed DTU testing archive; GT points are from the official DTU Points.zip.", "Common suite metrics do not replace the official DTU observability-mask evaluator."]},
                    "prediction_mesh": None,
                }
            )
            rows.append(_write_manifest(output, manifest))
    return rows


_PLY_SCALAR_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int16": "<i2",
    "uint16": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "int32": "<i4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def _parse_tnt_alignment(path: Path) -> dict[str, Any]:
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise CalibrationBuildError(f"Invalid T&T alignment matrix: {path}")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10):
        raise CalibrationBuildError(f"Invalid T&T homogeneous alignment row: {path}")
    linear = matrix[:3, :3]
    determinant = float(np.linalg.det(linear))
    if determinant <= 0.0:
        raise CalibrationBuildError(f"T&T alignment must be an orientation-preserving Sim(3): {path}")
    scale = float(np.cbrt(determinant))
    rotation = linear / scale
    orthogonality_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
    if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-8 or orthogonality_error > 1e-8:
        raise CalibrationBuildError(f"T&T alignment is not a valid Sim(3): {path}")
    return {
        "matrix": matrix,
        "scale": scale,
        "rotation": rotation,
        "translation": matrix[:3, 3].copy(),
        "rotation_orthogonality_error": orthogonality_error,
    }


def _transform_tnt_camera_pose(
    camera_to_colmap_world: np.ndarray, alignment: dict[str, Any]
) -> np.ndarray:
    pose = np.asarray(camera_to_colmap_world, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise CalibrationBuildError("Invalid T&T camera-to-world pose")
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = alignment["rotation"] @ pose[:3, :3]
    output[:3, 3] = (
        alignment["matrix"][:3, :3] @ pose[:3, 3] + alignment["translation"]
    )
    if not np.allclose(output[:3, :3].T @ output[:3, :3], np.eye(3), atol=1e-8):
        raise CalibrationBuildError("T&T aligned camera rotation is not orthonormal")
    return output


def _load_tnt_intrinsics(
    path: Path, image_archive: Path, camera_log: Path
) -> dict[str, Any]:
    document = load_json(path)
    if document.get("schema") != "genrecon.tnt-fixed-pose-intrinsics":
        raise CalibrationBuildError(f"Unexpected T&T calibration schema: {path}")
    if document.get("schema_version") != 2:
        raise CalibrationBuildError(f"Unsupported T&T calibration version: {path}")
    if document.get("protocol", {}).get("revision") != "v2-single-thread-geometry":
        raise CalibrationBuildError(f"Unsupported T&T calibration protocol: {path}")
    gates = document.get("quality_gates")
    if (
        document.get("result") != "pass"
        or not isinstance(gates, dict)
        or not gates
        or not all(gates.values())
    ):
        raise CalibrationBuildError(f"T&T calibration quality gates did not pass: {path}")
    expected = {
        "image_archive_sha256": sha256_file(image_archive),
        "camera_log_sha256": sha256_file(camera_log),
    }
    for key, value in expected.items():
        if document.get("source", {}).get(key) != value:
            raise CalibrationBuildError(f"T&T calibration source hash mismatch for {key}")
    calibration = document.get("calibration", {})
    params = np.asarray(calibration.get("params", []), dtype=np.float64)
    if (
        calibration.get("model") != "SIMPLE_RADIAL"
        or params.shape != (4,)
        or not np.isfinite(params).all()
        or calibration.get("width", 0) <= 0
        or calibration.get("height", 0) <= 0
    ):
        raise CalibrationBuildError(f"Invalid T&T recovered intrinsics: {path}")
    return document


def _tnt_undistortion(calibration: dict[str, Any]) -> dict[str, Any]:
    camera = calibration["calibration"]
    width, height = int(camera["width"]), int(camera["height"])
    focal, cx, cy, radial = (float(value) for value in camera["params"])
    source_intrinsic = np.asarray(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    distortion = np.asarray([radial, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    new_intrinsic, roi = cv2.getOptimalNewCameraMatrix(
        source_intrinsic,
        distortion,
        (width, height),
        alpha=0,
        newImgSize=(width, height),
    )
    x, y, output_width, output_height = (int(value) for value in roi)
    if output_width <= 0 or output_height <= 0:
        raise CalibrationBuildError("T&T undistortion produced an empty ROI")
    map_x, map_y = cv2.initUndistortRectifyMap(
        source_intrinsic,
        distortion,
        None,
        new_intrinsic,
        (width, height),
        cv2.CV_32FC1,
    )
    output_intrinsic = new_intrinsic.copy()
    output_intrinsic[0, 2] -= x
    output_intrinsic[1, 2] -= y
    return {
        "source_intrinsic": source_intrinsic,
        "distortion": distortion,
        "map_x": map_x,
        "map_y": map_y,
        "roi": [x, y, output_width, output_height],
        "intrinsic": output_intrinsic,
        "width": output_width,
        "height": output_height,
    }


def _point_in_polygon_xy(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    previous = polygon[-1]
    for current in polygon:
        x0, y0 = previous
        x1, y1 = current
        crossing = (y0 > y) != (y1 > y)
        denominator = y1 - y0
        x_intersection = (x1 - x0) * (y - y0) / (
            denominator if abs(denominator) > 1e-15 else 1e-15
        ) + x0
        inside ^= crossing & (x < x_intersection)
        previous = current
    return inside


def _read_exact(handle: Any, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        block = handle.read(remaining)
        if not block:
            raise CalibrationBuildError(f"Unexpected end of binary PLY ({remaining} bytes missing)")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def _read_binary_ply_header(handle: Any) -> tuple[int, np.dtype[Any]]:
    first = handle.readline()
    if first.strip() != b"ply":
        raise CalibrationBuildError("Invalid PLY signature in T&T scans archive")
    vertex_count = None
    current_element = None
    properties: list[tuple[str, str]] = []
    while True:
        raw = handle.readline()
        if not raw:
            raise CalibrationBuildError("Truncated PLY header in T&T scans archive")
        line = raw.decode("ascii").strip()
        if line == "end_header":
            break
        values = line.split()
        if values[:2] == ["format", "binary_little_endian"]:
            continue
        if values and values[0] == "format":
            raise CalibrationBuildError(f"Unsupported T&T PLY format: {line}")
        if len(values) == 3 and values[0] == "element":
            current_element = values[1]
            if current_element == "vertex":
                vertex_count = int(values[2])
            continue
        if values and values[0] == "property" and current_element == "vertex":
            if len(values) != 3 or values[1] == "list":
                raise CalibrationBuildError(f"Unsupported T&T vertex property: {line}")
            if values[1] not in _PLY_SCALAR_TYPES:
                raise CalibrationBuildError(f"Unsupported T&T PLY scalar type: {values[1]}")
            properties.append((values[2], _PLY_SCALAR_TYPES[values[1]]))
    if vertex_count is None or vertex_count <= 0:
        raise CalibrationBuildError("T&T PLY has no vertices")
    required = {"x", "y", "z", "red", "green", "blue"}
    if not required.issubset(name for name, _ in properties):
        raise CalibrationBuildError("T&T PLY is missing XYZ or RGB vertex properties")
    return vertex_count, np.dtype(properties, align=False)


def _write_binary_pointcloud(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    if points.shape != (len(colors), 3) or colors.shape[1:] != (3,):
        raise CalibrationBuildError("Point/color shape mismatch while writing T&T reference")
    payload = np.empty(
        len(points),
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
            align=False,
        ),
    )
    payload["x"], payload["y"], payload["z"] = points.T.astype(np.float32)
    payload["red"], payload["green"], payload["blue"] = colors.T.astype(np.uint8)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment GenRecon T&T official-scan 1cm voxel reference\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(payload.tobytes())


def _build_tnt_reference(
    scans_archive: Path,
    crop_json: Path,
    output_path: Path,
    metadata_path: Path,
    *,
    force: bool,
    voxel_m: float = 0.01,
    max_points: int | None = None,
) -> dict[str, Any]:
    source_hashes = {
        "scans_archive_sha256": sha256_file(scans_archive),
        "crop_json_sha256": sha256_file(crop_json),
    }
    if output_path.is_file() and metadata_path.is_file() and not force:
        metadata = load_json(metadata_path)
        if (
            metadata.get("source") == source_hashes
            and metadata.get("protocol", {}).get("voxel_m") == voxel_m
            and metadata.get("protocol", {}).get("max_points") == max_points
            and metadata.get("output", {}).get("sha256") == sha256_file(output_path)
        ):
            return metadata

    crop = load_json(crop_json)
    if crop.get("orthogonal_axis") != "Z":
        raise CalibrationBuildError("T&T adapter currently requires a Z-axis crop volume")
    polygon = np.asarray(crop.get("bounding_polygon", []), dtype=np.float64)[:, :2]
    z_min, z_max = float(crop["axis_min"]), float(crop["axis_max"])
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise CalibrationBuildError("Invalid T&T crop polygon")
    bounds_min = np.asarray([*polygon.min(axis=0), z_min], dtype=np.float64)
    bounds_max = np.asarray([*polygon.max(axis=0), z_max], dtype=np.float64)
    voxel_min = np.floor(bounds_min / voxel_m).astype(np.int64) - 1
    voxel_max = np.floor(bounds_max / voxel_m).astype(np.int64) + 1
    dimensions = voxel_max - voxel_min + 1
    if int(np.prod(dimensions.astype(object))) >= np.iinfo(np.int64).max:
        raise CalibrationBuildError("T&T voxel key range exceeds int64")

    all_keys: list[np.ndarray] = []
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    source_points = 0
    cropped_points = 0
    scan_names = []
    with ZipFile(scans_archive) as archive:
        names = safe_zip_names(archive)
        scan_names = sorted(
            name for name in names if Path(name).suffix.lower() == ".ply"
        )
        if not scan_names:
            raise CalibrationBuildError(f"No PLY scans in {scans_archive}")
        for name in scan_names:
            with archive.open(name) as handle:
                vertex_count, dtype = _read_binary_ply_header(handle)
                source_points += vertex_count
                remaining = vertex_count
                while remaining:
                    count = min(remaining, 500_000)
                    vertices = np.frombuffer(
                        _read_exact(handle, count * dtype.itemsize), dtype=dtype
                    )
                    points = np.column_stack(
                        (vertices["x"], vertices["y"], vertices["z"])
                    ).astype(np.float64)
                    colors = np.column_stack(
                        (vertices["red"], vertices["green"], vertices["blue"])
                    ).astype(np.uint8)
                    finite = np.isfinite(points).all(axis=1)
                    in_z = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
                    mask = finite & in_z & _point_in_polygon_xy(points, polygon)
                    points = points[mask]
                    colors = colors[mask]
                    cropped_points += len(points)
                    if len(points):
                        indices = np.floor(points / voxel_m).astype(np.int64) - voxel_min
                        keys = (
                            (indices[:, 0] * dimensions[1] + indices[:, 1])
                            * dimensions[2]
                            + indices[:, 2]
                        )
                        _, unique_indices = np.unique(keys, return_index=True)
                        unique_indices.sort()
                        all_keys.append(keys[unique_indices])
                        all_points.append(points[unique_indices].astype(np.float32))
                        all_colors.append(colors[unique_indices])
                    remaining -= count

    keys = np.concatenate(all_keys)
    points = np.concatenate(all_points)
    colors = np.concatenate(all_colors)
    _, unique_indices = np.unique(keys, return_index=True)
    unique_indices.sort()
    points = points[unique_indices]
    colors = colors[unique_indices]
    full_voxel_bounds = np.stack((points.min(axis=0), points.max(axis=0)))
    voxel_points = len(points)
    if max_points is not None and len(points) > max_points:
        rng = np.random.default_rng(42)
        selected = np.sort(rng.choice(len(points), max_points, replace=False))
        points = points[selected]
        colors = colors[selected]
    _write_binary_pointcloud(output_path, points, colors)
    metadata = {
        "schema": "genrecon.tnt-reference-build",
        "schema_version": 1,
        "source": source_hashes,
        "protocol": {
            "crop": "official SelectionPolygonVolume",
            "voxel_m": voxel_m,
            "max_points": max_points,
            "seed": 42,
            "overlap_policy": "one deterministic representative per global voxel",
        },
        "stats": {
            "scan_count": len(scan_names),
            "scan_names": scan_names,
            "source_points": source_points,
            "points_after_official_crop_before_deduplication": cropped_points,
            "global_voxel_points": voxel_points,
            "exported_points": len(points),
            "bounds_m": full_voxel_bounds.tolist(),
        },
        "output": {
            "path": str(output_path.resolve()),
            "size_bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
    }
    write_json(metadata_path, metadata)
    return metadata


def _validate_existing_frozen_hashes(
    source: Path, hash_groups: list[dict[str, str]]
) -> None:
    for hashes in hash_groups:
        for filename, expected_sha256 in hashes.items():
            path = source / filename
            if not path.is_file():
                continue
            actual_sha256 = sha256_file(path)
            if actual_sha256 != expected_sha256:
                raise CalibrationBuildError(
                    f"T&T frozen SHA256 mismatch for {path}: "
                    f"{actual_sha256} != {expected_sha256}"
                )


def prepare_tanks_and_temples(
    plan: dict[str, Any],
    output: Path,
    source_root: Path,
    *,
    force: bool,
) -> list[dict[str, Any]]:
    dataset = plan["datasets"]["tanks-and-temples-training"]
    unit_id = "tanks-and-temples-meetingroom"
    source = source_root / "tanks-and-temples"
    unit_dir = output / "units" / unit_id
    video = source / "Meetingroom.mp4"
    images = source / "Meetingroom.zip"
    camera_log = source / "Meetingroom_COLMAP.log"
    crop_json = source / "Meetingroom_crop.json"
    scans_archive = source / "Meetingroom_individual_scans.zip"
    alignment = source / "Meetingroom_alignment.txt"
    intrinsics_artifact = source / "Meetingroom_intrinsics.json"
    _validate_existing_frozen_hashes(
        source,
        [
            dataset.get("frozen_import_sha256", {}),
            dataset.get("frozen_derived_sha256", {}),
        ],
    )
    input_required = [video, images, camera_log, crop_json]
    input_valid = (
        all(path.is_file() for path in input_required)
        and file_prefix(video, 8)[4:8] == b"ftyp"
        and file_prefix(images, 4) == b"PK\x03\x04"
    )
    scans_valid = scans_archive.is_file() and file_prefix(scans_archive, 4) == b"PK\x03\x04"
    alignment_valid = alignment.is_file() and alignment.stat().st_size > 0
    calibration_valid = False
    calibration: dict[str, Any] | None = None
    calibration_error: str | None = None
    if input_valid and intrinsics_artifact.is_file():
        try:
            calibration = _load_tnt_intrinsics(intrinsics_artifact, images, camera_log)
            calibration_valid = True
        except Exception as exc:
            calibration_error = f"{type(exc).__name__}: {exc}"
    source_complete = input_valid and scans_valid and alignment_valid and calibration_valid
    if force and unit_dir.exists():
        shutil.rmtree(unit_dir)
    camera_records: list[dict[str, Any]] = []
    reference_path = unit_dir / "reference" / "Meetingroom_1cm_crop.ply"
    reference_build_path = unit_dir / "reference" / "build.json"
    reference_build: dict[str, Any] | None = None
    alignment_info: dict[str, Any] | None = None
    if input_valid and alignment_valid:
        alignment_info = _parse_tnt_alignment(alignment)
    if source_complete:
        assert calibration is not None and alignment_info is not None
        undistortion = _tnt_undistortion(calibration)
        poses = _read_redwood_poses(camera_log)
        with ZipFile(images) as archive:
            names = safe_zip_names(archive)
            image_names = sorted(
                name
                for name in names
                if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if len(image_names) != len(poses):
                raise CalibrationBuildError(
                    f"T&T image/pose count mismatch: {len(image_names)} != {len(poses)}"
                )
            selected = uniform_indices(len(image_names), 16)
            input_indices = selected[::2][:8]
            heldout_indices = [index for index in selected if index not in input_indices][:8]
            x, y, width, height = undistortion["roi"]
            for role, indices in (("conditioning", input_indices), ("heldout", heldout_indices)):
                for order, index in enumerate(indices):
                    path = unit_dir / "rgb" / role / f"{order:03d}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(io.BytesIO(archive.read(image_names[index]))) as opened:
                        rgb = np.asarray(opened.convert("RGB"))
                    undistorted = cv2.remap(
                        rgb,
                        undistortion["map_x"],
                        undistortion["map_y"],
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                    )[y : y + height, x : x + width]
                    if undistorted.shape[:2] != (height, width):
                        raise CalibrationBuildError("Unexpected T&T undistorted image shape")
                    Image.fromarray(undistorted).save(path, compress_level=6)
                    camera_to_gt = _transform_tnt_camera_pose(poses[index], alignment_info)
                    camera_records.append(
                        {
                            "role": role,
                            "order": order,
                            "source_index": index,
                            "source_image": image_names[index],
                            "rgb": relpath(path, unit_dir),
                            "camera_to_world": camera_to_gt.tolist(),
                            "camera_to_colmap_world": poses[index].tolist(),
                            "image_size": [width, height],
                            "intrinsics": undistortion["intrinsic"].tolist(),
                            "source_camera_model": calibration["calibration"],
                        }
                    )
        write_json(
            unit_dir / "cameras.json",
            {
                "schema": "genrecon.gt-camera-split",
                "schema_version": 1,
                "pose_convention": "camera-to-world",
                "world_frame": "official laser GT frame",
                "intrinsics_status": "recovered-with-all-official-poses-fixed",
                "intrinsics_artifact": relpath(intrinsics_artifact, unit_dir),
                "source_to_gt_sim3": {
                    "matrix": alignment_info["matrix"].tolist(),
                    "scale": alignment_info["scale"],
                    "rotation_orthogonality_error": alignment_info[
                        "rotation_orthogonality_error"
                    ],
                },
                "undistortion": {
                    "source_model": "SIMPLE_RADIAL",
                    "source_params": calibration["calibration"]["params"],
                    "roi": undistortion["roi"],
                    "output_model": "PINHOLE",
                    "output_intrinsic": undistortion["intrinsic"].tolist(),
                },
                "conditioning": [item for item in camera_records if item["role"] == "conditioning"],
                "heldout": [item for item in camera_records if item["role"] == "heldout"],
            },
        )
        reference_build = _build_tnt_reference(
            scans_archive,
            crop_json,
            reference_path,
            reference_build_path,
            force=force,
        )
        status = "prepared"
        blocker = None
    elif input_valid and not (scans_valid and alignment_valid):
        status = "blocked-reference-download"
        blocker = "Official Meetingroom laser scans/alignment are incomplete."
    elif input_valid:
        status = "blocked-calibration"
        blocker = calibration_error or "Fixed-pose Meetingroom intrinsics calibration is unavailable."
    else:
        status = "blocked-download"
        blocker = "Official source video/image/camera assets are incomplete."

    manifest = _base_manifest(
        unit_id=unit_id,
        dataset="tanks-and-temples-training",
        track=dataset["track"],
        gt_tier=dataset["gt_tier"],
        capture_kind=dataset["capture_kind"],
        license_status=dataset["license_status"],
        status=status,
    )
    manifest_path = _manifest_path(output, unit_id)
    source_paths = [
        *input_required,
        *([scans_archive] if scans_valid else []),
        *([alignment] if alignment_valid else []),
        *([intrinsics_artifact] if intrinsics_artifact.is_file() else []),
    ]
    manifest.update(
        {
            "source": {
                "scene_id": "Meetingroom",
                "records": _write_source_records(source_paths) if input_valid else [],
            },
            "input": {
                "conditioning_views": [item["rgb"] for item in camera_records if item["role"] == "conditioning"],
                "heldout_views": [item["rgb"] for item in camera_records if item["role"] == "heldout"],
                "source_video": str(video.resolve()),
                "image_archive": str(images.resolve()),
                "cameras": relpath(unit_dir / "cameras.json", manifest_path.parent) if camera_records else None,
                "pose_source": (
                    "official-training-COLMAP-log transformed by official Sim(3)"
                    if camera_records
                    else "blocked"
                ),
                "intrinsics_source": (
                    "fixed-official-pose track calibration"
                    if calibration_valid
                    else "blocked"
                ),
            },
            "reference": {
                "kind": "pointcloud",
                "paths": [relpath(reference_path, manifest_path.parent)] if reference_build else [],
                "alignment_file": relpath(alignment, manifest_path.parent) if alignment_valid else None,
                "official_crop_json": relpath(crop_json, manifest_path.parent) if crop_json.is_file() else None,
                "roi": "official-training-SelectionPolygonVolume",
                "scope": "official-crop-global-reference",
                "source_type": "official-prealigned-individual-laser-scans",
                "generation": reference_build,
            },
            "evaluation": {
                "alignment": "official-COLMAP-to-laser-GT-Sim3" if alignment_valid else "blocked",
                "pose_track": "GT-pose-official-alignment",
                "limitations": [
                    "The official log omits exact intrinsics; shared SIMPLE_RADIAL intrinsics are recovered from all 371 image tracks with official poses held fixed.",
                    "Heldout RGB is disjoint from the 8 conditioning views but contributed to intrinsics calibration and the official global COLMAP trajectory; it is not geometry-independent heldout.",
                    "The common point-cloud metrics under the official-crop scope are not a replacement for the official Tanks and Temples evaluator.",
                    "The laser reference is used only for evaluation, never as GenRecon conditioning geometry.",
                ],
            },
            "prediction_mesh": None,
            "blocker": blocker,
        }
    )
    return [_write_manifest(output, manifest)]


def _omni_blocked_manifest(
    dataset: dict[str, Any],
    output: Path,
    object_id: str,
    *,
    status: str,
    blocker: str,
    source_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    unit_dir = output / "units" / f"omniobject3d-{object_id}"
    if unit_dir.exists():
        shutil.rmtree(unit_dir)
    manifest = _base_manifest(
        unit_id=f"omniobject3d-{object_id}",
        dataset="omniobject3d",
        track=dataset["track"],
        gt_tier=dataset["gt_tier"],
        capture_kind=dataset["capture_kind"],
        license_status=dataset["license_status"],
        status=status,
    )
    manifest["physical_scene_group"] = f"omniobject3d:{object_id}"
    manifest.update(
        {
            "source": {
                "object_id": object_id,
                "category": _omni_category(object_id),
                "url": dataset["source_url"],
                "dataset_repo": OMNI_DATASET_REPO,
                "adapter_version": OMNI_ADAPTER_VERSION,
                "records": source_records or [],
            },
            "input": {
                "conditioning_views": [],
                "heldout_views": [],
                "pose_source": "blocked",
            },
            "reference": {
                "kind": "mesh",
                "paths": [],
                "roi": "full-object",
                "scope": "normalized-object-global-reference",
                "coordinate_units": "normalized-object",
                "source_type": "professional-real-object-scan",
            },
            "evaluation": {
                "alignment": status,
                "pose_track": "GT-pose-scan-render",
                "limitations": [blocker],
            },
            "prediction_mesh": None,
            "blocker": blocker,
        }
    )
    return _write_manifest(output, manifest)


def _omni_archive_record(
    index_by_path: dict[str, dict[str, Any]],
    openxlab_path: str,
    local_path: Path,
) -> dict[str, Any]:
    expected = index_by_path.get(openxlab_path)
    if expected is None:
        raise CalibrationBuildError(
            f"OpenXLab file index is missing {openxlab_path}"
        )
    if (
        not local_path.is_file()
        or local_path.stat().st_size != expected.get("size")
        or sha256_file(local_path) != expected.get("sha256")
    ):
        raise CalibrationBuildError(
            f"OmniObject3D archive differs from OpenXLab index: {local_path}"
        )
    return {
        "openxlab_path": openxlab_path,
        "local_path": local_path,
        "size_bytes": local_path.stat().st_size,
        "sha256": expected["sha256"],
    }


def _prepare_omni_unit(
    dataset: dict[str, Any],
    output: Path,
    object_id: str,
    image_record: dict[str, Any],
    scan_record: dict[str, Any],
    file_index_path: Path,
    *,
    force: bool,
) -> dict[str, Any]:
    category = _omni_category(object_id)
    unit_id = f"omniobject3d-{object_id}"
    unit_dir = output / "units" / unit_id
    if force and unit_dir.exists():
        shutil.rmtree(unit_dir)
    manifest_path = _manifest_path(output, unit_id)
    image_archive_path = image_record["local_path"]
    scan_archive_path = scan_record["local_path"]

    with tarfile.open(image_archive_path, "r:gz") as image_archive, tarfile.open(
        scan_archive_path, "r:gz"
    ) as scan_archive:
        image_members = safe_tar_members(image_archive)
        scan_members = safe_tar_members(scan_archive)
        transforms_member = _unique_tar_member(
            image_members,
            f"{object_id}/render/transforms.json",
            image_archive_path,
        )
        scan_member = _unique_tar_member(
            scan_members, f"{object_id}/Scan/Scan.obj", scan_archive_path
        )
        transforms_payload = _read_tar_member(image_archive, transforms_member)
        scan_payload = _read_tar_member(scan_archive, scan_member)
        try:
            transforms = json.loads(transforms_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CalibrationBuildError(
                f"Invalid OmniObject3D transforms for {object_id}: {exc}"
            ) from exc
        frames = transforms.get("frames")
        if not isinstance(frames, list) or len(frames) != 100:
            raise CalibrationBuildError(
                f"OmniObject3D {object_id} must contain exactly 100 rendered views"
            )
        source_names = [str(frame.get("file_path", "")) for frame in frames]
        expected_names = {f"r_{index}" for index in range(100)}
        if len(source_names) != len(set(source_names)) or set(source_names) != expected_names:
            raise CalibrationBuildError(
                f"OmniObject3D {object_id} has an invalid 100-view filename set"
            )
        try:
            official_scales = np.asarray(
                [frame.get("scale") for frame in frames], dtype=np.float64
            )
        except (TypeError, ValueError) as exc:
            raise CalibrationBuildError(
                f"OmniObject3D {object_id} has an invalid official render scale"
            ) from exc
        if (
            official_scales.shape != (100,)
            or not np.isfinite(official_scales).all()
            or np.any(official_scales <= 0.0)
            or not np.allclose(
                official_scales, official_scales[0], rtol=0.0, atol=1e-15
            )
        ):
            raise CalibrationBuildError(
                f"OmniObject3D {object_id} has inconsistent official render scales"
            )
        rgb_members = {}
        normal_members = {}
        depth_members = {}
        for source_name in source_names:
            rgb_members[source_name] = _unique_tar_member(
                image_members,
                f"{object_id}/render/images/{source_name}.png",
                image_archive_path,
            )
            normal_members[source_name] = _unique_tar_member(
                image_members,
                f"{object_id}/render/normals/{source_name}_normal.png",
                image_archive_path,
            )
            depth_members[source_name] = _unique_tar_member(
                image_members,
                f"{object_id}/render/depths/{source_name}_depth.exr",
                image_archive_path,
            )
        if any(
            member.size <= 0
            for member in [
                *rgb_members.values(),
                *normal_members.values(),
                *depth_members.values(),
            ]
        ):
            raise CalibrationBuildError(
                f"OmniObject3D {object_id} has an empty render member"
            )

        mesh = _load_omni_scan_mesh(scan_payload)
        selected_scale, scale_selection = _omni_select_render_scale(
            mesh, official_scales[0]
        )
        normalized_mesh, normalization = _omni_scan_to_render_mesh(
            mesh, selected_scale
        )
        normalization["scale_selection"] = scale_selection
        vertices = np.asarray(normalized_mesh.vertices, dtype=np.float64)
        faces = np.asarray(normalized_mesh.faces)
        pose_pairs = [
            _omni_blender_c2w_to_opencv_with_audit(
                frame.get("transform_matrix")
            )
            for frame in frames
        ]
        poses = [pair[0] for pair in pose_pairs]
        pose_audits = [pair[1] for pair in pose_pairs]
        centers = np.stack([pose[:3, 3] for pose in poses])
        conditioning_indices = farthest_point_indices(centers, 8)
        heldout_indices = farthest_point_indices(
            centers, 8, excluded=set(conditioning_indices)
        )
        selected = {
            "conditioning": conditioning_indices,
            "heldout": heldout_indices,
        }
        selected_indices = set(conditioning_indices + heldout_indices)
        selected_payloads = _read_tar_members_in_archive_order(
            image_archive,
            [
                *depth_members.values(),
                *(
                    rgb_members[source_names[index]]
                    for index in sorted(selected_indices)
                ),
            ],
        )

        image_metadata: dict[int, dict[str, Any]] = {}
        alignment_views = []
        raycast_scene = None
        for source_index, (frame, pose) in enumerate(zip(frames, poses)):
            source_name = str(frame["file_path"])
            depth_member = depth_members[source_name]
            depth_payload = selected_payloads[_normalized_tar_name(depth_member)]
            depth_mask, image_size, depth_bbox, depth_metadata = _omni_depth_mask(
                depth_payload
            )
            if image_size != (OMNI_RENDER_RESOLUTION, OMNI_RENDER_RESOLUTION):
                raise CalibrationBuildError(
                    f"OmniObject3D full render has unexpected dimensions {image_size}"
                )
            intrinsic = _omni_intrinsic(
                float(transforms.get("camera_angle_x")), image_size
            )
            projected_bbox = _omni_projection_bbox(
                vertices, pose, intrinsic, image_size
            )
            depth_bbox_array = np.asarray(depth_bbox, dtype=np.float64)
            fast_edge_error = np.abs(projected_bbox - depth_bbox_array)
            fast_bbox_iou = _bbox_iou(projected_bbox, depth_bbox_array)
            fast_pass = bool(
                float(fast_edge_error.max()) <= OMNI_MAX_SILHOUETTE_EDGE_ERROR_PX
                and fast_bbox_iou >= OMNI_MIN_SILHOUETTE_BBOX_IOU
            )
            acceptance_method = "fast-projected-vertices"
            accepted_bbox = projected_bbox
            edge_error = fast_edge_error
            bbox_iou = fast_bbox_iou
            raycast_diagnostic = None
            gate_pass = fast_pass
            if not fast_pass:
                if raycast_scene is None:
                    raycast_scene = _omni_raycast_scene(normalized_mesh)
                raycast_mask = _omni_raycast_mask(
                    raycast_scene, pose, intrinsic, image_size
                )
                raycast_bbox = _mask_bbox(raycast_mask)
                raycast_edge_error = np.abs(raycast_bbox - depth_bbox_array)
                raycast_bbox_iou = _bbox_iou(raycast_bbox, depth_bbox_array)
                raycast_mask_iou = _mask_iou(raycast_mask, depth_mask)
                gate_pass = bool(
                    float(raycast_edge_error.max())
                    <= OMNI_MAX_SILHOUETTE_EDGE_ERROR_PX
                    and raycast_bbox_iou >= OMNI_MIN_SILHOUETTE_BBOX_IOU
                    and raycast_mask_iou >= OMNI_MIN_RAYCAST_MASK_IOU
                )
                acceptance_method = "visibility-aware-raycast-fallback"
                accepted_bbox = raycast_bbox
                edge_error = raycast_edge_error
                bbox_iou = raycast_bbox_iou
                raycast_diagnostic = {
                    "raycast_bbox_xyxy": raycast_bbox.tolist(),
                    "max_edge_error_px": float(raycast_edge_error.max()),
                    "bbox_iou": raycast_bbox_iou,
                    "mask_iou": raycast_mask_iou,
                    "raycast_foreground_pixels": int(raycast_mask.sum()),
                    "official_foreground_pixels": int(depth_mask.sum()),
                    "pass": gate_pass,
                }
            diagnostic = {
                "source_index": source_index,
                "source_image": f"{source_name}.png",
                "camera_pose_audit": pose_audits[source_index],
                "depth_member": _normalized_tar_name(depth_member),
                "depth_size_bytes": len(depth_payload),
                "depth_sha256": _sha256_bytes(depth_payload),
                "depth_mask_bbox_xyxy": depth_bbox,
                "depth_mask": depth_metadata,
                "fast_projection": {
                    "projected_vertex_bbox_xyxy": projected_bbox.tolist(),
                    "max_edge_error_px": float(fast_edge_error.max()),
                    "bbox_iou": fast_bbox_iou,
                    "pass": fast_pass,
                },
                "raycast_fallback": raycast_diagnostic,
                "acceptance_method": acceptance_method,
                "accepted_scan_bbox_xyxy": accepted_bbox.tolist(),
                "max_edge_error_px": float(edge_error.max()),
                "bbox_iou": bbox_iou,
                "pass": gate_pass,
            }
            alignment_views.append(diagnostic)
            if not gate_pass:
                raise OmniAlignmentError(
                    f"OmniObject3D scan/render alignment failed for "
                    f"{object_id}/{source_name}: {diagnostic}"
                )
            if source_index in selected_indices:
                rgb_member = rgb_members[source_name]
                source_rgb_payload = selected_payloads[
                    _normalized_tar_name(rgb_member)
                ]
                rgb_payload = _omni_rgb_on_white(
                    source_rgb_payload, depth_mask, image_size
                )
                image_metadata[source_index] = {
                    "rgb_payload": rgb_payload,
                    "image_size": image_size,
                    "intrinsic": intrinsic,
                    "source_rgb_member": _normalized_tar_name(rgb_member),
                    "source_rgb_sha256": _sha256_bytes(source_rgb_payload),
                    "source_rgb_size_bytes": len(source_rgb_payload),
                    "depth_member": _normalized_tar_name(depth_member),
                    "depth_sha256": diagnostic["depth_sha256"],
                    "depth_size_bytes": len(depth_payload),
                    "depth_mask_bbox_xyxy": depth_bbox,
                    "alignment": diagnostic,
                }

    reference_path = unit_dir / "reference" / "scan_normalized.ply"
    _write_binary_mesh_ply(normalized_mesh, reference_path)
    del mesh, normalized_mesh
    camera_records: dict[str, list[dict[str, Any]]] = {
        "conditioning": [],
        "heldout": [],
    }
    conditioning_paths = []
    heldout_paths = []
    for role, indices in selected.items():
        for order, source_index in enumerate(indices):
            output_path = unit_dir / "rgb" / role / f"{order:03d}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(image_metadata[source_index]["rgb_payload"])
            relative_rgb = relpath(output_path, unit_dir)
            record = {
                "order": order,
                "source_index": source_index,
                "source_image": image_metadata[source_index]["source_rgb_member"],
                "rgb": relative_rgb,
                "camera_to_world": poses[source_index].tolist(),
                "camera_pose_audit": pose_audits[source_index],
                "intrinsics": image_metadata[source_index]["intrinsic"].tolist(),
                "width": image_metadata[source_index]["image_size"][0],
                "height": image_metadata[source_index]["image_size"][1],
                "source_rgb_member": image_metadata[source_index][
                    "source_rgb_member"
                ],
                "source_rgb_size_bytes": image_metadata[source_index][
                    "source_rgb_size_bytes"
                ],
                "source_rgb_sha256": image_metadata[source_index][
                    "source_rgb_sha256"
                ],
                "source_depth_member": image_metadata[source_index][
                    "depth_member"
                ],
                "source_depth_size_bytes": image_metadata[source_index][
                    "depth_size_bytes"
                ],
                "source_depth_sha256": image_metadata[source_index][
                    "depth_sha256"
                ],
                "depth_mask_bbox_xyxy": image_metadata[source_index][
                    "depth_mask_bbox_xyxy"
                ],
                "scan_projection_alignment": image_metadata[source_index]["alignment"],
            }
            camera_records[role].append(record)
            manifest_relative = relpath(output_path, manifest_path.parent)
            if role == "conditioning":
                conditioning_paths.append(manifest_relative)
            else:
                heldout_paths.append(manifest_relative)

    alignment_summary = {
        "schema": "genrecon.omniobject3d-scan-render-alignment",
        "schema_version": 3,
        "unit_id": unit_id,
        "method": "official raw scan transformed by an audited per-object scale selected from the full-render metadata or, only when that scale violates the published renderer target by more than 10x, recomputed from the published 0.99/max-abs formula; Blender OBJ import axis is then applied; all 100 official depth masks are first checked against projected vertex bounds, with visibility-aware Open3D raycasting required for any failed fast check",
        "thresholds": {
            "maximum_edge_error_px": OMNI_MAX_SILHOUETTE_EDGE_ERROR_PX,
            "minimum_bbox_iou": OMNI_MIN_SILHOUETTE_BBOX_IOU,
            "minimum_raycast_mask_iou": OMNI_MIN_RAYCAST_MASK_IOU,
            "published_render_target_max_abs_coordinate": OMNI_RENDER_TARGET_MAX_ABS,
            "recorded_scale_max_target_ratio": OMNI_RECORDED_SCALE_MAX_TARGET_RATIO,
            "maximum_camera_orthogonality_for_repair": OMNI_MAX_CAMERA_ORTHOGONALITY_FOR_REPAIR,
            "maximum_camera_rotation_repair": OMNI_MAX_CAMERA_ROTATION_REPAIR,
        },
        "view_count": len(alignment_views),
        "fast_projection_pass_count": sum(
            item["fast_projection"]["pass"] for item in alignment_views
        ),
        "raycast_fallback_count": sum(
            item["raycast_fallback"] is not None for item in alignment_views
        ),
        "bounded_camera_repair_count": sum(
            item["camera_pose_audit"]["method"] != "unchanged-within-tolerance"
            for item in alignment_views
        ),
        "maximum_edge_error_px": max(
            item["max_edge_error_px"] for item in alignment_views
        ),
        "minimum_bbox_iou": min(item["bbox_iou"] for item in alignment_views),
        "views": alignment_views,
        "result": "pass",
    }
    alignment_path = unit_dir / "scan_render_alignment.json"
    write_json(alignment_path, alignment_summary)
    camera_path = unit_dir / "cameras.json"
    write_json(
        camera_path,
        {
            "schema": "genrecon.gt-camera-split",
            "schema_version": 1,
            "pose_convention": "camera-to-world-opencv",
            "source_pose_convention": "Blender camera-to-world with local Y/Z sign conversion",
            "world_frame": "official normalized scan-render frame",
            "coordinate_units": "normalized-object",
            "conditioning": camera_records["conditioning"],
            "heldout": camera_records["heldout"],
        },
    )

    source_records = _write_source_records(
        [file_index_path, image_archive_path, scan_archive_path]
    )
    source_members = {
        "transforms": {
            "archive": image_record["openxlab_path"],
            "member": _normalized_tar_name(transforms_member),
            "size_bytes": len(transforms_payload),
            "sha256": _sha256_bytes(transforms_payload),
            "canonical_json_sha256": _canonical_json_sha256(transforms),
        },
        "scan_obj": {
            "archive": scan_record["openxlab_path"],
            "member": _normalized_tar_name(scan_member),
            "size_bytes": len(scan_payload),
            "sha256": _sha256_bytes(scan_payload),
        },
    }
    manifest = _base_manifest(
        unit_id=unit_id,
        dataset="omniobject3d",
        track=dataset["track"],
        gt_tier=dataset["gt_tier"],
        capture_kind=dataset["capture_kind"],
        license_status=dataset["license_status"],
    )
    manifest["physical_scene_group"] = f"omniobject3d:{object_id}"
    manifest.update(
        {
            "source": {
                "object_id": object_id,
                "category": category,
                "url": dataset["source_url"],
                "dataset_repo": OMNI_DATASET_REPO,
                "adapter_version": OMNI_ADAPTER_VERSION,
                "official_code_revision": OMNI_OFFICIAL_CODE_REVISION,
                "records": source_records,
                "archive_members": source_members,
            },
            "input": {
                "conditioning_views": conditioning_paths,
                "heldout_views": heldout_paths,
                "cameras": relpath(camera_path, manifest_path.parent),
                "pose_source": "official-100-view-Blender-camera-to-world",
                "split_policy": "deterministic-camera-center-FPS-8-conditioning-then-8-disjoint-heldout-from-100-official-views",
                "background_policy": "source RGB composited over white with the matching official depth-EXR foreground mask; source foreground pixels unchanged",
                "source_modality": "scan-derived-rendered-RGB",
                "auxiliary_sources_not_conditioning": [
                    "official depth EXRs used only for foreground masks and scan/render alignment audit; depth values are not exported",
                    "official normal render members checked for source completeness but not read by GenRecon",
                ],
            },
            "reference": {
                "kind": "mesh",
                "paths": [relpath(reference_path, manifest_path.parent)],
                "roi": "full-object",
                "scope": "normalized-object-global-reference",
                "coordinate_units": "normalized-object",
                "source_type": "professional-real-object-scan-normalized-to-official-render-frame",
                "normalization": normalization,
                "stats": {
                    "vertices": int(len(vertices)),
                    "faces": int(len(faces)),
                    "bounds": normalization["output_bounds"],
                },
                "scan_render_alignment": relpath(
                    alignment_path, manifest_path.parent
                ),
            },
            "evaluation": {
                "alignment": "provided-official-normalized-scan-render-frame",
                "pose_track": "GT-pose-scan-render",
                "primary_threshold_policy": "bbox-diagonal-normalized-only",
                "limitations": [
                    "Conditioning and heldout RGB are rendered from the same reference scan; neither split is geometry-independent.",
                    "This is the official 100-view normalized surface-reconstruction calibration domain, not the 24-view GET3D or real iPhone video domains.",
                    "Coordinates are normalized object units, not meters; absolute 2/5/10 cm scores are not valid for this unit.",
                    "The reference scan is evaluation-only and must not be read by GenRecon conditioning or alignment.",
                ],
            },
            "prediction_mesh": None,
            "blocker": None,
        }
    )
    return _write_manifest(output, manifest)


def prepare_omniobject3d(
    plan: dict[str, Any], output: Path, source_root: Path, *, force: bool
) -> list[dict[str, Any]]:
    dataset = plan["datasets"]["omniobject3d"]
    expected_unit_directories = {
        f"omniobject3d-{object_id}" for object_id in dataset["unit_ids"]
    }
    if force:
        for path in sorted((output / "units").glob("omniobject3d-*")):
            if path.is_dir() and path.name not in expected_unit_directories:
                shutil.rmtree(path)
    source = source_root / "omniobject3d"
    file_index_path = source / "metadata" / "openxlab_file_index.json"
    download_root = (
        source
        / "downloads"
        / "OpenXDLab___OmniObject3D-New"
    )
    if not file_index_path.is_file():
        blocker = (
            "OpenXLab metadata index is unavailable; authenticate locally and fetch "
            "the official file listing without committing AK/SK credentials."
        )
        return [
            _omni_blocked_manifest(
                dataset,
                output,
                object_id,
                status="blocked-auth",
                blocker=blocker,
            )
            for object_id in dataset["unit_ids"]
        ]
    file_index = load_json(file_index_path)
    if (
        file_index.get("schema") != "genrecon.omniobject3d-openxlab-file-index"
        or file_index.get("dataset_repo") != OMNI_DATASET_REPO
        or not isinstance(file_index.get("files"), list)
    ):
        raise CalibrationBuildError(
            f"Unexpected OmniObject3D OpenXLab index: {file_index_path}"
        )
    index_by_path = {item["path"]: item for item in file_index["files"]}
    rows = []
    for object_id in dataset["unit_ids"]:
        category = _omni_category(object_id)
        image_openxlab_path = f"/raw/blender_renders/{category}.tar.gz"
        scan_openxlab_path = f"/raw/raw_scans/{category}.tar.gz"
        image_path = download_root / image_openxlab_path.removeprefix("/")
        scan_path = download_root / scan_openxlab_path.removeprefix("/")
        missing = [
            str(path)
            for path in (image_path, scan_path)
            if not path.is_file()
        ]
        if missing:
            rows.append(
                _omni_blocked_manifest(
                    dataset,
                    output,
                    object_id,
                    status="missing-local-source",
                    blocker=f"Authorized OmniObject3D archives are not downloaded: {missing}",
                    source_records=_write_source_records([file_index_path]),
                )
            )
            continue
        image_record = _omni_archive_record(
            index_by_path, image_openxlab_path, image_path
        )
        scan_record = _omni_archive_record(
            index_by_path, scan_openxlab_path, scan_path
        )
        source_records = _write_source_records(
            [file_index_path, image_path, scan_path]
        )
        try:
            row = _prepare_omni_unit(
                dataset,
                output,
                object_id,
                image_record,
                scan_record,
                file_index_path,
                force=force,
            )
        except OmniAlignmentError as exc:
            row = _omni_blocked_manifest(
                dataset,
                output,
                object_id,
                status="alignment-failed",
                blocker=str(exc),
                source_records=source_records,
            )
        except OmniDependencyError:
            raise
        except (CalibrationBuildError, OSError, tarfile.TarError) as exc:
            row = _omni_blocked_manifest(
                dataset,
                output,
                object_id,
                status="source-incomplete",
                blocker=f"{type(exc).__name__}: {exc}",
                source_records=source_records,
            )
        rows.append(row)
    return rows


def _sha256_string(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _conditioning_manifest_contract_sha256(document: dict[str, Any]) -> str:
    canonical = {
        key: value
        for key, value in document.items()
        if key not in {"prediction_mesh", "prediction_provenance"}
    }
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _root_recorded_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _validate_representative_input_sources(
    document: dict[str, Any], manifest_path: Path, unit_id: str
) -> None:
    try:
        if (
            document.get("schema") != "genrecon.gt-representative-foundation-input"
            or document.get("schema_version") != 1
            or document.get("unit_id") != unit_id
            or document.get("track") != "GT-pose-foundation-pseudo-geometry"
        ):
            raise CalibrationBuildError("unexpected representative input schema")

        audit = document.get("source_audit", {})
        records = audit.get("source_files_read")
        expected_roles = [
            "unit-manifest",
            "conditioning-camera-metadata",
            *("conditioning-rgb" for _ in range(8)),
        ]
        if (
            not isinstance(records, list)
            or [record.get("role") for record in records] != expected_roles
            or audit.get("heldout_rgb_paths_read") != []
            or audit.get("depth_paths_read") != []
            or audit.get("reference_paths_read") != []
            or audit.get("conditioning_camera_records_used") != 8
            or audit.get("heldout_camera_records_used") != 0
        ):
            raise CalibrationBuildError("source audit is not conditioning-only 1+1+8")

        source_manifest = _root_recorded_path(document["source_manifest"])
        if _root_recorded_path(records[0]["path"]) != source_manifest:
            raise CalibrationBuildError("source manifest path differs from audit")
        source_document = load_json(source_manifest)
        if source_document.get("unit_id") != unit_id or source_document.get("status") != "prepared":
            raise CalibrationBuildError("source unit manifest is not the prepared unit")
        source_coordinate_units = source_document.get("reference", {}).get(
            "coordinate_units", "meters"
        )
        if document.get("coordinate_units", "meters") != source_coordinate_units:
            raise CalibrationBuildError(
                "representative input coordinate units differ from source unit"
            )
        conditioning_contract = _conditioning_manifest_contract_sha256(source_document)
        if (
            document.get("source_manifest_conditioning_contract_sha256")
            != conditioning_contract
            or records[0].get("conditioning_contract_sha256") != conditioning_contract
        ):
            raise CalibrationBuildError("conditioning manifest contract changed")

        source_root = source_manifest.parent
        expected_camera = (source_root / source_document["input"]["cameras"]).resolve()
        expected_rgb = [
            (source_root / value).resolve()
            for value in source_document["input"]["conditioning_views"]
        ]
        if len(expected_rgb) != 8 or len(set(expected_rgb)) != 8:
            raise CalibrationBuildError("conditioning split is not 8 unique RGBs")
        recorded_paths = [_root_recorded_path(record["path"]) for record in records]
        if recorded_paths[1:] != [expected_camera, *expected_rgb]:
            raise CalibrationBuildError("audited paths differ from conditioning split")
        for record, path in zip(records[1:], recorded_paths[1:]):
            if (
                not path.is_file()
                or path.stat().st_size != record.get("size_bytes_at_read")
                or sha256_file(path) != record.get("sha256_at_read")
            ):
                raise CalibrationBuildError(f"audited conditioning source changed: {path}")

        expected_assets = [
            "colmap_vggt/cameras.txt",
            "colmap_vggt/images.txt",
            "colmap_vggt/points3D.txt",
            *(f"rgb/{index:03d}.png" for index in range(8)),
        ]
        assets = document.get("genrecon_input_assets")
        if (
            not isinstance(assets, list)
            or [record.get("path") for record in assets] != expected_assets
        ):
            raise CalibrationBuildError("GenRecon input asset contract is missing or unordered")
        for record in assets:
            path = (manifest_path.parent / record["path"]).resolve()
            if (
                not path.is_file()
                or path.stat().st_size != record.get("size_bytes")
                or sha256_file(path) != record.get("sha256")
            ):
                raise CalibrationBuildError(f"GenRecon input asset changed: {path}")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CalibrationBuildError(f"invalid representative input sources: {exc}") from exc


def _representative_input_contract_sha256(document: dict[str, Any]) -> str:
    canonical = json.loads(json.dumps(document, allow_nan=False))
    canonical.pop("source_manifest_sha256_at_read", None)
    inference = canonical.get("inference", {})
    inference.pop("elapsed_seconds", None)
    inference.pop("peak_memory_mib", None)
    model = canonical.get("model", {})
    model.pop("checkpoint_path", None)
    for record in canonical.get("source_audit", {}).get("source_files_read", []):
        if record.get("role") == "unit-manifest":
            record.pop("size_bytes_at_read", None)
            record.pop("sha256_at_read", None)
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_normalized_prediction_camera_selection(
    package_source: dict[str, Any],
    reconstruction: Path | None,
    input_manifest: Path,
    input_document: dict[str, Any],
) -> None:
    if input_document.get("coordinate_units", "meters") != "normalized-object":
        return
    error = "normalized-object prediction camera-selection contract is invalid"
    try:
        if not isinstance(package_source, dict) or not isinstance(reconstruction, Path):
            raise TypeError("prediction source paths are missing")
        contract = input_document.get("genrecon_camera_selection", {})
        if not isinstance(contract, dict):
            raise TypeError("camera-selection contract is not an object")
        expected_area = float(contract.get("minimum_projected_chunk_area", -1.0))
        preflight_root = input_manifest.parent / "genrecon_preflight"
        preflight = load_json(preflight_root / "preflight.json")
        preflight_cameras = load_json(preflight_root / "cameras.json")
        reconstruction_args = load_json(reconstruction / "args.json")
        reconstruction_cameras = load_json(reconstruction / "cameras.json")
        chunk_records = preflight_cameras.get("chunks", [])
        area_summary = preflight.get("camera_selection", {})
        source_hashes = package_source.get("sha256", {})
        if not isinstance(source_hashes, dict):
            raise TypeError("prediction source hashes are not an object")
        areas = [
            float(item.get("cond2d_view", {}).get("projected_chunk_area", -1.0))
            for item in chunk_records
        ]
        valid = (
            contract.get("policy") == "frustum-and-minimum-projected-chunk-area"
            and contract.get("required_fallback_count") == 0
            and 0.0 <= expected_area <= 1.0
            and preflight.get("status") == "passed"
            and preflight.get("closest_camera_fallback_count") == 0
            and area_summary.get("policy")
            == "frustum-and-minimum-projected-chunk-area"
            and area_summary.get("minimum_projected_chunk_area") == expected_area
            and area_summary.get("required_fallback_count") == 0
            and len(chunk_records) == preflight.get("chunk_count")
            and len(chunk_records) > 0
            and all(
                item.get("cond2d_view", {}).get("selection_mode")
                == "visible-projected-area"
                and item.get("cond2d_view", {}).get("minimum_projected_chunk_area")
                == expected_area
                and area >= expected_area
                for item, area in zip(chunk_records, areas)
            )
            and area_summary.get("selected_projected_chunk_area_min") == min(areas)
            and area_summary.get("selected_projected_chunk_area_max") == max(areas)
            and area_summary.get("selected_chunks_meeting_area_gate") == len(areas)
            and reconstruction_args.get("min_projected_chunk_area") == expected_area
            and reconstruction_cameras == preflight_cameras
            and package_source.get("camera_selection") == contract
            and source_hashes.get("args") == sha256_file(reconstruction / "args.json")
            and source_hashes.get("cameras")
            == sha256_file(reconstruction / "cameras.json")
        )
    except (CalibrationBuildError, KeyError, OSError, TypeError, ValueError) as exc:
        raise CalibrationBuildError(error) from exc
    if not valid:
        raise CalibrationBuildError(error)


def validate_prediction_override(
    plan: dict[str, Any], unit_id: str
) -> dict[str, Any] | None:
    override = plan.get("prediction_overrides", {}).get(unit_id)
    if override is None:
        return None
    mesh = (ROOT / override["mesh"]).resolve()
    package_manifest = (ROOT / override["package_manifest"]).resolve()
    existing = [path.is_file() for path in (mesh, package_manifest)]
    if not any(existing):
        return None
    if not all(existing):
        raise CalibrationBuildError(
            f"Incomplete prediction override for {unit_id}: {mesh}, {package_manifest}"
        )
    package = load_json(package_manifest)
    expected_track = override["track"]
    alignment = package.get("alignment", {})
    matrix = np.asarray(alignment.get("work_to_official", []), dtype=np.float64)
    outputs = package.get("outputs", {})
    mesh_output = outputs.get("mesh", {})
    glb_output = outputs.get("glb", {})
    declared_mesh = (package_manifest.parent / mesh_output.get("path", "")).resolve()
    glb = (package_manifest.parent / glb_output.get("path", "")).resolve()
    source = package.get("source", {})
    source_hashes = source.get("sha256", {})
    reconstruction_value = source.get("reconstruction")
    input_manifest_value = source.get("input_manifest")
    reconstruction = (
        (ROOT / reconstruction_value).resolve()
        if isinstance(reconstruction_value, str)
        else None
    )
    input_manifest = (
        (ROOT / input_manifest_value).resolve()
        if isinstance(input_manifest_value, str)
        else None
    )
    input_document = (
        load_json(input_manifest)
        if input_manifest is not None and input_manifest.is_file()
        else None
    )
    input_matrix = np.asarray(
        (
            input_document.get("work_frame", {}).get("work_to_official", [])
            if input_document is not None
            else []
        ),
        dtype=np.float64,
    )
    source_mesh = reconstruction / "mesh.ply" if reconstruction is not None else None
    source_glb = reconstruction / "scene.glb" if reconstruction is not None else None
    input_contract = source.get("input_manifest_contract_sha256")
    if input_document is not None:
        _validate_representative_input_sources(input_document, input_manifest, unit_id)
        _validate_normalized_prediction_camera_selection(
            source, reconstruction, input_manifest, input_document
        )
    if (
        package.get("schema") != "genrecon.gt-representative-prediction-package"
        or package.get("schema_version") != 1
        or package.get("status") != "pass"
        or package.get("unit_id") != unit_id
        or package.get("track") != expected_track
        or package.get("coordinate_frame") != "declared calibration/reference frame"
        or input_document is None
        or package.get("coordinate_units", "meters")
        != input_document.get("coordinate_units", "meters")
        or matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
        or alignment.get("gt_geometry_icp_used") is not False
        or alignment.get("reference_geometry_used") is not False
        or alignment.get("heldout_views_used") is not False
        or alignment.get("method")
        != "conditioning-camera-only Sim(3), followed by exact work-to-official inverse"
        or declared_mesh != mesh
        or declared_mesh.parent != package_manifest.parent
        or glb.parent != package_manifest.parent
        or not glb.is_file()
        or mesh_output.get("size_bytes") != mesh.stat().st_size
        or mesh_output.get("sha256") != sha256_file(mesh)
        or glb_output.get("size_bytes") != glb.stat().st_size
        or glb_output.get("sha256") != sha256_file(glb)
        or source_mesh is None
        or source_glb is None
        or reconstruction != (package_manifest.parent.parent / "reconstruction").resolve()
        or not source_mesh.is_file()
        or not source_glb.is_file()
        or source_hashes.get("mesh") != sha256_file(source_mesh)
        or source_hashes.get("glb") != sha256_file(source_glb)
        or input_manifest is None
        or input_document is None
        or input_document.get("unit_id") != unit_id
        or input_matrix.shape != (4, 4)
        or not np.array_equal(matrix, input_matrix)
        or not _sha256_string(input_contract)
        or input_contract
        != _representative_input_contract_sha256(input_document)
    ):
        raise CalibrationBuildError(f"Invalid prediction override contract for {unit_id}")
    return {
        "mesh": mesh,
        "mesh_sha256": mesh_output["sha256"],
        "mesh_size_bytes": mesh_output["size_bytes"],
        "glb": glb,
        "glb_sha256": glb_output["sha256"],
        "glb_size_bytes": glb_output["size_bytes"],
        "input_manifest_contract_sha256": input_contract,
        "package_manifest": package_manifest,
        "package_manifest_sha256": sha256_file(package_manifest),
        "track": expected_track,
        "alignment": alignment.get("method"),
    }


def register_prediction_overrides(
    plan_path: Path, plan: dict[str, Any], output: Path
) -> dict[str, Any]:
    registry_path = output / "registry.json"
    if not registry_path.is_file():
        raise CalibrationBuildError(f"GT registry does not exist: {registry_path}")
    registry = load_json(registry_path)
    rows = registry.get("units", [])
    by_id = {row["unit_id"]: row for row in rows}
    unknown = sorted(set(plan.get("prediction_overrides", {})) - set(by_id))
    if unknown:
        raise CalibrationBuildError(f"Prediction overrides reference unknown units: {unknown}")
    registered = 0
    for unit_id in sorted(plan.get("prediction_overrides", {})):
        row = by_id[unit_id]
        manifest_path = output / row["manifest"]
        manifest = load_json(manifest_path)
        prediction = validate_prediction_override(plan, unit_id)
        if prediction is None:
            row["prediction_mesh"] = None
            manifest["prediction_mesh"] = None
            manifest.pop("prediction_provenance", None)
        else:
            prediction_value = relpath(prediction["mesh"], output)
            row["prediction_mesh"] = prediction_value
            manifest["prediction_mesh"] = prediction_value
            manifest["prediction_provenance"] = {
                "track": prediction["track"],
                "alignment": prediction["alignment"],
                "package_manifest": relpath(
                    prediction["package_manifest"], manifest_path.parent
                ),
                "package_manifest_sha256": prediction["package_manifest_sha256"],
                "mesh_sha256": prediction["mesh_sha256"],
                "mesh_size_bytes": prediction["mesh_size_bytes"],
                "glb": relpath(prediction["glb"], manifest_path.parent),
                "glb_sha256": prediction["glb_sha256"],
                "glb_size_bytes": prediction["glb_size_bytes"],
                "input_manifest_contract_sha256": prediction[
                    "input_manifest_contract_sha256"
                ],
                "reference_geometry_used_for_conditioning": False,
                "heldout_views_used_for_conditioning": False,
                "gt_geometry_icp_used": False,
            }
            registered += 1
        write_json(manifest_path, manifest)
    rebuilt = build_registry(plan_path, output, rows)
    write_summary(output, rebuilt)
    print(
        f"[gt-prediction-registration] registered={registered}/"
        f"{len(plan.get('prediction_overrides', {}))}"
    )
    return rebuilt


def build_registry(plan_path: Path, output: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    document = {
        "schema": REGISTRY_SCHEMA,
        "schema_version": 1,
        "plan": relpath(plan_path, output),
        "plan_sha256": sha256_file(plan_path),
        "summary": {
            "unit_count": len(rows),
            "statuses": dict(sorted(Counter(row["status"] for row in rows).items())),
            "datasets": dict(sorted(Counter(row["dataset"] for row in rows).items())),
            "tracks": dict(sorted(Counter(row["track"] for row in rows).items())),
            "gt_tiers": dict(sorted(Counter(row["gt_tier"] for row in rows).items())),
            "predictions_available": sum(bool(row.get("prediction_mesh")) for row in rows),
        },
        "units": sorted(rows, key=lambda row: row["unit_id"]),
    }
    write_json(output / "registry.json", document)
    return document


def write_summary(output: Path, registry: dict[str, Any]) -> None:
    fields = ["unit_id", "dataset", "track", "gt_tier", "status", "prediction_mesh", "blocker", "manifest"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in registry["units"]:
            writer.writerow({field: row.get(field) for field in fields})


def _strict_json_files(output: Path) -> tuple[int, list[str]]:
    count = 0
    errors = []
    for path in sorted(output.rglob("*.json")):
        if path.name in {"validation.json", "source_validation.json"}:
            continue
        try:
            load_json(path)
            count += 1
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return count, errors


def _validate_omni_prepared_package(
    manifest: dict[str, Any], manifest_path: Path, camera_split: dict[str, Any] | None
) -> None:
    unit_id = manifest["unit_id"]
    object_id = manifest.get("source", {}).get("object_id")
    if (
        manifest.get("dataset") != "omniobject3d"
        or manifest.get("status") != "prepared"
        or not isinstance(object_id, str)
        or unit_id != f"omniobject3d-{object_id}"
        or manifest.get("source", {}).get("adapter_version")
        != OMNI_ADAPTER_VERSION
        or manifest.get("source", {}).get("dataset_repo") != OMNI_DATASET_REPO
        or manifest.get("source", {}).get("official_code_revision")
        != OMNI_OFFICIAL_CODE_REVISION
    ):
        raise CalibrationBuildError("unexpected OmniObject3D package identity")

    unit_dir = manifest_path.parent
    input_document = manifest.get("input", {})
    conditioning = input_document.get("conditioning_views")
    heldout = input_document.get("heldout_views")
    if (
        not isinstance(conditioning, list)
        or not isinstance(heldout, list)
        or len(conditioning) != 8
        or len(heldout) != 8
        or len(set(conditioning + heldout)) != 16
        or input_document.get("pose_source")
        != "official-100-view-Blender-camera-to-world"
        or input_document.get("source_modality") != "scan-derived-rendered-RGB"
        or input_document.get("conditioning_depths") not in (None, [])
        or input_document.get("heldout_depths") not in (None, [])
        or list(unit_dir.rglob("*.exr"))
        or (unit_dir / "depth").exists()
    ):
        raise CalibrationBuildError("OmniObject3D input is not RGB-only 8+8")

    if (
        not isinstance(camera_split, dict)
        or camera_split.get("schema") != "genrecon.gt-camera-split"
        or camera_split.get("pose_convention") != "camera-to-world-opencv"
        or camera_split.get("coordinate_units") != "normalized-object"
    ):
        raise CalibrationBuildError("invalid OmniObject3D camera split schema")

    alignment_value = manifest.get("reference", {}).get("scan_render_alignment")
    if not isinstance(alignment_value, str):
        raise CalibrationBuildError("missing OmniObject3D alignment artifact")
    alignment_path = (unit_dir / alignment_value).resolve()
    alignment = load_json(alignment_path)
    thresholds = alignment.get("thresholds", {})
    expected_thresholds = {
        "maximum_edge_error_px": OMNI_MAX_SILHOUETTE_EDGE_ERROR_PX,
        "minimum_bbox_iou": OMNI_MIN_SILHOUETTE_BBOX_IOU,
        "minimum_raycast_mask_iou": OMNI_MIN_RAYCAST_MASK_IOU,
        "maximum_camera_orthogonality_for_repair": OMNI_MAX_CAMERA_ORTHOGONALITY_FOR_REPAIR,
        "maximum_camera_rotation_repair": OMNI_MAX_CAMERA_ROTATION_REPAIR,
        "published_render_target_max_abs_coordinate": OMNI_RENDER_TARGET_MAX_ABS,
        "recorded_scale_max_target_ratio": OMNI_RECORDED_SCALE_MAX_TARGET_RATIO,
    }
    views = alignment.get("views")
    if (
        alignment.get("schema")
        != "genrecon.omniobject3d-scan-render-alignment"
        or alignment.get("schema_version") != 3
        or alignment.get("unit_id") != unit_id
        or alignment.get("result") != "pass"
        or thresholds != expected_thresholds
        or not isinstance(views, list)
        or len(views) != 100
        or [view.get("source_index") for view in views] != list(range(100))
    ):
        raise CalibrationBuildError("invalid OmniObject3D 100-view alignment schema")

    fast_count = 0
    fallback_count = 0
    repair_count = 0
    for view in views:
        source_index = view["source_index"]
        fast = view.get("fast_projection")
        fallback = view.get("raycast_fallback")
        pose_audit = view.get("camera_pose_audit")
        if (
            view.get("pass") is not True
            or not isinstance(fast, dict)
            or not isinstance(pose_audit, dict)
            or view.get("source_image") != f"r_{source_index}.png"
            or view.get("depth_member")
            != f"{object_id}/render/depths/r_{source_index}_depth.exr"
            or not _sha256_string(view.get("depth_sha256"))
            or not isinstance(view.get("depth_size_bytes"), int)
            or view["depth_size_bytes"] <= 0
            or view.get("depth_mask", {}).get("background_sentinel")
            != OMNI_DEPTH_BACKGROUND_SENTINEL
            or view.get("depth_mask", {}).get("packed_channel") != "RGB"
            or view.get("depth_mask", {}).get("dtype") != "float32"
        ):
            raise CalibrationBuildError("invalid OmniObject3D view provenance")
        if fast.get("pass") is True:
            fast_count += 1
            if (
                fallback is not None
                or view.get("acceptance_method") != "fast-projected-vertices"
                or fast.get("max_edge_error_px")
                > OMNI_MAX_SILHOUETTE_EDGE_ERROR_PX
                or fast.get("bbox_iou") < OMNI_MIN_SILHOUETTE_BBOX_IOU
            ):
                raise CalibrationBuildError("invalid OmniObject3D fast acceptance")
        else:
            fallback_count += 1
            if (
                not isinstance(fallback, dict)
                or fallback.get("pass") is not True
                or view.get("acceptance_method")
                != "visibility-aware-raycast-fallback"
                or fallback.get("max_edge_error_px")
                > OMNI_MAX_SILHOUETTE_EDGE_ERROR_PX
                or fallback.get("bbox_iou") < OMNI_MIN_SILHOUETTE_BBOX_IOU
                or fallback.get("mask_iou") < OMNI_MIN_RAYCAST_MASK_IOU
            ):
                raise CalibrationBuildError("invalid OmniObject3D raycast acceptance")
        if pose_audit.get("method") == "nearest-SO3-SVD":
            repair_count += 1
            if (
                pose_audit.get("raw_orthogonality_max_abs") <= 1e-5
                or pose_audit.get("raw_orthogonality_max_abs")
                > OMNI_MAX_CAMERA_ORTHOGONALITY_FOR_REPAIR
                or pose_audit.get("max_abs_rotation_correction")
                > OMNI_MAX_CAMERA_ROTATION_REPAIR
            ):
                raise CalibrationBuildError("invalid OmniObject3D camera repair")
        elif (
            pose_audit.get("method") != "unchanged-within-tolerance"
            or pose_audit.get("max_abs_rotation_correction") != 0.0
        ):
            raise CalibrationBuildError("invalid OmniObject3D camera pose audit")

    if (
        alignment.get("view_count") != 100
        or alignment.get("fast_projection_pass_count") != fast_count
        or alignment.get("raycast_fallback_count") != fallback_count
        or alignment.get("bounded_camera_repair_count") != repair_count
        or not np.isclose(
            alignment.get("maximum_edge_error_px"),
            max(view["max_edge_error_px"] for view in views),
            rtol=0.0,
            atol=1e-12,
        )
        or not np.isclose(
            alignment.get("minimum_bbox_iou"),
            min(view["bbox_iou"] for view in views),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise CalibrationBuildError("OmniObject3D alignment summary is stale")

    normalization = manifest.get("reference", {}).get("normalization", {})
    scale_selection = normalization.get("scale_selection", {})
    source_radius = float(normalization.get("source_max_abs_coordinate", np.nan))
    selected_scale = float(normalization.get("selected_uniform_scale", np.nan))
    recorded_scale = float(scale_selection.get("recorded_scale", np.nan))
    recorded_output_radius = source_radius * recorded_scale
    expected_recovery = recorded_output_radius > (
        OMNI_RENDER_TARGET_MAX_ABS * OMNI_RECORDED_SCALE_MAX_TARGET_RATIO
    )
    expected_scale = (
        OMNI_RENDER_TARGET_MAX_ABS / source_radius
        if expected_recovery
        else recorded_scale
    )
    expected_scale_method = (
        "recomputed-published-renderer-scale-after-recorded-scale-sanity-failure"
        if expected_recovery
        else "recorded-full-render-scale"
    )
    if (
        normalization.get("method")
        != "audited-full-render-scale-and-Blender-OBJ-import-axis"
        or normalization.get("coordinate_units") != "normalized-object"
        or not np.array_equal(
            np.asarray(normalization.get("right_multiply_axis")),
            OMNI_SCAN_TO_RENDER_AXIS,
        )
        or not np.isfinite([source_radius, selected_scale, recorded_scale]).all()
        or source_radius <= 0.0
        or selected_scale <= 0.0
        or scale_selection.get("method") != expected_scale_method
        or scale_selection.get("recorded_scale_used") is expected_recovery
        or not np.isclose(selected_scale, expected_scale, rtol=0.0, atol=1e-15)
        or not np.isclose(
            scale_selection.get("recorded_output_max_abs_coordinate"),
            recorded_output_radius,
            rtol=0.0,
            atol=1e-9,
        )
        or not np.isclose(
            normalization.get("output_max_abs_coordinate"),
            source_radius * selected_scale,
            rtol=0.0,
            atol=1e-9,
        )
    ):
        raise CalibrationBuildError("invalid OmniObject3D scale selection")

    by_index = {view["source_index"]: view for view in views}
    selected_indices = []
    for role, expected_paths in (("conditioning", conditioning), ("heldout", heldout)):
        records = camera_split.get(role)
        if not isinstance(records, list) or len(records) != 8:
            raise CalibrationBuildError("invalid OmniObject3D camera role count")
        for order, (record, expected_path) in enumerate(zip(records, expected_paths)):
            source_index = record.get("source_index")
            expected_rgb = f"rgb/{role}/{order:03d}.png"
            selected_indices.append(source_index)
            pose = np.asarray(record.get("camera_to_world"), dtype=np.float64)
            intrinsic = np.asarray(record.get("intrinsics"), dtype=np.float64)
            if (
                record.get("order") != order
                or expected_path != expected_rgb
                or record.get("rgb") != expected_rgb
                or not isinstance(source_index, int)
                or source_index not in by_index
                or record.get("source_rgb_member")
                != f"{object_id}/render/images/r_{source_index}.png"
                or record.get("source_depth_member")
                != f"{object_id}/render/depths/r_{source_index}_depth.exr"
                or not _sha256_string(record.get("source_rgb_sha256"))
                or not _sha256_string(record.get("source_depth_sha256"))
                or record.get("source_depth_sha256")
                != by_index[source_index]["depth_sha256"]
                or record.get("scan_projection_alignment") != by_index[source_index]
                or record.get("camera_pose_audit")
                != by_index[source_index]["camera_pose_audit"]
                or pose.shape != (4, 4)
                or intrinsic.shape != (3, 3)
                or not np.isfinite(pose).all()
                or not np.isfinite(intrinsic).all()
                or not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9)
                or not np.allclose(
                    pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5
                )
                or np.linalg.det(pose[:3, :3]) <= 0.0
                or record.get("width") != OMNI_RENDER_RESOLUTION
                or record.get("height") != OMNI_RENDER_RESOLUTION
            ):
                raise CalibrationBuildError("invalid OmniObject3D selected camera")
    if len(set(selected_indices)) != 16:
        raise CalibrationBuildError("OmniObject3D selected camera indices overlap")


def validate(output: Path, plan: dict[str, Any]) -> dict[str, Any]:
    registry = load_json(output / "registry.json")
    expected = sum(len(dataset["unit_ids"]) for dataset in plan["datasets"].values())
    errors = []
    if plan.get("schema") != PLAN_SCHEMA:
        errors.append("plan schema mismatch")
    if plan.get("expected_unit_count", expected) != expected:
        errors.append(
            f"plan expected_unit_count {plan.get('expected_unit_count')} != dataset total {expected}"
        )
    if registry.get("schema") != REGISTRY_SCHEMA:
        errors.append("registry schema mismatch")
    registry_plan = (output / registry.get("plan", "")).resolve()
    if not registry_plan.is_file() or sha256_file(registry_plan) != registry.get("plan_sha256"):
        errors.append("registry plan path/hash mismatch")
    if len(registry.get("units", [])) != expected:
        errors.append(f"registry unit count {len(registry.get('units', []))} != {expected}")
    units = registry.get("units", [])
    expected_summary = {
        "unit_count": len(units),
        "statuses": dict(sorted(Counter(row["status"] for row in units).items())),
        "datasets": dict(sorted(Counter(row["dataset"] for row in units).items())),
        "tracks": dict(sorted(Counter(row["track"] for row in units).items())),
        "gt_tiers": dict(sorted(Counter(row["gt_tier"] for row in units).items())),
        "predictions_available": sum(bool(row.get("prediction_mesh")) for row in units),
    }
    if registry.get("summary") != expected_summary:
        errors.append("registry summary does not match unit rows")
    expected_manifests = {row["manifest"] for row in units}
    actual_manifests = {
        relpath(path, output) for path in (output / "units").glob("*/manifest.json")
    }
    if actual_manifests != expected_manifests:
        errors.append("unit manifest set does not match registry rows")
    seen = set()
    reference_files = 0
    reference_vertices = 0
    reference_faces = 0
    reference_bytes = 0
    source_records = 0
    source_hash_bytes = 0
    conditioning_views = 0
    heldout_views = 0
    decoded_images = 0
    decoded_depths = 0
    prediction_files = 0
    prediction_bytes = 0
    omni_prepared_packages = 0
    for row in registry.get("units", []):
        if row["unit_id"] in seen:
            errors.append(f"duplicate unit {row['unit_id']}")
        seen.add(row["unit_id"])
        manifest_path = output / row["manifest"]
        if not manifest_path.is_file():
            errors.append(f"missing manifest {manifest_path}")
            continue
        manifest = load_json(manifest_path)
        if row["unit_id"] in plan.get("prediction_overrides", {}):
            try:
                expected_prediction = validate_prediction_override(plan, row["unit_id"])
                expected_value = (
                    relpath(expected_prediction["mesh"], output)
                    if expected_prediction is not None
                    else None
                )
                if row.get("prediction_mesh") != expected_value:
                    raise CalibrationBuildError("registry prediction override is stale")
                if manifest.get("prediction_mesh") != expected_value:
                    raise CalibrationBuildError("unit manifest prediction override is stale")
                provenance = manifest.get("prediction_provenance")
                if expected_prediction is None:
                    if provenance is not None:
                        raise CalibrationBuildError(
                            "missing prediction retains prediction provenance"
                        )
                elif (
                    not isinstance(provenance, dict)
                    or provenance.get("track") != expected_prediction["track"]
                    or provenance.get("package_manifest_sha256")
                    != expected_prediction["package_manifest_sha256"]
                    or provenance.get("mesh_sha256")
                    != expected_prediction["mesh_sha256"]
                    or provenance.get("mesh_size_bytes")
                    != expected_prediction["mesh_size_bytes"]
                    or provenance.get("glb")
                    != relpath(expected_prediction["glb"], manifest_path.parent)
                    or provenance.get("glb_sha256")
                    != expected_prediction["glb_sha256"]
                    or provenance.get("glb_size_bytes")
                    != expected_prediction["glb_size_bytes"]
                    or provenance.get("input_manifest_contract_sha256")
                    != expected_prediction["input_manifest_contract_sha256"]
                    or provenance.get("reference_geometry_used_for_conditioning")
                    is not False
                    or provenance.get("heldout_views_used_for_conditioning") is not False
                    or provenance.get("gt_geometry_icp_used") is not False
                ):
                    raise CalibrationBuildError("prediction provenance contract mismatch")
            except Exception as exc:
                errors.append(
                    f"invalid prediction override for {row['unit_id']}: {exc}"
                )
        if manifest.get("schema") != SCHEMA:
            errors.append(f"manifest schema mismatch for {row['unit_id']}")
        for key in ("unit_id", "dataset", "track", "gt_tier", "physical_scene_group"):
            if manifest.get(key) != row.get(key):
                errors.append(f"manifest/registry {key} mismatch for {row['unit_id']}")
        if manifest.get("status") != row["status"]:
            errors.append(f"status mismatch for {row['unit_id']}")
        if row.get("prediction_mesh"):
            prediction = (output / row["prediction_mesh"]).resolve()
            if not prediction.is_file() or prediction.stat().st_size == 0:
                errors.append(f"missing prediction mesh for {row['unit_id']}: {prediction}")
            else:
                prediction_files += 1
                prediction_bytes += prediction.stat().st_size
        if manifest["status"] == "prepared":
            conditioning_views += len(manifest["input"]["conditioning_views"])
            heldout_views += len(manifest["input"]["heldout_views"])
            if not manifest["input"]["conditioning_views"]:
                errors.append(f"prepared unit has no conditioning views: {row['unit_id']}")
            if set(manifest["input"]["conditioning_views"]) & set(
                manifest["input"]["heldout_views"]
            ):
                errors.append(f"conditioning/heldout overlap for {row['unit_id']}")
            input_root_value = manifest["input"].get("root") or manifest["input"].get("images_root")
            input_root = (
                (manifest_path.parent / input_root_value).resolve()
                if input_root_value
                else manifest_path.parent
            )
            for role in ("conditioning_views", "heldout_views"):
                for value in manifest["input"][role]:
                    path = Path(value)
                    if not path.is_absolute():
                        path = (input_root / path).resolve()
                    if not path.is_file() or path.stat().st_size == 0:
                        errors.append(f"missing {role} image for {row['unit_id']}: {path}")
                        continue
                    try:
                        with Image.open(path) as image:
                            width, height = image.size
                            image.verify()
                        if width <= 0 or height <= 0:
                            raise CalibrationBuildError("non-positive image dimensions")
                        decoded_images += 1
                    except Exception as exc:
                        errors.append(f"invalid {role} image for {row['unit_id']}: {path}: {exc}")
            for role in ("conditioning_depths", "heldout_depths"):
                for value in manifest["input"].get(role, []):
                    path = Path(value)
                    if not path.is_absolute():
                        path = (input_root / path).resolve()
                    try:
                        with Image.open(path) as image:
                            width, height = image.size
                            image.verify()
                        if width <= 0 or height <= 0:
                            raise CalibrationBuildError("non-positive depth dimensions")
                        decoded_depths += 1
                    except Exception as exc:
                        errors.append(f"invalid {role} image for {row['unit_id']}: {path}: {exc}")
            camera_value = manifest["input"].get("cameras")
            camera_split = None
            if camera_value:
                camera_path = (manifest_path.parent / camera_value).resolve()
                try:
                    camera_split = load_json(camera_path)
                    if len(camera_split.get("conditioning", [])) != len(
                        manifest["input"]["conditioning_views"]
                    ) or len(camera_split.get("heldout", [])) != len(
                        manifest["input"]["heldout_views"]
                    ):
                        raise CalibrationBuildError("camera split count does not match input views")
                except Exception as exc:
                    errors.append(f"invalid camera split for {row['unit_id']}: {camera_path}: {exc}")
            if manifest["dataset"] == "omniobject3d":
                try:
                    _validate_omni_prepared_package(
                        manifest, manifest_path, camera_split
                    )
                    omni_prepared_packages += 1
                except Exception as exc:
                    errors.append(
                        f"invalid OmniObject3D package contract for "
                        f"{row['unit_id']}: {exc}"
                    )
            if manifest["dataset"] == "tanks-and-temples-training":
                try:
                    if manifest["reference"].get("scope") != "official-crop-global-reference":
                        raise CalibrationBuildError("unexpected T&T reference scope")
                    if manifest["evaluation"].get("pose_track") != "GT-pose-official-alignment":
                        raise CalibrationBuildError("unexpected T&T pose track")
                    if camera_split is None or camera_split.get("world_frame") != "official laser GT frame":
                        raise CalibrationBuildError("T&T cameras are not in the laser GT frame")
                    all_cameras = camera_split["conditioning"] + camera_split["heldout"]
                    source_indices = [item["source_index"] for item in all_cameras]
                    if len(source_indices) != 16 or len(set(source_indices)) != 16:
                        raise CalibrationBuildError("T&T camera split source indices overlap")
                    for camera in all_cameras:
                        pose = np.asarray(camera["camera_to_world"], dtype=np.float64)
                        intrinsic = np.asarray(camera["intrinsics"], dtype=np.float64)
                        if (
                            pose.shape != (4, 4)
                            or intrinsic.shape != (3, 3)
                            or not np.isfinite(pose).all()
                            or not np.isfinite(intrinsic).all()
                            or not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9)
                            or intrinsic[0, 0] <= 0.0
                            or intrinsic[1, 1] <= 0.0
                        ):
                            raise CalibrationBuildError("invalid T&T camera matrix")
                    generation = manifest["reference"].get("generation", {})
                    if generation.get("stats", {}).get("exported_points") != 10_321_864:
                        raise CalibrationBuildError("unexpected T&T reference point count")
                    calibration_path = (
                        camera_path.parent / camera_split["intrinsics_artifact"]
                    ).resolve()
                    calibration_document = load_json(calibration_path)
                    if (
                        calibration_document.get("schema_version") != 2
                        or calibration_document.get("result") != "pass"
                        or calibration_document.get("protocol", {}).get("revision")
                        != "v2-single-thread-geometry"
                    ):
                        raise CalibrationBuildError("invalid T&T calibration artifact")
                except Exception as exc:
                    errors.append(f"invalid T&T package contract for {row['unit_id']}: {exc}")
            for value in manifest["reference"]["paths"]:
                path = (manifest_path.parent / value).resolve()
                if not path.is_file() or path.stat().st_size == 0:
                    errors.append(f"missing reference for {row['unit_id']}: {path}")
                    continue
                reference_files += 1
                reference_bytes += path.stat().st_size
                try:
                    loaded = trimesh.load(str(path), process=False)
                    if not isinstance(loaded, (trimesh.Trimesh, trimesh.PointCloud)):
                        raise CalibrationBuildError(
                            f"unsupported geometry payload {type(loaded).__name__}"
                        )
                    vertices = np.asarray(loaded.vertices)
                    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
                        raise CalibrationBuildError("empty or invalid reference vertices")
                    if not np.isfinite(vertices).all():
                        raise CalibrationBuildError("non-finite reference vertices")
                    reference_vertices += len(vertices)
                    if isinstance(loaded, trimesh.Trimesh):
                        faces = np.asarray(loaded.faces)
                        if manifest["reference"]["kind"] == "mesh" and not len(faces):
                            raise CalibrationBuildError("mesh reference has no faces")
                        reference_faces += len(faces)
                    elif manifest["reference"]["kind"] == "mesh":
                        raise CalibrationBuildError("mesh reference loaded as point cloud")
                    del loaded, vertices
                except Exception as exc:
                    errors.append(f"invalid reference for {row['unit_id']}: {path}: {exc}")
        elif manifest["status"] == "blocked-auth" and manifest["dataset"] != "omniobject3d":
            errors.append(f"unexpected auth blocker for {row['unit_id']}")
        for record in manifest.get("source", {}).get("records", []):
            path = Path(record["path"])
            if not path.is_file() or path.stat().st_size != record["size_bytes"]:
                errors.append(f"source record size mismatch for {row['unit_id']}: {path}")
                continue
            source_records += 1
            source_hash_bytes += path.stat().st_size
            if sha256_file(path) != record["sha256"]:
                errors.append(f"source record hash mismatch for {row['unit_id']}: {path}")
    strict_json, json_errors = _strict_json_files(output)
    errors.extend(json_errors)
    counts = {
        "units": len(registry.get("units", [])),
        "statuses": registry.get("summary", {}).get("statuses", {}),
        "reference_files": reference_files,
        "reference_vertices": reference_vertices,
        "reference_faces": reference_faces,
        "reference_bytes": reference_bytes,
        "source_records": source_records,
        "source_hash_bytes": source_hash_bytes,
        "conditioning_views": conditioning_views,
        "heldout_views": heldout_views,
        "decoded_images": decoded_images,
        "decoded_depths": decoded_depths,
        "prediction_files": prediction_files,
        "prediction_bytes": prediction_bytes,
        "omni_prepared_packages": omni_prepared_packages,
        "strict_json_files": strict_json,
    }
    result = {
        "schema": "genrecon.gt-calibration-validation",
        "schema_version": 1,
        "result": "pass" if not errors else "fail",
        "counts": counts,
        "declared_blockers": [
            row for row in registry.get("units", []) if row["status"] != "prepared"
        ],
        "errors": errors,
    }
    write_json(output / "validation.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "register-predictions", "summarize", "validate", "all"),
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plan_path = args.plan.resolve()
    output = args.output.resolve()
    source_root = output / "sources"
    plan = load_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise SystemExit(f"Unexpected calibration plan schema in {plan_path}")
    selected = set(args.dataset or plan["datasets"])
    unknown = selected - set(plan["datasets"])
    if unknown:
        raise SystemExit(f"Unknown datasets: {sorted(unknown)}")
    output.mkdir(parents=True, exist_ok=True)
    if args.stage in {"prepare", "all"}:
        rows = []
        if "da3-scannetpp" in selected:
            rows.extend(prepare_scannetpp(plan, output))
        if "eth3d-indoor-training" in selected:
            rows.extend(prepare_eth3d(plan, output, source_root))
        if "seven-scenes" in selected:
            rows.extend(prepare_seven_scenes(plan, output, source_root, force=args.force))
        if "redwood-reconstruction" in selected:
            rows.extend(prepare_redwood(plan, output, source_root, force=args.force))
        if "dtu-mvs" in selected:
            rows.extend(prepare_dtu(plan, output, source_root, force=args.force))
        if "tanks-and-temples-training" in selected:
            rows.extend(
                prepare_tanks_and_temples(
                    plan, output, source_root, force=args.force
                )
            )
        if "omniobject3d" in selected:
            rows.extend(
                prepare_omniobject3d(
                    plan, output, source_root, force=args.force
                )
            )
        if selected != set(plan["datasets"]):
            existing = load_json(output / "registry.json")["units"] if (output / "registry.json").is_file() else []
            replaced = {row["dataset"] for row in rows}
            rows.extend(row for row in existing if row["dataset"] not in replaced)
        registry = build_registry(plan_path, output, rows)
        write_summary(output, registry)
        register_prediction_overrides(plan_path, plan, output)
    if args.stage == "register-predictions":
        register_prediction_overrides(plan_path, plan, output)
    if args.stage == "summarize":
        registry = load_json(output / "registry.json")
        write_summary(output, registry)
    if args.stage in {"validate", "all"}:
        result = validate(output, plan)
        print(f"[gt-calibration-validation] {result['result']} {result['counts']}")
        if result["errors"]:
            for error in result["errors"][:20]:
                print(f"  - {error}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
