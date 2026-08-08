#!/usr/bin/env python3
"""Prepare DA3-BENCH ScanNet++ scenes for GenRecon.

The Hugging Face ``depth-anything/DA3-BENCH`` archive stores a merged DSLR /
iPhone COLMAP model in binary form::

    <scene>/merge_dslr_iphone/
      colmap/sparse_render_rgb/{cameras,images,points3D}.bin
      images/iphone/...
      render_depth/...
    <scene>/scans/mesh_aligned_0.05.ply

GenRecon's ``Scannet_iphone`` mode expects a text COLMAP model and a flat RGB
directory under ``<scene>/iphone``.  This tool converts the binary model using
only the Python standard library, selects a deterministic set of registered
iPhone views, and creates relative symlinks for the RGB/depth/mesh assets.

The sparse point cloud is deliberately kept in full for scene chunking.  Its
tracks are pruned to the selected images and checked bidirectionally against
the selected images' POINTS2D records, so the emitted text model remains a
consistent COLMAP sub-model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable, Iterator, Sequence


class AdapterError(RuntimeError):
    """Raised when a DA3 scene is incomplete or its COLMAP model is invalid."""


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model_id: int
    model_name: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class ImageHeader:
    image_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    name: str
    num_points2d: int


@dataclass(frozen=True)
class RegisteredImage:
    image_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    name: str
    xys: tuple[tuple[float, float], ...]
    point3d_ids: tuple[int, ...]


@dataclass(frozen=True)
class Point3D:
    point3d_id: int
    xyz: tuple[float, float, float]
    rgb: tuple[int, int, int]
    error: float
    track: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SceneLayout:
    scene_id: str
    scene_root: Path
    model_dir: Path
    image_root: Path
    depth_root: Path
    mesh_path: Path


# The model identifiers and parameter counts are part of COLMAP's binary model
# format.  These are the models supported by the DA3 copy of COLMAP's official
# read_write_model.py at the time DA3-BENCH was published.
_CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

# inference/get_images.py can turn only these text camera models into the
# pinhole images consumed by GenRecon.  DA3-BENCH currently uses OPENCV.
_GENRECON_CAMERA_MODELS = {
    "SIMPLE_PINHOLE",
    "PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "OPENCV",
}

_MAX_RECORDS = 100_000_000
_MAX_CAMERAS = 1_000_000
_MAX_REGISTERED_IMAGES = 10_000_000
_MAX_OBSERVATIONS_PER_IMAGE = 10_000_000
_MAX_TRACK_LENGTH = 10_000_000
_BINARY_IMAGE_HEADER = struct.Struct("<idddddddi")
_BINARY_POINT_HEADER = struct.Struct("<QdddBBBd")
_BINARY_CAMERA_HEADER = struct.Struct("<iiQQ")
_BINARY_POINT2D = struct.Struct("<ddq")
_BINARY_TRACK = struct.Struct("<ii")
_UINT64 = struct.Struct("<Q")


def _format_float(value: float) -> str:
    """Round-trip-safe, deterministic text representation for COLMAP doubles."""

    return format(value, ".17g")


def _read_exact(handle: BinaryIO, size: int, *, context: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise AdapterError(
            f"Truncated COLMAP binary while reading {context}: "
            f"expected {size} bytes, got {len(data)}"
        )
    return data


def _read_count(
    handle: BinaryIO, *, context: str, limit: int = _MAX_RECORDS
) -> int:
    (count,) = _UINT64.unpack(_read_exact(handle, _UINT64.size, context=context))
    if count > limit:
        raise AdapterError(f"Implausible {context} count {count} (limit {limit})")
    return count


def _read_c_string(handle: BinaryIO, *, context: str) -> str:
    name = bytearray()
    while True:
        char = _read_exact(handle, 1, context=context)
        if char == b"\x00":
            break
        name.extend(char)
        if len(name) > 1_048_576:
            raise AdapterError(f"Unterminated or implausibly long {context}")
    try:
        decoded = name.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AdapterError(f"Invalid UTF-8 in {context}: {exc}") from exc
    if not decoded:
        raise AdapterError(f"Empty {context}")
    return decoded


def _assert_no_trailing_bytes(handle: BinaryIO, path: Path) -> None:
    position = handle.tell()
    size = os.fstat(handle.fileno()).st_size
    if position < size:
        raise AdapterError(
            f"Unexpected trailing bytes in COLMAP binary {path}: "
            f"parser stopped at {position} of {size} bytes"
        )
    if position > size:
        raise AdapterError(
            f"Truncated COLMAP binary {path}: parser advanced to {position}, "
            f"but file is only {size} bytes"
        )


def read_cameras_binary(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    try:
        with path.open("rb") as handle:
            count = _read_count(
                handle, context=f"camera count in {path}", limit=_MAX_CAMERAS
            )
            for index in range(count):
                camera_id, model_id, width, height = _BINARY_CAMERA_HEADER.unpack(
                    _read_exact(
                        handle,
                        _BINARY_CAMERA_HEADER.size,
                        context=f"camera {index} header in {path}",
                    )
                )
                if model_id not in _CAMERA_MODELS:
                    raise AdapterError(
                        f"Unknown COLMAP camera model id {model_id} for camera "
                        f"{camera_id} in {path}"
                    )
                model_name, num_params = _CAMERA_MODELS[model_id]
                params = struct.unpack(
                    "<" + "d" * num_params,
                    _read_exact(
                        handle,
                        8 * num_params,
                        context=f"camera {camera_id} parameters in {path}",
                    ),
                )
                if camera_id in cameras:
                    raise AdapterError(f"Duplicate camera id {camera_id} in {path}")
                if camera_id <= 0:
                    raise AdapterError(f"Invalid non-positive camera id {camera_id} in {path}")
                if width <= 0 or height <= 0:
                    raise AdapterError(
                        f"Camera {camera_id} has invalid dimensions {width}x{height}"
                    )
                if not all(math.isfinite(value) for value in params):
                    raise AdapterError(f"Camera {camera_id} contains non-finite parameters")
                if not params or params[0] <= 0 or (
                    model_name in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}
                    and params[1] <= 0
                ):
                    raise AdapterError(f"Camera {camera_id} has a non-positive focal length")
                cameras[camera_id] = Camera(
                    camera_id=camera_id,
                    model_id=model_id,
                    model_name=model_name,
                    width=width,
                    height=height,
                    params=tuple(params),
                )
            _assert_no_trailing_bytes(handle, path)
    except OSError as exc:
        raise AdapterError(f"Cannot read {path}: {exc}") from exc
    if not cameras:
        raise AdapterError(f"COLMAP model has no cameras: {path}")
    return cameras


def _read_image_record(
    handle: BinaryIO,
    *,
    path: Path,
    index: int,
    with_observations: bool,
) -> ImageHeader | RegisteredImage:
    values = _BINARY_IMAGE_HEADER.unpack(
        _read_exact(
            handle,
            _BINARY_IMAGE_HEADER.size,
            context=f"image {index} header in {path}",
        )
    )
    image_id = int(values[0])
    qvec = (float(values[1]), float(values[2]), float(values[3]), float(values[4]))
    tvec = (float(values[5]), float(values[6]), float(values[7]))
    camera_id = int(values[8])
    name = _read_c_string(handle, context=f"image {image_id} name in {path}")
    num_points2d = _read_count(
        handle,
        context=f"POINTS2D count for image {image_id} in {path}",
        limit=_MAX_OBSERVATIONS_PER_IMAGE,
    )
    observations_size = _BINARY_POINT2D.size * num_points2d
    if with_observations:
        raw = _read_exact(
            handle,
            observations_size,
            context=f"POINTS2D for image {image_id} in {path}",
        )
        triples = tuple(_BINARY_POINT2D.iter_unpack(raw))
        return RegisteredImage(
            image_id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=name,
            xys=tuple((float(x), float(y)) for x, y, _ in triples),
            point3d_ids=tuple(int(point3d_id) for _, _, point3d_id in triples),
        )
    handle.seek(observations_size, os.SEEK_CUR)
    return ImageHeader(
        image_id=image_id,
        qvec=qvec,
        tvec=tvec,
        camera_id=camera_id,
        name=name,
        num_points2d=num_points2d,
    )


def read_image_headers_binary(path: Path) -> list[ImageHeader]:
    images: list[ImageHeader] = []
    ids: set[int] = set()
    names: set[str] = set()
    try:
        with path.open("rb") as handle:
            count = _read_count(
                handle,
                context=f"registered image count in {path}",
                limit=_MAX_REGISTERED_IMAGES,
            )
            for index in range(count):
                record = _read_image_record(
                    handle, path=path, index=index, with_observations=False
                )
                assert isinstance(record, ImageHeader)
                if record.image_id <= 0:
                    raise AdapterError(
                        f"Invalid non-positive image id {record.image_id} in {path}"
                    )
                if record.image_id in ids:
                    raise AdapterError(f"Duplicate image id {record.image_id} in {path}")
                if record.name in names:
                    raise AdapterError(f"Duplicate image name {record.name!r} in {path}")
                ids.add(record.image_id)
                names.add(record.name)
                images.append(record)
            _assert_no_trailing_bytes(handle, path)
    except OSError as exc:
        raise AdapterError(f"Cannot read {path}: {exc}") from exc
    if not images:
        raise AdapterError(f"COLMAP model has no registered images: {path}")
    return images


def read_selected_images_binary(
    path: Path, selected_image_ids: set[int]
) -> dict[int, RegisteredImage]:
    selected: dict[int, RegisteredImage] = {}
    try:
        with path.open("rb") as handle:
            count = _read_count(
                handle,
                context=f"registered image count in {path}",
                limit=_MAX_REGISTERED_IMAGES,
            )
            for index in range(count):
                # Reading the fixed header and name is necessary to discover the id.
                header_pos = handle.tell()
                fixed = _BINARY_IMAGE_HEADER.unpack(
                    _read_exact(
                        handle,
                        _BINARY_IMAGE_HEADER.size,
                        context=f"image {index} header in {path}",
                    )
                )
                image_id = int(fixed[0])
                # Rewind so the shared parser handles the complete record.
                handle.seek(header_pos, os.SEEK_SET)
                record = _read_image_record(
                    handle,
                    path=path,
                    index=index,
                    with_observations=image_id in selected_image_ids,
                )
                if isinstance(record, RegisteredImage):
                    if record.image_id in selected:
                        raise AdapterError(f"Duplicate image id {record.image_id} in {path}")
                    selected[record.image_id] = record
            _assert_no_trailing_bytes(handle, path)
    except OSError as exc:
        raise AdapterError(f"Cannot read {path}: {exc}") from exc

    missing = selected_image_ids.difference(selected)
    if missing:
        raise AdapterError(f"Selected image ids absent from {path}: {sorted(missing)}")
    return selected


def _read_point3d(
    handle: BinaryIO, *, path: Path, index: int
) -> Point3D:
    values = _BINARY_POINT_HEADER.unpack(
        _read_exact(
            handle,
            _BINARY_POINT_HEADER.size,
            context=f"point3D {index} header in {path}",
        )
    )
    point3d_id = int(values[0])
    track_length = _read_count(
        handle,
        context=f"track length for point3D {point3d_id} in {path}",
        limit=_MAX_TRACK_LENGTH,
    )
    raw_track = _read_exact(
        handle,
        _BINARY_TRACK.size * track_length,
        context=f"track for point3D {point3d_id} in {path}",
    )
    point = Point3D(
        point3d_id=point3d_id,
        xyz=(float(values[1]), float(values[2]), float(values[3])),
        rgb=(int(values[4]), int(values[5]), int(values[6])),
        error=float(values[7]),
        track=tuple(
            (int(image_id), int(point2d_idx))
            for image_id, point2d_idx in _BINARY_TRACK.iter_unpack(raw_track)
        ),
    )
    if not all(math.isfinite(value) for value in (*point.xyz, point.error)):
        raise AdapterError(f"point3D {point3d_id} contains non-finite geometry/error")
    if point.error < 0:
        raise AdapterError(f"point3D {point3d_id} has negative reprojection error")
    return point


def _iter_points3d_binary(path: Path) -> Iterator[tuple[int, Point3D]]:
    try:
        with path.open("rb") as handle:
            count = _read_count(handle, context=f"point3D count in {path}")
            for index in range(count):
                yield count, _read_point3d(handle, path=path, index=index)
            _assert_no_trailing_bytes(handle, path)
    except OSError as exc:
        raise AdapterError(f"Cannot read {path}: {exc}") from exc


def _normalised_name(name: str) -> PurePosixPath:
    normalised = PurePosixPath(name.replace("\\", "/"))
    if normalised.is_absolute() or not normalised.parts or any(
        part in {"", ".", ".."} for part in normalised.parts
    ):
        raise AdapterError(f"Unsafe COLMAP image name: {name!r}")
    return normalised


def _source_image_path(image_root: Path, name: str) -> Path:
    relative = _normalised_name(name)
    candidate = image_root.joinpath(*relative.parts)
    # The archive is untrusted input: never follow a COLMAP name outside images/.
    root_resolved = image_root.resolve()
    candidate_resolved = candidate.resolve()
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise AdapterError(f"COLMAP image path escapes {image_root}: {name!r}") from exc
    return candidate


def _is_iphone_name(name: str) -> bool:
    # Match the DA3 loader's semantics (it checks for the substring "iphone")
    # while normalising case and path separators.
    return "iphone" in str(_normalised_name(name)).casefold()


def _round_ratio_ties_to_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    twice = 2 * remainder
    if twice < denominator:
        return quotient
    if twice > denominator:
        return quotient + 1
    return quotient if quotient % 2 == 0 else quotient + 1


def evenly_spaced_indices(population: int, count: int) -> list[int]:
    """Match ``numpy.linspace(...).round().astype(int)`` without NumPy."""

    if population < 0 or count < 0:
        raise ValueError("population and count must be non-negative")
    if count == 0 or population == 0:
        return []
    if population <= count:
        return list(range(population))
    if count == 1:
        return [0]
    denominator = count - 1
    return [
        _round_ratio_ties_to_even(index * (population - 1), denominator)
        for index in range(count)
    ]


def _camera_center(
    qvec: tuple[float, float, float, float],
    tvec: tuple[float, float, float],
) -> list[float]:
    norm = math.sqrt(sum(value * value for value in qvec))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise AdapterError(f"Invalid COLMAP quaternion: {qvec}")
    if abs(norm - 1.0) > 1e-3:
        raise AdapterError(f"COLMAP quaternion is not unit length ({norm:.8g}): {qvec}")
    qw, qx, qy, qz = (value / norm for value in qvec)
    rotation = (
        (
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw),
        ),
        (
            2 * (qx * qy + qz * qw),
            1 - 2 * (qx * qx + qz * qz),
            2 * (qy * qz - qx * qw),
        ),
        (
            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx * qx + qy * qy),
        ),
    )
    return [
        -sum(rotation[row][column] * tvec[row] for row in range(3))
        for column in range(3)
    ]


def _validate_regular_file(path: Path, *, label: str, prefix: bytes | None = None) -> None:
    if not path.is_file():
        raise AdapterError(f"Missing {label}: {path}")
    try:
        if path.stat().st_size <= 0:
            raise AdapterError(f"Empty {label}: {path}")
        if prefix is not None:
            with path.open("rb") as handle:
                actual = handle.read(len(prefix))
            if actual != prefix:
                raise AdapterError(
                    f"Invalid {label} signature at {path}: expected {prefix!r}, got {actual!r}"
                )
    except OSError as exc:
        raise AdapterError(f"Cannot inspect {label} {path}: {exc}") from exc


def _relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise AdapterError(f"Refusing to replace existing output path: {destination}")
    target = os.path.relpath(source.resolve(), start=destination.parent.resolve())
    destination.symlink_to(target)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_cameras_text(path: Path, cameras: Iterable[Camera]) -> None:
    ordered = sorted(cameras, key=lambda camera: camera.camera_id)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Camera list with one line of data per camera:\n")
        handle.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        handle.write(f"# Number of cameras: {len(ordered)}\n")
        for camera in ordered:
            fields = [
                str(camera.camera_id),
                camera.model_name,
                str(camera.width),
                str(camera.height),
                *(_format_float(value) for value in camera.params),
            ]
            handle.write(" ".join(fields) + "\n")


def _write_images_text(path: Path, images: Sequence[RegisteredImage]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        handle.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        handle.write(f"# Number of images: {len(images)}\n")
        for image in images:
            output_name = _normalised_name(image.name).name
            header = [
                str(image.image_id),
                *(_format_float(value) for value in image.qvec),
                *(_format_float(value) for value in image.tvec),
                str(image.camera_id),
                output_name,
            ]
            handle.write(" ".join(header) + "\n")
            observations: list[str] = []
            for (x, y), point3d_id in zip(image.xys, image.point3d_ids):
                observations.extend(
                    (_format_float(x), _format_float(y), str(point3d_id))
                )
            handle.write(" ".join(observations) + "\n")


def _write_points3d_text(
    source: Path,
    destination: Path,
    selected_images: dict[int, RegisteredImage],
) -> tuple[int, int]:
    expected: dict[int, set[tuple[int, int]]] = {}
    for image in selected_images.values():
        for point2d_idx, point3d_id in enumerate(image.point3d_ids):
            if point3d_id >= 0:
                expected.setdefault(point3d_id, set()).add(
                    (image.image_id, point2d_idx)
                )

    seen_expected: set[tuple[int, int, int]] = set()
    seen_point_ids: set[int] = set()
    point_count = 0
    retained_track_count = 0
    declared_count: int | None = None

    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# 3D point list with one line of data per point:\n")
        handle.write(
            "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
            "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )
        for declared_count, point in _iter_points3d_binary(source):
            if point.point3d_id in seen_point_ids:
                raise AdapterError(
                    f"Duplicate point3D id {point.point3d_id} in {source}"
                )
            seen_point_ids.add(point.point3d_id)
            point_count += 1
            retained_track: list[tuple[int, int]] = []
            local_pairs: set[tuple[int, int]] = set()
            for image_id, point2d_idx in point.track:
                if image_id not in selected_images:
                    continue
                pair = (image_id, point2d_idx)
                if pair in local_pairs:
                    raise AdapterError(
                        f"Duplicate track entry {pair} for point3D {point.point3d_id}"
                    )
                local_pairs.add(pair)
                image = selected_images[image_id]
                if point2d_idx < 0 or point2d_idx >= len(image.point3d_ids):
                    raise AdapterError(
                        f"point3D {point.point3d_id} track references invalid "
                        f"POINT2D index {point2d_idx} in image {image_id}"
                    )
                referenced_id = image.point3d_ids[point2d_idx]
                if referenced_id != point.point3d_id:
                    raise AdapterError(
                        f"Inconsistent COLMAP track: point3D {point.point3d_id} "
                        f"references image {image_id} POINT2D {point2d_idx}, which "
                        f"references point3D {referenced_id}"
                    )
                retained_track.append(pair)
                seen_expected.add((point.point3d_id, image_id, point2d_idx))

            fields = [
                str(point.point3d_id),
                *(_format_float(value) for value in point.xyz),
                *(str(value) for value in point.rgb),
                _format_float(point.error),
            ]
            for image_id, point2d_idx in retained_track:
                fields.extend((str(image_id), str(point2d_idx)))
            handle.write(" ".join(fields) + "\n")
            retained_track_count += len(retained_track)

    if declared_count is None:
        # The iterator yields no records for a valid zero-point file, so inspect
        # its count explicitly to distinguish it from an unread file.
        try:
            with source.open("rb") as binary:
                declared_count = _read_count(
                    binary, context=f"point3D count in {source}"
                )
        except OSError as exc:
            raise AdapterError(f"Cannot read {source}: {exc}") from exc
    if point_count != declared_count:
        raise AdapterError(
            f"Parsed {point_count} point3D records from {source}, expected {declared_count}"
        )
    if point_count == 0:
        raise AdapterError(f"COLMAP model has no sparse points: {source}")

    missing_reciprocal: list[tuple[int, int, int]] = []
    for point3d_id, observations in expected.items():
        if point3d_id not in seen_point_ids:
            for image_id, point2d_idx in observations:
                missing_reciprocal.append((point3d_id, image_id, point2d_idx))
            continue
        for image_id, point2d_idx in observations:
            triple = (point3d_id, image_id, point2d_idx)
            if triple not in seen_expected:
                missing_reciprocal.append(triple)
    if missing_reciprocal:
        preview = ", ".join(map(str, missing_reciprocal[:5]))
        suffix = " ..." if len(missing_reciprocal) > 5 else ""
        raise AdapterError(
            f"Selected image observations lack reciprocal point3D tracks: "
            f"{preview}{suffix}"
        )
    return point_count, retained_track_count


def _scene_layout(scene_root: Path) -> SceneLayout:
    scene_root = scene_root.resolve()
    return SceneLayout(
        scene_id=scene_root.name,
        scene_root=scene_root,
        model_dir=scene_root
        / "merge_dslr_iphone"
        / "colmap"
        / "sparse_render_rgb",
        image_root=scene_root / "merge_dslr_iphone" / "images",
        depth_root=scene_root / "merge_dslr_iphone" / "render_depth",
        mesh_path=scene_root / "scans" / "mesh_aligned_0.05.ply",
    )


def discover_source_scenes(input_root: Path) -> dict[str, Path]:
    """Discover extracted DA3-BENCH scene roots below ``input_root``."""

    input_root = input_root.expanduser().resolve()
    if not input_root.is_dir():
        raise AdapterError(f"Input root is not a directory: {input_root}")

    model_suffix = Path("merge_dslr_iphone/colmap/sparse_render_rgb/cameras.bin")
    direct = input_root / model_suffix
    candidates: list[Path]
    if direct.is_file():
        candidates = [input_root]
    else:
        candidates = []
        for cameras_bin in input_root.rglob("cameras.bin"):
            try:
                relative = cameras_bin.relative_to(input_root)
            except ValueError:
                continue
            if len(relative.parts) >= len(model_suffix.parts) and Path(
                *relative.parts[-len(model_suffix.parts) :]
            ) == model_suffix:
                candidates.append(cameras_bin.parents[3])

    scenes: dict[str, Path] = {}
    for candidate in sorted(set(candidates), key=lambda path: str(path)):
        scene_id = candidate.name
        if scene_id in scenes and scenes[scene_id] != candidate:
            raise AdapterError(
                f"Duplicate scene id {scene_id!r}: {scenes[scene_id]} and {candidate}"
            )
        scenes[scene_id] = candidate
    if not scenes:
        raise AdapterError(
            f"No DA3-BENCH scenes found below {input_root}; expected "
            f"<scene>/{model_suffix.parent}/{{cameras,images,points3D}}.bin"
        )
    return dict(sorted(scenes.items()))


def _validate_model_files(layout: SceneLayout) -> tuple[Path, Path, Path]:
    cameras_bin = layout.model_dir / "cameras.bin"
    images_bin = layout.model_dir / "images.bin"
    points_bin = layout.model_dir / "points3D.bin"
    for label, path in (
        ("COLMAP cameras.bin", cameras_bin),
        ("COLMAP images.bin", images_bin),
        ("COLMAP points3D.bin", points_bin),
    ):
        _validate_regular_file(path, label=label)
    if not layout.image_root.is_dir():
        raise AdapterError(f"Missing DA3 image directory: {layout.image_root}")
    return cameras_bin, images_bin, points_bin


def _install_stage(stage: Path, destination: Path, *, overwrite: bool) -> None:
    backup: Path | None = None
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            raise AdapterError(
                f"Output scene already exists: {destination} "
                "(pass --overwrite to replace it atomically)"
            )
        if destination.is_symlink() or not destination.is_dir():
            raise AdapterError(
                f"--overwrite only replaces an existing scene directory, not: {destination}"
            )
        backup = destination.parent / f".{destination.name}.backup-{os.getpid()}"
        if backup.exists() or backup.is_symlink():
            raise AdapterError(f"Stale adapter backup blocks overwrite: {backup}")
        os.replace(destination, backup)
    try:
        os.replace(stage, destination)
    except BaseException:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def prepare_scene(
    source_scene: Path,
    output_root: Path,
    *,
    num_views: int = 8,
    require_eval_assets: bool = True,
    overwrite: bool = False,
) -> dict[str, object]:
    """Convert one extracted DA3-BENCH scene and return its manifest."""

    if num_views <= 0:
        raise AdapterError(f"num_views must be positive, got {num_views}")
    layout = _scene_layout(source_scene)
    cameras_bin, images_bin, points_bin = _validate_model_files(layout)
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / layout.scene_id
    if (destination.exists() or destination.is_symlink()) and not overwrite:
        raise AdapterError(
            f"Output scene already exists: {destination} "
            "(pass --overwrite to replace it atomically)"
        )

    cameras = read_cameras_binary(cameras_bin)
    image_headers = read_image_headers_binary(images_bin)
    for image in image_headers:
        if image.camera_id not in cameras:
            raise AdapterError(
                f"Image {image.image_id} references missing camera {image.camera_id}"
            )
        if not all(math.isfinite(value) for value in (*image.qvec, *image.tvec)):
            raise AdapterError(f"Image {image.image_id} has non-finite pose values")

    iphone_headers = sorted(
        (image for image in image_headers if _is_iphone_name(image.name)),
        key=lambda image: image.name,
    )
    if len(iphone_headers) < num_views:
        raise AdapterError(
            f"Scene {layout.scene_id} has only {len(iphone_headers)} registered iPhone "
            f"views, fewer than requested {num_views}"
        )

    # Validate every registered iPhone image before sampling.  This prevents a
    # partial extraction from silently changing the evenly-spaced selection.
    source_images: dict[int, Path] = {}
    for image in iphone_headers:
        source_path = _source_image_path(layout.image_root, image.name)
        _validate_regular_file(source_path, label=f"registered iPhone RGB {image.name}")
        source_images[image.image_id] = source_path

    indices = evenly_spaced_indices(len(iphone_headers), num_views)
    selected_headers = [iphone_headers[index] for index in indices]
    selected_ids = {image.image_id for image in selected_headers}
    selected_by_id = read_selected_images_binary(images_bin, selected_ids)
    selected_images = [selected_by_id[image.image_id] for image in selected_headers]

    basenames: dict[str, str] = {}
    depth_names: dict[str, str] = {}
    selected_assets: list[tuple[RegisteredImage, Path, Path]] = []
    for image in selected_images:
        basename = _normalised_name(image.name).name
        previous = basenames.setdefault(basename, image.name)
        if previous != image.name:
            raise AdapterError(
                f"Selected image basename collision: {previous!r} and {image.name!r} "
                f"both map to {basename!r}"
            )
        depth_name = f"{PurePosixPath(basename).stem}.png"
        previous_depth = depth_names.setdefault(depth_name, image.name)
        if previous_depth != image.name:
            raise AdapterError(
                f"Selected depth basename collision: {previous_depth!r} and "
                f"{image.name!r} both map to {depth_name!r}"
            )
        depth_path = layout.depth_root / depth_name
        if require_eval_assets:
            _validate_regular_file(
                depth_path,
                label=f"render depth paired with {image.name}",
                prefix=b"\x89PNG\r\n\x1a\n",
            )
        selected_assets.append((image, source_images[image.image_id], depth_path))

    if require_eval_assets:
        _validate_regular_file(
            layout.mesh_path,
            label="aligned ScanNet++ evaluation mesh",
            prefix=b"ply",
        )

    stage = Path(
        tempfile.mkdtemp(prefix=f".{layout.scene_id}.tmp-", dir=str(output_root))
    )
    try:
        colmap_out = stage / "iphone" / "colmap"
        rgb_out = stage / "iphone" / "rgb"
        depth_out = stage / "render_depth"
        scans_out = stage / "scans"
        colmap_out.mkdir(parents=True)
        rgb_out.mkdir(parents=True)

        used_camera_ids = {image.camera_id for image in selected_images}
        selected_cameras = [cameras[camera_id] for camera_id in used_camera_ids]
        unsupported = sorted(
            camera.model_name
            for camera in selected_cameras
            if camera.model_name not in _GENRECON_CAMERA_MODELS
        )
        if unsupported:
            raise AdapterError(
                "Selected views use camera model(s) unsupported by GenRecon's "
                f"Scannet_iphone loader: {unsupported}"
            )

        _write_cameras_text(colmap_out / "cameras.txt", selected_cameras)
        _write_images_text(colmap_out / "images.txt", selected_images)
        point_count, retained_track_count = _write_points3d_text(
            points_bin,
            colmap_out / "points3D.txt",
            selected_by_id,
        )

        view_entries: list[dict[str, object]] = []
        for selection_order, (source_index, asset) in enumerate(
            zip(indices, selected_assets)
        ):
            image, source_image, source_depth = asset
            basename = _normalised_name(image.name).name
            output_image = rgb_out / basename
            _relative_symlink(source_image, output_image)
            depth_name = f"{PurePosixPath(basename).stem}.png"
            output_depth: str | None = None
            if source_depth.is_file():
                _relative_symlink(source_depth, depth_out / depth_name)
                output_depth = f"render_depth/{depth_name}"
            camera = cameras[image.camera_id]
            view_entries.append(
                {
                    "camera_center_world_m": _camera_center(image.qvec, image.tvec),
                    "camera_id": image.camera_id,
                    "camera_model": camera.model_name,
                    "eligible_sorted_index": source_index,
                    "image_id": image.image_id,
                    "intrinsics": {
                        "height": camera.height,
                        "params": list(camera.params),
                        "width": camera.width,
                    },
                    "original_name": image.name,
                    "output_depth": output_depth,
                    "output_image": f"iphone/rgb/{basename}",
                    "qvec_wxyz": list(image.qvec),
                    "selection_order": selection_order,
                    "tvec_world_to_camera": list(image.tvec),
                }
            )

        output_mesh: str | None = None
        if layout.mesh_path.is_file():
            _relative_symlink(
                layout.mesh_path, scans_out / "mesh_aligned_0.05.ply"
            )
            output_mesh = "scans/mesh_aligned_0.05.ply"

        manifest: dict[str, object] = {
            "counts": {
                "registered_images_all": len(image_headers),
                "registered_iphone_images": len(iphone_headers),
                "retained_sparse_tracks": retained_track_count,
                "selected_images": len(selected_images),
                "sparse_points_full": point_count,
            },
            "evaluation_assets_required": require_eval_assets,
            "genrecon": {
                "center_crop_required_for_exact_view_count": True,
                "colmap_text_dir": "iphone/colmap",
                "mode": "Scannet_iphone",
                "num_imgs_per_scene": len(selected_images),
            },
            "output_mesh": output_mesh,
            "scene_id": layout.scene_id,
            "schema": "genrecon.da3_scannetpp_adapter",
            "schema_version": 1,
            "selection": {
                "eligible_order": "lexicographic COLMAP image name",
                "indices": indices,
                "policy": "linspace endpoints inclusive, round ties to even",
                "requested_views": num_views,
            },
            "source": {
                "colmap_model": str(layout.model_dir),
                "mesh": str(layout.mesh_path),
                "scene_root": str(layout.scene_root),
            },
            "views": view_entries,
        }
        _write_json(stage / "selection.json", manifest)
        _install_stage(stage, destination, overwrite=overwrite)
        return manifest
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _load_prepared_scene_manifests(
    output_root: Path,
) -> dict[str, dict[str, object]]:
    prepared_results: dict[str, dict[str, object]] = {}
    if not output_root.exists():
        return prepared_results
    for selection_path in sorted(output_root.glob("*/selection.json")):
        try:
            prepared = json.loads(selection_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AdapterError(
                f"Cannot read prepared scene manifest {selection_path}: {exc}"
            ) from exc
        scene_id = prepared.get("scene_id")
        source = prepared.get("source")
        if (
            prepared.get("schema") != "genrecon.da3_scannetpp_adapter"
            or not isinstance(scene_id, str)
            or scene_id != selection_path.parent.name
            or not isinstance(prepared.get("counts"), dict)
            or not isinstance(source, dict)
            or not isinstance(source.get("scene_root"), str)
        ):
            raise AdapterError(f"Invalid prepared scene manifest: {selection_path}")
        prepared_results[scene_id] = prepared
    return prepared_results


def _validate_prepared_protocol(
    prepared_results: dict[str, dict[str, object]],
    *,
    num_views: int,
    require_eval_assets: bool,
) -> None:
    for scene_id, prepared in prepared_results.items():
        genrecon = prepared.get("genrecon")
        prepared_views = (
            genrecon.get("num_imgs_per_scene") if isinstance(genrecon, dict) else None
        )
        prepared_eval = prepared.get("evaluation_assets_required")
        if prepared_views != num_views or prepared_eval is not require_eval_assets:
            raise AdapterError(
                f"Output root already contains scene {scene_id} prepared with a "
                f"different protocol (num_views={prepared_views}, "
                f"require_eval_assets={prepared_eval}); use a separate output root"
            )


def prepare_dataset(
    input_root: Path,
    output_root: Path,
    *,
    scene_ids: Sequence[str] | None = None,
    num_views: int = 8,
    require_eval_assets: bool = True,
    overwrite: bool = False,
) -> dict[str, object]:
    scenes = discover_source_scenes(input_root)
    requested = list(scene_ids or scenes.keys())
    if not requested:
        raise AdapterError("No scenes requested")
    if len(requested) != len(set(requested)):
        raise AdapterError(f"Duplicate --scene values: {requested}")
    missing = sorted(set(requested).difference(scenes))
    if missing:
        raise AdapterError(
            f"Requested scene(s) not found: {missing}; available: {list(scenes)}"
        )

    output_root = output_root.expanduser().resolve()
    existing_results = _load_prepared_scene_manifests(output_root)
    _validate_prepared_protocol(
        existing_results,
        num_views=num_views,
        require_eval_assets=require_eval_assets,
    )
    for position, scene_id in enumerate(requested, start=1):
        print(
            f"[{position}/{len(requested)}] preparing DA3 ScanNet++ scene {scene_id}",
            file=sys.stderr,
        )
        prepare_scene(
            scenes[scene_id],
            output_root,
            num_views=num_views,
            require_eval_assets=require_eval_assets,
            overwrite=overwrite,
        )

    # Rebuild the root manifest from all prepared scene manifests. This keeps
    # separate --scene invocations additive instead of silently forgetting
    # scenes prepared by earlier invocations.
    prepared_results = _load_prepared_scene_manifests(output_root)
    _validate_prepared_protocol(
        prepared_results,
        num_views=num_views,
        require_eval_assets=require_eval_assets,
    )
    input_roots = sorted(
        {
            str(Path(result["source"]["scene_root"]).parent)
            for result in prepared_results.values()
        }
    )

    dataset_manifest: dict[str, object] = {
        "input_root": input_roots[0] if len(input_roots) == 1 else None,
        "input_roots": input_roots,
        "num_views_per_scene": num_views,
        "output_root": str(output_root),
        "require_eval_assets": require_eval_assets,
        "scene_ids": sorted(prepared_results),
        "scenes": [
            {
                "counts": result["counts"],
                "scene_id": result["scene_id"],
                "selection_manifest": f"{result['scene_id']}/selection.json",
            }
            for result in prepared_results.values()
        ],
        "schema": "genrecon.da3_scannetpp_dataset",
        "schema_version": 1,
    }
    _write_json(output_root / "da3_scannetpp_manifest.json", dataset_manifest)
    return dataset_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Extracted scannetpp.zip root, or one DA3-BENCH scene directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination containing one GenRecon-compatible directory per scene.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        dest="scene_ids",
        help="Prepare only this scene id; repeat for multiple scenes. Default: all.",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=8,
        help="Deterministic registered iPhone views per scene (default: 8).",
    )
    parser.add_argument(
        "--require-eval-assets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require/link each selected render_depth PNG and aligned mesh (default: yes).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing output scene directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = prepare_dataset(
            args.input_root,
            args.output_root,
            scene_ids=args.scene_ids,
            num_views=args.num_views,
            require_eval_assets=args.require_eval_assets,
            overwrite=args.overwrite,
        )
    except (AdapterError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
