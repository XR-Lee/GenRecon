#!/usr/bin/env python3
"""Estimate T&T Meetingroom intrinsics while keeping official poses fixed."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "gt-calibration-v1" / "sources" / "tanks-and-temples"
DEFAULT_OUTPUT = DEFAULT_SOURCE / "Meetingroom_intrinsics.json"
DEFAULT_WORK = ROOT / "outputs" / "gt-calibration-v1" / "tnt-meetingroom-intrinsics"
SCHEMA = "genrecon.tnt-fixed-pose-intrinsics"
CALIBRATION_REVISION = "v2-single-thread-geometry"


class TntCalibrationError(RuntimeError):
    pass


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def canonicalize_source_references(
    document: dict[str, Any], image_archive: Path, camera_log: Path
) -> bool:
    source = document.get("source")
    if not isinstance(source, dict):
        raise TntCalibrationError("Calibration artifact is missing source metadata")
    expected = {
        "image_archive": image_archive.name,
        "camera_log": camera_log.name,
    }
    changed = any(source.get(key) != value for key, value in expected.items())
    source.update(expected)
    return changed


def read_poses(path: Path) -> list[np.ndarray]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) % 5:
        raise TntCalibrationError(f"Camera log must contain five lines per pose: {path}")
    poses = []
    for offset in range(0, len(lines), 5):
        record_index = offset // 5
        try:
            metadata = [int(value) for value in lines[offset].split()]
        except ValueError as exc:
            raise TntCalibrationError(
                f"Invalid camera-log metadata at record {record_index}: {path}"
            ) from exc
        if metadata != [record_index, record_index, 0]:
            raise TntCalibrationError(
                f"Unexpected camera-log mapping at record {record_index}: {metadata}"
            )
        matrix = np.asarray(
            [
                [float(value) for value in lines[offset + row].split()]
                for row in range(1, 5)
            ],
            dtype=np.float64,
        )
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise TntCalibrationError(f"Invalid pose at record {offset // 5}: {path}")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise TntCalibrationError(
                f"Invalid homogeneous pose at record {offset // 5}: {path}"
            )
        poses.append(matrix)
    return poses


def archive_image_names(path: Path) -> list[str]:
    with ZipFile(path) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    if not names:
        raise TntCalibrationError(f"No images in {path}")
    return names


def extract_images(archive_path: Path, output: Path, names: list[str]) -> None:
    expected = {Path(name).name for name in names}
    stamp_path = output.parent / "images_source.json"
    existing = (
        {
            path.name
            for path in output.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }
        if output.is_dir()
        else set()
    )
    archive_sha256 = sha256_file(archive_path)
    stamp = (
        json.loads(stamp_path.read_text(encoding="utf-8"))
        if stamp_path.is_file()
        else {}
    )
    if (
        existing == expected
        and stamp.get("archive_sha256") == archive_sha256
        and stamp.get("image_names") == names
    ):
        return
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    with ZipFile(archive_path) as archive:
        for name in names:
            (output / Path(name).name).write_bytes(archive.read(name))
    write_json(
        stamp_path,
        {"archive_sha256": archive_sha256, "image_names": names},
    )


def create_ordered_database(
    pycolmap: Any,
    database_path: Path,
    images_dir: Path,
    *,
    width: int,
    height: int,
    focal: float,
    max_image_size: int,
    num_threads: int,
) -> None:
    if database_path.exists():
        database_path.unlink()
    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    with pycolmap.Database.open(str(database_path)) as database:
        camera = pycolmap.Camera(
            camera_id=1,
            model="SIMPLE_RADIAL",
            width=width,
            height=height,
            params=[focal, width / 2.0, height / 2.0, 0.0],
        )
        camera.has_prior_focal_length = True
        database.write_camera(camera, use_camera_id=True)
        for image_id, path in enumerate(image_paths, 1):
            database.write_image(
                pycolmap.Image(name=path.name, camera_id=1, image_id=image_id),
                use_image_id=True,
            )

    reader = pycolmap.ImageReaderOptions(existing_camera_id=1)
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = max_image_size
    extraction.num_threads = num_threads
    extraction.sift.max_num_features = 8192
    pycolmap.extract_features(
        str(database_path),
        str(images_dir),
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model="SIMPLE_RADIAL",
        reader_options=reader,
        extraction_options=extraction,
        device=pycolmap.Device.cpu,
    )


def match_database(pycolmap: Any, database_path: Path, *, num_threads: int) -> None:
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = num_threads
    matching.use_gpu = False
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = 10
    pairing.quadratic_overlap = True
    pairing.loop_detection = False
    verification = pycolmap.TwoViewGeometryOptions()
    verification.ransac.random_seed = 42
    pycolmap.match_sequential(
        str(database_path),
        matching_options=matching,
        pairing_options=pairing,
        verification_options=verification,
        device=pycolmap.Device.cpu,
    )


def write_known_pose_model(
    pycolmap: Any,
    database_path: Path,
    poses: list[np.ndarray],
    output: Path,
    *,
    focal: float,
    width: int,
    height: int,
) -> None:
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    (output / "cameras.txt").write_text(
        f"1 SIMPLE_RADIAL {width} {height} {focal:.17g} {width / 2:.17g} "
        f"{height / 2:.17g} 0\n",
        encoding="utf-8",
    )
    with pycolmap.Database.open(str(database_path)) as database:
        images = sorted(database.read_all_images(), key=lambda image: image.image_id)
    if len(images) != len(poses):
        raise TntCalibrationError(f"Image/pose count mismatch: {len(images)} != {len(poses)}")
    with (output / "images.txt").open("w", encoding="utf-8") as handle:
        for image, camera_to_world in zip(images, poses):
            world_to_camera = np.linalg.inv(camera_to_world)
            quaternion_xyzw = Rotation.from_matrix(world_to_camera[:3, :3]).as_quat()
            quaternion_wxyz = [quaternion_xyzw[3], *quaternion_xyzw[:3]]
            handle.write(
                f"{image.image_id} "
                f"{' '.join(f'{value:.17g}' for value in quaternion_wxyz)} "
                f"{' '.join(f'{value:.17g}' for value in world_to_camera[:3, 3])} "
                f"1 {image.name}\n\n"
            )
    (output / "points3D.txt").write_text("", encoding="utf-8")


def reconstruction_stats(reconstruction: Any) -> dict[str, Any]:
    reconstruction.update_point_3d_errors()
    supported_images = sum(
        image.num_points3D > 0 for image in reconstruction.images.values()
    )
    errors = np.asarray(
        [point.error for point in reconstruction.points3D.values()], dtype=np.float64
    )
    return {
        "registered_images": reconstruction.num_reg_images(),
        "supported_images": supported_images,
        "points3D": reconstruction.num_points3D(),
        "observations": reconstruction.compute_num_observations(),
        "mean_track_length": reconstruction.compute_mean_track_length(),
        "mean_reprojection_error_px": reconstruction.compute_mean_reprojection_error(),
        "point_error_median_px": float(np.median(errors)) if len(errors) else None,
        "point_error_p90_px": float(np.quantile(errors, 0.90)) if len(errors) else None,
    }


def compare_official_poses(
    reconstruction: Any, poses: list[np.ndarray]
) -> dict[str, Any]:
    matrix_deltas = []
    translation_deltas = []
    rotation_deltas_deg = []
    source_indices = set()
    for image in reconstruction.images.values():
        try:
            source_index = int(Path(image.name).stem) - 1
        except ValueError as exc:
            raise TntCalibrationError(f"Non-numeric Meetingroom image name: {image.name}") from exc
        if source_index < 0 or source_index >= len(poses) or source_index in source_indices:
            raise TntCalibrationError(f"Invalid or duplicate Meetingroom source index: {source_index}")
        source_indices.add(source_index)
        expected = np.linalg.inv(poses[source_index])[:3]
        actual = image.cam_from_world().matrix()
        matrix_deltas.append(float(np.linalg.norm(actual - expected)))
        translation_deltas.append(
            float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
        )
        delta_rotation = actual[:3, :3] @ expected[:3, :3].T
        rotation_deltas_deg.append(
            float(np.degrees(Rotation.from_matrix(delta_rotation).magnitude()))
        )
    if source_indices != set(range(len(poses))):
        raise TntCalibrationError("Final reconstruction does not map one-to-one to official poses")
    return {
        "compared_poses": len(source_indices),
        "max_matrix_frobenius_delta": max(matrix_deltas),
        "median_matrix_frobenius_delta": float(np.median(matrix_deltas)),
        "max_translation_delta": max(translation_deltas),
        "max_rotation_delta_deg": max(rotation_deltas_deg),
    }


def fixed_pose_bundle_adjustment(
    pycolmap: Any,
    reconstruction: Any,
    *,
    num_threads: int,
) -> dict[str, Any]:
    poses_before = {
        image_id: image.cam_from_world().matrix().copy()
        for image_id, image in reconstruction.images.items()
    }
    config = pycolmap.BundleAdjustmentConfig()
    for image_id, image in reconstruction.images.items():
        config.add_image(image_id)
        config.set_constant_rig_from_world_pose(image.frame_id)
    config.set_variable_cam_intrinsics(1)

    options = pycolmap.BundleAdjustmentOptions()
    options.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    options.loss_function_scale = 1.0
    options.refine_focal_length = True
    options.refine_principal_point = False
    options.refine_extra_params = True
    options.refine_rig_from_world = False
    options.refine_sensor_from_rig = False
    options.print_summary = True
    options.use_gpu = False
    options.solver_options.max_num_iterations = 100
    options.solver_options.num_threads = num_threads
    summary = pycolmap.create_default_bundle_adjuster(
        options, config, reconstruction
    ).solve()
    max_pose_delta = max(
        float(np.linalg.norm(poses_before[image_id] - image.cam_from_world().matrix()))
        for image_id, image in reconstruction.images.items()
    )
    reconstruction.update_point_3d_errors()
    return {
        "solution_usable": bool(summary.IsSolutionUsable()),
        "termination": str(summary.termination_type).split(".")[-1],
        "message": summary.message,
        "iterations": int(summary.num_successful_steps + summary.num_unsuccessful_steps),
        "residuals": int(summary.num_residuals),
        "initial_cost": float(summary.initial_cost),
        "final_cost": float(summary.final_cost),
        "max_fixed_pose_matrix_delta": max_pose_delta,
        "camera_params": reconstruction.camera(1).params.tolist(),
    }


def triangulate(
    pycolmap: Any,
    reconstruction: Any,
    database_path: Path,
    images_dir: Path,
    output: Path,
    *,
    num_threads: int,
) -> Any:
    shutil.rmtree(output, ignore_errors=True)
    options = pycolmap.IncrementalPipelineOptions()
    options.fix_existing_frames = True
    options.mapper.fix_existing_frames = True
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    options.num_threads = num_threads
    options.random_seed = 42
    options.triangulation.random_seed = 42
    return pycolmap.triangulate_points(
        reconstruction,
        str(database_path),
        str(images_dir),
        str(output),
        clear_points=True,
        options=options,
        refine_intrinsics=False,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import pycolmap
    except ModuleNotFoundError as exc:
        raise TntCalibrationError(
            "pycolmap is required; run with PYTHONPATH=/tmp/pycolmap-wheel"
        ) from exc

    source = args.source.resolve()
    image_archive = source / "Meetingroom.zip"
    camera_log = source / "Meetingroom_COLMAP.log"
    for path in (image_archive, camera_log):
        if not path.is_file():
            raise TntCalibrationError(f"Missing source: {path}")
    input_hashes = {
        "image_archive_sha256": sha256_file(image_archive),
        "camera_log_sha256": sha256_file(camera_log),
    }
    if args.output.is_file() and not args.force:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        existing_gates = existing.get("quality_gates")
        if (
            existing.get("schema") == SCHEMA
            and existing.get("schema_version") == 2
            and existing.get("result") == "pass"
            and existing.get("protocol", {}).get("revision") == CALIBRATION_REVISION
            and existing.get("protocol", {}).get("feature_max_image_size")
            == args.max_image_size
            and existing.get("software", {}).get("pycolmap_version")
            == pycolmap.__version__
            and isinstance(existing_gates, dict)
            and existing_gates
            and all(existing_gates.values())
            and all(
                existing.get("source", {}).get(key) == value
                for key, value in input_hashes.items()
            )
        ):
            if canonicalize_source_references(existing, image_archive, camera_log):
                write_json(args.output, existing)
                print(f"[tnt-calibration] normalized source references in {args.output}")
            print(f"[tnt-calibration] reuse {args.output}")
            return existing

    names = archive_image_names(image_archive)
    poses = read_poses(camera_log)
    if len(names) != len(poses):
        raise TntCalibrationError(f"Image/pose count mismatch: {len(names)} != {len(poses)}")
    if [int(Path(name).stem) for name in names] != list(range(1, len(names) + 1)):
        raise TntCalibrationError("Meetingroom image archive is not a contiguous 1-based sequence")
    with ZipFile(image_archive) as archive, Image.open(archive.open(names[0])) as image:
        width, height = image.size
        exif = image.getexif().get_ifd(34665)
        focal_mm = float(exif.get(37386)) if exif.get(37386) is not None else None
        focal_35mm = int(exif.get(41989)) if exif.get(41989) is not None else None

    work = args.work.resolve()
    images_dir = work / "images"
    work.mkdir(parents=True, exist_ok=True)
    extract_images(image_archive, images_dir, names)
    pycolmap.set_random_seed(42)
    database_path = work / "database.db"
    create_ordered_database(
        pycolmap,
        database_path,
        images_dir,
        width=width,
        height=height,
        focal=0.7 * width,
        max_image_size=args.max_image_size,
        num_threads=args.num_threads,
    )
    match_database(pycolmap, database_path, num_threads=1)
    with pycolmap.Database.open(str(database_path)) as database:
        database_stats = {
            "images": database.num_images(),
            "keypoints": database.num_keypoints(),
            "matched_pairs": database.num_matched_image_pairs(),
            "verified_pairs": database.num_verified_image_pairs(),
            "raw_matches": database.num_matches(),
            "geometric_inliers": database.num_inlier_matches(),
        }

    initial_model = work / "model-initial"
    write_known_pose_model(
        pycolmap,
        database_path,
        poses,
        initial_model,
        focal=0.7 * width,
        width=width,
        height=height,
    )
    reconstruction = triangulate(
        pycolmap,
        pycolmap.Reconstruction(str(initial_model)),
        database_path,
        images_dir,
        work / "model-triangulated-1",
        num_threads=1,
    )
    rounds = [
        {"stage": "initial-triangulation", "stats": reconstruction_stats(reconstruction)}
    ]
    for round_index in (1, 2):
        ba = fixed_pose_bundle_adjustment(pycolmap, reconstruction, num_threads=1)
        filtered_observations = pycolmap.ObservationManager(
            reconstruction
        ).filter_all_points3D(4.0, 1.0)
        rounds.append(
            {
                "stage": f"fixed-pose-ba-{round_index}",
                "bundle_adjustment": ba,
                "filtered_observations": int(filtered_observations),
                "stats": reconstruction_stats(reconstruction),
            }
        )
        if round_index == 1:
            reconstruction = triangulate(
                pycolmap,
                reconstruction,
                database_path,
                images_dir,
                work / "model-triangulated-2",
                num_threads=1,
            )
            rounds.append(
                {"stage": "retriangulation", "stats": reconstruction_stats(reconstruction)}
            )

    final_model = work / "model-final"
    shutil.rmtree(final_model, ignore_errors=True)
    final_model.mkdir(parents=True)
    reconstruction.write(str(final_model))
    final_stats = reconstruction_stats(reconstruction)
    official_pose_comparison = compare_official_poses(reconstruction, poses)
    focal, cx, cy, radial = (
        float(value) for value in reconstruction.camera(1).params.tolist()
    )
    max_pose_delta = max(
        item.get("bundle_adjustment", {}).get("max_fixed_pose_matrix_delta", 0.0)
        for item in rounds
    )
    ba_records = [item["bundle_adjustment"] for item in rounds if "bundle_adjustment" in item]
    first_params = np.asarray(ba_records[0]["camera_params"], dtype=np.float64)
    last_params = np.asarray(ba_records[-1]["camera_params"], dtype=np.float64)
    gates = {
        "all_images_registered": final_stats["registered_images"] == len(names),
        "image_support_fraction_at_least_0_9": final_stats["supported_images"]
        >= 0.9 * len(names),
        "points_at_least_5000": final_stats["points3D"] >= 5_000,
        "observations_at_least_10000": final_stats["observations"] >= 10_000,
        "mean_reprojection_error_at_most_2px": final_stats[
            "mean_reprojection_error_px"
        ]
        <= 2.0,
        "focal_ratio_plausible": 0.45 <= focal / width <= 0.9,
        "radial_parameter_plausible": abs(radial) <= 0.5,
        "official_poses_unchanged": (
            official_pose_comparison["max_matrix_frobenius_delta"] <= 1e-9
            and official_pose_comparison["max_translation_delta"] <= 1e-12
            and official_pose_comparison["max_rotation_delta_deg"] <= 1e-8
        ),
        "fixed_pose_ba_unchanged": max_pose_delta <= 1e-12,
        "bundle_adjustment_solution_usable": all(
            record["solution_usable"] for record in ba_records
        ),
        "bundle_adjustment_cost_decreased": all(
            record["final_cost"] < record["initial_cost"] for record in ba_records
        ),
        "intrinsics_stable_between_rounds": bool(
            abs(first_params[0] - last_params[0]) / last_params[0] <= 0.02
            and abs(first_params[3] - last_params[3]) <= 0.02
        ),
        "matching_support_sufficient": (
            database_stats["verified_pairs"] >= 1_000
            and database_stats["geometric_inliers"] >= 100_000
        ),
    }
    result = "pass" if all(gates.values()) else "fail"
    document = {
        "schema": SCHEMA,
        "schema_version": 2,
        "result": result,
        "source": {
            "image_archive": image_archive.name,
            "camera_log": camera_log.name,
            **input_hashes,
            "image_count": len(names),
            "image_size": [width, height],
            "camera_make": "Sony",
            "camera_model": "A7SM2",
            "exif_focal_length_mm": focal_mm,
            "exif_focal_length_35mm_equivalent_mm": focal_35mm,
        },
        "protocol": {
            "revision": CALIBRATION_REVISION,
            "seed": 42,
            "camera_model": "SIMPLE_RADIAL",
            "official_initial_intrinsics": {
                "focal_px": 0.7 * width,
                "cx_px": width / 2.0,
                "cy_px": height / 2.0,
                "radial_k1": 0.0,
            },
            "feature_max_image_size": args.max_image_size,
            "feature_max_count": 8192,
            "feature_extraction_threads": args.num_threads,
            "matching_triangulation_ba_threads": 1,
            "sequential_overlap": 10,
            "quadratic_overlap": True,
            "pose_policy": "all official camera-to-world poses held exactly constant",
            "optimized_parameters": ["shared_focal_length", "shared_radial_k1"],
            "fixed_parameters": ["all camera poses", "principal point"],
        },
        "software": {
            "pycolmap_version": pycolmap.__version__,
            "pycolmap_has_cuda": bool(pycolmap.has_cuda),
        },
        "database": database_stats,
        "rounds": rounds,
        "calibration": {
            "model": "SIMPLE_RADIAL",
            "width": width,
            "height": height,
            "params": [focal, cx, cy, radial],
            "focal_ratio_to_width": focal / width,
        },
        "final_stats": final_stats,
        "official_pose_comparison": official_pose_comparison,
        "quality_gates": gates,
        "limitations": [
            "The official log omits exact intrinsics; they are recovered from "
            "image tracks with official poses fixed.",
            "All 371 official images contribute to intrinsics calibration, so "
            "the 8 heldout views are not intrinsics-independent.",
            "Laser geometry is not used by this calibration step.",
        ],
    }
    write_json(args.output, document)
    print(
        f"[tnt-calibration] {result} f={focal:.6f}px k1={radial:.9f} "
        f"points={final_stats['points3D']} obs={final_stats['observations']} "
        f"error={final_stats['mean_reprojection_error_px']:.6f}px"
    )
    print(f"[tnt-calibration] wrote {args.output}")
    if result != "pass":
        failed = [name for name, passed in gates.items() if not passed]
        raise TntCalibrationError(f"Calibration quality gates failed: {failed}")
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--max-image-size", type=int, default=1600)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="CPU threads for per-image SIFT extraction; geometric stages stay single-threaded",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_image_size <= 0 or args.num_threads <= 0:
        raise SystemExit("Image size and thread count must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
