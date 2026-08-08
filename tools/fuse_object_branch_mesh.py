#!/usr/bin/env python3
"""Replace baseline faces inside registered object boxes with an object-branch mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def inside_boxes(points: np.ndarray, boxes: list[list[float]], margin: float) -> np.ndarray:
    result = np.zeros(len(points), dtype=bool)
    for x0, x1, y0, y1, z0, z1 in boxes:
        result |= (
            (points[:, 0] >= x0 - margin)
            & (points[:, 0] <= x1 + margin)
            & (points[:, 1] >= y0 - margin)
            & (points[:, 1] <= y1 + margin)
            & (points[:, 2] >= z0 - margin)
            & (points[:, 2] <= z1 + margin)
        )
    return result


def classify_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    boxes: list[list[float]],
    margin: float,
    batch_size: int,
) -> tuple[np.ndarray, int]:
    selected = np.zeros(len(faces), dtype=bool)
    crossing = 0
    for start in range(0, len(faces), batch_size):
        end = min(start + batch_size, len(faces))
        triangles = vertices[faces[start:end]]
        selected[start:end] = inside_boxes(triangles.mean(axis=1), boxes, margin)
        vertex_inside = inside_boxes(triangles.reshape(-1, 3), boxes, margin).reshape(-1, 3)
        crossing += int(np.count_nonzero(vertex_inside.any(axis=1) & ~vertex_inside.all(axis=1)))
    return selected, crossing


def restrict_removal_to_supported_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    candidate: np.ndarray,
    object_surface_vertices: np.ndarray,
    max_distance: float,
    batch_size: int,
) -> tuple[np.ndarray, int]:
    """Keep baseline fallback faces where the object branch has no nearby surface."""
    if max_distance <= 0:
        raise ValueError("baseline removal support distance must be positive")
    tree = cKDTree(object_surface_vertices)
    candidate_rows = np.flatnonzero(candidate)
    supported = np.zeros(len(faces), dtype=bool)
    for start in range(0, len(candidate_rows), batch_size):
        rows = candidate_rows[start : start + batch_size]
        centroids = vertices[faces[rows]].mean(axis=1)
        distance, _ = tree.query(centroids, k=1, workers=-1)
        supported[rows] = distance <= max_distance
    return supported, int(candidate.sum() - supported.sum())


def vertex_rgba(mesh: trimesh.Trimesh) -> np.ndarray:
    colors = np.asarray(mesh.visual.vertex_colors)
    if len(colors) != len(mesh.vertices):
        return np.full((len(mesh.vertices), 4), 255, dtype=np.uint8)
    if colors.shape[1] == 3:
        colors = np.concatenate([colors, np.full((len(colors), 1), 255, dtype=np.uint8)], axis=1)
    return colors[:, :4].astype(np.uint8, copy=False)


def write_binary_ply(
    path: Path,
    vertices: np.ndarray,
    colors: np.ndarray,
    baseline_faces: np.ndarray,
    object_faces: np.ndarray,
    object_vertex_offset: int,
    batch_size: int = 1_000_000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("alpha", "u1"),
        ]
    )
    face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
    face_count = len(baseline_faces) + len(object_faces)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n"
        f"element face {face_count}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        for start in range(0, len(vertices), batch_size):
            end = min(start + batch_size, len(vertices))
            block = np.empty(end - start, dtype=vertex_dtype)
            block["x"] = vertices[start:end, 0]
            block["y"] = vertices[start:end, 1]
            block["z"] = vertices[start:end, 2]
            block["red"] = colors[start:end, 0]
            block["green"] = colors[start:end, 1]
            block["blue"] = colors[start:end, 2]
            block["alpha"] = colors[start:end, 3]
            block.tofile(handle)
        for faces, offset in ((baseline_faces, 0), (object_faces, object_vertex_offset)):
            for start in range(0, len(faces), batch_size):
                end = min(start + batch_size, len(faces))
                block = np.empty(end - start, dtype=face_dtype)
                block["count"] = 3
                block["indices"] = faces[start:end] + offset
                block.tofile(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--object", type=Path, required=True)
    parser.add_argument("--box", type=float, nargs=6, action="append", required=True)
    parser.add_argument("--margin", type=float, default=0.015)
    parser.add_argument(
        "--baseline-removal-support-distance",
        type=float,
        help="Remove an in-box baseline face only when the object branch has a surface vertex "
        "within this distance; otherwise retain it as a coverage fallback.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=500_000)
    args = parser.parse_args()
    for box in args.box:
        if box[0] > box[1] or box[2] > box[3] or box[4] > box[5]:
            raise ValueError("box minima must not exceed maxima")

    baseline = trimesh.load(args.baseline, force="mesh", process=False)
    object_mesh = trimesh.load(args.object, force="mesh", process=False)
    baseline_vertices = np.asarray(baseline.vertices, dtype=np.float32)
    baseline_faces = np.asarray(baseline.faces, dtype=np.int32)
    object_vertices = np.asarray(object_mesh.vertices, dtype=np.float32)
    object_faces = np.asarray(object_mesh.faces, dtype=np.int32)
    baseline_owned, baseline_crossing = classify_faces(
        baseline_vertices, baseline_faces, args.box, args.margin, args.batch_size
    )
    baseline_owned_candidates = int(baseline_owned.sum())
    object_owned, object_crossing = classify_faces(
        object_vertices, object_faces, args.box, args.margin, args.batch_size
    )
    selected_object_faces = object_faces[object_owned]
    used_object_vertices, inverse = np.unique(selected_object_faces.reshape(-1), return_inverse=True)
    fallback_faces = 0
    if args.baseline_removal_support_distance is not None:
        baseline_owned, fallback_faces = restrict_removal_to_supported_faces(
            baseline_vertices,
            baseline_faces,
            baseline_owned,
            object_vertices[used_object_vertices],
            args.baseline_removal_support_distance,
            args.batch_size,
        )
    kept_baseline_faces = baseline_faces[~baseline_owned]
    compact_object_faces = inverse.reshape(-1, 3).astype(np.int32)
    compact_object_vertices = object_vertices[used_object_vertices]
    vertices = np.concatenate([baseline_vertices, compact_object_vertices], axis=0)
    colors = np.concatenate(
        [vertex_rgba(baseline), vertex_rgba(object_mesh)[used_object_vertices]], axis=0
    )
    write_binary_ply(
        args.output,
        vertices,
        colors,
        kept_baseline_faces,
        compact_object_faces,
        len(baseline_vertices),
    )
    summary = {
        "schema": "genrecon.object-branch-fusion",
        "baseline_path": str(args.baseline),
        "object_path": str(args.object),
        "output": str(args.output),
        "boxes_world": args.box,
        "margin_m": args.margin,
        "baseline": {
            "vertices": len(baseline_vertices),
            "faces": len(baseline_faces),
            "faces_in_ownership_boxes": baseline_owned_candidates,
            "faces_removed": int(baseline_owned.sum()),
            "fallback_faces_retained": fallback_faces,
            "removal_support_distance_m": args.baseline_removal_support_distance,
            "boundary_crossing_faces": baseline_crossing,
        },
        "object_branch": {
            "vertices": len(object_vertices),
            "faces": len(object_faces),
            "faces_selected": int(object_owned.sum()),
            "vertices_selected": len(compact_object_vertices),
            "boundary_crossing_faces": object_crossing,
        },
        "fused": {
            "vertices": len(vertices),
            "faces": len(kept_baseline_faces) + len(compact_object_faces),
            "file_bytes": args.output.stat().st_size,
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
