#!/usr/bin/env python3
"""Collect internet indoor-video candidates and build a local review index."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

try:
    import imageio_ffmpeg
except ImportError:  # pragma: no cover - exercised only outside the project environment.
    imageio_ffmpeg = None

VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".ogv", ".mkv", ".m4v"}
CATEGORY_LABELS = {
    "historic_residential": "历史住宅",
    "religious_tall_space": "宗教/挑高空间",
    "hospitality": "餐饮/宴会",
    "cultural_venue": "文化场馆",
    "museum": "博物馆",
    "education_technical": "学校/技术中心",
    "education": "学校",
    "auditorium": "礼堂/表演空间",
    "public_service": "公共服务设施",
    "industrial": "工业设施",
    "lab_training": "实验/训练设施",
    "residential_care": "居住/照护空间",
    "historic_commercial": "历史商业空间",
    "specialized_commercial": "专业商业空间",
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(value), ensure_ascii=False, indent=2) + "\n")


def fetch_bytes(url: str, *, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "GenRecon candidate collector/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_json_snapshot(url: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        payload = fetch_bytes(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def strip_html(value: Any) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def archive_video_files(document: dict[str, Any]) -> list[dict[str, Any]]:
    videos = []
    for item in document.get("files", []):
        suffix = Path(str(item.get("name", ""))).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            continue
        if numeric(item.get("width")) <= 0 or numeric(item.get("height")) <= 0:
            continue
        videos.append(item)
    if not videos:
        raise RuntimeError("Archive item exposes no dimensioned video file")
    return videos


def choose_archive_source(videos: list[dict[str, Any]]) -> dict[str, Any]:
    originals = [item for item in videos if item.get("source") == "original"]
    pool = originals or videos
    return max(
        pool,
        key=lambda item: (
            numeric(item.get("width")) * numeric(item.get("height")),
            numeric(item.get("size")),
        ),
    )


def choose_archive_preview(videos: list[dict[str, Any]]) -> dict[str, Any]:
    mp4 = [
        item
        for item in videos
        if Path(str(item.get("name", ""))).suffix.lower() == ".mp4"
        and item.get("source") == "derivative"
    ]
    browser = [item for item in mp4 if numeric(item.get("width")) >= 640]
    pool = browser or mp4
    if not pool:
        return choose_archive_source(videos)
    return min(
        pool,
        key=lambda item: (
            abs(numeric(item.get("width")) - 854) + abs(numeric(item.get("height")) - 480),
            numeric(item.get("size")),
        ),
    )


def archive_download_url(identifier: str, filename: str) -> str:
    return (
        "https://archive.org/download/"
        + urllib.parse.quote(identifier, safe="")
        + "/"
        + urllib.parse.quote(filename, safe="")
    )


def download_file(url: str, destination: Path, expected_size: int | None = None) -> None:
    if destination.is_file() and (expected_size is None or destination.stat().st_size == expected_size):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    command = [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "5",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        url,
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        if expected_size is None or not partial.is_file() or partial.stat().st_size != expected_size:
            raise subprocess.CalledProcessError(result.returncode, command)
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"Download size mismatch for {url}: got {partial.stat().st_size}, expected {expected_size}"
        )
    os.replace(partial, destination)


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def ffmpeg_executable(explicit: str | None) -> str:
    if explicit:
        return explicit
    if imageio_ffmpeg is not None:
        return imageio_ffmpeg.get_ffmpeg_exe()
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    raise RuntimeError("No FFmpeg executable is available")


def transcode_preview(ffmpeg: str, source: Path, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    temporary = destination.with_name(destination.name + ".part.mp4")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        "scale=854:480:force_original_aspect_ratio=decrease:force_divisible_by=2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "25",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        "-threads",
        "4",
        str(temporary),
    ]
    subprocess.run(command, check=True)
    os.replace(temporary, destination)


def extract_frame(ffmpeg: str, video: Path, timestamp: float, output: Path, width: int) -> None:
    height = int(round(width * 9 / 16))
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-an",
        "-vf",
        (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
        ),
        "-q:v",
        "3",
        str(output),
    ]
    subprocess.run(command, check=True)


def timestamp_label(seconds: float) -> str:
    whole = max(0, int(round(seconds)))
    return f"{whole // 60:02d}:{whole % 60:02d}"


def generate_visuals(ffmpeg: str, preview: Path, duration: float, directory: Path) -> None:
    contact = directory / "contact.jpg"
    poster = directory / "poster.jpg"
    if contact.is_file() and poster.is_file():
        return
    frames_dir = directory / ".contact_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    timestamps = [duration * (index + 0.5) / 12 for index in range(12)]
    frame_paths = []
    for index, timestamp in enumerate(timestamps):
        frame = frames_dir / f"frame_{index:02d}.jpg"
        extract_frame(ffmpeg, preview, timestamp, frame, 320)
        frame_paths.append(frame)

    gap = 6
    cell_width, cell_height = 320, 180
    canvas = Image.new("RGB", (4 * cell_width + 5 * gap, 3 * cell_height + 4 * gap), "#20252b")
    draw = ImageDraw.Draw(canvas)
    for index, (frame_path, timestamp) in enumerate(zip(frame_paths, timestamps)):
        with Image.open(frame_path) as frame:
            image = frame.convert("RGB")
        x = gap + (index % 4) * (cell_width + gap)
        y = gap + (index // 4) * (cell_height + gap)
        canvas.paste(image, (x, y))
        label = timestamp_label(timestamp)
        draw.rectangle((x + 5, y + 5, x + 48, y + 21), fill="#111418")
        draw.text((x + 9, y + 7), label, fill="#ffffff")
    canvas.save(contact, quality=90, subsampling=0)

    poster_index = 4
    with Image.open(frame_paths[poster_index]) as selected:
        selected.convert("RGB").resize((640, 360), Image.Resampling.LANCZOS).save(
            poster, quality=91, subsampling=0
        )
    shutil.rmtree(frames_dir)


def commons_api_url(title: str) -> str:
    parameters = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "imageinfo|revisions",
        "titles": title,
        "iiprop": "timestamp|user|url|size|sha1|mime|mediatype|extmetadata",
        "rvprop": "ids|timestamp|sha1",
        "rvlimit": "1",
    }
    return "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(parameters)


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def collect_archive_candidate(
    seed: dict[str, Any], root: Path, directory: Path, ffmpeg: str
) -> dict[str, Any]:
    identifier = seed["archive_identifier"]
    metadata_url = f"https://archive.org/metadata/{urllib.parse.quote(identifier, safe='')}"
    metadata = fetch_json_snapshot(metadata_url, directory / "source_metadata.json")
    item = metadata.get("metadata") or {}
    if not item:
        raise RuntimeError(f"Missing Internet Archive item: {identifier}")
    videos = archive_video_files(metadata)
    raw_remote = choose_archive_source(videos)
    preview_remote = choose_archive_preview(videos)
    raw_name = str(raw_remote["name"])
    preview_name = str(preview_remote["name"])
    raw_suffix = Path(raw_name).suffix.lower() or ".bin"
    raw_path = directory / f"source{raw_suffix}"
    preview_path = directory / "preview.mp4"
    raw_url = archive_download_url(identifier, raw_name)
    preview_url = archive_download_url(identifier, preview_name)
    raw_size = int(numeric(raw_remote.get("size"))) or None
    preview_size = int(numeric(preview_remote.get("size"))) or None

    print(f"[{seed['candidate_id']}] download source: {raw_name}", flush=True)
    download_file(raw_url, raw_path, raw_size)
    if raw_name == preview_name and raw_suffix == ".mp4":
        link_or_copy(raw_path, preview_path)
    elif Path(preview_name).suffix.lower() == ".mp4":
        print(f"[{seed['candidate_id']}] download preview: {preview_name}", flush=True)
        download_file(preview_url, preview_path, preview_size)
    else:
        print(f"[{seed['candidate_id']}] transcode browser preview", flush=True)
        transcode_preview(ffmpeg, raw_path, preview_path)

    duration = numeric(preview_remote.get("length"), numeric(raw_remote.get("length")))
    if duration <= 0:
        raise RuntimeError(f"No duration metadata for {identifier}")
    generate_visuals(ffmpeg, preview_path, duration, directory)
    license_url = item.get("licenseurl") or ""
    rights = strip_html(item.get("rights"))
    license_label = license_url or rights or "未声明"
    return {
        **seed,
        "source_platform": "Internet Archive",
        "source_page": f"https://archive.org/details/{identifier}",
        "creator": strip_html(item.get("creator")),
        "catalog_date": item.get("date"),
        "public_date": item.get("publicdate"),
        "capture_date": None,
        "capture_date_status": "未验证；Archive date 不能自动视为拍摄日期",
        "license": license_label,
        "license_url": license_url or None,
        "license_status": "declared" if license_url or rights else "not_declared",
        "description": strip_html(item.get("description")),
        "duration_s": duration,
        "source_video": {
            "remote_filename": raw_name,
            "remote_url": raw_url,
            "local_path": relative_path(raw_path, root),
            "size_bytes": raw_path.stat().st_size,
            "width": int(numeric(raw_remote.get("width"))),
            "height": int(numeric(raw_remote.get("height"))),
            "format": raw_remote.get("format"),
            "sha256": sha256_file(raw_path),
        },
        "preview": {
            "remote_filename": preview_name if preview_name != raw_name else None,
            "remote_url": preview_url if preview_name != raw_name else None,
            "local_path": relative_path(preview_path, root),
            "size_bytes": preview_path.stat().st_size,
            "width": int(numeric(preview_remote.get("width"))),
            "height": int(numeric(preview_remote.get("height"))),
            "sha256": sha256_file(preview_path),
        },
        "poster": relative_path(directory / "poster.jpg", root),
        "contact_sheet": relative_path(directory / "contact.jpg", root),
        "metadata_snapshot": relative_path(directory / "source_metadata.json", root),
        "collection_status": "downloaded_and_previewed",
    }


def collect_commons_candidate(
    seed: dict[str, Any], root: Path, directory: Path, ffmpeg: str
) -> dict[str, Any]:
    api_url = commons_api_url(seed["commons_title"])
    metadata = fetch_json_snapshot(api_url, directory / "source_metadata.json")
    page = metadata["query"]["pages"][0]
    info = page["imageinfo"][0]
    ext = info.get("extmetadata") or {}
    raw_url = str(info["url"])
    raw_suffix = Path(urllib.parse.urlparse(raw_url).path).suffix.lower() or ".bin"
    raw_path = directory / f"source{raw_suffix}"
    existing = seed.get("existing_source")
    if existing:
        existing_path = (root / existing).resolve()
        if not existing_path.is_file():
            raise FileNotFoundError(existing_path)
        link_or_copy(existing_path, raw_path)
    else:
        print(f"[{seed['candidate_id']}] download Commons source", flush=True)
        download_file(raw_url, raw_path, int(info["size"]))
    preview_path = directory / "preview.mp4"
    print(f"[{seed['candidate_id']}] transcode browser preview", flush=True)
    transcode_preview(ffmpeg, raw_path, preview_path)
    duration = numeric(info.get("duration"))
    generate_visuals(ffmpeg, preview_path, duration, directory)
    license_name = (ext.get("LicenseShortName") or {}).get("value") or "未声明"
    license_url = (ext.get("LicenseUrl") or {}).get("value")
    capture_date = (ext.get("DateTimeOriginal") or {}).get("value")
    description = strip_html((ext.get("ImageDescription") or {}).get("value"))
    return {
        **seed,
        "source_platform": "Wikimedia Commons",
        "source_page": info.get("descriptionurl") or seed["source_page"],
        "creator": info.get("user") or strip_html((ext.get("Artist") or {}).get("value")),
        "catalog_date": info.get("timestamp"),
        "public_date": info.get("timestamp"),
        "capture_date": capture_date,
        "capture_date_status": "Commons DateTimeOriginal",
        "license": license_name,
        "license_url": license_url,
        "license_status": "declared" if license_url else "not_declared",
        "description": description,
        "duration_s": duration,
        "source_video": {
            "remote_filename": Path(urllib.parse.urlparse(raw_url).path).name,
            "remote_url": raw_url,
            "local_path": relative_path(raw_path, root),
            "size_bytes": raw_path.stat().st_size,
            "width": int(info["width"]),
            "height": int(info["height"]),
            "format": info.get("mime"),
            "sha256": sha256_file(raw_path),
            "mediawiki_sha1": info.get("sha1"),
        },
        "preview": {
            "remote_filename": None,
            "remote_url": None,
            "local_path": relative_path(preview_path, root),
            "size_bytes": preview_path.stat().st_size,
            "width": 854 if int(info["width"]) >= int(info["height"]) else 270,
            "height": 480,
            "sha256": sha256_file(preview_path),
        },
        "poster": relative_path(directory / "poster.jpg", root),
        "contact_sheet": relative_path(directory / "contact.jpg", root),
        "metadata_snapshot": relative_path(directory / "source_metadata.json", root),
        "collection_status": "downloaded_and_previewed",
    }


def collect_one(seed: dict[str, Any], root: Path, ffmpeg: str) -> dict[str, Any]:
    directory = root / "candidates" / seed["candidate_id"]
    directory.mkdir(parents=True, exist_ok=True)
    if seed["source_kind"] == "internet_archive":
        result = collect_archive_candidate(seed, root, directory, ffmpeg)
    elif seed["source_kind"] == "wikimedia_commons":
        result = collect_commons_candidate(seed, root, directory, ffmpeg)
    else:
        raise ValueError(f"Unsupported source kind: {seed['source_kind']}")
    write_json(directory / "candidate.json", result)
    print(
        f"[{seed['candidate_id']}] ready: {result['duration_s']:.1f}s, "
        f"source={result['source_video']['size_bytes'] / 1048576:.1f}MiB",
        flush=True,
    )
    return result


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def format_size(size: int) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GiB"
    return f"{size / 1024**2:.1f} MiB"


def write_overview(root: Path, document: dict[str, Any]) -> None:
    cell_width, image_height, caption_height, gap = 320, 180, 54, 8
    columns, rows = 4, 5
    canvas = Image.new(
        "RGB",
        (
            columns * cell_width + (columns + 1) * gap,
            rows * (image_height + caption_height) + (rows + 1) * gap,
        ),
        "#20252b",
    )
    draw = ImageDraw.Draw(canvas)
    for index, item in enumerate(document["candidates"]):
        x = gap + (index % columns) * (cell_width + gap)
        y = gap + (index // columns) * (image_height + caption_height + gap)
        with Image.open(root / item["poster"]) as poster:
            image = poster.convert("RGB").resize(
                (cell_width, image_height), Image.Resampling.LANCZOS
            )
        canvas.paste(image, (x, y))
        draw.text(
            (x + 5, y + 5),
            f"{index + 1:02d}",
            fill="#ffffff",
            stroke_width=2,
            stroke_fill="#111418",
        )
        title = "\n".join(textwrap.wrap(item["title"], width=44)[:2])
        draw.text((x, y + image_height + 5), title, fill="#f5f6f7")
    canvas.save(root / "overview.jpg", quality=91, subsampling=0)


def write_markdown_index(root: Path, document: dict[str, Any]) -> None:
    lines = [
        "# Internet indoor raw video candidates v1",
        "",
        "> 这是视觉预览候选池，尚未进行 COLMAP、静态性、隐私或剪辑 gate。`index.html` 用于逐条播放和记录人工决定。",
        "",
        "[20 场景总览](overview.jpg)",
        "",
        "| ID | 场景 | 类别 | 来源 | 原片 | 时长 | 许可字段 | 预览 |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for item in document["candidates"]:
        source = item["source_video"]
        candidate_dir = f"candidates/{item['candidate_id']}"
        lines.append(
            "| {id} | [{title}]({page}) | {category} | {platform} | "
            "{width}x{height}, {size} | {duration} | {license} | "
            "[video]({preview}) / [contact]({contact}) |".format(
                id=item["candidate_id"],
                title=item["title"].replace("|", "\\|"),
                page=item["source_page"],
                category=CATEGORY_LABELS.get(item["category"], item["category"]),
                platform=item["source_platform"],
                width=source["width"],
                height=source["height"],
                size=format_size(source["size_bytes"]),
                duration=format_duration(item["duration_s"]),
                license=str(item["license"]).replace("|", "\\|"),
                preview=f"{candidate_dir}/preview.mp4",
                contact=f"{candidate_dir}/contact.jpg",
            )
        )
    lines.extend(
        [
            "",
            "## 汇总",
            "",
            f"- 候选：{document['summary']['candidate_count']} 个不同场景。",
            f"- 原片：{format_size(document['summary']['source_bytes_total'])}。",
            f"- 浏览代理：{format_size(document['summary']['preview_bytes_total'])}。",
            f"- 明确许可字段：{document['summary']['declared_license_count']} 个。",
            "- 未声明许可不等于无版权；这里只保留作内部人工筛选。",
            "",
        ]
    )
    (root / "INDEX.md").write_text("\n".join(lines))


def write_html_index(root: Path, document: dict[str, Any]) -> None:
    embedded = json.dumps(document, ensure_ascii=False).replace("</", "<\\/")
    html_text = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Internet indoor raw candidates</title>
<style>
:root { color-scheme: light; --bg:#f2f4f5; --panel:#fff; --ink:#20252b; --muted:#687078; --line:#d8dde1; --accent:#176b57; --warn:#9a4d19; --reject:#9b3434; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; letter-spacing:0; }
header { position:sticky; top:0; z-index:20; background:#fff; border-bottom:1px solid var(--line); }
.top { max-width:1500px; margin:auto; padding:14px 20px 10px; display:flex; align-items:flex-end; justify-content:space-between; gap:20px; }
h1 { margin:0; font-size:20px; font-weight:680; letter-spacing:0; }
.sub { color:var(--muted); margin-top:2px; font-size:12px; }
.stats { display:flex; flex-wrap:wrap; gap:14px; color:var(--muted); font-variant-numeric:tabular-nums; }
.stats strong { color:var(--ink); }
.filters { max-width:1500px; margin:auto; padding:0 20px 12px; display:grid; grid-template-columns:minmax(220px,1fr) repeat(3,minmax(150px,220px)) auto; gap:8px; }
input,select,textarea,button { font:inherit; letter-spacing:0; }
input,select,textarea { width:100%; border:1px solid #c8ced3; background:#fff; color:var(--ink); border-radius:5px; padding:8px 10px; }
button { border:1px solid #aeb6bc; background:#fff; color:var(--ink); border-radius:5px; padding:8px 12px; cursor:pointer; }
button:hover { border-color:var(--accent); color:var(--accent); }
main { max-width:1500px; margin:0 auto; padding:16px 20px 40px; }
.grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; align-items:start; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:7px; overflow:hidden; min-width:0; }
.card.hidden { display:none; }
.card-head { padding:12px 14px 10px; display:grid; grid-template-columns:1fr auto; gap:10px; border-bottom:1px solid #e4e7e9; }
.card h2 { margin:0; font-size:15px; line-height:1.3; letter-spacing:0; overflow-wrap:anywhere; }
.id { color:var(--muted); font:12px ui-monospace,SFMono-Regular,monospace; margin-bottom:3px; }
.badges { display:flex; flex-wrap:wrap; justify-content:flex-end; gap:5px; align-content:start; }
.badge { border:1px solid #cbd1d5; border-radius:999px; padding:2px 7px; color:#4d555c; font-size:11px; white-space:nowrap; }
.badge.license { border-color:#89b7a9; color:#176b57; }
video { display:block; width:100%; aspect-ratio:16/9; background:#111; object-fit:contain; }
.meta { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px 12px; padding:10px 14px; border-bottom:1px solid #e4e7e9; font-variant-numeric:tabular-nums; }
.meta div { min-width:0; }
.meta span { display:block; color:var(--muted); font-size:10px; text-transform:uppercase; margin-bottom:1px; }
.meta strong { display:block; font-size:12px; font-weight:600; overflow-wrap:anywhere; }
.interest { margin:0; padding:10px 14px; color:#3e454b; min-height:48px; }
.links { padding:0 14px 10px; display:flex; flex-wrap:wrap; gap:12px; }
a { color:#0c6250; text-decoration:none; }
a:hover { text-decoration:underline; }
details { border-top:1px solid #e4e7e9; }
summary { cursor:pointer; padding:9px 14px; color:#475159; }
details img { width:100%; height:auto; display:block; border-top:1px solid #e4e7e9; }
.review { border-top:1px solid #dfe3e6; padding:10px 14px 12px; display:grid; grid-template-columns:170px 1fr; gap:8px; }
.review textarea { min-height:38px; resize:vertical; }
.decision-keep { border-color:#4d9b81; color:#165f4e; }
.decision-reject { border-color:#c98787; color:var(--reject); }
.decision-maybe { border-color:#d4a66f; color:var(--warn); }
.empty { display:none; padding:60px 20px; text-align:center; color:var(--muted); }
@media (max-width:950px) { .grid{grid-template-columns:1fr}.filters{grid-template-columns:1fr 1fr}.top{align-items:flex-start;flex-direction:column}.meta{grid-template-columns:1fr 1fr}.review{grid-template-columns:1fr} }
@media (max-width:520px) { .filters{grid-template-columns:1fr}.top,main,.filters{padding-left:10px;padding-right:10px}.stats{gap:8px}.card-head{grid-template-columns:1fr}.badges{justify-content:flex-start} }
</style>
</head>
<body>
<header>
  <div class="top">
    <div><h1>Internet indoor raw candidates</h1><div class="sub">视觉候选池 v1 · 未执行 COLMAP / 隐私 / 静态性 gate</div></div>
    <div class="stats"><span>候选 <strong id="total"></strong></span><span>显示 <strong id="visible"></strong></span><span>保留 <strong id="kept"></strong></span><span>待定 <strong id="maybe"></strong></span><span>拒绝 <strong id="rejected"></strong></span></div>
  </div>
  <div class="filters">
    <input id="search" type="search" placeholder="搜索场景、类别、来源">
    <select id="category"><option value="">全部类别</option></select>
    <select id="platform"><option value="">全部来源</option></select>
    <select id="reviewFilter"><option value="">全部审查状态</option><option value="unreviewed">未审查</option><option value="keep">保留</option><option value="maybe">待定</option><option value="reject">拒绝</option></select>
    <button id="export" type="button">导出审查结果</button>
  </div>
</header>
<main><section id="grid" class="grid"></section><div id="empty" class="empty">没有匹配候选</div></main>
<script>
const data = __EMBEDDED__;
const labels = __CATEGORY_LABELS__;
const storageKey = 'genrecon.raw-candidates-v1.review';
let reviews = {};
try { reviews = JSON.parse(localStorage.getItem(storageKey) || '{}'); } catch (_) { reviews = {}; }
const esc = value => String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const duration = seconds => { const total=Math.round(seconds); return `${Math.floor(total/60)}:${String(total%60).padStart(2,'0')}`; };
const size = bytes => bytes >= 1073741824 ? `${(bytes/1073741824).toFixed(2)} GiB` : `${(bytes/1048576).toFixed(1)} MiB`;
const category = document.getElementById('category');
const platform = document.getElementById('platform');
[...new Set(data.candidates.map(x => x.category))].sort().forEach(x => category.insertAdjacentHTML('beforeend', `<option value="${esc(x)}">${esc(labels[x] || x)}</option>`));
[...new Set(data.candidates.map(x => x.source_platform))].sort().forEach(x => platform.insertAdjacentHTML('beforeend', `<option value="${esc(x)}">${esc(x)}</option>`));
function reviewFor(id) { return reviews[id] || {decision:'unreviewed',notes:''}; }
function decisionClass(value) { return value === 'keep' ? 'decision-keep' : value === 'reject' ? 'decision-reject' : value === 'maybe' ? 'decision-maybe' : ''; }
function render() {
  const grid = document.getElementById('grid');
  grid.innerHTML = data.candidates.map((x, index) => {
    const review = reviewFor(x.candidate_id);
    const declared = x.license_status === 'declared';
    return `<article class="card" data-id="${esc(x.candidate_id)}" data-search="${esc((x.title+' '+x.scene_label+' '+x.category+' '+x.source_platform).toLowerCase())}" data-category="${esc(x.category)}" data-platform="${esc(x.source_platform)}">
      <div class="card-head"><div><div class="id">${String(index+1).padStart(2,'0')} · ${esc(x.candidate_id)}</div><h2>${esc(x.title)}</h2></div><div class="badges"><span class="badge">${esc(labels[x.category] || x.category)}</span><span class="badge ${declared?'license':''}">${declared?'许可已声明':'许可未声明'}</span></div></div>
      <video controls preload="metadata" poster="${esc(x.poster)}" src="${esc(x.preview.local_path)}"></video>
      <div class="meta"><div><span>来源</span><strong>${esc(x.source_platform)}</strong></div><div><span>原片</span><strong>${x.source_video.width}x${x.source_video.height} · ${size(x.source_video.size_bytes)}</strong></div><div><span>时长</span><strong>${duration(x.duration_s)}</strong></div><div><span>日期</span><strong>${esc(x.capture_date || x.catalog_date || '未知')}</strong></div></div>
      <p class="interest">${esc(x.interest)}</p>
      <div class="links"><a href="${esc(x.source_page)}" target="_blank" rel="noreferrer">来源页</a><a href="${esc(x.source_video.local_path)}">本地原片</a><a href="${esc(x.metadata_snapshot)}">元数据</a><span>${esc(x.license)}</span></div>
      <details><summary>12 帧联系表</summary><img loading="lazy" src="${esc(x.contact_sheet)}" alt="${esc(x.title)} contact sheet"></details>
      <div class="review"><select class="decision ${decisionClass(review.decision)}" data-id="${esc(x.candidate_id)}"><option value="unreviewed" ${review.decision==='unreviewed'?'selected':''}>未审查</option><option value="keep" ${review.decision==='keep'?'selected':''}>保留</option><option value="maybe" ${review.decision==='maybe'?'selected':''}>待定</option><option value="reject" ${review.decision==='reject'?'selected':''}>拒绝</option></select><textarea class="notes" data-id="${esc(x.candidate_id)}" placeholder="场景、人物、剪辑、可用时间段备注">${esc(review.notes)}</textarea></div>
    </article>`;
  }).join('');
  grid.querySelectorAll('.decision').forEach(node => node.addEventListener('change', event => { const id=event.target.dataset.id; reviews[id]={...reviewFor(id),decision:event.target.value}; localStorage.setItem(storageKey,JSON.stringify(reviews)); event.target.className='decision '+decisionClass(event.target.value); applyFilters(); }));
  grid.querySelectorAll('.notes').forEach(node => node.addEventListener('input', event => { const id=event.target.dataset.id; reviews[id]={...reviewFor(id),notes:event.target.value}; localStorage.setItem(storageKey,JSON.stringify(reviews)); }));
  applyFilters();
}
function applyFilters() {
  const query=document.getElementById('search').value.trim().toLowerCase(); const cat=category.value; const source=platform.value; const status=document.getElementById('reviewFilter').value; let visible=0;
  document.querySelectorAll('.card').forEach(card => { const review=reviewFor(card.dataset.id); const show=(!query||card.dataset.search.includes(query))&&(!cat||card.dataset.category===cat)&&(!source||card.dataset.platform===source)&&(!status||review.decision===status); card.classList.toggle('hidden',!show); if(show) visible++; });
  const values=data.candidates.map(x=>reviewFor(x.candidate_id).decision); document.getElementById('total').textContent=data.candidates.length; document.getElementById('visible').textContent=visible; document.getElementById('kept').textContent=values.filter(x=>x==='keep').length; document.getElementById('maybe').textContent=values.filter(x=>x==='maybe').length; document.getElementById('rejected').textContent=values.filter(x=>x==='reject').length; document.getElementById('empty').style.display=visible?'none':'block';
}
['search','category','platform','reviewFilter'].forEach(id => document.getElementById(id).addEventListener(id==='search'?'input':'change',applyFilters));
document.getElementById('export').addEventListener('click', () => { const payload={schema:'genrecon.raw-candidate-review',schema_version:1,collection_id:data.collection_id,exported_utc:new Date().toISOString(),reviews:data.candidates.map(x=>({candidate_id:x.candidate_id,title:x.title,...reviewFor(x.candidate_id)}))}; const blob=new Blob([JSON.stringify(payload,null,2)+'\\n'],{type:'application/json'}); const link=document.createElement('a'); link.href=URL.createObjectURL(blob); link.download='raw-candidate-review.json'; link.click(); setTimeout(()=>URL.revokeObjectURL(link.href),1000); });
render();
</script>
</body>
</html>
"""
    html_text = html_text.replace("__EMBEDDED__", embedded).replace(
        "__CATEGORY_LABELS__", json.dumps(CATEGORY_LABELS, ensure_ascii=False)
    )
    (root / "index.html").write_text(html_text)


def validate_collection(root: Path, document: dict[str, Any]) -> None:
    if len(document["candidates"]) != 20:
        raise RuntimeError(f"Expected 20 candidates, got {len(document['candidates'])}")
    ids = [item["candidate_id"] for item in document["candidates"]]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Candidate IDs are not unique")
    required_paths = []
    for item in document["candidates"]:
        required_paths.extend(
            [
                item["source_video"]["local_path"],
                item["preview"]["local_path"],
                item["poster"],
                item["contact_sheet"],
                item["metadata_snapshot"],
            ]
        )
    missing = [path for path in required_paths if not (root / path).is_file()]
    if missing:
        raise RuntimeError(f"Missing collection artifacts: {missing[:5]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--workers", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()
    seed_path = (args.seed or root / "seed.json").resolve()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    seed_document = json.loads(seed_path.read_text())
    candidates = seed_document["candidates"]
    if len(candidates) != seed_document["policy"]["target_count"]:
        raise SystemExit("Seed count does not match policy target_count")
    root.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_executable(args.ffmpeg)
    print(f"[collector] root={root}", flush=True)
    print(f"[collector] candidates={len(candidates)} workers={args.workers}", flush=True)

    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(collect_one, item, root, ffmpeg): item for item in candidates}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                results[result["candidate_id"]] = result
            except Exception as exc:
                failures[item["candidate_id"]] = f"{type(exc).__name__}: {exc}"
                print(f"[{item['candidate_id']}] FAILED: {failures[item['candidate_id']]}", flush=True)
    if failures:
        write_json(root / "collection_failures.json", failures)
        raise SystemExit(f"Collection failed for {len(failures)} candidate(s)")

    ordered = [results[item["candidate_id"]] for item in candidates]
    generated = datetime.now(timezone.utc).isoformat()
    document = {
        "schema": "genrecon.internet-video-candidate-index",
        "schema_version": 1,
        "collection_id": seed_document["collection_id"],
        "generated_utc": generated,
        "policy": seed_document["policy"],
        "summary": {
            "candidate_count": len(ordered),
            "category_count": len({item["category"] for item in ordered}),
            "source_platform_count": len({item["source_platform"] for item in ordered}),
            "declared_license_count": sum(item["license_status"] == "declared" for item in ordered),
            "source_bytes_total": sum(item["source_video"]["size_bytes"] for item in ordered),
            "preview_bytes_total": sum(item["preview"]["size_bytes"] for item in ordered),
            "duration_s_total": sum(item["duration_s"] for item in ordered),
        },
        "candidates": ordered,
    }
    write_json(root / "index.json", document)
    write_overview(root, document)
    write_markdown_index(root, document)
    write_html_index(root, document)
    validate_collection(root, document)
    print(json.dumps(document["summary"], indent=2), flush=True)
    print(f"[collector] index={root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
