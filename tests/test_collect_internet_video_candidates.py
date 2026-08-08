import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.collect_internet_video_candidates import (
    archive_download_url,
    choose_archive_preview,
    choose_archive_source,
    write_html_index,
    write_json,
    write_overview,
)


class InternetVideoCandidateCollectorTests(unittest.TestCase):
    def test_archive_file_selection_prefers_original_hd_and_browser_proxy(self) -> None:
        files = [
            {
                "name": "scene.HD.mov",
                "source": "original",
                "width": "1920",
                "height": "1080",
                "size": "5000",
            },
            {
                "name": "scene.mp4",
                "source": "derivative",
                "width": "854",
                "height": "480",
                "size": "1000",
            },
            {
                "name": "scene_512kb.mp4",
                "source": "derivative",
                "width": "640",
                "height": "360",
                "size": "500",
            },
        ]
        self.assertEqual(choose_archive_source(files)["name"], "scene.HD.mov")
        self.assertEqual(choose_archive_preview(files)["name"], "scene.mp4")

    def test_archive_download_url_quotes_filename(self) -> None:
        url = archive_download_url("item-id", "A room #1.HD.mov")
        self.assertEqual(
            url,
            "https://archive.org/download/item-id/A%20room%20%231.HD.mov",
        )

    def test_json_writer_uses_null_for_non_finite_float(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            write_json(path, {"missing": float("nan")})
            raw = path.read_text()
            self.assertNotIn("NaN", raw)
            self.assertEqual(json.loads(raw), {"missing": None})

    def test_html_export_newline_remains_valid_javascript_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = {
                "collection_id": "test",
                "candidates": [],
            }
            write_html_index(root, document)
            page = (root / "index.html").read_text()
            self.assertIn("JSON.stringify(payload,null,2)+'\\n'", page)
            self.assertNotIn("JSON.stringify(payload,null,2)+'\n'", page)

    def test_overview_has_fixed_five_by_four_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = []
            for index in range(20):
                poster = root / f"poster_{index:02d}.jpg"
                Image.new("RGB", (640, 360), (index * 10, 80, 120)).save(poster)
                candidates.append({"poster": poster.name, "title": f"Candidate {index}"})
            write_overview(root, {"candidates": candidates})
            with Image.open(root / "overview.jpg") as overview:
                self.assertEqual(overview.size, (1320, 1218))


if __name__ == "__main__":
    unittest.main()
