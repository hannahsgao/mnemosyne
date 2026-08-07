from __future__ import annotations

import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
import tempfile
import threading
import unittest

from PIL import Image

from pipeline.met_visual import prepare_met_visual_subset


class _ImageHandler(BaseHTTPRequestHandler):
    payload = b""
    requests = 0

    def do_GET(self) -> None:  # noqa: N802
        type(self).requests += 1
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, format: str, *args: object) -> None:
        return


class MetVisualSubsetTests(unittest.TestCase):
    def setUp(self) -> None:
        image = Image.new("RGB", (120, 80), "navy")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image.close()
        _ImageHandler.payload = buffer.getvalue()
        _ImageHandler.requests = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _ImageHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_prepares_only_public_domain_met_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            met = root / "met"
            met.mkdir()
            fields = (
                "artwork_id",
                "source_id",
                "title",
                "date_start",
                "date_end",
                "public_domain",
            )
            with (met / "corpus.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "artwork_id": "MET_1",
                            "source_id": "1",
                            "title": "One",
                            "date_start": "1800",
                            "date_end": "1800",
                            "public_domain": "True",
                        },
                        {
                            "artwork_id": "MET_2",
                            "source_id": "2",
                            "title": "Two",
                            "date_start": "1900",
                            "date_end": "1900",
                            "public_domain": "True",
                        },
                        {
                            "artwork_id": "MET_3",
                            "source_id": "3",
                            "title": "Private",
                            "date_start": "1900",
                            "date_end": "1900",
                            "public_domain": "False",
                        },
                    ]
                )

            source = root / "ArtiFact_clean.csv"
            image_url = f"http://127.0.0.1:{self.server.server_port}/image.png"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("object_ID", "image_url"),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"object_ID": "MET_1", "image_url": image_url},
                        {"object_ID": "MET_2", "image_url": image_url},
                        {"object_ID": "MET_3", "image_url": image_url},
                        {"object_ID": "AIC_4", "image_url": image_url},
                    ]
                )

            output = root / "prepared" / "met-visual.csv"
            manifest = prepare_met_visual_subset(
                met,
                source,
                output,
                root / "prepared" / "images",
                sample_size=2,
                source_revision="pinned-revision",
                workers=2,
                max_dimension=64,
            )

            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["artwork_id"] for row in rows}, {"MET_1", "MET_2"})
            self.assertTrue(all(row["public_domain"] == "True" for row in rows))
            self.assertTrue(all(row["image_use_permitted"] == "True" for row in rows))
            self.assertTrue(all((output.parent / row["image_path"]).is_file() for row in rows))
            self.assertEqual(manifest["selection"]["prepared_rows"], 2)
            self.assertTrue(output.with_suffix(".manifest.json").is_file())

    def test_stream_manifest_does_not_download_or_store_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            met = root / "met"
            met.mkdir()
            with (met / "corpus.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("artwork_id", "source_id", "title", "date_start", "public_domain"),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "artwork_id": "MET_1",
                        "source_id": "1",
                        "title": "One",
                        "date_start": "1800",
                        "public_domain": "True",
                    }
                )
            source = root / "ArtiFact_clean.csv"
            image_url = f"http://127.0.0.1:{self.server.server_port}/image.png"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("object_ID", "image_url"), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerow({"object_ID": "MET_1", "image_url": image_url})

            output = root / "met-visual.csv"
            manifest = prepare_met_visual_subset(
                met,
                source,
                output,
                sample_size=0,
                source_revision="pinned-revision",
            )

            with output.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["image_url"], image_url)
            self.assertEqual(row["image_path"], "")
            self.assertEqual(row["image_use_permitted"], "True")
            self.assertEqual(_ImageHandler.requests, 0)
            self.assertEqual(manifest["images"]["storage"], "stream-at-embed-time")
            self.assertEqual(manifest["images"]["stored_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
