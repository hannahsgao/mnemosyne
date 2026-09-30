from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline import __version__
from pipeline.build import CorpusBuildError, build_corpus
from pipeline.cma_visual import (
    CMA_API_URL,
    CMA_CC0_URI,
    CMA_IMAGE_HOST,
    CMA_IMAGE_INPUT_POLICY,
    CMA_OPEN_DATA_URL,
    CMA_TIMELINE_DATE_SANITY_POLICY,
    CMA_TIMELINE_MAX_SPAN_YEARS,
    CMA_TIMELINE_MAX_YEAR,
    CMA_TIMELINE_MIN_YEAR,
    PHYSICAL_OBJECT_GROUPING_POLICY,
    VISUAL_SUBSET_SCHEMA_VERSION,
    _cacheable_availability,
    _validate_cma_image_url,
    prepare_cma_visual_subset,
)


REVISION = "1ac690143702714ba0481f89ae6faffde23f7399"


def _record(identifier: int, **updates: object) -> dict[str, object]:
    accession = f"2000.{identifier}"
    record: dict[str, object] = {
        "id": identifier,
        "accession_number": accession,
        "share_license_status": "CC0",
        "copyright": None,
        "record_type": "object",
        "title": f"Artwork {identifier}",
        "creation_date": "c. 1765" if identifier == 1 else "1900",
        "creation_date_earliest": 1760 if identifier == 1 else 1900,
        "creation_date_latest": 1770 if identifier == 1 else 1900,
        "creators": [
            {
                "description": "Unused Attribution",
                "use_in_caption": False,
            },
            {
                "description": f"Artist {identifier} (Example, 1800–1900)",
                "use_in_caption": True,
            },
        ],
        "artists_tags": ["painter", "painter", "example"],
        "culture": ["Example culture", "Second culture"],
        "technique": "oil on canvas",
        "department": "Paintings",
        "collection": "European Painting",
        "type": "Painting",
        "creditline": "Gift of Example Donor",
        "external_resources": {
            "wikidata": [f"https://www.wikidata.org/wiki/Q{identifier}"]
        },
        "url": f"https://clevelandart.org/art/{accession}",
        "images": {
            "web": {
                "url": (
                    f"https://{CMA_IMAGE_HOST}/{accession}/{accession}_web.jpg"
                ),
                "width": "900",
                "height": "700",
                "filesize": "12345",
            }
        },
    }
    record.update(updates)
    return record


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class CmaVisualSubsetTests(unittest.TestCase):
    def test_prepares_bulk_data_json_with_strict_rights_and_canonical_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data.json"
            invalid_host = _record(5)
            invalid_host["images"] = {
                "web": {
                    "url": "https://images.example.test/5_web.jpg",
                    "width": "900",
                    "height": "700",
                }
            }
            missing_image = _record(6)
            missing_image["images"] = {"web": None}
            _write_json(
                source,
                [
                    _record(2),
                    _record(1),
                    _record(3, share_license_status="Copyrighted"),
                    _record(4, copyright="© Example Estate"),
                    invalid_host,
                    missing_image,
                    _record(7, record_type="creator"),
                    _record(
                        8,
                        creation_date=None,
                        creation_date_earliest=None,
                        creation_date_latest=None,
                    ),
                ],
            )
            output = root / "prepared" / "cma.csv"

            manifest = prepare_cma_visual_subset(
                source,
                output,
                source_revision=REVISION,
                preflight=False,
            )

            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["artwork_id"] for row in rows], ["CMA_1", "CMA_2"])
            first = rows[0]
            self.assertEqual(first["physical_object_id"], "CMA_1")
            self.assertEqual(first["institution"], "cma")
            self.assertEqual(first["source_id"], "1")
            self.assertEqual(first["source_dataset_version"], REVISION)
            self.assertEqual(first["source_record_url"], "https://clevelandart.org/art/2000.1")
            self.assertEqual(first["artist"], "Artist 1 (Example, 1800–1900)")
            self.assertEqual(first["culture"], "Example culture; Second culture")
            self.assertEqual(first["tags"], "painter; example")
            self.assertEqual(first["object_type"], "Painting")
            self.assertEqual(first["classification"], "European Painting")
            self.assertEqual(first["date_display"], "c. 1765")
            self.assertEqual(first["date_start"], "1760")
            self.assertEqual(first["date_end"], "1770")
            self.assertEqual(first["date_qualifier"], "circa")
            self.assertEqual(first["metadata_license"], CMA_CC0_URI)
            self.assertEqual(first["image_rights_uri"], CMA_CC0_URI)
            self.assertEqual(first["public_domain"], "True")
            self.assertEqual(first["image_use_permitted"], "True")
            self.assertEqual(first["image_input_policy"], CMA_IMAGE_INPUT_POLICY)
            self.assertEqual(first["image_width"], "900")
            self.assertEqual(first["image_height"], "700")
            self.assertEqual(first["object_wikidata_url"], "https://www.wikidata.org/wiki/Q1")

            self.assertEqual(manifest["schema_version"], VISUAL_SUBSET_SCHEMA_VERSION)
            self.assertEqual(manifest["builder_version"], __version__)
            self.assertEqual(manifest["source"]["kind"], "cma-open-access-data-json")
            self.assertEqual(manifest["source"]["url"], CMA_OPEN_DATA_URL)
            self.assertEqual(manifest["source"]["revision"], REVISION)
            self.assertEqual(
                manifest["source"]["snapshot"]["sha256"],
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            self.assertEqual(manifest["source"]["snapshot"]["rows"], 8)
            self.assertEqual(manifest["selection"]["prepared_rows"], 2)
            self.assertEqual(manifest["selection"]["rejected_not_cc0"], 1)
            self.assertEqual(manifest["selection"]["rejected_nonblank_copyright"], 1)
            self.assertEqual(manifest["selection"]["rejected_invalid_web_image"], 1)
            self.assertEqual(manifest["selection"]["rejected_missing_web_image"], 1)
            self.assertEqual(manifest["selection"]["rejected_non_object_record"], 1)
            self.assertEqual(manifest["selection"]["rejected_without_creation_date"], 1)
            self.assertEqual(manifest["placeholder_basenames"], [])
            self.assertEqual(
                manifest["physical_object_grouping"]["policy"],
                PHYSICAL_OBJECT_GROUPING_POLICY,
            )
            self.assertEqual(manifest["images"]["stored_bytes"], 0)
            self.assertFalse(output.with_suffix(".incomplete.json").exists())

            corpus = root / "corpus"
            corpus_manifest = build_corpus(
                output,
                corpus,
                corpus_version="cma-fixture-v1",
                source_revision=REVISION,
                source_url=CMA_OPEN_DATA_URL,
                metadata_license=CMA_CC0_URI,
                source_kind="cma-open-access-data-json",
                source_payloads=(source, output.with_suffix(".manifest.json")),
            )
            self.assertEqual(corpus_manifest["counts"]["dated_rows"], 2)
            with (corpus / "images.manifest.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                image_rows = list(csv.DictReader(handle))
            self.assertTrue(
                all(row["permission_status"] == "public-domain" for row in image_rows)
            )
            self.assertTrue(
                all(
                    row["image_input_policy"] == CMA_IMAGE_INPUT_POLICY
                    for row in image_rows
                )
            )

    def test_accepts_api_search_and_single_artwork_snapshot_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                (
                    "search",
                    {"info": {"total": 42}, "data": [_record(1)]},
                    "api-artworks-response",
                    42,
                ),
                ("single", {"data": _record(2)}, "api-artwork-response", None),
                ("direct", _record(3), "direct-artwork-object", None),
            )
            for name, payload, expected_format, reported_total in cases:
                with self.subTest(name=name):
                    source = root / f"{name}.json"
                    output = root / f"{name}.csv"
                    _write_json(source, payload)
                    revision = hashlib.sha256(source.read_bytes()).hexdigest()
                    manifest = prepare_cma_visual_subset(
                        source,
                        output,
                        source_revision=revision,
                        preflight=False,
                    )
                    snapshot = manifest["source"]["snapshot"]
                    self.assertEqual(snapshot["format"], expected_format)
                    self.assertEqual(snapshot["rows"], 1)
                    self.assertEqual(manifest["source"]["url"], CMA_API_URL)
                    self.assertEqual(manifest["source"]["revision"], revision)
                    self.assertEqual(
                        manifest["source"]["revision_kind"], "snapshot-sha256"
                    )
                    if reported_total is None:
                        self.assertNotIn("reported_total", snapshot)
                    else:
                        self.assertEqual(snapshot["reported_total"], reported_total)

    def test_image_gate_rejects_insecure_untrusted_or_mutable_urls(self) -> None:
        _validate_cma_image_url(
            "https://openaccess-cdn.clevelandart.org/2000.1/2000.1_web.jpg"
        )
        invalid = (
            "http://openaccess-cdn.clevelandart.org/1_web.jpg",
            "https://images.example.test/1_web.jpg",
            "https://user@openaccess-cdn.clevelandart.org/1_web.jpg",
            "https://openaccess-cdn.clevelandart.org:444/1_web.jpg",
            "https://openaccess-cdn.clevelandart.org/1_web.jpg?token=mutable",
            "https://openaccess-cdn.clevelandart.org/1_web.tif",
        )
        for url in invalid:
            with self.subTest(url=url), self.assertRaises(ValueError):
                _validate_cma_image_url(url)

    def test_include_undated_keeps_zero_weight_rows_without_inventing_dates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data.json"
            output = root / "cma.csv"
            _write_json(
                source,
                [
                    _record(
                        1,
                        creation_date="Unknown",
                        creation_date_earliest=None,
                        creation_date_latest=None,
                    )
                ],
            )

            with self.assertRaisesRegex(CorpusBuildError, "no eligible"):
                prepare_cma_visual_subset(
                    source,
                    output,
                    source_revision=REVISION,
                    preflight=False,
                )
            self.assertTrue(output.with_suffix(".incomplete.json").is_file())

            manifest = prepare_cma_visual_subset(
                source,
                output,
                source_revision=REVISION,
                preflight=False,
                include_undated=True,
            )
            with output.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["date_display"], "")
            self.assertEqual(row["date_start"], "")
            self.assertEqual(row["date_end"], "")
            self.assertEqual(row["date_qualifier"], "unknown")
            self.assertEqual(row["date_parse_method"], "cma_unknown_creation_date")
            self.assertEqual(manifest["selection"]["objects_unknown_creation_date"], 1)
            self.assertEqual(manifest["selection"]["strict_timeline_candidates"], 0)

    def test_quarantines_untrustworthy_timeline_dates_without_clipping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data.json"
            output = root / "cma.csv"
            _write_json(
                source,
                [
                    _record(1),
                    _record(
                        2,
                        creation_date="before c. 10,000 BCE",
                        creation_date_earliest=-350_000,
                        creation_date_latest=-30_000,
                    ),
                    _record(
                        3,
                        creation_date="1920",
                        creation_date_earliest=1920,
                        creation_date_latest=10_920,
                    ),
                    _record(
                        4,
                        creation_date="1980–1801 BCE",
                        creation_date_earliest=-1980,
                        creation_date_latest=1801,
                    ),
                    _record(
                        5,
                        creation_date="c. 120,000–30,000 BCE",
                        creation_date_earliest=-9999,
                        creation_date_latest=-9998,
                    ),
                    _record(
                        6,
                        creation_date="c. 590–560 BCE, with modern repainting",
                        creation_date_earliest=-595,
                        creation_date_latest=1923,
                    ),
                    _record(
                        7,
                        creation_date="100 BCE–400 CE, or modern",
                        creation_date_earliest=-100,
                        creation_date_latest=1927,
                    ),
                    _record(
                        8,
                        creation_date="INVALID",
                        creation_date_earliest=-1400,
                        creation_date_latest=1914,
                    ),
                ],
            )

            manifest = prepare_cma_visual_subset(
                source,
                output,
                source_revision=REVISION,
                preflight=False,
            )

            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["artwork_id"] for row in rows], ["CMA_1"])
            selection = manifest["selection"]
            self.assertEqual(selection["rejected_untrusted_timeline_date"], 7)
            sanity = selection["timeline_date_sanity"]
            self.assertEqual(sanity["policy"], CMA_TIMELINE_DATE_SANITY_POLICY)
            self.assertEqual(sanity["supported_min_year"], CMA_TIMELINE_MIN_YEAR)
            self.assertEqual(sanity["supported_max_year"], CMA_TIMELINE_MAX_YEAR)
            self.assertEqual(
                sanity["maximum_span_years"], CMA_TIMELINE_MAX_SPAN_YEARS
            )
            self.assertEqual(sanity["quarantined_rows"], 7)
            self.assertEqual(
                [row["artwork_id"] for row in sanity["sample_rows"]],
                [f"CMA_{identifier}" for identifier in range(2, 9)],
            )
            self.assertEqual(
                sanity["reason_counts"]["after_supported_maximum"], 1
            )
            self.assertEqual(
                sanity["reason_counts"]["before_supported_minimum"], 1
            )
            self.assertEqual(
                sanity["reason_counts"]["literal_invalid_display"], 1
            )

    def test_sampling_is_seeded_and_output_order_is_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data.json"
            _write_json(source, [_record(identifier) for identifier in range(1, 8)])
            selected_ids: list[list[str]] = []
            for name in ("first", "second"):
                output = root / f"{name}.csv"
                prepare_cma_visual_subset(
                    source,
                    output,
                    source_revision=REVISION,
                    sample_size=3,
                    seed="fixed-test-seed",
                    preflight=False,
                )
                with output.open(encoding="utf-8", newline="") as handle:
                    selected_ids.append(
                        [row["artwork_id"] for row in csv.DictReader(handle)]
                    )
            self.assertEqual(selected_ids[0], selected_ids[1])
            self.assertEqual(selected_ids[0], sorted(selected_ids[0]))
            self.assertEqual(len(selected_ids[0]), 3)

    def test_preflight_cache_is_url_bound_and_never_freezes_transient_failures(self) -> None:
        self.assertFalse(_cacheable_availability(False, "HTTP 429"))
        self.assertFalse(_cacheable_availability(False, "HTTP 503"))
        self.assertFalse(_cacheable_availability(False, "timed out"))
        self.assertTrue(_cacheable_availability(False, "HTTP 404"))
        self.assertTrue(_cacheable_availability(True, ""))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data.json"
            output = root / "cma.csv"
            first = _record(1)
            second = _record(2)
            _write_json(source, [first, second])
            first_url = first["images"]["web"]["url"]  # type: ignore[index]
            second_url = second["images"]["web"]["url"]  # type: ignore[index]
            cache = output.with_suffix(".availability.csv")
            with cache.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("artwork_id", "image_url", "available", "reason"),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "artwork_id": "CMA_1",
                        "image_url": first_url,
                        "available": "True",
                        "reason": "",
                    }
                )
                writer.writerow(
                    {
                        "artwork_id": "CMA_2",
                        "image_url": "https://openaccess-cdn.clevelandart.org/stale.jpg",
                        "available": "True",
                        "reason": "",
                    }
                )

            with patch(
                "pipeline.cma_visual._remote_image_available", return_value=(True, "")
            ) as available:
                manifest = prepare_cma_visual_subset(
                    source,
                    output,
                    source_revision=REVISION,
                    preflight=True,
                )
            available.assert_called_once_with(second_url)
            self.assertTrue(manifest["images"]["availability_preflight"])
            self.assertEqual(manifest["selection"]["prepared_rows"], 2)

    def test_requires_pinned_revision_official_schema_and_nonoverlapping_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data.json"
            _write_json(source, [_record(1)])
            original = source.read_bytes()
            with self.assertRaisesRegex(CorpusBuildError, "pinned 40-character git SHA"):
                prepare_cma_visual_subset(
                    source,
                    root / "invalid.csv",
                    source_revision="master",
                    preflight=False,
                )

            api_snapshot = root / "api.json"
            _write_json(api_snapshot, {"data": _record(2)})
            with self.assertRaisesRegex(
                CorpusBuildError, "must equal the snapshot SHA-256"
            ):
                prepare_cma_visual_subset(
                    api_snapshot,
                    root / "api-invalid.csv",
                    source_revision=REVISION,
                    preflight=False,
                )
            with self.assertRaisesRegex(CorpusBuildError, "must not overwrite"):
                prepare_cma_visual_subset(
                    source,
                    source,
                    source_revision=REVISION,
                    preflight=False,
                )
            self.assertEqual(source.read_bytes(), original)

            malformed = root / "malformed.json"
            _write_json(malformed, {"data": "not artwork data"})
            with self.assertRaisesRegex(CorpusBuildError, "data must be an object or list"):
                prepare_cma_visual_subset(
                    malformed,
                    root / "malformed.csv",
                    source_revision=REVISION,
                    preflight=False,
                )

            duplicate = root / "duplicate.json"
            _write_json(duplicate, [_record(1), _record(1)])
            with self.assertRaisesRegex(CorpusBuildError, "duplicate id"):
                prepare_cma_visual_subset(
                    duplicate,
                    root / "duplicate.csv",
                    source_revision=REVISION,
                    preflight=False,
                )


if __name__ == "__main__":
    unittest.main()
