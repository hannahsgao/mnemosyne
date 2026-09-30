from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scipy import sparse

from pipeline import __version__
from pipeline.build import CorpusBuildError, build_corpus
from pipeline.smk_visual import (
    SMK_IMAGE_HOST,
    SMK_IMAGE_INPUT_POLICY,
    SMK_PUBLIC_DOMAIN_MARK_URI,
    VISUAL_SUBSET_SCHEMA_VERSION,
    _ValidatedSmkRedirectHandler,
    _cacheable_availability,
    _validate_smk_image_url,
    prepare_smk_visual_subset,
)


def _image_url(source_id: str) -> str:
    return (
        f"https://{SMK_IMAGE_HOST}/iiif/jp2/{source_id}.tif.jp2/"
        "full/!1024,/0/default.jpg"
    )


def _record(
    source_id: str,
    *,
    public_domain: object = True,
    rights: str = SMK_PUBLIC_DOMAIN_MARK_URI,
    has_image: object = True,
    image_thumbnail: str | None = None,
    production_date: object | None = None,
    production_dates_notes: object | None = None,
) -> dict[str, object]:
    return {
        "id": source_id,
        "object_number": f"KMS-{source_id}",
        "frontend_url": f"https://open.smk.dk/artwork/image/KMS-{source_id}",
        "titles": [{"title": f"Artwork {source_id}", "language": "English"}],
        "artist": [f"Artist {source_id}"],
        "production": [
            {
                "creator": f"Creator {source_id}",
                "creator_nationality": "Danish",
            }
        ],
        "production_date": (
            production_date
            if production_date is not None
            else [
                {
                    "start": "1900-01-01T00:00:00.000Z",
                    "end": "1900-12-31T00:00:00.000Z",
                    "period": "1900",
                }
            ]
        ),
        "production_dates_notes": (
            production_dates_notes
            if production_dates_notes is not None
            else ["Værkdatering: 1900"]
        ),
        "object_names": [{"name": "Painting"}, {"name": "Artwork"}],
        "techniques": ["Oil on canvas"],
        "materials": ["Canvas"],
        "responsible_department": "Paintings and Sculpture",
        "public_domain": public_domain,
        "rights": rights,
        "has_image": has_image,
        "image_thumbnail": (
            _image_url(source_id) if image_thumbnail is None else image_thumbnail
        ),
    }


def _write_json(path: Path, payload: object) -> str:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class SmkVisualSubsetTests(unittest.TestCase):
    def test_strict_rights_gate_uses_first_date_and_excludes_creator_activity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "smk-nightly.json"
            output = root / "prepared" / "smk-visual.csv"
            records = [
                _record(
                    "2_object",
                    production_date=[
                        {
                            "start": "1880-01-01T00:00:00.000Z",
                            "end": "1890-12-31T00:00:00.000Z",
                            "period": "ca. 1880-1890",
                        },
                        {
                            "start": "2000-01-01T00:00:00.000Z",
                            "end": "2000-12-31T00:00:00.000Z",
                            "period": "2000",
                        },
                    ],
                ),
                _record(
                    "1_object",
                    production_date=[
                        {
                            "start": "1500-01-01T00:00:00.000Z",
                            "end": "1550-12-31T00:00:00.000Z",
                            "period": "1500-1550",
                        }
                    ],
                    production_dates_notes=[
                        "Værkdatering: ca. 1500-1550",
                        "Dateringen følger kunstnerens virkeår, da værket er udateret",
                    ],
                ),
                _record("3_object", public_domain=False),
                _record(
                    "4_object",
                    rights="https://creativecommons.org/publicdomain/zero/1.0/",
                ),
                _record("5_object", has_image=False),
                _record("6_object", image_thumbnail=""),
                _record(
                    "7_object",
                    image_thumbnail="https://example.test/iiif/restricted.jpg",
                ),
                _record("8_object", public_domain="true"),
                _record("9_object", has_image=1),
            ]
            revision = _write_json(snapshot, records)

            with patch("pipeline.smk_visual._JSON_CHUNK_SIZE", 17):
                manifest = prepare_smk_visual_subset(
                    snapshot,
                    output,
                    source_revision=revision,
                    preflight=False,
                )

            rows = _read_rows(output)
            self.assertEqual(
                [row["artwork_id"] for row in rows],
                ["SMK_1_object", "SMK_2_object"],
            )
            activity, dated = rows
            self.assertEqual(activity["source_id"], "1_object")
            self.assertEqual(activity["physical_object_id"], "SMK_1_object")
            self.assertEqual(activity["institution"], "smk")
            self.assertEqual(activity["date_display"], "")
            self.assertEqual(activity["date_start"], "")
            self.assertEqual(activity["date_end"], "")
            self.assertEqual(
                activity["date_parse_method"], "smk_creator_activity_date_excluded"
            )
            self.assertEqual(dated["date_display"], "ca. 1880-1890")
            self.assertEqual(dated["date_start"], "1880")
            self.assertEqual(dated["date_end"], "1890")
            self.assertEqual(dated["date_qualifier"], "circa")
            self.assertEqual(dated["artist"], "Artist 2_object")
            self.assertEqual(dated["object_type"], "Painting")
            self.assertEqual(dated["medium"], "Oil on canvas")
            self.assertEqual(dated["image_rights_uri"], SMK_PUBLIC_DOMAIN_MARK_URI)
            self.assertEqual(dated["image_input_policy"], SMK_IMAGE_INPUT_POLICY)
            self.assertEqual(dated["image_use_permitted"], "True")

            self.assertEqual(manifest["schema_version"], VISUAL_SUBSET_SCHEMA_VERSION)
            self.assertEqual(manifest["builder_version"], __version__)
            self.assertEqual(manifest["source"]["revision"], revision)
            self.assertEqual(manifest["source"]["snapshot"]["sha256"], revision)
            self.assertEqual(manifest["selection"]["input_rows"], 9)
            self.assertEqual(manifest["selection"]["prepared_rows"], 2)
            self.assertEqual(manifest["selection"]["creator_activity_dates_excluded"], 1)
            self.assertEqual(manifest["selection"]["ignored_additional_production_dates"], 1)
            self.assertEqual(manifest["rights_gate"]["rejected_not_public_domain"], 2)
            self.assertEqual(manifest["rights_gate"]["rejected_wrong_rights_uri"], 1)
            self.assertEqual(manifest["rights_gate"]["rejected_without_image"], 2)
            self.assertEqual(
                manifest["rights_gate"]["rejected_without_image_thumbnail"], 1
            )
            self.assertEqual(manifest["rights_gate"]["rejected_invalid_image_url"], 1)
            self.assertEqual(manifest["placeholder_basenames"], [])
            self.assertEqual(manifest["images"]["stored_bytes"], 0)
            self.assertEqual(
                manifest["output"]["sha256"], hashlib.sha256(output.read_bytes()).hexdigest()
            )
            self.assertFalse(output.with_suffix(".incomplete.json").exists())

            corpus = root / "corpus"
            corpus_manifest = build_corpus(
                output,
                corpus,
                corpus_version="smk-fixture-v1",
                source_revision=revision,
                source_url="https://getallzip.open.smk.dk/smk_all_da.zip",
                source_kind="smk-open-local-json-snapshot",
                source_payloads=(snapshot, output.with_suffix(".manifest.json")),
            )
            self.assertEqual(corpus_manifest["counts"]["dated_rows"], 1)
            self.assertEqual(corpus_manifest["counts"]["unknown_date_rows"], 1)
            matrix = sparse.load_npz(corpus / "date-weights.npz")
            self.assertAlmostEqual(float(matrix[0].sum()), 0.0)
            self.assertAlmostEqual(float(matrix[1].sum()), 1.0)
            images = _read_rows(corpus / "images.manifest.csv")
            self.assertTrue(
                all(row["permission_status"] == "public-domain" for row in images)
            )

    def test_accepts_one_json_member_in_an_official_zip_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "smk_all_da.zip"
            payload = json.dumps([_record("1170012466_object")], ensure_ascii=False)
            with zipfile.ZipFile(snapshot, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("smk_all_da.json", payload)
            revision = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            output = root / "smk.csv"

            manifest = prepare_smk_visual_subset(
                snapshot,
                output,
                source_revision=revision,
                preflight=False,
            )

            rows = _read_rows(output)
            self.assertEqual(rows[0]["artwork_id"], "SMK_1170012466_object")
            source = manifest["source"]["snapshot"]
            self.assertEqual(source["container"], "zip")
            self.assertEqual(source["json_member"], "smk_all_da.json")
            self.assertEqual(source["json_member_bytes"], len(payload.encode("utf-8")))
            self.assertRegex(source["json_member_crc32"], r"^[0-9a-f]{8}$")

    def test_accepts_an_api_items_snapshot_and_english_activity_disclosure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "api-page.json"
            record = _record(
                "api_object",
                production_dates_notes=[
                    "Dating follows the creator's activity period because the work is undated."
                ],
            )
            revision = _write_json(
                snapshot,
                {"offset": 0, "rows": 1, "found": 1, "items": [record]},
            )
            output = root / "smk.csv"

            manifest = prepare_smk_visual_subset(
                snapshot,
                output,
                source_revision=revision,
                preflight=False,
            )

            row = _read_rows(output)[0]
            self.assertEqual(row["artwork_id"], "SMK_api_object")
            self.assertEqual(row["date_start"], "")
            self.assertEqual(row["date_end"], "")
            self.assertEqual(
                row["date_parse_method"], "smk_creator_activity_date_excluded"
            )
            self.assertEqual(manifest["source"]["snapshot"]["container"], "json")

    def test_does_not_fall_forward_when_the_first_production_date_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "smk.json"
            record = _record(
                "first-only",
                production_date=[
                    {"start": "", "end": "", "period": "Unknown"},
                    {
                        "start": "1999-01-01T00:00:00.000Z",
                        "end": "1999-12-31T00:00:00.000Z",
                        "period": "1999",
                    },
                ],
            )
            revision = _write_json(snapshot, [record])
            output = root / "smk.csv"

            manifest = prepare_smk_visual_subset(
                snapshot,
                output,
                source_revision=revision,
                preflight=False,
            )

            row = _read_rows(output)[0]
            self.assertEqual(row["date_display"], "")
            self.assertEqual(row["date_start"], "")
            self.assertEqual(row["date_end"], "")
            self.assertEqual(
                row["date_parse_method"], "smk_first_production_date_invalid"
            )
            self.assertEqual(
                manifest["selection"]["missing_or_invalid_first_production_date"], 1
            )
            self.assertEqual(
                manifest["selection"]["ignored_additional_production_dates"], 1
            )

    def test_requires_the_snapshot_content_digest_and_official_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "smk.json"
            revision = _write_json(snapshot, [_record("one")])

            with self.assertRaisesRegex(CorpusBuildError, "must match the SHA-256"):
                prepare_smk_visual_subset(
                    snapshot,
                    root / "wrong-digest.csv",
                    source_revision="0" * 64,
                    preflight=False,
                )
            self.assertFalse((root / "wrong-digest.incomplete.json").exists())

            malformed = root / "malformed.json"
            malformed_revision = _write_json(
                malformed,
                [{"id": "one", "public_domain": True}],
            )
            with self.assertRaisesRegex(CorpusBuildError, "missing required fields"):
                prepare_smk_visual_subset(
                    malformed,
                    root / "malformed.csv",
                    source_revision=malformed_revision,
                    preflight=False,
                )

            with self.assertRaisesRegex(CorpusBuildError, "lowercase SHA-256"):
                prepare_smk_visual_subset(
                    snapshot,
                    root / "uppercase.csv",
                    source_revision=revision.upper(),
                    preflight=False,
                )

    def test_sampling_is_stable_across_snapshot_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [_record(f"{index}_object") for index in range(8)]
            selected: list[list[str]] = []
            for name, ordered in (("forward", records), ("reverse", list(reversed(records)))):
                snapshot = root / f"{name}.json"
                revision = _write_json(snapshot, ordered)
                output = root / f"{name}.csv"
                prepare_smk_visual_subset(
                    snapshot,
                    output,
                    source_revision=revision,
                    sample_size=3,
                    seed="fixed-fixture-seed",
                    preflight=False,
                )
                selected.append([row["artwork_id"] for row in _read_rows(output)])
            self.assertEqual(selected[0], selected[1])

    def test_preflight_cache_is_url_bound_and_never_touches_restricted_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "smk.json"
            records = [
                _record("cached"),
                _record("changed"),
                _record("restricted", public_domain=False),
            ]
            revision = _write_json(snapshot, records)
            output = root / "smk.csv"
            availability = output.with_suffix(".availability.csv")
            with availability.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("artwork_id", "image_url", "available", "reason"),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "artwork_id": "SMK_cached",
                            "image_url": _image_url("cached"),
                            "available": "True",
                            "reason": "",
                        },
                        {
                            "artwork_id": "SMK_changed",
                            "image_url": _image_url("stale"),
                            "available": "True",
                            "reason": "",
                        },
                    ]
                )

            checked: list[str] = []

            def available(url: str) -> tuple[bool, str]:
                checked.append(url)
                return True, ""

            with patch("pipeline.smk_visual._remote_image_available", side_effect=available):
                manifest = prepare_smk_visual_subset(
                    snapshot,
                    output,
                    source_revision=revision,
                    workers=1,
                    preflight=True,
                )

            self.assertEqual(checked, [_image_url("changed")])
            self.assertNotIn("restricted", " ".join(checked))
            self.assertEqual(manifest["selection"]["prepared_rows"], 2)
            self.assertTrue(manifest["images"]["availability_preflight"])

    def test_transient_preflight_failure_is_retried_on_the_next_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "smk.json"
            revision = _write_json(snapshot, [_record("retry")])
            output = root / "smk.csv"
            with patch(
                "pipeline.smk_visual._remote_image_available",
                return_value=(False, "HTTP 503"),
            ):
                with self.assertRaisesRegex(CorpusBuildError, "no reachable SMK images"):
                    prepare_smk_visual_subset(
                        snapshot,
                        output,
                        source_revision=revision,
                        workers=1,
                        preflight=True,
                    )

            availability = output.with_suffix(".availability.csv")
            self.assertEqual(len(_read_rows(availability)), 0)
            with patch(
                "pipeline.smk_visual._remote_image_available",
                return_value=(True, ""),
            ) as retry:
                manifest = prepare_smk_visual_subset(
                    snapshot,
                    output,
                    source_revision=revision,
                    workers=1,
                    preflight=True,
                )
            retry.assert_called_once_with(_image_url("retry"))
            self.assertEqual(manifest["selection"]["prepared_rows"], 1)

    def test_image_urls_and_redirects_are_restricted_to_the_smk_thumbnail_host(self) -> None:
        _validate_smk_image_url(_image_url("safe"))
        invalid = (
            "http://iip-thumb.smk.dk/iiif/image.jpg",
            "https://example.test/iiif/image.jpg",
            "https://user:secret@iip-thumb.smk.dk/iiif/image.jpg",
            "https://iip-thumb.smk.dk:8443/iiif/image.jpg",
            "https://iip-thumb.smk.dk/iiif/image.jpg?download=1",
            "https://iip-thumb.smk.dk/iiif/image.jpg#fragment",
        )
        for url in invalid:
            with self.subTest(url=url), self.assertRaises(ValueError):
                _validate_smk_image_url(url)

        with self.assertRaisesRegex(ValueError, SMK_IMAGE_HOST.replace(".", r"\.")):
            _ValidatedSmkRedirectHandler().redirect_request(
                None,
                None,
                302,
                "Found",
                {},
                "https://example.test/redirected.jpg",
            )

        self.assertTrue(_cacheable_availability(True, ""))
        self.assertTrue(_cacheable_availability(False, "HTTP 404"))
        self.assertTrue(
            _cacheable_availability(False, "unexpected content type: text/html")
        )
        self.assertFalse(_cacheable_availability(False, "HTTP 429"))
        self.assertFalse(_cacheable_availability(False, "HTTP 503"))
        self.assertFalse(_cacheable_availability(False, "temporary DNS failure"))


if __name__ == "__main__":
    unittest.main()
