from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import threading
import unittest
from unittest.mock import call, Mock, patch
from urllib.error import HTTPError

from pipeline import __version__
from pipeline.aic_visual import (
    AIC_CC0_URI,
    AIC_IMAGE_INPUT_POLICY,
    AIC_IIIF_BASE,
    PHYSICAL_OBJECT_GROUPING_POLICY,
    VISUAL_SUBSET_SCHEMA_VERSION,
    _cacheable_availability,
    _iiif_image_url,
    _rank,
    _remote_image_available,
    _validate_iiif_base,
    compute_aic_snapshot_revision,
    prepare_aic_visual_subset,
)
from pipeline.build import CorpusBuildError, build_corpus


def _uuid(identifier: int) -> str:
    return f"{identifier:08x}-0000-0000-0000-{identifier:012x}"


def _record(
    identifier: int,
    *,
    public_domain: object = True,
    image_id: object | None = None,
    copyright_notice: object | None = None,
    date_display: str = "1900",
    date_start: object = 1900,
    date_end: object = 1900,
    date_qualifier_title: str | None = None,
) -> dict[str, object]:
    return {
        "id": identifier,
        "api_model": "artworks",
        "api_link": f"https://api.artic.edu/api/v1/artworks/{identifier}",
        "title": f"Object {identifier}",
        "date_start": date_start,
        "date_end": date_end,
        "date_display": date_display,
        "date_qualifier_title": date_qualifier_title,
        "artist_display": "Fallback Artist\nAmerican, 1900-1980",
        "artist_title": "Preferred Artist",
        "artist_titles": ["First Artist", "Second Artist", "first artist"],
        "place_of_origin": "United States",
        "description": "CC BY prose that must not enter the canonical row",
        "medium_display": "Oil on canvas",
        "credit_line": "Example Collection",
        "is_public_domain": public_domain,
        "copyright_notice": copyright_notice,
        "artwork_type_title": "Painting",
        "department_title": "Painting and Sculpture of Europe",
        "classification_title": "paintings (visual works)",
        "term_titles": ["painting", "oil paint", "Painting"],
        "image_id": _uuid(identifier) if image_id is None else image_id,
        "alt_image_ids": [_uuid(identifier + 1000)],
    }


def _write_dump(
    root: Path,
    records: list[dict[str, object]],
    *,
    iiif_url: str = AIC_IIIF_BASE,
) -> Path:
    dump = root / "aic-api-data"
    artworks = dump / "json" / "artworks"
    artworks.mkdir(parents=True)
    (dump / "json" / "config.json").write_text(
        json.dumps(
            {"iiif_url": iiif_url, "website_url": "http://www.artic.edu"},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for record in records:
        identifier = record["id"]
        (artworks / f"{identifier}.json").write_text(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return dump


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class AicVisualSubsetTests(unittest.TestCase):
    def test_prepares_only_strictly_eligible_rows_and_maps_canonical_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [
                {
                    **_record(
                        27992,
                        date_display="1884–1886",
                        date_start=1884,
                        date_end=1886,
                    ),
                    "title": "A Sunday on La Grande Jatte — 1884",
                },
                _record(2, public_domain=False),
                _record(3, public_domain="true"),
                _record(4, copyright_notice="© Example Artist"),
                _record(5, image_id=""),
                _record(6, image_id="not-a-uuid"),
                _record(
                    7,
                    copyright_notice="  ",
                    date_display="n.d.",
                    date_start=1845,
                    date_end=1921,
                    date_qualifier_title="Artist's working dates",
                ),
            ]
            dump = _write_dump(root, records)
            revision = compute_aic_snapshot_revision(dump)
            output = root / "prepared" / "aic-visual.csv"

            manifest = prepare_aic_visual_subset(
                dump,
                output,
                source_revision=revision,
                preflight=False,
            )

            rows = _read_rows(output)
            self.assertEqual(
                [row["artwork_id"] for row in rows], ["AIC_27992", "AIC_7"]
            )
            first, undated = rows
            self.assertEqual(first["physical_object_id"], "AIC_27992")
            self.assertEqual(first["institution"], "aic")
            self.assertEqual(first["source_id"], "27992")
            self.assertEqual(first["source_dataset_version"], revision)
            self.assertEqual(
                first["source_record_url"],
                "https://www.artic.edu/artworks/27992",
            )
            self.assertEqual(first["artist"], "First Artist; Second Artist")
            self.assertEqual(first["object_type"], "Painting")
            self.assertEqual(first["medium"], "Oil on canvas")
            self.assertEqual(first["department"], "Painting and Sculpture of Europe")
            self.assertEqual(first["classification"], "paintings (visual works)")
            self.assertEqual(first["geography"], "United States")
            self.assertEqual(first["tags"], "painting; oil paint")
            self.assertEqual(first["date_start"], "1884")
            self.assertEqual(first["date_end"], "1886")
            self.assertEqual(first["date_qualifier"], "range")
            self.assertEqual(first["metadata_license"], AIC_CC0_URI)
            self.assertEqual(first["image_rights_uri"], AIC_CC0_URI)
            self.assertEqual(first["public_domain"], "True")
            self.assertEqual(first["image_available"], "True")
            self.assertEqual(first["image_use_permitted"], "True")
            self.assertEqual(first["image_input_policy"], AIC_IMAGE_INPUT_POLICY)
            self.assertEqual(first["image_width"], "")
            self.assertEqual(first["image_height"], "")
            self.assertEqual(
                first["image_url"],
                f"{AIC_IIIF_BASE}/{_uuid(27992)}/full/843,/0/default.jpg",
            )
            self.assertNotIn("description", first)

            self.assertEqual(undated["date_display"], "n.d.")
            self.assertEqual(undated["date_start"], "")
            self.assertEqual(undated["date_end"], "")
            self.assertEqual(undated["date_qualifier"], "unknown")
            self.assertEqual(
                undated["date_parse_method"],
                "aic_non_work_date_bounds_excluded",
            )

            self.assertEqual(manifest["schema_version"], VISUAL_SUBSET_SCHEMA_VERSION)
            self.assertEqual(manifest["builder_version"], __version__)
            self.assertEqual(manifest["source"]["revision"], revision)
            self.assertEqual(
                manifest["source"]["revision_kind"],
                "content-inventory-sha256",
            )
            self.assertEqual(manifest["source"]["raw_snapshot"]["sha256"], revision)
            self.assertEqual(manifest["source"]["raw_snapshot"]["files"], 8)
            self.assertEqual(manifest["source"]["artworks"]["files"], 7)
            self.assertEqual(
                manifest["source"]["config"]["sha256"],
                hashlib.sha256(
                    (dump / "json" / "config.json").read_bytes()
                ).hexdigest(),
            )
            selection = manifest["selection"]
            self.assertEqual(selection["input_artwork_files"], 7)
            self.assertEqual(selection["eligible_candidates"], 2)
            self.assertEqual(selection["rejected_not_public_domain"], 2)
            self.assertEqual(selection["rejected_nonblank_copyright_notice"], 1)
            self.assertEqual(selection["rejected_missing_image_id"], 1)
            self.assertEqual(selection["rejected_invalid_image_id"], 1)
            self.assertEqual(selection["date_bounds_excluded_as_non_work_dates"], 1)
            self.assertEqual(manifest["rights_gate"]["institution"], "aic")
            self.assertEqual(
                manifest["physical_object_grouping"]["policy"],
                PHYSICAL_OBJECT_GROUPING_POLICY,
            )
            self.assertEqual(manifest["placeholder_basenames"], [])
            self.assertEqual(manifest["images"]["stored_bytes"], 0)
            self.assertEqual(
                manifest["output"]["sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )
            self.assertTrue(output.with_suffix(".manifest.json").is_file())
            self.assertFalse(output.with_suffix(".incomplete.json").exists())

    def test_output_builds_with_public_domain_permission_and_safe_dates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dump = _write_dump(
                root,
                [
                    _record(1, date_display="1900", date_start=1900, date_end=1900),
                    _record(
                        2,
                        date_display="n.d.",
                        date_start=1800,
                        date_end=1880,
                        date_qualifier_title="Artist's working dates",
                    ),
                ],
            )
            revision = compute_aic_snapshot_revision(dump)
            prepared = root / "aic.csv"
            prepare_aic_visual_subset(
                dump,
                prepared,
                source_revision=revision,
                preflight=False,
            )

            corpus = root / "corpus"
            manifest = build_corpus(
                prepared,
                corpus,
                corpus_version="aic-fixture-v1",
                source_revision=revision,
                retrieved_at="2026-09-03T00:00:00Z",
            )

            self.assertEqual(manifest["corpus"]["count"], 2)
            self.assertEqual(manifest["counts"]["dated_rows"], 1)
            self.assertEqual(manifest["counts"]["unknown_date_rows"], 1)
            with (corpus / "images.manifest.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                images = list(csv.DictReader(handle))
            self.assertTrue(
                all(row["permission_status"] == "public-domain" for row in images)
            )
            self.assertTrue(
                all(
                    row["image_url"].endswith("/full/843,/0/default.jpg")
                    for row in images
                )
            )

    def test_supports_prefixed_tar_bz2_and_checksums_original_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dump = _write_dump(root, [_record(11), _record(12)])
            archive = root / "artic-api-data.tar.bz2"
            with tarfile.open(archive, "w:bz2") as handle:
                handle.add(
                    dump / "json",
                    arcname="artic-api-data/json",
                    recursive=True,
                )
            revision = compute_aic_snapshot_revision(archive)
            directory_revision = compute_aic_snapshot_revision(dump)
            output = root / "from-archive.csv"

            manifest = prepare_aic_visual_subset(
                archive,
                output,
                source_revision=revision,
                preflight=False,
            )

            self.assertEqual(len(_read_rows(output)), 2)
            self.assertEqual(manifest["source"]["revision"], revision)
            self.assertEqual(manifest["source"]["revision_kind"], "archive-sha256")
            self.assertEqual(manifest["source"]["archive"]["sha256"], revision)
            self.assertEqual(
                manifest["source"]["raw_snapshot"]["sha256"], directory_revision
            )
            self.assertEqual(manifest["source"]["raw_snapshot"]["files"], 3)

    def test_preflight_is_sequential_rate_limited_and_uses_ranked_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dump = _write_dump(root, [_record(identifier) for identifier in range(1, 7)])
            revision = compute_aic_snapshot_revision(dump)
            output = root / "aic.csv"
            seed = "test-one-thread-preflight"
            ranked = sorted(
                [f"AIC_{identifier}" for identifier in range(1, 7)],
                key=lambda artwork_id: (_rank(seed, artwork_id), artwork_id),
            )
            checked: list[tuple[str, int]] = []

            def available(url: str, *, before_request) -> tuple[bool, str]:
                before_request()
                checked.append((url, threading.get_ident()))
                if len(checked) == 1:
                    return False, "HTTP 404"
                return True, ""

            with patch(
                "pipeline.aic_visual._remote_image_available", side_effect=available
            ), patch("pipeline.aic_visual.time.sleep") as sleep:
                manifest = prepare_aic_visual_subset(
                    dump,
                    output,
                    source_revision=revision,
                    sample_size=2,
                    seed=seed,
                    request_delay_seconds=1.0,
                )

            self.assertEqual(len(checked), 3)
            self.assertEqual({thread_id for _, thread_id in checked}, {threading.get_ident()})
            self.assertEqual(sleep.call_args_list, [call(1.0), call(1.0)])
            self.assertEqual(
                {row["artwork_id"] for row in _read_rows(output)}, set(ranked[1:3])
            )
            self.assertEqual(manifest["selection"]["examined_candidates"], 3)
            self.assertEqual(manifest["images"]["preflight_concurrency"], 1)
            self.assertEqual(manifest["images"]["preflight_live_requests"], 3)
            self.assertEqual(
                manifest["sample_failures"],
                [{"artwork_id": ranked[0], "reason": "HTTP 404"}],
            )
            availability = output.with_suffix(".availability.csv")
            self.assertEqual(
                manifest["images"]["availability_cache"]["sha256"],
                hashlib.sha256(availability.read_bytes()).hexdigest(),
            )

    def test_preflight_cache_is_bound_to_the_exact_url(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dump = _write_dump(root, [_record(1), _record(2)])
            revision = compute_aic_snapshot_revision(dump)
            output = root / "aic.csv"
            cache = output.with_suffix(".availability.csv")
            with cache.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("artwork_id", "image_url", "available", "reason"),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "artwork_id": "AIC_1",
                            "image_url": _iiif_image_url(AIC_IIIF_BASE, _uuid(1)),
                            "available": True,
                            "reason": "",
                        },
                        {
                            "artwork_id": "AIC_2",
                            "image_url": "https://www.artic.edu/iiif/2/"
                            f"{_uuid(2)}/full/600,/0/default.jpg",
                            "available": True,
                            "reason": "",
                        },
                    ]
                )

            def available(url: str, *, before_request) -> tuple[bool, str]:
                before_request()
                return True, ""

            with patch(
                "pipeline.aic_visual._remote_image_available", side_effect=available
            ) as remote, patch("pipeline.aic_visual.time.sleep") as sleep:
                manifest = prepare_aic_visual_subset(
                    dump,
                    output,
                    source_revision=revision,
                    preflight=True,
                )

            self.assertEqual(remote.call_count, 1)
            self.assertEqual(
                remote.call_args.args,
                (_iiif_image_url(AIC_IIIF_BASE, _uuid(2)),),
            )
            self.assertTrue(callable(remote.call_args.kwargs["before_request"]))
            sleep.assert_not_called()
            self.assertEqual(manifest["images"]["preflight_live_requests"], 1)
            self.assertEqual(manifest["selection"]["prepared_rows"], 2)

    def test_transient_availability_failures_are_not_cacheable(self) -> None:
        self.assertTrue(_cacheable_availability(True, ""))
        self.assertTrue(_cacheable_availability(False, "HTTP 404"))
        self.assertTrue(
            _cacheable_availability(False, "unexpected content type: text/html")
        )
        self.assertFalse(_cacheable_availability(False, "HTTP 429"))
        self.assertFalse(_cacheable_availability(False, "HTTP 503"))
        self.assertFalse(
            _cacheable_availability(False, "temporary DNS resolution failure")
        )

    def test_preflight_paces_each_retry_attempt(self) -> None:
        url = _iiif_image_url(AIC_IIIF_BASE, _uuid(1))

        class Response:
            headers = {"Content-Type": "image/jpeg"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def geturl(self) -> str:
                return url

        class Opener:
            def __init__(self) -> None:
                self.calls = 0

            def open(self, request, *, timeout):
                del request, timeout
                self.calls += 1
                if self.calls == 1:
                    raise HTTPError(url, 503, "unavailable", {}, None)
                return Response()

        opener = Opener()
        before_request = Mock()
        with (
            patch("pipeline.aic_visual._verified_opener", return_value=opener),
            patch("pipeline.aic_visual.time.sleep") as sleep,
        ):
            self.assertEqual(
                _remote_image_available(url, before_request=before_request),
                (True, ""),
            )

        self.assertEqual(before_request.call_count, 2)
        sleep.assert_called_once_with(0.25)

    def test_rejects_nonproduction_iiif_config_and_noncanonical_image_ids(self) -> None:
        for value, expected in (
            ("http://www.artic.edu/iiif/2", "must use https"),
            ("https://www-test.artic.edu/iiif/2", "production www.artic.edu"),
            ("https://www.artic.edu/iiif/3", "production /iiif/2"),
            ("https://www.artic.edu:444/iiif/2", "custom port"),
        ):
            with self.subTest(value=value), self.assertRaisesRegex(
                CorpusBuildError, expected
            ):
                _validate_iiif_base(value)

        with self.assertRaisesRegex(ValueError, "canonical UUID"):
            _iiif_image_url(AIC_IIIF_BASE, "not-a-uuid")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dump = _write_dump(root, [_record(1)])
            revision = compute_aic_snapshot_revision(dump)
            with self.assertRaisesRegex(CorpusBuildError, "exactly one worker"):
                prepare_aic_visual_subset(
                    dump,
                    root / "parallel.csv",
                    source_revision=revision,
                    workers=2,
                    preflight=False,
                )
            for label, delay in (
                ("short", 0.5),
                ("nan", float("nan")),
                ("infinity", float("inf")),
            ):
                with self.subTest(delay=label), self.assertRaisesRegex(
                    CorpusBuildError, "finite and at least 1 second"
                ):
                    prepare_aic_visual_subset(
                        dump,
                        root / f"{label}.csv",
                        source_revision=revision,
                        request_delay_seconds=delay,
                        preflight=False,
                    )

    def test_requires_matching_snapshot_checksum_and_official_record_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dump = _write_dump(root, [_record(1)])
            revision = compute_aic_snapshot_revision(dump)
            record_path = dump / "json" / "artworks" / "1.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["title"] = "Source changed after pinning"
            record_path.write_text(
                json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(CorpusBuildError, "does not match"):
                prepare_aic_visual_subset(
                    dump,
                    root / "changed.csv",
                    source_revision=revision,
                    preflight=False,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dump = _write_dump(root, [_record(1)])
            record_path = dump / "json" / "artworks" / "1.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["api_model"] = "agents"
            record_path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(CorpusBuildError, "api_model=artworks"):
                prepare_aic_visual_subset(
                    dump,
                    root / "wrong-model.csv",
                    source_revision="0" * 64,
                    preflight=False,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dump = _write_dump(root, [_record(1)])
            artwork = dump / "json" / "artworks" / "1.json"
            artwork.rename(dump / "json" / "artworks" / "2.json")
            with self.assertRaisesRegex(CorpusBuildError, "does not match filename"):
                prepare_aic_visual_subset(
                    dump,
                    root / "mismatch.csv",
                    source_revision="0" * 64,
                    preflight=False,
                )


if __name__ == "__main__":
    unittest.main()
