from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline import __version__
from pipeline.build import CorpusBuildError, build_corpus
from pipeline.nga_visual import (
    NGA_CC0_URI,
    NGA_IMAGE_INPUT_POLICY,
    PHYSICAL_OBJECT_GROUPING_POLICY,
    VISUAL_SUBSET_SCHEMA_VERSION,
    _iiif_derivative_url,
    _is_unknown_display_date,
    prepare_nga_visual_subset,
)


REVISION = "4a1aef41c56f4c20924ffe40898f9ffce000aabf"
OBJECT_FIELDS = (
    "objectid",
    "title",
    "displaydate",
    "beginyear",
    "endyear",
    "medium",
    "attribution",
    "creditline",
    "classification",
    "subclassification",
    "isvirtual",
    "departmentabbr",
    "wikidataid",
)
IMAGE_FIELDS = (
    "uuid",
    "iiifurl",
    "viewtype",
    "sequence",
    "width",
    "height",
    "openaccess",
    "depictstmsobjectid",
)
ASSOCIATION_FIELDS = (
    "parentobjectid",
    "childobjectid",
    "relationship",
)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _image(
    object_id: str,
    uuid: str,
    *,
    sequence: str = "0",
    width: str = "4000",
    height: str = "3000",
    openaccess: str = "1",
    viewtype: str = "primary",
    host: str = "api.nga.gov",
) -> dict[str, str]:
    return {
        "uuid": uuid,
        "iiifurl": f"https://{host}/iiif/{uuid}",
        "viewtype": viewtype,
        "sequence": sequence,
        "width": width,
        "height": height,
        "openaccess": openaccess,
        "depictstmsobjectid": object_id,
    }


def _object(
    object_id: str,
    *,
    title: str | None = None,
    displaydate: str = "1900",
    beginyear: str = "1900",
    endyear: str = "1900",
    isvirtual: str = "0",
) -> dict[str, str]:
    return {
        "objectid": object_id,
        "title": title or f"Object {object_id}",
        "displaydate": displaydate,
        "beginyear": beginyear,
        "endyear": endyear,
        "medium": "oil on canvas",
        "attribution": "Example Artist",
        "creditline": "Example Collection",
        "classification": "Painting",
        "subclassification": "Panel painting",
        "isvirtual": isvirtual,
        "departmentabbr": "PAINT",
        "wikidataid": "Q42",
    }


class NgaVisualSubsetTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        objects = root / "objects.csv"
        published = root / "published_images.csv"
        associations = root / "object_associations.csv"
        object_rows = [
            _object("1", title="Circa work", displaydate="c. 1900"),
            _object("2", title="Range work", displaydate="1910 to 1920", beginyear="1910", endyear="1920"),
            _object("3", title="Restricted image"),
            _object("4", title="Virtual record", isvirtual="1"),
            _object("5", title="Artist-life fallback", displaydate="", beginyear="1800", endyear="1880"),
            _object("6", title="Tiny source"),
            _object("7", title="Bad host"),
            _object(
                "8",
                title="Incomplete numeric date",
                displaydate="2nd/1st century BCE",
                beginyear="-199",
                endyear="",
            ),
            _object(
                "9",
                title="Explicitly unknown date",
                displaydate="n.d.",
                beginyear="1840",
                endyear="1917",
            ),
        ]
        uuids = {
            "one": "11111111-1111-4111-8111-111111111111",
            "two_late": "22222222-2222-4222-8222-222222222222",
            "two_tie_late": "33333333-3333-4333-8333-333333333333",
            "two_chosen": "00000000-0000-4000-8000-000000000000",
            "three": "44444444-4444-4444-8444-444444444444",
            "four": "55555555-5555-4555-8555-555555555555",
            "five": "66666666-6666-4666-8666-666666666666",
            "six": "77777777-7777-4777-8777-777777777777",
            "seven": "88888888-8888-4888-8888-888888888888",
            "eight": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "nine": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            "alternate": "99999999-9999-4999-8999-999999999999",
        }
        image_rows = [
            _image("2", uuids["two_late"], sequence="2"),
            _image("1", uuids["alternate"], viewtype="alternate"),
            _image("5", uuids["five"]),
            _image("2", uuids["two_tie_late"], sequence="1"),
            _image("3", uuids["three"], openaccess="0"),
            _image("4", uuids["four"]),
            _image("7", uuids["seven"], host="example.com"),
            _image("2", uuids["two_chosen"], sequence="1"),
            _image("6", uuids["six"], width="200", height="1000"),
            _image("1", uuids["one"], width="4000", height="3000"),
            _image("8", uuids["eight"]),
            _image("9", uuids["nine"]),
        ]
        _write_csv(objects, OBJECT_FIELDS, object_rows)
        _write_csv(published, IMAGE_FIELDS, image_rows)
        _write_csv(
            associations,
            ASSOCIATION_FIELDS,
            [
                {
                    "parentobjectid": "1",
                    "childobjectid": "2",
                    "relationship": "separable",
                }
            ],
        )
        return objects, published, associations

    def test_iiif_derivative_fits_normal_images_and_expands_panorama_bound(self) -> None:
        uuid = "11111111-1111-4111-8111-111111111111"
        base = f"https://api.nga.gov/iiif/{uuid}"
        normal = _iiif_derivative_url(base, uuid, 4000, 3000)
        panorama = _iiif_derivative_url(base, uuid, 8000, 1000)

        self.assertEqual(
            normal,
            (f"{base}/full/!1024,768/0/default.jpg", 1024, 768),
        )
        self.assertEqual(
            panorama,
            (f"{base}/full/!2048,256/0/default.jpg", 2048, 256),
        )
        with self.assertRaisesRegex(ValueError, "short side"):
            _iiif_derivative_url(base, uuid, 1000, 200)
        with self.assertRaisesRegex(ValueError, "api.nga.gov"):
            _iiif_derivative_url(
                f"https://example.com/iiif/{uuid}", uuid, 1000, 1000
            )

    def test_unknown_display_date_only_rejects_wholly_unknown_labels(self) -> None:
        for label in (
            "n.d.",
            "N. D.",
            "date unknown",
            "Unknown",
            "undated",
            "Model date unknown",
            "model date unknown; cast later",
        ):
            with self.subTest(rejected=label):
                self.assertTrue(_is_unknown_display_date(label))

        for label in (
            "designed before 1848, cast date unknown",
            "1788/1789, altered later (date unknown)",
            "model 1863/1865, cast date unknown",
        ):
            with self.subTest(retained=label):
                self.assertFalse(_is_unknown_display_date(label))

    def test_non_default_iiif_dimensions_have_truthful_input_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            objects, published, associations = self._write_fixture(root)
            output = root / "nga-custom-size.csv"
            manifest = prepare_nga_visual_subset(
                objects,
                published,
                output,
                object_associations_csv=associations,
                source_revision=REVISION,
                max_dimension=768,
                min_short_side=224,
                preflight=False,
            )

            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            expected_policy = "nga-iiif-fit-768-short-side-224/v1"
            self.assertTrue(
                all(row["image_input_policy"] == expected_policy for row in rows)
            )
            self.assertTrue(
                all("/full/!768,576/0/default.jpg" in row["image_url"] for row in rows)
            )
            self.assertEqual(manifest["images"]["input_policy"], expected_policy)

    def test_inseparable_chains_share_a_root_without_collapsing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            objects, published, associations = self._write_fixture(root)
            _write_csv(
                associations,
                ASSOCIATION_FIELDS,
                [
                    {
                        "parentobjectid": "3",
                        "childobjectid": "1",
                        "relationship": "inseparable",
                    },
                    {
                        "parentobjectid": "1",
                        "childobjectid": "2",
                        "relationship": "InSePaRaBlE",
                    },
                    {
                        "parentobjectid": "4",
                        "childobjectid": "5",
                        "relationship": "separable",
                    },
                ],
            )
            output = root / "nga-grouped.csv"
            manifest = prepare_nga_visual_subset(
                objects,
                published,
                output,
                object_associations_csv=associations,
                source_revision=REVISION,
                preflight=False,
            )

            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                [row["physical_object_id"] for row in rows], ["NGA_3", "NGA_3"]
            )
            grouping = manifest["physical_object_grouping"]
            self.assertEqual(grouping["inseparable_association_rows"], 2)
            self.assertEqual(grouping["separable_association_rows"], 1)
            self.assertEqual(grouping["maximum_inseparable_chain_depth"], 2)
            self.assertEqual(grouping["selected_rows"], 2)
            self.assertEqual(grouping["selected_distinct_physical_object_ids"], 1)
            self.assertEqual(grouping["selected_shared_physical_groups"], 1)
            self.assertEqual(grouping["rows_collapsed"], 0)

    def test_rejects_invalid_inseparable_association_graphs(self) -> None:
        cases = {
            "duplicate association": [
                ("1", "2", "inseparable"),
                ("1", "2", "INSEPARABLE"),
            ],
            "multiple parents": [
                ("1", "2", "inseparable"),
                ("3", "2", "inseparable"),
            ],
            "inseparable cycle": [
                ("1", "2", "inseparable"),
                ("2", "1", "inseparable"),
            ],
            "missing from objects.csv": [("1", "999", "inseparable")],
        }
        for expected, edges in cases.items():
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                objects, published, associations = self._write_fixture(root)
                _write_csv(
                    associations,
                    ASSOCIATION_FIELDS,
                    [
                        {
                            "parentobjectid": parent,
                            "childobjectid": child,
                            "relationship": relationship,
                        }
                        for parent, child, relationship in edges
                    ],
                )
                with self.assertRaisesRegex(CorpusBuildError, expected):
                    prepare_nga_visual_subset(
                        objects,
                        published,
                        root / "invalid.csv",
                        object_associations_csv=associations,
                        source_revision=REVISION,
                        preflight=False,
                    )

    def test_prepares_strict_rights_gated_rows_with_deterministic_primary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            objects, published, associations = self._write_fixture(root)
            output = root / "prepared" / "nga-visual.csv"

            manifest = prepare_nga_visual_subset(
                objects,
                published,
                output,
                object_associations_csv=associations,
                source_revision=REVISION,
                preflight=False,
            )

            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["artwork_id"] for row in rows], ["NGA_1", "NGA_2"])
            first, second = rows
            self.assertEqual(first["source_id"], "1")
            self.assertEqual(first["institution"], "nga")
            self.assertEqual(first["physical_object_id"], "NGA_1")
            self.assertEqual(second["physical_object_id"], "NGA_2")
            self.assertEqual(first["source_dataset_version"], REVISION)
            self.assertEqual(
                first["source_record_url"],
                "https://www.nga.gov/collection/art-object-page.1.html",
            )
            self.assertEqual(first["artist"], "Example Artist")
            self.assertEqual(first["object_type"], "Panel painting")
            self.assertEqual(first["object_wikidata_url"], "https://www.wikidata.org/wiki/Q42")
            self.assertEqual(first["date_qualifier"], "circa")
            self.assertEqual(first["public_domain"], "True")
            self.assertEqual(first["image_use_permitted"], "True")
            self.assertEqual(first["image_rights_uri"], NGA_CC0_URI)
            self.assertEqual(first["image_input_policy"], NGA_IMAGE_INPUT_POLICY)
            self.assertIn("/full/!1024,768/0/default.jpg", first["image_url"])
            self.assertIn(
                "00000000-0000-4000-8000-000000000000", second["image_url"]
            )

            self.assertEqual(manifest["schema_version"], VISUAL_SUBSET_SCHEMA_VERSION)
            self.assertEqual(manifest["builder_version"], __version__)
            self.assertEqual(manifest["source"]["revision"], REVISION)
            self.assertEqual(
                manifest["source"]["object_associations"]["rows"], 1
            )
            self.assertEqual(
                manifest["source"]["object_associations"]["sha256"],
                hashlib.sha256(associations.read_bytes()).hexdigest(),
            )
            self.assertEqual(manifest["selection"]["prepared_rows"], 2)
            self.assertEqual(manifest["selection"]["rejected_not_open_access"], 1)
            self.assertEqual(manifest["selection"]["rejected_virtual_objects"], 1)
            self.assertEqual(manifest["selection"]["rejected_without_display_date"], 1)
            self.assertEqual(manifest["selection"]["rejected_unknown_display_date"], 1)
            self.assertEqual(
                manifest["selection"]["rejected_without_numeric_date_bounds"], 1
            )
            self.assertEqual(manifest["selection"]["rejected_source_short_side"], 1)
            self.assertEqual(manifest["selection"]["rejected_invalid_image_metadata"], 1)
            self.assertEqual(manifest["placeholder_basenames"], [])
            self.assertEqual(manifest["rights_gate"]["institution"], "nga")
            self.assertEqual(
                manifest["physical_object_grouping"]["policy"],
                PHYSICAL_OBJECT_GROUPING_POLICY,
            )
            self.assertEqual(
                manifest["physical_object_grouping"][
                    "separable_association_rows"
                ],
                1,
            )
            self.assertEqual(
                manifest["physical_object_grouping"][
                    "selected_distinct_physical_object_ids"
                ],
                2,
            )
            self.assertEqual(
                manifest["physical_object_grouping"]["rows_collapsed"], 0
            )
            self.assertEqual(manifest["images"]["stored_bytes"], 0)
            self.assertTrue(output.with_suffix(".manifest.json").is_file())
            self.assertFalse(output.with_suffix(".incomplete.json").exists())

    def test_missing_displaydate_is_never_replaced_by_numeric_artist_life(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            objects, published, associations = self._write_fixture(root)
            output = root / "nga-with-undated.csv"
            prepare_nga_visual_subset(
                objects,
                published,
                output,
                object_associations_csv=associations,
                source_revision=REVISION,
                preflight=False,
                include_undated=True,
            )

            with output.open(encoding="utf-8", newline="") as handle:
                rows = {row["artwork_id"]: row for row in csv.DictReader(handle)}
            fallback = rows["NGA_5"]
            self.assertEqual(fallback["date_display"], "")
            self.assertEqual(fallback["date_start"], "")
            self.assertEqual(fallback["date_end"], "")
            self.assertEqual(
                fallback["date_parse_method"],
                "nga_strict_timeline_gate_excluded",
            )
            incomplete = rows["NGA_8"]
            self.assertEqual(incomplete["date_display"], "")
            self.assertEqual(incomplete["date_start"], "")
            self.assertEqual(incomplete["date_end"], "")
            self.assertEqual(
                incomplete["date_parse_method"],
                "nga_strict_timeline_gate_excluded",
            )
            unknown = rows["NGA_9"]
            self.assertEqual(unknown["date_display"], "")
            self.assertEqual(unknown["date_start"], "")
            self.assertEqual(unknown["date_end"], "")
            corpus_manifest = build_corpus(
                output,
                root / "with-undated-corpus",
                corpus_version="nga-with-undated-fixture-v1",
                source_revision=REVISION,
                retrieved_at="2026-08-14T00:00:00Z",
            )
            self.assertEqual(corpus_manifest["counts"]["dated_rows"], 2)
            self.assertEqual(corpus_manifest["counts"]["unknown_date_rows"], 3)

    def test_output_builds_with_public_domain_image_permissions_and_date_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            objects, published, associations = self._write_fixture(root)
            prepared = root / "nga.csv"
            prepare_nga_visual_subset(
                objects,
                published,
                prepared,
                object_associations_csv=associations,
                source_revision=REVISION,
                preflight=False,
            )
            corpus = root / "corpus"
            manifest = build_corpus(
                prepared,
                corpus,
                corpus_version="nga-fixture-v1",
                source_revision=REVISION,
                source_url="https://github.com/NationalGalleryOfArt/opendata",
                metadata_license=NGA_CC0_URI,
                retrieved_at="2026-08-14T00:00:00Z",
                source_kind="nga-open-data-local-csv",
            )

            self.assertEqual(manifest["corpus"]["count"], 2)
            self.assertEqual(manifest["counts"]["dated_rows"], 2)
            with (corpus / "images.manifest.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                images = list(csv.DictReader(handle))
            self.assertTrue(
                all(row["permission_status"] == "public-domain" for row in images)
            )
            self.assertTrue(
                all(
                    row["image_url"].startswith("https://api.nga.gov/iiif/")
                    for row in images
                )
            )
            self.assertTrue(
                all(row["image_input_policy"] == NGA_IMAGE_INPUT_POLICY for row in images)
            )

    def test_preflight_reuses_only_url_bound_cache_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            objects, published, associations = self._write_fixture(root)
            output = root / "nga.csv"
            uuid_one = "11111111-1111-4111-8111-111111111111"
            uuid_two = "00000000-0000-4000-8000-000000000000"
            urls = {
                "NGA_1": (
                    f"https://api.nga.gov/iiif/{uuid_one}/full/!1024,768/0/default.jpg"
                ),
                "NGA_2": (
                    f"https://api.nga.gov/iiif/{uuid_two}/full/!1024,768/0/default.jpg"
                ),
            }
            cache = output.with_suffix(".availability.csv")
            _write_csv(
                cache,
                ("artwork_id", "image_url", "available", "reason"),
                [
                    {
                        "artwork_id": artwork_id,
                        "image_url": image_url,
                        "available": "True",
                        "reason": "",
                    }
                    for artwork_id, image_url in urls.items()
                ],
            )
            with patch(
                "pipeline.nga_visual._remote_image_available",
                side_effect=AssertionError("cached URLs must not be checked again"),
            ):
                manifest = prepare_nga_visual_subset(
                    objects,
                    published,
                    output,
                    object_associations_csv=associations,
                    source_revision=REVISION,
                    preflight=True,
                )
            self.assertEqual(manifest["selection"]["prepared_rows"], 2)
            self.assertEqual(manifest["selection"]["examined_candidates"], 2)

    def test_requires_a_pinned_revision_and_official_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            objects, published, associations = self._write_fixture(root)
            original_objects = objects.read_bytes()
            with self.assertRaisesRegex(CorpusBuildError, "must not overwrite"):
                prepare_nga_visual_subset(
                    objects,
                    published,
                    objects,
                    object_associations_csv=associations,
                    source_revision=REVISION,
                    preflight=False,
                )
            self.assertEqual(objects.read_bytes(), original_objects)
            with self.assertRaisesRegex(
                CorpusBuildError, "pinned 40-character git SHA"
            ):
                prepare_nga_visual_subset(
                    objects,
                    published,
                    root / "moving.csv",
                    object_associations_csv=associations,
                    source_revision="main",
                    preflight=False,
                )

            malformed = root / "bad-images.csv"
            _write_csv(malformed, ("uuid",), [{"uuid": "not-enough"}])
            with self.assertRaisesRegex(CorpusBuildError, "missing required fields"):
                prepare_nga_visual_subset(
                    objects,
                    malformed,
                    root / "bad.csv",
                    object_associations_csv=associations,
                    source_revision=REVISION,
                    preflight=False,
                )

            malformed_associations = root / "bad-associations.csv"
            _write_csv(
                malformed_associations,
                ("parentobjectid",),
                [{"parentobjectid": "1"}],
            )
            with self.assertRaisesRegex(CorpusBuildError, "missing required fields"):
                prepare_nga_visual_subset(
                    objects,
                    published,
                    root / "bad-associations-output.csv",
                    object_associations_csv=malformed_associations,
                    source_revision=REVISION,
                    preflight=False,
                )


if __name__ == "__main__":
    unittest.main()
