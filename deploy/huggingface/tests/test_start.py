from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


START_PATH = Path(__file__).resolve().parents[1] / "start.py"
DEPLOY_ROOT = START_PATH.parent
DOCKERFILE_PATH = DEPLOY_ROOT / "Dockerfile"
README_PATH = DEPLOY_ROOT / "README.md"
SPEC = importlib.util.spec_from_file_location("mnemosyne_space_start", START_PATH)
assert SPEC is not None and SPEC.loader is not None
start = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(start)


def manifest_payload() -> dict[str, object]:
    return {
        "schema_version": start.EXPECTED_SCHEMA_VERSION,
        "corpus": {
            "id": start.EXPECTED_CORPUS_ID,
            "version": start.EXPECTED_CORPUS_ID,
            "count": start.EXPECTED_CORPUS_COUNT,
            "label": start.EXPECTED_CORPUS_LABEL,
            "countingUnit": start.EXPECTED_COUNTING_UNIT,
        },
        "merge": {
            "sources": [
                {
                    "bundle_index": bundle_index,
                    "institutions": list(institutions),
                    "row_start": row_start,
                    "row_end_exclusive": row_end,
                    "row_count": row_count,
                }
                for bundle_index, institutions, row_start, row_end, row_count
                in start.EXPECTED_MERGE_SOURCES
            ]
        },
        "model": {
            "id": start.EXPECTED_MODEL_ID,
            "revision": start.EXPECTED_MODEL_REVISION,
        },
        "matrix": {
            "rows": start.EXPECTED_ROWS,
            "dimensions": start.EXPECTED_DIMENSIONS,
            "dtype": "float32",
            "l2_normalized": True,
        },
        "bins": [{} for _ in range(1_703)],
        "artifacts": [
            {
                "path": f"payload/{index}.bin",
                "bytes": 1,
                "sha256": "0" * 64,
            }
            for index in range(start.EXPECTED_ARTIFACT_COUNT)
        ],
    }


def write_bundle(root: Path, payload: dict[str, object] | None = None) -> None:
    manifest = payload or manifest_payload()
    (root / "payload").mkdir(parents=True)
    for index in range(start.EXPECTED_ARTIFACT_COUNT):
        (root / "payload" / f"{index}.bin").write_bytes(b"x")
    (root / start.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


class ManifestTests(unittest.TestCase):
    def test_accepts_exact_pinned_bundle_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            write_bundle(root)

            manifest = start.load_and_validate_manifest(root)

            self.assertEqual(manifest["model"]["id"], start.EXPECTED_MODEL_ID)

    def test_rejects_wrong_model_revision(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            payload = manifest_payload()
            payload["model"]["revision"] = "moving-target"
            write_bundle(root, payload)

            with self.assertRaisesRegex(ValueError, "model revision"):
                start.load_and_validate_manifest(root)

    def test_rejects_wrong_corpus_label(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            payload = manifest_payload()
            payload["corpus"]["label"] = "The Met only"
            write_bundle(root, payload)

            with self.assertRaisesRegex(ValueError, "corpus label"):
                start.load_and_validate_manifest(root)

    def test_rejects_wrong_counting_unit(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            payload = manifest_payload()
            payload["corpus"]["countingUnit"] = "physical-object"
            write_bundle(root, payload)

            with self.assertRaisesRegex(ValueError, "counting unit"):
                start.load_and_validate_manifest(root)

    def test_rejects_wrong_merge_source_composition(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            payload = manifest_payload()
            payload["merge"]["sources"][1]["row_count"] -= 1
            write_bundle(root, payload)

            with self.assertRaisesRegex(ValueError, "merge sources"):
                start.load_and_validate_manifest(root)

    def test_rejects_artifact_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            payload = manifest_payload()
            payload["artifacts"][0]["path"] = "../secret"
            write_bundle(root, payload)

            with self.assertRaisesRegex(ValueError, "bundle root"):
                start.load_and_validate_manifest(root)


class HydrationTests(unittest.TestCase):
    def test_hydrates_declared_files_and_reuses_completed_destination(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            source = root / "source"
            destination = root / "local" / "artifacts"
            write_bundle(source)

            result = start.hydrate_artifacts(source, destination)
            reused = start.hydrate_artifacts(source, destination)

            self.assertEqual(result, destination)
            self.assertEqual(reused, destination)
            self.assertEqual((destination / "payload" / "0.bin").read_bytes(), b"x")
            self.assertTrue((destination / start.HYDRATION_MARKER).is_file())
            self.assertEqual((source / "payload" / "0.bin").read_bytes(), b"x")

    def test_copy_failure_leaves_no_visible_or_temporary_destination(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            source = root / "source"
            destination = root / "local" / "artifacts"
            write_bundle(source)

            with mock.patch.object(start.shutil, "copy2", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    start.hydrate_artifacts(source, destination)

            self.assertFalse(destination.exists())
            self.assertEqual(list(destination.parent.iterdir()), [])

    def test_source_file_symlink_cannot_escape_bundle_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            source = root / "source"
            write_bundle(source)
            outside = root / "outside.bin"
            outside.write_bytes(b"x")
            (source / "payload" / "0.bin").unlink()
            (source / "payload" / "0.bin").symlink_to(outside)

            with self.assertRaisesRegex(ValueError, "outside the bundle root"):
                start.hydrate_artifacts(source, root / "local")


class RuntimeProfileTests(unittest.TestCase):
    def test_artifact_source_uses_the_versioned_bucket_prefix(self) -> None:
        self.assertEqual(
            start.ARTIFACT_SOURCE,
            Path("/artifacts/releases") / start.EXPECTED_CORPUS_ID,
        )

    def test_exec_profile_is_one_offline_cpu_server_without_app_auth(self) -> None:
        argv = start.service_argv(Path("/tmp/test-artifacts"))

        self.assertEqual(argv[0], "mnemosyne-search")
        self.assertIn("--siglip2", argv)
        self.assertIn("--no-faiss", argv)
        self.assertEqual(argv[argv.index("--device") + 1], "cpu")
        self.assertEqual(argv[argv.index("--port") + 1], "7860")
        self.assertEqual(argv[argv.index("--http-auth-mode") + 1], "disabled")
        self.assertEqual(argv[argv.index("--max-concurrent-searches") + 1], "8")
        self.assertNotIn("--allow-model-download", argv)


class ContainerBuildTests(unittest.TestCase):
    def test_image_provisions_exact_snapshot_in_runtime_cache(self) -> None:
        dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")

        self.assertIn("snapshot_download(", dockerfile)
        self.assertIn(f'revision="{start.EXPECTED_MODEL_REVISION}"', dockerfile)
        self.assertIn("HF_HOME=/home/user/.cache/huggingface", dockerfile)
        self.assertIn("HF_HUB_OFFLINE=1", dockerfile)
        self.assertIn("TRANSFORMERS_OFFLINE=1", dockerfile)
        self.assertIn("chown -R 1000:1000 /home/user/.cache", dockerfile)

    def test_space_metadata_does_not_depend_on_platform_preload(self) -> None:
        readme = README_PATH.read_text(encoding="utf-8")

        frontmatter = readme.split("---", 2)[1]
        self.assertNotIn("preload_from_hub:", frontmatter)


if __name__ == "__main__":
    unittest.main()
