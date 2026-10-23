import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import launch
import model_cache


class CacheSecurityTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.data = b"verified synthetic weights"
        self.lock = {"model_id": "test/model", "revision": "a" * 40,
                     "files": {"model.safetensors": {"size": len(self.data),
                         "sha256": hashlib.sha256(self.data).hexdigest()}}}
        self.source = model_cache.cache_path(self.folder / "cache", self.lock)
        self.source.mkdir(parents=True)
        (self.source / "model.safetensors").write_bytes(self.data)
        self.private = self.folder / "private"

    def test_invalid_manifests_cannot_weaken_the_file_allowlist(self):
        invalid = [{}, {"*": {"size": 1, "sha256": "b" * 64}},
                   {"../weights": {"size": 1, "sha256": "b" * 64}},
                   {"weights": {"size": -1, "sha256": "b" * 64}},
                   {"weights": {"size": True, "sha256": "b" * 64}}]
        for files in invalid:
            with self.subTest(files=files):
                path = self.folder / "manifest.json"
                path.write_text(json.dumps({**self.lock, "files": files}))
                with self.assertRaises(ValueError):
                    model_cache.read_lock(path)

    def test_unlisted_loader_inputs_reject_the_cache(self):
        for name in ("model.safetensors.index.json", "extra.safetensors", "chat_template.jinja"):
            with self.subTest(name=name):
                extra = self.source / name
                extra.write_text("unverified")
                self.assertFalse(model_cache.verify_files(self.source, self.lock))
                with self.assertRaisesRegex(RuntimeError, "layout"):
                    model_cache.stage_verified_model(self.source, self.lock, self.private)
                extra.unlink()

    def test_hub_metadata_is_inert_and_never_reaches_the_engine(self):
        metadata = self.source / ".cache" / "huggingface" / "download"
        metadata.mkdir(parents=True)
        (metadata / "model.safetensors.metadata").write_text("synthetic bookkeeping")
        self.assertTrue(model_cache.verify_files(self.source, self.lock))
        target = model_cache.stage_verified_model(self.source, self.lock, self.private)
        self.assertEqual({p.name for p in target.iterdir()}, set(self.lock["files"]))
        self.assertTrue(model_cache.verify_files(target, self.lock))

    def test_snapshot_is_independent_of_later_pvc_writes(self):
        target = model_cache.stage_verified_model(self.source, self.lock, self.private)
        (self.source / "model.safetensors").write_bytes(b"damaged after copy")
        (self.source / "model.safetensors.index.json").write_text("unverified index")
        self.assertEqual((target / "model.safetensors").read_bytes(), self.data)
        self.assertTrue(model_cache.verify_files(target, self.lock))

    def test_snapshot_rechecks_bytes_instead_of_trusting_init(self):
        self.assertTrue(model_cache.verify_files(self.source, self.lock))
        (self.source / "model.safetensors").write_bytes(b"damaged")
        with self.assertRaisesRegex(RuntimeError, "integrity"):
            model_cache.stage_verified_model(self.source, self.lock, self.private)

    def test_container_restart_replaces_snapshot_without_accumulating_copies(self):
        first = model_cache.stage_verified_model(self.source, self.lock, self.private)
        second = model_cache.stage_verified_model(self.source, self.lock, self.private)
        self.assertEqual(first, second)
        self.assertEqual([p.name for p in self.private.iterdir()], ["model"])
        self.assertTrue(model_cache.verify_files(second, self.lock))

    def test_launcher_passes_only_private_verified_snapshot_to_vllm(self):
        manifest, secret = self.folder / "lock.json", self.folder / "token"
        manifest.write_text(json.dumps(self.lock))
        secret.write_text("s" * 48)
        with patch.dict(os.environ, {"MODEL_LOCK_PATH": str(manifest), "API_TOKEN_FILE": str(secret),
                "MODEL_STORAGE_DIR": str(self.source.parent), "VERIFIED_MODEL_DIR": str(self.private)}), patch(
                "launch.os.execvp") as execute, contextlib.redirect_stdout(io.StringIO()):
            launch.main()
        self.assertEqual(execute.call_args.args[1][2], str(self.private / "model"))
        self.assertNotEqual(execute.call_args.args[1][2], str(self.source))

    @unittest.skipUnless(os.name == "posix", "POSIX symlink tests run in Linux")
    def test_source_and_metadata_symlinks_are_rejected(self):
        alias = self.folder / "alias"
        alias.symlink_to(self.source, target_is_directory=True)
        self.assertFalse(model_cache.verify_files(alias, self.lock))
        metadata = self.source / ".cache"
        metadata.symlink_to(self.folder, target_is_directory=True)
        self.assertFalse(model_cache.verify_files(self.source, self.lock))

    @unittest.skipUnless(os.name == "posix", "POSIX symlink tests run in Linux")
    def test_verified_file_replaced_by_symlink_is_rejected(self):
        file = self.source / "model.safetensors"
        external = self.folder / "outside"
        external.write_bytes(self.data)
        file.unlink()
        file.symlink_to(external)
        self.assertFalse(model_cache.verify_files(self.source, self.lock))


if __name__ == "__main__":
    unittest.main()
