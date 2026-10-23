import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
if os.name == "posix":
    import download_model


@unittest.skipUnless(os.name == "posix", "POSIX downloader tests run in the Linux downloader image")
class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.root = self.folder / "models"
        self.data = b"valid synthetic weights"
        self.lock = {"model_id": "test/model", "revision": "a" * 40,
                     "files": {"weights.bin": {"size": len(self.data),
                                                "sha256": hashlib.sha256(self.data).hexdigest()}}}
        manifest = self.folder / "lock.json"
        manifest.write_text(json.dumps(self.lock), encoding="utf-8")
        self.enterContext(patch.dict(os.environ, {
            "MODEL_STORAGE_DIR": str(self.root), "MODEL_LOCK_PATH": str(manifest),
            "HF_HUB_OFFLINE": "0", "HF_TOKEN_FILE": str(self.folder / "absent-token")}))
        self.fetch = Mock(side_effect=self.write_download)
        self.enterContext(patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=self.fetch)}))
        self.events = self.enterContext(patch("download_model.event"))
        self.target = download_model.cache_path(self.root, self.lock)
        self.partial = self.root / (self.target.name + ".partial")

    def write_download(self, **kwargs):
        folder = Path(kwargs["local_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "weights.bin").write_bytes(self.data)

    def test_first_download_verifies_and_promotes_partial(self):
        download_model.prepare()
        self.assertEqual((self.target / "weights.bin").read_bytes(), self.data)
        self.assertFalse(self.partial.exists())
        self.assertEqual(self.fetch.call_args.kwargs["revision"], self.lock["revision"])
        self.assertFalse(self.fetch.call_args.kwargs["token"])

    def test_cache_hit_needs_neither_hub_import_nor_network(self):
        download_model.prepare()
        self.fetch.reset_mock()
        self.events.reset_mock()
        with patch.dict(sys.modules, {"huggingface_hub": None}):
            download_model.prepare()
        self.fetch.assert_not_called()
        self.events.assert_called_once_with("model_cache_hit", revision=self.lock["revision"], integrity="sha256_verified")

    def test_corrupted_final_cache_fails_before_hub_on_every_retry(self):
        self.target.mkdir(parents=True)
        damaged = self.target / "weights.bin"
        damaged.write_bytes(b"damaged")
        with patch.dict(sys.modules, {"huggingface_hub": None}):
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "Existing cache"):
                    download_model.prepare()
        self.fetch.assert_not_called()
        self.assertFalse(self.partial.exists())
        self.assertEqual(damaged.read_bytes(), b"damaged")

    def test_interrupted_download_keeps_partial_and_can_resume(self):
        def interrupt(**kwargs):
            self.partial.mkdir(parents=True)
            (self.partial / "weights.bin").write_bytes(b"partial")
            raise ConnectionError("synthetic failure")

        self.fetch.side_effect = interrupt
        with self.assertRaises(ConnectionError):
            download_model.prepare()
        self.assertTrue(self.partial.exists())
        self.assertFalse(self.target.exists())
        self.fetch.side_effect = self.write_download
        download_model.prepare()
        self.assertEqual((self.target / "weights.bin").read_bytes(), self.data)
        self.assertEqual(self.fetch.call_args_list[0].kwargs["local_dir"],
                         self.fetch.call_args_list[1].kwargs["local_dir"])

    def test_invalid_download_never_becomes_final_cache(self):
        self.fetch.side_effect = lambda **kwargs: Path(kwargs["local_dir"]).mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "Downloaded files"):
            download_model.prepare()
        self.assertFalse(self.target.exists())

    def test_offline_missing_cache_never_calls_hub(self):
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}), patch.dict(
                sys.modules, {"huggingface_hub": None}):
            with self.assertRaisesRegex(RuntimeError, "Offline cache"):
                download_model.prepare()
        self.fetch.assert_not_called()

    def test_unlisted_final_cache_file_fails_before_network(self):
        self.write_download(local_dir=self.target)
        (self.target / "model.safetensors.index.json").write_text("unverified")
        with self.assertRaisesRegex(RuntimeError, "Existing cache"):
            download_model.prepare()
        self.fetch.assert_not_called()

    def test_unlisted_partial_cache_file_fails_before_network(self):
        self.partial.mkdir(parents=True)
        (self.partial / "extra.safetensors").write_text("unverified")
        with self.assertRaisesRegex(RuntimeError, "Partial cache"):
            download_model.prepare()
        self.fetch.assert_not_called()

    def test_partial_directory_symlink_cannot_redirect_download(self):
        outside = self.folder / "outside"
        outside.mkdir()
        self.root.mkdir()
        self.partial.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "Partial cache"):
            download_model.prepare()
        self.fetch.assert_not_called()
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
