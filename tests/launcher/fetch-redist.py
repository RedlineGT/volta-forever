#!/usr/bin/env python3
"""Offline regressions for the component downloader. Run with python3 tests/launcher/fetch-redist.py."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/fetch-redist.py"


class FetchTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tree = self.root / "tree"
        self.cache = self.root / "cache"
        self.commands = self.root / "bin"
        self.commands.mkdir()
        self.tar = shutil.which("tar")
        # Exercise the urllib fallback with file:// URLs. The fixtures use plain tar.
        (self.commands / "tar").symlink_to(self.tar)
        self.manifest = {}

    def component(self, name, *, license=True, data=b"library"):
        path = self.root / (name + ".tar")
        with tarfile.open(path, "w") as archive:
            for member in [f"lib/{name}.so", f"include/{name}.h"] + (["LICENSE"] if license else []):
                item = tarfile.TarInfo("pkg/" + member)
                item.size = len(data)
                archive.addfile(item, io.BytesIO(data))
        self.manifest[name] = {"version": "1", "linux-x86_64": {
            "relative_path": path.name, "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }}
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))
        return path

    def launch(self, *options):
        return subprocess.Popen(
            [sys.executable, str(SCRIPT), self.root.as_uri(), str(self.root / "manifest.json"),
             str(self.cache), str(self.tree), "--discard", *options],
            env={**os.environ, "PATH": str(self.commands)}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        )

    def finish(self, proc, *, success=True):
        output = proc.communicate(timeout=15)[0]
        self.assertEqual(proc.returncode == 0, success, output)
        return output

    def test_retry_after_failure(self):
        first = self.component("a")
        self.component("b", license=False)
        self.finish(self.launch("--runtime"), success=False)
        self.assertTrue((self.tree / "lib/a.so").exists())
        self.assertFalse((self.cache / "a.tar").exists())
        first.unlink()  # The retry must use the receipt, even if the first download is now offline.
        self.component("b")
        self.finish(self.launch("--runtime"))

    def test_runtime_to_full(self):
        archive = self.component("a")
        self.finish(self.launch("--runtime"))
        self.assertFalse((self.tree / "include/a.h").exists())
        self.finish(self.launch())
        self.assertTrue((self.tree / "include/a.h").exists())
        archive.unlink()
        self.finish(self.launch("--runtime"))
        self.finish(self.launch())

    def test_legacy_receipt_is_safe_for_runtime_only(self):
        self.component("a")
        self.finish(self.launch("--runtime"))
        record = self.tree / "COMPONENTS.tsv"
        fields = record.read_text().splitlines()[0].split("\t")
        record.write_text("\t".join(fields[:4]) + "\n")
        self.assertIn("unpacked before", self.finish(self.launch("--runtime")))
        self.finish(self.launch())
        self.assertTrue((self.tree / "include/a.h").exists())

    def test_failed_replacement_invalidates_previous_receipt(self):
        self.component("a")
        self.finish(self.launch("--runtime"))
        self.component("a", license=False, data=b"changed")
        self.finish(self.launch("--runtime"), success=False)
        self.assertEqual((self.tree / "lib/a.so").read_bytes(), b"changed")
        self.component("a")
        self.finish(self.launch("--runtime"))
        self.assertEqual((self.tree / "lib/a.so").read_bytes(), b"library")

    def test_nccl_wheel_runtime_and_retry(self):
        archive = self.root / "nvidia_nccl_cu12-test.whl"
        license_name = "nvidia_nccl_cu12-2.32.6.dist-info/licenses/License.txt"
        lib = self.tree / "lib"
        lib.mkdir(parents=True)
        (lib / "libnccl.so.2.vf-part").write_bytes(b"partial from interrupted installation")
        for complete in (False, True):
            with zipfile.ZipFile(archive, "w") as wheel:
                wheel.writestr(license_name, "NCCL license")
                wheel.writestr("nvidia/nccl/include/nccl.h", "unused header")
                if complete:
                    wheel.writestr("nvidia/nccl/lib/libnccl.so.2", b"NCCL runtime")
            self.manifest["libnccl"] = {"version": "2.32.6", "linux-x86_64": {
                "relative_path": archive.name, "size": archive.stat().st_size,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }}
            (self.root / "manifest.json").write_text(json.dumps(self.manifest))
            self.finish(self.launch("--runtime", "--only", "libnccl"), success=complete)
            if not complete:
                self.assertFalse((self.tree / "lib/libnccl.so.2").exists())
                self.assertFalse((self.tree / "COMPONENTS.tsv").exists())
        self.assertEqual((self.tree / "lib/libnccl.so.2").read_bytes(), b"NCCL runtime")
        self.assertEqual((self.tree / "NCCL-LICENSE.txt").read_text(), "NCCL license")
        self.assertFalse((self.tree / "include/nccl.h").exists())
        self.assertFalse((lib / "libnccl.so.2.vf-part").exists())
        archive.unlink()
        self.assertIn("unpacked before", self.finish(self.launch("--runtime", "--only", "libnccl")))

    def test_concurrent_installs(self):
        self.component("a")
        # Slow extraction so both invocations overlap at the component being installed.
        (self.commands / "tar").unlink()
        wrapper = self.commands / "tar"
        wrapper.write_text(f'#!/bin/sh\n/bin/sleep 0.3\nexec "{self.tar}" "$@"\n')
        wrapper.chmod(0o755)
        processes = [self.launch("--runtime"), self.launch("--runtime")]
        outputs = [(p.communicate(timeout=15)[0], p.returncode) for p in processes]
        self.assertEqual([code for _, code in outputs], [0, 0], outputs)
        self.assertEqual(sum("get  a" in output for output, _ in outputs), 1, outputs)


if __name__ == "__main__":
    unittest.main()
