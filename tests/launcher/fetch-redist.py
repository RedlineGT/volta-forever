#!/usr/bin/env python3
"""Offline regressions for the component downloader. Run with python3 tests/launcher/fetch-redist.py."""
import hashlib
import http.server
import io
import json
import os
from pathlib import Path
import shutil
import signal
import ssl
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
from unittest import mock
import zipfile


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/fetch-redist.py"
# The script, given the signal VF_TEST_SIGNAL when it starts to write the NCCL runtime.
INTERRUPTED = """
import os, runpy, shutil, signal, sys
real = shutil.copyfileobj
def interrupted(*args, **kwargs):
    os.kill(os.getpid(), getattr(signal, os.environ["VF_TEST_SIGNAL"]))
    return real(*args, **kwargs)
shutil.copyfileobj = interrupted
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""

# The script, given the signal VF_TEST_SIGNAL once it has moved the first staged file into place.
PUBLISHING = """
import os, runpy, signal, sys
real = os.replace
def replace(source, *args, **kwargs):
    result = real(source, *args, **kwargs)
    if "/.vf-stage-" in str(source) and not os.environ.get("VF_TEST_SENT"):
        os.environ["VF_TEST_SENT"] = "1"
        os.kill(os.getpid(), getattr(signal, os.environ["VF_TEST_SIGNAL"]))
    return result
os.replace = replace
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


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

    def nccl_wheel(self):
        archive = self.root / "nvidia_nccl_cu12-test.whl"
        with zipfile.ZipFile(archive, "w") as wheel:
            wheel.writestr("nvidia_nccl_cu12-2.32.6.dist-info/licenses/License.txt", "NCCL license")
            wheel.writestr("nvidia/nccl/lib/libnccl.so.2", b"NCCL runtime")
        self.manifest["libnccl"] = {"version": "2.32.6", "linux-x86_64": {
            "relative_path": archive.name, "size": archive.stat().st_size,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        }}
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def launch(self, *options, interrupt=None, manifest=None, base=None, wrapper=INTERRUPTED):
        command, env = [sys.executable, str(SCRIPT)], {**os.environ, "PATH": str(self.commands)}
        if interrupt:
            command[1:1] = ["-c", wrapper]
            env["VF_TEST_SIGNAL"] = interrupt
        return subprocess.Popen(
            [*command, base or self.root.as_uri(), manifest or str(self.root / "manifest.json"),
             str(self.cache), str(self.tree), "--discard", *options],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
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
        self.assertFalse((self.tree / "lib/b.so").exists(), "a failed unpacking changed the tree")
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

    def test_failed_replacement_keeps_previous_version(self):
        # The new archive fails half-way (its library unpacked, its licence missing): the tree
        # keeps the previous version, and its receipt, rather than a mix of the two.
        self.component("a")
        self.finish(self.launch("--runtime"))
        self.component("a", license=False, data=b"changed")
        self.finish(self.launch("--runtime"), success=False)
        self.assertEqual((self.tree / "lib/a.so").read_bytes(), b"library")
        self.assertEqual(list(self.tree.glob(".vf-stage-*")), [])
        self.component("a")
        self.assertIn("unpacked before", self.finish(self.launch("--runtime")))
        self.assertEqual((self.tree / "lib/a.so").read_bytes(), b"library")
        self.component("a", data=b"changed")
        self.finish(self.launch("--runtime"))
        self.assertEqual((self.tree / "lib/a.so").read_bytes(), b"changed")

    def test_links_replace_links(self):
        # A library's links are moved as links, after the files that they name.
        for version in ("1.2", "1.3"):
            path = self.root / "x.tar"
            with tarfile.open(path, "w") as archive:
                for member in (f"lib/x.so.{version}", "LICENSE"):
                    item = tarfile.TarInfo("pkg/" + member)
                    item.size = len(version)
                    archive.addfile(item, io.BytesIO(version.encode()))
                for name, target in (("lib/x.so.1", f"x.so.{version}"), ("lib/x.so", "x.so.1")):
                    item = tarfile.TarInfo("pkg/" + name)
                    item.type, item.linkname = tarfile.SYMTYPE, target
                    archive.addfile(item)
            self.manifest["x"] = {"version": version, "linux-x86_64": {
                "relative_path": path.name, "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }}
            (self.root / "manifest.json").write_text(json.dumps(self.manifest))
            self.finish(self.launch("--runtime"))
            self.assertEqual(os.readlink(self.tree / "lib/x.so.1"), f"x.so.{version}")
            self.assertEqual(os.readlink(self.tree / "lib/x.so"), "x.so.1")
            self.assertEqual((self.tree / "lib/x.so").read_text(), version)

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
            output = self.finish(self.launch("--runtime", "--only", "libnccl"), success=complete)
            if not complete:
                # (A message, not a KeyError and its traceback, as until 2026-11-23.)
                self.assertIn("the NCCL wheel has no nvidia/nccl/lib/libnccl.so.2", output)
                self.assertNotIn("Traceback", output)
                self.assertFalse((self.tree / "lib/libnccl.so.2").exists())
                self.assertFalse((self.tree / "COMPONENTS.tsv").exists())
        self.assertEqual((self.tree / "lib/libnccl.so.2").read_bytes(), b"NCCL runtime")
        self.assertEqual((self.tree / "NCCL-LICENSE.txt").read_text(), "NCCL license")
        self.assertFalse((self.tree / "include/nccl.h").exists())
        self.assertFalse((lib / "libnccl.so.2.vf-part").exists())
        archive.unlink()
        self.assertIn("unpacked before", self.finish(self.launch("--runtime", "--only", "libnccl")))

    def test_interrupted_nccl_leaves_no_library_name(self):
        # Killed while it writes the NCCL runtime: nothing is left in lib/ that the launcher
        # could count as a library. SIGTERM ends it as Ctrl-C does, and it removes what it
        # staged; after SIGKILL, the next fetch does.
        self.nccl_wheel()
        for name, status in (("SIGTERM", 128 + signal.SIGTERM), ("SIGKILL", -signal.SIGKILL)):
            proc = self.launch("--runtime", "--only", "libnccl", interrupt=name)
            output = proc.communicate(timeout=15)[0]
            self.assertEqual(proc.returncode, status, output)
            lib = self.tree / "lib"
            self.assertEqual(sorted(lib.iterdir()) if lib.exists() else [], [], name)
            if name == "SIGTERM":
                self.assertEqual(list(self.tree.glob(".*")), [self.tree / ".vf-fetch.lock"])
        self.finish(self.launch("--runtime", "--only", "libnccl"))
        self.assertEqual((self.tree / "lib/libnccl.so.2").read_bytes(), b"NCCL runtime")
        self.assertEqual(list(self.tree.glob(".vf-stage-*")), [])

    def test_interrupted_publication(self):
        # Killed once the first file of a new version is in place: the journal says that the
        # tree is being changed (the launcher refuses it), and the next fetch completes it. Given
        # SIGTERM there, the fetch puts the whole version in place first. (Until 2026-11-23 the
        # tree was left with files of two versions, and nothing said so.)
        self.component("a")
        self.finish(self.launch("--runtime"))
        journal, record = self.tree / ".vf-publishing", self.tree / "COMPONENTS.tsv"
        for name, status in (("SIGKILL", -signal.SIGKILL), ("SIGTERM", 128 + signal.SIGTERM)):
            data = name.encode()
            archive = self.component("a", data=data)
            proc = self.launch("--runtime", interrupt=name, wrapper=PUBLISHING)
            output = proc.communicate(timeout=15)[0]
            self.assertEqual(proc.returncode, status, output)
            files = {(self.tree / "lib/a.so").read_bytes(), (self.tree / "LICENSE").read_bytes()}
            if name == "SIGKILL":
                self.assertEqual(len(files), 2, "the first file is not in place: the test tests nothing")
                self.assertTrue(journal.exists(), "files of two versions in the tree, and nothing says so")
                self.finish(self.launch("--runtime"))
            self.assertFalse(journal.exists(), name)
            self.assertEqual((self.tree / "lib/a.so").read_bytes(), data, name)
            self.assertEqual((self.tree / "LICENSE").read_bytes(), data, name)
            self.assertEqual(list(self.tree.glob(".vf-stage-*")), [], name)
            sha = hashlib.sha256(archive.read_bytes()).hexdigest()
            self.assertEqual(record.read_text().split("\t")[3], sha, name)
            self.assertIn("unpacked before", self.finish(self.launch("--runtime")), name)

    def test_publication_whose_staging_is_gone(self):
        # A journal whose staging directory is gone (removed by hand): the component is put in
        # place again, first, when it is fetched; another fetch stops and says what to fetch.
        self.component("a")
        self.component("b")
        self.finish(self.launch("--runtime"))
        journal = self.tree / ".vf-publishing"
        journal.write_text(json.dumps({"stage": ".vf-stage-gone", "receipt": ["a", "1", "a.tar", "0" * 64, "runtime"]}))
        self.assertIn("fetch a again (--only '^a$')", self.finish(self.launch("--runtime", "--only", "^b$"), success=False))
        self.assertTrue(journal.exists())
        output = self.finish(self.launch("--runtime"))
        self.assertIn("a is put in place again", output)
        self.assertIn("get  a", output)
        self.assertIn("have b 1 (unpacked before)", output)
        self.assertFalse(journal.exists())
        self.assertEqual((self.tree / "lib/a.so").read_bytes(), b"library")

    def test_leftovers_are_removed(self):
        self.component("a")
        leftovers = [self.tree / "lib/libnccl.so.2.vf-part", self.tree / "NCCL-LICENSE.txt.vf-part",
                     self.tree / ".vf-stage-old/lib/a.so", self.cache / ".dl-old"]
        for path in leftovers:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"left by an interrupted fetch")
        self.finish(self.launch("--runtime"))
        self.assertEqual([path for path in leftovers if path.exists()], [])
        self.assertFalse((self.tree / ".vf-stage-old").exists())
        self.assertEqual((self.tree / "lib/a.so").read_bytes(), b"library")

    def test_manifest_over_http(self):
        # Refused before anything is asked of the server, unless --allow-http.
        self.component("a")
        root, asked = str(self.root), []

        class Server(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=root, **kwargs)

            def do_GET(self):
                asked.append(self.path)
                super().do_GET()

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_port}/manifest.json"
        with mock.patch.dict(os.environ, no_proxy="*"):
            self.assertIn("--allow-http", self.finish(self.launch("--runtime", manifest=url), success=False))
            self.assertEqual(asked, [])
            self.finish(self.launch("--runtime", "--allow-http", manifest=url))
        self.assertEqual(asked, ["/manifest.json"])
        self.assertEqual((self.tree / "lib/a.so").read_bytes(), b"library")

    def serve(self, tls=False, moved_to=None):
        """A server of self.root on 127.0.0.1 (with TLS, by a certificate that the fetches are
        given to trust); /moved/X redirects to moved_to + X. Returns its URL and the paths asked."""
        root, asked = str(self.root), []

        class Server(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=root, **kwargs)

            def do_GET(self):
                asked.append(self.path)
                if moved_to and self.path.startswith("/moved/"):
                    self.send_response(302)
                    self.send_header("Location", moved_to + self.path[len("/moved/"):])
                    self.end_headers()
                    return
                super().do_GET()

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Server)
        if tls:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(self.certificate, self.key)
            server.socket = context.wrap_socket(server.socket, server_side=True)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"{'https' if tls else 'http'}://127.0.0.1:{server.server_port}", asked

    @unittest.skipUnless(shutil.which("openssl"), "needs openssl to make a certificate")
    def test_redirections_from_https(self):
        # From HTTPS, a redirection to plain HTTP is refused, for the manifest and for an archive,
        # unless --allow-http, before anything is asked over HTTP. (Until 2026-11-23 urllib
        # followed it.)
        self.certificate, self.key = self.root / "certificate.pem", self.root / "key.pem"
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                        "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1",
                        "-keyout", str(self.key), "-out", str(self.certificate)],
                       check=True, capture_output=True)
        self.component("a")
        plain, plain_asked = self.serve()
        secure, secure_asked = self.serve(tls=True, moved_to=plain + "/")
        with mock.patch.dict(os.environ, no_proxy="*", SSL_CERT_FILE=str(self.certificate)):
            # (The server is trusted: what is served over HTTPS without a redirection is taken.)
            self.finish(self.launch("--runtime", manifest=secure + "/manifest.json", base=secure))
            shutil.rmtree(self.tree)
            shutil.rmtree(self.cache)
            for manifest, base in [(secure + "/moved/manifest.json", secure), (None, secure + "/moved")]:
                plain_asked.clear()
                output = self.finish(self.launch("--runtime", manifest=manifest, base=base), success=False)
                self.assertIn("is refused (--allow-http takes plain HTTP)", output)
                self.assertEqual(plain_asked, [])
                self.assertFalse((self.tree / "lib/a.so").exists())
                self.finish(self.launch("--runtime", "--allow-http", manifest=manifest, base=base))
                self.assertNotEqual(plain_asked, [])
                self.assertEqual((self.tree / "lib/a.so").read_bytes(), b"library")
                shutil.rmtree(self.tree)
                shutil.rmtree(self.cache)
        self.assertIn("/moved/manifest.json", secure_asked)
        self.assertIn("/moved/a.tar", secure_asked)

    def test_redirections_that_aria2c_followed(self):
        # aria2c cannot be told not to follow a redirection: one that it followed from HTTPS to
        # HTTP refuses the archive afterwards, unless --allow-http. (A stand-in for aria2c, which
        # copies the archive and writes the line of its log of a redirection.)
        archive = self.component("a")
        for target, allowed in [("https://mirror.invalid/a.tar", True), ("http://mirror.invalid/a.tar", False)]:
            aria2c = self.commands / "aria2c"
            aria2c.write_text(
                "#!/bin/sh\nfor arg; do case $arg in --log=*) log=${arg#--log=} ;; esac; done\n"
                'while [ $# -gt 1 ]; do case $1 in -d) dir=$2 ;; -o) name=$2 ;; esac; shift; done\n'
                f'{shutil.which("cp")} "{archive}" "$dir/$name"\n'
                f'echo "2026-11-23 [NOTICE] [HttpResponse.cc:168] CUID#7 - Redirecting to {target}" > "$log"\n')
            aria2c.chmod(0o755)
            for options in [(), ("--allow-http",)]:
                output = self.finish(self.launch("--runtime", *options, base="https://origin.invalid"),
                                     success=allowed or bool(options))
                if not (allowed or options):
                    self.assertIn(f"https://origin.invalid/a.tar: the redirection to {target} is refused", output)
                    self.assertEqual(list(self.cache.iterdir()), [self.cache / ".vf-fetch.lock"])
                shutil.rmtree(self.tree, ignore_errors=True)
                shutil.rmtree(self.cache, ignore_errors=True)

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
