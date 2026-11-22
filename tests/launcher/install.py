#!/usr/bin/env python3
"""Exercise install.sh with local releases, a real vf venv and stand-ins for CUDA files."""
import hashlib
import http.server
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
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]
INSTALLER = Path(os.environ.get("VF_TEST_INSTALLER", ROOT / "install.sh")).resolve()
SHELL = os.environ.get("VF_TEST_SHELL", "dash")


class Installer(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="vf-install-test-")
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.prefix = self.root / "install prefix"
        self.bin = self.root / "user bin"
        self.base = self.root / "releases"
        self.base.mkdir()
        self.releases = {}
        for version in ("1.0.0", "1.1.0"):
            name = f"volta-forever-{version}-linux-x86_64"
            pkg = self.root / name
            (pkg / "bin").mkdir(parents=True)
            shutil.copy2(ROOT / "scripts/vf", pkg / "bin/vf")
            (pkg / "bin/volta-forever").symlink_to("vf")
            (pkg / "scripts").mkdir()
            (pkg / "scripts/fetch-redist.py").write_text(
                "from pathlib import Path\nimport sys\n"
                "lib = Path(sys.argv[4]) / 'lib'\nlib.mkdir(parents=True, exist_ok=True)\n"
                "(lib/'libcublas.so.12').write_bytes(b'CUDA fixture')\n"
                "(lib/'libcublasLt.so.12').write_bytes(b'CUDA fixture')\n"
                "(lib/'libnvJitLink.so.12').write_bytes(b'CUDA fixture')\n"
            )
            (pkg / "VERSION").write_text(f"{version} fixture\n")
            archive = self.base / (name + ".tar.gz")
            with tarfile.open(archive, "w:gz") as out:
                out.add(pkg, arcname=name)
            with (self.base / "SHA256SUMS").open("a") as sums:
                sums.write(f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n")
            self.releases[version] = pkg
        self.env = os.environ.copy()
        self.env.update(VF_PREFIX=str(self.prefix), VF_BIN=str(self.bin),
                        VF_INSTALL_FETCH="0", VF_INSTALL_BASE=self.base.as_uri(), VF_QUIET="1")
        self.shims = self.root / "shims"
        self.shims.mkdir()
        # Pause a particular system call, without a timing dependency or an install.sh test hook.
        for tool in ("flock", "mv"):
            real = shutil.which(tool)
            path = self.shims / tool
            path.write_text(f"""#!{sys.executable}
import os, pathlib, subprocess, sys
argv = sys.argv[1:]
mode = os.environ.get("SYNC_MODE")
sync = pathlib.Path(os.environ["SYNC_DIR"])
if {tool!r} == "flock" and mode == "observer":
    (sync / "second-lock").touch()
src, dst = map(pathlib.Path, argv[-2:]) if len(argv) >= 2 else (None, None)
payload = {tool!r} == "mv" and src is not None and (src / "bin/vf").is_file()
if {tool!r} == "mv" and mode == "interrupt" and (str(dst) == os.environ["VF_BIN"] + "/vf" or str(dst) == os.environ["LEGACY_DIR"]):
    (sync / "ready").touch()
    with (sync / "resume").open() as fifo:
        fifo.read()
rc = subprocess.run([{real!r}, *argv], close_fds=False).returncode
if payload and mode == "publisher" and rc == 0 and not (sync / "ready").exists():
    (sync / "ready").touch()
    with (sync / "resume").open() as fifo:
        fifo.read()
sys.exit(rc)
""")
            path.chmod(0o755)

    def install(self, version, mode=None, **settings):
        env = dict(self.env, VF_VERSION=version)
        if mode:
            env.update(PATH=str(self.shims) + ":" + env["PATH"], SYNC_MODE=mode,
                       SYNC_DIR=str(self.root), LEGACY_DIR=str(self.prefix / self.releases[version].name))
        env.update(settings)
        shell = [SHELL, "--posix"] if SHELL == "bash" else [SHELL]
        return subprocess.Popen([*shell, str(INSTALLER)], env=env, start_new_session=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def success(self, proc):
        output, _ = proc.communicate(timeout=30)
        self.assertEqual(proc.returncode, 0, output)

    def legacy(self):
        self.prefix.mkdir()
        self.bin.mkdir()
        first = self.prefix / self.releases["1.0.0"].name
        shutil.copytree(self.releases["1.0.0"], first, symlinks=True)
        for binary in ("vf", "volta-forever"):
            (self.bin / binary).symlink_to(first / "bin" / binary)
        lib = first / "cuda-12.9.2/lib"
        lib.mkdir(parents=True)
        (lib / "libcublas.so.12.9").write_bytes(b"CUDA fixture")
        (lib / "libcublas.so.12").symlink_to("libcublas.so.12.9")
        return first

    def check_venv(self, venv, version):
        check = "import os;from pathlib import Path;h=Path(os.environ['VOLTA_FOREVER_HOME']);" \
                "assert (h/'cuda-12.9.2/lib/libcublas.so.12').read_bytes()==b'CUDA fixture';" \
                "print(os.environ['VOLTA_FOREVER'])"
        actual = subprocess.check_output([venv / "bin/python", "-c", check], env=self.env, text=True)
        self.assertEqual(actual.strip(), version)

    def serve(self, asked, redirect=None, tls=None):
        """A server of the releases on 127.0.0.1, or of redirections to redirect + the path."""
        base = str(self.base)

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=base, **kwargs)

            def do_GET(self):
                asked.append(self.path)
                if redirect:
                    self.send_response(302)
                    self.send_header("Location", redirect + self.path)
                    self.end_headers()
                else:
                    super().do_GET()

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        if tls:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(*tls)
            server.socket = context.wrap_socket(server.socket, server_side=True)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_port

    def wait_for(self, condition, *procs):
        until = time.monotonic() + 15
        while not condition():
            if time.monotonic() >= until:
                for proc in procs:
                    self.stop(proc)
                self.fail("installer synchronization timed out")
            time.sleep(0.02)

    @staticmethod
    def stop(proc):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate(timeout=10)

    def test_upgrade_reuses_bytes_and_keeps_venvs(self):
        legacy = self.legacy()
        venv_old = self.root / "old-venv"
        venv_new = self.root / "new-venv"
        subprocess.run([self.bin / "vf", "venv", venv_old, "--", "--without-pip"], env=self.env,
                       check=True, stdout=subprocess.DEVNULL)
        self.check_venv(venv_old, "1.0.0")
        self.success(self.install("1.1.0"))
        new = self.prefix / self.releases["1.1.0"].name
        lib = Path("cuda-12.9.2/lib/libcublas.so.12")
        self.assertTrue((legacy / lib).is_file(), "the upgrade removed CUDA from the old release")
        self.assertTrue(os.path.samefile(legacy / lib, new / lib), "the upgrade duplicated the CUDA payload")
        self.assertTrue(os.path.samefile(legacy / "cuda-12.9.2", new / "cuda-12.9.2"),
                        "manual fetches must share one directory and the same download lock")
        self.check_venv(venv_old, "1.0.0")
        # Activated, the old environment is named by vf version and vf doctor of the new release.
        activated = dict(self.env, VIRTUAL_ENV=str(venv_old))
        shown = subprocess.run([self.bin / "vf", "version"], env=activated, capture_output=True, text=True)
        self.assertIn("1.1.0", shown.stdout)
        self.assertIn(f"{venv_old} runs through the release 1.0.0", shown.stderr)
        doctor = subprocess.run([self.bin / "vf", "doctor"], env=activated, capture_output=True, text=True)
        self.assertIn(f"NOTICE the environment {venv_old} runs through the release 1.0.0", doctor.stdout)
        subprocess.run([self.bin / "vf", "venv", venv_new, "--", "--without-pip"], env=self.env,
                       check=True, stdout=subprocess.DEVNULL)
        self.success(self.install("1.1.0"))
        self.success(self.install("1.0.0"))
        self.check_venv(venv_new, "1.1.0")
        self.check_venv(venv_old, "1.0.0")

    def interrupted_install(self, version):
        legacy = self.legacy()
        os.mkfifo(self.root / "resume")
        proc = self.install(version, "interrupt")
        self.addCleanup(self.stop, proc)
        self.wait_for((self.root / "ready").exists, proc)
        self.stop(proc)
        for cmd in ("vf", "volta-forever"):
            self.assertTrue((self.bin / cmd).is_file(), "the active launcher disappeared during the install")
            self.assertIn("1.0.0", subprocess.check_output([self.bin / cmd, "version"], text=True))
        self.assertTrue((legacy / "cuda-12.9.2/lib/libcublas.so.12").is_file())
        self.success(self.install(version))

    def test_interrupted_reinstall_preserves_live_commands(self):
        self.interrupted_install("1.0.0")

    def test_interrupted_upgrade_preserves_selected_release(self):
        self.interrupted_install("1.1.0")

    def concurrent_publishers(self, second_version):
        self.legacy()
        os.mkfifo(self.root / "resume")
        first = self.install("1.1.0", "publisher")
        self.addCleanup(self.stop, first)
        self.wait_for((self.root / "ready").exists, first)
        second = self.install(second_version, "observer")
        self.addCleanup(self.stop, second)
        self.wait_for(lambda: (self.root / "second-lock").exists() or second.poll() is not None, first, second)
        with (self.root / "resume").open("w") as fifo:
            fifo.write("continue\n")
        self.success(first)
        self.success(second)
        versions = [subprocess.check_output([self.bin / cmd, "version"], text=True)
                    for cmd in ("vf", "volta-forever")]
        self.assertEqual(*versions)
        self.assertTrue(any(v in versions[0] for v in ("1.1.0", second_version)))
        for pkg in self.releases.values():
            self.assertTrue((self.prefix / pkg.name / "cuda-12.9.2/lib/libcublas.so.12").is_file())


    def test_concurrent_reinstall(self):
        self.concurrent_publishers("1.1.0")

    def test_concurrent_versions(self):
        self.concurrent_publishers("1.0.0")

    def test_leftovers_of_killed_installers_are_removed(self):
        # Those not changed for a day; a younger one may be another installer's.
        self.prefix.mkdir()
        self.bin.mkdir()
        old = [self.prefix / ".vf-install.Old12345", self.bin / ".vf-links.Old12345"]
        young = [self.prefix / ".vf-install.New12345", self.bin / ".vf-links.New12345"]
        for path in old + young:
            (path / "volta-forever-0.0.1-linux-x86_64").mkdir(parents=True)
        for path in old:
            os.utime(path, (time.time() - 2 * 86400,) * 2)
        self.success(self.install("1.0.0"))
        self.assertEqual([path for path in old if os.path.lexists(path)], [])
        self.assertEqual([path for path in young if not path.is_dir()], [])
        self.assertTrue((self.prefix / ".vf-install.lock").is_file())

    def test_plain_http_refused_unless_allowed(self):
        asked = []
        base = f"http://127.0.0.1:{self.serve(asked)}"
        proc = self.install("1.0.0", VF_INSTALL_BASE=base, no_proxy="*")
        output, _ = proc.communicate(timeout=30)
        self.assertNotEqual(proc.returncode, 0, output)
        self.assertIn("VF_INSTALL_ALLOW_HTTP", output)
        self.assertEqual(asked, [])
        self.assertFalse((self.bin / "vf").exists())
        self.success(self.install("1.0.0", VF_INSTALL_BASE=base, VF_INSTALL_ALLOW_HTTP="1", no_proxy="*"))
        self.assertIn("1.0.0", subprocess.check_output([self.bin / "vf", "version"], text=True))

    def test_https_redirections(self):
        # Over HTTPS, a redirection to plain HTTP is refused, by curl and by wget; one to
        # HTTPS is followed.
        openssl = shutil.which("openssl")
        if openssl is None:
            self.skipTest("no openssl to make the server's certificate")
        key, cert = self.root / "key.pem", self.root / "cert.pem"
        subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key, "-out", cert,
                        "-days", "1", "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1"],
                       check=True, capture_output=True)
        (self.root / "wgetrc").write_text(f"ca_certificate = {cert}\n")
        trust = dict(CURL_CA_BUNDLE=str(cert), WGETRC=str(self.root / "wgetrc"), no_proxy="*")
        asked = []
        secure = self.serve(asked, tls=(cert, key))
        to_plain = self.serve(asked, redirect=f"http://127.0.0.1:{self.serve(asked)}", tls=(cert, key))
        to_secure = self.serve(asked, redirect=f"https://127.0.0.1:{secure}", tls=(cert, key))
        # The commands of PATH without curl, for wget.
        without_curl = self.root / "without-curl"
        without_curl.mkdir()
        for directory in self.env["PATH"].split(os.pathsep):
            try:
                entries = list(os.scandir(directory))
            except OSError:
                continue
            for entry in entries:
                link = without_curl / entry.name
                if entry.name != "curl" and not os.path.lexists(link):
                    link.symlink_to(entry.path)
        tried = 0
        for tool, path, refusal in (("curl", self.env["PATH"], '"http" not supported'),
                                    ("wget", str(without_curl), "redirected to another scheme than https://")):
            if shutil.which(tool, path=path) is None:
                continue
            tried += 1
            for port, works in ((to_plain, False), (secure, True), (to_secure, True)):
                place = self.root / f"{tool}-{port}"
                proc = self.install("1.0.0", PATH=path, VF_INSTALL_BASE=f"https://127.0.0.1:{port}",
                                    VF_PREFIX=str(place), VF_BIN=str(place / "bin"), **trust)
                output, _ = proc.communicate(timeout=30)
                self.assertEqual(proc.returncode == 0, works, (tool, port, output))
                self.assertEqual((place / "bin/vf").exists(), works, (tool, port, output))
                if not works:
                    self.assertIn(refusal, output)
        if not tried:
            self.skipTest("neither curl nor wget")

    def test_versions_share_later_fetches(self):
        self.success(self.install("1.0.0"))
        self.success(self.install("1.1.0"))
        old = self.prefix / self.releases["1.0.0"].name
        new = self.prefix / self.releases["1.1.0"].name
        subprocess.run([old / "bin/vf", "fetch"], env=self.env, check=True, stdout=subprocess.DEVNULL)
        for name in ("libcublas.so.12", "libcublasLt.so.12", "libnvJitLink.so.12"):
            path = Path("cuda-12.9.2/lib") / name
            self.assertTrue(os.path.samefile(old / path, new / path))


if __name__ == "__main__":
    unittest.main()
