#!/usr/bin/env python3
"""Exercise install.sh with local releases, a real vf venv and stand-ins for CUDA files."""
import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
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

    def install(self, version, mode=None):
        env = dict(self.env, VF_VERSION=version)
        if mode:
            env.update(PATH=str(self.shims) + ":" + env["PATH"], SYNC_MODE=mode,
                       SYNC_DIR=str(self.root), LEGACY_DIR=str(self.prefix / self.releases[version].name))
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
