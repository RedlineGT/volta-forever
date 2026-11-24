"""CPU regression for the report at exit written to a file while the process runs (VF_SUMMARY).

With VF_SUMMARY=FILE, the report that the layer prints when the process exits (here: the kernels
with emulated instructions given to the program, and with VF_STATS the calls counted) is also in
FILE.PID, rewritten every VF_SUMMARY_INTERVAL seconds when it changed and at exit: a process killed
with SIGKILL leaves the report as it was. Without the variable no thread is started. A child made
by fork() writes no report of its own, and does not stop on the lock of the writing thread. The
file is readable by its owner only, and no link put where it goes, or where it was written first,
leads the report into another file; nor one put where the profile goes (VF_PROFILE=FILE, FILE.PID at
exit). A stand-in driver (link_mock_driver.c) loads the modules.

Run with python3 tests/proxy/summary_file.py path/to/libcuda.so.1.
"""
import ctypes as c
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time

HEADER = b".version 8.8\n.target sm_75\n.address_size 64\n"


def emulated(i):
    """A kernel with an emulated instruction (tanh.approx.f32), of its own name."""
    return HEADER + b".visible .entry k%d()\n{\n .reg .f32 %%f<3>;\n tanh.approx.f32 %%f1, %%f2;\n ret;\n}\n" % i


def threads():
    return len(os.listdir("/proc/self/task"))


def child(proxy, mode):
    before = threads()
    api = c.CDLL(proxy)
    print(f"threads {before} {threads()}", flush=True)
    module, function = c.c_void_p(), c.c_void_p()

    def give(i):
        assert api.cuModuleLoadData(c.byref(module), emulated(i)) == 0
        assert api.cuModuleGetFunction(c.byref(function), module, b"k%d" % i) == 0
    give(1)
    if mode == "fork":
        # Children made while the thread writes the report, each giving a kernel (which takes the
        # announcer's lock that the thread reads) and exiting.
        pids = []
        for i in range(40):
            pid = os.fork()
            if pid == 0:
                give(2 + i % 3)
                sys.exit(0)
            pids.append(pid)
            time.sleep(0.05)
        for pid in pids:
            _, status = os.waitpid(pid, 0)
            assert status == 0, status
        print("children " + " ".join(map(str, pids)), flush=True)
    if mode == "exit":
        give(2)
        return
    if mode == "wait-exit":
        print("ready", flush=True)
        sys.stdin.readline()
        return
    print("ready", flush=True)
    if mode == "kill":
        sys.stdin.readline()
        give(2)
        print("ready", flush=True)
    time.sleep(600)


def wait_for(path, text, seconds=15):
    end = time.time() + seconds
    while time.time() < end:
        if path.exists() and text in path.read_text():
            return path.read_text()
        time.sleep(0.1)
    raise AssertionError(f"{path}: no '{text}' after {seconds} s: " + (path.read_text() if path.exists() else "no file"))


if __name__ == "__main__":
    if len(sys.argv) == 3:
        child(sys.argv[1], sys.argv[2])
        sys.exit(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mock = str(tmp / "driver.so")
        subprocess.run(["cc", "-shared", "-fPIC", "-o", mock, Path(__file__).with_name("link_mock_driver.c")],
                       check=True)
        base = dict(os.environ, VF_REAL_LIBCUDA=mock, VF_PROXY_CACHE="0")
        for name in ("VF_LOG", "VF_LOG_LEVEL", "VF_SUMMARY", "VF_SUMMARY_INTERVAL", "VF_STATS",
                     "VF_EMULATION_WARNINGS"):
            base.pop(name, None)
        summary = tmp / "summary"
        start = lambda mode, stderr=subprocess.DEVNULL, **env: subprocess.Popen(
            [sys.executable, __file__, sys.argv[1], mode], env=dict(base, **env), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=stderr, text=True)

        # Killed with SIGKILL: the file has the report as it was, rewritten when it changed.
        p = start("kill", VF_SUMMARY=str(summary), VF_SUMMARY_INTERVAL="1")
        before, after = map(int, p.stdout.readline().split()[1:])
        assert after == before + 1, (before, after)
        assert p.stdout.readline() == "ready\n"
        path = Path(f"{summary}.{p.pid}")
        text = wait_for(path, "1 kernel with emulated instructions was given to the program: 1 tanh")
        assert f"report of process {p.pid}, running for " in text.splitlines()[0], text
        p.stdin.write("more\n")
        p.stdin.flush()
        assert p.stdout.readline() == "ready\n"
        wait_for(path, "2 kernels with emulated instructions were given to the program: 2 tanh")
        p.send_signal(signal.SIGKILL)
        assert p.wait() == -signal.SIGKILL
        text = path.read_text()
        assert "running for" in text and "2 kernels with emulated" in text and "at its exit" not in text, text
        print("PASS a process killed with SIGKILL leaves its report, as it was when it last changed")

        # Links put, before the first write, where the file goes and where it was written first
        # (until 2026-11-24, FILE.PID.tmp): neither leads the report into the files they name.
        victims = [tmp / "victim-final", tmp / "victim-temporary"]
        for victim in victims:
            victim.write_text("victim\n")
        p = start("kill", VF_SUMMARY=str(summary), VF_SUMMARY_INTERVAL="1")
        path = Path(f"{summary}.{p.pid}")
        path.symlink_to(victims[0])
        Path(f"{path}.tmp").symlink_to(victims[1])
        p.stdout.readline()
        assert p.stdout.readline() == "ready\n"
        p.stdin.write("more\n")
        p.stdin.flush()
        assert p.stdout.readline() == "ready\n"
        end = time.time() + 15
        while time.time() < end and (path.is_symlink() or "2 kernels" not in path.read_text()):
            time.sleep(0.1)
        p.send_signal(signal.SIGKILL)
        p.wait()
        assert [victim.read_text() for victim in victims] == ["victim\n"] * 2, [v.read_text() for v in victims]
        mode = path.lstat().st_mode
        assert stat.S_ISREG(mode) and stat.S_IMODE(mode) == 0o600, oct(mode)
        assert "2 kernels with emulated" in path.read_text()
        assert os.readlink(f"{path}.tmp") == str(victims[1])
        print("PASS the file is its owner's only, and no link leads the report elsewhere")

        # The profile (VF_PROFILE): a link put where it goes, and where it was written first until
        # 2026-11-24 (FILE.PID itself, written through): neither leads it into the file it names.
        profile = tmp / "profile"
        victim = tmp / "victim-profile"
        victim.write_text("victim\n")
        p = start("wait-exit", VF_PROFILE=str(profile))
        path = Path(f"{profile}.{p.pid}")
        path.symlink_to(victim)
        p.stdout.readline()
        assert p.stdout.readline() == "ready\n"
        p.communicate("\n", timeout=60)
        assert p.returncode == 0
        assert victim.read_text() == "victim\n", victim.read_text()
        mode = path.lstat().st_mode
        assert stat.S_ISREG(mode) and stat.S_IMODE(mode) == 0o600, oct(mode)
        assert path.read_text().startswith("kernel\torigin\t") and "\nk1\t" in path.read_text(), path.read_text()
        assert sorted(f.name for f in tmp.glob(f"profile.{p.pid}*")) == [path.name]
        print("PASS the profile is its owner's only, and a link where it goes is replaced, not followed")

        # A file that cannot be written (a directory where it goes) is said once, and leaves nothing.
        p = start("kill", stderr=subprocess.PIPE, VF_SUMMARY=str(summary), VF_SUMMARY_INTERVAL="1")
        path = Path(f"{summary}.{p.pid}")
        path.mkdir()
        p.stdout.readline()
        assert p.stdout.readline() == "ready\n"
        time.sleep(1.5)
        p.stdin.write("more\n")
        p.stdin.flush()
        assert p.stdout.readline() == "ready\n"
        time.sleep(2.5)
        p.send_signal(signal.SIGKILL)
        _, err = p.communicate()
        said = [line for line in err.splitlines() if "VF_SUMMARY: cannot write" in line]
        assert len(said) == 1 and str(path) in said[0], err
        assert sorted(f.name for f in tmp.glob(f"summary.{p.pid}*")) == [path.name]
        print("PASS a file that cannot be written is said once, and leaves no temporary file")

        # Without VF_SUMMARY: no thread, no file.
        p = start("none")
        before, after = map(int, p.stdout.readline().split()[1:])
        assert after == before, (before, after)
        assert p.stdout.readline() == "ready\n"
        p.send_signal(signal.SIGKILL)
        p.wait()
        assert list(tmp.glob(f"summary.{p.pid}*")) == []
        print("PASS without VF_SUMMARY, no thread and no file")

        # An exit: the report at exit, with the calls counted (VF_STATS), before the first interval.
        p = start("exit", VF_SUMMARY=str(summary), VF_STATS="1")
        out, _ = p.communicate(timeout=60)
        assert p.returncode == 0, out
        text = Path(f"{summary}.{p.pid}").read_text()
        lines = text.splitlines()
        assert lines[0] == f"[volta-forever] report of process {p.pid}, at its exit", text
        assert "2 kernels with emulated instructions were given to the program: 2 tanh" in lines[1], text
        assert lines[2].startswith("[volta-forever] calls: ") and "cuModuleLoadData=2" in lines[2], text
        print("PASS at exit, the report as the log has it")

        # Children made by fork(): they end, and write no report.
        p = start("fork", VF_SUMMARY=str(summary), VF_SUMMARY_INTERVAL="1")
        p.stdout.readline()
        line = p.stdout.readline()
        assert line.startswith("children "), line
        assert p.stdout.readline() == "ready\n"
        wait_for(Path(f"{summary}.{p.pid}"), "1 kernel with emulated instructions was given")
        p.send_signal(signal.SIGKILL)
        p.wait()
        written = sorted(f.name for f in tmp.glob("summary.*"))
        for pid in line.split()[1:]:
            assert not any(name.startswith(f"summary.{pid}") for name in written), written
        print("PASS a child made by fork() ends, and writes no report of its own")
