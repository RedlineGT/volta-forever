"""CPU regression for the lines that other libraries put in VF_SUMMARY's file (vf_summary_add, vf_summary_remove).

A library registers a function of its own (fill) with the layer's vf_summary_add, looked up on the layer's handle;
each write of FILE.PID calls it, with the layer's lock held, and writes its lines after the layer's, each with
"[NAME] " in front; vf_summary_remove takes it back, waiting for a call under way, and, in the process that loaded
the layer while the file is still to be written, calls it once more and keeps its last report ("[NAME, last
report] "). With a stand-in driver (link_mock_driver.c) and a test library (summary_fill.c):
- the return codes, the 4 slots, a name that takes its slot back once its library is removed and is refused to
  another function while it is registered; the functions found on the layer's handle, not on the driver's;
- a fill's lines in the file while the process runs, written again when they change, and after SIGKILL; fills
  that write control bytes, bytes that are not UTF-8, a line separator before text that looks like the layer's
  and a bidirectional override over their whole buffer, and return more than its size, leave the file well
  formed (every line valid UTF-8 with a known prefix, no control character, no separator at which a reader's
  splitlines would cut, no bidirectional control; the text read up to the buffer's end and no further);
- a removal waits for the call under way (a fill of 300 ms), calls the fill once more and keeps its lines,
  marked; nothing is called after it; the same name registered again takes the slot back;
- a fill that calls the layer back is refused (EDEADLK), not left waiting; fills that break the contract: one
  that never returns (a fork from another thread does not wait for it, and the last write waits 2 s for it, then
  is made without the libraries' lines and says so), one that forks at each write (no deadlock), one that calls
  exit(3) (the process ends with that status and its last write is made, saying why it lacks those lines);
- children made by fork() while the thread writes: both functions do nothing there (ECHILD), nothing is called,
  the children end and write nothing; and a child made without fork()'s handlers (clone(2) itself) while the
  thread holds the lock, which the child then has held for ever: both functions return ECHILD there at once;
- the layer's thread blocks every signal, and the thread that loaded the layer keeps its mask;
- without VF_SUMMARY: no thread and no call; without the thread (it could not be started): the lines at exit;
- at exit, in both orders of the destructors (the library loaded after the layer and before it; with glibc 2.39
  the destructors run in the order of loading, and the two cases must show both orders): when the layer's comes
  first, its last write calls the fill and the removal after it calls nothing; when the library's comes first,
  its removal calls the fill once more, and that last report is kept and written;
- a library unloaded by dlclose while the thread runs removes itself in its destructor: its last report stays
  and the process lives on; pinned (RTLD_NODELETE), it stays loaded and its lines are written as they change.
  Control: without the removal (FILL_NO_REMOVE=1 in the test library), the thread calls into the library that
  is gone, and the process ends by SIGSEGV.

Run with python3 tests/proxy/summary_fill.py path/to/libcuda.so.1.
"""
import ctypes as c
import errno
import os
from pathlib import Path
import queue
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import warnings


def threads():
    return len(os.listdir("/proc/self/task"))


def load(path, mode=os.RTLD_LOCAL):
    lib = c.CDLL(path, mode=mode)
    lib.fill_bind.argtypes = [c.c_char_p]
    lib.fill_add.argtypes = [c.c_uint, c.c_char_p, c.c_char_p, c.c_long]
    lib.fill_remove.argtypes = [c.c_char_p, c.c_long]
    lib.fill_set_count.argtypes = [c.c_ulong]
    lib.fill_calls.restype = c.c_ulong
    for f in (lib.fill_now, lib.fill_slow_start, lib.fill_slow_end):
        f.restype = c.c_longlong
    lib.fill_slow_start.argtypes = lib.fill_slow_end.argtypes = [c.c_int]
    lib.fill_at_exit.argtypes = [c.c_char_p, c.c_long]
    return lib


def blocked(task):
    """The signals that a thread of this process blocks (its SigBlk), as a number."""
    with open(f"/proc/self/task/{task}/status") as status:
        return int(next(l for l in status if l.startswith("SigBlk:")).split()[1], 16)


def child(proxy, library, mode):
    warnings.simplefilter("ignore", DeprecationWarning)  # fork() with the layer's thread running
    say = lambda *words: print(*words, flush=True)
    mask = blocked(os.getpid())  # before the layer is loaded
    if mode == "order-library":  # the library first, the layer after it
        lib = load(library)
        api = c.CDLL(proxy)
    else:
        before = threads()
        api = c.CDLL(proxy)
        say("threads", before, threads())
        lib = load(library)
    assert lib.fill_bind(proxy.encode()) == 1
    add = lambda name, kind, ctx=0, abi=1: lib.fill_add(abi, name, kind, ctx)
    remove = lambda kind, ctx=0: lib.fill_remove(kind, ctx)

    if mode in ("codes", "codes-kept"):
        # The layer's own functions, in its libcuda.so.1 and not in the driver's.
        assert hasattr(api, "vf_summary_add") and hasattr(api, "vf_summary_remove")
        assert not hasattr(c.CDLL(os.environ["VF_REAL_LIBCUDA"]), "vf_summary_add")
        assert add(b"a", b"count", abi=2) == errno.ENOTSUP
        assert add(b"a", b"count", abi=0) == errno.ENOTSUP
        for name in (b"", b"x" * 65, b"a b", b"a]b", b"a\nb", b"volta-forever", b"VOLTA-FOREVER"):
            assert add(name, b"count") == errno.EINVAL, name
        assert add(b"a", b"no such fill") == errno.EINVAL
        assert add(b"x" * 64, b"count", 7) == 0  # the longest name
        assert add(b"a", b"count", 1) == 0
        assert add(b"b", b"count", 1) == errno.EEXIST  # the same fill and ctx under another name
        assert add(b"b", b"count", 2) == 0
        assert add(b"c", b"empty") == 0
        assert add(b"d", b"slow") == errno.ENOSPC  # 4 names
        assert add(b"a", b"count", 1) == 0  # the same registration again
        assert add(b"a", b"empty") == errno.EEXIST  # its name, to another fill while it is registered
        assert add(b"a", b"count", 5) == errno.EEXIST  # or to another ctx
        assert remove(b"count", 9) == errno.ENOENT
        assert remove(b"no such fill") == errno.EINVAL
        assert remove(b"count", 7) == 0 and remove(b"count", 7) == errno.ENOENT
        if mode == "codes":  # without VF_SUMMARY nothing is kept: the slot is free again
            assert add(b"d", b"slow") == 0
        else:  # a removed library's name keeps its slot (its last report is in the file), until it comes back
            assert add(b"d", b"slow") == errno.ENOSPC
            assert add(b"x" * 64, b"slow") == 0
        say("codes", lib.fill_calls())
        return

    if mode == "lines":
        assert add(b"counter", b"count") == 0
        assert add(b"bad", b"bad", 0) == 0  # returns its size plus one
        assert add(b"bad2", b"bad", 1) == 0  # returns SIZE_MAX
        assert add(b"empty", b"empty") == 0
        say("ready")
        sys.stdin.readline()
        lib.fill_set_count(1)
        say("ready")
        time.sleep(600)

    if mode == "remove":
        assert add(b"counter", b"count") == 0
        lib.fill_set_count(3)
        say("ready")
        sys.stdin.readline()
        r = remove(b"count")
        say("removed", r, lib.fill_calls())
        sys.stdin.readline()
        lib.fill_set_count(4)
        time.sleep(2.2)
        say("calls", lib.fill_calls())
        sys.stdin.readline()
        say("again", add(b"counter", b"count"))
        time.sleep(600)

    if mode == "slow":
        assert add(b"slow", b"slow") == 0
        # A call under way: started less than 200 ms ago (of its 300), not ended.
        end = time.time() + 15
        while time.time() < end:
            n = lib.fill_slow_calls()
            if n and not lib.fill_slow_end(n - 1) and 0 < lib.fill_now() - lib.fill_slow_start(n - 1) < 200000000:
                break
            time.sleep(0.005)
        else:
            raise AssertionError("no call of the slow fill seen under way")
        i = n - 1
        t0 = lib.fill_now()
        r = remove(b"slow")
        t1 = lib.fill_now()
        j = lib.fill_slow_calls() - 1
        s_i, e_i, s_j, e_j = (lib.fill_slow_start(i), lib.fill_slow_end(i), lib.fill_slow_start(j),
                              lib.fill_slow_end(j))
        # The removal waited for call i, then made its own (j), after it.
        assert r == 0 and j == i + 1 and s_i < t0 < e_i <= s_j < e_j <= t1, (r, i, j, s_i, t0, e_i, s_j, e_j, t1)
        time.sleep(2.2)
        assert lib.fill_slow_calls() == j + 1, lib.fill_slow_calls()
        say("slow", (e_i - t0) // 1000000, (t1 - t0) // 1000000)
        say("ready")
        time.sleep(600)

    if mode == "rawfork":
        assert add(b"slow", b"slow") == 0
        end = time.time() + 15
        while time.time() < end:  # the thread in the slow fill, the lock held for 200 ms more at least
            n = lib.fill_slow_calls()
            if n and not lib.fill_slow_end(n - 1) and 0 < lib.fill_now() - lib.fill_slow_start(n - 1) < 100000000:
                break
            time.sleep(0.002)
        else:
            raise AssertionError("no call of the slow fill seen under way")
        r, w = os.pipe()
        pid = lib.fill_raw_child(w)
        assert pid > 0, pid
        end = time.time() + 10
        while True:
            got, status = os.waitpid(pid, os.WNOHANG)
            if got:
                break
            if time.time() > end:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                raise AssertionError("the child made without fork()'s handlers did not end: it waited for the lock")
            time.sleep(0.01)
        codes = list(c.cast(c.create_string_buffer(os.read(r, 8), 8), c.POINTER(c.c_int * 2)).contents)
        assert os.waitstatus_to_exitcode(status) == 0 and codes == [errno.ECHILD, errno.ECHILD], (status, codes)
        say("rawfork", *codes)
        return

    if mode == "signals":  # the masks of the threads, the layer's thread found by its name
        tasks = os.listdir("/proc/self/task")
        name = lambda t: open(f"/proc/self/task/{t}/comm").read().strip()
        mine = [t for t in tasks if name(t) == "vf-summary"]
        say("masks", hex(mask), hex(blocked(os.getpid())), *(hex(blocked(t)) for t in mine))
        return

    if mode in ("hang", "forks", "exits"):  # fills that break the contract
        assert add(mode.encode(), mode.encode()) == 0
        lib.fill_arm(1)
        end = time.time() + 15
        while not lib.fill_entered() and time.time() < end:
            time.sleep(0.02)
        assert lib.fill_entered(), "the armed fill was not called"
        if mode == "exits":
            time.sleep(20)  # its exit(3) ends the process first
            say("still alive")
            return
        if mode == "hang":  # a fork from this thread, while the layer's thread is in the fill for ever
            t0 = time.time()
            pid = os.fork()
            if pid == 0:
                os._exit(0)
            os.waitpid(pid, 0)
            say("fork", round(1000 * (time.time() - t0)))
        else:  # the fill forks at each write
            end = time.time() + 15
            while lib.fill_calls() < 3 and time.time() < end:
                time.sleep(0.05)
            say("calls", lib.fill_calls())
        say("exiting")
        return  # exit()

    if mode == "reenter":
        assert add(b"re", b"reenter", 100) == 0
        assert add(b"counter", b"count") == 0
        end = time.time() + 15
        while lib.fill_calls() < 2 and time.time() < end:
            time.sleep(0.05)
        assert lib.fill_reentered(0) == errno.EDEADLK and lib.fill_reentered(1) == errno.EDEADLK
        lib.fill_set_count(5)  # the thread is still there: the file changes
        say("ready")
        time.sleep(600)

    if mode == "fork":
        assert add(b"slow", b"slow") == 0  # each write holds the lock for 300 ms
        assert add(b"counter", b"count") == 0
        lib.fill_at_exit(b"count", 0)
        end = time.time() + 15
        while lib.fill_slow_calls() < 1 and time.time() < end:
            time.sleep(0.05)
        pids = []
        for i in range(20):
            pid = os.fork()
            if pid == 0:
                calls = lib.fill_calls()
                codes = (add(b"child", b"empty"), remove(b"count"))
                time.sleep(0.3)
                ok = codes == (errno.ECHILD, errno.ECHILD) and lib.fill_calls() == calls
                sys.stdout.flush()
                sys.exit(0 if ok else 1)  # exit(): the library's destructor and the layer's run
            pids.append(pid)
            time.sleep(0.15)
        end = time.time() + 30
        for pid in pids:
            while True:
                got, status = os.waitpid(pid, os.WNOHANG)
                if got:
                    assert os.waitstatus_to_exitcode(status) == 0, (pid, status)
                    break
                if time.time() > end:
                    os.kill(pid, signal.SIGKILL)
                    raise AssertionError(f"child {pid} made by fork() did not end")
                time.sleep(0.05)
        say("children", *pids)
        say("ready")
        time.sleep(600)

    if mode in ("unset", "nothread", "order-layer", "order-library"):
        assert add(b"counter", b"count") == 0
        lib.fill_set_count(7)
        lib.fill_at_exit(b"count", 0)
        time.sleep(1.5 if mode in ("unset", "nothread") else 0)
        say("calls", lib.fill_calls())
        return  # exit()

    if mode in ("dlclose", "dlclose-pinned"):
        assert add(b"counter", b"count") == 0
        lib.fill_set_count(2)
        lib.fill_at_exit(b"count", 0)
        pinned = mode == "dlclose-pinned" and lib.fill_pin() == 1
        say("ready")
        sys.stdin.readline()
        import _ctypes
        _ctypes.dlclose(lib._handle)
        try:
            again = load(library, os.RTLD_NOLOAD | os.RTLD_LOCAL)
        except OSError:
            again = None
        say("loaded" if again else "unloaded", pinned)
        if again:
            again.fill_set_count(3)
        time.sleep(2.5)
        say("alive")
        sys.stdin.readline()
        return  # exit()
    raise AssertionError(f"unknown mode {mode}")


def wait_for(path, text, seconds=15):
    end = time.time() + seconds
    while time.time() < end:
        if path.exists() and text in path.read_text(errors="replace"):
            return path.read_text()
        time.sleep(0.1)
    raise AssertionError(f"{path}: no '{text}' after {seconds} s: " + (path.read_text(errors="replace")
                                                                       if path.exists() else "no file"))


def well_formed(path, names):
    """The file's text: valid UTF-8, every line with a known prefix and without control characters, line or
    paragraph separators or bidirectional controls, so that splitlines gives the lines that newlines make."""
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    assert text.endswith("\n"), text[-200:]
    lines = text.split("\n")[:-1]
    assert text.splitlines() == lines, [l for l in text.splitlines() if l not in lines][:3]
    prefixes = ["[volta-forever] "] + [f"[{n}] " for n in names] + [f"[{n}, last report] " for n in names]
    bidi = ("LRE", "RLE", "LRO", "RLO", "PDF", "LRI", "RLI", "FSI", "PDI")
    for line in lines:
        assert any(line.startswith(p) for p in prefixes), repr(line)
        assert not any(unicodedata.category(ch) == "Cc" or unicodedata.bidirectional(ch) in bidi or
                       ch in "\u2028\u2029\u200e\u200f\u061c" for ch in line), repr(line)
    return text


def of(text, name):
    """The lines of a library in the file's text, without their prefix."""
    return [line[len(name) + 3:] for line in text.splitlines() if line.startswith(f"[{name}] ")]


LIMIT = 60  # seconds: the longest that any wait of a case takes; then the case's processes are killed and it fails


class Child:
    """A case's process, in a session of its own (its children with it), its output read by threads: no wait of
    the test is unbounded, and a case that hangs is killed, all its processes, and fails."""

    def __init__(self, args, env, preexec=None):
        self.p = subprocess.Popen(args, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, preexec_fn=preexec, start_new_session=True)
        self.pid = self.p.pid
        self.out = queue.Queue()
        self.err = []
        self.readers = [threading.Thread(target=self._read, args=(self.p.stdout, self.out.put), daemon=True),
                        threading.Thread(target=self._read, args=(self.p.stderr, self.err.append), daemon=True)]
        for r in self.readers:
            r.start()

    @staticmethod
    def _read(f, put):
        for text in f:
            put(text)
        put(None)

    def errors(self):
        return "".join(e for e in self.err if e)[-3000:]

    def kill(self):
        try:
            os.killpg(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return self.p.wait()

    def line(self, seconds=LIMIT):
        """The next line of the process's output, as words."""
        try:
            text = self.out.get(timeout=seconds)
        except queue.Empty:
            self.kill()
            raise AssertionError(f"no line from process {self.pid} within {seconds} s: killed; "
                                 f"its standard error: {self.errors()}")
        if text is None:
            raise AssertionError(f"process {self.pid} ended (status {self.p.wait()}) before its line; "
                                 f"its standard error: {self.errors()}")
        return text.split()

    def send(self, text="go"):
        self.p.stdin.write(text + "\n")
        self.p.stdin.flush()

    def end(self, seconds=LIMIT):
        """Wait for the process's end: its status, the rest of its output and its standard error."""
        try:
            self.p.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            self.kill()
            raise AssertionError(f"process {self.pid} did not end within {seconds} s: killed; "
                                 f"its standard error: {self.errors()}")
        for r in self.readers:
            r.join(seconds)
        out = []
        while not self.out.empty():
            text = self.out.get()
            if text is not None:
                out.append(text)
        return self.p.returncode, "".join(out), "".join(e for e in self.err if e)

    def finish(self, seconds=LIMIT):
        """The end of a process that must exit with 0: its output and its standard error."""
        status, out, err = self.end(seconds)
        assert status == 0, (status, out, err)
        return out, err


if __name__ == "__main__":
    if len(sys.argv) == 4:
        child(*sys.argv[1:])
        sys.exit(0)
    sys.stdout.reconfigure(line_buffering=True)  # each PASS line out at once, whatever ends the test
    proxy = os.path.abspath(sys.argv[1])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mock, library = str(tmp / "driver.so"), str(tmp / "libfill.so")
        here = Path(__file__)
        subprocess.run(["cc", "-shared", "-fPIC", "-o", mock, here.with_name("link_mock_driver.c")], check=True)
        subprocess.run(["cc", "-shared", "-fPIC", "-O2", "-Wall", "-o", library, here.with_name("summary_fill.c"),
                        "-ldl"], check=True)
        base = dict(os.environ, VF_REAL_LIBCUDA=mock, VF_PROXY_CACHE="0")
        for name in ("VF_LOG", "VF_LOG_LEVEL", "VF_SUMMARY", "VF_SUMMARY_INTERVAL", "VF_STATS", "FILL_NO_REMOVE"):
            base.pop(name, None)
        summary = tmp / "summary"
        on = dict(VF_SUMMARY=str(summary), VF_SUMMARY_INTERVAL="1")
        start = lambda mode, preexec=None, **env: Child([sys.executable, __file__, proxy, library, mode],
                                                        dict(base, **env), preexec)

        # Return codes; the functions are the layer's.
        for mode, env in (("codes", {}), ("codes-kept", on)):
            p = start(mode, **env)
            p.line()
            out, err = p.finish()
            # Without VF_SUMMARY, no fill is called; with it, the removals called theirs.
            assert out.split()[0] == "codes" and (mode != "codes" or out.split() == ["codes", "0"]), (out, err)
        print("PASS the return codes, the 4 slots and a name that takes its slot back; the functions are the layer's")

        # Lines while the process runs, written again when they change, kept after SIGKILL; fills that misbehave.
        names = ["counter", "bad", "bad2", "empty"]
        p = start("lines", **on)
        p.line()
        assert p.line() == ["ready"]
        path = Path(f"{summary}.{p.pid}")
        wait_for(path, "[counter] count 0\n")
        text = well_formed(path, names)
        lines = text.splitlines()
        assert lines[0].startswith(f"[volta-forever] report of process {p.pid}, running for "), text
        for bad in ("bad", "bad2"):
            mine = of(text, bad)
            assert mine and mine[-1] == "END", mine[-3:]
            assert any("café" in m and "漢" in m and "[volta-forever] forged" in m and "esrever" in m
                       for m in mine), mine[:3]
            assert sum(len(m.encode()) + 1 for m in mine) <= 4096, sum(len(m.encode()) for m in mine)
        assert of(text, "empty") == [] and of(text, "counter") == ["count 0"], text
        # The libraries' lines come after the layer's, in the order of registration.
        order = [next(i for i, l in enumerate(lines) if l.startswith(f"[{n}] ")) for n in ("counter", "bad", "bad2")]
        assert order == sorted(order) and all(not l.startswith("[") or l.startswith("[volta-forever] ")
                                              for l in lines[:order[0]]), text
        p.send()
        assert p.line() == ["ready"]
        wait_for(path, "[counter] count 1\n")
        assert p.kill() == -signal.SIGKILL
        text = well_formed(path, names)
        assert "running for" in text and of(text, "counter") == ["count 1"] and "at its exit" not in text, text
        print("PASS a fill's lines after the layer's, written again when they change, kept after SIGKILL; fills that"
              " write control bytes and bytes that are not UTF-8 and return more than their buffer leave the file"
              " well formed, read up to the buffer's end")

        # A removal: the fill called once more, its lines kept and marked, nothing called after; the slot back.
        p = start("remove", **on)
        p.line()
        assert p.line() == ["ready"]
        path = Path(f"{summary}.{p.pid}")
        wait_for(path, "[counter] count 3\n")
        p.send()
        removed = p.line()
        assert removed[:2] == ["removed", "0"], removed
        wait_for(path, "[counter, last report] count 3\n")
        p.send()
        calls = p.line()
        assert calls == ["calls", removed[2]], (calls, removed)
        text = well_formed(path, ["counter"])
        assert "count 4" not in text and "[counter] " not in text, text
        p.send()
        assert p.line() == ["again", "0"]
        wait_for(path, "[counter] count 4\n")
        assert "last report" not in path.read_text()
        p.kill()
        print("PASS a removal calls the fill once more and keeps its lines, marked as its last report; nothing is"
              " called after it; the name registered again takes its slot back")

        # The removal waits for a call under way.
        p = start("slow", **on)
        p.line()
        slow = p.line()
        assert slow[0] == "slow", slow
        assert p.line() == ["ready"]
        path = Path(f"{summary}.{p.pid}")
        wait_for(path, "[slow, last report] slow\n")
        p.kill()
        print(f"PASS a removal waits for the call under way ({slow[1]} ms of it left) and returns after its own"
              f" last call ({slow[2]} ms in all); nothing is called after it")

        # A fill that calls the layer back.
        p = start("reenter", **on)
        p.line()
        assert p.line() == ["ready"]
        path = Path(f"{summary}.{p.pid}")
        wait_for(path, "[counter] count 5\n")
        text = well_formed(path, ["re", "counter"])
        assert of(text, "re") == [f"reenter add {errno.EDEADLK} remove {errno.EDEADLK}"], text
        p.kill()
        print("PASS a fill that calls vf_summary_add or vf_summary_remove is refused (EDEADLK), not left waiting")

        # Children made by fork() while the thread writes (and holds the lock for 300 ms).
        p = start("fork", **on)
        p.line()
        children = p.line()
        assert children[0] == "children", children
        assert p.line() == ["ready"]
        path = Path(f"{summary}.{p.pid}")
        wait_for(path, "[slow] slow\n")
        p.kill()
        _, _, err = p.end()
        written = sorted(f.name for f in tmp.glob("summary.*"))
        for pid in children[1:]:
            assert not any(name.startswith(f"summary.{pid}") for name in written), written
        ends = [l for l in err.splitlines() if l.startswith("fill library: removed at its end")]
        assert len(ends) == 20 and all(l.endswith(f"vf_summary_remove returned {errno.ECHILD}") and
                                       l.split("before ")[1].split(",")[0] == l.split("after ")[1].split(",")[0]
                                       for l in ends), err
        print("PASS children made by fork() while the thread writes: add and remove do nothing there (ECHILD), at"
              " their exit too, nothing is called, they end and write nothing")

        # A child made without fork()'s handlers while the thread holds the lock.
        p = start("rawfork", **on)
        p.line()
        out, err = p.finish()
        assert out.split() == ["rawfork", str(errno.ECHILD), str(errno.ECHILD)], (out, err)
        print("PASS a child made without fork()'s handlers while the thread held the lock (held for ever in it): add"
              " and remove return ECHILD at once")

        # The layer's thread blocks every signal; the thread that loaded the layer keeps its own mask.
        p = start("signals", **on)
        p.line()
        masks = p.line()
        p.finish()
        assert masks[0] == "masks" and len(masks) == 4, masks
        before, after, thread = (int(m, 16) for m in masks[1:])
        wanted = sum(1 << (n - 1) for n in (signal.SIGHUP, signal.SIGINT, signal.SIGQUIT, signal.SIGUSR1, signal.SIGUSR2,
                                            signal.SIGPIPE, signal.SIGALRM, signal.SIGTERM, signal.SIGCHLD,
                                            signal.SIGRTMIN))
        assert thread & wanted == wanted and after == before, masks
        print(f"PASS the layer's thread blocks every signal (its SigBlk {masks[3]}), the thread that loaded the layer"
              f" keeps its mask ({masks[2]})")

        # Without VF_SUMMARY: no thread, no call.
        p = start("unset")
        before, after = map(int, p.line()[1:])
        assert after == before, (before, after)
        out, err = p.finish()
        assert out.split() == ["calls", "0"], out
        assert "calls before 0, after 0, vf_summary_remove returned 0" in err, err
        assert list(tmp.glob(f"summary.{p.pid}*")) == []
        print("PASS without VF_SUMMARY, no thread, no call of a fill, no file")

        # Without the thread (it could not be started): the lines at exit.
        if os.geteuid() == 0:
            print("SKIP the lines at exit without the thread: root is not held to RLIMIT_NPROC")
        else:
            no_threads = lambda: resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
            p = start("nothread", no_threads, **on)
            p.line()
            out, err = p.finish()
            assert "cannot start the thread that writes the report" in err, err
            assert out.split() == ["calls", "0"], out
            text = well_formed(Path(f"{summary}.{p.pid}"), ["counter"])
            assert "at its exit" in text and of(text, "counter") == ["count 7"], text
            print("PASS a registration without the thread (it could not be started) is written at exit")

        # Fills that break the contract. One that never returns: a fork from another thread does not wait for it
        # (the fills have a lock of their own, which the handlers of pthread_atfork do not take), and the last write
        # waits 2 s for it, then is made without the other libraries' lines, and says so.
        p = start("hang", **on)
        p.line()
        fork = p.line()
        assert fork[0] == "fork" and int(fork[1]) < 3000, fork
        assert p.line() == ["exiting"]
        t0 = time.time()
        out, err = p.finish(30)
        took = time.time() - t0
        text = well_formed(Path(f"{summary}.{p.pid}"), ["hang"])
        assert text.startswith(f"[volta-forever] report of process {p.pid}, at its exit\n") and took < 10, (took, text)
        assert ("\n[volta-forever] the lines of other libraries are not in this last write: one of their functions had "
                "not returned 2 s after the exit began\n") in text and "\n[hang] " not in text, text
        # One that forks at each write: no deadlock (the fork's handlers do not wait for the fill's own lock).
        p = start("forks", **on)
        p.line()
        calls = p.line()
        assert calls[0] == "calls" and int(calls[1]) >= 3, calls
        assert p.line() == ["exiting"]
        p.finish()
        text = well_formed(Path(f"{summary}.{p.pid}"), ["forks"])
        forked = [int(l.split()[-1]) for l in text.splitlines() if l.startswith("[forks] forked ")]
        assert text.startswith(f"[volta-forever] report of process {p.pid}, at its exit\n") and forked and \
            forked[0] >= 3, text
        # One that calls exit(3): the process ends with that status, and the last write is made, without the other
        # libraries' lines (this thread was in one of their functions), and says so.
        p = start("exits", **on)
        p.line()
        status, out, err = p.end()
        assert status == 3 and "still alive" not in out, (status, out, err)
        text = well_formed(Path(f"{summary}.{p.pid}"), ["exits"])
        assert text.startswith(f"[volta-forever] report of process {p.pid}, at its exit\n"), text
        assert ("\n[volta-forever] the lines of other libraries are not in this last write: the process exited from "
                "one of their functions\n") in text, text
        print(f"PASS fills that break the contract: one that never returns delays the exit by 2 s ({took:.1f} s here), the"
              f" last write made without the libraries' lines and saying so, and a fork from another thread does not wait"
              f" for it ({fork[1]} ms); one that forks at each write is called {calls[1]} times; one that calls exit(3)"
              f" ends the process with that status, its last write made and saying why it lacks the libraries' lines")

        # At exit, in both orders of the destructors: which one came first is read from the removal (whether it
        # called the fill once more), and the file must hold what that order gives; the two cases must show both.
        orders = []
        for mode in ("order-layer", "order-library"):
            p = start(mode, **on)
            if mode == "order-layer":
                p.line()
            out, err = p.finish()
            text = well_formed(Path(f"{summary}.{p.pid}"), ["counter"])
            end = [l for l in err.splitlines() if l.startswith("fill library: removed at its end")]
            assert len(end) == 1 and end[0].endswith("vf_summary_remove returned 0"), err
            before, after = (int(end[0].split(w)[1].split(",")[0]) for w in ("before ", "after "))
            assert after in (before, before + 1), end
            library_first = after == before + 1  # its removal came before the last write, and called the fill
            mark = "[counter, last report] " if library_first else "[counter] "
            assert "at its exit" in text.splitlines()[0] and f"\n{mark}count 7\n" in text, (mode, end, text)
            orders.append(library_first)
        assert sorted(orders) == [False, True], f"both cases ran the destructors in one order: {orders}"
        print(f"PASS at exit, in both orders of the destructors (the {'library' if orders[0] else 'layer'}'s first"
              " for the library loaded after the layer): the layer's first, its last write calls the fill and the"
              " removal after it calls nothing; the library's first, its last report is kept and written")

        # A library unloaded while the thread runs, and one pinned.
        def unloaded(mode, want=None, **env):
            p = start(mode, **on, **env)
            p.line()
            assert p.line() == ["ready"]
            path = Path(f"{summary}.{p.pid}")
            wait_for(path, "[counter] count 2\n")
            p.send()
            if want is None:  # the control: the process must end by itself
                p.send()
                status, _, err = p.end(30)
                return status, err
            state = p.line()
            assert state[0] == want, state
            assert p.line() == ["alive"]
            p.send()
            out, err = p.finish()
            return well_formed(path, ["counter"]), err
        text, err = unloaded("dlclose", "unloaded")
        assert "removed at its end: calls before" in err and of(text, "counter, last report") == ["count 2"], text
        text, err = unloaded("dlclose-pinned", "loaded")
        assert of(text, "counter") == ["count 3"] and "last report" not in text, text
        print("PASS a library unloaded while the thread runs removes itself: its last report stays and the process"
              " lives on; pinned, it stays loaded and its lines are written as they change")
        # Control: without its removal, the thread calls into the library that is gone.
        status, err = unloaded("dlclose", FILL_NO_REMOVE="1")
        assert status == -signal.SIGSEGV, (status, err)
        print("PASS control: a library unloaded without removing itself is called by the thread, and the process ends"
              " by SIGSEGV")
