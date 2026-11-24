"""volta-forever: the helpers of VF_TRITON_TC, from the program's side.

The import hook and the rules that choose the kernels compiled for sm_80 are in
volta_forever_triton, which a Python reads when it imports Triton; this file is read by it at the
first kernel that may be compiled for sm_80, and by the helper (volta_forever_triton_helper.py),
which takes its messages from here. Python compiles each of them from its source (the
installation holds no bytecode, and nothing is written into it at run time): a process reads only
what it uses.

A compilation for sm_80 runs outside the program: Triton's compiler for sm_80 is not the one the
program would have run (7.5), and an error of LLVM or MLIR, or an assertion, ends the process it
runs in. It runs in a helper, a child of the program's process that compiles one kernel at a
time, started at the first kernel that needs one:

  - by the program's interpreter, with -I -S (nothing of the program's start, nothing of its
    sys.path but the directory that holds its Triton, after the standard library), an
    environment of its own (_helper_env: no secret of the program's, no LD_PRELOAD), the working
    directory /, its own session and process group (ptxas, which Triton runs, is in it), every
    other descriptor closed;
  - reached through a socket pair whose descriptor it inherits: no path names it, and no other
    process can reach it;
  - ended with its program (PR_SET_PDEATHSIG, the end of its socket), when it has had nothing to
    do for VF_TRITON_TC_IDLE seconds, or for half a second when a process of the user waits for a
    slot; waited for by a thread of its own (the wait of its pid alone: no other child of the
    program is taken), so that none is left as a zombie;
  - holding a slot (a place) while it lives: one of VF_TRITON_TC_JOBS lock files of this host in
    the directory of the user's helpers in Triton's cache directory (_core._own_dir), locked with
    flock; so that the user's helpers on a host are that many at most, and their memory with
    them. That cap is a best effort: where a lock file cannot be opened or locked for another
    reason than another holder (a file system without working locks, a directory that cannot be
    written), the process does not count places (said once, at VF_LOG_LEVEL=2) and its helpers
    start without one; and a process waits VF_TRITON_TC_WAIT seconds for a place once in its
    life: when none came, it waits for none again, and starts ALONE helper without a place when
    none is free (said once);
  - writing its temporary files (Triton's PTX, the cubin that ptxas writes, the tool's) in a
    directory of its own (mode 0700) in the temporary directory, where Triton's own compilation
    writes them (the directory of the user's helpers holds the lock files alone).

A process waits for each answer VF_TRITON_TC_TIMEOUT seconds at most (a time during which it was
itself stopped is not counted); the helper is then killed with its group. It takes from it only
an answer to its request (its digest) whose cubin is one for sm_80 that keeps the PTX of the
answer (_checked).
"""

import os
import sys


def _load(name, filename):
    """The module of this file's directory named so, read without writing its bytecode (the
    installation is not written to at run time): the one in sys.modules when it is that file's."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    module = sys.modules.get(name)
    if module is not None and getattr(module, "__file__", None) == path:
        return module
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    written = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        sys.modules[name] = module
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    finally:
        sys.dont_write_bytecode = written
    return module


# The module of the hook (the one that read this file puts itself in _volta_forever_core first).
_core = globals().get("_volta_forever_core") or _load("volta_forever_triton", "volta_forever_triton.py")
# (The state of the process, its lock and its lines are the hook's: one of each in a process.)
say, _kernel_name = _core.say, _core._kernel_name
_lock, _state = _core._lock, _core._state

TIMEOUT = 900.0
START = 20.0
IDLE = 10.0
WAIT = 60.0
# A message is taken in SEND seconds at most; the status of a helper whose socket ended, in REAP.
SEND = 30.0
REAP = 5.0
# The largest message that a helper reads (a request: a TTIR, usually under a MB) and that a
# process reads (an answer: its texts and its cubin in base64).
REQUEST_MAX = 64 << 20
ANSWER_MAX = 256 << 20
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0o2000000)


def _seconds(name, default):
    try:
        value = float(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return min(max(1.0, value), 86400.0) if value == value else default


def _jobs():
    """How many helpers the user's processes of this host have at most (VF_TRITON_TC_JOBS, else a
    quarter of the processors, from 2 to 8)."""
    given = os.environ.get("VF_TRITON_TC_JOBS", "")
    if given:
        try:
            return min(max(1, int(given)), 256)
        except ValueError:
            pass
    return max(2, min(8, (os.cpu_count() or 8) // 4))


def _host():
    """This host's name, as the files of its slots name it (one set of slots per host, where
    Triton's cache directory is shared by several)."""
    import re
    return re.sub(r"[^A-Za-z0-9_.]", "_", os.uname().nodename)[:64] or "localhost"


class HelperFailed(Exception):
    """A helper did not compile: (why, whether the kernel is the cause: a crash or a time out of
    its compilation, not the helper or its start; gone: the helper had ended on its own, as the
    request reached it)."""

    def __init__(self, why, kernel, gone=False):
        Exception.__init__(self, why)
        self.why = why
        self.kernel = kernel
        self.gone = gone


class _NoSlot(Exception):
    """No helper of this process was free in VF_TRITON_TC_WAIT seconds."""


class _NoCap(Exception):
    """A lock file of the places cannot be opened or locked for another reason than another holder
    (a file system whose locks do not work, a directory that cannot be written, what is not a
    regular file at a lock file's path): the places are not counted in this process."""


# The helpers that a process has at most without a place once it has waited VF_TRITON_TC_WAIT
# seconds for one in vain: from then on it waits for a place no more (places held by helpers that
# do not give them up, stopped or stuck in a compilation after their program died, cost a process
# one wait, not one a kernel).
ALONE = 1


# --- the cubin as the layer will take it -----------------------------------------------------

def keeps_ptx(cubin):
    """Whether ptxas kept the PTX in the cubin (the section that -lineinfo makes)."""
    return b".nv_debug_ptx_txt" in cubin


# No cubin larger than that is taken (PyTorch's largest is 22 MB).
CUBIN_MAX = 64 << 20


def cubin_problem(cubin, sm=80):
    """Why these bytes are not a cubin for sm (ELF64, little-endian, machine EM_CUDA, the SM in its
    flags: byte 0 for ABI 7, byte 1 for ABI 8, as crates/vf-fatbin reads it; CUBIN_MAX bytes at
    most; its PTX kept), or None."""
    import struct
    if not isinstance(cubin, bytes) or len(cubin) < 64:
        return "it is not a cubin (%s)" % ("%d bytes" % len(cubin) if isinstance(cubin, bytes) else type(cubin).__name__)
    if len(cubin) > CUBIN_MAX:
        return "its cubin has %d bytes" % len(cubin)
    if cubin[:4] != b"\x7fELF" or cubin[4] != 2 or cubin[5] != 1:
        return "it is not a cubin (no 64-bit ELF header)"
    machine, flags = struct.unpack_from("<H", cubin, 18)[0], struct.unpack_from("<I", cubin, 48)[0]
    if machine != 190:
        return "it is not a cubin (ELF machine %d)" % machine
    got = (flags >> 8) & 0xff if cubin[8] >= 8 else flags & 0xff
    if got != sm:
        return "its cubin is for sm_%d, not sm_%d" % (got, sm)
    if not keeps_ptx(cubin):
        return "its cubin does not keep its PTX"
    return None


def kept_ptx(cubin):
    """The text of the cubin's section .nv_debug_ptx_txt (the PTX that ptxas kept, its lines
    separated by NUL), or None."""
    import struct
    try:
        shoff = struct.unpack_from("<Q", cubin, 0x28)[0]
        size, count, names = struct.unpack_from("<HHH", cubin, 0x3a)
        if size < 64:
            return None
        table = struct.unpack_from("<IIQQQQ", cubin, shoff + names * size)[4]
        for i in range(count):
            name, _, _, _, offset, length = struct.unpack_from("<IIQQQQ", cubin, shoff + i * size)
            end = cubin.find(b"\0", table + name, table + name + 64)
            if end > 0 and cubin[table + name:end] == b".nv_debug_ptx_txt":
                if offset + length > len(cubin):
                    return None
                return cubin[offset:offset + length].replace(b"\0", b"\n").decode("utf-8", "replace")
    except (struct.error, ValueError, OverflowError):
        return None
    return None


def _ptx_lines(text):
    """The lines of a PTX as ptxas keeps them: without its comments, its .loc and .file lines, its
    DWARF sections and empty lines, each run of blanks one space."""
    import re
    text = re.split(r"\.section\s+\.debug", text, maxsplit=1)[0]
    text = re.sub(r"//[^\n]*", "", text)
    out = []
    for line in text.splitlines():
        line = " ".join(line.split())
        if line and not line.startswith((".loc", ".file")):
            out.append(line)
    return out


def ptx_problem(cubin, ptx):
    """Why the PTX that the cubin keeps (what the layer will compile for the V100) is not the PTX
    of the answer, compared line by line as ptxas keeps it, or None (a few ms for a kernel of
    Triton's; a PTX of more than 16 MiB is not compared)."""
    if len(ptx) > 16 << 20:
        return None
    kept = kept_ptx(cubin)
    if kept is None:
        return "its cubin's PTX could not be read"
    a, b = _ptx_lines(kept), _ptx_lines(ptx)
    if a == b:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return "the PTX that its cubin keeps is not its PTX (line %d: %r)" % (i + 1, x[:80])
    return "the PTX that its cubin keeps has %d lines, its PTX %d" % (len(a), len(b))


# --- messages --------------------------------------------------------------------------------
# A message: 8 bytes of size (little-endian), then its JSON. Nothing is unpickled, evaluated or
# imported from a message, either way.

def _message(obj):
    import json
    import struct
    data = json.dumps(obj, separators=(",", ":"), default=_jsonable).encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _jsonable(obj):
    import dataclasses
    if dataclasses.is_dataclass(obj):
        return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    if isinstance(obj, bytes):
        import base64
        return {"$bytes": base64.b64encode(obj).decode("ascii")}
    raise TypeError("%s is not sent to a helper" % type(obj).__name__)


def _decode(data):
    """A message's JSON as a dict, or None (not JSON, nested too deep for the parser, not a dict)."""
    import json
    try:
        obj = json.loads(data.decode("utf-8"))
    except (ValueError, RecursionError):
        return None
    return obj if isinstance(obj, dict) else None


def _wait_for(sock, event, timeout):
    """poll() of one socket (a process may have any number of descriptors open: no select)."""
    import select
    p = select.poll()
    p.register(sock.fileno(), event)
    return bool(p.poll(max(0.0, timeout) * 1000.0))


def _send(sock, data, timeout=SEND):
    """Send data in timeout seconds at most (socket.timeout after) on a non-blocking socket, with
    MSG_NOSIGNAL: a peer that closed gives an error (EPIPE), never SIGPIPE, which would end a
    program that sets it to its default."""
    import select
    import socket
    import time
    flags = getattr(socket, "MSG_NOSIGNAL", 0x4000)
    view = memoryview(data)
    deadline = time.monotonic() + timeout
    while len(view):
        try:
            view = view[sock.send(view, flags):]
            continue
        except (BlockingIOError, InterruptedError):
            pass
        left = deadline - time.monotonic()
        if left <= 0:
            raise socket.timeout("nothing of a message was taken in %g s" % timeout)
        _wait_for(sock, select.POLLOUT, left)


class _Inbox(object):
    """The next message of a non-blocking socket, read as it comes into a buffer of its size
    (nothing read is copied again), limit bytes at most."""

    def __init__(self, sock, limit):
        self.sock, self.limit = sock, limit
        self.buf, self.got, self.head = bytearray(8), 0, True

    def read(self):
        """The message once it is whole, or None (more is to come); EOFError when the peer closed,
        ValueError for a message larger than the limit."""
        import struct
        while True:
            if self.got == len(self.buf):
                if self.head:
                    size = struct.unpack("<Q", bytes(self.buf))[0]
                    if size > self.limit:
                        raise ValueError("a message of %d bytes" % size)
                    self.buf, self.got, self.head = bytearray(size), 0, False
                    continue
                data = bytes(self.buf)
                self.buf, self.got, self.head = bytearray(8), 0, True
                return data
            try:
                n = self.sock.recv_into(memoryview(self.buf)[self.got:], min(len(self.buf) - self.got, 1 << 22))
            except (BlockingIOError, InterruptedError):
                return None
            if n == 0:
                raise EOFError("the peer closed its socket")
            self.got += n


def _digest(request):
    """The digest of what a request asks (its TTIR, target, options and metadata as JSON), which
    the helper computes again and gives back with its answer: an answer is for that request."""
    import hashlib
    import json
    what = [request.get(k) for k in ("ttir", "target", "options", "metadata")]
    return hashlib.sha256(json.dumps(what, sort_keys=True, separators=(",", ":"), default=_jsonable)
                          .encode("utf-8")).hexdigest()


def _json(value):
    import json
    return json.loads(json.dumps(value, default=_jsonable))


def _identity():
    """What a helper's Triton must be: the program's (the interpreter, Triton's directory and its
    version; not the digest of its files, triton_key(), which reads all of libtriton.so again:
    0.4 s of each helper's start, for the same directory)."""
    with _lock:
        if "identity" in _state:
            return _state["identity"]
    import triton
    ident = {"python": os.path.realpath(sys.executable or "?"),
             "triton": os.path.realpath(os.path.dirname(triton.__file__)), "version": triton.__version__}
    with _lock:
        return _state.setdefault("identity", ident)


# --- a helper --------------------------------------------------------------------------------

# What a helper gets of the program's environment: these variables, and those whose names begin
# so (Triton's settings, those of LLVM and MLIR that Triton reads, ptxas's, the layer's), but for
# those of Triton's caches, which a helper does not use, and the layer's lines. The others
# (secrets, PYTHON* settings, LD_PRELOAD, LD_AUDIT...) are not given to it.
HELPER_ENV = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LANGUAGE", "TZ", "TMPDIR", "LD_LIBRARY_PATH", "USE_IR_LOC")
HELPER_ENV_PREFIXES = ("LC_", "TRITON_", "MLIR_", "LLVM_", "NVPTX_", "PTXAS_", "DISABLE_", "VF_")
HELPER_ENV_LEFT = ("TRITON_CACHE_DIR", "TRITON_HOME", "TRITON_DUMP_DIR", "TRITON_OVERRIDE_DIR", "TRITON_CACHE_MANAGER",
                   "TRITON_REMOTE_CACHE_BACKEND", "TRITON_REDIS_HOST", "TRITON_REDIS_PORT", "TRITON_REDIS_KEY_FORMAT",
                   "TRITON_PRINT_AUTOTUNING", "TRITON_CACHE_AUTOTUNING", "TRITON_INTERPRET", "TRITON_ALWAYS_COMPILE",
                   "TRITON_STORE_BINARY_ONLY", "VF_LOG", "VF_LOG_LEVEL")


def _helper_env():
    env = {k: v for k, v in os.environ.items()
           if (k in HELPER_ENV or k.startswith(HELPER_ENV_PREFIXES)) and k not in HELPER_ENV_LEFT}
    env["VF_TRITON_TC"] = "0"
    return env


# The signals of a crash of the compiler itself, which Triton's cache remembers for the kernel; any
# other end (SIGKILL as from the out-of-memory killer, SIGTERM, SIGINT or SIGHUP sent to a group of
# processes, an exit with no answer) may be otherwise another time: the kernel is compiled by
# Triton's own stages for this compilation, and not kept.
CRASH_SIGNALS = {4: "SIGILL", 6: "SIGABRT", 7: "SIGBUS", 8: "SIGFPE", 11: "SIGSEGV"}


def _open_lock(name):
    """A lock file of the directory of the user's helpers, opened (made with mode 0600) without
    following a link or waiting on a FIFO; OSError when it is not a regular file."""
    import errno
    import stat
    fd = os.open(os.path.join(_state["own"], name), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | _O_CLOEXEC,
                 0o600)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise OSError(errno.EINVAL, "not a regular file")
    return fd


# The errors of an open() that are this process's limits, not a property of the directory: that
# kernel is compiled by Triton's own stages, and not kept.
_LIMITS = ("EMFILE", "ENFILE", "ENOMEM")


def _limit(e):
    import errno
    return e.errno in [getattr(errno, name) for name in _LIMITS]


def _slot(pool):
    """A free slot of this host: a descriptor of its lock file, locked; None when another holder
    has each; _NoCap when a lock file cannot be opened or locked for another reason; OSError when
    this process can open no descriptor. The files of the slots that were taken are kept open by
    the pool for the next try (a lock tried on a network file system is one request, an open
    another)."""
    import fcntl
    import random
    order = list(range(_jobs()))
    random.shuffle(order)
    for i in order:
        name = "slot-%s-%d" % (_host(), i)
        fd, inode = pool.files.pop(name, (None, None))
        if fd is not None:
            try:
                st = os.fstat(fd)
                fd = fd if (st.st_dev, st.st_ino) == inode else None
            except OSError:
                fd = None   # (closed by the program, or another file now: the program's)
        if fd is None:
            try:
                fd = _open_lock(name)
            except OSError as e:
                if _limit(e):
                    raise
                raise _NoCap("%s: %s" % (name, e.strerror or e))
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            st = os.fstat(fd)
            pool.files[name] = (fd, (st.st_dev, st.st_ino))
        except OSError as e:
            os.close(fd)
            raise _NoCap("flock of %s: %s" % (name, e.strerror or e))
    return None


def _no_cap(why):
    """The places of this host are not counted in this process from now on: its helpers start
    without one, VF_TRITON_TC_JOBS of its own at most, and no kernel waits for a place; said once,
    at VF_LOG_LEVEL=2."""
    with _lock:
        first = "nocap" not in _state
        _state.setdefault("nocap", why)
    if first:
        say("VF_TRITON_TC: the places of the user's helpers on this host are not counted in this process (%s): "
            "its helpers start without one, %d at most" % (why, _jobs()), level=2)


def _stopped(pgid):
    """Whether a process of a helper's group is stopped (SIGSTOP, a debugger) or the helper's
    control group is frozen: then the time it took is not its compilation's."""
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                with open("/proc/%s/stat" % entry) as f:
                    fields = f.read().rsplit(")", 1)[1].split()
                if int(fields[2]) == pgid and fields[0] in ("T", "t"):
                    return True
            except (OSError, IndexError, ValueError):
                continue
    try:
        with open("/proc/%d/cgroup" % pgid) as f:
            group = [line[3:].strip() for line in f if line.startswith("0::")]
        with open("/sys/fs/cgroup%s/cgroup.events" % group[0]) as f:
            return "frozen 1" in f.read()
    except (OSError, IndexError):
        return False


class _Helper(object):
    """A helper of this process: its socket, its process, the work directory it said; ready (it
    imported the program's Triton), busy (a thread of this process uses it), lost (not to be used
    again), dead (its waiter reaped it)."""

    def __init__(self, pool, slot):
        """A helper started on a slot (the descriptor of its lock file, locked), or without one
        (None: the places are not counted, or this process waits for one no more)."""
        import socket
        import subprocess
        import threading
        import time
        self.pool, self.ready, self.busy, self.lost, self.dead = pool, False, True, False, False
        self.status = self.work = None
        self.written = ""
        self.ended = threading.Event()
        mine = theirs = waiting = None
        try:
            import triton
            root = os.path.dirname(os.path.dirname(os.path.abspath(triton.__file__)))
            mine, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
            if slot is not None:
                # (What tells it to give its slot up when a process waits for one.)
                try:
                    waiting = _open_lock("waiting-" + _host())
                    pool.held.add(waiting)
                except OSError as e:
                    if _limit(e):
                        raise
                    _no_cap("waiting-%s: %s" % (_host(), e.strerror or e))
            fds = (theirs.fileno(), -1 if slot is None else slot, -1 if waiting is None else waiting)
            here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "volta_forever_triton_helper.py")
            self.proc = subprocess.Popen(
                [_state["python"], "-I", "-S", here, "--helper"] + [str(fd) for fd in fds] + [str(os.getpid()), root],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=_helper_env(),
                cwd="/", pass_fds=[fd for fd in fds if fd >= 0], start_new_session=True)
        except BaseException:
            if mine is not None:
                mine.close()
            raise
        finally:
            if slot is not None:
                pool.held.discard(slot)
                os.close(slot)   # (the helper holds it: the slot is free when the helper ends, however)
            if theirs is not None:
                theirs.close()
            if waiting is not None:
                pool.held.discard(waiting)
                os.close(waiting)
        mine.setblocking(False)
        self.sock, self.pid, self.started = mine, self.proc.pid, time.monotonic()
        st = os.fstat(mine.fileno())
        self.inode = (st.st_dev, st.st_ino)
        self.inbox = _Inbox(mine, ANSWER_MAX)
        threading.Thread(target=self._reap, name="volta-forever-helper-%d" % self.pid, daemon=True).start()

    def _reap(self):
        """The helper's waiter: its status when it ends (the wait of its pid alone), what it wrote,
        its work directory removed (it does so itself when it ends otherwise than by a crash);
        what its group still runs (ptxas, when the helper was killed) killed before it is reaped
        (its pid, and so its group's number, still taken)."""
        import signal
        try:
            if hasattr(os, "waitid"):
                os.waitid(os.P_PID, self.pid, os.WEXITED | os.WNOWAIT)
                try:
                    os.killpg(self.pid, signal.SIGKILL)
                except OSError:
                    pass
            self.proc.wait()
            if not hasattr(os, "waitid"):   # (PyPy: the group's number, which its members still hold)
                try:
                    os.killpg(self.pid, signal.SIGKILL)
                except OSError:
                    pass
        except Exception:  # noqa: BLE001 (a program that reaps every child itself: no status)
            pass
        if self.work:
            try:
                with open(os.path.join(self.work, "stderr"), "rb") as f:
                    self.written = f.read()[-20000:].decode("utf-8", "replace")
            except OSError:
                pass
            import shutil
            shutil.rmtree(self.work, ignore_errors=True)
        with self.pool.cond:
            self.status, self.dead = self.proc.returncode, True
            if not self.busy:
                self.pool.drop(self)
            self.pool.cond.notify_all()
        self.ended.set()
        if not self.ready and not self.lost and self.status not in (0, None):
            # (Ended by a signal or an error before this process read that it was ready, as when its
            # import of Triton does not end: another helper would do the same.)
            _broken("its helper ended at its start (status %s)" % self.status)

    def usable(self):
        """Whether an idle helper can take a request: its socket is still this process's, and the
        helper has not ended it (killed while it was idle, before its waiter saw it)."""
        import select
        if not self.ours():
            return False
        if not _wait_for(self.sock, select.POLLIN, 0):
            return True
        try:
            import socket
            return self.sock.recv(1, socket.MSG_PEEK) != b""
        except (BlockingIOError, InterruptedError):
            return True
        except OSError:
            return False

    def ours(self):
        """Whether the socket's descriptor is still this process's socket (a program may close
        descriptors, and open others in their place)."""
        try:
            st = os.fstat(self.sock.fileno())
        except (OSError, ValueError):
            return False
        return (st.st_dev, st.st_ino) == self.inode

    def close(self):
        """Its socket closed, when the descriptor is still this one's (a descriptor that is the
        program's now is left to it)."""
        if self.ours():
            self.sock.close()
        else:
            try:
                self.sock.detach()
            except OSError:
                pass

    def kill(self):
        import signal
        self.lost = True
        if not self.dead:
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except OSError:
                try:
                    os.kill(self.pid, signal.SIGKILL)
                except OSError:
                    pass

    def receive(self, timeout):
        """The helper's next message (a dict), or None when none came in timeout seconds; a time
        during which this process was itself stopped (job control, a debugger) is not counted.
        EOFError when the helper ended its socket; ValueError for what is not a message."""
        import select
        import time
        if not self.ours():
            raise EOFError("its socket is no longer this process's")
        deadline = time.monotonic() + timeout
        while True:
            data = self.inbox.read()
            if data is not None:
                message = _decode(data)
                if message is None:
                    raise ValueError("it sent what is not a message")
                return message
            now = time.monotonic()
            left = deadline - now
            if left <= 0:
                return None
            step = min(left, 1.0)
            _wait_for(self.sock, select.POLLIN, step)
            late = time.monotonic() - now - step
            if late > 2.0:
                deadline += late

    def hello(self, wait):
        """Whether the helper is ready: its first message says that it imported the program's
        Triton (True); none came in wait seconds (False: it is still importing). HelperFailed
        when it failed or ended."""
        try:
            message = self.receive(wait)
        except (EOFError, ValueError, OSError) as e:
            self.kill()
            self.ended.wait(REAP)
            why = "status %s" % self.status if self.status is not None else str(e)
            raise HelperFailed("its helper ended at its start (%s)" % why, False)
        if message is None:
            return False
        if message.get("kind") == "ready" and message.get("identity") == _identity():
            work = message.get("work")
            self.ready, self.work = True, work if isinstance(work, str) else None
            return True
        self.kill()
        if message.get("kind") == "ready":
            raise HelperFailed("its helper imported another Triton than the program's (%s, not %s)"
                               % (message.get("identity"), _identity()), False)
        raise HelperFailed("its helper did not start: %s" % str(message.get("why", message.get("kind")))[:300], False)

    def ask(self, request, timeout):
        """The helper's answer to a request; HelperFailed when there is none: the kernel's when the
        helper ended by a crash or did not answer in timeout seconds (it is then killed with its
        group), else not."""
        import socket
        try:
            if not self.ours():
                raise EOFError("its socket is no longer this process's")
            _send(self.sock, _message(request))
            answer = self.receive(timeout)
        except socket.timeout:
            stopped = _stopped(self.pid)
            self.kill()
            raise HelperFailed("its helper took no request in %g s%s" % (SEND, " (stopped)" if stopped else ""), False)
        except ValueError as e:
            self.kill()
            raise HelperFailed("its helper's answer was not read: %s" % e, False)
        except (EOFError, OSError):
            return self.ended_without_answer()
        if answer is None:
            stopped = _stopped(self.pid)
            self.kill()
            if stopped:
                raise HelperFailed("its helper was stopped from outside (SIGSTOP, a debugger, a frozen control group) "
                                   "past %g s" % timeout, False)
            raise HelperFailed("its sm_80 compilation did not end in %g s" % timeout, True)
        return answer

    def ended_without_answer(self):
        """HelperFailed for a helper whose socket ended with no answer, by the helper's status."""
        self.lost = True
        if not self.ended.wait(REAP):
            self.kill()
            raise HelperFailed("its helper closed its socket with no answer", False)
        status = self.status
        if self.written.strip():
            say("what the sm_80 compilation wrote before its helper ended:\n%s" % self.written.strip(), level=3)
        if status == 0:
            raise HelperFailed("its helper had ended (status 0)", False, gone=True)
        if status is not None and -status in CRASH_SIGNALS:
            raise HelperFailed("its sm_80 compilation ended its process (%s)" % CRASH_SIGNALS[-status], True)
        if status is not None and status < 0:
            raise HelperFailed("its sm_80 compilation was ended from outside (signal %d)" % -status, False)
        raise HelperFailed("its helper ended with no answer (status %s)" % status, False)


# --- the helpers of this process -------------------------------------------------------------

class _Pool(object):
    """The helpers of this process (of its pid: a child made by fork() has its own)."""

    def __init__(self):
        import threading
        self.pid = os.getpid()
        self.cond = threading.Condition(threading.Lock())
        self.helpers = []
        self.starting = 0
        # (Descriptors of locks taken and not yet given to a helper: a child made by fork() at that
        # moment closes its copies, which would hold the lock.)
        self.held = set()
        self.files = {}

    def drop(self, helper):
        if helper in self.helpers:
            self.helpers.remove(helper)
            helper.close()


_pool = None


def _forget(pool):
    """In a child made by fork(): the parent's helpers are the parent's. The child's copies of
    their sockets are closed (a helper then ends when the parent does), and their Popen objects
    never wait for a pid of the child's."""
    for h in list(pool.helpers):
        if h.proc.returncode is None:
            h.proc.returncode = -1
        h.close()


def _in_child():
    global _pool
    old, _pool = _pool, None
    if old is not None and old.pid != os.getpid():
        for fd in list(old.held) + [fd for fd, _ in old.files.values()]:
            try:
                os.close(fd)
            except OSError:
                pass
        _forget(old)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_in_child)


def _the_pool():
    global _pool
    with _lock:
        if _pool is None or _pool.pid != os.getpid():
            old, _pool = _pool, _Pool()
            if old is not None:
                _forget(old)   # (a fork that ran no handler: a C library's)
        return _pool


def _take():
    """A helper of this process for one compilation, made busy: an idle one, or one started on a
    free slot of this host. While the slots are taken, this process holds the lock "waiting"
    shared, which tells the idle helpers of the user's other processes to give theirs up, for
    VF_TRITON_TC_WAIT seconds once in its life: when no slot came, it waits for none again, and
    starts ALONE helper without a slot when none is free (said once). Where the slots are not
    counted (_no_cap), its helpers start without one. _NoSlot when the helpers of this process are
    all busy for VF_TRITON_TC_WAIT seconds. It may still be starting."""
    import fcntl
    import time
    pool = _the_pool()
    wait = _seconds("VF_TRITON_TC_WAIT", WAIT)
    end = time.monotonic() + wait
    waiting = slot = None
    pause = 0.02
    try:
        with pool.cond:
            while True:
                if _state.get("broken"):
                    raise HelperFailed(_state["broken"], False)
                free = [h for h in pool.helpers if not (h.busy or h.lost or h.dead)]
                for h in [h for h in free if not h.usable()]:
                    h.lost = True
                    pool.drop(h)
                    free.remove(h)
                if free:
                    free.sort(key=lambda h: not h.ready)
                    free[0].busy = True
                    return free[0]
                alive = len(pool.helpers) + pool.starting
                if alive < _jobs():
                    if not _state.get("nocap"):
                        try:
                            slot = _slot(pool)
                        except _NoCap as e:
                            _no_cap(str(e))
                    if slot is not None:
                        pool.held.add(slot)
                        break
                    if _state.get("nocap") or (_state.get("alone") and alive < ALONE):
                        break   # (a helper without a slot)
                    if waiting is None and not _state.get("alone"):
                        try:
                            waiting = _open_lock("waiting-" + _host())
                            pool.held.add(waiting)
                            fcntl.flock(waiting, fcntl.LOCK_SH | fcntl.LOCK_NB)
                        except OSError as e:
                            if waiting is not None:
                                pool.held.discard(waiting)
                                os.close(waiting)
                                waiting = None
                            if _limit(e):
                                raise
                            if not isinstance(e, BlockingIOError):   # (not a helper that looks at it alone)
                                _no_cap("waiting-%s: %s" % (_host(), e.strerror or e))
                                break
                left = end - time.monotonic()
                if left <= 0:
                    if alive < _jobs() and not _state.get("alone"):
                        with _lock:
                            _state["alone"] = True
                        say("VF_TRITON_TC: no place for a helper of this host was free in %g s (VF_TRITON_TC_JOBS: %d): "
                            "this process waits for one no more, and compiles with %d helper of its own without a "
                            "place when none is free" % (wait, _jobs(), ALONE), once="alone")
                        if alive < ALONE:
                            break
                    raise _NoSlot("no helper of this process was free in %g s (VF_TRITON_TC_JOBS: %d)" % (wait, _jobs()))
                pool.cond.wait(min(left, pause))
                pause = min(pause * 1.5, 0.2)
            pool.starting += 1
    finally:
        if waiting is not None:
            pool.held.discard(waiting)
            os.close(waiting)
    try:
        helper = _Helper(pool, slot)
    except BaseException:
        with pool.cond:
            pool.starting -= 1
            pool.cond.notify_all()
        raise
    with pool.cond:
        pool.starting -= 1
        pool.helpers.append(helper)
    return helper


def _give(helper):
    pool = helper.pool
    with pool.cond:
        helper.busy = False
        if helper.dead or helper.lost:
            pool.drop(helper)
        pool.cond.notify_all()


def _ready(helper):
    """Whether a helper that was starting is ready now: the first kernel of a process waits for its
    import of Triton VF_TRITON_TC_START seconds at most (counted from its start); then no kernel
    waits for it again (each is compiled by Triton's own stages, not kept, until it is ready)."""
    import time
    start = _seconds("VF_TRITON_TC_START", START)
    with _lock:
        late = _state.get("late")
    if helper.hello(0.0 if late else max(0.0, start - (time.monotonic() - helper.started))):
        return True
    with _lock:
        _state["late"] = True
    _quiet("VF_TRITON_TC: a helper still imports Triton after %g s: kernels are compiled by Triton's own stages, "
           "and not kept in Triton's cache, until it is ready" % start, "late")
    return False


def _quiet(line, once):
    """Say a line once, after which each kernel's line of that kind is for VF_LOG_LEVEL=2."""
    with _lock:
        _state["quiet"] = True
    say(line, once=once)


def _broken(why):
    with _lock:
        first = not _state.get("broken")
        _state["broken"] = _state.get("broken") or why
    if first:
        say("VF_TRITON_TC: every kernel of this process is compiled by Triton's own stages: %s" % why)


# --- what a process takes from its helper ---------------------------------------------------

# The fields of a kernel's metadata that Triton's stages (and compile80, num_stages) set, in Triton
# 3.3 to 3.9; an answer may change no other field of those that the request had.
STAGE_FIELDS = ("name", "num_stages", "num_warps", "shared", "tmem_size", "global_scratch_size", "global_scratch_align",
                "profile_scratch_size", "profile_scratch_align", "tensordesc_meta", "cluster_dims")


def _checked(answer, ttir, metadata, options):
    """(result, None) for an answer "ok" that is a kernel for sm_80 of this request, else (None,
    why): it is checked before anything of it reaches Triton or its cache."""
    import base64
    import binascii
    import re
    result = answer.get("result")
    if not isinstance(result, dict) or not all(isinstance(result.get(k), str) for k in ("ttgir", "llir", "ptx")):
        return None, "it has no stages"
    raw = result.get("cubin")
    raw = raw.get("$bytes") if isinstance(raw, dict) else None
    if not isinstance(raw, str) or len(raw) > CUBIN_MAX * 4 // 3 + 4:
        return None, "it has no cubin"
    try:
        cubin = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        return None, "it has no cubin"
    why = cubin_problem(cubin) or ptx_problem(cubin, result["ptx"])
    if why:
        return None, why
    if not re.search(r"^\s*\.target\s+sm_80\b", result["ptx"], re.M):
        return None, "its PTX is not for sm_80"
    stages, asked = result.get("stages"), max(1, getattr(options, "num_stages", 1) or 1)
    if not isinstance(stages, int) or isinstance(stages, bool) or not 1 <= stages <= asked:
        return None, "its number of stages is %r" % (stages,)
    md = result.get("metadata")
    if not isinstance(md, dict):
        return None, "it has no metadata"
    md = {k: v for k, v in md.items() if k not in ("target", "hash")}
    for k, v in md.items():
        if k in metadata and k not in STAGE_FIELDS and v != _json(metadata[k]):
            return None, "it changes the metadata's %s" % k
    shared = md.get("shared")
    if not isinstance(shared, int) or isinstance(shared, bool) or not 0 <= shared <= _core.VOLTA_SHARED:
        return None, "its shared memory is %r" % (shared,)
    if md.get("num_stages") != stages or md.get("name") != _kernel_name(ttir):
        return None, "its metadata is not that of this kernel"
    return dict(result, cubin=cubin, metadata=md), None


def compile80_isolated(backend, ttir, metadata, options):
    """The sm_80 compilation of a kernel in a helper of this process: (result, why, transient).
    HelperFailed when the kernel ended or hung its helper, or no helper can be used."""
    import errno
    timeout = _seconds("VF_TRITON_TC_TIMEOUT", TIMEOUT)
    request = {"kind": "compile", "ttir": ttir, "target": backend.target, "options": options,
               "metadata": dict(metadata)}
    try:
        digest = request["digest"] = _digest(request)
    except (TypeError, ValueError) as e:
        return None, "its compilation could not be sent to a helper (%s)" % e, False
    for attempt in (1, 2):
        try:
            helper = _take()
        except _NoSlot as e:
            _quiet("VF_TRITON_TC: %s: kernels are compiled by Triton's own stages, and not kept in Triton's cache, "
                   "while its helpers are busy" % e, "slots")
            return None, str(e), True
        except HelperFailed:
            raise
        except Exception as e:  # noqa: BLE001 (a helper that cannot be started: the process goes on without)
            why = "a helper could not be started: %s: %s" % (type(e).__name__, str(e).strip()[:200])
            if isinstance(e, OSError) and e.errno in (errno.EAGAIN, errno.ENOMEM, errno.EMFILE, errno.ENFILE):
                return None, why, True   # (no process or descriptor this time: another time, perhaps)
            _broken(why)
            raise HelperFailed(why, False)
        result = refused = None
        try:
            if not helper.ready and not _ready(helper):
                return None, "its helper is still starting (it imports Triton)", True
            answer = helper.ask(request, timeout)
            if answer.get("digest") != digest:
                refused = "it is not the answer to its request"
            elif answer.get("kind") == "ok":
                result, refused = _checked(answer, ttir, metadata, options)
            if refused:
                helper.kill()   # (before another thread may take it)
        except HelperFailed as e:
            if e.gone and attempt == 1:
                continue   # (it ended when its idle time was up, as the request reached it)
            if not helper.ready:
                _broken(e.why)
            raise
        finally:
            _give(helper)
        kind, why = answer.get("kind"), str(answer.get("why", answer.get("kind")))[:1000]
        written = str(answer.get("stderr") or "").strip()
        if written:
            # (What Triton's compiler wrote, its dumps and "please share the reproducer" included: for
            # VF_LOG_LEVEL=3; the line of the kernel says what came of it.)
            say("what the sm_80 compilation of %s wrote:\n%s" % (_kernel_name(ttir), written[-20000:]), level=3)
            if kind != "ok":
                why += " (VF_LOG_LEVEL=3 shows what the compiler wrote)"
        if result is not None:
            return result, None, False
        if refused:
            why = "its helper's answer was refused: %s" % refused
            _broken(why + "; no helper is used again by this process")
            return None, why, True
        if kind == "error":
            raise HelperFailed(why, True)
        return None, why, kind == "transient"
    raise HelperFailed("its helper had ended twice as a request reached it", False)
