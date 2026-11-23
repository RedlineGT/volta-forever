"""CPU regression for the end of a child of a process that uses the nvJitLink adapter.

A child made by fork() while another thread of the parent waits in a write of the adapter's log
(with the log's lock), and that ends with exit(), ends: the summary of the kernels with emulated
instructions, which the adapter writes when a process ends, is that of the process that loaded it.
(Until 2026-11-23 the child waited for ever for the lock of the log.) A stand-in for nvJitLink
12.9.2, in which every function succeeds, does the linking.

Run with python3 tests/proxy/jitlink_fork.py path/to/libnvJitLink.so.13.
"""
import ctypes as c
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import termios
import threading
import time
import warnings

MOCK = """
#include <stddef.h>
static int handle;
int nvJitLinkCreate(void **h, unsigned n, const char **o) { (void)n; (void)o; *h = &handle; return 0; }
int nvJitLinkDestroy(void **h) { *h = NULL; return 0; }
int nvJitLinkAddData(void *h, int t, const void *d, size_t s, const char *n) { return 0; }
int nvJitLinkAddFile(void *h, int t, const char *f) { return 0; }
int nvJitLinkComplete(void *h) { return 0; }
int nvJitLinkGetLinkedCubinSize(void *h, size_t *s) { *s = 0; return 0; }
int nvJitLinkGetLinkedCubin(void *h, void *b) { return 0; }
int nvJitLinkGetLinkedPtxSize(void *h, size_t *s) { *s = 0; return 0; }
int nvJitLinkGetLinkedPtx(void *h, char *b) { return 0; }
int nvJitLinkGetErrorLogSize(void *h, size_t *s) { *s = 0; return 0; }
int nvJitLinkGetErrorLog(void *h, char *b) { return 0; }
int nvJitLinkGetInfoLogSize(void *h, size_t *s) { *s = 0; return 0; }
int nvJitLinkGetInfoLog(void *h, char *b) { return 0; }
"""

# A kernel with an emulated instruction (tanh.approx.f32): the adapter announces it, and says how
# many there were when the process ends.
EMULATED = (b".version 8.8\n.target sm_75\n.address_size 64\n"
            b".visible .entry k(.param .u64 p)\n{\n .reg .f32 %f<3>;\n .reg .b64 %rd<2>;\n"
            b" ld.param.u64 %rd1, [p];\n ld.global.f32 %f1, [%rd1];\n tanh.approx.f32 %f2, %f1;\n"
            b" st.global.f32 [%rd1], %f2;\n ret;\n}\n\0")
# Adapted with a line of log each time (VF_LOG_LEVEL=2).
PLAIN = b".version 8.8\n.target sm_75\n.address_size 64\n.visible .entry plain()\n{\n ret;\n}\n\0"
SYS_WRITE = 1  # x86_64


def in_write(tid):
    try:
        return Path(f"/proc/self/task/{tid}/syscall").read_text().split()[0] == str(SYS_WRITE)
    except (OSError, IndexError):
        return False


def child_case(adapter, fifo):
    """In a process of its own, as the adapter reads its settings when it is loaded."""
    pipe = os.open(fifo, os.O_RDWR)  # the adapter's open of its log does not wait for a reader
    lib = c.CDLL(adapter)
    handle = c.c_void_p()
    assert lib.nvJitLinkCreate(c.byref(handle), 1, (c.c_char_p * 1)(b"-arch=sm_70")) == 0
    assert lib.nvJitLinkAddData(handle, 2, EMULATED, c.c_size_t(len(EMULATED)), b"emulated.ptx") == 0

    stop, tid = threading.Event(), []

    def adapt():
        tid.append(threading.get_native_id())
        while not stop.is_set():
            lib.nvJitLinkAddData(handle, 2, PLAIN, c.c_size_t(len(PLAIN)), b"plain.ptx")

    thread = threading.Thread(target=adapt)
    thread.start()
    # Until the pipe is full and the thread waits in a write of the log, with the log's lock.
    queued, waits = 0, False
    for _ in range(1000):
        before = queued
        time.sleep(0.01)
        size = c.c_int()
        fcntl.ioctl(pipe, termios.FIONREAD, size)
        queued = size.value
        waits = queued > 32768 and queued == before and tid and in_write(tid[0])
        if waits:
            break
    if not waits:
        print(f"the thread did not come to wait in a write of the log ({queued} bytes in the pipe)")
        os._exit(2)

    libc = c.CDLL(None)
    warnings.simplefilter("ignore", DeprecationWarning)  # that fork() may lead to deadlocks: the point
    child = os.fork()
    if child == 0:
        libc.exit(0)  # the C library's exit(): the adapter's .fini_array runs
    ended, status = False, 0
    for _ in range(500):
        pid, status = os.waitpid(child, os.WNOHANG)
        if pid == child:
            ended = True
            break
        time.sleep(0.01)
    if not ended:
        os.kill(child, signal.SIGKILL)
        os.waitpid(child, 0)

    # Let the thread go: what it writes is read until it is done.
    stop.set()
    os.set_blocking(pipe, False)
    while thread.is_alive():
        try:
            while os.read(pipe, 65536):
                pass
        except BlockingIOError:
            pass
        thread.join(0.001)
    assert lib.nvJitLinkDestroy(c.byref(handle)) == 0
    os.close(pipe)
    if ended and os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0:
        print("ok     fork_exit_child_ends  the child ended")
        return 0
    print("WRONG  fork_exit_child_ends  " + ("the child did not end in 5 s" if not ended else f"status {status}"))
    return 1


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--case":
        sys.exit(child_case(sys.argv[2], sys.argv[3]))
    if len(sys.argv) != 2:
        sys.exit("usage: python3 tests/proxy/jitlink_fork.py ADAPTER (the layer's, as dist/lib/libnvJitLink.so.13)")
    adapter = str(Path(sys.argv[1]).resolve(strict=True))
    with tempfile.TemporaryDirectory() as tmp:
        mock = Path(tmp) / "libnvJitLink.so.12"
        subprocess.run(["cc", "-shared", "-fPIC", "-x", "c", "-", "-o", str(mock)], input=MOCK, text=True, check=True)
        fifo = Path(tmp) / "log.fifo"
        os.mkfifo(fifo, 0o600)
        env = dict(os.environ, VF_JITLINK_REAL=str(mock), VF_LOG=str(fifo), VF_LOG_LEVEL="2",
                   VF_EMULATION_WARNINGS="20")
        env.pop("VF_JITLINK_DUMP", None)
        status = subprocess.run([sys.executable, __file__, "--case", adapter, str(fifo)], env=env).returncode
    print("PASS  nvJitLink adapter at the end of a child" if status == 0 else "FAIL  nvJitLink adapter at the end of a child")
    sys.exit(status)
