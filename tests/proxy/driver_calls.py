"""The calls that reach the driver, recorded: what the layer makes of a program's calls, and the calls
it makes of its own. Without a GPU, against the stand-ins of the other tests.

A recorder stands between the layer and a stand-in for the driver: a library that exports every
function of NVIDIA's libcuda.so.1 (crates/vf-proxy/symbols/), each of which writes its name, with its
process and thread, to the file DRIVER_CALLS names, then jumps to the stand-in's function of that
name (a function the stand-in does not have returns 801, CUDA_ERROR_NOT_SUPPORTED). The stand-in is
a dependency of the recorder, so that a test that looks up its variables in the library it gave as
the driver finds them.

    python3 tests/proxy/driver_calls.py PROXY [BASE]

runs, each in a process of its own: the cases of tests/proxy/lifecycle.c against its stand-in, and
the cases of tests/proxy/fault.py in which nothing faults against its stand-in; each with the
report of faults (the default) and with VF_FAULT_REPORT=0. With BASE (the libcuda.so.1 of a build
without the report of faults, as volta-forever 0.1.5), each is run with it too. It checks that,
thread by thread, the calls are the same:

- with VF_FAULT_REPORT=0 and with BASE: the very same;
- with the report on and with VF_FAULT_REPORT=0: the same, but for the calls that the report makes
  without a fault, each of which is said, which are only these two getters: cuCtxGetCurrent after a
  synchronisation of the current context, of its legacy stream or of the thread's own default stream
  that succeeded (which context's launches it covers); and cuStreamGetCaptureInfo_v2 (or its _ptsz
  form), while a capture is under way in the process, of the stream that the program launched into,
  or whose capture it ends (whether the launch is a node of a graph). Nothing at exit.

Prints one line a case, then PASS or FAIL; exit status 1 on FAIL. Needs gcc and
tests/kernels/k_fit_test.135.sm75.cubin.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
# The calls the report may make without a fault (see above).
GETTERS = {"cuCtxGetCurrent", "cuStreamGetCaptureInfo_v2", "cuStreamGetCaptureInfo_v2_ptsz"}


def recorder(stand_in, directory):
    """A recorder in front of `stand_in` (a built library), in `directory`."""
    names = (ROOT / "crates/vf-proxy/symbols/libcuda-580.178.04.txt").read_text().split()
    lines = ["#define _GNU_SOURCE", "#include <dlfcn.h>", "#include <fcntl.h>", "#include <stdio.h>",
             "#include <stdlib.h>", "#include <sys/syscall.h>", "#include <unistd.h>",
             f"#define N {len(names)}",
             "static const char *const names[N] = {" + ", ".join(f'"{n}"' for n in names) + "};",
             "__attribute__((visibility(\"hidden\"))) void *targets[N];",
             "static int fd = -1;",
             "static int absent(void) { return 801; }",
             "__attribute__((constructor)) static void setup(void) {",
             "    void *inner = dlopen(getenv(\"DRIVER_CALLS_INNER\"), RTLD_NOW | RTLD_NOLOAD);",
             "    for (int i = 0; i < N; i++) {",
             "        targets[i] = inner ? dlsym(inner, names[i]) : 0;",
             "        if (!targets[i]) targets[i] = (void *)absent;",
             "    }",
             "    const char *log = getenv(\"DRIVER_CALLS\");",
             "    if (log) fd = open(log, O_WRONLY | O_CREAT | O_APPEND, 0600);",
             "}",
             "__attribute__((visibility(\"hidden\"), used)) void driver_call(long i) {",
             "    if (fd < 0) return;",
             "    char line[160];",
             "    int n = snprintf(line, sizeof line, \"%d %ld %s\\n\", getpid(), (long)syscall(SYS_gettid), names[i]);",
             "    if (write(fd, line, n) != n) return;",
             "}"]
    saved = ["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9", "%rax"]
    for i, name in enumerate(names):
        body = [f"push {r}" for r in saved] + ["sub $128, %rsp"]
        body += [f"movdqu %xmm{x}, {16 * x}(%rsp)" for x in range(8)]
        body += [f"mov ${i}, %edi", "call driver_call"]
        body += [f"movdqu {16 * x}(%rsp), %xmm{x}" for x in range(8)]
        body += ["add $128, %rsp"] + [f"pop {r}" for r in reversed(saved)]
        body += [f"jmp *targets+{8 * i}(%rip)"]
        lines.append(f"__attribute__((naked)) void {name}(void) {{ __asm__(\"" + "\\n".join(body) + "\"); }")
    source = Path(directory) / "recorder.c"
    source.write_text("\n".join(lines) + "\n")
    out = Path(directory) / "librecorder.so"
    subprocess.run(["gcc", "-shared", "-fPIC", "-O1", "-o", str(out), str(source), "-Wl,--no-as-needed",
                    str(stand_in), f"-Wl,-rpath,{Path(stand_in).parent}", "-ldl"], check=True)
    return out


def threads(log):
    """The calls of `log`, as a list of each thread's, processes and threads in the order they first
    called."""
    order, calls = [], {}
    for line in log.read_text().splitlines() if log.exists() else []:
        pid, tid, name = line.split()
        if (pid, tid) not in calls:
            order.append((pid, tid))
            calls[(pid, tid)] = []
        calls[(pid, tid)].append(name)
    return [calls[key] for key in order]


def extra(with_report, without):
    """What one thread called with the report beyond what it called without (in the same order
    otherwise): the getters, as {name: count}; None when it called anything else, or not in the
    same order."""
    extra, j = {}, 0
    for name in with_report:
        if j < len(without) and name == without[j]:
            j += 1
        elif name in GETTERS:
            extra[name] = extra.get(name, 0) + 1
        else:
            return None
    return extra if j == len(without) else None


def main():
    proxy = os.path.abspath(sys.argv[1])
    base = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else None
    failed = 0
    with tempfile.TemporaryDirectory(prefix="vf-driver-calls.") as tmp:
        tmp = Path(tmp)
        cubin = ROOT / "tests/kernels/k_fit_test.135.sm75.cubin"
        subprocess.run(["gcc", "-shared", "-fPIC", "-O0", "-Wl,-Bsymbolic", "-o", str(tmp / "lifecycle_mock.so"),
                        str(HERE / "lifecycle_mock_driver.c")], check=True)
        subprocess.run(["gcc", "-O2", "-Wall", "-o", str(tmp / "lifecycle"), str(HERE / "lifecycle.c"), "-ldl",
                        "-lpthread"], check=True)
        subprocess.run(["gcc", "-shared", "-fPIC", "-Wl,-Bsymbolic", "-o", str(tmp / "fault_mock.so"),
                        str(HERE / "fault_mock_driver.c")], check=True)
        (tmp / "lifecycle_rec").mkdir()
        (tmp / "fault_rec").mkdir()
        stand_ins = {"lifecycle": recorder(tmp / "lifecycle_mock.so", tmp / "lifecycle_rec"),
                     "fault": recorder(tmp / "fault_mock.so", tmp / "fault_rec")}
        inner = {"lifecycle": tmp / "lifecycle_mock.so", "fault": tmp / "fault_mock.so"}
        cases = [("lifecycle", name) for name in ("fork_exit", "contexts", "lazy_lock", "cache_temporaries",
                                                  "file_path")]
        cases += [("fault", name) for name in ("nofault", "nofault-graph", "nofault-release", "nofault-capture-exit")]

        def run(which, name, library, settings, label):
            log = tmp / f"{which}-{name}-{label}.calls"
            log.unlink(missing_ok=True)
            work = tmp / f"{which}-{name}-{label}"
            work.mkdir()
            env = dict(os.environ, VF_REAL_LIBCUDA=str(stand_ins[which]), DRIVER_CALLS=str(log),
                       DRIVER_CALLS_INNER=str(inner[which]), VF_PROXY_CACHE="0", VF_LOG_LEVEL="0",
                       VF_EMULATION_WARNINGS="0", CUBIN=str(cubin), TMPDIR=str(work), **settings)
            for name_ in ("VF_FAULT_TRACE", "VF_PROFILE", "VF_STATS", "VF_SAFE"):
                if name_ not in settings:
                    env.pop(name_, None)
            if which == "lifecycle":
                command = [str(tmp / "lifecycle"), name, str(library), str(work)]
            else:
                command = [sys.executable, str(HERE / "fault.py"), str(library), name]
            done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=300)
            return done.returncode, threads(log)

        for which, name in cases:
            label = f"{which}.{name}"
            status, on = run(which, name, proxy, {}, "on")
            status_off, off = run(which, name, proxy, {"VF_FAULT_REPORT": "0"}, "off")
            if status or status_off:
                print(f"FAIL  {label}: the case failed ({status}, {status_off})")
                failed += 1
                continue
            same = True
            said = []
            if len(on) != len(off):
                same = False
                said.append(f"{len(on)} threads called the driver with the report, {len(off)} without")
            for number, (a, b) in enumerate(zip(on, off)):
                found = extra(a, b)
                if found is None:
                    same = False
                    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                    said.append(f"thread {number + 1}: {len(a)} calls with the report, {len(b)} without, "
                                f"other from call {first + 1}: {a[first:first + 6]} for {b[first:first + 6]}")
                    continue
                if found:
                    said.append(f"thread {number + 1}: with the report, also "
                                + ", ".join(f"{count} {name}" for name, count in sorted(found.items())))
            if base:
                status_base, at_base = run(which, name, base, {}, "base")
                if status_base or at_base != off:
                    same = False
                    said.append(f"VF_FAULT_REPORT=0 differs from the base build ({status_base}: "
                                f"{sum(map(len, at_base))} calls, {sum(map(len, off))} with VF_FAULT_REPORT=0)")
                else:
                    said.append("VF_FAULT_REPORT=0 and the base build: the very same calls")
            calls = sum(map(len, off))
            print(f"{'ok  ' if same else 'FAIL'}  {label}: {calls} calls without the report"
                  + "".join(f"; {s}" for s in said))
            failed += not same
    print("PASS  driver_calls" if not failed else "FAIL  driver_calls")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
