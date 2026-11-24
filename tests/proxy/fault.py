"""After a fault, the layer names the kernels that may have faulted (the launches of the context that
no synchronisation that succeeded covers: not those of another context, nor those made before a
synchronisation of their stream, the thread's own default stream included; those of every thread),
with what it made of their image, once, at level 0, and in the report at exit; under
VF_FAULT_TRACE=1 it names the kernel alone. Without a GPU, against a
stand-in for the driver (fault_mock_driver.c) that shows a fault as NVIDIA's driver does (700 from
the call after the launch of write_null on).

Run with python3 tests/proxy/fault.py path/to/libcuda.so.1. Each case runs in a process of its own.
The cases that fault on a V100 are test_fault.c's (tests/gpu/run.sh with VF_TEST_FAULTS=1).
"""
import ctypes as c
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading

PTX = (b".version 7.0\n.target sm_75\n.address_size 64\n"
       b".visible .entry add_one(.param .u64 p)\n{\n ret;\n}\n"
       b".visible .entry write_null(.param .u64 p)\n{\n ret;\n}\n")
FAULT = "a kernel faulted on the GPU"


class Run:
    def __init__(self, proxy):
        self.api = c.CDLL(proxy)
        assert self.api.cuInit(0) == 0
        self.context, self.functions = self.new_context()

    def new_context(self):
        """A context made current, with the module loaded in it: the context and its functions."""
        context, module = c.c_void_p(), c.c_void_p()
        assert self.api.cuCtxCreate_v2(c.byref(context), 0, 0) == 0
        assert self.api.cuModuleLoadData(c.byref(module), PTX) == 0
        functions = {}
        for name in ("add_one", "write_null"):
            f = c.c_void_p()
            assert self.api.cuModuleGetFunction(c.byref(f), module, name.encode()) == 0
            functions[name] = f
        return context, functions

    def launch(self, name, stream=None, grid=1, functions=None, own_stream=False):
        word = c.c_uint64(0)
        args = (c.c_void_p * 1)(c.cast(c.byref(word), c.c_void_p))
        launch = self.api.cuLaunchKernel_ptsz if own_stream else self.api.cuLaunchKernel
        return launch((functions or self.functions)[name], grid, 1, 1, 32, 1, 1, 0, stream, args, None)

    def stream(self):
        s = c.c_void_p()
        assert self.api.cuStreamCreate(c.byref(s), 1) == 0
        return s

    def copy(self):
        host = c.c_uint32(0)
        return self.api.cuMemcpyDtoH_v2(c.byref(host), c.c_uint64(0x7f0000000000), 4)


def case(name, proxy):
    run = Run(proxy)
    out = []
    if name in ("candidates", "nofault", "exact"):
        out.append(run.launch("add_one", grid=1))
        out.append(run.api.cuCtxSynchronize())
        out.append(run.launch("add_one", grid=2))
        out.append(run.api.cuStreamSynchronize(None))
        out.append(run.launch("add_one", grid=3))
        out.append(run.launch("add_one" if name == "nofault" else "write_null", grid=4))
        out.append(run.launch("add_one", grid=5))
        out.append(run.api.cuCtxSynchronize())
        # Met again: said once.
        out.append(run.api.cuCtxSynchronize())
        out.append(run.launch("add_one", grid=6))
    elif name == "streams":
        first, second = run.stream(), run.stream()
        out.append(run.launch("add_one", first, grid=1))
        out.append(run.api.cuStreamSynchronize(first))
        out.append(run.launch("write_null", second, grid=2))
        out.append(run.api.cuStreamSynchronize(first))
    elif name in ("graph", "graph-exact"):
        stream = run.stream()
        graph, exec_ = c.c_void_p(), c.c_void_p()
        out.append(run.launch("add_one", stream, grid=1))
        out.append(run.api.cuStreamSynchronize(stream))
        out.append(run.api.cuStreamBeginCapture_v2(stream, 2))
        out.append(run.launch("add_one", stream, grid=2))
        out.append(run.launch("write_null", stream, grid=3))
        out.append(run.api.cuStreamEndCapture(stream, c.byref(graph)))
        out.append(run.api.cuGraphInstantiateWithFlags(c.byref(exec_), graph, c.c_ulonglong(0)))
        out.append(run.api.cuGraphLaunch(exec_, stream))
        out.append(run.api.cuStreamSynchronize(stream))
    elif name == "contexts":
        # A launch in another context, not synchronised there, is not a candidate of this one's fault.
        first = run.functions
        other, second = run.new_context()
        out.append(run.api.cuCtxSetCurrent(run.context))
        out.append(run.launch("add_one", grid=1, functions=first))
        out.append(run.api.cuCtxSetCurrent(other))
        out.append(run.launch("add_one", grid=2, functions=second))
        out.append(run.launch("write_null", grid=3, functions=second))
        out.append(run.api.cuCtxSynchronize())
    elif name == "context-gone":
        # A launch in a context destroyed since is not a candidate of a fault in the next one.
        out.append(run.launch("add_one", grid=1))
        out.append(run.api.cuCtxDestroy_v2(run.context))
        other, second = run.new_context()
        out.append(run.launch("add_one", grid=2, functions=second))
        out.append(run.launch("write_null", grid=3, functions=second))
        out.append(run.api.cuCtxSynchronize())
    elif name == "own-stream":
        # The thread's own default stream: its synchronisation covers the launches made on it before.
        out.append(run.launch("add_one", grid=1, own_stream=True))
        out.append(run.api.cuStreamSynchronize_ptsz(None))
        out.append(run.launch("write_null", grid=2, own_stream=True))
        out.append(run.launch("add_one", grid=3, own_stream=True))
        out.append(run.api.cuCtxSynchronize())
    elif name == "threads":
        # Launches of two threads, each holding its record of launches: all are candidates.
        out.append(run.launch("add_one", grid=1))
        def other_thread():
            out.append(run.api.cuCtxSetCurrent(run.context))
            out.append(run.launch("add_one", grid=2))
        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join()
        out.append(run.launch("write_null", grid=3))
        out.append(run.api.cuCtxSynchronize())
    elif name == "exact-capture":
        # Exact mode: a launch is not waited for while a graph is being captured (a synchronisation
        # could be refused and end the capture), and is waited for once the capture ended.
        mock = c.CDLL(os.environ["VF_REAL_LIBCUDA"])
        waits = c.c_uint.in_dll(mock, "synchronized")
        captured, other = run.stream(), run.stream()
        graph = c.c_void_p()
        out.append(run.api.cuStreamBeginCapture_v2(captured, 2))
        before = waits.value
        out.append(run.launch("add_one", other, grid=1))
        out.append(waits.value - before)
        out.append(run.api.cuStreamEndCapture(captured, c.byref(graph)))
        out.append(run.launch("add_one", other, grid=2))
        out.append(waits.value - before)
    elif name in ("copy", "exit"):
        out.append(run.launch("add_one", grid=1))
        out.append(run.launch("write_null", grid=2))
        # A copy is not routed through the layer: the fault is found later.
        out.append(run.copy())
        if name == "copy":
            out.append(run.api.cuCtxDestroy_v2(run.context))
    print("results", " ".join(map(str, out)), flush=True)


def checks(name, log, summary, results):
    lines = [line.removeprefix("[volta-forever] ") for line in log.splitlines()]
    faults = [i for i, line in enumerate(lines) if "faulted on the GPU" in line]
    names = []
    if faults:
        for line in lines[faults[0] + 1:]:
            if not line.startswith("  "):
                break
            names.append(line.strip().split(" ")[0])
    found = []

    def expect(what, ok):
        found.append((what, ok))

    if name == "nofault":
        expect("no fault is said", not faults)
        expect("no fault in the report at exit", "faulted" not in summary)
        expect("every call returned 0", results == ["0"] * 10)
        return found
    if name == "exact-capture":
        expect("no fault is said", not faults)
        expect("not waited for during a capture, waited for after", results == ["0", "0", "0", "0", "0", "1"])
        return found
    expect("one fault is said, once", len(faults) == 1)
    if not faults:
        return found
    head = lines[faults[0]]
    if name == "candidates":
        expect("cuCtxSynchronize met 700", "cuCtxSynchronize returned 700 (CUDA_ERROR_ILLEGAL_ADDRESS)" in head)
        expect("the three launches since the stream's synchronisation, the last first",
               "one of the 3 launched in this context since its last synchronisation" in head and
               names == ["add_one", "write_null", "add_one"])
        expect("their grids", re.search(r"add_one \(cuLaunchKernel, grid 5x1x1, block 32x1x1, the null stream\)",
                                        log) is not None and "grid 4x1x1" in log and "grid 2x1x1" not in log)
        expect("what the layer made of the image", "PTX for sm_75 retargeted by the layer" in log and
               re.search(r"the image given to the driver: \d+ bytes, XXH64 [0-9a-f]{16}", log) is not None)
        expect("the program's image without its digest at level 1",
               f"the program's image: cuModuleLoadData of {len(PTX)} bytes (VF_LOG_LEVEL=2 prints its XXH64" in log)
        expect("how to name it alone", "VF_FAULT_TRACE=1 synchronises every launch" in log)
        expect("the report at exit", "a kernel faulted: cuCtxSynchronize returned 700 (CUDA_ERROR_ILLEGAL_ADDRESS); "
               "the candidates, the last launched first: add_one, write_null, add_one" in summary)
    elif name == "exact":
        expect("named alone", head.startswith("kernel write_null faulted on the GPU: the synchronisation of the "
                                              "stream after cuLaunchKernel (VF_FAULT_TRACE=1) returned 700"))
        expect("its launch returned the fault", results[5] == "700")
        expect("the report at exit", "a kernel faulted: the synchronisation of the stream after cuLaunchKernel "
               "(VF_FAULT_TRACE=1) returned 700 (CUDA_ERROR_ILLEGAL_ADDRESS); the kernel: write_null" in summary)
    elif name == "streams":
        expect("the other stream's synchronisation does not cover it", names == ["write_null"])
    elif name == "contexts":
        expect("the launches of this context only", names == ["write_null", "add_one"] and "grid 1x1x1" not in log)
        expect("no launch said to be of another context", "of a context that the layer does not know" not in log)
    elif name == "context-gone":
        expect("the launches of the new context only", names == ["write_null", "add_one"] and "grid 1x1x1" not in log)
    elif name == "own-stream":
        expect("the launches made after the synchronisation of the thread's stream",
               names == ["add_one", "write_null"] and "grid 1x1x1" not in log and
               "cuLaunchKernel_ptsz, grid 2x1x1, block 32x1x1, the thread's default stream" in log)
    elif name == "threads":
        threads = {grid: re.search(rf"\(cuLaunchKernel, grid {grid}x1x1, block 32x1x1, the null stream, thread (\d+)\)",
                                   log) for grid in (1, 2, 3)}
        expect("the launches of both threads, each thread's the last first, named by thread",
               names == ["write_null", "add_one", "add_one"] and all(threads.values()) and
               threads[1].group(1) == threads[3].group(1) != threads[2].group(1))
    elif name == "graph":
        expect("a graph launch", "the kernel is the one launched in this context since its" in head)
        expect("its kernel nodes as captured, not as launches",
               re.search(r"a CUDA graph \(cuGraphLaunch, stream 0x[0-9a-f]+\): its kernels ran without passing through "
                         r"the layer; it was made of 2 kernel nodes, of 2 kernels:", log) is not None and
               "    add_one (grid 2x1x1, block 32x1x1)" in log and "    write_null (grid 3x1x1, block 32x1x1)" in log)
    elif name == "graph-exact":
        expect("the graph named", head.startswith("a kernel of a CUDA graph faulted on the GPU") and
               "    write_null (grid 3x1x1" in log)
        expect("the graph's launch returned the fault", results[7] == "700")
    elif name == "copy":
        expect("found before the context was destroyed",
               "cuStreamQuery, asked by the layer before the context was destroyed returned 700" in head)
        expect("the copy met it first", results[2] == "700")
        expect("both launches", names == ["write_null", "add_one"])
    elif name == "exit":
        expect("found at exit", "cuStreamQuery, asked by the layer at exit returned 700" in head)
        expect("the report at exit", "asked by the layer at exit returned 700" in summary)
    return found


def main():
    if len(sys.argv) == 3:
        case(sys.argv[2], sys.argv[1])
        return 0
    proxy = os.path.abspath(sys.argv[1])
    here = Path(__file__).resolve().parent
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        mock = str(Path(tmp) / "libcuda_mock.so")
        subprocess.run(["cc", "-shared", "-fPIC", "-Wl,-Bsymbolic", "-o", mock, str(here / "fault_mock_driver.c")],
                       check=True)
        cases = [("candidates", {}), ("nofault", {}), ("exact", {"VF_FAULT_TRACE": "1"}), ("streams", {}),
                 ("contexts", {}), ("context-gone", {}), ("own-stream", {}), ("threads", {}),
                 ("graph", {}), ("graph-exact", {"VF_FAULT_TRACE": "1"}), ("exact-capture", {"VF_FAULT_TRACE": "1"}),
                 ("copy", {}), ("exit", {}),
                 # Exact mode needs the exact value.
                 ("candidates", {"VF_FAULT_TRACE": "true"})]
        for name, settings in cases:
            log, summary = Path(tmp) / f"{name}.log", Path(tmp) / f"{name}.summary"
            log.unlink(missing_ok=True)
            env = dict(os.environ, VF_REAL_LIBCUDA=mock, VF_PROXY_CACHE="0", VF_LOG=str(log),
                       VF_SUMMARY=str(summary), **settings)
            env.pop("VF_LOG_LEVEL", None)
            done = subprocess.run([sys.executable, __file__, proxy, name], env=env, capture_output=True, text=True)
            results = next((line.split()[1:] for line in done.stdout.splitlines() if line.startswith("results")), [])
            text = log.read_text() if log.exists() else ""
            summaries = list(Path(tmp).glob(f"{name}.summary.*"))
            report = summaries[0].read_text() if summaries else ""
            for s in summaries:
                s.unlink()
            label = name + "".join(f" {k}={v}" for k, v in settings.items())
            outcome = checks(name, text, report, results) if done.returncode == 0 else [("the case ran", False)]
            for what, ok in outcome:
                print(f"  {'ok    ' if ok else 'WRONG '} {label}: {what}")
                failed += not ok
            if any(not ok for _, ok in outcome):
                print(done.stdout + done.stderr + text)
    print(f"{failed} wrong" if failed else "all checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
