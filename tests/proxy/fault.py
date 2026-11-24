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
import signal
import subprocess
import sys
import tempfile
import threading
import time

PTX = (b".version 7.0\n.target sm_75\n.address_size 64\n"
       b".visible .entry add_one(.param .u64 p)\n{\n ret;\n}\n"
       b".visible .entry write_null(.param .u64 p)\n{\n ret;\n}\n"
       b".visible .entry ecc_fail(.param .u64 p)\n{\n ret;\n}\n")


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
        for name in ("add_one", "write_null", "ecc_fail"):
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
    if name in ("candidates", "nofault", "exact", "off"):
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
        # Whom a lookup leads to: the layer's own synchronisation, or, with VF_FAULT_REPORT=0, the
        # driver's (as without the report of faults; the launch is the layer's while it adapts code).
        mock = c.CDLL(os.environ["VF_REAL_LIBCUDA"])
        for function in ("cuStreamSynchronize", "cuLaunchKernel"):
            pointer = c.c_void_p()
            assert run.api.cuGetProcAddress(function.encode(), c.byref(pointer), 12000, c.c_uint64(0)) == 0
            ours = pointer.value == c.cast(getattr(mock, function), c.c_void_p).value
            out.append(f"{function}:{'driver' if ours else 'layer'}")
        out.append(c.c_uint.in_dll(mock, "synchronized").value)
    elif name == "nofault-graph":
        # A graph captured, instantiated, launched and destroyed, and streams made and destroyed:
        # nothing faults.
        stream = run.stream()
        graph, exec_ = c.c_void_p(), c.c_void_p()
        out.append(run.launch("add_one", stream, grid=1))
        out.append(run.api.cuStreamSynchronize(stream))
        out.append(run.api.cuStreamQuery(stream))
        out.append(run.api.cuStreamBeginCapture_v2(stream, 2))
        out.append(run.launch("add_one", stream, grid=2))
        out.append(run.launch("add_one", None, grid=3, own_stream=True))
        out.append(run.api.cuStreamEndCapture(stream, c.byref(graph)))
        out.append(run.api.cuGraphInstantiateWithFlags(c.byref(exec_), graph, c.c_ulonglong(0)))
        out.append(run.api.cuGraphLaunch(exec_, stream))
        out.append(run.api.cuStreamSynchronize(stream))
        out.append(run.api.cuStreamSynchronize_ptsz(None))
        out.append(run.api.cuGraphExecDestroy(exec_))
        out.append(run.api.cuGraphDestroy(graph))
        out.append(run.api.cuStreamDestroy_v2(stream))
        out.append(run.api.cuCtxDestroy_v2(run.context))
    elif name in ("nofault-release", "nofault-capture-exit"):
        # The capture of a blocking stream of the primary context, during which the program retains
        # and releases the primary context (not for the last time) and destroys another context: a
        # use of the legacy stream of the primary context would invalidate the capture. With
        # nofault-capture-exit, the program ends while the capture is under way.
        primary, module, function = c.c_void_p(), c.c_void_p(), c.c_void_p()
        out.append(run.api.cuDevicePrimaryCtxRetain(c.byref(primary), 0))
        out.append(run.api.cuCtxSetCurrent(primary))
        out.append(run.api.cuModuleLoadData(c.byref(module), PTX))
        out.append(run.api.cuModuleGetFunction(c.byref(function), module, b"add_one"))
        stream = c.c_void_p()
        out.append(run.api.cuStreamCreate(c.byref(stream), 0))
        out.append(run.launch("add_one", stream, grid=1, functions={"add_one": function}))
        out.append(run.api.cuStreamSynchronize(stream))
        out.append(run.api.cuStreamBeginCapture_v2(stream, 2))
        out.append(run.launch("add_one", stream, grid=2, functions={"add_one": function}))
        again = c.c_void_p()
        out.append(run.api.cuDevicePrimaryCtxRetain(c.byref(again), 0))
        out.append(run.api.cuDevicePrimaryCtxRelease_v2(0))
        out.append(run.api.cuCtxDestroy_v2(run.context))
        out.append(run.launch("add_one", stream, grid=3, functions={"add_one": function}))
        if name == "nofault-release":
            graph = c.c_void_p()
            out.append(run.api.cuStreamEndCapture(stream, c.byref(graph)))
            out.append(run.api.cuDevicePrimaryCtxRelease_v2(0))
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
    elif name == "deadlock":
        # A report, which looks up the context of every launch kept (here 120 threads' 256 each), while
        # another thread destroys contexts that have modules: contexts::destroyed holds its lock while
        # it has fault forget the names of each module, so no lock of fault.rs may be held while
        # contexts is asked. (Each launching thread holds its record until all have launched.)
        barrier = threading.Barrier(120)

        def launcher():
            run.api.cuCtxSetCurrent(run.context)
            for i in range(256):
                run.launch("add_one", grid=1 + i)
            barrier.wait()
        threads = [threading.Thread(target=launcher) for _ in range(120)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        out.append(run.launch("write_null", grid=7))
        stop, destroyed = threading.Event(), [0]

        def destroyer():
            while not stop.is_set():
                context = c.c_void_p()
                run.api.cuCtxCreate_v2(c.byref(context), 0, 0)
                for _ in range(16):
                    module = c.c_void_p()
                    run.api.cuModuleLoadData(c.byref(module), PTX)
                run.api.cuCtxDestroy_v2(context)
                destroyed[0] += 1
        other = threading.Thread(target=destroyer)
        other.start()
        while destroyed[0] < 3:
            time.sleep(0.001)
        out.append(run.api.cuCtxSynchronize())
        stop.set()
        other.join()
    elif name in ("other-context-stream", "other-context-event", "no-context"):
        # A fault of context A met by a wait for a stream (or an event) of A made while context B is
        # current, where B has launches of its own not synchronised: the fault is A's. (With
        # no-context, the wait is made with no context current.)
        a, functions_a = run.context, run.functions
        stream, event = run.stream(), c.c_void_p()
        assert run.api.cuEventCreate(c.byref(event), 0) == 0
        b, functions_b = run.new_context()
        out.append(run.launch("add_one", grid=21, functions=functions_b))
        out.append(run.launch("add_one", grid=22, functions=functions_b))
        out.append(run.api.cuCtxSetCurrent(a))
        out.append(run.launch("write_null", stream, grid=11, functions=functions_a))
        out.append(run.api.cuEventRecord(event, stream))
        out.append(run.api.cuCtxSetCurrent(None if name == "no-context" else b))
        if name == "other-context-stream":
            out.append(run.api.cuStreamSynchronize(stream))
        else:
            out.append(run.api.cuEventSynchronize(event))
        # A, later: its own synchronisation.
        out.append(run.api.cuCtxSetCurrent(a))
        out.append(run.api.cuCtxSynchronize())
    elif name == "ecc":
        # An uncorrectable ECC error (214), as an aged V100's memory reports one: not a kernel's fault.
        out.append(run.launch("add_one", grid=1))
        out.append(run.launch("ecc_fail", grid=2))
        out.append(run.api.cuCtxSynchronize())
    elif name == "wrapped":
        # More launches than a thread's record keeps since the last synchronisation: the report at exit
        # says that launches before those named may be candidates too.
        out.append(run.launch("write_null", grid=1))
        for i in range(300):
            run.launch("add_one", grid=2 + i)
        out.append(run.api.cuCtxSynchronize())
    elif name == "unseen":
        # VF_ADAPT=0 VF_FIT=0: a launch through the driver's own entry, which the layer does not see.
        pointer = c.c_void_p()
        assert run.api.cuGetProcAddress(b"cuLaunchKernel", c.byref(pointer), 12000, c.c_uint64(0)) == 0
        launch = c.CFUNCTYPE(c.c_int, *([c.c_void_p] + [c.c_uint] * 7 + [c.c_void_p] * 3))(pointer.value)
        word = c.c_uint64(0)
        args = (c.c_void_p * 1)(c.cast(c.byref(word), c.c_void_p))
        out.append(launch(run.functions["write_null"], 1, 1, 1, 32, 1, 1, 0, None, args, None))
        out.append(run.api.cuCtxSynchronize())
    elif name == "exact-done":
        # Exact mode, the fault seen one wait late (MOCK_DELAY=1, as on a GPU where write_null still
        # runs when the next wait returns): write_null is launched while another stream is captured
        # (not waited for), then add_one twice on a third stream; the wait after the first succeeds,
        # the wait after the second meets the fault.
        captured, other, third = run.stream(), run.stream(), run.stream()
        graph = c.c_void_p()
        out.append(run.api.cuStreamBeginCapture_v2(captured, 2))
        out.append(run.launch("write_null", other, grid=1))
        out.append(run.launch("add_one", captured, grid=2))
        out.append(run.api.cuStreamEndCapture(captured, c.byref(graph)))
        out.append(run.launch("add_one", third, grid=3))
        out.append(run.launch("add_one", third, grid=4))
    elif name == "exec-reuse":
        # An instantiated graph destroyed after its launch (it may be destroyed while it runs), and
        # another instantiated at its handle (MOCK_EXEC_FIXED=1): the launch is of the first one.
        stream = run.stream()
        graphs, execs = [c.c_void_p(), c.c_void_p()], [c.c_void_p(), c.c_void_p()]
        out.append(run.api.cuStreamBeginCapture_v2(stream, 2))
        out.append(run.launch("add_one", stream, grid=1))
        out.append(run.launch("write_null", stream, grid=2))
        out.append(run.api.cuStreamEndCapture(stream, c.byref(graphs[0])))
        out.append(run.api.cuGraphInstantiateWithFlags(c.byref(execs[0]), graphs[0], c.c_ulonglong(0)))
        out.append(run.api.cuGraphLaunch(execs[0], stream))
        out.append(run.api.cuGraphExecDestroy(execs[0]))
        out.append(run.api.cuStreamBeginCapture_v2(stream, 2))
        out.append(run.launch("add_one", stream, grid=3))
        out.append(run.api.cuStreamEndCapture(stream, c.byref(graphs[1])))
        out.append(run.api.cuGraphInstantiateWithFlags(c.byref(execs[1]), graphs[1], c.c_ulonglong(0)))
        out.append(run.api.cuStreamSynchronize(stream))
        out.append("same-handle" if execs[0].value == execs[1].value else "other-handle")
    elif name == "own-stream-threads":
        # A thread's own default stream is not the next thread's, even when that thread is given the
        # ended one's record of launches: its synchronisation covers its own launches only. (The fault
        # is seen one wait late, MOCK_DELAY=1, as on a GPU where write_null still runs.)
        def first():
            out.append(run.api.cuCtxSetCurrent(run.context))
            out.append(run.launch("write_null", grid=1, own_stream=True))

        def second():
            out.append(run.api.cuCtxSetCurrent(run.context))
            out.append(run.launch("add_one", grid=2, own_stream=True))
            out.append(run.api.cuStreamSynchronize_ptsz(None))
        for target in (first, second):
            thread = threading.Thread(target=target)
            thread.start()
            thread.join()
        out.append(run.api.cuCtxSynchronize())
    elif name == "unload-failed":
        # The unload of a module that fails (here, after the fault, which only a copy met) leaves it
        # loaded: its kernels are still named as what the layer made of them.
        module, other = c.c_void_p(), c.c_void_p()
        assert run.api.cuModuleLoadData(c.byref(module), PTX) == 0
        assert run.api.cuModuleGetFunction(c.byref(other), module, b"add_one") == 0
        out.append(run.launch("add_one", grid=1, functions={"add_one": other}))
        out.append(run.launch("write_null", grid=2))
        out.append(run.copy())
        out.append(run.api.cuModuleUnload(module))
    elif name == "profile":
        # VF_PROFILE waits for each launch: its wait that met the fault is the one reported, and its
        # waits that succeeded cover what they waited for.
        out.append(run.launch("add_one", grid=1))
        out.append(run.launch("write_null", grid=2))
        out.append(run.launch("add_one", grid=3))
        out.append(run.api.cuCtxSynchronize())
    elif name in ("sigexit", "sigexit-report"):
        # exit() called by a handler of a signal (a handler in C: libc's exit itself, given the
        # signal's number) delivered during a call of the driver that holds a lock of the driver's
        # (MOCK_SIGNAL=1: the stand-in's cuCtxSynchronize raises SIGUSR1 while it holds the lock that
        # its cuStreamQuery and cuCtxPushCurrent_v2 take), after launches: the process ends, with
        # status 10 (SIGUSR1's number). With sigexit-report, after a fault reported: the report at exit
        # still has it.
        libc = c.CDLL(None)
        libc.signal.argtypes = [c.c_int, c.c_void_p]
        libc.signal.restype = c.c_void_p
        libc.signal(signal.SIGUSR1, c.cast(libc.exit, c.c_void_p))
        out.append(run.launch("add_one", grid=1))
        if name == "sigexit-report":
            out.append(run.launch("write_null", grid=2))
            out.append(run.api.cuStreamSynchronize(None))
        print("results", " ".join(map(str, out)), flush=True)
        run.api.cuCtxSynchronize()
        print("not exited", flush=True)
    elif name in ("copy", "copy-destroyed", "exit"):
        out.append(run.launch("add_one", grid=1))
        out.append(run.launch("write_null", grid=2))
        # A copy is not routed through the layer: the fault is found at the next call that is (here
        # a launch); not when the context is destroyed (the layer does not ask the driver then), nor
        # at exit (it asks the driver nothing then).
        out.append(run.copy())
        if name == "copy":
            out.append(run.launch("add_one", grid=3))
        if name == "copy-destroyed":
            out.append(run.api.cuCtxDestroy_v2(run.context))
    print("results", " ".join(map(str, out)), flush=True)


def checks(name, log, summary, results):
    lines = [line.removeprefix("[volta-forever] ") for line in log.splitlines()]
    faults = [i for i, line in enumerate(lines) if "can run nothing more" in line]
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
        expect("every call returned 0", results[:10] == ["0"] * 10)
        return found
    if name == "off":
        # VF_FAULT_REPORT=0: what the program gets is the driver's, nothing is kept or said, nothing
        # is waited for even with VF_FAULT_TRACE=1, and lookups of the synchronisations lead to the
        # driver.
        expect("no fault is said", not faults and "faulted" not in summary and "fault" not in log)
        expect("every call returned what the driver returned",
               results[:10] == ["0"] * 7 + ["700"] * 3)
        expect("a lookup of cuStreamSynchronize leads to the driver, of cuLaunchKernel to the layer",
               results[10:12] == ["cuStreamSynchronize:driver", "cuLaunchKernel:layer"])
        expect("nothing waited for but the program's one stream synchronisation", results[12] == "1")
        return found
    if name == "copy-destroyed":
        expect("nothing is said: the layer does not ask the driver before a context is destroyed",
               not faults and "faulted" not in summary and results == ["0", "0", "700", "0"])
        return found
    if name == "exit":
        expect("nothing is said: the layer asks the driver nothing at exit",
               not faults and "failed" not in summary and results == ["0", "0", "700"])
        return found
    if name == "sigexit":
        expect("the process ended in exit() from the signal handler, and nothing is said",
               results == ["0"] and not faults)
        return found
    if name == "nofault-graph":
        expect("no fault is said, and every call returned 0", not faults and results == ["0"] * 15)
        return found
    if name in ("nofault-release", "nofault-capture-exit"):
        expect("no fault is said", not faults)
        expect("the capture was not invalidated: every call returned 0, its end too",
               results == ["0"] * (15 if name == "nofault-release" else 13))
        return found
    if name == "exact-capture":
        expect("no fault is said", not faults)
        expect("not waited for during a capture, waited for after", results == ["0", "0", "0", "0", "0", "1"])
        return found
    if name in ("other-context-event", "no-context"):
        # The context is not known at the wait: said so, with the launches of every context, and no
        # context taken as reported; A's own synchronisation then names A's.
        expect("two reports: the first of a context the layer cannot tell, the second A's",
               len(faults) == 2 and "a CUDA context that the layer cannot tell" in lines[faults[0]]
               and "cuCtxSynchronize returned 700" in lines[faults[1]] and "cannot tell" not in lines[faults[1]])
        if len(faults) == 2:
            first = [line.strip().split(" ")[0] for line in lines[faults[0] + 1:faults[1]] if line.startswith("  ")]
            second = [line.strip().split(" ")[0] for line in lines[faults[1] + 1:] if line.startswith("  ")]
            expect("the first names the launches of both contexts, each with its context",
                   sorted(first) == ["add_one", "add_one", "write_null"] and
                   len(re.findall(r", context 0x[0-9a-f]+\)", "\n".join(lines[faults[0]:faults[1]]))) == 3)
            expect("the second names A's launch alone", second == ["write_null"])
        expect("no claim of a context 0", "context 0)" not in log and "context 0x0" not in log)
        return found
    expect("one fault is said, once", len(faults) == 1)
    if not faults:
        return found
    head = lines[faults[0]]
    if name == "candidates":
        expect("lookups of the synchronisations lead to the layer",
               results[10:12] == ["cuStreamSynchronize:layer", "cuLaunchKernel:layer"])
        expect("cuCtxSynchronize met 700", "cuCtxSynchronize returned 700 (CUDA_ERROR_ILLEGAL_ADDRESS)" in head)
        expect("not said to be a kernel's for certain: a copy or a memset may have failed instead",
               head.startswith("work on the GPU failed:") and "a copy, a memset" in head and "a kernel faulted on" not in log)
        expect("the three launches since the stream's synchronisation, the last first",
               "3 launches were made in this context since its last synchronisation" in head and
               names == ["add_one", "write_null", "add_one"])
        expect("their grids", re.search(r"add_one \(cuLaunchKernel, grid 5x1x1, block 32x1x1, the null stream\)",
                                        log) is not None and "grid 4x1x1" in log and "grid 2x1x1" not in log)
        expect("what the layer made of the image", "PTX for sm_75 retargeted by the layer" in log and
               re.search(r"the image given to the driver: \d+ bytes, XXH64 [0-9a-f]{16}", log) is not None)
        expect("the program's image without its digest at level 1",
               f"the program's image: cuModuleLoadData of {len(PTX)} bytes (VF_LOG_LEVEL=2 prints its XXH64" in log)
        expect("how to name it alone", "VF_FAULT_TRACE=1 synchronises every launch" in log)
        expect("the report at exit", "work on the GPU failed: cuCtxSynchronize returned 700 (CUDA_ERROR_ILLEGAL_ADDRESS); "
               "the candidates, the last launched first: add_one, write_null, add_one" in summary)
    elif name == "exact":
        expect("named alone, with what is not ruled out",
               head.startswith("the synchronisation right after the launch of write_null met the fault: the "
                               "synchronisation of the stream after cuLaunchKernel (VF_FAULT_TRACE=1) returned 700")
               and "(a copy, a memset, a host function) is not ruled out" in head and names == ["write_null"])
        expect("its launch returned the fault", results[5] == "700")
        expect("the report at exit", "work on the GPU failed: the synchronisation of the stream after cuLaunchKernel "
               "(VF_FAULT_TRACE=1) returned 700 (CUDA_ERROR_ILLEGAL_ADDRESS); the launch whose synchronisation met "
               "it: write_null (work not seen on its stream before it not ruled out)" in summary)
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
        expect("a graph launch", "1 launch was made in this context since its" in head)
        expect("its kernel nodes as captured, not as launches, with what the layer does not follow",
               re.search(r"a CUDA graph \(cuGraphLaunch, stream 0x[0-9a-f]+\): its kernels ran without passing through "
                         r"the layer; the layer saw 2 kernel nodes added to it, of 2 kernels \(it does not follow child "
                         r"graphs, conditional bodies", log) is not None and
               "    add_one (grid 2x1x1, block 32x1x1)" in log and "    write_null (grid 3x1x1, block 32x1x1)" in log)
    elif name == "graph-exact":
        expect("the graph named", head.startswith("the synchronisation right after the launch of a CUDA graph met "
                                                  "the fault") and
               "    write_null (grid 3x1x1" in log)
        expect("the graph's launch returned the fault", results[7] == "700")
    elif name == "deadlock":
        count = re.search(r"(\d+) launches were made", head)
        expect("the report ended while contexts were destroyed, and named the launches of the 120 threads",
               results == ["0", "700"] and count is not None and int(count.group(1)) >= 120 * 255)
    elif name == "other-context-stream":
        expect("the fault is A's, which its stream's launch tells: write_null alone, not B's launches",
               "cuStreamSynchronize returned 700" in head and names == ["write_null"])
    elif name == "ecc":
        expect("an error of the hardware, said so, not a kernel's fault",
               head.startswith("the GPU's memory reported an uncorrectable ECC error: cuCtxSynchronize returned 214 "
                               "(CUDA_ERROR_ECC_UNCORRECTABLE)") and "an error of the hardware, not of a kernel" in head
               and "faulted" not in log)
        expect("what ran then is named", names == ["ecc_fail", "add_one"])
        expect("the report at exit says it as such", summary.count("the GPU's memory reported an uncorrectable ECC "
                                                                   "error: cuCtxSynchronize returned 214") == 1)
    elif name == "wrapped":
        expect("the line and the report at exit say that earlier launches may be candidates",
               "(and maybe launches made before these, which the layer no longer keeps" in head and
               "(and maybe launches made before these, which the layer no longer keeps" in summary)
    elif name == "unseen":
        expect("no launch known, and nothing asserted of where the kernel was launched",
               "The layer kept no launch of this context that no synchronisation that succeeded covers" in head
               and "the kernel was launched where" not in log and "may be what failed" in head)
    elif name == "exact-done":
        expect("the last launch's wait met the fault and it returned it", results == ["0"] * 5 + ["700"])
        expect("not said to be the kernel: the launch not waited for is named with it",
               "2 launches were made" in head and names == ["add_one", "write_null"] and "grid 4x1x1" in log
               and "grid 1x1x1" in log and "grid 3x1x1" not in log)
        expect("the launch whose wait met it is said", "whose synchronisation right after it met the fault" in log)
    elif name == "exec-reuse":
        expect("the second graph was given the first one's handle", results[-1] == "same-handle")
        expect("the launch is named with the kernel nodes of the graph launched, not of the one at its handle since",
               "    add_one (grid 1x1x1" in log and "    write_null (grid 2x1x1" in log and "grid 3x1x1" not in log)
    elif name == "own-stream-threads":
        expect("the second thread's synchronisation succeeded, the main thread's met the fault",
               results == ["0", "0", "0", "0", "0", "700"])
        expect("the first thread's launch on its own default stream is named, the second's is not",
               names == ["write_null"] and "grid 1x1x1" in log and "grid 2x1x1" not in log)
    elif name == "unload-failed":
        expect("the unload met the fault", results[2:] == ["700", "700"] and "cuModuleUnload returned 700" in head)
        line = next((line for line in lines if "grid 1x1x1" in line), "")
        expect("the kernel of the module still loaded is named with its image",
               "PTX for sm_75 retargeted by the layer" in line and "unloaded since" not in line)
    elif name == "profile":
        expect("the profile's wait after write_null met it",
               "the synchronisation of the stream after a launch (VF_PROFILE) returned 700" in head)
        expect("write_null alone: the profile's wait after add_one covered it", names == ["write_null"])
        expect("the launch whose wait met it, named as such", "the synchronisation right after the launch of "
               "write_null met the fault" in head)
    elif name == "copy":
        expect("found at the next launch", "cuLaunchKernel returned 700" in head)
        expect("the copy met it first", results[2:] == ["700", "700"])
        expect("both launches", names == ["write_null", "add_one"])
    elif name == "sigexit-report":
        expect("the fault reported, and the process ended in exit() from the signal handler",
               results == ["0", "0", "700"] and "cuStreamSynchronize returned 700" in head)
        at_exit = "work on the GPU failed: cuStreamSynchronize returned 700 (CUDA_ERROR_ILLEGAL_ADDRESS); the candidates"
        expect("the report at exit, made by the handler's exit, has it (the log's and VF_SUMMARY's)",
               summary.count(at_exit) == 1 and sum(line.startswith(at_exit) for line in lines) == 1)
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
        cases = [("deadlock", {}), ("candidates", {}), ("nofault", {}), ("nofault-graph", {}),
                 ("nofault-release", {}), ("nofault-capture-exit", {}), ("exact", {"VF_FAULT_TRACE": "1"}),
                 ("streams", {}), ("contexts", {}), ("context-gone", {}),
                 ("other-context-stream", {}), ("other-context-event", {}), ("no-context", {}), ("own-stream", {}),
                 ("own-stream-threads", {"MOCK_DELAY": "1"}), ("threads", {}), ("ecc", {}), ("wrapped", {}),
                 ("unseen", {"VF_ADAPT": "0", "VF_FIT": "0"}),
                 ("graph", {}), ("graph-exact", {"VF_FAULT_TRACE": "1"}),
                 ("exec-reuse", {"MOCK_EXEC_FIXED": "1"}), ("exact-capture", {"VF_FAULT_TRACE": "1"}),
                 ("exact-done", {"VF_FAULT_TRACE": "1", "MOCK_DELAY": "1"}),
                 ("copy", {}), ("copy-destroyed", {}), ("exit", {}), ("unload-failed", {}), ("profile", {"VF_PROFILE": str(Path(tmp) / "profile")}),
                 ("sigexit", {"MOCK_SIGNAL": "1"}), ("sigexit-report", {"MOCK_SIGNAL": "1"}),
                 # Exact mode needs the exact value.
                 ("candidates", {"VF_FAULT_TRACE": "true"}),
                 # The report off: as without it, exact mode too; only the exact value turns it off.
                 ("off", {"VF_FAULT_REPORT": "0"}), ("off", {"VF_FAULT_REPORT": "0", "VF_FAULT_TRACE": "1"}),
                 ("candidates", {"VF_FAULT_REPORT": "false"})]
        for name, settings in cases:
            log, summary = Path(tmp) / f"{name}.log", Path(tmp) / f"{name}.summary"
            log.unlink(missing_ok=True)
            env = dict(os.environ, VF_REAL_LIBCUDA=mock, VF_PROXY_CACHE="0", VF_LOG=str(log),
                       VF_SUMMARY=str(summary), **settings)
            env.pop("VF_LOG_LEVEL", None)
            # (The cases of a signal end in exit(10) from its handler, and at once.)
            status, limit = (10, 20) if name.startswith("sigexit") else (0, 120)
            try:
                done = subprocess.run([sys.executable, __file__, proxy, name], env=env, capture_output=True,
                                      text=True, timeout=limit)
            except subprocess.TimeoutExpired as hung:
                text_of = lambda s: s.decode(errors="replace") if isinstance(s, bytes) else (s or "")
                done = subprocess.CompletedProcess(hung.cmd, -9, text_of(hung.stdout), text_of(hung.stderr))
                print(f"  {name}: still running after {limit} s, stopped")
            results = next((line.split()[1:] for line in done.stdout.splitlines() if line.startswith("results")), [])
            text = log.read_text() if log.exists() else ""
            summaries = list(Path(tmp).glob(f"{name}.summary.*"))
            report = summaries[0].read_text() if summaries else ""
            for s in summaries:
                s.unlink()
            label = name + "".join(f" {k}={v}" for k, v in settings.items())
            outcome = (checks(name, text, report, results) if done.returncode == status
                       else [(f"the case ran (status {done.returncode}, {status} expected)", False)])
            for what, ok in outcome:
                print(f"  {'ok    ' if ok else 'WRONG '} {label}: {what}")
                failed += not ok
            if any(not ok for _, ok in outcome):
                print(done.stdout + done.stderr + text)
    print(f"{failed} wrong" if failed else "all checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
