"""A kernel that the layer refused, of which the program holds a handle that the layer did not give,
is not launched, recorded in a graph or asked about: the call returns 218 and the driver is not
called. Without a GPU, against a stand-in for the driver (guard_mock_driver.c), which the test also
calls directly for the handles: as a program reaches the driver through a private interface (cuFFT
takes the function of each module it loads from an entry of a table of cuGetExportTable).

Run with python3 tests/proxy/launch_guard.py path/to/libcuda.so.1. Each case runs in a process of
its own, with deferred libraries and without, and with the second translation at launch off.
"""
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REFUSED = 218
PTX = (b".version 8.8\n.target sm_75\n.address_size 64\n"
       b".visible .entry bad()\n{\n barrier.cluster.arrive;\n ret;\n}\n"
       b".visible .entry good()\n{\n ret;\n}\n")
# No kernel refused, and one whose only entry is refused.
GOOD = PTX.replace(b"bad()\n{\n barrier.cluster.arrive;\n ret;\n}\n.visible .entry ", b"")
BAD = PTX.replace(b".visible .entry good()\n{\n ret;\n}\n", b"")
assert b"bad" not in GOOD and b"good" not in BAD


class LaunchConfig(c.Structure):
    _fields_ = [("grid", c.c_uint * 3), ("block", c.c_uint * 3), ("shared", c.c_uint), ("stream", c.c_void_p),
                ("attributes", c.c_void_p), ("count", c.c_uint)]


class KernelNode(c.Structure):  # CUDA_KERNEL_NODE_PARAMS_v2 (v1: the first 7 fields)
    _fields_ = [("function", c.c_void_p), ("grid", c.c_uint * 3), ("block", c.c_uint * 3), ("shared", c.c_uint),
                ("params", c.c_void_p), ("extra", c.c_void_p), ("kernel", c.c_void_p), ("context", c.c_void_p)]


class GraphNode(c.Structure):  # CUgraphNodeParams
    _fields_ = [("kind", c.c_int), ("reserved", c.c_int * 3), ("kernel", KernelNode), ("pad", c.c_byte * 160),
                ("reserved2", c.c_longlong)]


class LaunchParams(c.Structure):  # CUDA_LAUNCH_PARAMS
    _fields_ = [("function", c.c_void_p), ("grid", c.c_uint * 3), ("block", c.c_uint * 3), ("shared", c.c_uint),
                ("stream", c.c_void_p), ("params", c.c_void_p)]


assert c.sizeof(KernelNode) == 72 and KernelNode.kernel.offset == 56 and c.sizeof(LaunchParams) == 56
assert GraphNode.kernel.offset == 16 and c.sizeof(GraphNode) == 256


class Run:
    def __init__(self, proxy, mock):
        self.api = c.CDLL(proxy)
        self.driver = c.CDLL(mock)
        self.context = c.c_void_p()
        assert self.api.cuInit(0) == 0
        assert self.api.cuCtxCreate_v2(c.byref(self.context), 0, 0) == 0

    def count(self, name):
        return c.c_uint.in_dll(self.driver, name).value

    def set(self, name, value):
        c.c_size_t.in_dll(self.driver, name).value = value

    def load(self, image, library=False):
        handle = c.c_void_p()
        if library:
            assert self.api.cuLibraryLoadData(c.byref(handle), image, None, None, 0, None, None, 0) == 0
        else:
            assert self.api.cuModuleLoadData(c.byref(handle), image) == 0
        return handle

    def private(self, module, name):
        """The function `name` of `module`, from the driver: not through the layer."""
        function = c.c_void_p()
        assert self.driver.cuModuleGetFunction(c.byref(function), module, name) == 0, name
        return function

    def calls(self, handle):
        """Every call of the layer that launches `handle`, records it in a graph or asks about it, with
        the stand-in's counter it changes when the driver is called."""
        api, h = self.api, c.c_void_p(handle)
        config = LaunchConfig((1, 1, 1), (1, 1, 1), 0, None, None, 0)
        value, size, offset = c.c_int(), c.c_size_t(), c.c_size_t()
        node, v1, v2 = c.c_void_p(), KernelNode(h), KernelNode(h)
        graph = c.c_void_p(0x600000)
        any_node = GraphNode(0, (0, 0, 0), KernelNode(h))
        multi = (LaunchParams * 2)(LaunchParams(h), LaunchParams(h))
        pointer = c.c_void_p()
        assert api.cuGetProcAddress(b"cuLaunchKernel", c.byref(pointer), 12000, 0) == 0
        looked_up = c.CFUNCTYPE(c.c_int, *[c.c_void_p] + [c.c_uint] * 7 + [c.c_void_p] * 3)(pointer.value)
        grid = (1, 1, 1, 1, 1, 1, 0, None, None, None)
        return [
            ("cuLaunchKernel", "launched", lambda: api.cuLaunchKernel(h, *grid)),
            ("cuLaunchKernel by cuGetProcAddress", "launched", lambda: looked_up(h, *grid)),
            ("cuLaunchKernel_ptsz", "launched", lambda: api.cuLaunchKernel_ptsz(h, *grid)),
            ("cuLaunchKernelEx", "launched", lambda: api.cuLaunchKernelEx(c.byref(config), h, None, None)),
            ("cuLaunchKernelEx_ptsz", "launched", lambda: api.cuLaunchKernelEx_ptsz(c.byref(config), h, None, None)),
            ("cuLaunch", "launched", lambda: api.cuLaunch(h)),
            ("cuLaunchGrid", "launched", lambda: api.cuLaunchGrid(h, 1, 1)),
            ("cuLaunchGridAsync", "launched", lambda: api.cuLaunchGridAsync(h, 1, 1, None)),
            ("cuLaunchCooperativeKernel", "launched", lambda: api.cuLaunchCooperativeKernel(h, *grid[:9])),
            ("cuLaunchCooperativeKernel_ptsz", "launched", lambda: api.cuLaunchCooperativeKernel_ptsz(h, *grid[:9])),
            ("cuLaunchCooperativeKernelMultiDevice", "launched",
             lambda: api.cuLaunchCooperativeKernelMultiDevice(multi, 2, 0)),
            ("cuGraphAddKernelNode", "nodes", lambda: api.cuGraphAddKernelNode(c.byref(node), graph, None, 0, c.byref(v1))),
            ("cuGraphAddKernelNode_v2", "nodes",
             lambda: api.cuGraphAddKernelNode_v2(c.byref(node), graph, None, 0, c.byref(v2))),
            ("cuGraphKernelNodeSetParams", "nodes", lambda: api.cuGraphKernelNodeSetParams(node, c.byref(v1))),
            ("cuGraphKernelNodeSetParams_v2", "nodes", lambda: api.cuGraphKernelNodeSetParams_v2(node, c.byref(v2))),
            ("cuGraphExecKernelNodeSetParams", "nodes",
             lambda: api.cuGraphExecKernelNodeSetParams(graph, node, c.byref(v1))),
            ("cuGraphExecKernelNodeSetParams_v2", "nodes",
             lambda: api.cuGraphExecKernelNodeSetParams_v2(graph, node, c.byref(v2))),
            ("cuGraphAddNode", "nodes", lambda: api.cuGraphAddNode(c.byref(node), graph, None, 0, c.byref(any_node))),
            ("cuGraphAddNode_v2", "nodes",
             lambda: api.cuGraphAddNode_v2(c.byref(node), graph, None, None, 0, c.byref(any_node))),
            ("cuGraphNodeSetParams", "nodes", lambda: api.cuGraphNodeSetParams(node, c.byref(any_node))),
            ("cuGraphExecNodeSetParams", "nodes", lambda: api.cuGraphExecNodeSetParams(graph, node, c.byref(any_node))),
            ("cuFuncGetAttribute", "asked", lambda: api.cuFuncGetAttribute(c.byref(value), 0, h)),
            ("cuFuncSetAttribute", "asked", lambda: api.cuFuncSetAttribute(h, 8, 0)),
            ("cuFuncSetCacheConfig", "asked", lambda: api.cuFuncSetCacheConfig(h, 0)),
            ("cuFuncSetSharedMemConfig", "asked", lambda: api.cuFuncSetSharedMemConfig(h, 0)),
            ("cuFuncGetParamInfo", "asked", lambda: api.cuFuncGetParamInfo(h, 0, c.byref(offset), c.byref(size))),
            ("cuFuncLoad", "asked", lambda: api.cuFuncLoad(h)),
            ("cuOccupancyMaxActiveBlocksPerMultiprocessor", "asked",
             lambda: api.cuOccupancyMaxActiveBlocksPerMultiprocessor(c.byref(value), h, 32, c.c_size_t(0))),
            ("cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", "asked",
             lambda: api.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(c.byref(value), h, 32, c.c_size_t(0), 0)),
            ("cuOccupancyMaxPotentialBlockSize", "asked",
             lambda: api.cuOccupancyMaxPotentialBlockSize(c.byref(value), c.byref(value), h, None, c.c_size_t(0), 0)),
            ("cuOccupancyMaxPotentialBlockSizeWithFlags", "asked",
             lambda: api.cuOccupancyMaxPotentialBlockSizeWithFlags(c.byref(value), c.byref(value), h, None,
                                                                   c.c_size_t(0), 0, 0)),
            ("cuOccupancyAvailableDynamicSMemPerBlock", "asked",
             lambda: api.cuOccupancyAvailableDynamicSMemPerBlock(c.byref(size), h, 1, 32)),
            ("cuOccupancyMaxPotentialClusterSize", "asked",
             lambda: api.cuOccupancyMaxPotentialClusterSize(c.byref(value), h, c.byref(config))),
            ("cuOccupancyMaxActiveClusters", "asked",
             lambda: api.cuOccupancyMaxActiveClusters(c.byref(value), h, c.byref(config))),
        ]

    def kernel_calls(self, kernel, device=0):
        """The calls that take a library's kernel (CUkernel), those that name a device with `device`."""
        api, k = self.api, c.c_void_p(kernel)
        value, size, offset, node = c.c_int(), c.c_size_t(), c.c_size_t(), c.c_void_p()
        by_kernel = KernelNode(None, kernel=k)
        any_node = GraphNode(0, (0, 0, 0), KernelNode(None, kernel=k))
        grid = (1, 1, 1, 1, 1, 1, 0, None, None, None)
        return [
            ("cuLaunchKernel with a kernel", "launched", lambda: api.cuLaunchKernel(k, *grid)),
            ("cuGraphAddKernelNode_v2 with a kernel", "nodes",
             lambda: api.cuGraphAddKernelNode_v2(c.byref(node), c.c_void_p(0x600000), None, 0, c.byref(by_kernel))),
            ("cuGraphAddNode with a kernel", "nodes",
             lambda: api.cuGraphAddNode(c.byref(node), c.c_void_p(0x600000), None, 0, c.byref(any_node))),
            ("cuKernelGetAttribute", "asked", lambda: api.cuKernelGetAttribute(c.byref(value), 0, k, device)),
            ("cuKernelSetAttribute", "asked", lambda: api.cuKernelSetAttribute(8, 0, k, device)),
            ("cuKernelSetCacheConfig", "asked", lambda: api.cuKernelSetCacheConfig(k, 0, device)),
            ("cuKernelGetParamInfo", "asked", lambda: api.cuKernelGetParamInfo(k, 0, c.byref(offset), c.byref(size))),
        ]

    def refused(self, handle, calls=None, what=""):
        """Every call with `handle` returns 218, and the driver is not called."""
        for name, counter, call in calls or self.calls(handle):
            before = self.count(counter)
            got = call()
            assert got == REFUSED and self.count(counter) == before, f"{name} {what}: {got}, {counter} {before} -> {self.count(counter)}"

    def allowed(self, handle, calls=None, what=""):
        """Every call with `handle` reaches the driver."""
        for name, counter, call in calls or self.calls(handle):
            before = self.count(counter)
            got = call()
            assert got == 0 and self.count(counter) > before, f"{name} {what}: {got}, {counter} {before} -> {self.count(counter)}"


def module_case(run):
    """A module: the refused kernel by the driver directly and by a private table, through every call."""
    module = run.load(PTX)
    function = c.c_void_p(0xDEADBEEF)
    assert run.api.cuModuleGetFunction(c.byref(function), module, b"bad") == REFUSED and function.value is None
    bad, good = run.private(module, b"bad"), run.private(module, b"good")
    # The private interface: a table of cuGetExportTable, forwarded untouched by the layer.
    table = c.POINTER(c.c_void_p)()
    assert run.api.cuGetExportTable(c.byref(table), b"\0" * 16) == 0
    first = c.CFUNCTYPE(c.c_int, c.c_void_p, c.POINTER(c.c_void_p))(table[1])
    from_table = c.c_void_p()
    assert first(module, c.byref(from_table)) == 0 and from_table.value == bad.value
    run.refused(bad.value, what="(a refused kernel)")
    run.allowed(good.value, what="(a kernel of the same module)")
    run.refused(bad.value, what="(again)")
    # What was found allowed is not asked again; what is refused is.
    identified = run.count("identified")
    run.allowed(good.value, what="(a third time)")
    assert run.count("identified") == identified, (identified, run.count("identified"))
    # The functions given by an enumeration of the driver itself.
    count = c.c_uint()
    assert run.driver.cuModuleGetFunctionCount(c.byref(count), module) == 0 and count.value == 2
    listed = (c.c_void_p * 2)()
    assert run.driver.cuModuleEnumerateFunctions(listed, 2, module) == 0
    assert sorted(listed) == sorted([bad.value, good.value])
    assert run.api.cuModuleUnload(module) == 0
    # The driver gives the handle of the refused kernel to a kernel of a module with nothing refused,
    # and the handle of the kernel found allowed to a refused one.
    run.set("next_function", bad.value)
    module = run.load(GOOD)
    assert run.private(module, b"good").value == bad.value
    run.allowed(bad.value, what="(a handle given again to a kernel that runs)")
    assert run.api.cuModuleUnload(module) == 0
    run.set("next_function", good.value)
    module = run.load(BAD)
    assert run.private(module, b"bad").value == good.value
    run.refused(good.value, what="(a handle given again to a refused kernel)")
    assert run.api.cuModuleUnload(module) == 0


def fast_case(run):
    """While no module has a refused kernel, the driver is not asked what a handle is."""
    module = run.load(GOOD)
    good = run.private(module, b"good")
    for _ in range(3):
        run.allowed(good.value)
    assert run.count("identified") == 0, run.count("identified")
    other = run.load(PTX)
    run.allowed(good.value)
    once = run.count("identified")
    assert once > 0
    for _ in range(3):
        run.allowed(good.value)
    assert run.count("identified") == once, (once, run.count("identified"))
    run.refused(run.private(other, b"bad").value)
    # Unloaded, the module with a refused kernel leaves nothing that is asked about.
    assert run.api.cuModuleUnload(other) == 0
    identified = run.count("identified")
    for _ in range(3):
        run.allowed(good.value)
    assert run.count("identified") == identified, (identified, run.count("identified"))
    assert run.api.cuModuleUnload(module) == 0


def library_case(run, lazy):
    """A library: its kernel and its function in a context, taken from the driver directly."""
    library = run.load(PTX, library=True)
    kernel = c.c_void_p(0xDEADBEEF)
    assert run.api.cuLibraryGetKernel(c.byref(kernel), library, b"bad") == REFUSED and kernel.value is None
    assert run.api.cuLibraryGetKernel(c.byref(kernel), library, b"good") == 0
    good_kernel = kernel.value
    run.allowed(good_kernel, run.kernel_calls(good_kernel), "(a kernel that runs)")
    if not lazy:
        bad_kernel = c.c_void_p()
        assert run.driver.cuLibraryGetKernel(c.byref(bad_kernel), library, b"bad") == 0
        run.refused(bad_kernel.value, run.kernel_calls(bad_kernel.value), "(a refused kernel)")
        # Its function, from the layer: not given; from the driver, in a context where the program
        # never asked for the library's module.
        function = c.c_void_p(0xDEADBEEF)
        assert run.api.cuKernelGetFunction(c.byref(function), bad_kernel) == REFUSED and function.value is None
        assert run.driver.cuKernelGetFunction(c.byref(function), bad_kernel) == 0
        run.refused(function.value, what="(the function of a refused kernel in a context)")
        other = c.c_void_p()
        assert run.api.cuCtxCreate_v2(c.byref(other), 0, 0) == 0
        assert run.driver.cuKernelGetFunction(c.byref(function), bad_kernel) == 0
        run.refused(function.value, what="(in a second context)")
        good_function = c.c_void_p()
        assert run.driver.cuKernelGetFunction(c.byref(good_function), c.c_void_p(good_kernel)) == 0
        run.allowed(good_function.value, what="(a function that runs, in a second context)")
        # A runtime may ask for the function of a kernel before each launch: once the layer knows the
        # module of the library in the context, that asks the driver nothing more at a launch.
        for _ in range(3):
            assert run.api.cuKernelGetFunction(c.byref(good_function), c.c_void_p(good_kernel)) == 0
            identified = run.count("identified")
            run.allowed(good_function.value, what="(after cuKernelGetFunction)")
            once = run.count("identified") - identified
        assert once == 0, f"the driver was asked {once} times what the handle is"
        assert run.api.cuCtxDestroy_v2(other) == 0
        # On a later GPU, the library's kernels run from its own code: the refused one, and its
        # function there, are launched; back on the V100, the kernel is refused again.
        later = c.c_void_p()
        assert run.api.cuCtxCreate_v2(c.byref(later), 0, 1) == 0
        assert run.driver.cuKernelGetFunction(c.byref(function), bad_kernel) == 0
        run.allowed(function.value, what="(the function of a refused kernel on a later GPU)")
        run.allowed(bad_kernel.value, run.kernel_calls(bad_kernel.value, 1), "(a refused kernel on a later GPU)")
        assert run.api.cuCtxSetCurrent(run.context) == 0
        run.refused(bad_kernel.value, run.kernel_calls(bad_kernel.value), "(back on the V100)")
        # A kernel node that names the kernel and a context runs in that context, whichever is current.
        node = c.c_void_p()
        for current, context, error in ((run.context, later, 0), (later, run.context, REFUSED)):
            assert run.api.cuCtxSetCurrent(current) == 0
            params = KernelNode(None, kernel=bad_kernel, context=context)
            nodes = run.count("nodes")
            got = run.api.cuGraphAddKernelNode_v2(c.byref(node), c.c_void_p(0x600000), None, 0, c.byref(params))
            assert got == error and run.count("nodes") == nodes + (error == 0), (
                f"a node of the refused kernel in {'the later GPU' if error == 0 else 'the V100'}'s context: {got}")
        assert run.api.cuCtxDestroy_v2(later) == 0
        assert run.api.cuCtxSetCurrent(run.context) == 0
    # The module of the library (the whole module of a deferred library), and its functions taken
    # from the driver.
    module = c.c_void_p()
    assert run.api.cuLibraryGetModule(c.byref(module), library) == 0
    run.refused(run.private(module, b"bad").value, what="(of the library's module)")
    run.allowed(run.private(module, b"good").value, what="(of the library's module)")
    if lazy:
        # A deferred library gives every GPU the PTX adapted for the V100, in which the kernel traps:
        # on a later GPU too, it is refused.
        later = c.c_void_p()
        assert run.api.cuCtxCreate_v2(c.byref(later), 0, 1) == 0
        assert run.api.cuLibraryGetModule(c.byref(module), library) == 0
        run.refused(run.private(module, b"bad").value, what="(of a deferred library's module on a later GPU)")
        run.allowed(run.private(module, b"good").value, what="(of a deferred library's module on a later GPU)")
        assert run.api.cuCtxDestroy_v2(later) == 0
        assert run.api.cuCtxSetCurrent(run.context) == 0
    assert run.api.cuLibraryUnload(library) == 0


def context_case(run):
    """A module that goes with its context leaves nothing refused under the handles of its functions."""
    module = run.load(PTX)
    bad = run.private(module, b"bad")
    run.refused(bad.value)
    assert run.api.cuCtxDestroy_v2(run.context) == 0
    assert run.api.cuCtxCreate_v2(c.byref(run.context), 0, 0) == 0
    run.set("next_function", bad.value)
    module = run.load(GOOD)
    assert run.private(module, b"good").value == bad.value
    run.allowed(bad.value, what="(a handle of a context destroyed, given again)")
    # The other way: a function of a library in a context, found allowed; the context goes (destroyed,
    # or a green context destroyed), and the driver gives the handle to the refused kernel of the
    # library in the module that it makes for the library in a new context. The layer sees neither
    # module: only the end of the context tells it that the handle may be given again.
    library = run.load(PTX, library=True)
    if os.environ.get("VF_LAZY") != "0":
        return
    good_kernel, bad_kernel = c.c_void_p(), c.c_void_p()
    assert run.driver.cuLibraryGetKernel(c.byref(good_kernel), library, b"good") == 0
    assert run.driver.cuLibraryGetKernel(c.byref(bad_kernel), library, b"bad") == 0
    # (A context in which nothing was loaded through the layer.)
    assert run.api.cuCtxCreate_v2(c.byref(run.context), 0, 0) == 0
    for destroy in (run.api.cuCtxDestroy_v2, run.api.cuGreenCtxDestroy):
        function = c.c_void_p()
        assert run.driver.cuKernelGetFunction(c.byref(function), good_kernel) == 0
        good = function.value
        run.allowed(good, what="(a library's function in a context)")
        assert destroy(run.context) == 0
        assert run.api.cuCtxCreate_v2(c.byref(run.context), 0, 0) == 0
        run.set("next_function", good)
        assert run.driver.cuKernelGetFunction(c.byref(function), bad_kernel) == 0 and function.value == good
        run.refused(good, what=f"(given again to a refused kernel after {destroy.__name__})")
    assert run.api.cuLibraryUnload(library) == 0


def log_case(run, log):
    """The refusal is said once, at the default level, with its reason."""
    module = run.load(PTX)
    bad = run.private(module, b"bad")
    run.refused(bad.value)
    run.refused(bad.value)
    text = Path(log).read_text()
    lines = [line for line in text.splitlines() if "holds a handle to it, which a lookup by name refuses" in line]
    assert len(lines) == 1, text
    assert "kernel bad cannot run on this GPU: barrier.cluster.arrive: no Volta sequence" in lines[0], lines[0]
    assert "cuLaunchKernel with it is refused (error 218)" in lines[0], lines[0]


CASES = {
    "module": module_case,
    "fast": fast_case,
    "library": lambda run: library_case(run, os.environ.get("VF_LAZY") != "0"),
    "contexts": context_case,
    "log": lambda run: log_case(run, os.environ["VF_LOG"]),
}

if __name__ == "__main__":
    if len(sys.argv) == 4:
        CASES[sys.argv[3]](Run(sys.argv[1], sys.argv[2]))
        sys.exit(0)
    proxy = sys.argv[1]
    with tempfile.TemporaryDirectory() as tmp:
        mock = str(Path(tmp) / "guard.so")
        subprocess.run(["cc", "-shared", "-fPIC", "-Wl,-Bsymbolic", "-o", mock,
                        Path(__file__).with_name("guard_mock_driver.c")], check=True)
        failed = 0
        for case in CASES:
            for settings in ({}, {"VF_LAZY": "0"}, {"VF_FIT": "0"}):
                env = dict(os.environ, VF_REAL_LIBCUDA=mock, VF_PROXY_CACHE="0", VF_LOG=str(Path(tmp) / f"{case}.log"),
                           VF_LOG_LEVEL="1", **settings)
                Path(env["VF_LOG"]).unlink(missing_ok=True)
                done = subprocess.run([sys.executable, __file__, proxy, mock, case], env=env)
                label = " ".join(f"{k}={v}" for k, v in settings.items())
                print(f"{'ok  ' if done.returncode == 0 else 'FAIL'}  {case} {label}".rstrip())
                failed |= done.returncode != 0
        print("FAIL launch guard" if failed else "PASS launch guard: refused kernels are not launched, recorded or asked about")
        sys.exit(1 if failed else 0)
