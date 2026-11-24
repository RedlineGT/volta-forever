"""A kernel without code (its body in the image is a trap and nothing else, as a kernel built for other
architectures only is emitted) is given at a lookup and may be asked about, as NVIDIA's driver does;
a call that would run it (a launch of any kind, a kernel node of a graph) returns 218 without the
driver being called, with one line that names the kernel and its image. Launched, its trap would end
the context. Without a GPU, against the stand-in for the driver of launch_guard.py
(guard_mock_driver.c), with the calls of that test.

Run with python3 tests/proxy/nocode_launch.py path/to/libcuda.so.1. Each case runs in a process of its
own, with deferred libraries and without.
"""
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent))
from launch_guard import REFUSED, Run  # noqa: E402

# no_code is what nvcc 13 writes for an instantiation built without code for the target. trap_if
# traps behind a guard, after_store after a store: code, as `good` is. empty does nothing.
PTX = (b".version 8.8\n.target sm_75\n.address_size 64\n"
       b".visible .entry no_code(\n.param .u64 p\n)\n{\n\n\n\t// begin inline asm\n\ttrap;\n\t// end inline asm\n\tret;\n\n}\n"
       b".visible .entry good()\n{\n ret_label:\n mov.u32 %r1, 1;\n ret;\n}\n"
       b".visible .entry trap_if()\n{\n .reg .pred %p<2>;\n setp.eq.u32 %p1, 1, 0;\n @%p1 trap;\n ret;\n}\n"
       b".visible .entry after_store()\n{\n .reg .b64 %rd<2>;\n st.global.u32 [%rd1], 1;\n trap;\n ret;\n}\n"
       b".visible .entry empty()\n{\n ret;\n}\n")
LINE = "kernel no_code has no code in its image ("


def split(calls):
    """The calls that run the kernel, and those that ask about it or set something of it."""
    run = [call for call in calls if call[1] in ("launched", "nodes")]
    ask = [call for call in calls if call[1] == "asked"]
    assert run and ask and len(run) + len(ask) == len(calls)
    return run, ask


def lookup_is_silent(log):
    text = Path(log).read_text() if Path(log).exists() else ""
    assert "no_code" not in text, text


def module_case(run, log):
    module = run.load(PTX)
    function = c.c_void_p()
    # Given at the lookup, and nothing said.
    assert run.api.cuModuleGetFunction(c.byref(function), module, b"no_code") == 0 and function.value
    lookup_is_silent(log)
    launches, questions = split(run.calls(function.value))
    run.allowed(function.value, questions, "(a question about a kernel without code)")
    run.refused(function.value, launches, "(a kernel without code)")
    # The same handle, from the driver directly (a private interface), is the same kernel.
    private = run.private(module, b"no_code").value
    run.refused(private, split(run.calls(private))[0], "(from the driver)")
    # Kernels that are code, and the empty one, run.
    for name in (b"good", b"trap_if", b"after_store", b"empty"):
        other = c.c_void_p()
        assert run.api.cuModuleGetFunction(c.byref(other), module, name) == 0
        run.allowed(other.value, what=f"({name.decode()})")
    # Asked again after a launch was refused: still answered.
    run.allowed(function.value, questions, "(a question after a refused launch)")
    run.refused(function.value, launches, "(again)")
    assert run.api.cuModuleUnload(module) == 0
    # Its handle given again to a kernel that runs: allowed.
    run.set("next_function", function.value)
    module = run.load(b".version 8.8\n.target sm_75\n.address_size 64\n.visible .entry good()\n{\n mov.u32 %r1, 1;\n ret;\n}\n")
    assert run.private(module, b"good").value == function.value
    run.allowed(function.value, what="(a handle given again to a kernel with code)")
    assert run.api.cuModuleUnload(module) == 0


def library_case(run, log):
    library = run.load(PTX, library=True)
    kernel = c.c_void_p()
    assert run.api.cuLibraryGetKernel(c.byref(kernel), library, b"no_code") == 0 and kernel.value
    lookup_is_silent(log)
    launches, questions = split(run.kernel_calls(kernel.value))
    run.allowed(kernel.value, questions, "(a question about a library's kernel without code)")
    run.refused(kernel.value, launches, "(a library's kernel without code)")
    # Its function in the context: given, and the same.
    function = c.c_void_p()
    assert run.api.cuKernelGetFunction(c.byref(function), kernel) == 0 and function.value
    launches, questions = split(run.calls(function.value))
    run.allowed(function.value, questions, "(a question about its function)")
    run.refused(function.value, launches, "(its function)")
    good = c.c_void_p()
    assert run.api.cuLibraryGetKernel(c.byref(good), library, b"good") == 0
    run.allowed(good.value, run.kernel_calls(good.value), "(a library's kernel with code)")
    # The module of the library, and the kernel's function there from the driver.
    module = c.c_void_p()
    assert run.api.cuLibraryGetModule(c.byref(module), library) == 0
    of_module = run.private(module, b"no_code").value
    launches, _ = split(run.calls(of_module))
    run.refused(of_module, launches, "(of the library's module)")
    run.allowed(run.private(module, b"good").value, what="(of the library's module)")
    assert run.api.cuLibraryUnload(library) == 0


def log_case(run, log):
    """Said once for the kernel, at the default level, with its image by size and digest."""
    module = run.load(PTX)
    function = c.c_void_p()
    assert run.api.cuModuleGetFunction(c.byref(function), module, b"no_code") == 0
    launches, _ = split(run.calls(function.value))
    run.refused(function.value, launches)
    run.refused(function.value, launches)
    text = Path(log).read_text()
    lines = [line for line in text.splitlines() if LINE in line]
    assert len(lines) == 1, text
    assert f"its image ({len(PTX) + 1} bytes, XXH64 " in lines[0] or f"its image ({len(PTX)} bytes, XXH64 " in lines[0], lines[0]
    assert lines[0].endswith("its body is a trap and nothing else; cuLaunchKernel with it is refused (error 218)"), lines[0]
    assert "cannot run on this GPU" not in text, text


CASES = {
    "module": module_case,
    "library": library_case,
    "log": log_case,
}

if __name__ == "__main__":
    if len(sys.argv) == 4:
        CASES[sys.argv[3]](Run(sys.argv[1], sys.argv[2]), os.environ["VF_LOG"])
        sys.exit(0)
    proxy = sys.argv[1]
    with tempfile.TemporaryDirectory() as tmp:
        mock = str(Path(tmp) / "guard.so")
        subprocess.run(["cc", "-shared", "-fPIC", "-Wl,-Bsymbolic", "-o", mock,
                        Path(__file__).with_name("guard_mock_driver.c")], check=True)
        failed = 0
        for case in CASES:
            for settings in ({}, {"VF_LAZY": "0"}):
                env = dict(os.environ, VF_REAL_LIBCUDA=mock, VF_PROXY_CACHE="0", VF_LOG=str(Path(tmp) / f"{case}.log"),
                           VF_LOG_LEVEL="1", **settings)
                Path(env["VF_LOG"]).unlink(missing_ok=True)
                done = subprocess.run([sys.executable, __file__, proxy, mock, case], env=env)
                label = " ".join(f"{k}={v}" for k, v in settings.items())
                print(f"{'ok  ' if done.returncode == 0 else 'FAIL'}  {case} {label}".rstrip())
                failed |= done.returncode != 0
        print("FAIL kernels without code" if failed else
              "PASS kernels without code: given at a lookup, asked about, never launched or recorded")
        sys.exit(1 if failed else 0)
