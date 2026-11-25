"""A kernel that only the image's machine code for the capability told has is given from that code
in a module or library of the layer's own (crates/vf-proxy/src/told.rs): what holds of that, where
a GPU is not needed to see it. Against the stand-in for the driver of launch_guard.py
(guard_mock_driver.c), which has the kernels of the machine code it is given, with the image of
tests/kernels/k_told.cu and the program told 8.0:

  contexts   a module of one context, asked while another is current: not given (the layer would
             load in the other), nothing loaded; given in its own, from a module loaded there
  library    the same for the module that the driver gives for a library in a context; the
             library's kernel is given in any
  race       the program unloads its module, from another thread, while the layer loads the code
             for it: not given, and what was loaded is unloaded
  unload     what was loaded for a module or library is unloaded with it
  restore    an unload that the driver refuses leaves the kernel given, and what was loaded
  off        with VF_TOLD_CODE=0 nothing is loaded and the lookup fails
  linked     an image built with -rdc (relocatable cubins, a call into another unit), loaded or
             linked: the kernel that only its sm_80 code has is not given, and nothing is loaded

Run with python3 tests/proxy/told.py path/to/libcuda.so.1. Each case runs in a process of its own.
"""
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

sys.path.insert(0, str(Path(__file__).parent))
from launch_guard import Run  # noqa: E402

KERNELS = Path(__file__).resolve().parents[1] / "kernels"
IMAGE = (KERNELS / "k_told.135.sm75-sm80.fatbin").read_bytes()
RELOCATABLE = (KERNELS / "k_told_rdc.135.sm75-sm80.fatbin").read_bytes()
NAME = b"k_bf16_sum"
GIVEN = "is given from the machine code for sm_80"
ANOTHER = "its module is of another context than the current one"
NOT_FOUND = 500


def log():
    path = Path(os.environ["VF_LOG"])
    return path.read_text() if path.exists() else ""


def function(run, module, name=NAME):
    out = c.c_void_p()
    return run.api.cuModuleGetFunction(c.byref(out), module, name), out.value


def kernel(run, library, name=NAME):
    out = c.c_void_p()
    return run.api.cuLibraryGetKernel(c.byref(out), library, name), out.value


def context_of(run, handle):
    """The context in which the driver loaded the module of the function `handle`."""
    module = c.c_void_p()
    assert run.driver.cuFuncGetModule(c.byref(module), c.c_void_p(handle)) == 0
    run.driver.context_of_object.restype = c.c_size_t
    return module.value, run.driver.context_of_object(module)


def other_context(run):
    other = c.c_void_p()
    assert run.api.cuCtxCreate_v2(c.byref(other), 0, 0) == 0 and other.value != run.context.value
    return other


def contexts_case(run):
    module = run.load(IMAGE)
    loads = run.count("loads")
    other = other_context(run)  # (now the current one)
    result, given = function(run, module)
    assert result == NOT_FOUND and not given, (result, given)
    assert run.count("loads") == loads and ANOTHER in log() and GIVEN not in log(), log()
    # In its own context: given, from a module that the driver loaded there.
    assert run.api.cuCtxSetCurrent(run.context) == 0
    result, given = function(run, module)
    assert result == 0 and given and run.count("loads") == loads + 1 and GIVEN in log(), (result, log())
    second, context = context_of(run, given)
    assert second != module.value and context == run.context.value, (second, context)
    # In the other again: not given there, the one above or another (nothing more is loaded).
    assert run.api.cuCtxSetCurrent(other) == 0
    for name in (NAME, b"k_divides"):
        result, given = function(run, module, name)
        assert result == NOT_FOUND and not given and run.count("loads") == loads + 1, (name, result)


def library_case(run):
    library = run.load(IMAGE, library=True)
    first = c.c_void_p()
    assert run.api.cuLibraryGetModule(c.byref(first), library) == 0
    other = other_context(run)
    loads = run.count("loads")
    # The module that the driver gave for the library in the first context, asked in the other.
    result, given = function(run, first)
    assert result == NOT_FOUND and not given and run.count("loads") == loads and ANOTHER in log(), (result, log())
    # The library's module of the current context: given, a function of a module of this context.
    here = c.c_void_p()
    assert run.api.cuLibraryGetModule(c.byref(here), library) == 0 and here.value != first.value
    result, given = function(run, here)
    assert result == 0 and given, (result, log())
    assert context_of(run, given)[1] == other.value
    # A kernel of the library is of no context.
    assert run.api.cuCtxSetCurrent(run.context) == 0
    result, given = kernel(run, library)
    assert result == 0 and given, result


def race_case(run):
    module = run.load(IMAGE)
    done = []

    def unload():
        done.append(run.api.cuModuleUnload(module))

    def meanwhile():
        thread = threading.Thread(target=unload)
        thread.start()
        thread.join(20)
        done.append(thread.is_alive())

    hook = c.CFUNCTYPE(None)(meanwhile)
    c.c_void_p.in_dll(run.driver, "before_load").value = c.cast(hook, c.c_void_p).value
    result, given = function(run, module)
    # The unload was made while the layer loaded (it does not wait for the lookup), the lookup
    # fails, and what the layer loaded is unloaded: nothing is left.
    assert done == [0, False], done
    assert result == NOT_FOUND and not given and "unloaded meanwhile" in log(), (result, log())
    assert run.count("loads") == 2 and run.count("unloads") == 2 and run.driver.live_objects() == 0


def lookup(run, handle, library):
    return (kernel if library else function)(run, handle)


def unload(run, handle, library):
    return (run.api.cuLibraryUnload if library else run.api.cuModuleUnload)(handle)


def unload_case(run, library):
    handle = run.load(IMAGE, library=library)
    result, given = lookup(run, handle, library)
    assert result == 0 and given and run.driver.live_objects() == 2, (result, log())
    assert unload(run, handle, library) == 0
    assert run.driver.live_objects() == 0 and run.count("loads") == run.count("unloads") == 2
    # Loaded again (the driver may give the same handle): given again, from code loaded again.
    again = run.load(IMAGE, library=library)
    result, given = lookup(run, again, library)
    assert result == 0 and given and run.driver.live_objects() == 2 and run.count("loads") == 4, result


def restore_case(run, library):
    handle = run.load(IMAGE, library=library)
    result, given = lookup(run, handle, library)
    assert result == 0 and given
    loads = run.count("loads")
    c.c_size_t.in_dll(run.driver, "unload_fails_for").value = handle.value
    assert unload(run, handle, library) == 201
    # The driver did not unload it: the kernel is the program's as it was, and nothing is loaded
    # again for it.
    assert run.driver.live_objects() == 2 and run.count("unloads") == 0
    assert lookup(run, handle, library) == (0, given) and run.count("loads") == loads
    c.c_size_t.in_dll(run.driver, "unload_fails_for").value = 0
    assert unload(run, handle, library) == 0 and run.driver.live_objects() == 0


def off_case(run):
    module = run.load(IMAGE)
    loads = run.count("loads")
    result, given = function(run, module)
    assert result == NOT_FOUND and not given and run.count("loads") == loads and GIVEN not in log(), (result, log())


def linked_case(run):
    name = b"k_rdc_calls_out"
    # Loaded as it is (the driver's own answer to relocatable code is not the stand-in's).
    module = c.c_void_p()
    loads = run.count("loads")
    if run.api.cuModuleLoadData(c.byref(module), RELOCATABLE) == 0:
        loads = run.count("loads")
        result, given = function(run, module, name)
        assert result == NOT_FOUND and not given, result
    assert run.count("loads") == loads and GIVEN not in log(), log()
    # Linked: the module is of the image that the driver's linker made.
    made = b".version 8.8\n.target sm_70\n.address_size 64\n.visible .entry k_rdc_both()\n{\n ret;\n}\n\0"
    c.c_void_p.in_dll(run.driver, "linked").value = c.cast(c.c_char_p(made), c.c_void_p).value
    c.c_size_t.in_dll(run.driver, "linked_size").value = len(made)
    state, image, size = c.c_void_p(), c.c_void_p(), c.c_size_t()
    assert run.api.cuLinkCreate_v2(0, None, None, c.byref(state)) == 0
    added = run.api.cuLinkAddData_v2(state, 2, RELOCATABLE, c.c_size_t(len(RELOCATABLE)), b"unit", 0, None, None)
    if added == 0 and run.api.cuLinkComplete(state, c.byref(image), c.byref(size)) == 0:
        assert run.api.cuModuleLoadData(c.byref(module), image) == 0
        loads = run.count("loads")
        assert function(run, module, b"k_rdc_both")[0] == 0
        result, given = function(run, module, name)
        assert result == NOT_FOUND and not given, result
    assert run.count("loads") == loads and GIVEN not in log(), log()
    print(f"   (the relocatable input: cuLinkAddData {added})")


CASES = {
    "contexts": contexts_case,
    "library": library_case,
    "race": race_case,
    "unload-module": lambda run: unload_case(run, False),
    "unload-library": lambda run: unload_case(run, True),
    "restore-module": lambda run: restore_case(run, False),
    "restore-library": lambda run: restore_case(run, True),
    "off": off_case,
    "linked": linked_case,
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
            for settings in ({}, {"VF_LAZY": "0"}, {"VF_SAFE": "1"}):
                if case == "off":
                    settings = dict(settings, VF_TOLD_CODE="0")
                env = dict(os.environ, VF_REAL_LIBCUDA=mock, VF_PROXY_CACHE="0", VF_LOG=str(Path(tmp) / f"{case}.log"),
                           VF_LOG_LEVEL="1", VF_CAPABILITY="8.0", **settings)
                Path(env["VF_LOG"]).unlink(missing_ok=True)
                done = subprocess.run([sys.executable, __file__, proxy, mock, case], env=env, timeout=120)
                label = " ".join(f"{k}={v}" for k, v in settings.items())
                print(f"{'ok  ' if done.returncode == 0 else 'FAIL'}  {case} {label}".rstrip())
                failed |= done.returncode != 0
        print("FAIL told" if failed else
              "PASS told: a kernel of the code for the capability told is given in its module's context, and goes with it")
        sys.exit(1 if failed else 0)
