"""CPU nvJitLink regression for native CUDA host objects and static libraries, and for null
arguments and empty inputs.

Usage: python3 tests/proxy/jitlink_objects.py adapter.so cuda-12.9.2-directory
"""
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def check(lib, path, kind, memory):
    h = c.c_void_p()
    options = (c.c_char_p * 1)(b"-arch=sm_70")
    assert lib.nvJitLinkCreate(c.byref(h), 1, options) == 0
    try:
        def add(data, type_, name):
            return lib.nvJitLinkAddData(h, type_, data, c.c_size_t(len(data)), name)

        # Force the linker to extract the archive member to resolve a device symbol.
        caller = (b".version 8.8\n.target sm_70\n.address_size 64\n"
                  b".extern .func helper();\n.visible .entry use_helper() { call helper, (); ret; }\n\0")
        assert add(caller, 2, b"caller.ptx") == 0
        if memory:
            rc = add(path.read_bytes(), kind, os.fsencode(path))
        else:
            rc = lib.nvJitLinkAddFile(h, kind, os.fsencode(path))
        assert rc == 0, (path.name, kind, memory, rc)
        assert lib.nvJitLinkComplete(h) == 0, (path.name, kind, memory)
        size = c.c_size_t()
        assert lib.nvJitLinkGetLinkedCubinSize(h, c.byref(size)) == 0
        cubin = c.create_string_buffer(size.value)
        assert lib.nvJitLinkGetLinkedCubin(h, cubin) == 0
        assert cubin.raw[:4] == b"\x7fELF" and cubin.raw[0x30] == 70
    finally:
        assert lib.nvJitLinkDestroy(c.byref(h)) == 0


def null_arguments(lib):
    """What nvJitLink 12.9.2 refuses with NVJITLINK_ERROR_INVALID_INPUT (3): a null array of
    options, even of none, and an input of no bytes or at a null address, of every kind. A null
    option is refused too (12.9.2 reads through it, as the adapter did through a null array)."""
    h = c.c_void_p()
    assert lib.nvJitLinkCreate(c.byref(h), 1, None) == 3
    assert lib.nvJitLinkCreate(c.byref(h), 0, None) == 3
    assert lib.nvJitLinkCreate(c.byref(h), 2, (c.c_char_p * 2)(b"-arch=sm_70", None)) == 3
    assert lib.nvJitLinkCreate(c.byref(h), 1, (c.c_char_p * 1)(b"-arch=sm_70")) == 0
    try:
        for kind in (1, 2, 3, 4, 5, 6, 10):
            for data, size in ((None, 0), (b".version 8.8\n", 0), (None, 16)):
                rc = lib.nvJitLinkAddData(h, kind, data, c.c_size_t(size), b"empty")
                assert rc == 3, (kind, data, size, rc)
    finally:
        assert lib.nvJitLinkDestroy(c.byref(h)) == 0


if __name__ == "__main__":
    adapter = str(Path(sys.argv[1]).resolve())
    toolkit = Path(sys.argv[2]).resolve()
    os.environ["VF_JITLINK_REAL"] = str(toolkit / "lib/libnvJitLink.so.12")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "helper.cu").write_text('extern "C" __device__ void helper() { asm volatile("membar.cta;"); }\n')
        subprocess.run([toolkit / "bin/nvcc", "-allow-unsupported-compiler", "-arch=sm_70",
                        "-dc", "helper.cu", "-o", "helper.o"], cwd=work, check=True)
        subprocess.run(["ar", "rcs", "helper.a", "helper.o"], cwd=work, check=True)
        lib = c.CDLL(adapter)
        null_arguments(lib)
        for file, explicit_type in [("helper.o", 5), ("helper.a", 6)]:
            for kind in (explicit_type, 10):  # NVJITLINK_INPUT_ANY
                for memory in (True, False):
                    check(lib, work / file, kind, memory)
        print("PASS host objects and archives through AddData/AddFile and explicit/automatic types, "
              "null arguments and empty inputs")
