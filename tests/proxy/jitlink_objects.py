"""CPU nvJitLink regression for native CUDA host objects and static libraries, for null
arguments and empty inputs, and for kernels that cannot run on Volta: PTX or machine code in which
every kernel traps is refused, as PTX text and in a fat binary alike; with one kernel that runs, both
are linked, and the cubin that the link makes names the kernel that traps in a note, which a link
of that cubin keeps.

Usage: python3 tests/proxy/jitlink_objects.py adapter.so cuda-12.9.2-directory
"""
import ctypes as c
import functools
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import refusal_note


@functools.cache
def nvcc_command(toolkit):
    """The nvcc of TOOLKIT and what this machine needs to compile with it (scripts/nvcc-host.sh:
    nothing on most systems)."""
    nvcc = str(toolkit / "bin/nvcc")
    helper = Path(__file__).resolve().parents[2] / "scripts/nvcc-host.sh"
    added = subprocess.run(["bash", str(helper), "--args", nvcc], check=True, stdout=subprocess.PIPE,
                           text=True).stdout.splitlines()
    return [nvcc, *added]


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
        assert refusal_note.read(cubin.raw) == [], "a note in a link without a kernel that traps"
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


# A kernel that uses an instruction with no Volta sequence (clusters of blocks), and one that runs.
BAD = 'extern "C" __global__ void bad() { asm volatile("barrier.cluster.arrive;"); }\n'
GOOD = 'extern "C" __global__ void good(int *p) { *p = 1; }\n'
# A kernel whose Turing machine code the translator refuses (no translation of IMMA).
IMMA = ('extern "C" __global__ void imma(const unsigned *in, unsigned *out) {\n'
        '    unsigned d0 = in[0], d1 = in[1];\n'
        '    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"\n'
        '                 : "+r"(d0), "+r"(d1) : "r"(in[2]), "r"(in[3]));\n'
        '    out[0] = d0; out[1] = d1;\n'
        '}\n')
NVJITLINK_ERROR_PTX_COMPILE = 4
NVJITLINK_ERROR_UNRECOGNIZED_INPUT = 8


def linked_cubin(lib, h):
    size = c.c_size_t()
    assert lib.nvJitLinkGetLinkedCubinSize(h, c.byref(size)) == 0
    cubin = c.create_string_buffer(size.value)
    assert lib.nvJitLinkGetLinkedCubin(h, cubin) == 0
    return cubin.raw


def refusals(lib, toolkit, work, log):
    """Every kernel traps: refused, by AddData and AddFile, with the type given and with
    NVJITLINK_INPUT_ANY, as PTX text and as a fat binary (compressed or not) of that PTX alike;
    also relocatable Turing machine code that the translator refuses. (Until 2026-11-23 the fat
    binaries were linked into machine code that only traps.) With one kernel that runs, the same
    inputs are linked, the PTX and the machine code, and the kernel that traps is said in the log
    for each, as the layer says a kernel that it does not give to the program, and named with its
    reason in the note of the cubin that the link makes (until 2026-11-23 nothing in the cubin
    said it, and the layer gave it to a program that loaded the cubin), which a link of that cubin
    (as a cubin, and in a fat binary) keeps."""
    nvcc = [*nvcc_command(toolkit), "-allow-unsupported-compiler", "-Wno-deprecated-gpu-targets"]
    inputs = []
    for name, source in [("bad", BAD), ("half", BAD + GOOD)]:
        (work / f"{name}.cu").write_text(source)
        subprocess.run(nvcc + ["-ptx", "-arch=compute_90", f"{name}.cu", "-o", f"{name}.ptx"], cwd=work, check=True)
        subprocess.run(nvcc + ["-fatbin", "-gencode", "arch=compute_90,code=compute_90", f"{name}.cu",
                               "-o", f"{name}.fatbin"], cwd=work, check=True)
        subprocess.run(nvcc + ["-fatbin", "-Xfatbin=-compress=false", "-gencode", "arch=compute_90,code=compute_90",
                               f"{name}.cu", "-o", f"{name}.raw.fatbin"], cwd=work, check=True)
        inputs += [(name, f"{name}.ptx", 2), (name, f"{name}.fatbin", 4), (name, f"{name}.raw.fatbin", 4)]
    (work / "imma.cu").write_text(IMMA)
    subprocess.run(nvcc + ["-fatbin", "-rdc=true", "-gencode", "arch=compute_75,code=sm_75", "imma.cu",
                           "-o", "imma.fatbin"], cwd=work, check=True)
    inputs.append(("imma", "imma.fatbin", 4))
    (work / "halfimma.cu").write_text(IMMA + GOOD)
    subprocess.run(nvcc + ["-fatbin", "-rdc=true", "-gencode", "arch=compute_75,code=sm_75", "halfimma.cu",
                           "-o", "halfimma.fatbin"], cwd=work, check=True)
    inputs.append(("halfimma", "halfimma.fatbin", 4))
    wrong = []
    for name, file, explicit_type in inputs:
        path = work / file
        data = path.read_bytes() + (b"\0" if explicit_type == 2 else b"")
        for kind in (explicit_type, 10):
            for memory in (True, False):
                h = c.c_void_p()
                assert lib.nvJitLinkCreate(c.byref(h), 1, (c.c_char_p * 1)(b"-arch=sm_70")) == 0
                said = log.stat().st_size
                if memory:
                    rc = lib.nvJitLinkAddData(h, kind, data, c.c_size_t(len(data)), os.fsencode(path))
                else:
                    rc = lib.nvJitLinkAddFile(h, kind, os.fsencode(path))
                complete = lib.nvJitLinkComplete(h) if rc == 0 else None
                note = refusal_note.read(linked_cubin(lib, h)) if complete == 0 else None
                with open(log, "rb") as f:
                    f.seek(said)
                    lines = f.read()
                case = (file, kind, "AddData" if memory else "AddFile")
                trapping = {"half": "bad", "halfimma": "imma"}.get(name)
                if trapping and complete == 0 and memory and kind == explicit_type:
                    # The cubin linked again, as a cubin and in a fat binary: the note is kept.
                    output = linked_cubin(lib, h)
                    for again_type, again in ((1, output), (4, fat_binary(output))):
                        g = c.c_void_p()
                        assert lib.nvJitLinkCreate(c.byref(g), 1, (c.c_char_p * 1)(b"-arch=sm_70")) == 0
                        assert lib.nvJitLinkAddData(g, again_type, again, c.c_size_t(len(again)), b"again") == 0
                        assert lib.nvJitLinkComplete(g) == 0
                        kept = refusal_note.read(linked_cubin(lib, g))
                        assert lib.nvJitLinkDestroy(c.byref(g)) == 0
                        if kept != note:
                            wrong.append(f"{case}: linked again as input type {again_type}, the note is {kept}")
                assert lib.nvJitLinkDestroy(c.byref(h)) == 0
                if name == "bad":
                    want = (NVJITLINK_ERROR_PTX_COMPILE, None)
                elif name == "imma":
                    want = (NVJITLINK_ERROR_UNRECOGNIZED_INPUT, None)
                else:
                    want = (0, 0)
                if (rc, complete) != want:
                    wrong.append(f"{case}: AddData/AddFile {rc}, Complete {complete}, want {want}")
                elif trapping and f"kernel {trapping} cannot run on this GPU".encode() not in lines:
                    wrong.append(f"{case}: linked without saying that kernel {trapping} traps")
                elif trapping and ([k for k, _, reason in note if reason] != [trapping]):
                    wrong.append(f"{case}: the linked cubin names {note}, not kernel {trapping} with its reason")
                elif not trapping and b"compilation refused" not in lines:
                    wrong.append(f"{case}: refused without saying why")
    assert not wrong, "\n".join(wrong)
    return len(inputs) * 4


def fat_binary(cubin):
    """`cubin` as the one entry of a fat binary, for sm_70 and stored as it is (the header of the
    sm_70 entry of a fat binary of the repository's, with this size)."""
    native = Path(__file__).resolve().parents[2] / "crates/vf-fatbin/tests/data/elf70-elf75-ptx70-lz4.fatbin"
    header = bytearray(native.read_bytes()[16:16 + 64])
    stored = cubin + bytes(-len(cubin) % 8)
    struct.pack_into("<QI", header, 8, len(stored), 0)
    struct.pack_into("<Q", header, 0x28, struct.unpack_from("<Q", header, 0x28)[0] & ~0xB000)
    struct.pack_into("<Q", header, 0x38, 0)
    assert struct.unpack_from("<HI", header, 0)[0] == 2 and struct.unpack_from("<I", header, 0x1C)[0] == 70
    body = bytes(header) + stored
    return struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(body)) + body


def file_names(lib, work):
    """AddFile reads a file by the bytes of its name, one that is not UTF-8 too, and not the file
    whose name its text gives, with U+FFFD for each byte that is not (here one in which every
    kernel traps, which is refused); for PTX, given as such and as NVJITLINK_INPUT_ANY. (Until
    2026-11-23 the adapter opened that other file, or found none.)"""
    head = b".version 8.8\n.target sm_75\n.address_size 64\n"
    asked = work / os.fsdecode(b"k\xff.ptx")
    asked.write_bytes(head + b".visible .entry good() { ret; }\n")
    other = work / "k\ufffd.ptx"
    for decoy in (False, True):
        if decoy:
            other.write_bytes(head + b".visible .entry bad() { barrier.cluster.arrive; ret; }\n")
        for kind in (2, 10):
            h = c.c_void_p()
            assert lib.nvJitLinkCreate(c.byref(h), 1, (c.c_char_p * 1)(b"-arch=sm_70")) == 0
            rc = lib.nvJitLinkAddFile(h, kind, os.fsencode(asked))
            complete = lib.nvJitLinkComplete(h) if rc == 0 else None
            assert lib.nvJitLinkDestroy(c.byref(h)) == 0
            assert (rc, complete) == (0, 0), (os.fsencode(asked), kind, "with" if decoy else "without", rc, complete)
    other.unlink()


if __name__ == "__main__":
    adapter = str(Path(sys.argv[1]).resolve())
    toolkit = Path(sys.argv[2]).resolve()
    os.environ["VF_JITLINK_REAL"] = str(toolkit / "lib/libnvJitLink.so.12")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        log = work / "vf.log"
        os.environ["VF_LOG"] = str(log)
        log.write_bytes(b"")
        (work / "helper.cu").write_text('extern "C" __device__ void helper() { asm volatile("membar.cta;"); }\n')
        subprocess.run([*nvcc_command(toolkit), "-allow-unsupported-compiler", "-arch=sm_70",
                        "-dc", "helper.cu", "-o", "helper.o"], cwd=work, check=True)
        subprocess.run(["ar", "rcs", "helper.a", "helper.o"], cwd=work, check=True)
        lib = c.CDLL(adapter)
        null_arguments(lib)
        file_names(lib, work)
        for file, explicit_type in [("helper.o", 5), ("helper.a", 6)]:
            for kind in (explicit_type, 10):  # NVJITLINK_INPUT_ANY
                for memory in (True, False):
                    check(lib, work / file, kind, memory)
        refused = refusals(lib, toolkit, work, log)
        print("PASS host objects and archives through AddData/AddFile and explicit/automatic types, "
              f"null arguments and empty inputs, a file whose name is not UTF-8, {refused} inputs of kernels that trap")
