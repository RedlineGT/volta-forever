"""CPU regression for kernels refused through the driver's linker.

A kernel that cannot run on the GPU, in an input of cuLinkAddData, is refused when the program asks
for it in the image that cuLinkComplete made: as the program got it, and as a copy loaded after the
link is gone. A stand-in driver (link_mock_driver.c) links into a fixed cubin and has every kernel.

Run with python3 tests/proxy/link_refusals.py path/to/libcuda.so.1.
"""
import ctypes as c
import mmap
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

PTX = (b".version 8.8\n.target sm_75\n.address_size 64\n"
       b".visible .entry bad()\n{\n barrier.cluster.arrive;\n ret;\n}\n"
       b".visible .entry good()\n{\n ret;\n}\n")
INPUT_PTX = 1
INPUT_FATBINARY = 2
CUBIN = Path(__file__).resolve().parents[1] / "kernels" / "k_basic.128.sm70.cubin"


def lookup(api, module, name):
    function = c.c_void_p(0xDEADBEEF)
    status = api.cuModuleGetFunction(c.byref(function), module, name)
    return status, function.value


def load(api, image):
    module = c.c_void_p()
    assert api.cuModuleLoadData(c.byref(module), image) == 0
    return module


def check(proxy, mock):
    api, backend = c.CDLL(proxy), c.CDLL(mock)
    output = CUBIN.read_bytes()
    linked = c.create_string_buffer(output, len(output))
    c.c_void_p.in_dll(backend, "linked").value = c.addressof(linked)
    c.c_size_t.in_dll(backend, "linked_size").value = len(output)

    state = c.c_void_p()
    assert api.cuLinkCreate_v2(0, None, None, c.byref(state)) == 0
    assert api.cuLinkAddData_v2(state, INPUT_PTX, PTX, len(PTX), b"bad.ptx", 0, None, None) == 0
    image, size = c.c_void_p(), c.c_size_t()
    assert api.cuLinkComplete(state, c.byref(image), c.byref(size)) == 0
    module = load(api, image)
    assert lookup(api, module, b"bad") == (218, None), lookup(api, module, b"bad")
    assert lookup(api, module, b"good")[0] == 0
    copy = c.create_string_buffer(c.string_at(image, size.value), size.value)
    assert api.cuLinkDestroy(state) == 0
    assert c.c_uint.in_dll(backend, "destroyed").value == 1
    assert lookup(api, load(api, copy), b"bad") == (218, None)
    # And in a fat binary of the program's own, as the machine code for this GPU.
    native = (Path(__file__).resolve().parents[2] / "crates/vf-fatbin/tests/data/elf70-elf75-ptx70-lz4.fatbin")
    header = bytearray(native.read_bytes()[16:16 + 64])
    stored = output + bytes(-len(output) % 8)
    struct.pack_into("<QI", header, 8, len(stored), 0)
    struct.pack_into("<Q", header, 0x28, struct.unpack_from("<Q", header, 0x28)[0] & ~0xB000)
    struct.pack_into("<Q", header, 0x38, 0)
    assert struct.unpack_from("<HI", header, 0)[0] == 2 and struct.unpack_from("<I", header, 0x1C)[0] == 70
    body = bytes(header) + stored
    wrapped = struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(body)) + body
    assert lookup(api, load(api, wrapped), b"bad") == (218, None)
    assert lookup(api, load(api, wrapped), b"good")[0] == 0

    # Other bytes are another image, and a link with nothing refused, given the handle of the link
    # that is gone, gives nothing to its image.
    other = bytearray(output)
    other[len(other) // 2] ^= 0xff
    assert lookup(api, load(api, bytes(other)), b"bad")[0] == 0
    good = PTX.replace(b"barrier.cluster.arrive;", b"ret;")
    other_linked = c.create_string_buffer(bytes(other), len(other))
    c.c_void_p.in_dll(backend, "linked").value = c.addressof(other_linked)
    assert api.cuLinkCreate_v2(0, None, None, c.byref(state)) == 0
    assert api.cuLinkAddData_v2(state, INPUT_PTX, good, len(good), b"good.ptx", 0, None, None) == 0
    assert api.cuLinkComplete(state, c.byref(image), c.byref(size)) == 0
    assert lookup(api, load(api, image), b"bad")[0] == 0
    assert api.cuLinkDestroy(state) == 0
    assert c.c_uint.in_dll(backend, "inputs").value == 2

    # A link whose options name the architecture of its output (CU_JIT_TARGET) is given its inputs
    # for that one: adapted for this GPU when it is this GPU's, as they are when it is a later one.
    for target, expected in ((None, b".target sm_70"), (70, b".target sm_70"), (90, b".target sm_75"),
                             (0x10000 + 90, b".target sm_75")):
        options, values = (c.c_int * 1)(9), (c.c_void_p * 1)(target)
        count = 0 if target is None else 1
        assert api.cuLinkCreate_v2(count, options, values, c.byref(state)) == 0
        assert api.cuLinkAddData_v2(state, INPUT_PTX, PTX, len(PTX), b"bad.ptx", 0, None, None) == 0
        assert c.c_char_p(c.addressof(c.c_char.in_dll(backend, "seen_target"))).value == expected, target
        assert api.cuLinkDestroy(state) == 0

    # The image that a link gives without its size (sizeOut may be NULL) is known all the same.
    third = bytearray(output)
    third[len(third) // 3] ^= 0xff
    third_linked = c.create_string_buffer(bytes(third), len(third))
    c.c_void_p.in_dll(backend, "linked").value = c.addressof(third_linked)
    assert api.cuLinkCreate_v2(0, None, None, c.byref(state)) == 0
    assert api.cuLinkAddData_v2(state, INPUT_PTX, PTX, len(PTX), b"bad.ptx", 0, None, None) == 0
    assert api.cuLinkComplete(state, c.byref(image), None) == 0
    assert lookup(api, load(api, image), b"bad") == (218, None)
    assert api.cuLinkDestroy(state) == 0

    # What another thread does with a handle that the driver gives again once it is free, while
    # the destruction of a link or the unload of a module returns, is not dropped with what was
    # known of the handle before: a link made then, and a module loaded then.
    fourth = bytearray(output)
    fourth[len(fourth) // 4] ^= 0xff
    fourth_linked = c.create_string_buffer(bytes(fourth), len(fourth))
    c.c_void_p.in_dll(backend, "linked").value = c.addressof(fourth_linked)
    assert api.cuLinkCreate_v2(0, None, None, c.byref(state)) == 0
    assert api.cuLinkAddData_v2(state, INPUT_PTX, good, len(good), b"good.ptx", 0, None, None) == 0
    again = c.c_void_p()

    def link_meanwhile():
        assert api.cuLinkCreate_v2(0, None, None, c.byref(again)) == 0 and again.value == state.value
        assert api.cuLinkAddData_v2(again, INPUT_PTX, PTX, len(PTX), b"bad.ptx", 0, None, None) == 0

    meanwhile = c.CFUNCTYPE(None)(link_meanwhile)
    c.c_void_p.in_dll(backend, "while_freed").value = c.cast(meanwhile, c.c_void_p).value
    assert api.cuLinkDestroy(state) == 0
    c.c_void_p.in_dll(backend, "while_freed").value = None
    assert api.cuLinkComplete(again, c.byref(image), c.byref(size)) == 0
    assert lookup(api, load(api, image), b"bad") == (218, None)
    assert api.cuLinkDestroy(again) == 0
    first, second = load(api, PTX), c.c_void_p()

    def load_meanwhile():
        c.c_size_t.in_dll(backend, "next_module").value = first.value
        assert api.cuModuleLoadData(c.byref(second), PTX) == 0 and second.value == first.value

    meanwhile = c.CFUNCTYPE(None)(load_meanwhile)
    c.c_void_p.in_dll(backend, "while_freed").value = c.cast(meanwhile, c.c_void_p).value
    assert api.cuModuleUnload(first) == 0
    c.c_void_p.in_dll(backend, "while_freed").value = None
    assert lookup(api, second, b"bad") == (218, None)

    # A fat binary given with its size is read within it, whatever its header says: one cut short,
    # just before memory that cannot be read, is given to the linker as it is.
    fat = (Path(__file__).resolve().parents[2] / "crates/vf-fatbin/tests/data/elf75-ptx75-zstd.fatbin").read_bytes()
    page = mmap.PAGESIZE
    guard = -(-len(fat) // page) * page
    pages = mmap.mmap(-1, page + guard)
    start = c.addressof(c.c_char.from_buffer(pages))
    libc = c.CDLL(None)
    libc.mprotect.argtypes = (c.c_void_p, c.c_size_t, c.c_int)
    assert libc.mprotect(start + page, guard, 0) == 0
    cut = 200
    pages[page - cut:page] = fat[:cut]
    state = c.c_void_p()
    assert api.cuLinkCreate_v2(0, None, None, c.byref(state)) == 0
    inputs = c.c_uint.in_dll(backend, "inputs").value
    assert api.cuLinkAddData_v2(state, INPUT_FATBINARY, c.c_void_p(start + page - cut), cut, None, 0, None, None) == 0
    assert c.c_uint.in_dll(backend, "inputs").value == inputs + 1
    assert api.cuLinkDestroy(state) == 0


if __name__ == "__main__":
    if len(sys.argv) == 3:
        check(sys.argv[1], sys.argv[2])
    else:
        with tempfile.TemporaryDirectory() as tmp:
            mock = str(Path(tmp) / "link.so")
            subprocess.run(["cc", "-shared", "-fPIC", "-o", mock, Path(__file__).with_name("link_mock_driver.c")],
                           check=True)
            env = dict(os.environ, VF_REAL_LIBCUDA=mock, VF_PROXY_CACHE="0")
            subprocess.run([sys.executable, __file__, sys.argv[1], mock], env=env, check=True)
            print("PASS kernels refused through the linker")
