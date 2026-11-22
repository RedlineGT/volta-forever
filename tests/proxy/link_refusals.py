"""CPU regression for kernels refused through the driver's linker.

A kernel that cannot run on the GPU, in an input of cuLinkAddData, is refused when the program asks
for it in the image that cuLinkComplete made: as the program got it, and as a copy loaded after the
link is gone. A stand-in driver (link_mock_driver.c) links into a fixed cubin and has every kernel.

Run with python3 tests/proxy/link_refusals.py path/to/libcuda.so.1.
"""
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PTX = (b".version 8.8\n.target sm_75\n.address_size 64\n"
       b".visible .entry bad()\n{\n barrier.cluster.arrive;\n ret;\n}\n"
       b".visible .entry good()\n{\n ret;\n}\n")
INPUT_PTX = 1
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
