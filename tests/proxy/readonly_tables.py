"""CPU regression for the read-only data that kernels compiled alone hold copies of.

A deferred library (`VF_LAZY`) gives a kernel that only reads a table of its module (a variable with
an initializer that no function writes) from a sub-module of its own, which holds a copy of the
table, instead of compiling the whole module for it. While such a copy is loaded, the program is not
given the table by name (CUDA_ERROR_NOT_SUPPORTED, 801: NVIDIA's driver would give it): a write to the
copy it would be given would not reach the kernel. A table that the program was given first stays the
whole module's. Each round in a process of its own, with a stand-in for the driver that keeps the text
of what it is given; the second load of the module in a round is set up from the proxy cache.

Run with python3 tests/proxy/readonly_tables.py path/to/libcuda.so.1.
"""
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PTX = b"""\
.version 8.8
.target sm_75
.address_size 64

.global .align 4 .b8 table[8] = {1, 0, 0, 0, 2, 0, 0, 0};
.global .align 4 .b32 counter;

.visible .entry reader(
\t.param .u64 reader_param_0
)
{
\t.reg .b32 %r<2>;
\t.reg .b64 %rd<3>;
\tld.param.u64 %rd1, [reader_param_0];
\tcvta.to.global.u64 %rd2, %rd1;
\tld.global.u32 %r1, [table+4];
\tst.global.u32 [%rd2], %r1;
\tret;
}

.visible .entry other_reader(
\t.param .u64 other_reader_param_0
)
{
\t.reg .b32 %r<2>;
\t.reg .b64 %rd<3>;
\tld.param.u64 %rd1, [other_reader_param_0];
\tcvta.to.global.u64 %rd2, %rd1;
\tld.global.u32 %r1, [table];
\tst.global.u32 [%rd2], %r1;
\tret;
}

.visible .entry counting()
{
\t.reg .b32 %r<2>;
\tatom.global.add.u32 %r1, [counter], 1;
\tret;
}
"""


def text(mock, function, *args):
    f = getattr(mock, function)
    f.restype = c.c_char_p
    return f(*args).decode()


def whole(image):
    return all(name in image for name in ("reader(", "other_reader(", "counting("))


def copies_first(api, mock):
    """A kernel that reads the table is taken first: it is given from a sub-module with the table in
    it, and the table is then refused by every name the program could ask for it with."""
    library = c.c_void_p()
    assert api.cuLibraryLoadData(c.byref(library), PTX, None, None, 0, None, None, 0) == 0
    kernel = c.c_void_p()
    assert api.cuLibraryGetKernel(c.byref(kernel), library, b"reader") == 0 and kernel.value
    image = text(mock, "last_image")
    assert "reader(" in image and not whole(image), "the whole module was compiled for a kernel that reads a table:\n" + image
    assert ".global .align 4 .b8 table[8] = {1, 0, 0, 0, 2, 0, 0, 0};" in image, image
    address, size = c.c_ulonglong(7), c.c_size_t(7)
    # The program asks for the table: refused, and not as a name that the module does not have (500).
    assert api.cuLibraryGetGlobal(c.byref(address), c.byref(size), library, b"table") == 801
    assert address.value == 7, "an address was given"
    function, module = c.c_void_p(), c.c_void_p()
    assert api.cuKernelGetFunction(c.byref(function), kernel) == 0
    assert api.cuFuncGetModule(c.byref(module), function) == 0
    assert api.cuModuleGetGlobal_v2(c.byref(address), c.byref(size), module, b"table") == 801
    # A name that the module does not have is answered as the driver answers it.
    assert api.cuLibraryGetGlobal(c.byref(address), c.byref(size), library, b"absent") == 500
    # A variable that kernels write has one copy, the whole module's, and is given.
    assert api.cuLibraryGetGlobal(c.byref(address), c.byref(size), library, b"counter") == 0
    assert address.value != 7
    loaded = text(mock, "last_image")
    assert whole(loaded), loaded
    whole_module = c.c_void_p()
    assert api.cuLibraryGetModule(c.byref(whole_module), library) == 0
    for get in (api.cuModuleGetGlobal, api.cuModuleGetGlobal_v2):
        assert get(c.byref(address), c.byref(size), whole_module, b"table") == 801
        assert get(c.byref(address), c.byref(size), whole_module, b"counter") == 0
    assert api.cuLibraryUnload(library) == 0


def named_first(api, mock):
    """The program is given the table first: the kernels that read it come from the whole module,
    whose copy it was given, and it can be given again."""
    library = c.c_void_p()
    assert api.cuLibraryLoadData(c.byref(library), PTX, None, None, 0, None, None, 0) == 0
    address, size = c.c_ulonglong(7), c.c_size_t(7)
    assert api.cuLibraryGetGlobal(c.byref(address), c.byref(size), library, b"table") == 0
    kernel = c.c_void_p()
    loads = c.c_uint.in_dll(mock, "loads").value
    assert api.cuLibraryGetKernel(c.byref(kernel), library, b"reader") == 0 and kernel.value
    assert c.c_uint.in_dll(mock, "loads").value == loads, "a sub-module was compiled"
    assert api.cuLibraryGetGlobal(c.byref(address), c.byref(size), library, b"table") == 0
    assert api.cuLibraryUnload(library) == 0


def check(proxy, mock_path, round_name):
    api = c.CDLL(proxy)
    mock = c.CDLL(mock_path)
    rounds = {"copies_first": copies_first, "named_first": named_first}
    # The second time, the library is set up from what the first left in the proxy cache.
    for _ in range(2):
        rounds[round_name](api, mock)


def main():
    if len(sys.argv) == 4:
        check(sys.argv[1], sys.argv[2], sys.argv[3])
        return
    proxy = str(Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory() as tmp:
        mock = str(Path(tmp) / "tables.so")
        subprocess.run(["cc", "-shared", "-fPIC", "-Wl,-Bsymbolic", "-o", mock,
                        Path(__file__).with_name("tables_mock_driver.c")], check=True)
        for round_name in ("copies_first", "named_first"):
            cache = Path(tmp) / f"cache-{round_name}"
            cache.mkdir(mode=0o700)
            env = dict(os.environ, VF_REAL_LIBCUDA=mock, VF_PROXY_CACHE=str(cache), VF_LAZY="1")
            env.pop("VF_ADAPT", None)
            subprocess.run([sys.executable, __file__, proxy, mock, round_name], env=env, check=True)
            print(f"PASS {round_name}")


if __name__ == "__main__":
    main()
