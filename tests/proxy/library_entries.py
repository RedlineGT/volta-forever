"""CPU regression for refused handles and CUDA input preservation on internal images.

Run with python3 tests/proxy/library_entries.py path/to/libcuda.so.1.
"""
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def check(proxy, mock):
    api = c.CDLL(proxy)
    backend = c.CDLL(mock)
    ptx = (b".version 8.8\n.target sm_75\n.address_size 64\n"
           b".visible .entry bad()\n{\n barrier.cluster.arrive;\n ret;\n}\n"
           b".visible .entry good()\n{\n ret;\n}\n")
    library = c.c_void_p()
    options = (c.c_int * 1)(1)  # CU_LIBRARY_BINARY_IS_PRESERVED
    values = (c.c_void_p * 1)(None)
    assert api.cuLibraryLoadData(c.byref(library), ptx, None, None, 0, options, values, 1) == 0
    # A refused lookup leaves NULL in the handle, as NVIDIA's driver does for a name it does not
    # have: a program that ignores the error must not launch what the variable held before.
    kernel = c.c_void_p(0xDEADBEEF)
    assert api.cuLibraryGetKernel(c.byref(kernel), library, b"bad") == 218
    assert kernel.value is None, hex(kernel.value)
    assert api.cuLibraryGetKernel(c.byref(kernel), library, b"good") == 0
    function, alias = c.c_void_p(), c.c_void_p()
    assert api.cuKernelGetFunction(c.byref(function), kernel) == 0
    assert api.cuFuncGetModule(c.byref(alias), function) == 0
    count = c.c_uint()
    assert api.cuModuleGetFunctionCount(c.byref(count), alias) == 0 and count.value == 1
    alias_entries = (c.c_void_p * 2)(None, 12345)
    assert api.cuModuleEnumerateFunctions(alias_entries, 2, alias) == 0
    assert alias_entries[0] % 16 == 2 and alias_entries[1] == 12345
    module = c.c_void_p()
    assert api.cuLibraryGetModule(c.byref(module), library) == 0
    direct = c.c_void_p()
    assert api.cuModuleLoadData(c.byref(direct), ptx) == 0
    function = c.c_void_p(0xDEADBEEF)
    assert api.cuModuleGetFunction(c.byref(function), direct, b"bad") == 218
    assert function.value is None, hex(function.value)

    for owner, count_fn, enum_name in (
        (library, api.cuLibraryGetKernelCount, "cuLibraryEnumerateKernels"),
        (module, api.cuModuleGetFunctionCount, "cuModuleEnumerateFunctions"),
        (direct, api.cuModuleGetFunctionCount, "cuModuleEnumerateFunctions"),
    ):
        count = c.c_uint()
        assert count_fn(c.byref(count), owner) == 0 and count.value == 1, (enum_name, count.value)
        ptr = c.c_void_p()
        assert api.cuGetProcAddress(enum_name.encode(), c.byref(ptr), 12040, 0) == 0
        lookup = c.CFUNCTYPE(c.c_int, c.POINTER(c.c_void_p), c.c_uint, c.c_void_p)(ptr.value)
        for enum in (getattr(api, enum_name), lookup):
            for cap in (1, 2):
                entries = (c.c_void_p * 2)(None, 12345)
                assert enum(entries, cap, owner) == 0
                assert entries[0] % 16 == 2 and entries[1] == 12345, list(entries)
    # A context-owned module can go away before its library. Have the mock
    # recycle that alias as each of a native module, an adapted module, and a
    # second library with the opposite refusal set (including on a cache hit).
    native = ptx.replace(b"sm_75", b"sm_70").replace(b"barrier.cluster.arrive;", b"ret;")
    other = ptx.replace(b"bad()", b"temp()").replace(b"good()", b"bad()").replace(b"temp()", b"good()")
    for image, as_library, allowed in ((native, False, 2), (other, False, 1), (other, True, 1)):
        assert api.cuLibraryGetModule(c.byref(module), library) == 0
        c.c_size_t.in_dll(backend, "next_handle").value = module.value
        fresh = c.c_void_p()
        if as_library:
            assert api.cuLibraryLoadData(c.byref(fresh), image, None, None, 0, None, None, 0) == 0
            get_count, enumerate_ = api.cuLibraryGetKernelCount, api.cuLibraryEnumerateKernels
        else:
            assert api.cuModuleLoadData(c.byref(fresh), image) == 0
            get_count, enumerate_ = api.cuModuleGetFunctionCount, api.cuModuleEnumerateFunctions
        assert fresh.value == module.value
        assert get_count(c.byref(count), fresh) == 0 and count.value == allowed, (as_library, count.value, allowed)
        entries = (c.c_void_p * 2)(None, None)
        assert enumerate_(entries, 2, fresh) == 0
        assert entries[0] % 16 == 1
        assert (entries[1] is None) == (allowed == 1)
        assert (api.cuLibraryUnload if as_library else api.cuModuleUnload)(fresh) == 0
    c.c_uint.in_dll(backend, "force_name_fallback").value = 1
    assert api.cuLibraryGetKernelCount(c.byref(count), library) == 0 and count.value == 1
    assert c.c_uint.in_dll(backend, "invalid_function_names").value == 0
    assert c.c_uint.in_dll(backend, "preserved").value == 0
    assert api.cuLibraryUnload(library) == 0
    assert api.cuModuleUnload(direct) == 0
    # With native PTX, the driver receives the original pointer and its hint.
    assert api.cuLibraryLoadData(c.byref(library), native, None, None, 0, options, values, 1) == 0
    assert c.c_uint.in_dll(backend, "preserved").value == 1
    assert api.cuLibraryUnload(library) == 0
    # A lookup that fails leaves NULL in the handle, in a deferred library too: a kernel that the
    # driver's compiler refuses only when the whole module is loaded for it (the layer learns it
    # during that lookup), and a name that the library does not have.
    refused = (b".version 8.8\n.target sm_75\n.address_size 64\n"
               b".visible .entry refused()\n{\n .reg .b32 %r<2>;\n mov.u32 %r1, 12345;\n ret;\n}\n"
               b".visible .entry good()\n{\n ret;\n}\n")
    assert api.cuLibraryLoadData(c.byref(library), refused, None, None, 0, None, None, 0) == 0
    for name, error in ((b"absent", 500), (b"refused", 218), (b"refused", 218), (b"absent", 500), (b"good", 0)):
        kernel = c.c_void_p(0xDEADBEEF)
        assert api.cuLibraryGetKernel(c.byref(kernel), library, name) == error, name
        assert (kernel.value is None) == (error != 0), (name, kernel.value)
    assert api.cuLibraryUnload(library) == 0
    assert c.c_uint.in_dll(backend, "loads").value == c.c_uint.in_dll(backend, "unloads").value


if __name__ == "__main__":
    if len(sys.argv) == 3:
        check(sys.argv[1], sys.argv[2])
    else:
        with tempfile.TemporaryDirectory() as tmp:
            mock = str(Path(tmp) / "entries.so")
            subprocess.run(["cc", "-shared", "-fPIC", "-Wl,-Bsymbolic", "-o", mock,
                            Path(__file__).with_name("entries_mock_driver.c")], check=True)
            cache = str(Path(tmp) / "cache")
            for lazy, cache_dir in (("0", "0"), ("1", "0"), ("1", cache), ("1", cache)):
                env = dict(os.environ, VF_REAL_LIBCUDA=mock, VF_PROXY_CACHE=cache_dir, VF_LAZY=lazy)
                subprocess.run([sys.executable, __file__, sys.argv[1], mock], env=env, check=True)
            print("PASS library and module enumerations, eager/lazy image ownership")
