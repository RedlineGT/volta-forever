"""Ask for the lookup tables of tests/kernels/k_tables.cu by name, through the module or the library
interface, before or after the kernel that reads them, and check what is answered.

    python3 tests/gpu/tables.py IMAGE module|library STEP[=EXPECTED]...

The image is loaded once (cuModuleLoadData or cuLibraryLoadData) and the steps run in their order:
    run          launch `reader` (one block of 4 threads) and read the 4 words it stores (each is
                 table[i] + ctable[i] + ns::ntable[i]: 166,288,410,532 as compiled)
    get:NAME     cuModuleGetGlobal_v2 on the module, or cuLibraryGetGlobal on the library
    mod:NAME     cuModuleGetGlobal_v2 on the module that cuLibraryGetModule gives (library)
    fn:NAME      cuModuleGetGlobal_v2 on the module of the function of `reader` (cuLibraryGetKernel,
                 cuKernelGetFunction, cuFuncGetModule) (library)
    set:NAME     write 1000 + i into word i of NAME, at the address that get:NAME gives
EXPECTED is the CUDA status of a lookup, or the words that `run` must read. Each step prints a line;
the last line is "all results match" when every expected value was met. The driver is
$DRVRUN_LIBCUDA (default: libcuda.so.1, NVIDIA's own).
"""
import ctypes as C
import os
from pathlib import Path
import sys


def check(status, api):
    if status:
        raise RuntimeError(f"{api} returned CUDA status {status}")


def main(path, interface, steps):
    cuda = C.CDLL(os.environ.get("DRVRUN_LIBCUDA", "libcuda.so.1"))
    image = C.create_string_buffer(Path(path).read_bytes())
    context, device = C.c_void_p(), C.c_int()
    check(cuda.cuInit(0), "cuInit")
    check(cuda.cuDeviceGet(C.byref(device), 0), "cuDeviceGet")
    check(cuda.cuCtxCreate_v2(C.byref(context), 0, device), "cuCtxCreate_v2")
    owner = C.c_void_p()
    if interface == "module":
        check(cuda.cuModuleLoadData(C.byref(owner), image), "cuModuleLoadData")
    else:
        check(cuda.cuLibraryLoadData(C.byref(owner), image, None, None, 0, None, None, 0), "cuLibraryLoadData")
    out = C.c_ulonglong()
    check(cuda.cuMemAlloc_v2(C.byref(out), 16), "cuMemAlloc_v2")

    def function():
        handle = C.c_void_p()
        if interface == "module":
            check(cuda.cuModuleGetFunction(C.byref(handle), owner, b"reader"), "cuModuleGetFunction")
            return handle
        kernel = C.c_void_p()
        check(cuda.cuLibraryGetKernel(C.byref(kernel), owner, b"reader"), "cuLibraryGetKernel")
        check(cuda.cuKernelGetFunction(C.byref(handle), kernel), "cuKernelGetFunction")
        return handle

    def lookup(how, name):
        address, size = C.c_ulonglong(7), C.c_size_t(7)
        if how == "get" and interface == "module":
            status = cuda.cuModuleGetGlobal_v2(C.byref(address), C.byref(size), owner, name.encode())
        elif how == "get":
            status = cuda.cuLibraryGetGlobal(C.byref(address), C.byref(size), owner, name.encode())
        else:
            module = C.c_void_p()
            if how == "mod":
                check(cuda.cuLibraryGetModule(C.byref(module), owner), "cuLibraryGetModule")
            else:
                check(cuda.cuFuncGetModule(C.byref(module), function()), "cuFuncGetModule")
            status = cuda.cuModuleGetGlobal_v2(C.byref(address), C.byref(size), module, name.encode())
        return status, address.value, size.value

    def words(address, count):
        host = (C.c_uint * count)()
        check(cuda.cuMemcpyDtoH_v2(host, C.c_ulonglong(address), 4 * count), "cuMemcpyDtoH_v2")
        return ",".join(str(w) for w in host)

    matched = True
    for step in steps:
        what, _, expected = step.partition("=")
        if what == "run":
            params = (C.c_void_p * 1)(C.cast(C.pointer(out), C.c_void_p))
            check(cuda.cuLaunchKernel(function(), 1, 1, 1, 4, 1, 1, 0, None, params, None), "cuLaunchKernel")
            check(cuda.cuCtxSynchronize(), "cuCtxSynchronize")
            got = words(out.value, 4)
            print(f"run reader -> {got}", flush=True)
        elif what.startswith("set:"):
            status, address, size = lookup("get", what[4:])
            check(status, f"get:{what[4:]}")
            host = (C.c_uint * (size // 4))(*(1000 + i for i in range(size // 4)))
            check(cuda.cuMemcpyHtoD_v2(C.c_ulonglong(address), host, size), "cuMemcpyHtoD_v2")
            got = "0"
            print(f"{what} -> written", flush=True)
        else:
            how, _, name = what.partition(":")
            status, address, size = lookup(how, name)
            got = str(status)
            shown = f", size {size}, words {words(address, size // 4)}" if status == 0 else ""
            print(f"{how}:{name} -> {status}{shown}", flush=True)
        if expected and got != expected:
            print(f"    expected {expected}", flush=True)
            matched = False
    check(cuda.cuMemFree_v2(out), "cuMemFree_v2")
    unload = cuda.cuModuleUnload if interface == "module" else cuda.cuLibraryUnload
    check(unload(owner), "unload")
    check(cuda.cuCtxDestroy_v2(context), "cuCtxDestroy_v2")
    if not matched:
        raise SystemExit("some results do not match")
    print("all results match")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
