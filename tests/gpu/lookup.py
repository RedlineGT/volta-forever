"""Check lookup status and output handles through the CUDA module and library APIs."""
import ctypes as C
import os
from pathlib import Path
import sys


def check(status, api):
    if status:
        raise RuntimeError(f"{api} returned CUDA status {status}")


def main(path, cases):
    cuda = C.CDLL(os.environ.get("DRVRUN_LIBCUDA", "libcuda.so.1"))
    image = C.create_string_buffer(Path(path).read_bytes())
    context, device = C.c_void_p(), C.c_int()
    check(cuda.cuInit(0), "cuInit")
    check(cuda.cuDeviceGet(C.byref(device), 0), "cuDeviceGet")
    check(cuda.cuCtxCreate_v2(C.byref(context), 0, device), "cuCtxCreate_v2")
    for api in ("module", "library"):
        owner = C.c_void_p()
        if api == "module":
            check(cuda.cuModuleLoadData(C.byref(owner), image), "cuModuleLoadData")
            get, unload = cuda.cuModuleGetFunction, cuda.cuModuleUnload
        else:
            check(cuda.cuLibraryLoadData(C.byref(owner), image, None, None, 0, None, None, 0), "cuLibraryLoadData")
            get, unload = cuda.cuLibraryGetKernel, cuda.cuLibraryUnload
        for case in cases:
            name, expected = case.split("=")
            handle = C.c_void_p()
            status = get(C.byref(handle), owner, name.encode("ascii"))
            print(f"{api} {name} -> {status}, handle={bool(handle)}", flush=True)
            if status != int(expected) or bool(handle) != (status == 0):
                raise RuntimeError(f"{api} {name} expected CUDA status {expected} and matching handle")
        check(unload(owner), f"{api} unload")
    check(cuda.cuCtxDestroy_v2(context), "cuCtxDestroy_v2")
    print("all lookup results match")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
