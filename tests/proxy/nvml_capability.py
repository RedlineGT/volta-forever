"""Exercise NVML lookup through ctypes, the path used by pynvml, without a GPU.

Run with python3 tests/proxy/nvml_capability.py path/to/libvf_audit.so.
"""
import _ctypes
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


MOCK = """
#include <stdint.h>
int nvmlDeviceGetCudaComputeCapability(void *device, int *major, int *minor) {
    if (!device || !major || !minor) return 2;
    *major = (uintptr_t)device == 1 ? 7 : 8;
    *minor = (uintptr_t)device == 1 ? 0 : 6;
    return 0;
}
"""


def check(library, expected):
    handle = ctypes.CDLL(str(library))
    query = handle.nvmlDeviceGetCudaComputeCapability
    integer = ctypes.c_int
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(integer), ctypes.POINTER(integer)]
    a, b = integer(), integer()
    assert query(1, ctypes.byref(a), ctypes.byref(b)) == 0
    assert (a.value, b.value) == expected, (a.value, b.value)
    assert query(2, ctypes.byref(a), ctypes.byref(b)) == 0
    assert (a.value, b.value) == (8, 6)
    assert query(None, ctypes.byref(a), ctypes.byref(b)) == 2
    assert (a.value, b.value) == (8, 6)
    assert query(1, None, ctypes.byref(b)) == 2
    _ctypes.dlclose(handle._handle)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python3 tests/proxy/nvml_capability.py AUDIT_LIBRARY (the layer's, as dist/lib/libvf_audit.so)")
    if sys.argv[1] == "--probe":
        directory = Path(sys.argv[2])
        expected = tuple(map(int, sys.argv[3].split(".")))
        check(directory / "libnvidia-ml.so.1", expected)
        check(directory / "libunrelated.so", (7, 0))
        check(directory / "libnvidia-ml.so.1", expected)
        # A native caller with a DT_NEEDED reference to NVML uses the same wrapper.
        assert ctypes.CDLL(str(directory / "libconsumer.so")).probe() == 10 * expected[0] + expected[1]
    else:
        audit = Path(sys.argv[1]).resolve(strict=True)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            mock = directory / "libnvidia-ml.so.1"
            subprocess.run(["cc", "-shared", "-fPIC", "-x", "c", "-", "-o", str(mock)],
                           input=MOCK, text=True, check=True)
            shutil.copyfile(mock, directory / "libunrelated.so")
            subprocess.run(["cc", "-shared", "-fPIC", "-o", str(directory / "libconsumer.so"),
                            "-x", "c", "-", "-L", tmp, "-l:libnvidia-ml.so.1", "-Wl,-rpath,$ORIGIN"],
                           input="""extern int nvmlDeviceGetCudaComputeCapability(void *, int *, int *);
                           int probe(void) {
                               int major, minor;
                               return nvmlDeviceGetCudaComputeCapability((void *)1, &major, &minor)
                                   ? -1 : 10 * major + minor;
                           }""", text=True, check=True)
            env = {k: v for k, v in os.environ.items() if k not in ("VOLTA_FOREVER", "VF_CAPABILITY")}
            env["LD_AUDIT"] = str(audit)
            for settings, expected in [
                ({}, "7.0"),
                ({"VOLTA_FOREVER": "test"}, "7.5"),
                ({"VOLTA_FOREVER": "test", "VF_CAPABILITY": "8.0"}, "8.0"),
                ({"VOLTA_FOREVER": "test", "VF_CAPABILITY": "12.0"}, "12.0"),
                ({"VOLTA_FOREVER": "test", "VF_CAPABILITY": "4294967295.0"}, "7.5"),
            ]:
                subprocess.run([sys.executable, __file__, "--probe", tmp, expected],
                               env={**env, **settings}, check=True)
