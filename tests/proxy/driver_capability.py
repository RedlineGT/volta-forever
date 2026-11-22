"""Check CUDA device and kernel capability queries through callers with different GPU images.

Run with python3 tests/proxy/driver_capability.py path/to/libcuda.so.1.
"""
import _ctypes
import ctypes as c
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


DRIVER = """
static const int sm[] = {70, 75, 80, 86};
int cuDeviceGetAttribute(int *v, int a, int d) {
    if (!v) return 101;
    if (d == 4 && a == 75) { *v = 7; return 0; }
    if (d == 5 && a == 76) { *v = 0; return 0; }
    if (d < 0 || d > 3) return 101;
    *v = a == 75 ? sm[d] / 10 : a == 76 ? sm[d] % 10 : 42;
    return 0;
}
int cuDeviceComputeCapability(int *major, int *minor, int d) {
    if (!major || !minor || d < 0 || d > 3) return 101;
    *major = sm[d] / 10;
    *minor = sm[d] % 10;
    return 0;
}
int cuFuncGetAttribute(int *v, int a, void *f) { *v = a == 6 ? 70 : 42; return 0; }
int cuKernelGetAttribute(int *v, int a, void *k, int d) { return cuFuncGetAttribute(v, a, k); }
"""

# A normal (non-tail) call makes this DSO the return address seen by the proxy.
CLIENT = """
int query(int (*f)(int *, int, int), int *v, int attr, int dev) { return f(v, attr, dev); }
int both(int (*f)(int *, int *, int), int *a, int *b, int dev) { return f(a, b, dev); }
int binary(int (*f)(int *, int, void *), int *v) { return f(v, 6, (void *)1); }
int kernel(int (*f)(int *, int, void *, int), int *v) { return f(v, 6, (void *)1, 0); }
"""


def probe(api, client, device_sm, binary_sm):
    val, a, b = c.c_int(), c.c_int(), c.c_int()
    for dev, sm in enumerate((device_sm, 75, 80, 86)):
        assert client.both(api.cuDeviceComputeCapability, c.byref(a), c.byref(b), dev) == 0
        assert (a.value, b.value) == divmod(sm, 10), (client._name, dev, a.value, b.value, sm)
        for attr, expected in ((75, sm // 10), (76, sm % 10), (1, 42)):
            assert client.query(api.cuDeviceGetAttribute, c.byref(val), attr, dev) == 0
            assert val.value == expected, (client._name, dev, attr, val.value, expected)
    # A failed lookup of the complementary attribute cannot turn a successful native answer into a guess.
    for dev, attr, expected in ((4, 75, 7), (5, 76, 0)):
        assert client.query(api.cuDeviceGetAttribute, c.byref(val), attr, dev) == 0 and val.value == expected
    a.value = b.value = val.value = -1
    assert client.both(api.cuDeviceComputeCapability, c.byref(a), c.byref(b), -1) == 101
    assert client.both(api.cuDeviceComputeCapability, None, c.byref(b), 0) == 101
    assert client.query(api.cuDeviceGetAttribute, c.byref(val), 75, -1) == 101
    assert client.query(api.cuDeviceGetAttribute, None, 75, 0) == 101
    assert (a.value, b.value, val.value) == (-1, -1, -1)
    assert client.binary(api.cuFuncGetAttribute, c.byref(val)) == 0 and val.value == binary_sm
    assert client.kernel(api.cuKernelGetAttribute, c.byref(val)) == 0 and val.value == binary_sm


def run(proxy, directory):
    api = c.CDLL(str(proxy))
    launched = "VOLTA_FOREVER" in os.environ
    override = os.environ.get("VF_CAPABILITY")
    given = int(override.replace(".", "")) if override else None
    for name, sm, binary in (
        ("libcudart.so.12", given or 70, given or 70),
        ("libcudart.so.13", given or 75, max(70, given or 75)),
    ):
        probe(api, c.CDLL(str(directory / name)), sm, binary)
    # Reopen the same name after dlclose, with different embedded code at its usual load address.
    previous_address = None
    for name, sm, binary in (
        ("plain", given or (75 if launched else 70), 70 if launched else given or 70),
        ("volta", 70, 70),
        ("turing", given or 75, max(70, given or 75)),
        ("plain", given or (75 if launched else 70), 70 if launched else given or 70),
    ):
        path = directory / "reloaded.so"
        shutil.copyfile(directory / name, path)
        client = c.CDLL(str(path))
        address = c.cast(client.query, c.c_void_p).value
        assert previous_address in (None, address), "the loader must reuse the address to exercise the stale-cache case"
        previous_address = address
        probe(api, client, sm, binary)
        _ctypes.dlclose(client._handle)
    # The loader can still identify this mapping after unlink, but VF cannot inspect its code.
    path = directory / "unreadable.so"
    shutil.copyfile(directory / "plain", path)
    client = c.CDLL(str(path))
    path.unlink()
    probe(api, client, given or 70, given or 70)
    _ctypes.dlclose(client._handle)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python3 tests/proxy/driver_capability.py LIBCUDA (the layer's, as dist/lib/libcuda.so.1)")
    proxy = Path(sys.argv[1]).resolve(strict=True)
    if len(sys.argv) == 3:
        run(proxy, Path(sys.argv[2]))
    else:
        data = Path(__file__).resolve().parents[2] / "crates/vf-fatbin/tests/data"
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name, source in (("mock.so", DRIVER), ("plain", CLIENT)):
                subprocess.run(["cc", "-shared", "-fPIC", "-O0", "-Wl,-Bsymbolic", "-x", "c", "-",
                                "-o", str(directory / name)], input=source, text=True, check=True)
            for name in ("libcudart.so.12", "libcudart.so.13"):
                shutil.copyfile(directory / "plain", directory / name)
            for name, fixture in (("volta", "elf70-elf75-ptx70-lz4"), ("turing", "elf75-ptx75-zstd")):
                subprocess.run(["objcopy", "--add-section", f".nv_fatbin={data / fixture}.fatbin",
                                directory / "plain", directory / name], check=True)
            env = {k: v for k, v in os.environ.items() if k not in ("VOLTA_FOREVER", "VF_CAPABILITY")}
            env.update(VF_REAL_LIBCUDA=str(directory / "mock.so"), VF_LOG_LEVEL="0")
            for launched, override in ((True, None), (True, "7.0"), (True, "8.0"), (True, "12.0"), (False, None),
                                       (False, "8.0")):
                settings = {"VOLTA_FOREVER": "test"} if launched else {}
                if override:
                    settings["VF_CAPABILITY"] = override
                subprocess.run([sys.executable, __file__, proxy, directory], env={**env, **settings}, check=True)
        print("PASS direct, native and CUDA runtime callers, mixed devices, kernel binaries and reloaded DSOs")
