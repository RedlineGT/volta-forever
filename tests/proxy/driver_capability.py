"""Check CUDA device and kernel capability queries through callers with different GPU images."""
import _ctypes
import ctypes as c
import os
from pathlib import Path
import shutil
import struct
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


def fat_binary(entries):
    """A fat binary of (header, stored payload) entries."""
    body = b"".join(header + stored for header, stored in entries)
    return struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(body)) + body


def entries(image):
    """The (header, stored payload) entries of a fat binary."""
    size, offset, found = struct.unpack_from("<Q", image, 8)[0], 16, []
    while offset < 16 + size:
        header_size = struct.unpack_from("<I", image, offset + 4)[0]
        stored_size = struct.unpack_from("<Q", image, offset + 8)[0]
        found.append((image[offset:offset + header_size], image[offset + header_size:offset + header_size + stored_size]))
        offset += header_size + stored_size
    return found


def machine_code(header, cubin, sm):
    """An entry of machine code for `sm`, stored as it is, with the other fields of `header`."""
    stored = cubin + bytes(-len(cubin) % 8)
    header = bytearray(header)
    struct.pack_into("<QI", header, 8, len(stored), 0)
    struct.pack_into("<I", header, 0x1C, sm)
    flags = struct.unpack_from("<Q", header, 0x28)[0] & ~0xB000  # not compressed
    struct.pack_into("<Q", header, 0x28, flags)
    struct.pack_into("<Q", header, 0x38, 0)
    return bytes(header), stored


def limited(given, code_sm):
    """What a caller whose code runs from machine code for `code_sm` is told when VF_CAPABILITY is
    `given`: no later generation than that code's."""
    return code_sm if given and code_sm and given // 10 > code_sm // 10 else given


def asked():
    """What VF_CAPABILITY asks for, as an SM number: None when it is not set, and when it is not a
    capability written <major>.<minor> of 7.0 or more (said, and not used)."""
    major, dot, minor = os.environ.get("VF_CAPABILITY", "").strip().partition(".")
    if not (major.isdigit() and dot and minor.isdigit() and len(minor) == 1):
        return None
    sm = int(major) * 10 + int(minor)
    return sm if sm >= 70 else None


def run(proxy, directory):
    api = c.CDLL(str(proxy))
    launched = "VOLTA_FOREVER" in os.environ
    given = asked()
    # Machine code only, as PyTorch's (from sm_75) and SGLang's kernels (from sm_80): told no later
    # generation than the code that runs, whatever VF_CAPABILITY asks.
    for name, code_sm in (("turing_only", 75), ("ampere_only", 80)):
        sm = limited(given, code_sm) or code_sm
        probe(api, c.CDLL(str(directory / name)), sm, sm)
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


def run_runtimes(proxy, directory):
    """A CUDA runtime answers for the code of the libraries loaded with it (DT_NEEDED)."""
    api = c.CDLL(str(proxy))
    given = asked()
    for runtime, user, code_sm in (("libcudart.so.13", "user_turing.so", 75), ("libcudart.so.14", "user_ptx.so", None),
                                   ("libcudart.so.15", "user_ampere.so", 80)):
        c.CDLL(str(directory / "runtimes" / user))  # and the runtime it needs
        sm = limited(given, code_sm) or 75
        probe(api, c.CDLL(str(directory / "runtimes" / runtime)), sm, sm)


def build(directory):
    data = Path(__file__).resolve().parents[2] / "crates/vf-fatbin/tests/data"
    cubin80 = Path(__file__).resolve().parents[2] / "tests/kernels/translate/k_ampere.sm80.cubin"
    for name, source in (("mock.so", DRIVER), ("plain", CLIENT)):
        subprocess.run(["cc", "-shared", "-fPIC", "-O0", "-Wl,-Bsymbolic", "-x", "c", "-",
                        "-o", str(directory / name)], input=source, text=True, check=True)
    for name in ("libcudart.so.12", "libcudart.so.13"):
        shutil.copyfile(directory / "plain", directory / name)
    turing, _ = entries((data / "elf75-ptx75-zstd.fatbin").read_bytes())
    (directory / "turing_only.fatbin").write_bytes(fat_binary([turing]))
    (directory / "ampere_only.fatbin").write_bytes(fat_binary([machine_code(turing[0], cubin80.read_bytes(), 80)]))
    for name, fixture in (("volta", data / "elf70-elf75-ptx70-lz4.fatbin"), ("turing", data / "elf75-ptx75-zstd.fatbin"),
                          ("turing_only", directory / "turing_only.fatbin"),
                          ("ampere_only", directory / "ampere_only.fatbin")):
        subprocess.run(["objcopy", "--add-section", f".nv_fatbin={fixture}", directory / "plain", directory / name],
                       check=True)
    runtimes = directory / "runtimes"
    runtimes.mkdir()
    for runtime, user, fixture in (("libcudart.so.13", "user_turing.so", directory / "turing_only.fatbin"),
                                   ("libcudart.so.14", "user_ptx.so", data / "elf75-ptx75-zstd.fatbin"),
                                   ("libcudart.so.15", "user_ampere.so", directory / "ampere_only.fatbin")):
        subprocess.run(["cc", "-shared", "-fPIC", "-O0", "-Wl,-Bsymbolic", f"-Wl,-soname,{runtime}", "-x", "c", "-",
                        "-o", str(runtimes / runtime)], input=CLIENT, text=True, check=True)
        subprocess.run(["cc", "-shared", "-fPIC", "-x", "c", "-", "-o", str(runtimes / user), "-L", str(runtimes),
                        "-Wl,--no-as-needed", f"-l:{runtime}", "-Wl,-rpath,$ORIGIN"],
                       input="int user(void) { return 0; }\n", text=True, check=True)
        subprocess.run(["objcopy", "--add-section", f".nv_fatbin={fixture}", runtimes / user], check=True)


if __name__ == "__main__":
    proxy = Path(sys.argv[1]).resolve(strict=True)
    if len(sys.argv) == 4 and sys.argv[3] == "--runtimes":
        run_runtimes(proxy, Path(sys.argv[2]))
    elif len(sys.argv) == 3:
        run(proxy, Path(sys.argv[2]))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            build(directory)
            env = {k: v for k, v in os.environ.items() if k not in ("VOLTA_FOREVER", "VF_CAPABILITY")}
            env.update(VF_REAL_LIBCUDA=str(directory / "mock.so"), VF_LOG_LEVEL="0")
            for launched, override in ((True, None), (True, "7.0"), (True, "8.0"), (True, "12.0"), (False, None),
                                       (False, "8.0"), (True, "8"), (True, "sm_80"), (True, "6.1"), (True, "0.0")):
                settings = {"VOLTA_FOREVER": "test"} if launched else {}
                if override:
                    settings["VF_CAPABILITY"] = override
                for mode in ([], ["--runtimes"]):
                    subprocess.run([sys.executable, __file__, proxy, directory, *mode], env={**env, **settings},
                                   check=True)
            # A value that is not used is said once, at the default log level; one that is used is not.
            for override, said in (("8", "VF_CAPABILITY=8 is not a compute capability"),
                                   ("6.1", "VF_CAPABILITY=6.1 is below the 7.0 of the GPU"), ("8.0", None)):
                settings = {"VOLTA_FOREVER": "test", "VF_CAPABILITY": override, "VF_LOG_LEVEL": "1"}
                log = subprocess.run([sys.executable, __file__, proxy, directory], env={**env, **settings},
                                     check=True, capture_output=True, text=True).stderr
                lines = [line for line in log.splitlines() if "is not used" in line]
                if said:
                    assert len(lines) == 1 and lines[0].startswith(f"[volta-forever] {said}"), log
                else:
                    assert not lines, log
        print("PASS direct, native and CUDA runtime callers, mixed devices, kernel binaries, reloaded DSOs, "
              "machine code that limits what its callers are told, and values of VF_CAPABILITY that are not used")
