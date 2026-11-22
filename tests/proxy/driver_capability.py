"""Check CUDA device and kernel capability queries through callers with different GPU images.

Run with python3 tests/proxy/driver_capability.py path/to/libcuda.so.1.
"""
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
#define _GNU_SOURCE
#include <stddef.h>
#include <string.h>
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
/* The device of the current context (-1: no context), and the .target line of the last PTX given
   to the loading functions ("file" for cuModuleLoad). */
int mock_device = -1;
char mock_seen[64];
static int seen(const char *what, size_t length) {
    const char *line = memmem(what, length, ".target", 7);
    size_t n = 0;
    if (!line) line = what;
    while (line + n < what + length && line[n] != '\\n' && n < sizeof mock_seen - 1) n++;
    memset(mock_seen, 0, sizeof mock_seen);
    memcpy(mock_seen, line, n);
    return 0;
}
int cuCtxGetDevice(int *d) { if (mock_device < 0) return 201; *d = mock_device; return 0; }
int cuModuleLoadData(void **m, const void *image) { *m = (void *)0x2000; return seen(image, strlen(image)); }
int cuModuleLoadDataEx(void **m, const void *image, unsigned n, int *o, void **v) { return cuModuleLoadData(m, image); }
int cuModuleLoad(void **m, const char *path) { *m = (void *)0x2000; return seen("file", 4); }
int cuModuleUnload(void *m) { return 0; }
int cuLinkAddData_v2(void *s, int type, void *data, size_t size, const char *name, unsigned n, int *o, void **v) {
    return seen(data, size);
}
"""

# A normal (non-tail) call makes this DSO the return address seen by the proxy.
CLIENT = """
int query(int (*f)(int *, int, int), int *v, int attr, int dev) { return f(v, attr, dev); }
int both(int (*f)(int *, int *, int), int *a, int *b, int dev) { return f(a, b, dev); }
int binary(int (*f)(int *, int, void *), int *v) { return f(v, 6, (void *)1); }
int kernel(int (*f)(int *, int, void *, int), int *v) { return f(v, 6, (void *)1, 0); }
"""


# A main program that asks the driver itself:   main PROXY [DIRECTORY-TO-ENTER-FIRST]
MAIN = """
#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>
int main(int argc, char **argv) {
    void *proxy = dlopen(argv[1], RTLD_NOW);
    if (!proxy || (argc > 2 && chdir(argv[2]))) return 2;
    int (*query)(int *, int *, int) = (int (*)(int *, int *, int))dlsym(proxy, "cuDeviceComputeCapability");
    int major = -1, minor = -1, status = query(&major, &minor, 0);
    printf("%d.%d %d\\n", major, minor, status);
    return 0;
}
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
    `given`: no later generation than that code's, unless VF_CAPABILITY_ALL=1."""
    if os.environ.get("VF_CAPABILITY_ALL") == "1":
        return given
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
    # cuDNN's graph library, which has no code, chooses among the engines of the library beside it,
    # which it loads later: told what that code allows.
    told = limited(given, 75)
    probe(api, c.CDLL(str(directory / "cudnn/libcudnn_graph.so.9")), told or (75 if launched else 70),
          70 if launched else told or 70)
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
    (directory / "cudnn").mkdir()
    shutil.copyfile(directory / "plain", directory / "cudnn/libcudnn_graph.so.9")
    shutil.copyfile(directory / "turing_only", directory / "cudnn/libcudnn_engines_precompiled.so.9")
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
    (directory / "bin").mkdir()
    (directory / "elsewhere").mkdir()
    subprocess.run(["cc", "-O0", "-x", "c", "-", "-o", str(directory / "main"), "-ldl"], input=MAIN, text=True,
                   check=True)
    for name, fixture in (("main75", data / "elf75-ptx75-zstd.fatbin"),
                          ("main_turing_only", directory / "turing_only.fatbin")):
        subprocess.run(["objcopy", "--add-section", f".nv_fatbin={fixture}", directory / "main",
                        directory / "bin" / name], check=True)


def run_mixed(proxy, directory):
    """In a machine that also has a later card, a module loaded into a context on that card is
    given as the program gave it; one loaded into a context on the V100 is adapted, and so is one
    loaded when the driver does not say which device the context is on."""
    api = c.CDLL(str(proxy))
    mock = c.CDLL(str(directory / "mock.so"))
    device = c.c_int.in_dll(mock, "mock_device")
    ptx_file = Path(__file__).resolve().parents[2] / "tests/kernels/k_basic.135.sm75.ptx"
    ptx = ptx_file.read_bytes()
    module = c.c_void_p()
    loads = {
        "cuModuleLoadData": lambda: api.cuModuleLoadData(c.byref(module), c.c_char_p(ptx)),
        "cuModuleLoadDataEx": lambda: api.cuModuleLoadDataEx(c.byref(module), c.c_char_p(ptx), 0, None, None),
        "cuModuleLoad": lambda: api.cuModuleLoad(c.byref(module), c.c_char_p(bytes(ptx_file))),
        "cuLinkAddData_v2": lambda: api.cuLinkAddData_v2(None, 1, c.c_char_p(ptx), c.c_size_t(len(ptx)), None, 0,
                                                         None, None),
    }
    for ordinal, adapted in ((-1, True), (0, True), (1, False), (3, False)):
        device.value = ordinal
        for name, load in loads.items():
            assert load() == 0, name
            seen = (c.c_char * 64).in_dll(mock, "mock_seen").value
            expected = b".target sm_70" if adapted else b"file" if name == "cuModuleLoad" else b".target sm_75"
            assert seen == expected, (name, ordinal, seen)


def run_main(proxy, directory, env):
    """The main program is told what its code says, however it was started (by its path, by name
    through PATH, before a change of directory)."""
    programs = directory / "bin"
    for name, override, told in (("main75", None, "7.5"), ("main75", "8.0", "8.0"),
                                 ("main_turing_only", "8.0", "7.5"), ("main_turing_only", None, "7.5")):
        settings = {**env, "VOLTA_FOREVER": "test", "PATH": f"{programs}:{os.environ['PATH']}"}
        if override:
            settings["VF_CAPABILITY"] = override
        for command, cwd in (([str(programs / name), str(proxy)], directory), ([name, str(proxy)], directory / "elsewhere"),
                             ([f"./{name}", str(proxy), "/"], programs)):
            out = subprocess.run(command, cwd=cwd, env=settings, check=True, capture_output=True, text=True).stdout
            assert out == f"{told} 0\n", (command, cwd, override, out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python3 tests/proxy/driver_capability.py LIBCUDA (the layer's, as dist/lib/libcuda.so.1)")
    proxy = Path(sys.argv[1]).resolve(strict=True)
    if len(sys.argv) == 4 and sys.argv[3] == "--runtimes":
        run_runtimes(proxy, Path(sys.argv[2]))
    elif len(sys.argv) == 4 and sys.argv[3] == "--mixed":
        run_mixed(proxy, Path(sys.argv[2]))
    elif len(sys.argv) == 3:
        run(proxy, Path(sys.argv[2]))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            build(directory)
            env = {k: v for k, v in os.environ.items() if k not in ("VOLTA_FOREVER", "VF_CAPABILITY")}
            env.update(VF_REAL_LIBCUDA=str(directory / "mock.so"), VF_LOG_LEVEL="0")
            for launched, override in ((True, None), (True, "7.0"), (True, "8.0"), (True, "12.0"), (False, None),
                                       (False, "8.0"), (True, "8"), (True, "sm_80"), (True, "6.1"), (True, "0.0"),
                                       (True, "8.0 all"), (True, "12.0 all")):
                settings = {"VOLTA_FOREVER": "test"} if launched else {}
                if override:
                    settings["VF_CAPABILITY"], _, everyone = override.partition(" ")
                    if everyone:
                        settings["VF_CAPABILITY_ALL"] = "1"
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
            run_main(proxy, directory, env)
            subprocess.run([sys.executable, __file__, proxy, directory, "--mixed"],
                           env={**env, "VOLTA_FOREVER": "test", "VF_PROXY_CACHE": "0"}, check=True)
        print("PASS direct, native and CUDA runtime callers, mixed devices, kernel binaries, reloaded DSOs, "
              "machine code that limits what its callers are told, values of VF_CAPABILITY that are not used, "
              "main programs however they were started, and modules loaded on a later card")
