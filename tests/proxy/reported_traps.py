"""CPU regression for the reported traps: which kernels that may reach a multimem instruction the
layer gives, when it refuses them instead, and what it says.

The layer replaces each instruction of the multimem family by a reported trap, and gives the kernels
that may reach one, only where no GPU of the process can take part in multicast: NVIDIA's driver
says CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED is 0 for each (crates/vf-proxy/src/images.rs,
`without_multicast`). A stand-in driver (link_mock_driver.c, with the device functions below) says
what each case needs:

- two GPUs without multicast, the second with it, an attribute that cannot be read, and
  VF_REPORTED_TRAPS=0: a kernel of NCCL's name given (one line for its image at the default level,
  a line for the kernel from level 2, nothing about NCCL's CUDA 12 build) or refused (with that
  line);
- a large image (16 MiB of PTX or more) given with reported traps, which the stand-in takes a second
  to load, as the driver takes minutes to compile one: the layer says what is being compiled before
  the load begins, once in the process, and names NCCL's CUDA 12 build only for kernels of NCCL's
  collectives.

Run with python3 tests/proxy/reported_traps.py path/to/libcuda.so.1.
"""
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile

DEVICES = r"""
#include <stdlib.h>
#include <string.h>
/* MOCK_MULTICAST: one character for each device, 1 for a device with multicast; MOCK_ATTRIBUTE_FAILS:
 * the attribute cannot be read. */
int cuDeviceGetCount(int *count) {
    const char *flags = getenv("MOCK_MULTICAST");
    *count = flags ? (int)strlen(flags) : 1;
    return 0;
}
int cuDeviceGet(int *device, int ordinal) { *device = ordinal; return 0; }
int cuDeviceGetAttribute(int *value, int attribute, int device) {
    const char *flags = getenv("MOCK_MULTICAST");
    if (attribute == 132) {
        if (getenv("MOCK_ATTRIBUTE_FAILS")) return 1;
        *value = flags && flags[device] == '1';
        return 0;
    }
    *value = attribute == 75 ? 7 : 0;
    return 0;
}
"""

NCCL = b"_Z39ncclDevKernel_AllReduce_Sum_f32_RING_LL24ncclDevKernelArgsStorageILm4096EE"
OTHER = b"gather_multicast"


def module(kernel, filler=0):
    """PTX with `kernel`, which calls a function that may read with multimem behind a flag, as NCCL's
    do; and `filler` bytes more of plain kernels."""
    text = (b".version 8.8\n.target sm_90\n.address_size 64\n"
            b".func (.param .b32 r) polled(.param .b64 a)\n{\n .reg .pred %p<2>;\n .reg .b32 %r<3>;\n"
            b" .reg .b64 %rd<2>;\n ld.param.u64 %rd1, [a];\n ld.u32 %r1, [%rd1+8];\n setp.ne.u32 %p1, %r1, 0;\n"
            b" @%p1 multimem.ld_reduce.acquire.sys.global.min.u32 %r2, [%rd1];\n"
            b" @!%p1 ld.volatile.global.u32 %r2, [%rd1];\n st.param.b32 [r], %r2;\n ret;\n}\n"
            b".visible .entry " + kernel + b"(.param .u64 a)\n{\n .reg .b64 %rd<2>;\n .reg .b32 %r<2>;\n"
            b" ld.param.u64 %rd1, [a];\n {\n .param .b64 x;\n .param .b32 y;\n st.param.b64 [x], %rd1;\n"
            b" call (y), polled, (x);\n ld.param.b32 %r1, [y];\n }\n st.u32 [%rd1], %r1;\n ret;\n}\n")
    body = b" add.u32 %r1, %r1, 1;\n" * 4000
    n = 0
    while filler > 0:
        function = b".visible .entry plain_%d(.param .u64 a)\n{\n .reg .b32 %%r<2>;\n" % n + body + b" ret;\n}\n"
        text += function
        filler -= len(function)
        n += 1
    return text


GIVEN = "is given with 1 instruction that cannot run on this GPU replaced by a reported trap: " \
        "multimem.ld_reduce.acquire.sys.global.min.u32 in polled"
IMAGE = "1 kernel is given with instructions that cannot run on this GPU replaced by reported traps"
REFUSED = "cannot run on this GPU and is not given to the program: multimem.ld_reduce.acquire.sys.global.min.u32"
LARGE = "MB of PTX that the layer made of it (its kernels given with reported traps)"
REMEDY = "vf fetch --only libnccl"


def child(proxy, mock, mode):
    api, backend = c.CDLL(proxy), c.CDLL(mock)
    handle, function = c.c_void_p(), c.c_void_p()
    if mode == "lookup":
        assert api.cuModuleLoadData(c.byref(handle), module(NCCL)) == 0
        sys.stderr.flush()
        print(api.cuModuleGetFunction(c.byref(function), handle, NCCL), flush=True)
    elif mode in ("large-nccl", "large-other"):
        name = NCCL if mode == "large-nccl" else OTHER
        image = module(name, 17 << 20)
        for _ in range(2):
            assert api.cuModuleLoadData(c.byref(handle), image) == 0
            print(api.cuModuleGetFunction(c.byref(function), handle, name), flush=True)


def run(proxy, mock, mode, **settings):
    env = dict(os.environ, VF_REAL_LIBCUDA=mock, VF_PROXY_CACHE="0", MOCK_MULTICAST="00")
    for variable in ("VF_LOG_LEVEL", "VF_LOG", "VF_SAFE", "VF_EMULATION_WARNINGS", "VF_REPORTED_TRAPS",
                     "MOCK_LOAD_SLEEP", "MOCK_ATTRIBUTE_FAILS"):
        env.pop(variable, None)
    env.update(settings)
    return subprocess.run([sys.executable, __file__, proxy, mock, mode], env=env, check=True,
                          capture_output=True, text=True)


def check(proxy, mock):
    for name, settings, given in (
        ("two GPUs without multicast", {}, True),
        ("the second GPU with multicast", {"MOCK_MULTICAST": "01"}, False),
        ("an attribute that cannot be read", {"MOCK_ATTRIBUTE_FAILS": "1"}, False),
        ("VF_REPORTED_TRAPS=0", {"VF_REPORTED_TRAPS": "0"}, False),
    ):
        done = run(proxy, mock, "lookup", **settings)
        status = int(done.stdout.split()[-1])
        if given:
            assert status == 0, (name, done.stdout, done.stderr)
            # One line for the image at the default level, which names the kernel; no line for the
            # kernel itself (from level 2), nothing about NCCL's CUDA 12 build.
            assert done.stderr.count(IMAGE) == 1 and NCCL.decode() in done.stderr, (name, done.stderr)
            assert GIVEN not in done.stderr and REMEDY not in done.stderr, (name, done.stderr)
            detailed = run(proxy, mock, "lookup", VF_LOG_LEVEL="2", **settings)
            assert detailed.stderr.count(GIVEN) == 1, (name, detailed.stderr)
        else:
            assert status == 218, (name, done.stdout, done.stderr)
            assert REFUSED in done.stderr and "reported trap" not in done.stderr, (name, done.stderr)
            # A refused kernel of NCCL's still brings the remedy.
            assert REMEDY in done.stderr, (name, done.stderr)
        print(f"PASS {name}: {'given with a reported trap' if given else 'refused'}")

    # A large image: the line comes before the driver's load begins, once in the process.
    for mode in ("large-nccl", "large-other"):
        done = run(proxy, mock, mode, MOCK_LOAD_SLEEP="1")
        err = done.stderr
        assert err.count(LARGE) == 1, (mode, err[:3000])
        assert err.index(LARGE) < err.index("mock: cuModuleLoadData begins"), (mode, err[:3000])
        assert err.count("mock: cuModuleLoadData begins") == 2, (mode, err[:3000])
        assert done.stdout.split() == ["0", "0"], (mode, done.stdout)
        assert (REMEDY in err) == (mode == "large-nccl"), (mode, err[:3000])
        print(f"PASS a large image ({mode}): said before the driver's load, once")


if __name__ == "__main__":
    if len(sys.argv) == 4:
        child(*sys.argv[1:])
        sys.exit(0)
    with tempfile.TemporaryDirectory() as tmp:
        mock = str(Path(tmp) / "driver.so")
        devices = Path(tmp) / "devices.c"
        devices.write_text(DEVICES)
        subprocess.run(["cc", "-shared", "-fPIC", "-o", mock, Path(__file__).with_name("link_mock_driver.c"), devices],
                       check=True)
        check(sys.argv[1], mock)
