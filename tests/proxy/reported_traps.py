"""CPU regression for the reported traps: which kernels that may reach a multimem instruction the
layer gives, and when it refuses them instead.

The layer replaces each instruction of the multimem family by a reported trap, and gives the kernels
that may reach one, only where no GPU of the process can take part in multicast: NVIDIA's driver
says CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED is 0 for each (crates/vf-proxy/src/images.rs,
`without_multicast`). A stand-in driver (link_mock_driver.c, with the device functions below) says
what each case needs: two GPUs without multicast; the second with it; an attribute that cannot be
read; and VF_REPORTED_TRAPS=0. A kernel of NCCL's name given so brings no line about NCCL's CUDA 12
build, which is for a refused one.

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
# A kernel that calls a function that may read with multimem, behind a flag, as NCCL's do.
PTX = (b".version 8.8\n.target sm_90\n.address_size 64\n"
       b".func (.param .b32 r) polled(.param .b64 a)\n{\n .reg .pred %p<2>;\n .reg .b32 %r<3>;\n .reg .b64 %rd<2>;\n"
       b" ld.param.u64 %rd1, [a];\n ld.u32 %r1, [%rd1+8];\n setp.ne.u32 %p1, %r1, 0;\n"
       b" @%p1 multimem.ld_reduce.acquire.sys.global.min.u32 %r2, [%rd1];\n"
       b" @!%p1 ld.volatile.global.u32 %r2, [%rd1];\n st.param.b32 [r], %r2;\n ret;\n}\n"
       b".visible .entry " + NCCL + b"(.param .u64 a)\n{\n .reg .b64 %rd<2>;\n .reg .b32 %r<2>;\n"
       b" ld.param.u64 %rd1, [a];\n {\n .param .b64 x;\n .param .b32 y;\n st.param.b64 [x], %rd1;\n"
       b" call (y), polled, (x);\n ld.param.b32 %r1, [y];\n }\n st.u32 [%rd1], %r1;\n ret;\n}\n")
GIVEN = "is given with 1 instruction that cannot run on this GPU replaced by a reported trap: " \
        "multimem.ld_reduce.acquire.sys.global.min.u32 in polled"
REFUSED = "cannot run on this GPU and is not given to the program: multimem.ld_reduce.acquire.sys.global.min.u32"


def lookup(proxy):
    api = c.CDLL(proxy)
    module, function = c.c_void_p(), c.c_void_p()
    assert api.cuModuleLoadData(c.byref(module), PTX) == 0
    sys.stderr.flush()
    print(api.cuModuleGetFunction(c.byref(function), module, NCCL), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        lookup(sys.argv[1])
        sys.exit(0)
    with tempfile.TemporaryDirectory() as tmp:
        mock = str(Path(tmp) / "driver.so")
        devices = Path(tmp) / "devices.c"
        devices.write_text(DEVICES)
        subprocess.run(["cc", "-shared", "-fPIC", "-o", mock, Path(__file__).with_name("link_mock_driver.c"), devices],
                       check=True)
        for name, settings, given in (
            ("two GPUs without multicast", {"MOCK_MULTICAST": "00"}, True),
            ("the second GPU with multicast", {"MOCK_MULTICAST": "01"}, False),
            ("an attribute that cannot be read", {"MOCK_MULTICAST": "00", "MOCK_ATTRIBUTE_FAILS": "1"}, False),
            ("VF_REPORTED_TRAPS=0", {"MOCK_MULTICAST": "00", "VF_REPORTED_TRAPS": "0"}, False),
        ):
            env = dict(os.environ, VF_REAL_LIBCUDA=mock, VF_PROXY_CACHE="0", **settings)
            for variable in ("VF_LOG_LEVEL", "VF_LOG", "VF_SAFE", "VF_EMULATION_WARNINGS"):
                env.pop(variable, None)
            if "VF_REPORTED_TRAPS" not in settings:
                env.pop("VF_REPORTED_TRAPS", None)
            run = subprocess.run([sys.executable, __file__, sys.argv[1], mock], env=env, check=True,
                                 capture_output=True, text=True)
            status = int(run.stdout.split()[-1])
            if given:
                assert status == 0, (name, run.stdout, run.stderr)
                assert run.stderr.count(GIVEN) == 1, (name, run.stderr)
                assert "vf fetch --only libnccl" not in run.stderr, (name, run.stderr)
            else:
                assert status == 218, (name, run.stdout, run.stderr)
                assert REFUSED in run.stderr and "reported trap" not in run.stderr, (name, run.stderr)
                # A refused kernel of NCCL's still brings the remedy.
                assert "vf fetch --only libnccl" in run.stderr, (name, run.stderr)
            print(f"PASS {name}: {'given with a reported trap' if given else 'refused'}")
