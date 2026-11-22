"""CPU regression for what the layer says when it refuses a kernel of NCCL.

The CUDA 13 build of NCCL cannot run its collectives on a V100; when the layer refuses one of its
kernels, it says once, at the default log level, that NCCL's CUDA 12 build is the remedy and how
to fetch it. A stand-in driver (link_mock_driver.c) loads the modules.

Run with python3 tests/proxy/nccl_remedy.py path/to/libcuda.so.1.
"""
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile

NCCL = (b"_Z39ncclDevKernel_AllReduce_Sum_f32_RING_LL24ncclDevKernelArgsStorageILm4096EE",
        b"_Z22ncclDevKernel_SendRecv24ncclDevKernelArgsStorageILm4096EE")
REMEDY = "vf fetch --only libnccl"


def entry(name):
    return b".visible .entry " + name + b"()\n{\n barrier.cluster.arrive;\n ret;\n}\n"


def check(proxy):
    api = c.CDLL(proxy)
    ptx = b".version 8.8\n.target sm_75\n.address_size 64\n" + entry(b"bad") + b"".join(map(entry, NCCL))
    module, function = c.c_void_p(), c.c_void_p()
    assert api.cuModuleLoadData(c.byref(module), ptx) == 0
    sys.stderr.flush()
    for name in (b"bad", NCCL[0], NCCL[0], NCCL[1]):
        assert api.cuModuleGetFunction(c.byref(function), module, name) == 218, name


if __name__ == "__main__":
    if len(sys.argv) == 3:
        check(sys.argv[1])
    else:
        with tempfile.TemporaryDirectory() as tmp:
            mock = str(Path(tmp) / "driver.so")
            subprocess.run(["cc", "-shared", "-fPIC", "-o", mock, Path(__file__).with_name("link_mock_driver.c")],
                           check=True)
            env = dict(os.environ, VF_REAL_LIBCUDA=mock, VF_PROXY_CACHE="0")
            env.pop("VF_LOG_LEVEL", None)
            env.pop("VF_LOG", None)
            run = subprocess.run([sys.executable, __file__, sys.argv[1], mock], env=env, check=True,
                                 stderr=subprocess.PIPE, text=True)
            lines = run.stderr.splitlines()
            said = [i for i, line in enumerate(lines) if REMEDY in line]
            refused = [i for i, line in enumerate(lines) if "cannot run on this GPU" in line]
            assert len(said) == 1, run.stderr
            assert NCCL[0].decode() in lines[said[0]], lines[said[0]]
            # Not for the kernel that is not NCCL's, and right after the first of NCCL's.
            assert len(refused) == 3 and "kernel bad " in lines[refused[0]] and said[0] == refused[1] + 1, run.stderr
            print("PASS the remedy for NCCL's kernels, said once")
