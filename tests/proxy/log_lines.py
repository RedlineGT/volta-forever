"""CPU regression for what the layer's lines say of the images it loads.

From VF_LOG_LEVEL=2, a line about an image that a program loads names it: by the file's name for
a load from a file (cuModuleLoad), by its size and XXH64 digest for a load from memory
(cuModuleLoadData, cuLibraryLoadData), so that two images loaded for kernels of one name, two
specializations of a kernel, can be told apart. At the default level these lines are not
printed. A stand-in driver (link_mock_driver.c) loads the modules.

Run with python3 tests/proxy/log_lines.py path/to/libcuda.so.1.
"""
import ctypes as c
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

HEADER = b".version 8.8\n.target sm_75\n.address_size 64\n"
# Two kernels of one name, of different code (two specializations).
FIRST = HEADER + b".visible .entry k()\n{\n ret;\n}\n"
SECOND = HEADER + b".visible .entry k()\n{\n .reg .b32 %r<2>;\n mov.b32 %r1, 1;\n ret;\n}\n"


def xxh64(data, seed=0):
    """XXH64 of `data` (the specification of xxHash), for the digests the layer prints."""
    p1, p2, p3, p4, p5 = (0x9E3779B185EBCA87, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9,
                          0x85EBCA77C2B2AE63, 0x27D4EB2F165667C5)
    mask = (1 << 64) - 1
    rotl = lambda x, r: ((x << r) | (x >> (64 - r))) & mask
    word = lambda i: int.from_bytes(data[i:i + 8], "little")

    def round_(acc, lane):
        return (rotl((acc + lane * p2) & mask, 31) * p1) & mask

    def merge(acc, v):
        return ((acc ^ round_(0, v)) * p1 + p4) & mask
    n, i = len(data), 0
    if n >= 32:
        v = [(seed + p1 + p2) & mask, (seed + p2) & mask, seed, (seed - p1) & mask]
        while i + 32 <= n:
            v = [round_(v[j], word(i + 8 * j)) for j in range(4)]
            i += 32
        h = (rotl(v[0], 1) + rotl(v[1], 7) + rotl(v[2], 12) + rotl(v[3], 18)) & mask
        for lane in v:
            h = merge(h, lane)
    else:
        h = (seed + p5) & mask
    h = (h + n) & mask
    while i + 8 <= n:
        h = (rotl(h ^ round_(0, word(i)), 27) * p1 + p4) & mask
        i += 8
    if i + 4 <= n:
        h = (rotl(h ^ (int.from_bytes(data[i:i + 4], "little") * p1 & mask), 23) * p2 + p3) & mask
        i += 4
    while i < n:
        h = (rotl(h ^ (data[i] * p5 & mask), 11) * p1) & mask
        i += 1
    h = ((h ^ (h >> 33)) * p2) & mask
    h = ((h ^ (h >> 29)) * p3) & mask
    return h ^ (h >> 32)


def loads(proxy, path):
    api = c.CDLL(proxy)
    module, library = c.c_void_p(), c.c_void_p()
    sys.stderr.flush()
    for image in (FIRST, SECOND):
        assert api.cuModuleLoadData(c.byref(module), image) == 0
    assert api.cuModuleLoad(c.byref(module), path.encode()) == 0
    assert api.cuLibraryLoadData(c.byref(library), SECOND, None, None, 0, None, None, 0) == 0


if __name__ == "__main__":
    if len(sys.argv) == 4:
        loads(sys.argv[1], sys.argv[3])
        sys.exit(0)
    assert xxh64(b"") == 0xEF46DB3751D8E999 and xxh64(b"abc") == 0x44BC2CF5AD770999
    assert xxh64(b"Nobody inspects the spammish repetition") == 0xFBCEA83C8A378BF1
    with tempfile.TemporaryDirectory() as tmp:
        mock = str(Path(tmp) / "driver.so")
        subprocess.run(["cc", "-shared", "-fPIC", "-o", mock, Path(__file__).with_name("link_mock_driver.c")],
                       check=True)
        path = Path(tmp) / "kernel-first.ptx"
        path.write_bytes(FIRST)
        env = dict(os.environ, VF_REAL_LIBCUDA=mock, VF_PROXY_CACHE="0")
        env.pop("VF_LOG", None)
        said = {}
        for level in ("1", "2"):
            run = subprocess.run([sys.executable, __file__, sys.argv[1], mock, str(path)],
                                 env=dict(env, VF_LOG_LEVEL=level), check=True, stderr=subprocess.PIPE, text=True)
            said[level] = [line for line in run.stderr.splitlines() if "without sm_70 code" in line]
        lines = said["2"]
        named = [re.search(r"\] (cu\w+) of (\d+) bytes, XXH64 ([0-9a-f]{16}): PTX text without", line)
                 for line in lines]
        memory = [m.groups() for m in named if m]
        assert memory == [("cuModuleLoadData", str(len(FIRST)), f"{xxh64(FIRST):016x}"),
                          ("cuModuleLoadData", str(len(SECOND)), f"{xxh64(SECOND):016x}"),
                          ("cuLibraryLoadData", str(len(SECOND)), f"{xxh64(SECOND):016x}")], "\n".join(lines)
        assert sum(f"] cuModuleLoad of {path}: PTX text without" in line for line in lines) == 1, "\n".join(lines)
        assert len(lines) == 4, "\n".join(lines)
        print("PASS each image named at level 2: the file, or its size and digest")
        assert said["1"] == [], "\n".join(said["1"])
        print("PASS nothing of it at the default level")
