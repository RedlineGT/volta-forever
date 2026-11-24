"""CPU regression for what the layer's lines say of the images it loads and of their kernels.

From VF_LOG_LEVEL=2, a line about an image that a program loads names it: by the file's name for
a load from a file (cuModuleLoad), by its size and XXH64 digest for a load from memory
(cuModuleLoadData, cuLibraryLoadData), so that two images loaded for kernels of one name, two
specializations of a kernel, can be told apart. At the default level these lines are not
printed.

At the default level, "emulated in NAME" and "kernel NAME cannot run" are said once for each
kernel: a kernel with emulated instructions is its name and its code, a refused kernel its name
and why it is refused. A second kernel of a name, of other code or refused otherwise, is said too
(from level 2 with its image's digest, which the line of its load printed); the same image loaded
again is not said again. Past
VF_EMULATION_WARNINGS kernels with emulated instructions, and as many refusals of a name said
before, a line says that the next are not listed. What is kept of the kernels said is bounded:
past 4096 kernels of one name from as many images, the next are neither listed nor counted, and a
line says so. A stand-in driver (link_mock_driver.c) loads the modules.

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


def entry(name, body):
    return HEADER + b".visible .entry " + name + b"()\n{\n .reg .f32 %f<3>;\n" + body + b" ret;\n}\n"


# A kernel with an emulated instruction (tanh.approx.f32 has no Volta instruction), in four images of
# different code; and one that is refused, in four images: two for barrier.cluster.arrive (of
# different code, which becomes the same trap), one for barrier.cluster.wait, one for
# fence.proxy.async (none has a Volta sequence).
EMULATED = [entry(b"k", b" tanh.approx.f32 %f1, %f2;\n" + b" add.f32 %f1, %f1, %f2;\n" * i) for i in range(4)]
REFUSED = [entry(b"r", b" barrier.cluster.arrive;\n"), entry(b"r", b" barrier.cluster.arrive;\n add.f32 %f1, %f1, %f2;\n"),
           entry(b"r", b" barrier.cluster.wait;\n"), entry(b"r", b" fence.proxy.async;\n")]
WHY = ["barrier.cluster.arrive", "barrier.cluster.arrive", "barrier.cluster.wait", "fence.proxy.async"]


def kernels(proxy):
    api = c.CDLL(proxy)
    module, function = c.c_void_p(), c.c_void_p()
    sys.stderr.flush()
    # The first image twice (two modules of the same code), then the others.
    for image in [EMULATED[0]] + EMULATED:
        assert api.cuModuleLoadData(c.byref(module), image) == 0
        assert api.cuModuleGetFunction(c.byref(function), module, b"k") == 0
        assert api.cuModuleGetFunction(c.byref(function), module, b"k") == 0
    for image in [REFUSED[0]] + REFUSED:
        assert api.cuModuleLoadData(c.byref(module), image) == 0
        assert api.cuModuleGetFunction(c.byref(function), module, b"r") == 218


# More kernels of one name, each of its own code, than the layer keeps of the kernels it said.
MANY = 4096 + 50


def many(proxy):
    api = c.CDLL(proxy)
    module, function = c.c_void_p(), c.c_void_p()
    for i in range(MANY):
        image = entry(b"k", b" tanh.approx.f32 %%f1, %%f2;\n add.f32 %%f1, %%f1, 0f%08X;\n" % (0x3F800000 + i))
        assert api.cuModuleLoadData(c.byref(module), image) == 0
        assert api.cuModuleGetFunction(c.byref(function), module, b"k") == 0


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
        {"kernels": kernels, "many": many}.get(sys.argv[3], lambda proxy: loads(proxy, sys.argv[3]))(sys.argv[1])
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

        def kernel_lines(limit, level="1"):
            settings = {"VF_LOG_LEVEL": level, **({"VF_EMULATION_WARNINGS": limit} if limit else {})}
            run = subprocess.run([sys.executable, __file__, sys.argv[1], mock, "kernels"],
                                 env=dict(env, **settings), check=True, stderr=subprocess.PIPE, text=True)
            return [line.split("] ", 1)[1] for line in run.stderr.splitlines()
                    if "emulated in " in line or "kernel r " in line or "further kernels" in line
                    or "not said any more" in line]
        image = lambda data: f", image XXH64 {xxh64(data):016x}"
        # At level 1 no digest is taken: another kernel of a name is said as such; at level 2, with the
        # digest of its image that the line of its load printed.
        for level, shown in (("1", lambda data: ""), ("2", image)):
            lines = kernel_lines(None, level)
            assert len(lines) == 7, "\n".join(lines)
            assert lines[0] == "  emulated in k: 1 tanh", lines[0]
            for i in (1, 2, 3):
                assert lines[i] == f"  emulated in k (another kernel of this name{shown(EMULATED[i])}): 1 tanh", lines[i]
            assert lines[4].startswith("kernel r cannot run on this GPU and is not given to the program: ") and \
                WHY[0] in lines[4] and "another" not in lines[4], lines[4]
            # The second image of barrier.cluster.arrive is refused as the first: not said again.
            for i, line in zip((2, 3), lines[5:], strict=True):
                assert line.startswith("kernel r cannot run") and WHY[i] in line and \
                    line.endswith(f"(another kernel of this name{shown(REFUSED[i])})"), line
        print("PASS each kernel of a name said once: by its code, or for a refusal by its reason")
        lines = kernel_lines("1")
        assert len(lines) == 5, "\n".join(lines)
        assert lines[0] == "  emulated in k: 1 tanh" and "counted, not listed" in lines[1], "\n".join(lines)
        assert "another" not in lines[2] and WHY[2] in lines[3] and lines[3].endswith("(another kernel of this name)"), \
            "\n".join(lines)
        assert lines[4].startswith("kernels refused under a name already said are not said any more"), lines[4]
        print("PASS within VF_EMULATION_WARNINGS")

        run = subprocess.run([sys.executable, __file__, sys.argv[1], mock, "many"], env=dict(env, VF_LOG_LEVEL="1"),
                             check=True, stderr=subprocess.PIPE, text=True)
        lines = [line.split("] ", 1)[1] for line in run.stderr.splitlines() if "forwarding to" not in line]
        listed = [line for line in lines if "emulated in k" in line]
        beyond = [line for line in lines if "neither listed nor counted" in line]
        assert len(listed) == 20 and len(beyond) == 1 and "more than 4096 were given" in beyond[0], "\n".join(lines)
        assert lines[-1] == ("4096 kernels with emulated instructions were given to the program, and more that were "
                             "not counted: 4096 tanh"), lines[-1]
        print(f"PASS {MANY} kernels of one name from as many images: 4096 kept, the next neither listed nor counted")
