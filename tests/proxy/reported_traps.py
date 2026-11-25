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
- what the driver answered: said from level 2 when it is asked; at the default level, nothing for
  a program without a multimem instruction whatever the answer, nor for one whose kernels are
  given; for one refused because of the answer (a GPU with multicast, an attribute that cannot be
  read), the answer once, after the refusal;
- a large image (16 MiB of PTX or more) given with reported traps, which the stand-in takes a second
  to load, as the driver takes minutes to compile one: the layer says what is being compiled before
  the load begins, once in the process, and names NCCL's CUDA 12 build only for kernels of NCCL's
  collectives;
- a kernel linked by the driver's linker from PTX with a multimem instruction, whose cubin is then
  given as the relocatable fat binary of a wrapper of the CUDA runtime: the line of its image names
  it (until 2026-11-25 what the link knew of it was dropped there).

Run with python3 tests/proxy/reported_traps.py path/to/libcuda.so.1.
"""
import ctypes as c
import os
from pathlib import Path
import struct
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
    static int asked;
    if (attribute == 132) {
        if (getenv("MOCK_ATTRIBUTE_FAILS")) return 1;
        if (getenv("MOCK_ATTRIBUTE_FAILS_ONCE") && asked++ == 0) return 1;
        *value = flags && flags[device] == '1';
        return 0;
    }
    *value = attribute == 75 ? 7 : 0;
    return 0;
}
"""

NCCL = b"_Z39ncclDevKernel_AllReduce_Sum_f32_RING_LL24ncclDevKernelArgsStorageILm4096EE"
OTHER = b"gather_multicast"


PLAIN = (b".version 8.8\n.target sm_90\n.address_size 64\n"
         b".visible .entry plain(.param .u64 a)\n{\n .reg .b64 %rd<2>;\n ld.param.u64 %rd1, [a];\n"
         b" st.u32 [%rd1], 1;\n ret;\n}\n")


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
MULTICAST = "take part in multicast"
NO_GPU = "no GPU of this process can take part in multicast (2 asked): instructions that need a multicast " \
         "address are replaced by reported traps"
SECOND = "GPU 1 of this process can take part in multicast (CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED): the " \
         "kernels that may reach a multimem instruction are refused, not given with reported traps"
UNREAD = "the driver did not say whether GPU 0 can take part in multicast: the kernels that may reach a " \
         "multimem instruction are refused, not given with reported traps"
WRAPPER_MAGIC = 0x466243B1
CUBIN = Path(__file__).resolve().parents[1] / "kernels" / "k_basic.128.sm70.cubin"


class Wrapper(c.Structure):
    _fields_ = [("magic", c.c_uint), ("version", c.c_uint), ("data", c.c_void_p), ("extra", c.c_void_p)]


def fat_binary(cubin, sm=70):
    """`cubin` as the only entry of a fat binary, for `sm` (as link_refusals.py makes it)."""
    native = (Path(__file__).resolve().parents[2] / "crates/vf-fatbin/tests/data/elf70-elf75-ptx70-lz4.fatbin")
    header = bytearray(native.read_bytes()[16:16 + 64])
    stored = cubin + bytes(-len(cubin) % 8)
    struct.pack_into("<QI", header, 8, len(stored), 0)
    struct.pack_into("<Q", header, 0x28, struct.unpack_from("<Q", header, 0x28)[0] & ~0xB000)
    struct.pack_into("<Q", header, 0x38, 0)
    struct.pack_into("<I", header, 0x1C, sm)
    body = bytes(header) + stored
    return struct.pack("<IHHQ", 0xBA55ED50, 1, 16, len(body)) + body


def child(proxy, mock, mode):
    api, backend = c.CDLL(proxy), c.CDLL(mock)
    handle, function = c.c_void_p(), c.c_void_p()
    if mode == "plain":
        assert api.cuModuleLoadData(c.byref(handle), PLAIN) == 0
        print(api.cuModuleGetFunction(c.byref(function), handle, b"plain"), flush=True)
    elif mode in ("lookup", "lookup-twice"):
        for _ in range(1 if mode == "lookup" else 2):
            assert api.cuModuleLoadData(c.byref(handle), module(NCCL)) == 0
            sys.stderr.flush()
            print(api.cuModuleGetFunction(c.byref(function), handle, NCCL), flush=True)
    elif mode in ("large-nccl", "large-other"):
        name = NCCL if mode == "large-nccl" else OTHER
        image = module(name, 17 << 20)
        for _ in range(2):
            assert api.cuModuleLoadData(c.byref(handle), image) == 0
            print(api.cuModuleGetFunction(c.byref(function), handle, name), flush=True)
    elif mode == "large-link":
        # The program links the large PTX itself: the driver compiles it in cuLinkComplete.
        output = CUBIN.read_bytes()
        linked = c.create_string_buffer(output, len(output))
        c.c_void_p.in_dll(backend, "linked").value = c.addressof(linked)
        c.c_size_t.in_dll(backend, "linked_size").value = len(output)
        state, image, size = c.c_void_p(), c.c_void_p(), c.c_size_t()
        ptx = module(NCCL, 17 << 20)
        assert api.cuLinkCreate_v2(0, None, None, c.byref(state)) == 0
        assert api.cuLinkAddData_v2(state, 1, ptx, len(ptx), b"large.ptx", 0, None, None) == 0
        assert api.cuLinkComplete(state, c.byref(image), c.byref(size)) == 0
        print(0, flush=True)
    elif mode == "linked":
        output = CUBIN.read_bytes()
        linked = c.create_string_buffer(output, len(output))
        c.c_void_p.in_dll(backend, "linked").value = c.addressof(linked)
        c.c_size_t.in_dll(backend, "linked_size").value = len(output)
        state, image, size = c.c_void_p(), c.c_void_p(), c.c_size_t()
        ptx = module(OTHER)
        assert api.cuLinkCreate_v2(0, None, None, c.byref(state)) == 0
        assert api.cuLinkAddData_v2(state, 1, ptx, len(ptx), b"multicast.ptx", 0, None, None) == 0
        assert api.cuLinkComplete(state, c.byref(image), c.byref(size)) == 0
        # The cubin that the link made, as the relocatable fat binary of a version-2 wrapper whose
        # first fat binary has nothing for this GPU (an entry said to be for sm_90).
        cubin = c.string_at(image, size.value)
        member = c.create_string_buffer(fat_binary(cubin))
        first = c.create_string_buffer(fat_binary(cubin, 90))
        members = (c.c_void_p * 2)(c.addressof(member), None)
        wrapper = Wrapper(WRAPPER_MAGIC, 2, c.addressof(first), c.addressof(members))
        sys.stderr.write("wrapper load\n")
        sys.stderr.flush()
        assert api.cuModuleLoadFatBinary(c.byref(handle), c.byref(wrapper)) == 0
        print(api.cuModuleGetFunction(c.byref(function), handle, OTHER), flush=True)


def run(proxy, mock, mode, **settings):
    env = dict(os.environ, VF_REAL_LIBCUDA=mock, VF_PROXY_CACHE="0", MOCK_MULTICAST="00")
    for variable in ("VF_LOG_LEVEL", "VF_LOG", "VF_SAFE", "VF_EMULATION_WARNINGS", "VF_REPORTED_TRAPS",
                     "MOCK_LOAD_SLEEP", "MOCK_ATTRIBUTE_FAILS", "MOCK_ATTRIBUTE_FAILS_ONCE"):
        env.pop(variable, None)
    env.update(settings)
    return subprocess.run([sys.executable, __file__, proxy, mock, mode], env=env, check=True,
                          capture_output=True, text=True)


def check(proxy, mock):
    for name, settings, given, answer in (
        ("two GPUs without multicast", {}, True, NO_GPU),
        ("the second GPU with multicast", {"MOCK_MULTICAST": "01"}, False, SECOND),
        ("an attribute that cannot be read", {"MOCK_ATTRIBUTE_FAILS": "1"}, False, UNREAD),
        ("VF_REPORTED_TRAPS=0", {"VF_REPORTED_TRAPS": "0"}, False, None),
    ):
        done = run(proxy, mock, "lookup", **settings)
        detailed = run(proxy, mock, "lookup", VF_LOG_LEVEL="2", **settings)
        status = int(done.stdout.split()[-1])
        if given:
            assert status == 0, (name, done.stdout, done.stderr)
            # One line for the image at the default level, which names the kernel and says why; no
            # line for the kernel itself (from level 2), nothing about NCCL's CUDA 12 build, nothing
            # of the driver's answer (from level 2).
            assert done.stderr.count(IMAGE) == 1 and NCCL.decode() in done.stderr, (name, done.stderr)
            assert "which no GPU of this process can have" in done.stderr, (name, done.stderr)
            assert GIVEN not in done.stderr and REMEDY not in done.stderr, (name, done.stderr)
            assert MULTICAST not in done.stderr, (name, done.stderr)
            assert detailed.stderr.count(GIVEN) == 1, (name, detailed.stderr)
        else:
            assert status == 218, (name, done.stdout, done.stderr)
            # (Of reported traps, only the answer says something: they are not used.)
            rest = done.stderr.replace(answer, "") if answer else done.stderr
            assert REFUSED in rest and "reported trap" not in rest, (name, done.stderr)
            # A refused kernel of NCCL's still brings the remedy.
            assert REMEDY in done.stderr, (name, done.stderr)
            # The answer that refused it, once, after the refusal; none with VF_REPORTED_TRAPS=0 (the
            # driver is not asked).
            if answer:
                assert done.stderr.count(answer) == 1, (name, done.stderr)
                assert done.stderr.index(REFUSED) < done.stderr.index(answer), (name, done.stderr)
            else:
                assert MULTICAST not in done.stderr + detailed.stderr, (name, done.stderr, detailed.stderr)
        # From level 2, the answer once, when the driver is asked.
        if answer:
            assert detailed.stderr.count(answer) == 1, (name, detailed.stderr)
        print(f"PASS {name}: {'given with a reported trap' if given else 'refused'}")

    # A program without a multimem instruction: nothing of multicast at the default level, whatever
    # the driver answers; the answer from level 2.
    for name, settings, answer in (
        ("two GPUs without multicast", {}, NO_GPU),
        ("the second GPU with multicast", {"MOCK_MULTICAST": "01"}, SECOND),
        ("an attribute that cannot be read", {"MOCK_ATTRIBUTE_FAILS": "1"}, UNREAD),
    ):
        done = run(proxy, mock, "plain", **settings)
        assert done.stdout.split() == ["0"], (name, done.stdout, done.stderr)
        assert MULTICAST not in done.stderr, (name, done.stderr)
        detailed = run(proxy, mock, "plain", VF_LOG_LEVEL="2", **settings)
        assert detailed.stderr.count(answer) == 1, (name, detailed.stderr)
        print(f"PASS a program without a multimem instruction ({name}): the driver's answer from level 2 only")

    # An answer that cannot be had once is kept for the process: the second load is refused too, as
    # the cache's key and the adaptation say (until 2026-11-25 it was asked again, and given).
    done = run(proxy, mock, "lookup-twice", MOCK_ATTRIBUTE_FAILS_ONCE="1")
    assert done.stdout.split() == ["218", "218"], (done.stdout, done.stderr)
    print("PASS an answer of the driver that failed once: kept for the process")

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
    # Linked by the program: the line comes before the driver's link completes (it compiles there).
    done = run(proxy, mock, "large-link", MOCK_LOAD_SLEEP="1")
    err = done.stderr
    assert err.count(LARGE) == 1 and REMEDY in err, err[:3000]
    assert err.index(LARGE) < err.index("mock: cuLinkComplete begins"), err[:3000]
    print("PASS a large image linked by the program: said before the driver's link completes")

    # What the driver's link knew of its input reaches the image that it made, given as the
    # relocatable fat binary of a wrapper.
    done = run(proxy, mock, "linked")
    after = done.stderr[done.stderr.index("wrapper load"):]
    assert IMAGE in after and OTHER.decode() in after, done.stderr
    assert done.stdout.split()[-1] == "0", done.stdout
    print("PASS a cubin made by the driver's linker, in a wrapper: its kernel said given with a reported trap")


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
