"""Link, through the layer's nvJitLink, PTX in which one kernel cannot run on Volta (`bad`, which
asks for a cluster of blocks) and one can (`good`), and write the cubin made to OUTPUT; it must name
`bad` in its note (`.note.vf.refused`). run.sh then loads it with NVIDIA's driver and through the
layer (lookup.py).

usage: VF_JITLINK_REAL=.../libnvJitLink.so.12 python3 jitlink-marked.py libnvJitLink.so.13 OUTPUT
"""
import ctypes as C
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "proxy"))
import refusal_note  # noqa: E402

PTX = (b".version 8.8\n.target sm_80\n.address_size 64\n"
       b".visible .entry bad()\n{\n barrier.cluster.arrive;\n ret;\n}\n"
       b".visible .entry good(.param .u64 out)\n{\n .reg .b64 %a;\n .reg .b32 %v;\n"
       b" ld.param.u64 %a, [out];\n mov.u32 %v, 1;\n st.global.u32 [%a], %v;\n ret;\n}\n\0")


def main(adapter, output):
    lib = C.CDLL(adapter)
    handle = C.c_void_p()
    assert lib.nvJitLinkCreate(C.byref(handle), 1, (C.c_char_p * 1)(b"-arch=sm_80")) == 0
    assert lib.nvJitLinkAddData(handle, 2, PTX, C.c_size_t(len(PTX)), b"half.ptx") == 0
    assert lib.nvJitLinkComplete(handle) == 0
    size = C.c_size_t()
    assert lib.nvJitLinkGetLinkedCubinSize(handle, C.byref(size)) == 0
    cubin = C.create_string_buffer(size.value)
    assert lib.nvJitLinkGetLinkedCubin(handle, cubin) == 0
    assert lib.nvJitLinkDestroy(C.byref(handle)) == 0
    note = refusal_note.read(cubin.raw)
    print(f"linked cubin of {size.value} bytes for sm_{cubin.raw[0x30]}, its note: {note}")
    assert cubin.raw[0x30] == 70 and [kernel for kernel, _, _ in note] == ["bad"], note
    Path(output).write_bytes(cubin.raw)
    print("marked cubin written")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
