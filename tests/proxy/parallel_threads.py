"""CPU test for VF_THREADS and RLIMIT_NPROC thread-creation refusal fallback.

Under RLIMIT_NPROC=1, a large module that triggers parallel PTX lowering and module analysis must
still be adapted without panicking, falling back to the calling thread: vf-translate adapt must
write the same bytes with VF_THREADS=1, with VF_THREADS=4 and under the limit; against a stand-in
driver (link_mock_driver.c), cuModuleLoadData and cuLibraryLoadData must adapt it and give its
kernel, with no internal error and nothing loaded as is (what the driver is given is not compared).

Run with python3 tests/proxy/parallel_threads.py path/to/libcuda.so.1.
"""
import ctypes as c
import hashlib
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile


def build_large_module():
    header = (
        b".version 8.8\n.target sm_80\n.address_size 64\n"
        b".global .align 4 .u32 g_ro[4] = {1, 2, 3, 4};\n"
        b".global .align 4 .u32 g_rw = 0;\n"
        b".func helper(.param .u32 x)\n{\n .reg .u64 %rd<2>;\n mov.u64 %rd0, g_rw;\n st.global.u32 [%rd0], 7;\n ret;\n}\n"
    )
    pad = b" add.s32 %r1, %r1, 1;\n" * 350
    chunks = [header]
    lower_body_bytes = 0
    for i in range(20):
        extra = (
            b" mov.u64 %rd0, g_ro;\n ld.global.u32 %r0, [%rd0];\n"
            if i % 2 == 0
            else b""
        ) + b" tanh.approx.f32 %f0, %f1;\n"
        body = (
            b" .reg .s32 %r<4>;\n .reg .u64 %rd<2>;\n .reg .f32 %f<4>;\n"
            + pad
            + extra
            + b" ret;\n"
        )
        lower_body_bytes += len(body)
        chunks.append(f".visible .entry k_{i}()\n{{\n".encode() + body + b"}\n")
    ptx = b"".join(chunks)
    assert lower_body_bytes > 131_072
    assert len(ptx) > 131_072
    return ptx


def run_proxy_load(proxy, ptx_bytes):
    api = c.CDLL(proxy)
    mod, lib_h, fn, kern = c.c_void_p(), c.c_void_p(), c.c_void_p(), c.c_void_p()
    assert api.cuModuleLoadData(c.byref(mod), ptx_bytes) == 0, "cuModuleLoadData failed"
    assert api.cuModuleGetFunction(c.byref(fn), mod, b"k_1") == 0, "cuModuleGetFunction failed"
    assert api.cuLibraryLoadData(c.byref(lib_h), ptx_bytes, None, None, 0, None, None, 0) == 0, "cuLibraryLoadData failed"
    assert api.cuLibraryGetKernel(c.byref(kern), lib_h, b"k_1") == 0, "cuLibraryGetKernel failed"


def limit_nproc():
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[2] == "proxy":
        ptx_bytes = Path(sys.argv[3]).read_bytes()
        run_proxy_load(sys.argv[1], ptx_bytes)
        sys.exit(0)

    proxy = Path(sys.argv[1]).resolve()
    vf_translate = proxy.parent.parent / "bin" / "vf-translate"
    ptx = build_large_module()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ptx_path = tmp_path / "large.ptx"
        ptx_path.write_bytes(ptx)

        if vf_translate.is_file():
            out_1 = tmp_path / "out_1.ptx"
            out_4 = tmp_path / "out_4.ptx"
            out_nproc = tmp_path / "out_nproc.ptx"
            subprocess.run([str(vf_translate), "adapt", str(ptx_path), str(out_1)],
                           env=dict(os.environ, VF_THREADS="1"), check=True, stdout=subprocess.DEVNULL)
            subprocess.run([str(vf_translate), "adapt", str(ptx_path), str(out_4)],
                           env=dict(os.environ, VF_THREADS="4"), check=True, stdout=subprocess.DEVNULL)
            subprocess.run([str(vf_translate), "adapt", str(ptx_path), str(out_nproc)],
                           env=dict(os.environ, VF_THREADS="4"), preexec_fn=limit_nproc,
                           check=True, stdout=subprocess.DEVNULL)
            sha1 = hashlib.sha256(out_1.read_bytes()).hexdigest()
            assert hashlib.sha256(out_4.read_bytes()).hexdigest() == sha1
            assert hashlib.sha256(out_nproc.read_bytes()).hexdigest() == sha1
            print("PASS  vf-translate adapt with VF_THREADS=1/4 and RLIMIT_NPROC=1")

        mock = tmp_path / "driver.so"
        subprocess.run(["cc", "-shared", "-fPIC", "-o", str(mock),
                        str(Path(__file__).with_name("link_mock_driver.c"))], check=True)
        env = dict(os.environ, VF_REAL_LIBCUDA=str(mock), VF_PROXY_CACHE="0", VF_LOG_LEVEL="2", VF_THREADS="4")
        env.pop("VF_LOG", None)
        res = subprocess.run(
            [sys.executable, __file__, str(proxy), "proxy", str(ptx_path)],
            env=env,
            preexec_fn=limit_nproc,
            check=True,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert "internal error while adapting" not in res.stderr, res.stderr
        assert "loaded as is" not in res.stderr, res.stderr
        assert "cuModuleLoadData" in res.stderr and "PTX for sm_80 retargeted" in res.stderr, res.stderr
        assert "cuLibraryLoadData" in res.stderr, res.stderr
        print("PASS  cuModuleLoadData + cuLibraryLoadData under RLIMIT_NPROC=1 (calling-thread fallback)")
