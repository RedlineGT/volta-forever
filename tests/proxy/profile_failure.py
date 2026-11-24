"""With VF_PROFILE, the first synchronisation after a launch that returns an error is said at once, with
the kernel it followed: a kernel that faults ends the context, the program learns it at a later call,
and the line names the kernel. Without a GPU, against the stand-in for the driver of launch_guard.py
(guard_mock_driver.c), whose synchronisation returns what the test sets.

Run with python3 tests/proxy/profile_failure.py path/to/libcuda.so.1.
"""
import ctypes as c
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent))
from launch_guard import Run  # noqa: E402

PTX = (b".version 8.8\n.target sm_75\n.address_size 64\n"
       b".visible .entry first()\n{\n mov.u32 %r1, 1;\n ret;\n}\n"
       b".visible .entry faulting()\n{\n mov.u32 %r1, 2;\n ret;\n}\n")
LINE = "profiling: the synchronisation after a launch of "


def case(run, log):
    module = run.load(PTX)
    functions = {}
    for name in (b"first", b"faulting"):
        function = c.c_void_p()
        assert run.api.cuModuleGetFunction(c.byref(function), module, name) == 0
        functions[name] = function
    grid = (2, 1, 1, 64, 1, 1, 0, None, None, None)
    assert run.api.cuLaunchKernel(functions[b"first"], *grid) == 0
    assert LINE not in (Path(log).read_text() if Path(log).exists() else "")
    # The kernel faults: its synchronisation, and every one after it, return 719.
    c.c_int.in_dll(run.driver, "sync_result").value = 719
    assert run.api.cuLaunchKernel(functions[b"faulting"], *grid) == 0
    assert run.api.cuLaunchKernel(functions[b"first"], *grid) == 0
    lines = [line for line in Path(log).read_text().splitlines() if LINE in line]
    assert len(lines) == 1, lines
    assert lines[0].endswith(
        "launch of faulting (grid 2x1x1, block 64x1x1, 0 bytes of dynamic shared memory) returned error 719, "
        "the first that one did"), lines[0]


if __name__ == "__main__":
    if len(sys.argv) == 3:
        case(Run(sys.argv[1], sys.argv[2]), os.environ["VF_LOG"])
        sys.exit(0)
    proxy = sys.argv[1]
    with tempfile.TemporaryDirectory() as tmp:
        mock = str(Path(tmp) / "guard.so")
        subprocess.run(["cc", "-shared", "-fPIC", "-Wl,-Bsymbolic", "-o", mock,
                        Path(__file__).with_name("guard_mock_driver.c")], check=True)
        env = dict(os.environ, VF_REAL_LIBCUDA=mock, VF_PROXY_CACHE="0", VF_LOG=str(Path(tmp) / "profile.log"),
                   VF_LOG_LEVEL="1", VF_PROFILE=str(Path(tmp) / "profile"))
        done = subprocess.run([sys.executable, __file__, proxy, mock], env=env)
        print("PASS the first failed synchronisation of the profile names its kernel" if done.returncode == 0
              else "FAIL the first failed synchronisation of the profile")
        sys.exit(done.returncode)
