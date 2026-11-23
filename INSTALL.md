# volta-forever: installation

A compatibility layer that lets CUDA 13 programs (built for sm_75 and later, with no code for
Tesla V100) run on V100s with the 580 driver.

## Requirements

- **GPU**: NVIDIA Tesla V100 (compute capability 7.0)
- **Driver**: NVIDIA 580.178.04 (tested; the last branch with Volta support)
- **OS**: Linux x86_64, glibc 2.34 or later. The installation was run in fresh systems of
  Ubuntu 22.04, 24.04 and 26.04 and of Debian 12 and 13 (2026-11-21; no GPU in them: what was
  seen is that everything installs, that the binaries start, and that the downloads are
  made). RHEL 9 and Fedora 36 or later have the C library it takes; not tried.
- **On a fresh Debian or Ubuntu**, what the commands below use and the system does not have
  yet: `curl` (for the installer), `python3` and `xz-utils` (for `vf fetch`), `python3-venv`
  (for `vf venv`). The installer and the launcher say which one is missing; they install
  nothing themselves.
- **CUDA 12.9.2 runtime libraries**: not included; `vf fetch` downloads them (below)

## 1. Install

With the installer:

```sh
curl -fsSL https://raw.githubusercontent.com/nbald/volta-forever/main/install.sh | bash
```

`install.sh` is at the root of the repository; read it before you run it, or download it and
run it as a file. What it does, in order:

1. checks the machine: Linux, x86-64, glibc 2.34 or later, and the tools it uses (`tar`,
   `gzip`, `sha256sum`, `mktemp`, `flock`, `curl` or `wget`);
2. downloads the archive of its version and the file `SHA256SUMS` from the project's releases,
   over HTTPS only, redirections included, and stops if the archive's SHA-256 is not the one
   that the file gives;
3. unpacks the archive under `~/.local/share/volta-forever` (`$XDG_DATA_HOME/volta-forever`
   when that is set; `/opt/volta-forever` for root) and links `vf` and `volta-forever` in
   `~/.local/bin` (`/usr/local/bin` for root). The CUDA 12.9.2 libraries that an earlier
   installation downloaded are kept, and what an installer that was killed left there more
   than a day before is removed;
4. runs `vf fetch` (section 2: NVIDIA's cuBLAS, from NVIDIA's servers), when `python3` and `xz`
   are there; when one is not, it says which and leaves that step to you;
5. runs `vf doctor`, and says so if the directory of the commands is not on your `PATH`.

It installs no package of the system, asks for no password and writes nowhere else.

| Setting (environment) | |
|---|---|
| `VF_VERSION` | the version to install (default: the one the script was published with) |
| `VF_PREFIX`, `VF_BIN` | where versions are unpacked, and where the commands are linked |
| `VF_INSTALL_FETCH=0` | do not run `vf fetch` |
| `VF_INSTALL_BASE` | where the archive and `SHA256SUMS` are taken from (a mirror, a directory served locally): an `https://` URL, or a `file://` one; `VF_INSTALL_ALLOW_HTTP=1` takes an `http://` one too |

Reinstalling or switching releases keeps existing `vf venv` environments pinned to their installed launcher: with such an environment activated, `vf version` and `vf doctor` say when it runs another release than theirs, and `vf venv DIR` makes it run through the release of that `vf`. Releases share one `cuda-12.9.2` directory under `VF_PREFIX`. When upgrading an older installation, this path points to its existing CUDA directory so manual `vf fetch` commands share the same downloads. Keep that directory when removing individual old versions.

To remove it: delete the directory of `VF_PREFIX` and the two links.

By hand, from a release's archive:

```sh
tar -xzf volta-forever-VERSION-linux-x86_64.tar.gz      # or: tar --zstd -xf ...tar.zst
export PATH=$PWD/volta-forever-VERSION-linux-x86_64/bin:$PATH
```

`bin/vf` is the launcher: everything below goes through it. `bin/volta-forever` is a link to
it, the same program under its long name (in the fish shell, `vf` may already be the command
of VirtualFish). It finds the rest of the archive from its own place, so it may also be called
by its full path, or through a link from a directory of the `PATH`.

## 2. Download the CUDA 12.9.2 libraries

The layer does not redistribute NVIDIA's libraries. The launcher downloads the ones your
program needs from NVIDIA's public redistributable archives, checks them (SHA-256) and unpacks
them into `cuda-12.9.2/` of the archive's directory, where it looks for them afterwards:

```sh
vf fetch           # cuBLAS, cuBLASLt, nvJitLink: 1.0 GB to download, 0.95 GB on disk
vf fetch --only libnccl  # multi-GPU collectives: 337 MB to download, 421 MB on disk
vf fetch --all     # also cuFFT, cuSOLVER, cuSPARSE, cuRAND, NPP, nvJPEG, NVRTC and NCCL
```

NCCL comes from NVIDIA's `nvidia-nccl-cu12` wheel. VF verifies its SHA-256 and installs the shared library and its license into the same tree. [NVIDIA NCCL on PyPI](https://pypi.org/project/nvidia-nccl-cu12/2.32.6/)

It needs `python3` and GNU `tar` (and uses `aria2c` when there is one). It unpacks the shared
libraries and NVIDIA's licence, nothing else, and removes each archive once it is unpacked.
Each archive, and the NCCL library, is unpacked in a staging directory and moved into place: a
fetch that fails or is stopped (Ctrl-C, SIGTERM, SIGHUP) leaves the files that were there before,
and the next one removes what it left. It can be run again: what is unpacked already is not
downloaded again. `--only REGEX` chooses the
packages of NVIDIA's manifest by name, `--to DIR` another directory (then tell the launcher
where: `--cuda12 DIR/lib`, or `VF_CUDA12=DIR/lib`).

**cuBLAS alone, or all of them.** cuBLAS is the one that a program cannot keep from CUDA 13:
the translator refuses some kernels of its CUDA 13 build, and the operations that need them
fail. With cuBLAS alone (`vf fetch`), a program's other CUDA libraries are its own CUDA 13
builds, whose kernels the layer translates when the program loads them. With all of them
(`vf fetch --all`), each is served by its CUDA 12.9.2 build, which has code for Volta: nothing
of it is translated, and nothing of it waits for a translation at the first run. What was run
with which:

| Program | CUDA 12.9.2 libraries behind it |
|---|---|
| PyTorch (stock wheel, sm_75 machine code only) | cuBLAS alone: the 925 operations and the 32 models of `LIMITS.md` (the transforms through the wheel's own cuFFT), and factorisations of `torch.linalg` and a sparse product through the wheel's own cuSOLVER and cuSPARSE. All of them: the 925 operations and the 32 models again |
| llama.cpp (official CUDA 13 build) | cuBLAS |
| vLLM, SGLang (pip) | cuBLAS alone (environment set by hand); all of them (through the launcher) |
| ZML source example (bundled CUDA libraries opened by absolute path) | cuBLAS, cuFFT, cuSOLVER, NVRTC |

If you already have the CUDA 12.9 toolkit installed, the launcher finds it in
`/usr/local/cuda-12.9/lib64`; elsewhere, name its library directory with `--cuda12` or
`VF_CUDA12`. The launcher refuses a directory whose path has a `:` (the lists it passes on are
separated by colons): name it by a link without one. `vf doctor` says which directory it took
and why. Section 4 has the command that `fetch` runs, for those who would rather run it
themselves.

## 3. Run a program

Put `vf` in front of the command:

```sh
vf python3 -c "import torch; print(torch.cuda.get_device_name())"
vf llama-completion -m model.gguf -p "Hello" -n 48 -ngl 99
```

The launcher sets the environment for that command and runs it: nothing stays in the shell,
and the exit status is the command's. `vf doctor` says what it finds and what is missing (the
layer's files, the card, the driver, the C library, which CUDA 12.9.2 directory it took and
why, the library that each forwarding library calls, the CUDA 12 NCCL, and the
`LD_LIBRARY_PATH` and `LD_AUDIT` that a command would get), and ends with a status other than
0 when something required is missing (of the CUDA 12.9.2 libraries, cuBLAS is required; it
names the others that are not there in one line).

**Python: pip and virtual environments.** pip looks at neither the card nor the CUDA of the
system: install the ordinary wheels for CUDA 13 (`pip install torch --index-url
https://download.pytorch.org/whl/cu130`). They bring CUDA 13 libraries of their own, which the
layer replaces when the program runs. A virtual environment can be made to run through the
layer by itself:

```sh
vf venv .venv                 # python3 -m venv .venv, then every Python of it runs through the layer
.venv/bin/pip install ...     # no prefix: pip, python and the scripts that pip puts in .venv/bin
.venv/bin/python train.py
```

`vf venv DIR` makes the environment when DIR is not one yet (`--python PYTHON` names the
Python to make it with, and what follows `--` goes to `python -m venv`), or takes one that is
there (made by `python -m venv` or by `uv venv`: a directory with a `pyvenv.cfg`). When Python
cannot make the environment (on Debian and Ubuntu, without the package `python3-venv`), it
says what Python says and removes what the attempt left, so that the same command works once
the package is there. It puts a module in the environment's
`site-packages` that every Python of the environment imports when it starts, and that starts
Python again through the launcher when the layer's environment is not set yet: activated or
not, started by a service or by an editor. So a package that compiles GPU code when it is
installed, and asks the GPU what to compile for, is told what a program is told (7.5), and not
`sm_70`, which CUDA 13's compiler refuses. `VF_VENV=0` in front of a command starts that Python
as it is; `vf venv --off DIR` takes the module out again. It needs Python 3.10 or later, and
it is for Python: another program run from the environment's shell still wants `vf` in front.

For a whole shell session:

```sh
eval "$(vf env)"
```

(`env` prints `export` lines; applied twice, it does not repeat the directories.)

| | |
|---|---|
| `vf [OPTIONS] [--] COMMAND [ARGS...]` | run COMMAND through the layer (`run` may be written before the options; needed for a command named like one of the words below) |
| `vf env [OPTIONS]` | print the environment as `export` lines |
| `vf venv [--python PYTHON] DIR [-- OPTIONS]`, `vf venv --off DIR` | a virtual environment of Python that runs through the layer by itself |
| `vf fetch [--all \| --only REGEX] [--to DIR]` | download the CUDA 12.9.2 libraries |
| `vf ptxas ARGS...` | assemble file-backed PTX for the V100 using the CUDA 12.9 assembler |
| `vf doctor` | check the installation |
| `vf inspect`, `extract`, `adapt`, `translate` | the offline tool (`bin/vf-translate`; "Tools", below) |
| `vf version`, `help` | |
| `--cuda12 DIR` | the directory of the CUDA 12.9.2 libraries (else `VF_CUDA12`, then `cuda-12.9.2/lib` of the installation, then in a source tree `toolkits/cuda-12.9.2/lib`, then `/usr/local/cuda-12.9/lib64`); a path with a `:` is refused |
| `--map FILE` | substitutions of your own, which come before the launcher's (the format of section 4, B) |
| `--no-audit` | do not set `LD_AUDIT` (a program that finds all its libraries through the loader does without it); an audit library of the layer that `LD_AUDIT` already names (after `eval "$(vf env)"`) is taken out of it |

**What the launcher sets.**

- `LD_LIBRARY_PATH`: the layer's `lib` (the proxy `libcuda.so.1`, the nvJitLink adapter), its
  `forward` (the libraries with CUDA 13's names), the CUDA 12.9.2 directory, then what was there.
- `LD_AUDIT` = the audit library, then those that were in `LD_AUDIT` (an audit library of the
  layer is not named twice): they see the names that it chose, and have the last word. And
  `VF_AUDIT_MAP` = its table of substitutions: for each forwarding library whose CUDA 12.9.2
  library is there, its name to the forwarding library, and the name of the CUDA 12.9.2
  library that it calls to that library; `libnvJitLink.so.13` to the layer's nvJitLink
  adapter; `libnccl.so.2` to NCCL's CUDA 12 build when it was fetched. So a program that
  carries CUDA 13 libraries of its own and finds them by a recorded path (a PyTorch wheel) or
  opens them by absolute path gets the forwarding ones instead. Entries that you already have
  in `VF_AUDIT_MAP` stay first, and the launcher run under itself (`vf vf ...`, or after
  `eval "$(vf env)"`) does not add its own again.
- `VOLTA_FOREVER` (the version) and `VOLTA_FOREVER_HOME` (the archive's directory): how a
  script can tell that it runs through the launcher.
- For the command it runs (not in what `vf env` prints): `VF_CUDA12`, the CUDA 12.9.2
  directory it took, and, for a Bazel program, `RUNFILES_DIR` (below).
- With `VF_JIT_LINEINFO=1`, `NVCC_APPEND_FLAGS` gains `-lineinfo` (after what it already holds;
  not twice) and `CUTE_DSL_LINEINFO=1` is set unless it is set already. This is for a program told
  8.0 or later (`VF_CAPABILITY`) that compiles CUDA while it runs (PyTorch extensions, FlashInfer,
  TileLang through nvcc; NVIDIA's CuTe DSL, CUTLASS kernels written in Python, without it): such
  compilers target the capability they are told and keep machine code only, of which the layer can
  translate little when it is for sm_80 or later. With line information they also keep each
  cubin's PTX, which the layer compiles again for sm_70. The machine code is the same; the cubins
  grow by 60 to 120 %. NVRTC (PyTorch's jiterator, CuPy) reads neither. Caches keyed without the
  environment (PyTorch extensions, FlashInfer, TileLang, ccache) keep what they built before:
  clear them once. `vf doctor` suggests it when `VF_CAPABILITY` is 8.0 or later. It is not set by
  default because the launcher cannot see a compiler's command line: next to a `-G` there,
  `-lineinfo` makes nvcc warn twice, and fail under `-Werror all-warnings`. It is left out when
  `NVCC_APPEND_FLAGS` itself holds `-G`. `vf env` prints the two variables only when it sets them.

A Bazel program that ships a CUDA toolkit in its runfiles can invoke an external assembler through an absolute path. When you start it with `vf`, the launcher prepares a private runfiles directory with the VF assembler and links to the program's other resources. It uses the local temporary directory when the user's cache cannot store private links. Install NVIDIA's `cuda-nvcc-12-9` package for this path. VF looks for `bin/ptxas` beside the selected CUDA 12.9 libraries and under `/usr/local/cuda-12.9`. Set `VF_PTXAS_REAL` to the absolute path of the CUDA 12.9 assembler to select another installation. For a file on disk, run `vf ptxas -arch=sm_80 kernel.ptx -o kernel.cubin` to produce code for sm_70. `vf ptxas` reads the options of `ptxas` 13.5 that take a value as `ptxas` does, ends with status 255 on the errors it finds itself, as `ptxas` does, and passes on the assembler's messages with the names of your files. The private runfiles directories are kept under `$XDG_CACHE_HOME/volta-forever/runfiles-v1` (`~/.cache/...`).

**When only some of the CUDA 12.9.2 libraries are there** (cuBLAS alone, after `vf fetch`
without `--all`), a forwarding library that has nothing to forward to is not used: it is left
out of the table, and the directory of the forwarding libraries is left off the library path.
The program then keeps its own library of that name (a wheel's cuFFT, cuSOLVER or cuSPARSE
for CUDA 13), and the layer translates its kernels like the rest of the program's code. That
is how the PyTorch tests of `LIMITS.md` are run when their environment is set by hand. With
`--no-audit` the forwarding libraries can only be found by their names: their directory stays
on the path, and the launcher says which of them have nothing behind them (a program that
calls one of those stops there).

When no CUDA 12.9.2 library is found at all the command still runs, with one line on standard
error (`VF_QUIET=1` leaves it out).

**A program that wants to know.** A program built for later GPUs is told the compute
capability that its code starts at (7.5), so that it loads that code: the driver's answers do
not tell it that the card is a Volta with the layer in front of it. A program that has kernels
written for Volta beside its code for later GPUs can ask, and prefer them: `libcuda.so.1` has
the functions `volta_forever_version()` (the version, a C string) and `volta_forever_abi()`
(the number of this interface, 1) only when it is the layer's.
`dlsym(RTLD_DEFAULT, "volta_forever_version")` gives the first or nothing; in Python,
`ctypes.CDLL("libcuda.so.1").volta_forever_version` or an `AttributeError`.

## 4. By hand

What the launcher does can be done without it. The CUDA 12.9.2 libraries:

```sh
# Needs: python3 and tar. Uses aria2c when available (faster); falls back to Python's urllib.
python3 scripts/fetch-redist.py \
    https://developer.download.nvidia.com/compute/cuda/redist \
    https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.2.json \
    dl-cuda-12.9.2 \
    cuda-12.9.2 \
    --only '^(libcublas|libcufft|libcusolver|libcusparse|libcurand|libnpp|libnvjpeg|cuda_nvrtc|libnvjitlink)$'
```

The manifest argument can be a local JSON file or an https:// URL (an http:// one is refused
unless `--allow-http` is given: the manifest holds the SHA-256 that the archives are checked
against). The script downloads and verifies (SHA-256) the archives listed in it, then unpacks
them into `cuda-12.9.2/`, headers and static libraries included, and keeps the archives (6.8 GB
unpacked for these nine). `vf fetch` adds `--runtime` (the shared libraries and the licence
alone), `--discard` (each archive removed once unpacked) and `--with-nccl` (NVIDIA's CUDA 12
NCCL wheel among the packages that `--only` can choose, as `libnccl`).

Then one of three ways to run a program.

### A. Library path (works when the program finds its libraries through the loader)

```sh
S=/path/to/volta-forever
T=/path/to/cuda-12.9.2/lib

LD_LIBRARY_PATH=$S/lib:$S/forward:$T:$LD_LIBRARY_PATH \
    some-cuda-13-program
```

The program finds the proxy (`libcuda.so.1`) and the forwarding libraries (the CUDA 13 names)
in `$S/lib` and `$S/forward`, and the forwarding libraries find the CUDA 12.9.2 originals in
`$T`.

**PyTorch needs the audit library (way B below), not way A alone.** A stock PyTorch wheel
finds NVIDIA's libraries by a path recorded in its own (RPATH), not by `LD_LIBRARY_PATH`. With
way A alone, the wheel's own cuBLAS 13 is used instead of the forwarding one; its sm_75 kernels
are translated, but some of them are refused by the translator, and the operations that need
them fail.

### B. Audit library with a map (for PyTorch and programs that open libraries by absolute path)

The audit library (`libvf_audit.so`) intercepts the loader's search and substitutes
libraries by file name. The substitution table is read from `VF_AUDIT_MAP` (environment,
colon-separated `name=path` pairs; highest priority) and from a map file named by
`VF_AUDIT_MAP_FILE` or `vf-audit.map` next to the audit library. The first matching entry
wins, and environment entries come before those of the file. An empty replacement makes the
library not found. Relative paths in the file are resolved relative to the file's directory.
(`crates/vf-audit/src/lib.rs`)

**Example: a stock PyTorch wheel (the wheel carries its own cuBLAS)**

Write a map file `torch-audit.map` (the actual map used by `tests/gpu/torch.sh`):

```
libcublas.so.13=/path/to/volta-forever/forward/libcublas.so.13
libcublasLt.so.13=/path/to/volta-forever/forward/libcublasLt.so.13
libcublas.so.12=/path/to/cuda-12.9.2/lib/libcublas.so.12
libcublasLt.so.12=/path/to/cuda-12.9.2/lib/libcublasLt.so.12
```

The proxy (`libcuda.so.1`) is found by the library path, not through the map.

Run:

```sh
S=/path/to/volta-forever
T=/path/to/cuda-12.9.2/lib

LD_AUDIT=$S/lib/libvf_audit.so \
VF_AUDIT_MAP_FILE=torch-audit.map \
LD_LIBRARY_PATH=$S/lib:$T:$LD_LIBRARY_PATH \
    python3 -c "import torch; print(torch.cuda.get_device_name())"
```

**Example from the historical ZML source test**

The default `tests/gpu/zml.sh` run keeps the build’s bundled cuDNN 9.24.0 and maps the following CUDA libraries.

```
libcuda.so.1=/path/to/volta-forever/lib/libcuda.so.1
libnvJitLink.so.13=/path/to/volta-forever/lib/libnvJitLink.so.13
libnvJitLink.so.12=/path/to/cuda-12.9.2/lib/libnvJitLink.so.12
libcublas.so.13=/path/to/volta-forever/forward/libcublas.so.13
libcublasLt.so.13=/path/to/volta-forever/forward/libcublasLt.so.13
libcufft.so.12=/path/to/volta-forever/forward/libcufft.so.12
libcusolver.so.12=/path/to/volta-forever/forward/libcusolver.so.12
libnvrtc.so.13=/path/to/volta-forever/forward/libnvrtc.so.13
```

A separate historical comparison set `VF_ZML_CUDNN=9.10.2` (a setting of `tests/gpu/zml.sh`) and made that older library report 9.24.0 with `@cudnn-version`. Use the cuDNN build the application expects. The verified 35B LLMD server runs through `vf ./llmd` with its matching runfiles and no manual audit map. [LLMD launch command](docs/guides/zml.md#5-benchmarks)

### C. Forwarding libraries on the library path

Place the forwarding libraries and the CUDA 12.9.2 originals on the library path so the
program finds them before the CUDA 13 ones it carries:

```sh
LD_LIBRARY_PATH=/path/to/volta-forever/lib:/path/to/volta-forever/forward:/path/to/cuda-12.9.2/lib:$LD_LIBRARY_PATH \
    some-cuda-13-program
```

### Programs that were run this way

On one Tesla V100, with no switch set (`LIMITS.md` has the counts and what does not work):

| Program | How | What to know |
|---|---|---|
| PyTorch 2.14 (wheel for CUDA 13) | B, with the map for its cuBLAS | 924 of 925 operations tested give the CPU's results (the other one the CPU does not do), 32 of 32 small whole models |
| llama.cpp (official CUDA 13 build) | A or C | flash attention on or off; quantized and NVFP4 weights |
| Historical ZML source test | B, with the map above | the test opens bundled CUDA libraries by absolute path |
| vLLM 0.34 (installed with pip) | B, as PyTorch | the engine and its HTTP server |
| SGLang 0.5 (installed with pip) | B, as PyTorch | the engine and its HTTP server, with its default attention (FlashInfer) or with `--attention-backend triton`. Five of FlashInfer's eighteen prefill kernels were refused when they were last counted (`LIMITS.md`): if a model or a batch asks for one of them, the error names it, and `--attention-backend triton` goes without them |

A kernel that the layer cannot run is refused when the program asks for it (the program gets
error 218 and a NULL handle, and on standard error a line names the kernel and says why:
`LIMITS.md`, "Kernels refused"); it is never run wrong knowingly. The first run of a program translates its machine code and lets the driver compile
what it compiles (`What the first run costs`, below).

## 5. Verify that it works

`vf doctor`, then a PyTorch test that exercises the proxy and cuBLAS forwarding
through the audit library (needs a stock PyTorch wheel for CUDA 13, e.g. `torch==2.14.0` from
`download.pytorch.org/whl/cu130`):

```sh
vf python3 -c "
import torch
print('device:', torch.cuda.get_device_name())
a = torch.randn(256, 256, device='cuda')
b = torch.randn(256, 256, device='cuda')
c = torch.mm(a, b)
print('matmul shape:', c.shape)
print('sum:', c.sum().item())
print('ok')
"
```

Expected output:

```
device: Tesla V100-SXM2-32GB
matmul shape: torch.Size([256, 256])
sum: <a number>
ok
```

## Environment variables

Every variable that the layer, the launcher and the installer read is here (the installer's are
in section 1). The ones of the test scripts are in `tests/README.md`; `docs/architecture.md`,
"Switches and diagnostics", groups these by purpose.

**Proxy (`libcuda.so.1`):**

| Variable | Default | Description |
|---|---|---|
| `VF_LOG_LEVEL` | `1` | 0: fatal only; 1: kernels refused, emulated, images adapted; 2: what was adapted; 3: everything |
| `VF_LOG` | stderr | Log to a file instead of standard error |
| `VF_EMULATION_WARNINGS` | `20` | How many kernels with emulated instructions are listed; 0 for none |
| `VF_PROXY_CACHE` | `$XDG_CACHE_HOME/volta-forever/proxy`, else `~/.cache/volta-forever/proxy` | Directory for the proxy's cache (translated machine code, adapted PTX); `0` or empty to disable. Must be a directory of the user that no one else can write (group- or world-writable directories are refused, and the cache is then off); the layer makes it with mode 0700. The cache can grow to 2 GB (`VF_PROXY_CACHE_MAX`). On a system whose home directory is on a RAM filesystem, this is memory. A write that fails leaves no temporary file, and temporary files left more than ten minutes before are removed at the first use of the cache in a process. |
| `VF_PROXY_CACHE_MAX` | `2147483648` (2 GB) | Maximum size of the cache in bytes |
| `VF_CAPABILITY` | auto | Compute capability reported to callers without Volta code, as `<major>.<minor>` (default: the lowest the program has code for, which is 7.5 for programs with only sm_75 machine code). `VF_CAPABILITY=8.0` reports Ampere and prefers compatible PTX for the highest target up to 8.0, except to a caller whose code, or the code of the libraries that use it (for a CUDA runtime), runs from sm_75 machine code, such as PyTorch's, or from PTX for sm_75 only: that caller is still told 7.5, and the images it loads are adapted for 7.5, with a line at the default log level. A value that is not `<major>.<minor>` (`8`, `80`, `sm_80`) or that is below 7.0 is not used, and the layer says so. What it does not cover: `LIMITS.md`, "The capability told with `VF_CAPABILITY`"; how images are chosen: `docs/architecture.md`, section 3. |
| `VF_CAPABILITY_ALL` | `0` | `1`: report `VF_CAPABILITY` also to the callers that are otherwise still told 7.5. Their kernels for the reported capability then compute nothing, or stop the GPU: only for a program that does not launch them, as SGLang with the settings of its guide, which selects its quantized experts from PyTorch's capability. |
| `VF_TRANSLATE` | `1` | `0`: disable machine-code translation; `first`: translate even where PTX exists; `force`: translate even where sm_70 code exists |
| `VF_PAIRS` | `1` | `0`: each Turing tensor-core product of machine code is translated by itself; by default two of them are done together on whole rows where the program loads them that way (`docs/pairs.md`) |
| `VF_BORROW` | `7` | Where the translation of machine code takes the registers it needs. `7`: above the program's own wherever that translates, and for a kernel that would otherwise be refused for want of registers, among the program's registers that hold nothing, then in the thread's local memory, kernels with tensor-core products included. `6`: the same, such kernels with products refused (the default until 2026-11-20). `0`: above the program's only (kernels that name nearly every register are refused). `9`: try to preserve the source kernel’s maximum block size when translating machine code for CUDA graphs. [Graph limits](LIMITS.md#cuda-graphs). Other values are for experiments (`docs/translator.md`, "Registers") |
| `VF_FIT` | `1` | `0`: a launch that the driver refuses for its registers (`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`) is refused to the program. By default the kernel is then launched from a second translation of its module in which no kernel takes more registers than it declared (slower code; made once, at the first refused launch). Not for a kernel that can reach data of its module (`docs/architecture.md`, "Launch") |
| `VF_FIT_SHARED` | `0` | Set to `1` to retry a rejected dynamic shared-memory opt-in on a V100 at the physical maximum available to that kernel. Each launch must still fit its static and dynamic allocation within the device limit. |
| `VF_BF16_TC_GROUPS` | `0` | Set to `1` to use Volta FP16 Tensor Cores for groups of BF16 matrix products when every multiplicand converts exactly to normal FP16 or signed zero. FP32 accumulation may round differently. This can increase first-use compilation time. |
| `VF_BF16_TC_TILES` | `0` | Set to `1` to use Volta Tensor Cores for two-by-two BF16 matrix-product tiles. A warp uses this path when its inputs convert exactly to normal FP16 or signed zero. Other inputs keep the scalar path. FP32 accumulation can round differently. |
| `VF_BF16_COLUMN` | off | For four successive BF16 matrix products sharing an accumulator, use the single-column path when only column 0 of B is active and the warp passes the operand checks. `ordered` preserves the scalar FP32 FMA order. `tree` sums four partial results and can round differently, especially when large terms cancel. |
| `VF_BF16_ROW` | off | For pairs of BF16 matrix products sharing a left fragment, test whether only its first row contributes. `ordered` preserves the scalar FP32 FMA order. `tree` sums four partial dots before adding the accumulator and can round differently. Other inputs take the ordinary BF16 path. |
| `VF_ALIGNED_ROWS` | `1` | `0`: do not take it that the program keeps the rows that `ldmatrix` reads at multiples of 16 bytes, as the PTX manual requires: test the addresses when the code runs. For a program that stops with a misaligned-address fault (error 716); emulated tensor-core kernels are slower with it |
| `VF_LAZY` | `1` | `0`: let the driver compile a whole module when it is loaded, not each kernel when the program asks for it (a much longer first run) |
| `VF_ADAPT` | `1` | `0`: give the driver every image as it is (to see what a program does without the layer's rewriting) |
| `VF_REAL_LIBCUDA` | auto | Path to NVIDIA's real `libcuda.so.1` when it cannot be found by itself (by default: the library of the loaded kernel module's version in the system's library directories) |
| `VF_STATS` | `0` | `1`: route every driver function through the layer and print call counts and the most launched kernels at exit |
| `VF_PROFILE` | off | `FILE`: write a per-kernel profile to `FILE.PID` at exit. Every launch is then synchronised on its stream: a profile is not a benchmark. Each kernel keeps the counts of its first 64 distinct grid, block and shared-memory sizes. Only the process that loaded the layer writes it, and the `VF_STATS` report: a child made by `fork()` (a daemon, a pre-fork worker) writes neither, even when it does all the CUDA work |
| `VF_TIMING` | `0` | `1`: print the time of each stage for every module and kernel |
| `VF_TRACE_LAUNCH` | `0` | `1`: route kernel launches through the layer and log those that the driver refuses (kernel, grid, block, shared memory, launch attributes) |
| `VF_TRANSLATE_DUMP` | off | `DIR` (which must exist): write every cubin given to the translator there as it was received, as `PID-NUMBER.sm_SM.cubin`, to translate and validate it apart (`reference/validate.py`) |
| `VF_BORROW_SAY` | off | Set: the translator says on standard error, for each function, what registers it borrowed (as `reference/translate.py` does) |

The reports at exit (`VF_STATS`, `VF_PROFILE`, the summaries of emulated instructions and of
second translations) are printed by the process that loaded the layer, not by a child that it
made with `fork()`.

**Audit library (`libvf_audit.so`):**

| Variable | Default | Description |
|---|---|---|
| `VF_AUDIT_MAP_FILE` | `vf-audit.map` beside the library, when there is one | Path to a map file (one `name=path` per line; `#` comments; empty path = not found) |
| `VF_AUDIT_MAP` | none | Map entries in the environment, colon-separated; these come before the file's entries |

It also reads `VF_LOG`, `VF_LOG_LEVEL` (2: each substitution) and `VF_CAPABILITY`, which it
reports to NVML (`nvmlDeviceGetCudaComputeCapability`) on a V100.

**nvJitLink adapter (`libnvJitLink.so.13`):**

| Variable | Default | Description |
|---|---|---|
| `VF_JITLINK_REAL` | `libnvJitLink.so.12`, as the loader finds it | The nvJitLink 12.9 library to use |
| `VF_JITLINK_REAL_DIR` | none | A directory: take `libnvJitLink.so.12` there (when `VF_JITLINK_REAL` is not set) |
| `VF_JITLINK_DUMP` | off | `DIR`: write every PTX input there as the program gave it, as `PID-NUMBER.ptx` |

It adapts PTX as the proxy does, with the same switches of the rewriting (`VF_CAPABILITY`,
`VF_ALIGNED_ROWS`, `VF_BF16_*`), and also reads `VF_LOG`, `VF_LOG_LEVEL` and
`VF_EMULATION_WARNINGS`.

**Forwarding libraries (`forward/`):**

| Variable | Default | Description |
|---|---|---|
| `VF_FWD_<KEY>` | the CUDA 12.9.2 library's name | The library that one forwarding library opens instead, by path or name. KEY is its short name in capitals, `-` written `_`: `VF_FWD_CUBLAS`, `VF_FWD_CUBLASLT`, `VF_FWD_CUFFT`, `VF_FWD_NVRTC_BUILTINS`, `VF_FWD_NPPIF`, ... (`crates/vf-fwdgen/src/libs.rs`) |
| `VF_FWD_VERBOSE` | off | Set: each forwarding library says which library it opened and each function that it binds |

**Launcher (`vf`) and `vf ptxas`:**

| Variable | Default | Description |
|---|---|---|
| `VF_CUDA12` | found (section 2) | The directory of the CUDA 12.9.2 libraries (`--cuda12` comes first). `vf ptxas` takes `bin/ptxas` beside it |
| `VF_QUIET` | `0` | `1`: no line when the CUDA 12.9.2 libraries are not found |
| `VF_VENV` | unset | `0`: a Python of an environment made by `vf venv` starts as it is, not through the layer |
| `VF_PTXAS_REAL` | `bin/ptxas` beside `VF_CUDA12`, else `/usr/local/cuda-12.9/bin/ptxas` | The CUDA 12.9 assembler that `vf ptxas` runs |

## What the first run costs

The first time a program runs through the layer, the proxy:

1. **Adapts** each module's PTX for Volta (rewrites instructions Volta lacks) and **analyses**
   what each kernel depends on. For llama.cpp (12 modules of 14-20 MB each), this takes about
   5.6 seconds (3.1 s for the rewriting, 2.5 s for the analysis).
2. **Translates** sm_75 machine code and supported sm_80 functions to sm_70. A stock PyTorch
   wheel includes kernels that are distributed as machine code only.
3. The **driver compiles** the rewritten PTX. Large attention kernels take 1.1 to 1.8 seconds
   each when the driver's own cache of compiled kernels is empty.

The proxy caches what it worked out in `~/.cache/volta-forever/proxy/` by default. With a warm
cache, llama.cpp's 48-token generation takes 1.3 seconds where a cold cache takes 6.9 seconds.
For PyTorch, a battery of 820 operations takes 5 to 7 seconds with a warm cache against 63
seconds without (`docs/performance.md`, "Where the time goes"). The cache directory must be owned by the
user and not writable by others; on a machine whose home is on a tmpfs, it is in memory and
lost on reboot.

## Licence

The layer is distributed under the Apache License, version 2.0 (`LICENSE`), except the
instruction tables of the machine-code translator, which derive from descriptions distributed
under the GNU Lesser General Public License, version 2.1, and are distributed under it
(`LICENSE.LGPL-2.1`). The binaries contain those tables: `NOTICE` says where they come from,
where the complete source of the binaries is, and which Rust crates are linked in, with their
licence texts. The archive contains no NVIDIA software; `scripts/fetch-redist.py` downloads
the CUDA 12.9 libraries from NVIDIA's public archives, under NVIDIA's licence.

The layer is not NVIDIA's and is not endorsed by NVIDIA.

## The build this archive was made from

The `VERSION` file holds the version, the commit of the source repository and `-dirty` when
tracked files differed from that commit. The source of that commit is the complete source of
the binaries: `scripts/build.sh` and `scripts/build-forwarders.sh` build them, and
`scripts/package.sh` makes this archive.

## Tools

The offline tool is `bin/vf-translate`; the launcher has its four functions as subcommands:

```sh
vf inspect FILE...                  # which architectures a file carries
vf extract FILE DIR [SM]            # extract the code from a fat binary
vf adapt IN OUT                     # adapt PTX for Volta (what the proxy does)
vf translate IN OUT [MAP [REPORT]]  # translate supported sm_75 and sm_80 machine code to sm_70
```

`translate` writes beside MAP the file `MAP.lives` that `reference/validate.py` reads with it,
and REPORT has one line a function: `ok`, or why it is refused.
