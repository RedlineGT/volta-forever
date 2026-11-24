"""volta-forever: Triton's tl.dot on the products that the layer runs fast (VF_TRITON_TC).

On a Tesla V100 the layer tells a program 7.5, and Triton, told 7.5, computes tl.dot with float32
fma: it writes matrix-product instructions only for 8.0 and later (LIMITS.md, "Speed"). Triton
takes the capability from PyTorch, whose answer must stay 7.5 for PyTorch's own kernels, so the
layer cannot tell Triton something else. This module changes what Triton compiles, not what anyone
is told: a kernel that Triton compiles for 7.5, on a machine whose GPUs are all Voltas behind the
layer, and whose products (tl.dot) all multiply two operands of one of the types that VF_TRITON_TC
names, is compiled for sm_80 instead, as TRITON_OVERRIDE_ARCH=sm80 would compile it with its PTX
and its cubin for sm_80 too, when all of this holds:

  - the sm_80 kernel's shared memory fits a block of a Volta (96 KiB); if it does not, it is
    compiled again with one pipeline stage fewer, down to one;
  - its cubin keeps its PTX (ptxas -lineinfo, which Triton passes), which the layer compiles for
    the V100 when the kernel is loaded;
  - the layer's offline tool (vf-translate adapt), given that cubin, retargets its PTX with no
    kernel refused and no function made to trap: what the layer will do with it at load;
  - nothing of the sm_80 compilation failed.

Any other kernel, and one for which any of these fails, is compiled by Triton's own stages, as
without this module. The types: float16 (f16; the layer's products on Volta's tensor cores,
whose sums are truncated: docs/emulations.md, section 1), bfloat16 (bf16; the layer's scalar
products, float32 fma in the order of k) and int8 (i8; the layer's integer products, exact).
VF_TRITON_TC=1 is f16,bf16,i8; a list of some of them takes those alone.

While this is on, the caches have a mark of their own, "-vf-tc<rules>-<types>-<version of the
layer>-<the layer's settings>", at the end of two of Triton's keys: the hash of its NVIDIA backend for 7.5, which Triton
puts in the key of every kernel of its cache and PyTorch's Inductor in the keys of its kernels,
and triton_key(), which Triton puts there too and Inductor in the keys of its compiled graphs and
of its autotuning. What was compiled or tuned without it is not taken, nor the reverse.

The launcher puts the directory of this file first in PYTHONPATH when VF_TRITON_TC is set; its
sitecustomize installs an import hook on three modules of Triton: triton.backends (the table of
Triton's backends), whose class CUDABackend it patches (add_stages, hash) once Triton has made
that table, and triton.runtime.cache or triton.compiler.compiler, whichever defines triton_key
(Triton 3.5 and later, 3.3 and 3.4), whose triton_key it wraps. Nothing of Triton is imported
before the program imports it. The stages of a
kernel that is compiled for sm_80 call four other methods of CUDABackend (make_ttgir, make_llir,
make_ptx, make_cubin) as Triton's own stages call them; they have had these names and parameters
from Triton 3.3 to 3.9. When they do not, this module says so once and changes nothing.

The compilation for sm_80 runs in a helper process of the program's own, which a crash of
Triton's compiler cannot take down with it: how a process starts and asks its helpers is in
volta_forever_triton_client, read at the first kernel that may be compiled for sm_80 (client()),
and the helper itself in volta_forever_triton_helper, which only the helper reads. Python compiles
each module from its source here (the installation holds no bytecode, and nothing is written into
it at run time), so a process reads only what it uses.
"""

import os
import sys

import _thread

# The version of the rules that decide which kernels are compiled for sm_80 (in the cache key).
RULES = "3"
# The kinds of products that are left to 7.5 although they may be compiled for sm_80: measured
# slower that way on a V100, through the layer, than Triton's fma code for 7.5 (below 0.97 of its
# speed at 1, 16 or 256 rows, in one sweep; a line within the noise of such measures:
# docs/performance.md, "With VF_TRITON_TC"). A kernel all of whose
# products are of one kind is compiled for 7.5. Each kind: the operand types, the accumulator types,
# how B is stored ("n": by rows of N, its elements of one k next to each other in memory, as the
# shared memory of the sm_80 kernel shows it), and the cells measured slower: a tile (M, N, K) and
# the stages asked (None: any). A tile or a number of stages that was not measured is not left so.
SLOWER = (
    {"why": "bfloat16 products of a skinny tile measured slower, B stored by rows",
     "types": ("bf16",), "acc": ("f32",), "b": "n",
     "cells": {(16, 128, 32): None, (16, 128, 64): (1, 3), (16, 64, 32): (1,), (16, 64, 64): None,
               (32, 128, 32): (1, 3), (32, 64, 32): None, (32, 64, 64): None}},
    {"why": "float16 products with float16 sums of a skinny tile measured slower, B stored by rows",
     "types": ("f16",), "acc": ("f16",), "b": "n",
     "cells": {(16, 128, 32): None, (16, 128, 64): (3,), (32, 64, 32): (1,), (16, 256, 32): (2,)}},
)
# The types of the operands of a product that may be compiled for sm_80, by their name in Triton's
# IR, and what VF_TRITON_TC=1 takes.
KNOWN_TYPES = ("f16", "bf16", "i8")
# Shared memory that a block can have on a Volta GPU (cudaDevAttrMaxSharedMemoryPerBlockOptin).
VOLTA_SHARED = 98304
# The Volta GPUs, by the names that the driver gives them.
VOLTA_NAMES = ("V100", "TITAN V", "Titan V", "GV100")
# The layer's settings with which it would not run the PTX that its tool checked: nothing is done.
OFF_SETTINGS = (
    ("VF_ADAPT", ("0",), "the layer gives the driver the sm_80 machine code as it is"),
    ("VF_TRANSLATE", ("first", "force"), "the layer takes a kernel's machine code before its PTX"),
)
# The layer's settings that change what it makes of a cubin's PTX, or what its tool says of it
# (Policy::with_environment in crates/vf-adapt, and the proxy's): their values are in the mark of
# the caches.
MARKED_SETTINGS = ("VF_ADAPT", "VF_ALIGNED_ROWS", "VF_BF16_COLUMN", "VF_BF16_ROW", "VF_BF16_TC_GROUPS",
                   "VF_BF16_TC_TILES", "VF_BORROW", "VF_CAPABILITY", "VF_CAPABILITY_ALL", "VF_FIT",
                   "VF_FIT_SHARED", "VF_LAZY", "VF_PAIRS", "VF_TRANSLATE")
# The modules whose import is hooked: Triton's backends, whose table has the class CUDABackend in
# its entry "nvidia" (Triton 3.3 loads the module of that class under another name), and the two
# that define triton_key in one version or another.
TARGETS = ("triton.backends", "triton.runtime.cache", "triton.compiler.compiler")

_lock = _thread.allocate_lock()
_said = set()
_state = {}
# Whether the last key that this thread computed for a kernel (CUDABackend.hash) had the mark: a
# kernel whose key had it is not kept in Triton's cache after a failure that may be otherwise
# another time; one whose key had not (the process no longer uses the switch) is Triton's own.
_keys = _thread._local()


def _after_fork():
    """In a child made by fork(): the lock is free. Another thread of the parent may have held it
    at the fork; that thread is not in the child, which would wait for it at its next kernel or
    line. (It is held across no I/O and no fork of this module's, so nothing in the child holds
    it.)"""
    if _lock.locked():
        try:
            _lock.release()
        except RuntimeError:
            pass


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


# --- the lines of the layer -------------------------------------------------------------------

def _level():
    try:
        return int(os.environ.get("VF_LOG_LEVEL", "1"))
    except ValueError:
        return 1


def say(message, level=1, once=None):
    """A line as the layer writes its own: to VF_LOG when it is set, else to standard error."""
    if _level() < level:
        return
    if once is not None:
        with _lock:
            if once in _said:
                return
            _said.add(once)
    line = "[volta-forever] " + message + "\n"
    path = os.environ.get("VF_LOG")
    try:
        if path:
            with open(path, "a") as f:
                f.write(line)
        else:
            sys.stderr.write(line)
            sys.stderr.flush()
    except Exception:  # noqa: BLE001 (a line that cannot be written is no reason to fail)
        pass


# --- whether this process compiles for Voltas behind the layer -------------------------------

def types():
    """The types of products that VF_TRITON_TC asks for: a tuple, empty when it is off."""
    value = os.environ.get("VF_TRITON_TC", "").strip()
    if value in ("", "0"):
        return ()
    if value == "1":
        return KNOWN_TYPES
    asked = [t.strip() for t in value.split(",") if t.strip()]
    unknown = [t for t in asked if t not in KNOWN_TYPES]
    if unknown:
        say("VF_TRITON_TC=%s: %s is not one of %s, and is not used" % (value, ", ".join(unknown),
                                                                         ", ".join(KNOWN_TYPES)), once="types")
    return tuple(t for t in KNOWN_TYPES if t in asked)


def _gpu_names(root="/proc/driver/nvidia/gpus"):
    """The names of the machine's NVIDIA GPUs, from the files of the driver in /proc. No CUDA
    call: a process that forks after this must still be able to use CUDA in its children.
    VF_TRITON_TC_GPUS (a list separated by commas) takes their place on a machine without the
    driver's directory alone (the tests' machines): it cannot make a machine with another GPU
    pass for one whose GPUs are all Voltas."""
    names = []
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        if os.path.isdir(os.path.dirname(root)):
            return names
        given = os.environ.get("VF_TRITON_TC_GPUS", "")
        return [n.strip() for n in given.split(",") if n.strip()]
    for entry in entries:
        try:
            with open(os.path.join(root, entry, "information")) as f:
                for line in f:
                    if line.startswith("Model:"):
                        names.append(line.split(":", 1)[1].strip())
                        break
        except OSError:
            continue
    return names


def _layer_version():
    """The version of the layer when the libcuda.so.1 of this process is the layer's, else None
    (INSTALL.md, "A program that wants to know")."""
    try:
        import ctypes
        lib = ctypes.CDLL("libcuda.so.1")
        version = lib.volta_forever_version
        version.restype = ctypes.c_char_p
        return version().decode("ascii", "replace")
    except (OSError, AttributeError, ValueError):
        return None


def _tool():
    """The layer's offline tool, which says what the layer does with a cubin: that of this file's
    own installation, in an archive's layout (bin/) or a source tree's (dist/bin/); not that of
    VOLTA_FOREVER_HOME, which may be another release's launcher."""
    home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in (os.path.join(home, "bin", "vf-translate"), os.path.join(home, "dist", "bin", "vf-translate")):
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def active():
    """(True, mark) when kernels told 7.5 may be compiled for sm_80 in this process, else (False,
    why). Decided once, at the first compilation, and said once."""
    with _lock:
        if "active" in _state:
            return _state["active"]
    result = _decide()
    with _lock:
        first = "active" not in _state
        _state.setdefault("active", result)
        result = _state["active"]
    if first and types():
        version = _state.get("version", "?")
        if result[0]:
            say("Triton %s: a kernel whose products are all of %s is compiled for sm_80 (VF_TRITON_TC)"
                % (version, ", ".join(types())), level=2)
        else:
            say("Triton %s: VF_TRITON_TC is not used: %s" % (version, result[1]))
    return result


def _safe():
    """VF_SAFE=1 (exactly, as the launcher's safe_mode reads it), read once in a process: safe mode,
    in which VF_TRITON_TC is off."""
    with _lock:
        if "safe" not in _state:
            _state["safe"] = os.environ.get("VF_SAFE") == "1"
        return _state["safe"]


def _decide():
    asked = types()
    if not asked:
        return False, "VF_TRITON_TC is not set"
    if _safe():
        return False, "VF_SAFE=1"
    capability = os.environ.get("VF_CAPABILITY", "").strip()
    if capability not in ("", "7.5"):
        return False, "VF_CAPABILITY is %s; it is for the default capability, 7.5, alone" % capability
    for name, values, why in OFF_SETTINGS:
        value = os.environ.get(name, "").strip()
        if value in values:
            return False, "%s=%s: %s" % (name, value, why)
    version = _layer_version()
    if version is None:
        return False, "the libcuda.so.1 of this process is not the layer's"
    with _lock:
        _state["layer"] = version
    names = _gpu_names()
    if not names:
        return False, "no GPU was found in /proc/driver/nvidia/gpus"
    others = [n for n in names if not any(v in n for v in VOLTA_NAMES)]
    if others:
        return False, "the machine has a GPU that is not a Volta (%s)" % others[0]
    tool = _tool()
    if tool is None:
        return False, "the layer's tool vf-translate was not found"
    # (What does not depend on a kernel is decided here, before any key has the mark: a process
    # that cannot start a helper compiles as without the switch, and Triton's cache keeps its
    # kernels so. The directory of the places is not: the client finds it at the first kernel
    # that may be compiled for sm_80, and does without it.)
    try:
        exe = _interpreter()
    except OSError as e:
        return False, str(e)
    with _lock:
        _state.update(tool=tool, types=asked, python=exe)
    return True, "vf-tc%s-%s-%s-%s" % (RULES, "+".join(asked), version, _settings_mark())


def _interpreter():
    """The interpreter to start a helper with: sys.executable when it is known to be this Python
    (the file that the process runs, /proc/self/exe, and a name of Python's or PyPy's), else
    OSError. A frozen or embedded application's sys.executable is the application, which must not
    be started again with the helper's arguments."""
    import re
    exe = sys.executable
    if getattr(sys, "frozen", False):
        raise OSError("this program is a frozen application, with no interpreter to start a helper with")
    if not exe:
        raise OSError("this Python does not say where it is (sys.executable)")
    real = os.path.realpath(exe)
    try:
        running = os.path.realpath("/proc/self/exe")
    except OSError:
        running = real
    names = (os.path.basename(exe), os.path.basename(real))
    if real != running or not any(re.match(r"(python|pypy)[0-9.]*[dmt]*(-c)?(\.exe)?\Z", n) for n in names):
        raise OSError("sys.executable (%s) is not known to be the Python interpreter of this process" % exe)
    return exe


def _cache_dir():
    """Triton's cache directory, as Triton finds it now: TRITON_CACHE_DIR, else .triton/cache in
    TRITON_HOME or the home directory (knobs.cache.dir in Triton 3.4 to 3.9, which reads the
    environment each time, default_cache_dir() in 3.3)."""
    try:
        from triton import knobs
        return os.path.abspath(knobs.cache.dir)
    except (ImportError, AttributeError):
        pass
    given = os.environ.get("TRITON_CACHE_DIR", "").strip()
    if given:
        return os.path.abspath(given)
    try:
        from triton.runtime.cache import default_cache_dir
        return os.path.abspath(default_cache_dir())
    except (ImportError, AttributeError):
        return os.path.join(os.environ.get("TRITON_HOME") or os.path.expanduser("~"), ".triton", "cache")


def _settings_mark():
    """What the layer's settings that change what it makes of a cubin's PTX, or what its tool says
    of it, are in this process: "default", or a digest of those that are set."""
    import hashlib
    given = ["%s=%s" % (name, os.environ[name]) for name in MARKED_SETTINGS if os.environ.get(name, "") != ""]
    if not given:
        return "default"
    return hashlib.sha256(";".join(given).encode()).hexdigest()[:12]


# --- which kernels ---------------------------------------------------------------------------

_DOT = r"=\s*tt\.dot\s.*?:\s*tensor<([^>]*)>\s*\*\s*tensor<([^>]*)>\s*->\s*tensor<([^>]*)>"


def _element(tensor_type):
    return tensor_type.rsplit("x", 1)[-1].strip()


def eligible(ttir, allowed):
    """None when the kernel of this TTIR (Triton's IR before layouts, as text) may be compiled for
    sm_80, else why not: it must have a tl.dot, each of its tl.dot must multiply two operands of
    one type that is allowed, and it must have no FP8, FP6 or FP4 value (Triton converts FP8
    otherwise for 7.5 and for 8.0) and no scaled product."""
    import re
    if "tt.dot_scaled" in ttir:
        return "it has a scaled product"
    if re.search(r"f[468]E\d", ttir):
        return "it has FP8 (or FP6, FP4) values"
    dots = [line for line in ttir.splitlines() if re.search(r"=\s*tt\.dot\s", line)]
    if not dots:
        return "it has no tl.dot"
    for line in dots:
        m = re.search(_DOT, line)
        if m is None:
            return "a tl.dot that was not understood"
        a, b, _ = (_element(t) for t in m.groups())
        if a != b or a not in allowed:
            return "a tl.dot of %s by %s" % (a, b)
    return None


def _products(ttir):
    """Each tl.dot of a TTIR: (operand type, accumulator type, M, N, K of its tile)."""
    import re
    out = []
    for line in ttir.splitlines():
        m = re.search(_DOT, line)
        if m is None:
            continue
        a, b, c = m.groups()
        dims_a = [int(x) for x in a.split("x")[:-1] if x.strip().isdigit()]
        dims_c = [int(x) for x in c.split("x")[:-1] if x.strip().isdigit()]
        if len(dims_a) < 2 or len(dims_c) < 2:
            continue
        out.append((_element(a), _element(c), dims_c[-2], dims_c[-1], dims_a[-1]))
    return out


def b_storage(ttgir):
    """How the B operands of the products of an sm_80 TTGIR are stored in shared memory: "n" when
    every one is by rows of N (its last dimension the fastest), "k" when every one is by columns,
    else None (mixed, or not loaded from shared memory)."""
    import re
    orders = {}
    for line in ttgir.splitlines():
        m = re.match(r"\s*(#\w+)\s*=\s*#[\w.]*shared<\{(.*)\}>", line)
        if m:
            o = re.search(r"order\s*=\s*\[([\d,\s]+)\]", m.group(2))
            if o:
                orders[m.group(1)] = [int(x) for x in o.group(1).split(",")]
    seen = set()
    for line in ttgir.splitlines():
        if "local_load" not in line or not re.search(r"dot_op<\{opIdx\s*=\s*1\b", line):
            continue
        m = re.search(r"memdesc<([^,>]*),\s*(#\w+)", line)
        order = orders.get(m.group(2)) if m else None
        if not order:
            return None
        rank = len(m.group(1).split("x")) - 1
        seen.add("n" if order[0] == rank - 1 else "k")
    return seen.pop() if len(seen) == 1 else None


def slower(ttir, ttgir, stages=None):
    """Why a kernel that may be compiled for sm_80 is left to 7.5 all the same, or None: every one of
    its products is of a kind that SLOWER names (TTIR: the types and the tile; the sm_80 TTGIR: how
    B is stored; the stages asked)."""
    if not SLOWER or _triton_version() < (3, 4):
        return None   # (Triton 3.3 cannot compile a product of halves for 7.5: sm_80 is the only way)
    products = _products(ttir)
    if not products:
        return None
    storage = None
    for rule in SLOWER:
        if not all(_matches(rule, p, stages) for p in products):
            continue
        if "b" in rule:
            if storage is None:
                storage = b_storage(ttgir) or "?"
            if storage != rule["b"]:
                continue
        return "%s (measured slower for sm_80 on a V100)" % rule["why"]
    return None


def _triton_version():
    """Triton's version as (major, minor), or (99, 0) when it is not known."""
    import re
    try:
        import triton
        m = re.match(r"(\d+)\.(\d+)", triton.__version__)
        return (int(m.group(1)), int(m.group(2))) if m else (99, 0)
    except Exception:  # noqa: BLE001
        return (99, 0)


def _matches(rule, product, stages=None):
    a, acc, m, n, k = product
    big = (0, 1 << 62)
    if "cells" in rule:
        if (m, n, k) not in rule["cells"]:
            return False
        asked = rule["cells"][(m, n, k)]
        if asked is not None and stages not in asked:
            return False
    return (a in rule.get("types", (a,)) and acc in rule.get("acc", (acc,))
            and rule.get("m", big)[0] <= m <= rule.get("m", big)[1]
            and rule.get("n", big)[0] <= n <= rule.get("n", big)[1]
            and rule.get("elements", big)[0] <= m * n * k <= rule.get("elements", big)[1])


def _kernel_name(ttir):
    import re
    m = re.search(r"tt\.func\s+(?:public\s+)?@([\w$.]+)", ttir)
    return m.group(1) if m else "?"


# --- the helpers ---------------------------------------------------------------------------

def client():
    """volta_forever_triton_client of this file's directory (how a program starts its helpers and
    what it takes from them), read once, at its first use, without writing its bytecode."""
    module = _state.get("client_module")
    if module is not None:
        return module
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "volta_forever_triton_client.py")
    spec = importlib.util.spec_from_file_location("volta_forever_triton_client", path)
    module = importlib.util.module_from_spec(spec)
    module._volta_forever_core = sys.modules[__name__]
    written = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = written
    with _lock:
        _state.setdefault("client_module", module)
        return _state["client_module"]


def _marked():
    """Whether the keys of this process have the mark now: the switch is on, and the process has
    not given it up (no helper could be used: then it compiles as without the switch, and
    Triton's cache keeps what it compiles as it keeps it without the switch)."""
    on, mark = active()
    return (mark if on and not _state.get("broken") else None)


def _not_kept(metadata):
    """The compilation of this metadata gives the program the kernel of Triton's own stages after a
    failure that may be otherwise another time (the tool did not run, a full disk): Triton's cache
    keeps its files but not the group that makes the next compilation take them."""
    key = metadata.get("hash")
    if not key:
        return
    names = {key}
    try:
        # (The name of the group in Triton's cache: the hash in base 32, in Triton 3.3 to 3.9.)
        import base64
        names.add(base64.b32encode(bytes.fromhex(key)).decode("utf-8").rstrip("="))
    except ValueError:
        pass
    with _lock:
        _state.setdefault("not_kept", set()).update(names)


def _replace_stages(backend, stages, options, allowed):
    """In place of Triton's stages ttgir, llir, ptx and cubin: the first compiles the kernel for
    sm_80 when it may and returns its TTGIR, the others what the sm_80 compilation made; or, for
    any other kernel, Triton's own stages."""
    original = dict(stages)
    made = {}

    def ttgir(src, metadata):
        name = "?"
        try:
            ttir = str(src)
            name = _kernel_name(ttir)
            why = eligible(ttir, allowed)
            if why is None:
                svc = client()
                said = False
                try:
                    result, why, transient = svc.compile80_isolated(backend, ttir, metadata, options)
                except svc.HelperFailed as e:
                    # A crash or a time out of the kernel's compilation: Triton's own stages, kept in
                    # its cache (not tried again), said at the default level; a helper that could not
                    # start: not kept.
                    result, why, transient = None, e.why, not e.kernel
                    if e.kernel:
                        say("Triton kernel %s: compiled for sm_75: %s" % (name, why))
                        said = True
                if transient and getattr(_keys, "marked", True):
                    _not_kept(metadata)
                if result is not None:
                    metadata.update(result["metadata"])
                    made.update(result)
                    asked = max(1, options.num_stages)
                    fewer = "" if result["stages"] == asked else \
                        ", %d stages of the %d asked, for its shared memory" % (result["stages"], asked)
                    say("Triton kernel %s: compiled for sm_80%s" % (name, fewer), level=2)
                    # (Its TTGIR as text: Triton keeps what a stage gives as text, and the next stage
                    # is this module's; nothing of the sm_80 compilation is parsed in this process.)
                    return result["ttgir"]
                if transient:
                    # (When no helper can be used in this process, or one is still starting, or
                    # no slot is free, that was said once: each kernel's line is for VF_LOG_LEVEL=2.)
                    say("Triton kernel %s: compiled for sm_75, not kept in Triton's cache: %s" % (name, why),
                        level=2 if _state.get("broken") or _state.get("quiet") else 1)
                elif not said:
                    say("Triton kernel %s: compiled for sm_75: %s" % (name, why), level=2)
            else:
                say("Triton kernel %s: compiled for sm_75: %s" % (name, why), level=3)
        except Exception as e:  # noqa: BLE001 (Triton's own compilation, whatever went wrong here)
            made.clear()
            if isinstance(e, (OSError, MemoryError)):
                _not_kept(metadata)
            first = str(e).strip().splitlines()[0][:300] if str(e).strip() else ""
            say("Triton kernel %s: compiled for sm_75 after an error of its sm_80 compilation: %s: %s"
                % (name, type(e).__name__, first))
        return original["ttgir"](src, metadata)

    def after(stage):
        def run(src, metadata):
            if made:
                return made[stage]
            return original[stage](src, metadata)
        run.__qualname__ = "volta_forever_triton.%s" % stage
        return run

    stages["ttgir"] = ttgir
    for stage in ("llir", "ptx", "cubin"):
        stages[stage] = after(stage)


def _flags():
    """Triton's settings that read or replace what a stage makes (TRITON_KERNEL_OVERRIDE,
    USE_IR_LOC, TRITON_DISABLE_LINE_INFO, LLVM_EXTRACT_DI_LOCAL_VARIABLES): from its knobs, or
    from the environment for a Triton without them (3.3)."""
    try:
        from triton import knobs
        comp = knobs.compilation
        return (comp.override, comp.use_ir_loc, comp.disable_line_info,
                getattr(comp, "dump_ir_extract_di_local_variables", False))
    except ImportError:
        env = os.environ.get
        return (env("TRITON_KERNEL_OVERRIDE", "0") == "1", env("USE_IR_LOC"),
                env("TRITON_DISABLE_LINE_INFO", "0") != "0", env("LLVM_EXTRACT_DI_LOCAL_VARIABLES", "0") == "1")


def _triton_stage(fn):
    """A stage of Triton's own add_stages (not one that a hook put there)."""
    return getattr(fn, "__qualname__", "").endswith("add_stages.<locals>.<lambda>")


# The options of ptxas that a compilation may give it (Triton's ptx_options, from PTXAS_OPTIONS or
# the kernel's): those that Triton itself puts on ptxas's command line in 3.3 to 3.9, the count of
# registers and the warnings; none names a file. A value of the form NAME=N takes digits.
PTXAS_FLAGS = ("-v", "--verbose", "-lineinfo", "--generate-line-info", "-suppress-debug-info", "-g", "--device-debug",
               "--fmad=true", "--fmad=false", "-O0", "-O1", "-O2", "-O3", "-warn-spills", "--warn-on-spills",
               "-warn-lmem-usage", "--warn-on-local-memory-usage", "--warn-on-double-precision-use")
PTXAS_NUMBERS = ("--opt-level", "--regAllocOptLevel", "--maxrregcount", "-maxrregcount")


def options_problem(options, triton_dir=None, default_libs=()):
    """Why a compilation with these options (Triton's CUDAOptions, or the dict of a request) is not
    compiled by a helper, or None: an IR given in place of a stage (a path), an instrumentation,
    options of ptxas that are not of PTXAS_FLAGS and PTXAS_NUMBERS (no --options-file, no path),
    libraries to link (extern_libs) that are not files of Triton's own directory or the ones Triton
    takes by default (default_libs). Each text is bounded before it is looked at (a path of 4,096
    bytes at most before realpath, whose cost grows with its components)."""
    get = options.get if isinstance(options, dict) else (lambda name, value=None: getattr(options, name, value))
    if get("ir_override"):
        return "its compilation gives an IR in place of a stage (ir_override)"
    if get("instrumentation_mode"):
        return "its compilation is instrumented (instrumentation_mode)"
    ptx = get("ptx_options")
    if ptx:
        if not isinstance(ptx, str) or len(ptx) > 4096:
            return "its ptx_options are not a text of 4,096 characters at most"
        words = ptx.split(" ")
        i = 0
        while i < len(words):
            w, name = words[i], words[i].split("=", 1)[0]
            if w in PTXAS_FLAGS:
                i += 1
            elif name in PTXAS_NUMBERS and "=" in w and w.split("=", 1)[1].isdigit():
                i += 1
            elif w in PTXAS_NUMBERS and i + 1 < len(words) and words[i + 1].isdigit():
                i += 2
            else:
                return "its ptx_options give ptxas %r, which a helper does not give it" % w[:80]
    libs = get("extern_libs")
    if libs:
        try:
            pairs = [(name, path) for name, path in (libs.items() if isinstance(libs, dict) else libs)]
        except (TypeError, ValueError):
            return "its extern_libs are not pairs of a name and a file"
        if len(pairs) > 64:
            return "its extern_libs name %d files" % len(pairs)
        if triton_dir is None:
            triton = sys.modules.get("triton")
            triton_dir = os.path.dirname(getattr(triton, "__file__", "") or "") or None
        root = os.path.realpath(triton_dir) + os.sep if triton_dir else None
        given = {(str(n), str(p)) for n, p in default_libs or ()}
        for name, path in pairs:
            if not isinstance(name, str) or not isinstance(path, str) or len(name) > 256 or len(path) > 4096:
                return "its extern_libs are not pairs of a name and a file"
            if (name, path) in given:
                continue
            real = os.path.realpath(path)
            if root is None or not real.startswith(root) or not os.path.isfile(real):
                return "its extern_libs name %s, which is not a file of Triton's directory" % path[:200]
    return None


def _default_libs(cls):
    """The libraries that Triton's options link by default (libdevice: Triton's, or that of
    TRITON_LIBDEVICE_PATH), for options of class cls; () when they cannot be made."""
    with _lock:
        known = _state.setdefault("default_libs", {})
        if cls in known:
            return known[cls]
    try:
        libs = tuple(tuple(p) for p in (getattr(cls(), "extern_libs", None) or ()))
    except Exception:  # noqa: BLE001
        libs = ()
    with _lock:
        return known.setdefault(cls, libs)


def applies(backend, stages, options, language):
    """Whether the stages of this compilation may be replaced: told 7.5 and not given another
    architecture (TRITON_OVERRIDE_ARCH), a Triton kernel (not Gluon), Triton's own stages, one CTA,
    none of the settings that read or replace what a stage makes, no instrumentation."""
    target = backend.target
    if getattr(target, "backend", None) != "cuda" or getattr(target, "arch", None) != 75:
        return False
    if getattr(options, "arch", None) != "sm75":
        return False
    if language is not None and getattr(language, "name", "TRITON") != "TRITON":
        return False
    if not active()[0] or _state.get("broken"):
        return False   # (a process that can use no helper compiles as without the switch)
    if list(stages) != ["ttir", "ttgir", "llir", "ptx", "cubin"] or not all(_triton_stage(stages[k]) for k in stages):
        say("Triton: the stages of a compilation (%s) are not the ones this layer knows: such kernels are compiled "
            "for sm_75" % ", ".join(stages), once="stages")
        return False
    why = options_problem(options, default_libs=_default_libs(type(options)))
    if why:
        if not (getattr(options, "ir_override", None) or getattr(options, "instrumentation_mode", "")):
            say("Triton: a kernel is compiled for sm_75 when %s" % why, once="options:" + why)
        return False
    if getattr(options, "num_ctas", 1) != 1 or getattr(type(backend), "instrumentation", None):
        return False
    return not any(_flags())


# --- the patch -------------------------------------------------------------------------------

def _known(cls):
    """None when CUDABackend has the methods and parameters that this module calls, else what
    differs."""
    import inspect
    want = {
        "make_ttgir": ["mod", "metadata", "opt", "capability"],
        "make_llir": ["self", "src", "metadata", "options", "capability"],
        "make_ptx": ["self", "src", "metadata", "opt", "capability"],
        "make_cubin": ["self", "src", "metadata", "opt", "capability"],
    }
    for name, params in want.items():
        fn = getattr(cls, name, None)
        if fn is None:
            return "CUDABackend.%s is missing" % name
        try:
            got = list(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            return "CUDABackend.%s has no signature" % name
        if got != params:
            return "CUDABackend.%s takes %s" % (name, ", ".join(got))
    try:
        got = list(inspect.signature(cls.add_stages).parameters)
    except (TypeError, ValueError, AttributeError):
        return "CUDABackend.add_stages is missing"
    if got not in (["self", "stages", "options"], ["self", "stages", "options", "language"]):
        return "CUDABackend.add_stages takes %s" % ", ".join(got)
    if not callable(getattr(cls, "hash", None)):
        return "CUDABackend.hash is missing"
    options = getattr(getattr(cls, "parse_options", None), "__globals__", {}).get("CUDAOptions")
    fields = getattr(options, "__dataclass_fields__", {})
    for field in ("arch", "num_stages"):
        if field not in fields:
            return "CUDAOptions has no %s" % field
    return None


def patch(module):
    """Patch a module of Triton that has just been run (TARGETS)."""
    if getattr(module, "__name__", "") == "triton.backends":
        _patch_backend(module)
    else:
        _patch_key(module)


def _patch_key(module):
    """Wrap the triton_key that this module defines (not one it imported), once: the mark at its
    end while this is active and Triton's backend was taken; and keep its FileCacheManager from
    writing the group of a compilation that _not_kept names."""
    _patch_cache(module)
    key = module.__dict__.get("triton_key")
    if key is None or getattr(key, "_volta_forever", False) or getattr(key, "__module__", None) != module.__name__:
        return

    def triton_key():
        k = key()
        try:
            mark = _marked()
            if mark and _state.get("backend_ok"):
                return "%s-%s" % (k, mark)
        except Exception:  # noqa: BLE001
            pass
        return k

    triton_key.__wrapped__ = key
    triton_key.__doc__ = key.__doc__
    triton_key._volta_forever = True
    module.triton_key = triton_key


def _patch_cache(module):
    cls = module.__dict__.get("FileCacheManager")
    put_group = getattr(cls, "put_group", None)
    if cls is None or not callable(put_group) or cls.__dict__.get("_volta_forever", False):
        return

    def vf_put_group(self, filename, group):
        if getattr(self, "key", None) in _state.get("not_kept", ()):
            return None
        return put_group(self, filename, group)

    vf_put_group.__wrapped__ = put_group
    cls.put_group = vf_put_group
    cls._volta_forever = True


def _triton_files(cls=None):
    """What the files of this process's Triton with which a helper compiles are: [device, inode,
    size, modification time in ns] of its library (triton._C.libtriton, the file that Python
    mapped) and of the module of its NVIDIA backend (CUDABackend's, cls), None for one that cannot
    be read. Recorded when Triton is imported, and compared with those of a helper's Triton
    (volta_forever_triton_client._identity): a Triton replaced or changed in place under a running
    program with the same version (another build reinstalled, an editable install edited, an
    environment made again) is not taken. Two stat() calls."""
    if cls is None:
        try:
            from triton.backends import backends
            cls = backends["nvidia"].compiler
        except Exception:  # noqa: BLE001
            cls = None
    paths = (getattr(sys.modules.get("triton._C.libtriton"), "__file__", None),
             getattr(getattr(cls, "make_cubin", None), "__globals__", {}).get("__file__"))
    out = []
    for path in paths:
        try:
            st = os.stat(path)
            out.append([st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns])
        except (OSError, TypeError, ValueError):
            out.append(None)
    return out


# The files of _triton_files, as a line names them.
TRITON_FILES = ("its library (triton._C.libtriton)", "its NVIDIA backend (CUDABackend's module)")


def _patch_backend(module):
    """Patch the class CUDABackend of the table of Triton's backends, once."""
    nvidia = getattr(module, "backends", {}).get("nvidia")
    cls = getattr(nvidia, "compiler", None)
    if cls is None or getattr(cls, "_volta_forever", False):
        return
    try:
        import triton
        version = getattr(triton, "__version__", "?")
    except Exception:  # noqa: BLE001 (triton is being imported: its version is set before its backends)
        version = "?"
    _state["version"] = version
    why = "the backend nvidia has %s, not CUDABackend" % cls.__name__ if cls.__name__ != "CUDABackend" else _known(cls)
    if why:
        say("Triton %s is not one this layer knows (%s): VF_TRITON_TC is not used" % (version, why), once="unknown")
        return
    add_stages = cls.add_stages
    backend_hash = cls.hash
    _state["files"] = _triton_files(cls)

    def vf_add_stages(self, stages, options, *rest):
        add_stages(self, stages, options, *rest)
        try:
            if applies(self, stages, options, rest[0] if rest else None):
                _replace_stages(self, stages, options, _state["types"])
        except Exception as e:  # noqa: BLE001
            say("Triton: its stages were left as they are after an error: %s: %s" % (type(e).__name__, e),
                once="add_stages")

    def vf_hash(self):
        h = backend_hash(self)
        try:
            target = self.target
            if target.backend == "cuda" and target.arch == 75:
                mark = _marked()
                _keys.marked = bool(mark)
                if mark:
                    return "%s-%s" % (h, mark)
        except Exception:  # noqa: BLE001
            pass
        return h

    vf_add_stages.__wrapped__ = add_stages
    vf_hash.__wrapped__ = backend_hash
    cls.add_stages = vf_add_stages
    cls.hash = vf_hash
    cls._volta_forever = True
    _state["backend_ok"] = True


# --- the import hook -------------------------------------------------------------------------

class _Loader:
    """The loader of a hooked module: it runs the module, then patches it."""

    def __init__(self, loader):
        self._loader = loader

    def __getattr__(self, name):
        return getattr(self._loader, name)

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        try:
            patch(module)
        except Exception as e:  # noqa: BLE001
            say("Triton was left as it is after an error: %s: %s" % (type(e).__name__, e), once="patch")


class Finder:
    """On sys.meta_path: finds a hooked module with the finders after it, and wraps its loader."""
    volta_forever_triton = True

    def find_spec(self, name, path=None, target=None):
        if name not in TARGETS:
            return None
        for finder in sys.meta_path:
            if finder is self or getattr(finder, "volta_forever_triton", False):
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            try:
                spec = find(name, path, target)
            except Exception:  # noqa: BLE001
                continue
            if spec is not None:
                break
        else:
            return None
        if spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        spec.loader = _Loader(spec.loader)
        return spec


def install():
    """Install the import hook once; patch at once when Triton's backends are imported already."""
    if not any(getattr(f, "volta_forever_triton", False) for f in sys.meta_path):
        sys.meta_path.insert(0, Finder())
    for name in TARGETS:
        module = sys.modules.get(name)
        if module is not None:
            patch(module)

