"""volta-forever: run by Python when it starts (its module site), from the directory that the
launcher puts first in PYTHONPATH when VF_TRITON_TC is set (INSTALL.md).

It installs an import hook on three modules of Triton; when Triton imports the first of them,
volta_forever_triton is read and patches them, once in a process. Then it runs the sitecustomize
that this one hides, as Python would have run it without this one: the first that the finders of
sys.meta_path give when no directory of the layer's python/ is on sys.path (a distribution's, a
virtual environment's, one that an editable install's finder gives, a user's own). Python runs no
sitecustomize with -S or -I, and reads no PYTHONPATH with -E: Triton then compiles as without it.
importlib.reload of the hidden module runs its code again in the same module object, and leaves
nothing of this one in it. On a Python that has no sitecustomize of its own, this module stays in
sys.modules under that name, without a public name. It is short on purpose: Python compiles it
from its source at every start (its directory holds no bytecode, and nothing is written there).
"""
import os as _volta_forever_os
import sys as _volta_forever_sys


def _volta_forever_triton():
    """Install the hook, once in a process; False when this file runs again (importlib.reload,
    another copy of it). Nothing is imported here but what Python has imported already."""
    os, sys = _volta_forever_os, _volta_forever_sys
    if getattr(sys, "_volta_forever_site", None) is not None:
        return False
    here = sys._volta_forever_site = os.path.dirname(os.path.abspath(__file__))
    if os.environ.get("VF_TRITON_TC", "").strip() not in ("", "0") and os.environ.get("VF_SAFE") != "1" \
            and sys.version_info >= (3, 9) and not any(getattr(f, "volta_forever_triton", False) for f in sys.meta_path):
        sys.meta_path.insert(0, _volta_forever_Finder(here))
    return True


class _volta_forever_Finder(object):
    """On sys.meta_path: for a module of Triton that is patched, the finder of volta_forever_triton,
    which finds it with the finders after this one and wraps its loader."""
    volta_forever_triton = True
    targets = ("triton.backends", "triton.runtime.cache", "triton.compiler.compiler")

    def __init__(self, here):
        self.here = here

    def find_spec(self, name, path=None, target=None):
        if name not in self.targets:
            return None
        try:
            return _volta_forever_module(self.here).Finder().find_spec(name, path, target)
        except Exception as e:  # noqa: BLE001 (Triton is left as it is)
            sys = _volta_forever_sys
            if not getattr(sys, "_volta_forever_said", False):
                sys._volta_forever_said = True
                sys.stderr.write("[volta-forever] VF_TRITON_TC is not used: %s: %s\n" % (type(e).__name__, e))
            return None


def _volta_forever_module(here):
    """volta_forever_triton of this file's directory, read once, without writing its bytecode."""
    sys = _volta_forever_sys
    module = sys.modules.get("volta_forever_triton")
    path = _volta_forever_os.path.join(here, "volta_forever_triton.py")
    if module is not None and getattr(module, "__file__", None) == path:
        return module
    import importlib.util
    spec = importlib.util.spec_from_file_location("volta_forever_triton", path)
    module = importlib.util.module_from_spec(spec)
    written, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        sys.modules["volta_forever_triton"] = module
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("volta_forever_triton", None)
        raise
    finally:
        sys.dont_write_bytecode = written
    return module


def _volta_forever_hidden():
    """The spec of the sitecustomize that this one hides, or None: what the finders give while no
    copy of the layer's python/ is on sys.path (two would run each other without end) and no module
    of that name is in sys.modules (both are put back after)."""
    os, sys = _volta_forever_os, _volta_forever_sys
    path = list(sys.path)
    current = sys.modules.pop("sitecustomize", None)
    try:
        kept = []
        for entry in path:
            try:
                if isinstance(entry, str) and os.path.isfile(os.path.join(entry or os.curdir, "volta_forever_triton.py")):
                    continue
            except (OSError, ValueError, TypeError):
                pass
            kept.append(entry)
        sys.path[:] = kept
        for finder in list(sys.meta_path):
            find = getattr(finder, "find_spec", None)
            if find is None or getattr(finder, "volta_forever_triton", False):
                continue
            try:
                spec = find("sitecustomize", None, None)
            except Exception:  # noqa: BLE001 (a finder that fails finds nothing here)
                spec = None
            if spec is not None and spec.loader is not None:
                return spec
        return None
    finally:
        sys.path[:] = path
        if current is not None:
            sys.modules["sitecustomize"] = current


def _volta_forever_chain(first):
    sys = _volta_forever_sys
    spec = _volta_forever_hidden()
    if spec is None:
        return
    if first:
        # The hidden module, as Python would have made it: `import sitecustomize` gives it.
        try:
            from _frozen_importlib import module_from_spec
        except ImportError:
            from importlib.util import module_from_spec
        module = sys.modules["sitecustomize"] = module_from_spec(spec)
        spec.loader.exec_module(module)
        return
    # Run again: the hidden module's code, in the module object being run, which then holds
    # nothing of this file.
    module = sys.modules.get("sitecustomize")
    if module is None:
        return
    ns = vars(module)
    mine = {k: v for k, v in ns.items() if k.startswith("_volta_forever_")}
    for attr, value in (("__spec__", spec), ("__loader__", spec.loader), ("__file__", spec.origin),
                        ("__cached__", getattr(spec, "cached", None))):
        try:
            setattr(module, attr, value)
        except (AttributeError, TypeError):
            pass
    try:
        spec.loader.exec_module(module)
    finally:
        for k, v in mine.items():
            if ns.get(k) is v:
                del ns[k]


_volta_forever_chain(_volta_forever_triton())
