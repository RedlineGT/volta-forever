#!/usr/bin/env python3
"""Download NVIDIA redistributable archives listed in a redistrib manifest,
verify their SHA-256 and unpack them into one merged tree.

usage: fetch-redist.py BASE_URL MANIFEST DL_DIR TREE_DIR [--skip REGEX] [--only REGEX]
                       [--platform linux-x86_64] [--variant NAME] [--runtime] [--discard]
                       [--allow-http]

BASE_URL   e.g. https://developer.download.nvidia.com/compute/cuda/redist
MANIFEST   a local JSON file or an https:// URL to one
DL_DIR     where the archives are kept (one file per component)
TREE_DIR   merged tree (bin/, include/, lib/, nvvm/ ...), like /usr/local/cuda
--variant  for manifests with per-CUDA variants (cuDNN: cuda12 / cuda13)
--runtime  unpack the shared libraries (lib/*.so*) and the licence alone: what a program
           needs to run, without the headers and the static libraries
--discard  remove each archive once it is unpacked; what TREE_DIR/COMPONENTS.tsv records
           as unpacked is then not downloaded again
--allow-http
           take a MANIFEST given by an http:// URL, and follow a redirection from HTTPS to
           HTTP (refused otherwise, for the manifest and for every download: the manifest holds
           the SHA-256 that the archives are checked against, and over plain HTTP whoever is on
           the way can change both)

Needs: Python 3, GNU tar. Uses aria2c when available; falls back to Python's own
urllib (slower, single-threaded). Re-runnable: verified archives are not downloaded again.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import zipfile


# NVIDIA publishes NCCL separately from its CUDA redistrib manifest.
NCCL_VERSION = "2.32.6"
NCCL_WHEEL = {
    "relative_path": "nvidia_nccl_cu12-2.32.6-py3-none-manylinux_2_27_x86_64.whl",
    "base_url": "https://files.pythonhosted.org/packages/95/9f/ba87c8199a53ea3058640607c242174c8537e9e64d7a17991f332cd1c391",
    "sha256": "2a6e6d5ddfc08970eca872be3dfa2650b24fa10a2043974465a9cae9c64ed6d8",
    "size": 336541086,
}

# What is being unpacked is written in a directory of TREE_DIR whose name starts with this,
# then moved into place: no program loads from there, and no name pattern of the launcher
# matches it. (Earlier releases wrote the NCCL runtime as lib/libnccl.so.2.vf-part.)
STAGE_PREFIX = ".vf-stage-"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _has_aria2c():
    return shutil.which("aria2c") is not None


def redirection_refused(origin, target, allow_http):
    """Why a redirection from the URL origin to the URL target is not followed, or None: from
    HTTPS to HTTPS only, or also to HTTP with allow_http; from HTTP to HTTP or HTTPS; never to
    another scheme. (What comes over plain HTTP, whoever is on the way can change: a manifest,
    and with it the SHA-256 that the archives are checked against.)"""
    from urllib.parse import urlsplit
    scheme = urlsplit(target).scheme.lower()
    if scheme == "https" or (scheme == "http" and (allow_http or urlsplit(origin).scheme.lower() == "http")):
        return None
    return f"{origin}: the redirection to {target} is refused" + (
        " (--allow-http takes plain HTTP)" if scheme == "http" else "")


def urlopen(url, allow_http):
    """urllib's urlopen, whose redirections are only those that redirection_refused lets through:
    another ends the program."""
    import urllib.request

    class Redirections(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            refused = redirection_refused(req.full_url, newurl, allow_http)
            if refused:
                raise SystemExit(refused)
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    return urllib.request.build_opener(Redirections).open(url)


def download(url, dest_dir, dest_name, expected_sha256, lock_fds, allow_http=False):
    """Download url to dest_dir/dest_name, verify SHA-256. Returns the path."""
    dest = os.path.join(dest_dir, dest_name)
    if _has_aria2c():
        # aria2c cannot be told which redirections to follow: those it followed are read from its
        # log, and an archive that came by one that is refused is removed, and the program ends.
        log = os.path.join(dest_dir, f".dl-{dest_name}.log")
        try:
            subprocess.run(
                ["aria2c", "-q", "-x", "8", "-s", "8", "--allow-overwrite=true",
                 "--auto-file-renaming=false", f"--log={log}", "--log-level=notice",
                 "-d", dest_dir, "-o", dest_name, url],
                check=True, pass_fds=lock_fds,
            )
            with open(log, errors="replace") as lines:
                targets = [line.split("Redirecting to ", 1)[1].strip() for line in lines if "Redirecting to " in line]
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(log)
        origin = url
        for target in targets:
            refused = redirection_refused(origin, target, allow_http)
            if refused:
                os.unlink(dest)
                sys.exit(refused)
            origin = target
    else:
        # Fall back to Python's urllib: download to a temporary file, rename on success.
        tmp_fd, tmp_path = tempfile.mkstemp(dir=dest_dir, prefix=".dl-")
        try:
            with os.fdopen(tmp_fd, "wb") as tmp_f:
                with urlopen(url, allow_http) as resp:
                    while True:
                        chunk = resp.read(1 << 22)
                        if not chunk:
                            break
                        tmp_f.write(chunk)
            os.rename(tmp_path, dest)
        except BaseException:
            os.unlink(tmp_path)
            raise
    got = sha256(dest)
    if got != expected_sha256:
        sys.exit(f"SHA-256 mismatch for {dest}: {got} != {expected_sha256}")
    return dest


def load_manifest(path_or_url, allow_http=False):
    """Load a JSON manifest from a local file or an https:// URL (http:// with allow_http)."""
    if path_or_url.startswith("http://") and not allow_http:
        sys.exit(f"{path_or_url}: a manifest over plain HTTP is refused, since it holds the SHA-256 "
                 "that the archives are checked against (--allow-http takes it)")
    if path_or_url.startswith("https://") or path_or_url.startswith("http://"):
        with urlopen(path_or_url, allow_http) as resp:
            return json.loads(resp.read())
    return json.load(open(path_or_url))


def stop(signum, frame):
    raise SystemExit(128 + signum)


def main():
    # SIGTERM and SIGHUP end a fetch as Ctrl-C does, by an exception: what it staged is removed.
    for signum in (signal.SIGTERM, signal.SIGHUP):
        if signal.getsignal(signum) == signal.SIG_DFL:
            signal.signal(signum, stop)
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("manifest", help="local JSON file or https:// URL")
    ap.add_argument("dl_dir")
    ap.add_argument("tree_dir")
    ap.add_argument("--skip", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--platform", default="linux-x86_64")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--runtime", action="store_true")
    ap.add_argument("--discard", action="store_true")
    ap.add_argument("--with-nccl", action="store_true", help="include NVIDIA's CUDA 12 NCCL runtime wheel among selectable components")
    ap.add_argument("--allow-http", action="store_true",
                    help="take a manifest given by an http:// URL, and redirections from HTTPS to HTTP")
    a = ap.parse_args()
    if a.with_nccl and not a.runtime:
        ap.error("--with-nccl requires --runtime")

    if not _has_aria2c():
        print("note: aria2c not found; using Python's urllib (slower)", file=sys.stderr)

    manifest = load_manifest(a.manifest, a.allow_http)
    if a.with_nccl:
        manifest["libnccl"] = {"version": NCCL_VERSION, "linux-x86_64": NCCL_WHEEL}
    os.makedirs(a.dl_dir, exist_ok=True)
    os.makedirs(a.tree_dir, exist_ok=True)
    rows = []
    for name, comp in sorted(manifest.items()):
        if not isinstance(comp, dict) or a.platform not in comp:
            continue
        if a.skip and re.search(a.skip, name):
            continue
        if a.only and not re.search(a.only, name):
            continue
        entry = comp[a.platform]
        if "relative_path" not in entry:  # per-variant entries
            if a.variant is None or a.variant not in entry:
                print(f"skip {name}: variants {sorted(entry)} (use --variant)", file=sys.stderr)
                continue
            entry = entry[a.variant]
        rows.append((name, comp.get("version", "?"), entry))

    # Both the merged install tree and the archive cache may be shared by fetches.
    # Lock them in a consistent order, including when two installs share just their cache.
    with contextlib.ExitStack() as locks:
        directories = sorted({os.path.realpath(a.tree_dir), os.path.realpath(a.dl_dir)})
        lock_fds = []
        for directory in directories:
            lock = locks.enter_context(open(os.path.join(directory, ".vf-fetch.lock"), "a"))
            fcntl.flock(lock, fcntl.LOCK_EX)
            lock_fds.append(lock.fileno())
        # The active child keeps the locks if the parent is killed mid-download or extraction.
        install_components(a, rows, lock_fds)
    print(f"done: {len(rows)} components -> {a.tree_dir}")


def read_receipts(path):
    receipts = {}
    try:
        with open(path) as record:
            for line in record:
                fields = line.rstrip("\n").split("\t")
                if len(fields) == 4:
                    # Older receipts guarantee the runtime subset only.
                    fields.append("runtime")
                if len(fields) == 5:
                    receipts[fields[0]] = fields
    except FileNotFoundError:
        pass
    return receipts


def write_receipts(path, receipts):
    # Writers hold the tree lock. Publish a complete snapshot so a killed fetch
    # leaves either the previous receipts or the new ones available to its retry.
    tmp = path + ".tmp"
    with open(tmp, "w") as record:
        for fields in receipts.values():
            record.write("\t".join(fields) + "\n")
        record.flush()
        os.fsync(record.fileno())
    os.replace(tmp, path)


def remove_leftovers(tree, dl_dir):
    """What an interrupted fetch left. The caller holds the locks: nothing else writes there."""
    for stage in Path(tree).glob(STAGE_PREFIX + "*"):
        shutil.rmtree(stage, ignore_errors=True)
    for part in [*Path(tree).glob("*.vf-part"), *Path(tree, "lib").glob("*.vf-part")]:
        part.unlink(missing_ok=True)
    for part in Path(dl_dir).glob(".dl-*"):
        part.unlink(missing_ok=True)


def move_into_place(stage, tree):
    """Move each file staged under stage to the same place under tree, by a rename: the regular
    files first, then the symbolic links, so that a link does not name a file that is not
    there yet."""
    links = []
    for directory, subdirectories, names in os.walk(stage):
        target = os.path.join(tree, os.path.relpath(directory, stage))
        os.makedirs(target, exist_ok=True)
        for name in [n for n in subdirectories if os.path.islink(os.path.join(directory, n))]:
            subdirectories.remove(name)
            names.append(name)
        for name in names:
            source = os.path.join(directory, name)
            if os.path.islink(source):
                links.append((source, os.path.join(target, name)))
            else:
                os.replace(source, os.path.join(target, name))
    for source, dest in links:
        os.replace(source, dest)


def stage_nccl_runtime(wheel, stage, version):
    files = [
        (f"nvidia_nccl_cu12-{version}.dist-info/licenses/License.txt", "NCCL-LICENSE.txt"),
        ("nvidia/nccl/lib/libnccl.so.2", "lib/libnccl.so.2"),
    ]
    stage = Path(stage)
    (stage / "lib").mkdir()
    with zipfile.ZipFile(wheel) as archive:
        for member, dest in files:
            with archive.open(member) as src, (stage / dest).open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)


def install_components(a, rows, lock_fds):
    remove_leftovers(a.tree_dir, a.dl_dir)
    record = os.path.join(a.tree_dir, "COMPONENTS.tsv")
    receipts = read_receipts(record)
    requested = "runtime" if a.runtime else "full"
    extract = ["--strip-components=1"]
    if a.runtime:
        extract += ["--wildcards", "--exclude=*/lib/stubs/*", "*/lib/*.so*", "*/LICENSE"]

    # (Said before anything is downloaded: tar starts xz for these archives.)
    if any(entry["relative_path"].endswith(".xz") for _, _, entry in rows) and shutil.which("xz") is None:
        sys.exit("xz is needed to unpack NVIDIA's archives and was not found "
                 "(Debian and Ubuntu: the package xz-utils; Fedora and RHEL: xz)")

    for name, version, entry in rows:
        url = f"{entry.get('base_url', a.base_url)}/{entry['relative_path']}"
        dest = os.path.join(a.dl_dir, os.path.basename(entry["relative_path"]))
        expected = entry["sha256"]
        prior = receipts.get(name)
        same_archive = prior and prior[3] == expected
        if a.discard and same_archive and prior[4] in (requested, "full"):
            print(f"have {name} {version} (unpacked before)", flush=True)
            continue
        if not (os.path.exists(dest) and sha256(dest) == entry["sha256"]):
            print(f"get  {name} {version} ({int(entry['size']) / 1e6:.1f} MB)", flush=True)
            download(url, a.dl_dir, os.path.basename(dest), expected, lock_fds, a.allow_http)
        else:
            print(f"have {name} {version}", flush=True)
        # Unpacked in a directory of the tree, then moved into place: a failure on the way (a
        # full disk, a damaged archive) leaves the tree as it was. Writers hold the tree lock;
        # what a killed fetch staged (the 421 MB NCCL runtime, say), the next one removes.
        stage = tempfile.mkdtemp(prefix=STAGE_PREFIX, dir=a.tree_dir)
        try:
            if name == "libnccl" and dest.endswith(".whl"):
                stage_nccl_runtime(dest, stage, version)
            elif subprocess.run(["tar", "-xf", dest, "-C", stage] + extract, pass_fds=lock_fds).returncode != 0:
                sys.exit(f"cannot unpack {dest} into {a.tree_dir}")
            # The files of the previous version are replaced from here: its receipt goes first.
            if prior:
                del receipts[name]
                write_receipts(record, receipts)
            move_into_place(stage, a.tree_dir)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        installed = "full" if same_archive and prior[4] == "full" else requested
        receipts[name] = [name, version, os.path.basename(dest), expected, installed]
        write_receipts(record, receipts)
        if a.discard:
            os.remove(dest)


if __name__ == "__main__":
    main()
