//! A Bazel application can resolve the whole CUDA toolkit by directory and then append
//! "bin/ptxas". Preserve that lookup and the original resources using directory symlinks.
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{DirBuilderExt, symlink};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// What a view rebuilds of the runfiles: its directories, and its links (path in the view, what
/// it points to).
type Layout = (BTreeSet<PathBuf>, BTreeMap<PathBuf, PathBuf>);

/// A view not used for this long, that cannot work, is removed (`remove_stale`).
const UNUSED: Duration = Duration::from_secs(24 * 60 * 60);

pub fn view(source: &Path, compiler: &Path, cache: &Path) -> io::Result<PathBuf> {
    let source = source.canonicalize()?;
    let Some((directories, links)) = layout(&source, compiler)? else {
        return Ok(source);
    };
    let cache = match private_cache(cache) {
        Ok(()) => cache.to_path_buf(),
        // A root-squashed NFS home or a read-only cache may still hold the engine.
        // A per-user private directory on the local system can hold its resource links.
        Err(e)
            if matches!(
                e.kind(),
                io::ErrorKind::PermissionDenied | io::ErrorKind::ReadOnlyFilesystem
            ) =>
        {
            let local = std::env::temp_dir().join(format!("volta-forever-runfiles-{}", effective_uid()?));
            private_cache(&local)?;
            local
        }
        Err(e) => return Err(e),
    };
    let final_dir = cache.join(name(&links));
    if final_dir.is_dir() {
        touch(&final_dir);
        return Ok(final_dir);
    }
    remove_stale(&cache);
    let staging = (0..)
        .find_map(|i: u32| {
            let p = cache.join(format!(".tmp-{}-{i}", std::process::id()));
            match fs::DirBuilder::new().mode(0o700).create(&p) {
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => None,
                result => Some(result.map(|()| p)),
            }
        })
        .unwrap()?;
    let populate = || -> io::Result<()> {
        for dir in directories {
            if !dir.as_os_str().is_empty() {
                fs::create_dir(staging.join(dir))?;
            }
        }
        for (rel, target) in links {
            symlink(target, staging.join(rel))?;
        }
        Ok(())
    };
    if let Err(error) = populate() {
        fs::remove_dir_all(staging)?;
        return Err(error);
    }
    if let Err(error) = fs::rename(&staging, &final_dir) {
        fs::remove_dir_all(staging)?;
        if !matches!(
            error.kind(),
            io::ErrorKind::AlreadyExists | io::ErrorKind::DirectoryNotEmpty
        ) || !final_dir.is_dir()
        {
            return Err(error);
        }
    }
    // After a successful rename, another caller may already own the staging name.
    Ok(final_dir)
}

/// The layout of the view of `source`, or None when it has no assembler but `compiler`.
fn layout(source: &Path, compiler: &Path) -> io::Result<Option<Layout>> {
    let mut compilers = BTreeSet::new();
    for entry in fs::read_dir(source)? {
        let repo = PathBuf::from(entry?.file_name());
        for bin in ["sandbox/bin/ptxas", "bin/ptxas"] {
            let rel = repo.join(bin);
            let orig = source.join(&rel);
            if orig.is_file() && orig.canonicalize()? != compiler {
                compilers.insert(rel);
            }
        }
    }
    if compilers.is_empty() {
        return Ok(None);
    }
    let directories: BTreeSet<PathBuf> = compilers
        .iter()
        .flat_map(|p| p.ancestors().skip(1))
        .map(Path::to_path_buf)
        .collect();
    let mut links = BTreeMap::new();
    for dir in &directories {
        for child in fs::read_dir(source.join(dir))? {
            let rel = dir.join(child?.file_name());
            if !directories.contains(rel.as_path()) {
                let value = if compilers.contains(&rel) {
                    compiler.to_owned()
                } else {
                    source.join(&rel)
                };
                links.insert(rel, value);
            }
        }
    }
    Ok(Some((directories, links)))
}

/// The name of the view of a layout: a hash of the complete layout, by an algorithm of its own
/// (FNV-1a, 64 bits), so that it depends on nothing else (`DefaultHasher` may change with the
/// version of Rust). Replacing a linked file is immediately visible. Adding a sibling to one of
/// the rebuilt directories changes the name on the next launch.
fn name(links: &BTreeMap<PathBuf, PathBuf>) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for (rel, target) in links {
        for part in [rel, target] {
            for &byte in part.as_os_str().as_bytes().iter().chain(&[0]) {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
            }
        }
    }
    format!("{hash:016x}")
}

/// The view `dir` is used now (see `remove_stale`).
fn touch(dir: &Path) {
    let _ = fs::File::open(dir).and_then(|dir| dir.set_modified(SystemTime::now()));
}

/// The assembler `compiler` was started by the path `argv0`: when that is its link in a view
/// (`<view>/<repo>/bin/ptxas` or `<view>/<repo>/sandbox/bin/ptxas`), the view is used now. An
/// engine whose `RUNFILES_DIR` is a view starts the assembler through it each time it compiles,
/// and may not ask for its view again for as long as it runs.
pub fn used(argv0: &Path, compiler: &Path) {
    let link = fs::symlink_metadata(argv0).is_ok_and(|m| m.file_type().is_symlink());
    if argv0.file_name() != Some(OsStr::new("ptxas")) || !link || argv0.canonicalize().ok().as_deref() != Some(compiler)
    {
        return;
    }
    let mut ancestors = argv0.ancestors().skip(1);
    if ancestors.next().and_then(Path::file_name) != Some(OsStr::new("bin")) {
        return;
    }
    // `<repo>` or `sandbox`, then the view, or `<repo>` then the view.
    let above: Vec<&Path> = ancestors.skip(1).take(2).collect();
    for view in above {
        if view.file_name().and_then(OsStr::to_str).is_some_and(view_name) {
            touch(view);
        }
    }
}

/// A name that a view can have.
fn view_name(name: &str) -> bool {
    name.len() == 16 && name.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Remove from `cache` what nothing will use again: the staging directories of processes that
/// ended before they renamed them, and the views not used for a day (`UNUSED`) that cannot work:
/// one of their links leads nowhere (the runfiles they link are gone, or the assembler is). A
/// view whose links all lead somewhere is left, whatever its name, the assembler it links or the
/// layout its runfiles have now: it may be the `RUNFILES_DIR` of an engine that runs, which does
/// not ask for its view again (an engine started by an earlier version, or before its runfiles
/// changed). What cannot be read or removed is left: another process may be removing it too.
fn remove_stale(cache: &Path) {
    let Ok(entries) = fs::read_dir(cache) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if let Some((pid, _)) = name.strip_prefix(".tmp-").and_then(|rest| rest.split_once('-')) {
            if pid.parse::<u32>().is_ok() && !Path::new("/proc").join(pid).exists() {
                let _ = fs::remove_dir_all(&path);
            }
        } else if view_name(&name)
            && fs::symlink_metadata(&path)
                .and_then(|m| m.modified())
                .is_ok_and(|used| used.elapsed().is_ok_and(|age| age > UNUSED))
            && dead(&path)
        {
            let _ = fs::remove_dir_all(&path);
        }
    }
}

/// Whether the view `dir` cannot work: one of its links leads nowhere. (One that cannot be read
/// is taken to work.)
fn dead(dir: &Path) -> bool {
    let mut links = BTreeMap::new();
    if read_links(dir, Path::new(""), &mut links).is_err() {
        return false;
    }
    links
        .keys()
        .any(|rel| fs::metadata(dir.join(rel)).is_err_and(|e| e.kind() == io::ErrorKind::NotFound))
}

/// The links under `dir`, by their path from the view (`rel`).
fn read_links(dir: &Path, rel: &Path, links: &mut BTreeMap<PathBuf, PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir.join(rel))? {
        let entry = entry?;
        let path = rel.join(entry.file_name());
        if entry.file_type()?.is_symlink() {
            links.insert(path.clone(), fs::read_link(dir.join(&path))?);
        } else {
            read_links(dir, &path, links)?;
        }
    }
    Ok(())
}

fn effective_uid() -> io::Result<u32> {
    use std::os::unix::fs::MetadataExt;
    Ok(fs::metadata("/proc/self")?.uid())
}

fn private_cache(cache: &Path) -> io::Result<()> {
    use std::os::unix::fs::MetadataExt;
    fs::DirBuilder::new().recursive(true).mode(0o700).create(cache)?;
    let m = fs::symlink_metadata(cache)?;
    let owner = effective_uid()?;
    if m.is_dir() && m.uid() == owner && m.mode() & 0o077 == 0 {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "the VF runfiles cache must be a private directory owned by the current user",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_view_is_named_by_the_fnv_1a_hash_of_its_links() {
        // (The values: FNV-1a of each path and each target, each followed by a zero byte,
        // computed apart from this code.)
        let links = BTreeMap::from([
            (PathBuf::from("cuda/bin/ptxas"), PathBuf::from("/opt/vf/bin/vf-ptxas")),
            (
                PathBuf::from("cuda/bin/nvlink"),
                PathBuf::from("/app.runfiles/cuda/bin/nvlink"),
            ),
        ]);
        assert_eq!(name(&links), "666a022d32197c05");
        assert_eq!(name(&BTreeMap::new()), "cbf29ce484222325");
    }
}
