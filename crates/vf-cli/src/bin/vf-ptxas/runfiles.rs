//! A Bazel application can resolve the whole CUDA toolkit by directory and then append
//! "bin/ptxas". Preserve that lookup and the original resources using directory symlinks.
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{DirBuilderExt, symlink};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// What a view rebuilds of the runfiles: its directories, and its links (path in the view, what
/// it points to).
type Layout = (BTreeSet<PathBuf>, BTreeMap<PathBuf, PathBuf>);

/// A view not used for this long, that its runfiles would not give now, is removed.
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
        // (Used: see `remove_stale`.)
        let _ = fs::File::open(&final_dir).and_then(|dir| dir.set_modified(SystemTime::now()));
        return Ok(final_dir);
    }
    remove_stale(&cache, compiler);
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

/// Remove from `cache` what nothing will use again: the staging directories of processes that
/// ended before they renamed them, and the views not used for a day (`UNUSED`) that their
/// runfiles would not give now: these are gone, or their layout changed, or the view links
/// another assembler, or it was named otherwise (by an earlier version). What cannot be read or
/// removed is left: another process may be removing it too.
fn remove_stale(cache: &Path, compiler: &Path) {
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
        } else if name.len() == 16
            && name.bytes().all(|b| b.is_ascii_hexdigit())
            && fs::symlink_metadata(&path)
                .and_then(|m| m.modified())
                .is_ok_and(|used| used.elapsed().is_ok_and(|age| age > UNUSED))
            && !current(&path, &name, compiler)
        {
            let _ = fs::remove_dir_all(&path);
        }
    }
}

/// Whether the view `dir`, of that name, is the one its runfiles give now. (When it says nothing
/// of where they are, all its links being to the assembler, it is taken to be.)
fn current(dir: &Path, name: &str, compiler: &Path) -> bool {
    let mut links = BTreeMap::new();
    if read_links(dir, Path::new(""), &mut links).is_err() {
        return false;
    }
    // The runfiles: what a link that is not to the assembler points to, less the link's own path.
    let source = links.iter().find_map(|(rel, target)| {
        let source = rel.components().try_fold(target.as_path(), |path, _| path.parent())?;
        (target != compiler && source.join(rel) == *target).then(|| source.to_path_buf())
    });
    let Some(source) = source else {
        return true;
    };
    match layout(&source, compiler) {
        Ok(Some((_, now))) => now == links && self::name(&now) == name,
        _ => false,
    }
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
