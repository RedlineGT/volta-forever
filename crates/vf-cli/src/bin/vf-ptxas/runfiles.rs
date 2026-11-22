//! A Bazel application can resolve the whole CUDA toolkit by directory and then append
//! "bin/ptxas". Preserve that lookup and the original resources using directory symlinks.
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io;
use std::os::unix::fs::{DirBuilderExt, symlink};
use std::path::{Path, PathBuf};

pub fn view(source: &Path, compiler: &Path, cache: &Path) -> io::Result<PathBuf> {
    let source = source.canonicalize()?;
    let mut compilers = BTreeSet::new();
    for entry in fs::read_dir(&source)? {
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
        return Ok(source);
    }
    let directories: BTreeSet<_> = compilers.iter().flat_map(|p| p.ancestors().skip(1)).collect();
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
    // Hash the complete layout. Replacing a linked file is immediately visible. Adding
    // a sibling to one of the rebuilt directories changes the key on the next launch.
    let mut hash = DefaultHasher::new();
    links.hash(&mut hash);
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
    let final_dir = cache.join(format!("{:016x}", hash.finish()));
    if final_dir.is_dir() {
        return Ok(final_dir);
    }
    let staging = (0..)
        .find_map(|i: u32| {
            let p = cache.join(format!(".tmp-{}-{i}", std::process::id()));
            match fs::DirBuilder::new().mode(0o700).create(&p) {
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => None,
                result => Some(result.map(|()| p)),
            }
        })
        .unwrap()?;
    let publish = || -> io::Result<()> {
        for dir in directories {
            if !dir.as_os_str().is_empty() {
                fs::create_dir(staging.join(dir))?;
            }
        }
        for (rel, target) in links {
            symlink(target, staging.join(rel))?;
        }
        match fs::rename(&staging, &final_dir) {
            Err(_) if final_dir.is_dir() => Ok(()), // A concurrent launch won the rename.
            result => result,
        }
    };
    let result = publish();
    if staging.exists() {
        fs::remove_dir_all(staging)?;
    }
    result.map(|()| final_dir)
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
