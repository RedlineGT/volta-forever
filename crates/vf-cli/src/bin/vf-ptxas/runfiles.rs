//! A Bazel application can resolve the whole CUDA toolkit by directory and then append
//! "bin/ptxas". Preserve that lookup and the original resources using directory symlinks.
//!
//! What is in use is known by a lock: each view `<cache>/<name>` has a lock file
//! `<cache>/<name>.lock`, and the process that makes the view (in a staging directory
//! `.tmp-<name>-<pid>-<n>`, then renamed) and the program that runs with it hold a shared lock
//! on it. It is an open file description lock: the program and every process it starts inherit
//! it, it is let go when the last of them ends, and it is seen by every machine that shares the
//! cache over NFS (where flock is not: the server's own flocks and its clients' do not meet).
//! A view and its staging directories are removed only by a process that takes the exclusive
//! lock, which none of them can while something holds it. Where such locks are not seen by all
//! who may share the cache (NFS mounted with local locks, FUSE), or cannot be taken, nothing is
//! removed.
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_int;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, symlink};
use std::path::{Path, PathBuf};

/// What a view rebuilds of the runfiles: its directories, and its links (path in the view, what
/// it points to).
type Layout = (BTreeSet<PathBuf>, BTreeMap<PathBuf, PathBuf>);

/// The view of the runfiles `source` (or `source` itself, when it has no assembler but
/// `compiler`), and the shared lock on it: the view is not removed for as long as the file is
/// open, in this process or in one that inherits it (`inherited`). None for the file when the
/// file system of the cache takes no lock.
pub fn view(source: &Path, compiler: &Path, cache: &Path) -> io::Result<(PathBuf, Option<File>)> {
    let source = source.canonicalize()?;
    let Some((directories, links)) = layout(&source, compiler)? else {
        return Ok((source, None));
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
    let name = name(&links);
    let held = hold(&cache, &name)?;
    let final_dir = cache.join(&name);
    if final_dir.is_dir() {
        return Ok((final_dir, held));
    }
    remove_stale(&cache, &name);
    let staging = (0..)
        .find_map(|i: u32| {
            let p = cache.join(format!(".tmp-{name}-{}-{i}", std::process::id()));
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
    Ok((final_dir, held))
}

/// Keep `file` open in the programs this process becomes or starts (its descriptor is not
/// closed by `exec`), so that they hold its lock.
pub fn inherited(file: &File) -> io::Result<()> {
    // SAFETY: a descriptor of this process; F_SETFD takes an int.
    if unsafe { fcntl(file.as_raw_fd(), F_SETFD, 0 as c_int) } == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

unsafe extern "C" {
    fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
}

const F_SETFD: c_int = 2;
const F_OFD_SETLK: c_int = 37;
const F_OFD_SETLKW: c_int = 38;
const F_RDLCK: i16 = 0;
const F_WRLCK: i16 = 1;
const EINTR: i32 = 4;
const EAGAIN: i32 = 11;
const EACCES: i32 = 13;

/// `struct flock` of Linux on 64-bit systems.
#[repr(C)]
struct Flock {
    l_type: i16,
    l_whence: i16,
    l_start: i64,
    l_len: i64,
    l_pid: i32,
}

/// Take an open file description lock on the whole of `file`, shared or exclusive, waiting for
/// it or not: Ok(false) when another holds it (and this does not wait).
fn lock(file: &File, exclusive: bool, wait: bool) -> io::Result<bool> {
    let mut flock = Flock {
        l_type: if exclusive { F_WRLCK } else { F_RDLCK },
        l_whence: 0,
        l_start: 0,
        l_len: 0,
        l_pid: 0,
    };
    let command = if wait { F_OFD_SETLKW } else { F_OFD_SETLK };
    loop {
        // SAFETY: a descriptor of this process, and a `struct flock` that outlives the call.
        if unsafe { fcntl(file.as_raw_fd(), command, &mut flock as *mut Flock) } == 0 {
            return Ok(true);
        }
        let error = io::Error::last_os_error();
        match error.raw_os_error() {
            Some(EINTR) => continue,
            Some(EAGAIN | EACCES) if !wait => return Ok(false),
            _ => return Err(error),
        }
    }
}

/// The lock file of the view `name`, opened (made when it is not there).
fn lock_file(cache: &Path, name: &str) -> io::Result<(PathBuf, File)> {
    let path = cache.join(format!("{name}.lock"));
    // (Read and write: on NFS, a lock is taken on the server, which wants both.)
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(&path)?;
    Ok((path, file))
}

/// Whether `path` names `file` now: it was not removed, or made again, meanwhile. The name is
/// opened again, not looked up in what the system remembers of it, which on NFS may be what it
/// named a minute ago. (Closing that second description lets go of no lock of the first.)
fn names(path: &Path, file: &File) -> bool {
    match (File::open(path).and_then(|f| f.metadata()), file.metadata()) {
        (Ok(now), Ok(held)) => (now.dev(), now.ino()) == (held.dev(), held.ino()),
        _ => false,
    }
}

/// The shared lock on the view `name`, waited for (a process that removes the view holds the
/// exclusive one for as long as that takes). None when the file system takes no lock: the
/// view is then used without one, and `remove_stale` removes nothing.
fn hold(cache: &Path, name: &str) -> io::Result<Option<File>> {
    for _ in 0..100 {
        let (path, file) = lock_file(cache, name)?;
        let locked = lock(&file, false, true);
        if names(&path, &file) {
            return Ok(locked.ok().map(|_| file));
        }
        // Removed with its view while this waited for it (NFS may say so by an error): again,
        // with the lock file made next.
    }
    Err(io::Error::other(format!(
        "the lock file of the runfiles view {name} is removed whenever it is taken"
    )))
}

/// The exclusive lock on the view `name`, or None: something holds it (a program that runs with
/// the view, a process that makes it), the file system takes no lock, or the lock file was
/// removed meanwhile.
fn exclusive(cache: &Path, name: &str) -> Option<File> {
    let (path, file) = lock_file(cache, name).ok()?;
    (lock(&file, true, false).ok()? && names(&path, &file)).then_some(file)
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

/// A name that a view can have.
fn view_name(name: &str) -> bool {
    name.len() == 16 && name.bytes().all(|b| b.is_ascii_hexdigit())
}

/// The view that an entry of the cache is of: `<name>` (the view), `<name>.lock` (its lock
/// file), `.tmp-<name>-<pid>-<n>` (a staging directory of it, or a view being removed).
fn of_view(entry: &str) -> Option<&str> {
    let name = match entry.strip_prefix(".tmp-") {
        Some(rest) => rest.split_once('-')?.0,
        None => entry.strip_suffix(".lock").unwrap_or(entry),
    };
    view_name(name).then_some(name)
}

/// Remove from `cache` the views that nothing holds, but `own`, with their staging directories
/// and their lock files: no program runs with them, no process makes them (`view` then makes
/// one again). A view is first renamed as a staging directory, so that one half removed is not
/// taken for a view. What cannot be removed is left, with its lock file.
fn remove_stale(cache: &Path, own: &str) {
    if !locks_shared(cache) {
        return;
    }
    let Ok(entries) = fs::read_dir(cache) else {
        return;
    };
    let mut views: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for entry in entries.flatten() {
        let file = entry.file_name();
        let Some(name) = file.to_str().and_then(of_view) else {
            continue;
        };
        if name != own {
            let list = views.entry(name.to_string()).or_default();
            if entry.file_type().is_ok_and(|t| t.is_dir()) {
                list.push(entry.path());
            }
        }
    }
    for (name, directories) in views {
        let Some(held) = exclusive(cache, &name) else {
            continue;
        };
        let mut left = false;
        for directory in directories {
            let doomed = if directory.file_name().and_then(|n| n.to_str()) == Some(name.as_str()) {
                let doomed = cache.join(format!(".tmp-{name}-{}-gone", std::process::id()));
                if fs::rename(&directory, &doomed).is_err() {
                    left = true;
                    continue;
                }
                doomed
            } else {
                directory
            };
            left |= fs::remove_dir_all(&doomed).is_err();
        }
        if !left {
            let _ = fs::remove_file(cache.join(format!("{name}.lock")));
        }
        drop(held);
    }
}

/// Whether the locks taken on files of `cache` are seen by every process that may share it:
/// not on NFS mounted with local locks (`local_lock=all` or `posix`, `nolock`), nor on FUSE,
/// whose locks are the local system's unless the file system says otherwise; and not when
/// where the cache is mounted cannot be read.
fn locks_shared(cache: &Path) -> bool {
    let (Ok(cache), Ok(table)) = (cache.canonicalize(), fs::read("/proc/self/mountinfo")) else {
        return false;
    };
    // The mount whose point is the longest prefix of the cache's path; of two at one point, the
    // later (mounted over the other: autofs, then NFS).
    let mut found: Option<(usize, String, String)> = None;
    for line in table.split(|&b| b == b'\n') {
        let line = String::from_utf8_lossy(line);
        let fields: Vec<&str> = line.split(' ').collect();
        let Some(dash) = fields.iter().position(|f| *f == "-") else {
            continue;
        };
        let (Some(point), Some(options), Some(kind), Some(more)) =
            (fields.get(4), fields.get(5), fields.get(dash + 1), fields.get(dash + 3))
        else {
            continue;
        };
        let point = PathBuf::from(unescape(point));
        if cache.starts_with(&point) {
            let depth = point.components().count();
            if found.as_ref().is_none_or(|(d, _, _)| depth >= *d) {
                found = Some((depth, kind.to_string(), format!("{options},{more}")));
            }
        }
    }
    let Some((_, kind, options)) = found else {
        return false;
    };
    let local = options
        .split(',')
        .any(|o| matches!(o, "local_lock=all" | "local_lock=posix" | "nolock"));
    !(local || kind == "fuse" || kind.starts_with("fuse."))
}

/// A path of /proc/self/mountinfo, whose space, tab, end of line and backslash are written in
/// octal (`\040`).
fn unescape(field: &str) -> std::ffi::OsString {
    use std::os::unix::ffi::OsStringExt;
    let bytes = field.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let octal = bytes
            .get(i + 1..i + 4)
            .filter(|d| bytes[i] == b'\\' && d.iter().all(|b| (b'0'..=b'7').contains(b)));
        match octal {
            Some(d) => {
                out.push(d.iter().fold(0u8, |v, b| v.wrapping_mul(8).wrapping_add(b - b'0')));
                i += 4;
            }
            None => {
                out.push(bytes[i]);
                i += 1;
            }
        }
    }
    std::ffi::OsString::from_vec(out)
}

fn effective_uid() -> io::Result<u32> {
    Ok(fs::metadata("/proc/self")?.uid())
}

fn private_cache(cache: &Path) -> io::Result<()> {
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

    fn directory() -> PathBuf {
        let tmp = std::process::Command::new("mktemp").arg("-d").output().unwrap();
        assert!(tmp.status.success());
        PathBuf::from(std::str::from_utf8(&tmp.stdout).unwrap().trim())
            .canonicalize()
            .unwrap()
    }

    #[test]
    fn runfiles_views_are_removed_once_nothing_holds_them() {
        // (Until 2026-11-23 a view was removed when it had not been used for a day and one of its
        // links led nowhere, whatever ran with it: a view whose assembler worked, of runfiles
        // that had lost an optional tool, was removed under the engine that ran with it. A
        // staging directory was removed when its process was not in /proc, which does not show
        // the processes of another PID namespace that shares the cache.)
        let dir = directory();
        let (cache, compiler) = (dir.join("cache"), dir.join("vf-ptxas"));
        fs::write(&compiler, "VF").unwrap();
        let app = |name: &str| {
            let src = dir.join(name);
            fs::create_dir_all(src.join("cuda/bin")).unwrap();
            fs::write(src.join("cuda/bin/ptxas"), "NVIDIA").unwrap();
            fs::write(src.join("cuda/bin/nvlink"), "linker").unwrap();
            fs::write(src.join("cuda/bin/optional"), "tool").unwrap();
            src
        };
        let two_days_ago = |path: &Path| {
            let when = std::time::SystemTime::now() - std::time::Duration::from_secs(2 * 24 * 60 * 60);
            File::open(path).unwrap().set_modified(when).unwrap();
        };
        // Programs run with two views; one of them has lost an optional tool (its link leads
        // nowhere) and is two days old. A third view is no longer held.
        let (held, dangling, free) = (app("held.runfiles"), app("dangling.runfiles"), app("free.runfiles"));
        let (held_view, held_lock) = view(&held, &compiler, &cache).unwrap();
        let (dangling_view, dangling_lock) = view(&dangling, &compiler, &cache).unwrap();
        let (free_view, free_lock) = view(&free, &compiler, &cache).unwrap();
        assert!(held_lock.is_some() && dangling_lock.is_some() && free_lock.is_some());
        drop(free_lock);
        fs::remove_file(dangling.join("cuda/bin/optional")).unwrap();
        two_days_ago(&dangling_view);
        assert!(fs::metadata(dangling_view.join("cuda/bin/optional")).is_err());
        assert_eq!(fs::read_to_string(dangling_view.join("cuda/bin/ptxas")).unwrap(), "VF");
        // A view without a lock file, which nothing holds; a lock file without its view.
        let bare = cache.join("0123456789abcdef");
        fs::create_dir(&bare).unwrap();
        symlink(&compiler, bare.join("ptxas")).unwrap();
        let orphan = cache.join("aabbccddeeff0011.lock");
        fs::write(&orphan, "").unwrap();
        // A staging directory of a process that runs, in another PID namespace (a number that no
        // process has here), and holds the lock of its view; one of a process that ended; a file
        // of someone else.
        let making = cache.join(".tmp-00112233445566ff-4294967295-0");
        let ended = cache.join(".tmp-0011223344556677-4294967295-0");
        fs::create_dir(&making).unwrap();
        fs::create_dir(&ended).unwrap();
        let maker = hold(&cache, "00112233445566ff").unwrap();
        assert!(maker.is_some());
        let other = cache.join("notes.txt");
        fs::write(&other, "notes").unwrap();
        // A new view: what nothing holds goes, with its lock file; nothing it links goes.
        let (new_view, _new_lock) = view(&app("new.runfiles"), &compiler, &cache).unwrap();
        let lock_of = |view: &Path| view.with_extension("lock");
        for (path, there) in [
            (&held_view, true),
            (&lock_of(&held_view), true),
            (&dangling_view, true),
            (&lock_of(&dangling_view), true),
            (&making, true),
            (&other, true),
            (&new_view, true),
            (&free_view, false),
            (&lock_of(&free_view), false),
            (&bare, false),
            (&lock_of(&bare), false),
            (&orphan, false),
            (&ended, false),
        ] {
            assert_eq!(fs::symlink_metadata(path).is_ok(), there, "{}", path.display());
        }
        assert_eq!(fs::read_to_string(free.join("cuda/bin/nvlink")).unwrap(), "linker");
        assert_eq!(fs::read_to_string(&compiler).unwrap(), "VF");
        // Once they end, the next view made removes theirs.
        drop((held_lock, dangling_lock, maker));
        let (last_view, _last_lock) = view(&app("last.runfiles"), &compiler, &cache).unwrap();
        let mut left: Vec<_> = fs::read_dir(&cache)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        left.sort();
        let new = new_view.file_name().unwrap().to_str().unwrap();
        let last = last_view.file_name().unwrap().to_str().unwrap();
        let mut want = vec![
            new.to_string(),
            format!("{new}.lock"),
            last.to_string(),
            format!("{last}.lock"),
            "notes.txt".to_string(),
        ];
        want.sort();
        // (The view made before is not held either: `_new_lock` is, by this test.)
        assert_eq!(left, want);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_lock_waits_for_the_removal_and_is_taken_again() {
        // A process that removes a view holds its exclusive lock: one that asks for the view
        // meanwhile waits, then takes the lock file made after (the one it waited on is gone).
        let dir = directory();
        let cache = dir.join("cache");
        fs::DirBuilder::new().mode(0o700).create(&cache).unwrap();
        let name = "fedcba9876543210";
        let remover = exclusive(&cache, name).unwrap();
        assert!(exclusive(&cache, name).is_none());
        let waiter = std::thread::spawn({
            let cache = cache.clone();
            move || hold(&cache, name).unwrap().unwrap()
        });
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert!(!waiter.is_finished());
        fs::remove_file(cache.join(format!("{name}.lock"))).unwrap();
        drop(remover);
        let held = waiter.join().unwrap();
        assert!(names(&cache.join(format!("{name}.lock")), &held));
        assert!(exclusive(&cache, name).is_none());
        drop(held);
        assert!(exclusive(&cache, name).is_some());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn mount_points_are_read_with_their_escapes() {
        assert_eq!(unescape(r"/mnt/a\040b\134c"), std::ffi::OsString::from("/mnt/a b\\c"));
        assert_eq!(unescape(r"/plain\08"), std::ffi::OsString::from(r"/plain\08"));
    }
}
