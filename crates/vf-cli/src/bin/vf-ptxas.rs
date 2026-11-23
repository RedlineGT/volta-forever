//! CUDA 12.9 ptxas front end for compilers that hand PTX to a subprocess.
//! Keep all backend tuning options. Adapt file inputs to the Volta offline ISA.
#[path = "vf-ptxas/runfiles.rs"]
mod runfiles;

use std::ffi::{OsStr, OsString, c_char, c_int, c_uint, c_ulong};
use std::fs::{self, File};
use std::io::{self, BufRead, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::{
    ffi::OsStrExt,
    fs::MetadataExt,
    process::{CommandExt, ExitStatusExt},
};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use vf_adapt::{Outcome, Policy};

unsafe extern "C" {
    fn memfd_create(name: *const c_char, flags: c_uint) -> c_int;
    fn prctl(option: c_int, ...) -> c_int;
    fn getpid() -> c_int;
    fn getppid() -> c_int;
    fn signal(signum: c_int, handler: usize) -> usize;
    fn raise(signum: c_int) -> c_int;
}

const PR_SET_PDEATHSIG: c_int = 1;
const SIGTERM: c_int = 15;
const SIG_DFL: usize = 0;
/// What ptxas ends with on every error it finds, and so this program on those it finds itself.
const FATAL: u8 = 255;

fn backend() -> Result<PathBuf, String> {
    let path = match std::env::var_os("VF_PTXAS_REAL") {
        Some(path) => PathBuf::from(path),
        None => {
            let lib = std::env::var_os("VF_CUDA12").map(PathBuf::from);
            lib.as_deref()
                .and_then(Path::parent)
                .map(|dir| dir.join("bin/ptxas"))
                .filter(|p| p.is_file())
                .unwrap_or_else(|| PathBuf::from("/usr/local/cuda-12.9/bin/ptxas"))
        }
    };
    let stat = path.metadata().map_err(|e| {
        format!(
            "{}: {e} (CUDA 12.9 ptxas; install cuda-nvcc-12-9 or set VF_PTXAS_REAL)",
            path.display()
        )
    })?;
    let own = fs::metadata("/proc/self/exe").map_err(|e| e.to_string())?;
    if (own.dev(), own.ino()) == (stat.dev(), stat.ino()) {
        return Err("the configured assembler resolves to the VF front end".into());
    }
    Ok(path)
}

fn target(name: &OsStr) -> Option<&'static str> {
    let name = name.to_str()?;
    let (prefix, suffix) = name.split_once('_')?;
    let digits = suffix.strip_suffix(['a', 'f']).unwrap_or(suffix);
    let sm: u32 = digits.parse().ok()?;
    if sm < 70 {
        return None;
    }
    match prefix {
        "sm" => Some("sm_70"),
        "compute" => Some("compute_70"),
        _ => None,
    }
}

// These options consume the following argument when their value is not joined with '=': those of
// ptxas 13.5 (each given alone, it says "argument expected"), but the ones handled in `compile`.
// In particular, the output file must never be opened as a PTX input. (`-O` takes its level joined,
// `-O3`: alone, it is an option of its own, and what follows is an input.)
fn consumes_value(option: &[u8]) -> bool {
    matches!(
        option,
        b"-o"
            | b"--output-file"
            | b"-e"
            | b"--entry"
            | b"--opt-level"
            | b"-m"
            | b"--machine"
            | b"-Ofc"
            | b"--Ofast-compile"
            | b"--allow-expensive-optimizations"
            | b"-allow-expensive-optimizations"
            | b"-func-maxrregcount"
            | b"--device-function-maxrregcount"
            | b"--device-stack-protector"
            | b"-device-stack-protector"
            | b"-fmad"
            | b"--fmad"
            | b"-dlcm"
            | b"--def-load-cache"
            | b"-dscm"
            | b"--def-store-cache"
            | b"-flcm"
            | b"--force-load-cache"
            | b"-fscm"
            | b"--force-store-cache"
            | b"--maxntid"
            | b"-maxntid"
            | b"--maxrregcount"
            | b"-maxrregcount"
            | b"--minnctapersm"
            | b"-minnctapersm"
            | b"--position-independent-code"
            | b"-pic"
            | b"-regUsageLevel"
            | b"--register-usage-level"
            | b"--split-compile"
            | b"-split-compile"
            | b"--apply-controls"
            | b"-apply-controls"
            | b"--enable-func-clone-sc"
            | b"-enable-func-clone-sc"
            | b"--sanitize"
            | b"-sanitize"
    )
}

fn adapted_input(data: &[u8], policy: &Policy) -> Result<Option<File>, String> {
    let adapted = policy.adapt_ptx(data);
    match adapted.outcome {
        Outcome::Native => return Ok(None),
        Outcome::Retargeted { trapped, .. } if !trapped.is_empty() => {
            return Err(format!("{}: {}", trapped[0].function, trapped[0].reason));
        }
        Outcome::Retargeted { .. } => {}
        _ => return Err("expected a PTX module that VF can compile for sm_70".into()),
    }
    let bytes = adapted.image.ok_or("missing retargeted PTX")?;
    // Keep the fd open across exec. The assembler reopens /proc/self/fd/N at offset zero.
    let fd = unsafe { memfd_create(c"vf-ptx".as_ptr(), 0) };
    if fd == -1 {
        return Err(io::Error::last_os_error().to_string());
    }
    let mut file = unsafe { File::from_raw_fd(fd) };
    file.write_all(&bytes).map_err(|e| e.to_string())?;
    Ok(Some(file))
}

fn compile(args: Vec<OsString>) -> Result<ExitCode, String> {
    let backend = backend()?;
    let mut process = Command::new(&backend);
    if args
        .iter()
        .any(|s| matches!(s.as_bytes(), b"-V" | b"--version" | b"-h" | b"--help"))
    {
        return Err(process.args(args).exec().to_string());
    }
    let policy = Policy::cuda_12_9().with_environment();
    // The adapted inputs, and what the user called them.
    let mut retained = Vec::new();
    let mut names = Vec::new();
    let mut explicit_arch = false;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let bytes = arg.as_bytes();
        let mut pieces = bytes.splitn(2, |b| *b == b'=');
        let name = pieces.next().unwrap();
        let value = pieces.next().map(OsStr::from_bytes);
        match name {
            b"-arch" | b"--gpu-name" => {
                let value = value
                    .map(OsStr::to_owned)
                    .or_else(|| args.next())
                    .ok_or("missing --gpu-name value")?;
                let value = target(&value).map(OsStr::new).unwrap_or(&value);
                process.arg(OsStr::from_bytes(name)).arg(value);
                explicit_arch = true;
            }
            b"-optf" | b"--options-file" | b"-ias" | b"--input-as-string" => {
                return Err(format!(
                    "{} requires file-backed PTX arguments to VF",
                    OsStr::from_bytes(name).to_string_lossy()
                ));
            }
            _ if bytes.starts_with(b"-") => {
                let next = value.is_none() && consumes_value(name);
                process.arg(arg);
                if next {
                    process.arg(args.next().ok_or("missing assembler option value")?);
                }
            }
            _ => {
                let data = fs::read(&arg).map_err(|e| format!("{}: {e}", Path::new(&arg).display()))?;
                if let Some(file) =
                    adapted_input(&data, &policy).map_err(|e| format!("{}: {e}", Path::new(&arg).display()))?
                {
                    let given = format!("/proc/self/fd/{}", file.as_raw_fd());
                    process.arg(&given);
                    names.push((given.into_bytes(), arg.as_bytes().to_vec()));
                    retained.push(file);
                } else {
                    process.arg(arg);
                }
            }
        }
    }
    if !explicit_arch {
        process.arg("-arch=sm_70");
    }
    if names.is_empty() {
        return Err(process.exec().to_string());
    }
    assemble(process, &names).map_err(|e| format!("{}: {e}", backend.display()))
}

/// Run the assembler on adapted inputs, which it is given as `/proc/self/fd/N`: its messages
/// (all on its standard error) name the files the user gave instead (`names`: what it was given,
/// what the user gave). This ends with the status the assembler ends with, or by the signal that
/// ended it; the assembler is sent SIGTERM if this ends first.
fn assemble(mut process: Command, names: &[(Vec<u8>, Vec<u8>)]) -> io::Result<ExitCode> {
    let parent = unsafe { getpid() };
    unsafe {
        process.pre_exec(move || {
            prctl(PR_SET_PDEATHSIG, SIGTERM as c_ulong);
            if getppid() != parent {
                // (This front end has ended already: ESRCH, and nothing that allocates.)
                return Err(io::Error::from_raw_os_error(3));
            }
            Ok(())
        });
    }
    let mut child = process.stderr(Stdio::piped()).spawn()?;
    let mut messages = io::BufReader::new(child.stderr.take().expect("piped"));
    let mut stderr = io::stderr().lock();
    let mut line = Vec::new();
    loop {
        line.clear();
        match messages.read_until(b'\n', &mut line) {
            Ok(0) => break,
            Ok(_) => {
                let _ = stderr.write_all(&renamed(&line, names));
                let _ = stderr.flush();
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
    drop(messages);
    let status = child.wait()?;
    if let Some(number) = status.signal() {
        unsafe {
            signal(number, SIG_DFL);
            raise(number);
        }
    }
    Ok(ExitCode::from(status.code().map_or(FATAL, |code| code as u8)))
}

/// `line` with each name the assembler was given (`/proc/self/fd/3`, not followed by a digit)
/// replaced by the one the user gave.
fn renamed(line: &[u8], names: &[(Vec<u8>, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(line.len());
    let mut at = 0;
    'scan: while at < line.len() {
        for (given, user) in names {
            if line[at..].starts_with(given) && !line.get(at + given.len()).is_some_and(u8::is_ascii_digit) {
                out.extend_from_slice(user);
                at += given.len();
                continue 'scan;
            }
        }
        out.push(line[at]);
        at += 1;
    }
    out
}

fn run(args: Vec<OsString>) -> Result<ExitCode, String> {
    if args.first().is_none_or(|a| a != "--vf-runfiles") {
        return compile(args);
    }
    let [_, source, cache] = args.as_slice() else {
        return Err("--vf-runfiles requires a source and cache directory".into());
    };
    let write_view = || -> io::Result<()> {
        let path = runfiles::view(Path::new(source), &std::env::current_exe()?, Path::new(cache))?;
        let mut out = io::stdout().lock();
        out.write_all(path.as_os_str().as_bytes())?;
        out.write_all(b"\n")
    };
    write_view().map_err(|e| format!("CUDA runfiles: {e}"))?;
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    let mut args = std::env::args_os();
    if let (Some(argv0), Ok(exe)) = (args.next(), std::env::current_exe()) {
        runfiles::used(Path::new(&argv0), &exe);
    }
    match run(args.collect()) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("vf-ptxas: {error}");
            ExitCode::from(FATAL)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_name_the_inputs_as_the_user_gave_them() {
        let names = [(b"/proc/self/fd/3".to_vec(), b"kernel.ptx".to_vec())];
        assert_eq!(
            renamed(b"ptxas /proc/self/fd/3, line 2; error\n", &names),
            b"ptxas kernel.ptx, line 2; error\n"
        );
        assert_eq!(renamed(b"/proc/self/fd/30\n", &names), b"/proc/self/fd/30\n");
        assert_eq!(renamed(b"/proc/self/fd/3", &names), b"kernel.ptx");
    }

    #[test]
    fn recent_targets_retain_architecture_kind() {
        for name in ["sm_70", "sm_80", "sm_90a", "sm_120f"] {
            assert_eq!(target(OsStr::new(name)), Some("sm_70"));
        }
        assert_eq!(target(OsStr::new("compute_120")), Some("compute_70"));
        for name in ["sm_60", "bad_80", "sm_8x"] {
            assert_eq!(target(OsStr::new(name)), None);
        }
    }

    #[test]
    fn cuda_runfiles_share_resources_across_concurrent_launches() {
        let tmp = std::process::Command::new("mktemp").arg("-d").output().unwrap();
        assert!(tmp.status.success());
        let dir = Path::new(std::str::from_utf8(&tmp.stdout).unwrap().trim());
        let src = dir.join("app.runfiles");
        let cache = dir.join("cache");
        let compiler = dir.join("vf-ptxas");
        let bin = Path::new("toolkit/sandbox/bin");
        fs::create_dir_all(src.join(bin)).unwrap();
        fs::write(&compiler, "VF").unwrap();
        fs::write(src.join(bin).join("ptxas"), "NVIDIA").unwrap();
        fs::write(src.join(bin).join("nvlink"), "linker").unwrap();
        fs::write(src.join("_repo_mapping"), "repositories").unwrap();
        let view = std::thread::scope(|s| {
            let threads: Vec<_> = (0..4)
                .map(|_| s.spawn(|| runfiles::view(&src, &compiler, &cache).unwrap()))
                .collect();
            let views: Vec<_> = threads.into_iter().map(|h| h.join().unwrap()).collect();
            assert!(views.iter().all(|p| p == &views[0]));
            views[0].clone()
        });
        assert_eq!(fs::read_to_string(view.join(bin).join("ptxas")).unwrap(), "VF");
        assert_eq!(fs::read_to_string(src.join(bin).join("ptxas")).unwrap(), "NVIDIA");
        assert_eq!(fs::read_to_string(view.join("_repo_mapping")).unwrap(), "repositories");
        assert_eq!(fs::read_to_string(view.join(bin).join("nvlink")).unwrap(), "linker");
        fs::write(src.join(bin).join("new-tool"), "new resource").unwrap();
        let next = runfiles::view(&src, &compiler, &cache).unwrap();
        assert_ne!(next, view);
        assert_eq!(
            fs::read_to_string(next.join(bin).join("new-tool")).unwrap(),
            "new resource"
        );
        assert_eq!(runfiles::view(&next, &compiler, &cache).unwrap(), next);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn runfiles_views_are_named_the_same_by_every_build() {
        // The name of a view is the FNV-1a hash of its links (each path and each target with a
        // zero byte after it). (Until 2026-11-22 it was DefaultHasher's, which may change from
        // one version of Rust to the next: each upgrade left the views made before it.)
        let tmp = std::process::Command::new("mktemp").arg("-d").output().unwrap();
        assert!(tmp.status.success());
        let dir = Path::new(std::str::from_utf8(&tmp.stdout).unwrap().trim())
            .canonicalize()
            .unwrap();
        let src = dir.join("app.runfiles");
        let compiler = dir.join("vf-ptxas");
        fs::create_dir_all(src.join("cuda/bin")).unwrap();
        fs::write(&compiler, "VF").unwrap();
        fs::write(src.join("cuda/bin/ptxas"), "NVIDIA").unwrap();
        fs::write(src.join("cuda/bin/nvlink"), "linker").unwrap();
        let links = [
            ("cuda/bin/nvlink", src.join("cuda/bin/nvlink")),
            ("cuda/bin/ptxas", compiler.clone()),
        ];
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for (rel, target) in &links {
            for part in [rel.as_bytes(), target.as_os_str().as_bytes()] {
                for &byte in part.iter().chain(&[0]) {
                    hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
                }
            }
        }
        let view = runfiles::view(&src, &compiler, &dir.join("cache")).unwrap();
        assert_eq!(view, dir.join("cache").join(format!("{hash:016x}")));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn runfiles_views_that_cannot_work_are_removed_and_no_other() {
        let tmp = std::process::Command::new("mktemp").arg("-d").output().unwrap();
        assert!(tmp.status.success());
        let dir = Path::new(std::str::from_utf8(&tmp.stdout).unwrap().trim())
            .canonicalize()
            .unwrap();
        let (cache, compiler) = (dir.join("cache"), dir.join("vf-ptxas"));
        fs::write(&compiler, "VF").unwrap();
        let app = |name: &str| {
            let src = dir.join(name);
            fs::create_dir_all(src.join("cuda/bin")).unwrap();
            fs::write(src.join("cuda/bin/ptxas"), "NVIDIA").unwrap();
            fs::write(src.join("cuda/bin/nvlink"), "linker").unwrap();
            src
        };
        let days_ago = |path: &Path, days: u64| {
            let when = std::time::SystemTime::now() - std::time::Duration::from_secs(days * 24 * 60 * 60);
            File::open(path).unwrap().set_modified(when).unwrap();
        };
        let recently = |path: &Path| {
            fs::metadata(path).unwrap().modified().unwrap().elapsed().unwrap() < std::time::Duration::from_secs(3600)
        };
        // Views of three applications, made two days ago; one is asked for again today, and the
        // assembler is started through another (its engine compiles).
        let (kept, changed, recent) = (app("kept.runfiles"), app("changed.runfiles"), app("recent.runfiles"));
        let kept_view = runfiles::view(&kept, &compiler, &cache).unwrap();
        let changed_view = runfiles::view(&changed, &compiler, &cache).unwrap();
        let recent_view = runfiles::view(&recent, &compiler, &cache).unwrap();
        for view in [&kept_view, &changed_view, &recent_view] {
            days_ago(view, 2);
        }
        assert_eq!(runfiles::view(&recent, &compiler, &cache).unwrap(), recent_view);
        assert!(recently(&recent_view));
        runfiles::used(&changed_view.join("cuda/bin/ptxas"), &compiler);
        assert!(recently(&changed_view));
        // Not through a view, or not to this assembler: nothing is used.
        runfiles::used(&changed.join("cuda/bin/ptxas"), &compiler);
        runfiles::used(&kept_view.join("cuda/bin/nvlink"), &compiler);
        runfiles::used(&kept_view.join("cuda/bin/ptxas"), &dir.join("another"));
        assert!(!recently(&kept_view));
        days_ago(&changed_view, 2);
        // Two of them change; a view of runfiles that are gone; one of an assembler that is gone
        // (a release removed); what a process that ended left, and what one that runs (this one)
        // is making.
        fs::write(changed.join("cuda/bin/nvcc"), "compiler driver").unwrap();
        fs::write(recent.join("cuda/bin/nvcc"), "compiler driver").unwrap();
        let gone = cache.join("0123456789abcdef");
        fs::create_dir(&gone).unwrap();
        std::os::unix::fs::symlink(dir.join("gone.runfiles/cuda/bin/nvcc"), gone.join("nvcc")).unwrap();
        days_ago(&gone, 2);
        let removed = cache.join("00112233445566ff");
        fs::create_dir_all(removed.join("cuda/bin")).unwrap();
        std::os::unix::fs::symlink(dir.join("removed/vf-ptxas"), removed.join("cuda/bin/ptxas")).unwrap();
        std::os::unix::fs::symlink(kept.join("cuda/bin/nvlink"), removed.join("cuda/bin/nvlink")).unwrap();
        days_ago(&removed, 2);
        // The view of the first, as a version that named views otherwise made it, and which an
        // engine started by that version may hold: it works, and is left.
        let renamed = cache.join("fedcba9876543210");
        fs::create_dir_all(renamed.join("cuda/bin")).unwrap();
        std::os::unix::fs::symlink(&compiler, renamed.join("cuda/bin/ptxas")).unwrap();
        std::os::unix::fs::symlink(kept.join("cuda/bin/nvlink"), renamed.join("cuda/bin/nvlink")).unwrap();
        days_ago(&renamed, 2);
        // What this system's processes are making: one that ended, one that runs (this one); what
        // another system that shares the cache is making, whose processes are not seen from here,
        // and what one made a day ago; what an earlier version (without the system's identity)
        // left a day ago, and what it is making.
        let boot = runfiles::boot();
        let (ended, running) = (
            cache.join(format!(".tmp-{boot}-4294967295-0")),
            cache.join(format!(".tmp-{boot}-{}-7", std::process::id())),
        );
        let (elsewhere, elsewhere_old) = (
            cache.join(".tmp-00000000-0000-0000-0000-000000000000-4294967295-0"),
            cache.join(".tmp-00000000-0000-0000-0000-000000000000-4294967295-1"),
        );
        let (earlier, earlier_old) = (cache.join(".tmp-4294967295-0"), cache.join(".tmp-4294967295-1"));
        for staging in [&ended, &running, &elsewhere, &elsewhere_old, &earlier, &earlier_old] {
            fs::create_dir(staging).unwrap();
        }
        days_ago(&elsewhere_old, 2);
        days_ago(&earlier_old, 2);
        // A new view: what cannot work goes, with what it holds and nothing it points to. A view
        // that works stays, also when its runfiles changed since (an engine started before holds
        // it).
        let changed_now = runfiles::view(&changed, &compiler, &cache).unwrap();
        assert_ne!(changed_now, changed_view);
        for (path, there) in [
            (&kept_view, true),
            (&recent_view, true),
            (&running, true),
            (&elsewhere, true),
            (&earlier, true),
            (&elsewhere_old, false),
            (&earlier_old, false),
            (&changed_now, true),
            (&changed_view, true),
            (&renamed, true),
            (&gone, false),
            (&removed, false),
            (&ended, false),
        ] {
            assert_eq!(path.exists(), there, "{}", path.display());
        }
        assert_eq!(fs::read_to_string(changed.join("cuda/bin/ptxas")).unwrap(), "NVIDIA");
        assert_eq!(fs::read_to_string(kept.join("cuda/bin/nvlink")).unwrap(), "linker");
        assert_eq!(fs::read_to_string(&compiler).unwrap(), "VF");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn concurrent_runfiles_for_different_apps_preserve_each_layout() {
        const THREADS: usize = 12;
        let tmp = std::process::Command::new("mktemp").arg("-d").output().unwrap();
        assert!(tmp.status.success());
        let root = PathBuf::from(std::str::from_utf8(&tmp.stdout).unwrap().trim());
        let compiler = root.join("vf-ptxas");
        fs::write(&compiler, "VF").unwrap();
        let start = std::sync::Barrier::new(THREADS);
        std::thread::scope(|scope| {
            for i in 0..THREADS {
                let src = root.join(format!("app-{i}.runfiles"));
                let bin = src.join("cuda/bin");
                fs::create_dir_all(&bin).unwrap();
                fs::write(bin.join("ptxas"), "NVIDIA").unwrap();
                fs::write(bin.join("nvlink"), i.to_string()).unwrap();
                let (start, root, compiler) = (&start, &root, &compiler);
                scope.spawn(move || {
                    start.wait();
                    for round in 0..24 {
                        let view = runfiles::view(&src, compiler, &root.join(round.to_string())).unwrap();
                        assert_eq!(fs::read_to_string(view.join("cuda/bin/nvlink")).unwrap(), i.to_string());
                        assert_eq!(fs::read_to_string(view.join("cuda/bin/ptxas")).unwrap(), "VF");
                    }
                });
            }
        });
        fs::remove_dir_all(root).unwrap();
    }
}
