//! CUDA 12.9 ptxas front end for compilers that hand PTX to a subprocess.
//! Keep all backend tuning options. Adapt file inputs to the Volta offline ISA.
#[path = "vf-ptxas/runfiles.rs"]
mod runfiles;

use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io::{self, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::{ffi::OsStrExt, fs::MetadataExt, process::CommandExt};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use vf_adapt::{Outcome, Policy};

unsafe extern "C" {
    fn memfd_create(name: *const std::ffi::c_char, flags: std::ffi::c_uint) -> std::ffi::c_int;
}

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

// These options consume the following argument when their value is not joined with '='.
// In particular, the output file must never be opened as a PTX input.
fn consumes_value(option: &[u8]) -> bool {
    matches!(
        option,
        b"-o"
            | b"--output-file"
            | b"-e"
            | b"--entry"
            | b"-O"
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

fn compile(args: Vec<OsString>) -> Result<(), String> {
    let backend = backend()?;
    let mut process = Command::new(&backend);
    if args
        .iter()
        .any(|s| matches!(s.as_bytes(), b"-V" | b"--version" | b"-h" | b"--help"))
    {
        return Err(process.args(args).exec().to_string());
    }
    let policy = Policy::cuda_12_9().with_environment();
    let mut retained = Vec::new();
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
                    process.arg(format!("/proc/self/fd/{}", file.as_raw_fd()));
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
    Err(process.exec().to_string())
}

fn run(args: Vec<OsString>) -> Result<(), String> {
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
    write_view().map_err(|e| format!("CUDA runfiles: {e}"))
}

fn main() -> ExitCode {
    match run(std::env::args_os().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("vf-ptxas: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
