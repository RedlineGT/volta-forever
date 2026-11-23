//! vf-ptxas in front of a stand-in for ptxas that says what it was given.

use std::fs;
use std::os::unix::{fs::PermissionsExt, process::ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// What it was given: options, values, and inputs with their `.target` line. Then it ends with
/// the status `VF_TEST_STATUS` says, or by the signal `VF_TEST_SIGNAL` names.
const ASSEMBLER: &str = r#"#!/bin/sh
for a in "$@"; do
    case $a in
        -*) echo "option $a" >&2 ;;
        *) if [ -f "$a" ]; then echo "input $a: $(grep -m1 '^\.target' "$a")" >&2; else echo "value $a" >&2; fi ;;
    esac
done
[ -n "$VF_TEST_SIGNAL" ] && kill -"$VF_TEST_SIGNAL" $$
exit "${VF_TEST_STATUS:-0}"
"#;

/// A directory with the stand-in (`ptxas`), PTX for sm_75 (`kernel.ptx`) and a file that is not
/// PTX (`controls.txt`).
fn directory() -> PathBuf {
    let tmp = Command::new("mktemp").arg("-d").output().unwrap();
    assert!(tmp.status.success());
    let dir = PathBuf::from(std::str::from_utf8(&tmp.stdout).unwrap().trim());
    fs::write(dir.join("ptxas"), ASSEMBLER).unwrap();
    fs::set_permissions(dir.join("ptxas"), fs::Permissions::from_mode(0o755)).unwrap();
    let ptx = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/kernels/k_basic.135.sm75.ptx");
    fs::copy(ptx, dir.join("kernel.ptx")).unwrap();
    fs::write(dir.join("controls.txt"), "controls\n").unwrap();
    dir
}

fn run(dir: &Path, args: &[&str], env: &[(&str, &str)]) -> (Output, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_vf-ptxas"))
        .current_dir(dir)
        .env("VF_PTXAS_REAL", dir.join("ptxas"))
        .envs(env.iter().copied())
        .args(args)
        .output()
        .unwrap();
    let messages = String::from_utf8_lossy(&output.stderr).into_owned();
    (output, messages)
}

#[test]
fn messages_name_the_inputs_as_given() {
    // The assembler is given the adapted PTX as /proc/self/fd/N; its messages name what the user
    // gave. (Until 2026-11-22 they named /proc/self/fd/3.)
    let dir = directory();
    let (output, messages) = run(&dir, &["-arch=sm_75", "-o", "out.cubin", "kernel.ptx"], &[]);
    assert_eq!(output.status.code(), Some(0), "{messages}");
    assert!(messages.contains("input kernel.ptx: .target sm_70\n"), "{messages}");
    assert!(!messages.contains("/proc/self/fd"), "{messages}");
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn options_of_ptxas_13_5_take_their_value() {
    // (Until 2026-11-22 controls.txt was read as PTX, and so were true and memcheck; and `-O`
    // took the input that followed it for its value.)
    let dir = directory();
    let (output, messages) = run(
        &dir,
        &[
            "-arch=sm_75",
            "--apply-controls",
            "controls.txt",
            "-enable-func-clone-sc",
            "true",
            "--sanitize",
            "memcheck",
            "-o",
            "out.cubin",
            "kernel.ptx",
        ],
        &[],
    );
    assert_eq!(output.status.code(), Some(0), "{messages}");
    for line in [
        "option -arch\nvalue sm_70\n",
        "option --apply-controls\ninput controls.txt: \n",
        "option -enable-func-clone-sc\nvalue true\n",
        "option --sanitize\nvalue memcheck\n",
        "option -o\nvalue out.cubin\n",
        "input kernel.ptx: .target sm_70\n",
    ] {
        assert!(messages.contains(line), "{line:?} not in:\n{messages}");
    }
    // `-O` takes its level joined (`-O3`); what follows it alone is an input.
    let (output, messages) = run(&dir, &["-arch=sm_75", "-O", "kernel.ptx"], &[]);
    assert_eq!(output.status.code(), Some(0), "{messages}");
    assert!(
        messages.contains("option -O\ninput kernel.ptx: .target sm_70\n"),
        "{messages}"
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn the_status_is_the_assemblers_or_255_as_ptxas_has_it() {
    let dir = directory();
    let (output, _) = run(&dir, &["kernel.ptx"], &[("VF_TEST_STATUS", "7")]);
    assert_eq!(output.status.code(), Some(7));
    let (output, _) = run(&dir, &["kernel.ptx"], &[("VF_TEST_SIGNAL", "TERM")]);
    assert_eq!(output.status.signal(), Some(15));
    // What this program finds itself (until 2026-11-22: status 1).
    for (args, why) in [
        (&["missing.ptx"][..], "missing.ptx: No such file"),
        (&["controls.txt"][..], "controls.txt: expected a PTX module"),
        (&["-o"][..], "missing assembler option value"),
    ] {
        let (output, messages) = run(&dir, args, &[]);
        assert_eq!(output.status.code(), Some(255), "{args:?}: {messages}");
        assert!(
            messages.starts_with("vf-ptxas: ") && messages.contains(why),
            "{messages}"
        );
    }
    let (output, _) = run(&dir, &["kernel.ptx"], &[("VF_PTXAS_REAL", "/nonexistent/ptxas")]);
    assert_eq!(output.status.code(), Some(255));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_runfiles_view_is_kept_while_the_program_started_with_it_runs() {
    // `--vf-runfiles SOURCE CACHE COMMAND...` runs COMMAND with RUNFILES_DIR set to the view, and
    // the lock of the view held by COMMAND and what it becomes or starts. (Until 2026-11-23 the
    // view was printed and nothing held it.)
    let dir = directory();
    let cache = dir.join("cache");
    for app in ["a", "b", "c"] {
        let bin = dir.join(format!("{app}.runfiles/cuda/bin"));
        fs::create_dir_all(&bin).unwrap();
        fs::write(bin.join("ptxas"), "NVIDIA").unwrap();
        fs::write(bin.join("nvlink"), app).unwrap();
    }
    let with_view = |app: &str| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_vf-ptxas"));
        command
            .arg("--vf-runfiles")
            .arg(dir.join(format!("{app}.runfiles")))
            .arg(&cache);
        command
    };
    // A program started with the view of a says its RUNFILES_DIR, then becomes `sleep`.
    let said = dir.join("said");
    let mut program = with_view("a")
        .args([
            "sh",
            "-c",
            "printf %s \"$RUNFILES_DIR\" > \"$0.part\"; mv \"$0.part\" \"$0\"; exec sleep 60",
        ])
        .arg(&said)
        .spawn()
        .unwrap();
    let start = std::time::Instant::now();
    while !said.exists() {
        assert!(
            start.elapsed() < std::time::Duration::from_secs(20),
            "the program said nothing"
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let view_a = PathBuf::from(fs::read_to_string(&said).unwrap());
    assert!(view_a.starts_with(&cache), "{}", view_a.display());
    assert_eq!(fs::read_to_string(view_a.join("cuda/bin/nvlink")).unwrap(), "a");
    assert_eq!(
        fs::read_link(view_a.join("cuda/bin/ptxas")).unwrap(),
        Path::new(env!("CARGO_BIN_EXE_vf-ptxas")).canonicalize().unwrap()
    );
    // Another program with the view of b, while a's runs: a's is left.
    assert!(with_view("b").arg("true").status().unwrap().success());
    assert!(view_a.is_dir());
    // Once a's has ended, the next view made removes its view.
    program.kill().unwrap();
    program.wait().unwrap();
    assert!(with_view("c").arg("true").status().unwrap().success());
    assert!(!view_a.exists());
    // What a command that cannot be run ends with, as a shell has it.
    let output = with_view("c").arg("no-such-command-here").output().unwrap();
    assert_eq!(output.status.code(), Some(127));
    assert!(String::from_utf8_lossy(&output.stderr).contains("no-such-command-here"));
    let output = with_view("c").arg(dir.join("said")).output().unwrap();
    assert_eq!(output.status.code(), Some(126));
    // Runfiles that are not there: 1, and the command is not run.
    let output = Command::new(env!("CARGO_BIN_EXE_vf-ptxas"))
        .arg("--vf-runfiles")
        .arg(dir.join("none.runfiles"))
        .arg(&cache)
        .arg("true")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).starts_with("vf-ptxas: CUDA runfiles: "));
    fs::remove_dir_all(dir).unwrap();
}
