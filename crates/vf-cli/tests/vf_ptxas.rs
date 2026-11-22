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
