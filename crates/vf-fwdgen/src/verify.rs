//! Host-side checks of the generated libraries (no GPU): the exports and their versions are those of
//! the CUDA 13.5 library, nothing is imported but from the C library, every forwarded function exists in
//! the 12.9 library, and the library loads and does not load the 12.9 one before a call.

use crate::{Args, bf16, elf, include_dir, selected};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;

pub fn run(args: &Args) -> Result<(), String> {
    let _ = include_dir;
    let mut failed = 0;
    let mut variants = 0;
    for spec in selected(args)? {
        let built_path = args.out.join(spec.new_name);
        let problems = check_library(args, spec, &built_path, &args.out.join("gen").join(spec.new_name), &[])?;
        failed += say(spec.new_name, &problems, args, spec)?;
        if bf16::variant_of(spec.key).is_some() {
            let gdir = args.out.join("gen").join(bf16::DIR).join(spec.new_name);
            let mut problems = check_library(
                args,
                spec,
                &args.out.join(bf16::DIR).join(spec.new_name),
                &gdir,
                &["vf_icpt", "vf_icpt_resolve", "vf_fwd_slow"],
            )?;
            problems.extend(bf16::check_variant(spec, &gdir));
            failed += say(&format!("{}/{}", bf16::DIR, spec.new_name), &problems, args, spec)?;
            variants += 1;
        }
    }
    if variants > 0 {
        let mut problems = bf16::check_runtime(&args.out.join(bf16::DIR).join(bf16::RUNTIME));
        problems.extend(bf16::check_kernels());
        if problems.is_empty() {
            println!(
                "PASS  {}/{} (exports the runtime's functions only; nothing imported but libc; the kernels of kernels.manifest, sm_70 machine code only)",
                bf16::DIR,
                bf16::RUNTIME
            );
        } else {
            failed += 1;
            println!("FAIL  {}/{}", bf16::DIR, bf16::RUNTIME);
            for p in problems.iter().take(20) {
                println!("    {p}");
            }
        }
    }
    if failed > 0 {
        Err(format!("{failed} libraries failed"))
    } else {
        Ok(())
    }
}

fn say(name: &str, problems: &[String], args: &Args, spec: &crate::libs::LibSpec) -> Result<usize, String> {
    if problems.is_empty() {
        let new = elf::read(&args.new.join(spec.new_name))?;
        println!(
            "PASS  {} ({} exports, versions and soname as in CUDA 13.5; loads; nothing imported but libc)",
            name,
            new.exports.len()
        );
        Ok(0)
    } else {
        println!("FAIL  {name}");
        for p in problems.iter().take(20) {
            println!("    {p}");
        }
        Ok(1)
    }
}

/// The checks of one built forwarding library (a default one or a variant).
fn check_library(
    args: &Args,
    spec: &crate::libs::LibSpec,
    built_path: &std::path::Path,
    gdir: &std::path::Path,
    internal: &[&str],
) -> Result<Vec<String>, String> {
    let mut problems: Vec<String> = Vec::new();
    {
        let new = elf::read(&args.new.join(spec.new_name))?;
        let old = elf::read(&args.old.join(spec.old_name))?;
        match elf::read(built_path) {
            Err(e) => problems.push(e),
            Ok(built) => {
                if built.soname.as_deref() != Some(spec.new_name) {
                    problems.push(format!("SONAME is {:?}", built.soname));
                }
                let want: BTreeMap<&str, (&Option<String>, u8)> = new
                    .exports
                    .iter()
                    .map(|s| (s.name.as_str(), (&s.version, s.kind)))
                    .collect();
                let have: BTreeMap<&str, (&Option<String>, u8)> = built
                    .exports
                    .iter()
                    .map(|s| (s.name.as_str(), (&s.version, s.kind)))
                    .collect();
                for (n, v) in &want {
                    match have.get(n) {
                        None => problems.push(format!("missing export {n}")),
                        Some(h) if h != v => problems.push(format!("{n}: version/type {h:?}, 13.5 has {v:?}")),
                        _ => {}
                    }
                }
                for n in have.keys().filter(|n| !want.contains_key(*n)) {
                    problems.push(format!("extra export {n}"));
                }
                let verdefs: Vec<&str> = built.verdefs.iter().map(|(_, n, _)| n.as_str()).collect();
                let want_verdefs: Vec<&str> = new.verdefs.iter().map(|(_, n, _)| n.as_str()).collect();
                if verdefs != want_verdefs {
                    problems.push(format!("version definitions {verdefs:?}, 13.5 has {want_verdefs:?}"));
                }
                if built.needed.iter().any(|n| {
                    !(n.starts_with("libc.so")
                        || n.starts_with("ld-linux")
                        || n.starts_with("libdl.so")
                        || n.starts_with("libpthread.so"))
                }) {
                    problems.push(format!("needs {:?}", built.needed));
                }
                // Imports: all from the C library (resolved by the loader of this very process).
                for imp in built.imports.iter().filter(|s| !s.weak) {
                    let c = CString::new(imp.name.as_str()).unwrap();
                    // SAFETY: dlsym with a valid handle constant and a NUL-terminated name.
                    if unsafe { libc::dlsym(libc::RTLD_DEFAULT, c.as_ptr()) }.is_null() {
                        problems.push(format!("undefined symbol {} is not from the C library", imp.name));
                    }
                }
                // Every forwarded target exists in the 12.9 library.
                let old_names: BTreeSet<&str> = old.exports.iter().map(|s| s.name.as_str()).collect();
                let manifest = std::fs::read_to_string(gdir.join("manifest.tsv")).map_err(|e| e.to_string())?;
                let mut listed = 0;
                for line in manifest.lines() {
                    let f: Vec<&str> = line.split('\t').collect();
                    listed += 1;
                    if f[1] != "stub" && !old_names.contains(f[2]) {
                        problems.push(format!(
                            "{} forwards to {}, which {} does not export",
                            f[0], f[2], spec.old_name
                        ));
                    }
                }
                if listed != new.exports.len() {
                    problems.push(format!(
                        "manifest lists {listed} functions, 13.5 exports {}",
                        new.exports.len()
                    ));
                }
                // Loads, without loading the 12.9 library (it is loaded by the first call).
                let path = CString::new(built_path.to_str().unwrap()).unwrap();
                // SAFETY: loading a library we just built; its constructors do nothing.
                let handle = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
                if handle.is_null() {
                    // SAFETY: dlerror returns a valid string after a failed dlopen.
                    let msg = unsafe { std::ffi::CStr::from_ptr(libc::dlerror()) }
                        .to_string_lossy()
                        .into_owned();
                    problems.push(format!("dlopen: {msg}"));
                } else {
                    let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
                    let real = std::fs::canonicalize(args.old.join(spec.old_name)).ok();
                    if let Some(real) = real
                        && spec.old_name != spec.new_name
                        && maps.contains(real.to_str().unwrap_or("?"))
                    {
                        problems.push("the 12.9 library was loaded at dlopen (binding is not lazy)".into());
                    }
                    for n in ["vf_fwd_resolve", "vf_slots"].iter().chain(internal) {
                        let c = CString::new(*n).unwrap();
                        // SAFETY: as above.
                        if !unsafe { libc::dlsym(handle, c.as_ptr()) }.is_null() {
                            problems.push(format!("internal symbol {n} is exported"));
                        }
                    }
                    // SAFETY: closes the handle opened above.
                    unsafe { libc::dlclose(handle) };
                }
            }
        }
    }
    Ok(problems)
}
