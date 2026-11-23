//! vf-fwdgen: builds the forwarding libraries of the CUDA 13 -> CUDA 12.9 library pairs.
//!
//!   vf-fwdgen generate --old DIR12/lib --new DIR13/lib --out DIR [--only key,key]
//!   vf-fwdgen verify   --old DIR12/lib --new DIR13/lib --out DIR [--only key,key]
//!   vf-fwdgen list
//!
//! `generate` reads the export list (with symbol versions) of each CUDA 13 library and of the CUDA 12.9
//! library that serves it, the prototypes in both toolkits' headers (`DIR/../include`), decides what
//! each function does (see plan.rs), writes the sources under `OUT/gen/` and builds the libraries into
//! `OUT/`, with a report. `verify` checks the built libraries on the host, without a GPU. `list`
//! prints the names of the libraries that a run without `--only` builds, one a line
//! (`scripts/package.sh` takes nothing less for an archive).
//!
//! For cuBLAS and cuBLASLt, `generate` also builds the opt-in bf16 variants and their runtime into
//! `OUT/bf16tc/` (bf16.rs), after checking the runtime's declarations against both toolkits'
//! headers; `verify` checks them too.

mod bf16;
mod elf;
mod emit;
mod libs;
mod plan;
mod protos;
mod report;
mod verify;

use libs::{LIBS, LibSpec};
use protos::Proto;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub struct Args {
    pub old: PathBuf,
    pub new: PathBuf,
    pub out: PathBuf,
    pub only: Option<Vec<String>>,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let (mut old, mut new, mut out, mut only) = (None, None, None, None);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = || it.next().cloned().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--old" => old = Some(PathBuf::from(value()?)),
            "--new" => new = Some(PathBuf::from(value()?)),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--only" => only = Some(value()?.split(',').map(str::to_string).collect()),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(Args {
        old: old.ok_or("--old is required")?,
        new: new.ok_or("--new is required")?,
        out: out.ok_or("--out is required")?,
        only,
    })
}

pub fn selected<'a>(args: &Args) -> Result<Vec<&'a LibSpec>, String> {
    match &args.only {
        None => Ok(LIBS.iter().collect()),
        Some(keys) => keys
            .iter()
            .map(|k| LIBS.iter().find(|l| l.key == k).ok_or(format!("no library `{k}`")))
            .collect(),
    }
}

pub fn include_dir(lib_dir: &Path) -> PathBuf {
    lib_dir.join("..").join("include")
}

type ProtoMap = BTreeMap<String, Proto>;

/// Prototypes of every header group of a library in one toolkit, cached by group.
struct ProtoCache {
    cache: BTreeMap<(String, String), ProtoMap>,
}

impl ProtoCache {
    fn get(&mut self, spec: &LibSpec, include: &Path, work: &Path) -> Result<ProtoMap, String> {
        let mut all = ProtoMap::new();
        for (i, group) in spec.headers.iter().enumerate() {
            let key = (include.display().to_string(), group.join(","));
            if !self.cache.contains_key(&key) {
                let tag = format!(
                    "{}-{}",
                    if key.0.contains("12.9") { "old" } else { "new" },
                    group.join("+").replace(['.', '/'], "_")
                );
                let _ = i;
                self.cache.insert(
                    key.clone(),
                    protos::read_group(include, group, spec.defines, work, &tag)?,
                );
            }
            for (name, p) in &self.cache[&key] {
                all.entry(name.clone()).or_insert_with(|| p.clone());
            }
        }
        Ok(all)
    }
}

/// How the status enumerators changed: (renumbered pairs (12.9, 13.5), only in 12.9, only in 13.5).
pub struct StatusDiff {
    pub renumbered: Vec<(String, i64, i64)>,
    pub only_old: Vec<String>,
    pub only_new: Vec<String>,
}

fn status_diff(spec: &LibSpec, args: &Args, work: &Path) -> Result<Option<StatusDiff>, String> {
    let Some(status) = &spec.status else { return Ok(None) };
    let group = spec.headers[0];
    let tag = |v: &str| format!("enum-{v}-{}", status.ty);
    let old = protos::enum_values(&include_dir(&args.old), group, status.ty, work, &tag("old"))?;
    let new = protos::enum_values(&include_dir(&args.new), group, status.ty, work, &tag("new"))?;
    let old_map: BTreeMap<&str, i64> = old.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    let new_map: BTreeMap<&str, i64> = new.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    let mut diff = StatusDiff {
        renumbered: Vec::new(),
        only_old: Vec::new(),
        only_new: Vec::new(),
    };
    for (n, v) in &old_map {
        match new_map.get(n) {
            Some(w) if w != v => diff.renumbered.push((n.to_string(), *v, *w)),
            Some(_) => {}
            None => diff.only_old.push(n.to_string()),
        }
    }
    diff.only_new = new_map
        .keys()
        .filter(|n| !old_map.contains_key(*n))
        .map(|n| n.to_string())
        .collect();
    Ok(Some(diff))
}

fn type_sizes(spec: &LibSpec, args: &Args, protos: &ProtoMap, work: &Path) -> Result<BTreeMap<String, usize>, String> {
    let mut types = std::collections::BTreeSet::new();
    for p in protos.values() {
        for param in protos::split_params(&p.params) {
            if protos::builtin_size(&param).is_none() {
                types.insert(plan::by_value_type(&param));
            }
        }
    }
    let types: Vec<String> = types.into_iter().collect();
    let headers: Vec<&str> = spec.headers.iter().flat_map(|g| g.iter().copied()).collect();
    protos::type_sizes(
        &include_dir(&args.new),
        &headers,
        &types,
        work,
        &format!("types-{}", spec.key),
    )
}

fn type_abi_classes(
    spec: &LibSpec,
    args: &Args,
    sizes: &BTreeMap<String, usize>,
    work: &Path,
) -> Result<BTreeMap<String, protos::AbiClass>, String> {
    let types: Vec<(String, usize)> = sizes.iter().map(|(t, s)| (t.clone(), *s)).collect();
    let headers: Vec<&str> = spec.headers.iter().flat_map(|g| g.iter().copied()).collect();
    protos::type_abi_classes(
        &include_dir(&args.new),
        &headers,
        &types,
        work,
        &format!("types-{}", spec.key),
    )
}

fn generate(args: &Args) -> Result<(), String> {
    let gen_dir = args.out.join("gen");
    let headers_dir = gen_dir.join("headers");
    std::fs::create_dir_all(&args.out).map_err(|e| e.to_string())?;
    let mut cache = ProtoCache { cache: BTreeMap::new() };
    let mut results = Vec::new();
    let mut variants = 0;
    for spec in selected(args)? {
        let new = elf::read(&args.new.join(spec.new_name))?;
        let old = elf::read(&args.old.join(spec.old_name))?;
        if new.soname.as_deref() != Some(spec.new_name) {
            return Err(format!("{}: SONAME is {:?}", spec.new_name, new.soname));
        }
        let new_protos = cache.get(spec, &include_dir(&args.new), &headers_dir)?;
        let old_protos = cache.get(spec, &include_dir(&args.old), &headers_dir)?;
        let diff = status_diff(spec, args, &headers_dir)?;
        let renumber = diff.as_ref().is_some_and(|d| !d.renumbered.is_empty());
        // The stack a call's arguments take is needed where the returned status is renumbered.
        let (sizes, abi_classes) = if renumber {
            let s = type_sizes(spec, args, &new_protos, &headers_dir)?;
            let ac = type_abi_classes(spec, args, &s, &headers_dir)?;
            (s, ac)
        } else {
            (BTreeMap::new(), BTreeMap::new())
        };
        let inputs = plan::Inputs {
            spec,
            new: &new,
            old: &old,
            new_protos: &new_protos,
            old_protos: &old_protos,
            renumber,
            sizes: &sizes,
            abi_classes: &abi_classes,
        };
        let entries = plan::plan(&inputs)?;
        let status_args = plan::status_arguments(&inputs, &entries);
        if !status_args.is_empty() {
            return Err(format!(
                "{}: functions take the renumbered status as an argument without an adapter: {status_args:?}",
                spec.new_name
            ));
        }
        let renumbering: emit::Renumbering = diff
            .as_ref()
            .map(|d| d.renumbered.iter().map(|(_, o, n)| (*o, *n)).collect())
            .unwrap_or_default();
        let target = emit::target_name(spec, &args.old)?;
        let dir = gen_dir.join(spec.new_name);
        let slots = emit::write_sources(spec, &entries, &target, &renumbering, &dir)?;
        emit::write_status_source(spec, &dir)?;
        report::write_manifest(&dir, &entries)?;
        emit::build(spec, &entries, &dir, &include_dir(&args.new), &args.out)?;
        eprintln!("built {} ({} exports, {slots} jumps)", spec.new_name, entries.len());
        if let Some(variant) = bf16::variant_of(spec.key) {
            if variants == 0 {
                let mut problems = bf16::check_declarations(args, &headers_dir)?;
                problems.extend(bf16::check_kernels());
                if !problems.is_empty() {
                    return Err(format!(
                        "the bf16 runtime does not match the toolkits or its kernels:\n  {}",
                        problems.join("\n  ")
                    ));
                }
            }
            let vdir = gen_dir.join(bf16::DIR).join(spec.new_name);
            let n = bf16::write_variant_sources(spec, variant, &entries, &target, &renumbering, &vdir)?;
            emit::write_status_source(spec, &vdir)?;
            report::write_manifest(&vdir, &entries)?;
            emit::build(
                spec,
                &entries,
                &vdir,
                &include_dir(&args.new),
                &args.out.join(bf16::DIR),
            )?;
            eprintln!(
                "built {}/{} ({n} functions for the bf16 runtime)",
                bf16::DIR,
                spec.new_name
            );
            variants += 1;
        }
        results.push(report::LibResult {
            spec,
            target,
            entries,
            diff,
            new,
            old,
        });
    }
    if variants > 0 {
        let rt = bf16::build_runtime(&gen_dir, &args.out)?;
        eprintln!("built {}", rt.display());
    }
    let text = report::render(&results);
    std::fs::write(args.out.join("REPORT.txt"), &text).map_err(|e| e.to_string())?;
    print!("{}", report::summary(&results));
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("generate") => parse_args(&args[1..]).and_then(|a| generate(&a)),
        Some("verify") => parse_args(&args[1..]).and_then(|a| verify::run(&a)),
        Some("list") if args.len() == 1 => {
            for spec in libs::LIBS {
                println!("{}", spec.new_name);
            }
            Ok(())
        }
        // The packaged runtime of the bf16 variants against the kernels and manifest this generator was
        // built with, without the toolkits (scripts/package.sh)
        Some("check-bf16") if args.len() == 2 => {
            let dir = std::path::Path::new(&args[1]);
            let mut problems = bf16::check_runtime(&dir.join(bf16::RUNTIME));
            problems.extend(bf16::check_kernels());
            if problems.is_empty() {
                println!("{}: the runtime of the bf16 variants holds the kernels of kernels.manifest", dir.display());
                Ok(())
            } else {
                Err(problems.join("; "))
            }
        }
        _ => Err(
            "usage: vf-fwdgen (generate|verify) --old DIR --new DIR --out DIR [--only key,key]\n       vf-fwdgen list\n       vf-fwdgen check-bf16 DIR"
                .to_string(),
        ),
    };
    if let Err(e) = result {
        eprintln!("vf-fwdgen: {e}");
        std::process::exit(1);
    }
}
