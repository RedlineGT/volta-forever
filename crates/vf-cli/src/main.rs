//! `vf-translate`: look at the GPU code a file carries, adapt an image offline, or translate supported machine
//! code to sm_70.
//!
//! ```text
//! vf-translate inspect FILE...        fat binaries found in each file (a library, an executable, a raw image)
//! vf-translate extract FILE DIR [SM]  write the decompressed entries of FILE's fat binaries into DIR
//!                             (NNNN.sm_75.ptx, NNNN.sm_86.cubin ...), all or those for architecture SM
//! vf-translate adapt IN OUT           write the image a Volta driver should load in place of IN
//! vf-translate translate IN OUT [MAP [REPORT]]
//!                             write the sm_70 cubin for a supported sm_75 or sm_80 cubin IN; MAP gets one line per
//!                             instruction of the result (function, new offset, old offset, what was
//!                             done), MAP.lives what reference/validate.py reads beside it (where the
//!                             uniform predicates live, what was borrowed), REPORT one line per
//!                             function (`ok` or why it was refused, the name)
//! ```

use std::collections::BTreeMap;
use std::process::ExitCode;

use vf_adapt::{Outcome, Policy};
use vf_fatbin::{self as fatbin, KIND_ELF, KIND_LTO_IR, KIND_PTX};

fn kind_name(kind: u16) -> String {
    match kind {
        KIND_PTX => "PTX".to_string(),
        KIND_ELF => "machine code".to_string(),
        KIND_LTO_IR => "LTO-IR".to_string(),
        other => format!("kind {other}"),
    }
}

/// One line per (kind, architecture) with the number of entries and their stored size.
fn inspect(path: &str) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let is_text = bytes
        .first()
        .is_some_and(|b| b.is_ascii_graphic() || b.is_ascii_whitespace());
    if let Some(header) = vf_ptx::header(&bytes).filter(|_| is_text) {
        println!(
            "{path}: PTX text, version {}.{}, target sm_{}",
            header.version.0, header.version.1, header.target
        );
        return Ok(());
    }
    let containers = fatbin::find_containers(&bytes);
    if containers.is_empty() {
        match fatbin::cubin_sm(&bytes) {
            Some(sm) if bytes.get(0x12..0x14) == Some(&[0xbe, 0x00]) => {
                println!("{path}: cubin for sm_{sm}")
            }
            _ => println!("{path}: no GPU code found"),
        }
        return Ok(());
    }
    let mut table: BTreeMap<(u16, u32, String), (usize, usize)> = BTreeMap::new();
    for &offset in &containers {
        let fat = fatbin::parse(&bytes[offset..]).map_err(|e| format!("{path}: {e}"))?;
        for entry in &fat.entries {
            let row = table.entry((entry.kind(), entry.sm(), entry.arch_name())).or_default();
            row.0 += 1;
            row.1 += entry.stored.len();
        }
    }
    println!("{path}: {} fat binaries", containers.len());
    for ((kind, _, arch), (count, size)) in table {
        println!(
            "  {:12} {arch:8} {count:6} entries {:10.1} kB",
            kind_name(kind),
            size as f64 / 1e3
        );
    }
    Ok(())
}

/// One file per entry, numbered by fat binary.
fn extract(path: &str, dir: &str, only_sm: Option<u32>) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
    let mut written = 0;
    for (index, &offset) in fatbin::find_containers(&bytes).iter().enumerate() {
        let fat = fatbin::parse(&bytes[offset..]).map_err(|e| format!("{path}: {e}"))?;
        let mut occurrences = BTreeMap::new();
        for entry in fat.entries.iter().filter(|e| only_sm.is_none_or(|sm| e.sm() == sm)) {
            let extension = match entry.kind() {
                KIND_PTX => "ptx",
                KIND_ELF => "cubin",
                KIND_LTO_IR => "ltoir",
                _ => "bin",
            };
            let arch = entry.arch_name();
            let occurrence = occurrences.entry((arch.clone(), extension)).or_insert(0usize);
            *occurrence += 1;
            let name = if *occurrence == 1 {
                format!("{dir}/{index:04}.{arch}.{extension}")
            } else {
                format!("{dir}/{index:04}-{occurrence}.{arch}.{extension}")
            };
            let payload = entry
                .payload()
                .map_err(|e| format!("{path}: fat binary {index}: {e}"))?;
            std::fs::write(&name, &payload).map_err(|e| format!("{name}: {e}"))?;
            written += 1;
        }
    }
    println!("{path}: {written} entries written to {dir}");
    Ok(())
}

/// The kernels without code (`vf_adapt::Stubs`): what the layer does with each.
fn print_stubs(stubs: &vf_adapt::Stubs) {
    for kernel in &stubs.trap_only {
        println!("  without code (a trap): {kernel}: given at a lookup, its launch is refused");
    }
    for kernel in &stubs.empty {
        println!("  empty: {kernel}");
    }
}

fn adapt(input: &str, output: &str) -> Result<(), String> {
    let bytes = std::fs::read(input).map_err(|e| format!("{input}: {e}"))?;
    let policy = Policy::default().with_environment();
    let adapted = if bytes.starts_with(&fatbin::MAGIC.to_le_bytes()) {
        let fat = fatbin::parse(&bytes).map_err(|e| format!("{input}: {e}"))?;
        policy.adapt_fatbin(&fat).map_err(|e| format!("{input}: {e}"))?
    } else if fatbin::is_elf(&bytes) {
        policy.adapt_cubin(&bytes)
    } else {
        policy.adapt_ptx(&bytes)
    };
    match &adapted.outcome {
        Outcome::Native => println!("{input}: already holds code for sm_{}; copied", policy.device_sm),
        Outcome::Retargeted {
            entries,
            from_sm,
            tensor_rewritten,
            tanh_expanded,
            trapped,
            refused,
            emulated,
            stubs,
        } => {
            println!(
                "{input}: PTX for sm_{from_sm} retargeted to sm_{} ({entries} modules, {tensor_rewritten} \
                 tensor-core instructions rewritten, {tanh_expanded} tanh expanded, {} functions made to trap)",
                policy.device_sm,
                trapped.len()
            );
            for function in trapped {
                println!("  trap: {}: {}", function.function, function.reason);
            }
            for kernel in refused {
                println!(
                    "  refused kernel: {} ({}: {})",
                    kernel.kernel, kernel.function, kernel.reason
                );
            }
            for kernel in emulated {
                println!("  emulated in kernel: {}: {}", kernel.kernel, kernel.counts.summary());
            }
            print_stubs(stubs);
        }
        Outcome::Translated {
            entries,
            from_sm,
            functions,
            refused,
            emulated,
            stubs,
            ..
        } => {
            println!(
                "{input}: machine code for sm_{from_sm} translated to sm_{} ({entries} modules, {functions} \
                 functions, {} refused, {} with instructions added)",
                policy.device_sm,
                refused.len(),
                emulated.len()
            );
            for kernel in refused {
                println!("  refused: {}: {}", kernel.kernel, kernel.reason);
            }
            print_stubs(stubs);
        }
        Outcome::NotTranslated { from_sm, reason } => {
            return Err(format!(
                "{input}: machine code for sm_{from_sm} that could not be read: {reason}"
            ));
        }
        Outcome::NeedsTranslation { lowest_sm } => {
            return Err(format!(
                "{input}: machine code for sm_{lowest_sm} and later only: needs translation"
            ));
        }
        Outcome::Unknown => return Err(format!("{input}: not a fat binary, a cubin or PTX text")),
    }
    std::fs::write(output, adapted.image.as_deref().unwrap_or(&bytes)).map_err(|e| format!("{output}: {e}"))
}

/// The sm_70 cubin for a supported cubin, with what was refused on standard output.
fn translate(input: &str, output: &str, map: Option<&str>, report: Option<&str>) -> Result<(), String> {
    let bytes = std::fs::read(input).map_err(|e| format!("{input}: {e}"))?;
    if !vf_fatbin::is_elf(&bytes) {
        return Err(format!("{input}: not a cubin (ELF file)"));
    }
    let opts = vf_sass::translate::Options::from_environment();
    let result = vf_sass::translate::translate_with_options(&bytes, map.is_some(), &opts)
        .map_err(|e| format!("{input}: {e}"))?;
    std::fs::write(output, &result.bytes).map_err(|e| format!("{output}: {e}"))?;
    if let (Some(path), Some(rows)) = (map, &result.map) {
        let mut text = String::new();
        for (function, new, old, what) in rows {
            text.push_str(&format!("{function}\t{new:x}\t{old:x}\t{what}\n"));
        }
        std::fs::write(path, text).map_err(|e| format!("{path}: {e}"))?;
    }
    if let (Some(map), Some(lives)) = (map, &result.lives) {
        // As reference/translate.py writes it (`write_lives`).
        let mut text = String::new();
        for (function, first, second, how, offsets) in lives {
            let offsets: Vec<String> = offsets.iter().map(|at| format!("{at:x}")).collect();
            text.push_str(&format!(
                "{function}\t{first}\t{second}\t{how}\t{}\n",
                offsets.join(",")
            ));
        }
        let path = format!("{map}.lives");
        std::fs::write(&path, text).map_err(|e| format!("{path}: {e}"))?;
    }
    if let Some(path) = report {
        let mut text = String::new();
        for function in &result.report {
            text.push_str(&format!(
                "{}\t{}\n",
                function.refused.as_deref().unwrap_or("ok"),
                function.name
            ));
        }
        std::fs::write(path, text).map_err(|e| format!("{path}: {e}"))?;
    }
    let name = std::path::Path::new(input)
        .file_name()
        .map_or_else(|| input.into(), |n| n.to_string_lossy());
    let mut refused: Vec<(&str, &str)> = result
        .report
        .iter()
        .filter_map(|function| Some((function.name.as_str(), function.refused.as_deref()?)))
        .collect();
    refused.sort();
    println!("{name}: {} functions, {} refused", result.report.len(), refused.len());
    for (function, why) in refused {
        println!("  refused {}: {why}", function.chars().take(60).collect::<String>());
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let result = match args.as_slice() {
        ["inspect", files @ ..] if !files.is_empty() => files.iter().try_for_each(|file| inspect(file)),
        ["extract", file, dir] => extract(file, dir, None),
        ["extract", file, dir, sm] => match sm.trim_start_matches("sm_").parse() {
            Ok(sm) => extract(file, dir, Some(sm)),
            Err(_) => Err(format!("{sm}: not an architecture (75 or sm_75)")),
        },
        ["adapt", input, output] => adapt(input, output),
        ["translate", input, output] => translate(input, output, None, None),
        ["translate", input, output, map] => translate(input, output, Some(map), None),
        ["translate", input, output, map, report] => translate(input, output, Some(map), Some(report)),
        _ => {
            eprintln!(
                "usage: vf-translate inspect FILE...\n       vf-translate extract FILE DIR [SM]\n       \
                 vf-translate adapt IN OUT\n       vf-translate translate IN OUT [MAP [REPORT]]\n\
                 (the launcher has them as subcommands: vf inspect FILE...)"
            );
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("vf-translate: {message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vf_fatbin::NewEntry;

    #[test]
    fn extract_preserves_multiple_ptx_entries_for_one_architecture() {
        let fixture = include_bytes!("../../vf-fatbin/tests/data/elf75-ptx75-zstd.fatbin");
        let fat = fatbin::parse(fixture).unwrap();
        let ptx = fat.entries.iter().find(|entry| entry.kind() == KIND_PTX).unwrap();
        let data = fatbin::build(&[
            NewEntry::replace(ptx, b"first".to_vec(), 75, (8, 8)),
            NewEntry::replace(ptx, b"second".to_vec(), 75, (8, 8)),
            NewEntry::replace(ptx, b"third".to_vec(), 80, (8, 8)),
        ]);
        let tmp = std::process::Command::new("mktemp").arg("-d").output().unwrap();
        assert!(tmp.status.success());
        let path = std::path::Path::new(std::str::from_utf8(&tmp.stdout).unwrap().trim());
        let input = path.join("input.fatbin");
        std::fs::write(&input, data).unwrap();
        extract(input.to_str().unwrap(), path.to_str().unwrap(), Some(75)).unwrap();
        assert_eq!(std::fs::read(path.join("0000.sm_75.ptx")).unwrap(), b"first");
        assert_eq!(std::fs::read(path.join("0000-2.sm_75.ptx")).unwrap(), b"second");
        assert!(!path.join("0000.sm_80.ptx").exists());
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn translate_writes_the_lives_beside_the_map() {
        // What reference/translate.py writes for this cubin; reference/validate.py reads it beside
        // the map. (Until 2026-11-22 there was no such file: every uniform predicate "had no life".)
        let cubin = include_bytes!("../../../tests/kernels/translate/k_memory.sm75.cubin");
        let tmp = std::process::Command::new("mktemp").arg("-d").output().unwrap();
        assert!(tmp.status.success());
        let path = std::path::Path::new(std::str::from_utf8(&tmp.stdout).unwrap().trim());
        let input = path.join("k_memory.sm75.cubin");
        std::fs::write(&input, cubin).unwrap();
        let (output, map) = (path.join("out.cubin"), path.join("out.map"));
        translate(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            Some(map.to_str().unwrap()),
            None,
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(path.join("out.map.lives")).unwrap(),
            "k_generic_index\t0\t2\tfree\tb0,c0,d0,e0,f0,100\nk_generic\t0\t0\tfree\t30,40,50,60,70,80\n"
        );
        std::fs::remove_dir_all(path).unwrap();
    }
}
