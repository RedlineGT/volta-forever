//! Telling the user which kernels run emulated instructions.
//!
//! A kernel that uses instructions this GPU lacks runs through longer sequences, slower than on the
//! GPU the program was built for. It is announced once, when it is given to the program; past a number
//! of kernels the following are only counted, and the totals are given when the process ends. A kernel
//! is its name and the image it comes from: two kernels of one name in two images (two
//! specializations of one kernel) are two kernels (`Said`).

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use vf_ptx::EmulatedCounts;

/// Kernel names are shortened to this many characters in the list (C++ names run to hundreds).
const NAME_SHOWN: usize = 100;

/// How many kernels a record of what was said keeps (`Said`, the refusals of the proxy): beyond them
/// (thousands of specializations that a program's autotuning makes, for one), what was not said is
/// not said, and one line says so; the memory they take stays bounded.
pub const KEPT_KERNELS: usize = 4096;

/// One load of an image: the handle the driver gave it and the number of that load (a handle that
/// the driver gives again, after an unload, is another load). `Load::default()`: one load for all.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Load {
    pub handle: usize,
    pub number: u64,
}

/// What `Said::see` makes of a kernel.
#[derive(Debug, PartialEq, Eq)]
pub enum Seen {
    /// The first kernel of its name.
    First,
    /// A kernel of a name seen before, of another image than those seen.
    Another,
    /// A kernel seen before: from the same load, or from an image of the same digest.
    Again,
    /// A kernel not seen before, beyond the `KEPT_KERNELS` kept: not kept.
    Full,
}

/// The kernels of each name seen: the first by its load, the others by the digest of their image. A
/// digest is asked for (`see`) only when a kernel of a name seen before comes from another load, so
/// that nothing is computed for the kernels that are seen once, or from one load.
#[derive(Default)]
pub struct Said {
    first: HashMap<String, Load>,
    images: HashSet<(String, u64)>,
}

impl Said {
    /// Kernel `name` of `load` is given or refused. `digest(load)` gives the digest of the image of a
    /// load, or 0 when it is not known (images of unknown digests are taken as one).
    pub fn see(&mut self, name: &str, load: Load, digest: &dyn Fn(Load) -> u64) -> Seen {
        let Some(&first) = self.first.get(name) else {
            if self.len() >= KEPT_KERNELS {
                return Seen::Full;
            }
            self.first.insert(name.to_string(), load);
            return Seen::First;
        };
        if first == load {
            return Seen::Again;
        }
        let image = digest(load);
        if image != 0 && image == digest(first) {
            return Seen::Again;
        }
        let kernel = (name.to_string(), image);
        if self.images.contains(&kernel) {
            return Seen::Again;
        }
        if self.len() >= KEPT_KERNELS {
            return Seen::Full;
        }
        self.images.insert(kernel);
        Seen::Another
    }

    /// How many kernels are kept.
    pub fn len(&self) -> usize {
        self.first.len() + self.images.len()
    }

    pub fn is_empty(&self) -> bool {
        self.first.is_empty()
    }
}

#[derive(Default)]
struct State {
    said: Said,
    /// How many kernels were announced.
    count: usize,
    totals: EmulatedCounts,
    /// Kernels were given beyond those that `said` keeps: not counted.
    beyond: bool,
}

/// The kernels announced so far in a process.
pub struct Announcer {
    state: Mutex<Option<State>>,
}

impl Default for Announcer {
    fn default() -> Self {
        Announcer::new()
    }
}

impl Announcer {
    pub const fn new() -> Announcer {
        Announcer {
            state: Mutex::new(None),
        }
    }

    /// The lines to print when `kernel`, which has `counts` of emulated instructions, is given to the
    /// program from `load` (`Said::see`, with `digest`): none if it was announced before or has none.
    /// The first `limit` kernels are listed, or all of them with `all`; with a limit of zero nothing
    /// is ever printed. A kernel of a name announced before, from another image, is listed as another,
    /// with `shown` (what names its image, when that is known: "image XXH64 ...").
    #[allow(clippy::too_many_arguments)]
    pub fn given(
        &self,
        kernel: &str,
        load: Load,
        digest: &dyn Fn(Load) -> u64,
        shown: Option<&str>,
        counts: &EmulatedCounts,
        limit: usize,
        all: bool,
    ) -> Vec<String> {
        let mut lines = Vec::new();
        if !counts.any() {
            return lines;
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let state = state.get_or_insert_with(State::default);
        let again = match state.said.see(kernel, load, digest) {
            Seen::Again => return lines,
            Seen::First => false,
            Seen::Another => true,
            Seen::Full => {
                if !std::mem::replace(&mut state.beyond, true) && limit > 0 {
                    lines.push(format!(
                        "  further kernels with emulated instructions are neither listed nor counted: more than \
                         {KEPT_KERNELS} were given"
                    ));
                }
                return lines;
            }
        };
        state.count += 1;
        state.totals.add(counts);
        let rank = state.count;
        if limit == 0 {
            return lines;
        }
        if rank == 1 {
            lines.push(
                "warning: this GPU lacks instructions that kernels of the program were compiled with; they are \
                 emulated, and these kernels run slower than on the GPU they were built for:"
                    .to_string(),
            );
        }
        if rank <= limit || all {
            let name = match kernel.char_indices().nth(NAME_SHOWN) {
                Some((end, _)) if !all => format!("{}...", &kernel[..end]),
                _ => kernel.to_string(),
            };
            let which = match (again, shown) {
                (false, _) => String::new(),
                (true, None) => " (another kernel of this name)".to_string(),
                (true, Some(shown)) => format!(" (another kernel of this name, {shown})"),
            };
            lines.push(format!("  emulated in {name}{which}: {}", counts.summary()));
        } else if rank == limit + 1 {
            lines.push(
                "  further kernels with emulated instructions are counted, not listed (VF_LOG_LEVEL=2 lists them \
                 all, VF_EMULATION_WARNINGS=N lists N)"
                    .to_string(),
            );
        }
        lines
    }

    /// The line to print when the process ends, if kernels with emulated instructions were given to
    /// the program.
    pub fn summary(&self, limit: usize) -> Option<String> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let state = state.as_ref()?;
        if limit == 0 || state.count == 0 {
            return None;
        }
        let count = state.count;
        Some(format!(
            "{count} {} with emulated instructions {} given to the program{}: {}",
            if count == 1 { "kernel" } else { "kernels" },
            if count == 1 { "was" } else { "were" },
            if state.beyond {
                ", and more that were not counted"
            } else {
                ""
            },
            state.totals.summary()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(tanh: usize) -> EmulatedCounts {
        EmulatedCounts {
            tanh,
            ..EmulatedCounts::default()
        }
    }

    /// Kernels of one load, whose image has no digest: a kernel is its name.
    fn by_name(announcer: &Announcer, kernel: &str, counts: &EmulatedCounts, limit: usize, all: bool) -> Vec<String> {
        announcer.given(kernel, Load::default(), &|_| 0, None, counts, limit, all)
    }

    /// A kernel of load `number`, whose image's digest is `number / 10`, shown as such.
    fn of_load(announcer: &Announcer, kernel: &str, number: u64, counts: &EmulatedCounts) -> Vec<String> {
        let shown = format!("image XXH64 {:016x}", number / 10);
        announcer.given(
            kernel,
            Load { handle: 1, number },
            &|load| load.number / 10,
            Some(&shown),
            counts,
            3,
            false,
        )
    }

    #[test]
    fn announces_each_kernel_once_up_to_the_limit() {
        let announcer = Announcer::new();
        assert_eq!(announcer.summary(2), None);
        assert!(by_name(&announcer, "plain", &counts(0), 2, false).is_empty());
        let first = by_name(&announcer, "a", &counts(1), 2, false);
        assert_eq!(first.len(), 2);
        assert!(first[0].starts_with("warning: this GPU lacks instructions"));
        assert_eq!(first[1], "  emulated in a: 1 tanh");
        assert!(by_name(&announcer, "a", &counts(1), 2, false).is_empty());
        assert_eq!(
            by_name(&announcer, "b", &counts(2), 2, false),
            ["  emulated in b: 2 tanh"]
        );
        let third = by_name(&announcer, "c", &counts(4), 2, false);
        assert_eq!(third.len(), 1);
        assert!(third[0].contains("counted, not listed"));
        assert!(by_name(&announcer, "d", &counts(8), 2, false).is_empty());
        // Listed whatever the limit when all are asked for, with the whole name.
        let long = "k".repeat(300);
        assert_eq!(
            by_name(&announcer, &long, &counts(16), 2, true),
            [format!("  emulated in {long}: 16 tanh")]
        );
        assert_eq!(
            announcer.summary(2).unwrap(),
            "5 kernels with emulated instructions were given to the program: 31 tanh"
        );
    }

    #[test]
    fn shortens_long_names_and_can_be_silenced() {
        let announcer = Announcer::new();
        let long = "k".repeat(300);
        let lines = by_name(&announcer, &long, &counts(1), 20, false);
        assert_eq!(lines[1], format!("  emulated in {}...: 1 tanh", "k".repeat(100)));
        assert_eq!(
            announcer.summary(20).unwrap(),
            "1 kernel with emulated instructions was given to the program: 1 tanh"
        );
        let silent = Announcer::new();
        assert!(by_name(&silent, "a", &counts(1), 0, false).is_empty());
        assert_eq!(silent.summary(0), None);
    }

    #[test]
    fn a_kernel_is_its_name_and_its_image() {
        let announcer = Announcer::new();
        assert_eq!(of_load(&announcer, "k", 10, &counts(1)).len(), 2);
        // The same kernel given again, from the same load, or from another load of the same image:
        // nothing.
        assert!(of_load(&announcer, "k", 10, &counts(1)).is_empty());
        assert!(of_load(&announcer, "k", 11, &counts(1)).is_empty());
        // Another kernel of the name, from another image: announced, with its image.
        assert_eq!(
            of_load(&announcer, "k", 220, &counts(2)),
            ["  emulated in k (another kernel of this name, image XXH64 0000000000000016): 2 tanh"]
        );
        assert!(of_load(&announcer, "k", 221, &counts(2)).is_empty());
        let unknown = announcer.given("k", Load { handle: 2, number: 3 }, &|_| 0, None, &counts(4), 3, false);
        assert_eq!(unknown, ["  emulated in k (another kernel of this name): 4 tanh"]);
        // Still within the limit, counted as kernels of their own.
        let fourth = of_load(&announcer, "k", 440, &counts(8));
        assert!(fourth[0].contains("counted, not listed"), "{fourth:?}");
        assert!(of_load(&announcer, "k", 550, &counts(16)).is_empty());
        assert_eq!(
            announcer.summary(3).unwrap(),
            "5 kernels with emulated instructions were given to the program: 31 tanh"
        );
    }

    #[test]
    fn a_digest_is_asked_for_only_when_another_load_has_a_kernel_of_the_name() {
        let asked = std::cell::Cell::new(0);
        let digest = |load: Load| {
            asked.set(asked.get() + 1);
            load.number / 10
        };
        let mut said = Said::default();
        let load = |number| Load { handle: 7, number };
        assert_eq!(said.see("k", load(10), &digest), Seen::First);
        assert_eq!(said.see("k", load(10), &digest), Seen::Again);
        assert_eq!(said.see("other", load(20), &digest), Seen::First);
        assert_eq!(asked.get(), 0);
        // Another load of the first's image, then another image.
        assert_eq!(said.see("k", load(11), &digest), Seen::Again);
        assert_eq!(said.see("k", load(30), &digest), Seen::Another);
        assert_eq!(said.see("k", load(31), &digest), Seen::Again);
        assert_eq!(asked.get(), 6);
        assert_eq!(said.len(), 3);
    }

    #[test]
    fn what_is_kept_of_the_kernels_said_is_bounded() {
        // Kernels of one name from many images, one load each (the digest of load n is n).
        let announcer = Announcer::new();
        let mut lines = Vec::new();
        for n in 1..=KEPT_KERNELS as u64 + 100 {
            let load = Load {
                handle: n as usize,
                number: n,
            };
            lines.extend(announcer.given("k", load, &|load| load.number, None, &counts(1), 20, false));
        }
        let state = announcer.state.lock().unwrap();
        assert_eq!(state.as_ref().unwrap().said.len(), KEPT_KERNELS);
        drop(state);
        assert_eq!(
            lines
                .iter()
                .filter(|l| l.contains("neither listed nor counted"))
                .count(),
            1,
            "{lines:?}"
        );
        assert_eq!(lines.len(), 1 + 20 + 1 + 1);
        assert_eq!(
            announcer.summary(20).unwrap(),
            format!(
                "{KEPT_KERNELS} kernels with emulated instructions were given to the program, and more that were not \
                 counted: {KEPT_KERNELS} tanh"
            )
        );
        // A kernel kept is still known for one beyond.
        let first = Load { handle: 1, number: 1 };
        assert!(
            announcer
                .given("k", first, &|load| load.number, None, &counts(1), 20, false)
                .is_empty()
        );
    }
}
