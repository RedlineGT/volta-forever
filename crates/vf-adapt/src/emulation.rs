//! Telling the user which kernels run emulated instructions.
//!
//! A kernel that uses instructions this GPU lacks runs through longer sequences, slower than on the
//! GPU the program was built for. It is announced once, when it is given to the program; past a number
//! of kernels the following are only counted, and the totals are given when the process ends. A kernel
//! is its name and the image it comes from: two kernels of one name in two images (two
//! specializations of one kernel) are two kernels.

use std::collections::HashSet;
use std::sync::Mutex;

use vf_ptx::EmulatedCounts;

/// Kernel names are shortened to this many characters in the list (C++ names run to hundreds).
const NAME_SHOWN: usize = 100;

#[derive(Default)]
struct State {
    /// The kernels announced, by name and digest of their image.
    announced: HashSet<(String, u64)>,
    /// Their names.
    names: HashSet<String>,
    totals: EmulatedCounts,
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
    /// program from an image whose digest is `image` (0: not known): none if it was announced before
    /// or has none. The first `limit` kernels are listed, or all of them with `all`; with a limit of
    /// zero nothing is ever printed. A kernel of a name announced before, from another image, is
    /// listed with that image's digest.
    pub fn given(&self, kernel: &str, image: u64, counts: &EmulatedCounts, limit: usize, all: bool) -> Vec<String> {
        let mut lines = Vec::new();
        if !counts.any() {
            return lines;
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let state = state.get_or_insert_with(State::default);
        if !state.announced.insert((kernel.to_string(), image)) {
            return lines;
        }
        let again = !state.names.insert(kernel.to_string());
        state.totals.add(counts);
        let rank = state.announced.len();
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
            let which = match (again, image) {
                (false, _) => String::new(),
                (true, 0) => " (another kernel of this name)".to_string(),
                (true, image) => format!(" (another kernel of this name, image XXH64 {image:016x})"),
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
        if limit == 0 || state.announced.is_empty() {
            return None;
        }
        let count = state.announced.len();
        Some(format!(
            "{count} {} with emulated instructions {} given to the program: {}",
            if count == 1 { "kernel" } else { "kernels" },
            if count == 1 { "was" } else { "were" },
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

    #[test]
    fn announces_each_kernel_once_up_to_the_limit() {
        let announcer = Announcer::new();
        assert_eq!(announcer.summary(2), None);
        assert!(announcer.given("plain", 0, &counts(0), 2, false).is_empty());
        let first = announcer.given("a", 0, &counts(1), 2, false);
        assert_eq!(first.len(), 2);
        assert!(first[0].starts_with("warning: this GPU lacks instructions"));
        assert_eq!(first[1], "  emulated in a: 1 tanh");
        assert!(announcer.given("a", 0, &counts(1), 2, false).is_empty());
        assert_eq!(
            announcer.given("b", 0, &counts(2), 2, false),
            ["  emulated in b: 2 tanh"]
        );
        let third = announcer.given("c", 0, &counts(4), 2, false);
        assert_eq!(third.len(), 1);
        assert!(third[0].contains("counted, not listed"));
        assert!(announcer.given("d", 0, &counts(8), 2, false).is_empty());
        // Listed whatever the limit when all are asked for, with the whole name.
        let long = "k".repeat(300);
        assert_eq!(
            announcer.given(&long, 0, &counts(16), 2, true),
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
        let lines = announcer.given(&long, 0, &counts(1), 20, false);
        assert_eq!(lines[1], format!("  emulated in {}...: 1 tanh", "k".repeat(100)));
        assert_eq!(
            announcer.summary(20).unwrap(),
            "1 kernel with emulated instructions was given to the program: 1 tanh"
        );
        let silent = Announcer::new();
        assert!(silent.given("a", 0, &counts(1), 0, false).is_empty());
        assert_eq!(silent.summary(0), None);
    }

    #[test]
    fn a_kernel_is_its_name_and_its_image() {
        let announcer = Announcer::new();
        assert_eq!(announcer.given("k", 0x11, &counts(1), 3, false).len(), 2);
        // The same kernel given again, from the same image: nothing.
        assert!(announcer.given("k", 0x11, &counts(1), 3, false).is_empty());
        // Another kernel of the name, from another image: announced, with its image.
        assert_eq!(
            announcer.given("k", 0x22, &counts(2), 3, false),
            ["  emulated in k (another kernel of this name, image XXH64 0000000000000022): 2 tanh"]
        );
        assert_eq!(
            announcer.given("k", 0, &counts(4), 3, false),
            ["  emulated in k (another kernel of this name): 4 tanh"]
        );
        // Still within the limit, counted as kernels of their own.
        let fourth = announcer.given("k", 0x44, &counts(8), 3, false);
        assert!(fourth[0].contains("counted, not listed"), "{fourth:?}");
        assert!(announcer.given("k", 0x55, &counts(16), 3, false).is_empty());
        assert_eq!(
            announcer.summary(3).unwrap(),
            "5 kernels with emulated instructions were given to the program: 31 tanh"
        );
    }
}
