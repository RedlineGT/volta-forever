//! GV100 register limit for an entire CTA. A graph records its kernel before
//! the driver checks this limit, so a launch-time retry can arrive too late.

/// Maximum registers per thread consistent with every previously launchable
/// CTA under the function's register claim and declared block dimensions.
pub(super) fn budget(registers: u32, info: &[u8]) -> u32 {
    // GV100's four subpartitions each own 16,384 registers and allocate
    // a warp's 32 registers per thread in units of 256 registers.
    let regs_per_warp = (32 * registers.max(1)).next_multiple_of(256);
    let hardware_threads = (16_384 / regs_per_warp * 4 * 32).min(1024);
    let threads = declared_threads(info).unwrap_or(1024).min(hardware_threads);
    // The per-block admission check rounds warp count up to four
    // subpartitions; each 128-thread group consumes the same allocation.
    let groups = threads.div_ceil(128).max(1);
    (65_536 / (128 * groups) / 8 * 8).min(255)
}

/// An absent or unrecognized shape never narrows the hardware bound.
fn declared_threads(mut info: &[u8]) -> Option<u32> {
    let mut maximum = 1024;
    while !info.is_empty() {
        let [format, kind, lo, hi] = *info.first_chunk::<4>()?;
        let size = if format == 4 {
            u16::from_le_bytes([lo, hi]) as usize
        } else {
            0
        };
        let (payload, rest) = info.get(4..)?.split_at_checked(size)?;
        if matches!(kind, 0x05 | 0x10) {
            // EIATTR_MAX_THREADS / EIATTR_REQNTID
            if format != 4 || size != 12 {
                return None;
            }
            let dims = payload.as_chunks::<4>().0.iter().copied().map(u32::from_le_bytes);
            let n = dims.into_iter().try_fold(1u32, u32::checked_mul)?;
            if n == 0 {
                return None;
            }
            maximum = maximum.min(n);
        }
        info = rest;
    }
    Some(maximum)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn shape(kind: u8, dims: [u32; 3]) -> Vec<u8> {
        let mut info = vec![4, kind, 12, 0];
        for d in dims {
            info.extend_from_slice(&d.to_le_bytes());
        }
        info
    }
    #[test]
    fn register_thresholds_include_whole_warps_and_scheduler_partitions() {
        assert_eq!(budget(12, &[]), 64);
        assert_eq!(budget(64, &[]), 64);
        assert_eq!(budget(65, &[]), 72);
        assert_eq!(budget(80, &[]), 80);
        assert_eq!(budget(129, &[]), 168);
        assert_eq!(budget(255, &[]), 255);
        for src in 1..=255 {
            for threads in 1u32..=1024 {
                let allowed = |r: u32| threads.div_ceil(128) * 4 * (r * 32).next_multiple_of(256) <= 65536;
                assert_eq!(allowed(src), allowed(budget(src, &[])));
            }
        }
    }
    #[test]
    fn small_shapes_keep_the_plain_first_image() {
        for kind in [0x05, 0x10] {
            assert_eq!(budget(64, &shape(kind, [32, 4, 2])), 255);
            assert_eq!(budget(64, &shape(kind, [512, 1, 1])), 128);
            assert_eq!(budget(64, &shape(kind, [897, 1, 1])), 64);
        }
        assert_eq!(budget(64, &[4, 0x10, 12, 0, 1, 0]), 64);
        assert_eq!(budget(64, &shape(0x05, [0, 1, 1])), 64);
        assert_eq!(budget(64, &shape(0x10, [u32::MAX, 4, 4])), 64);
    }
}
