//! Instructions that Volta runs through a longer sequence than the one instruction the program was
//! compiled with. What a kernel has of them is counted, so that the user can be told that it runs
//! slower than on the GPU it was built for.

/// Counts, by kind, of the instructions of a function that `lower_to_volta` replaces by a slower
/// sequence. They are counted in the text of the program, before it is rewritten.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct EmulatedCounts {
    /// Tensor-core products of the shapes of Turing and later (`mma.m16n8k8`, `m16n8k16`, ...).
    pub fp16_tensor: usize,
    /// Integer tensor-core products (`mma.m8n8k16` and Ampere's shapes on int8).
    pub integer_tensor: usize,
    /// FP8 tensor-core products, which run on Volta's tensor cores after a conversion to halves.
    pub fp8_tensor: usize,
    /// Block-scaled tensor-core products (NVFP4, MXFP4, MXF8F6F4): E2M1 or FP8 operands with block
    /// scales, run on Volta's tensor cores after decoding to halves and applying the scales.
    pub fp4_tensor: usize,
    /// Tensor-core products of bf16, tf32 or f64. Some BF16 groups can run on Volta Tensor Cores.
    pub thread_tensor: usize,
    /// `ldmatrix` instructions.
    pub ldmatrix: usize,
    /// `movmatrix` instructions.
    pub movmatrix: usize,
    /// Asynchronous copies (`cp.async` with addresses; the group instructions are not counted).
    pub cp_async: usize,
    /// `tanh.approx.f32` instructions.
    pub tanh: usize,
    /// Conversions of two floats to a packed pair (`cvt.rn.f16x2.f32`, `cvt.rn.bf16x2.f32`).
    pub cvt_packed: usize,
    /// Conversions to and from FP8.
    pub fp8_cvt: usize,
    /// `min` and `max` that return NaN.
    pub nan_minmax: usize,
    /// Other instructions of later targets with a plain equivalent: bf16 and tf32 conversions,
    /// reductions over the warp, `min`/`max` with `.relu`, loads and stores of 256 bits, arithmetic on
    /// pairs of floats.
    pub other: usize,
    /// Not of PTX: the instructions that the translation of machine code for a later GPU added to a
    /// function (sequences for the instructions this GPU lacks, copies, waits). Counted by the
    /// translator (`vf-sass`), not by `count_emulated`.
    pub machine_code: usize,
}

impl EmulatedCounts {
    fn fields(&self) -> [(usize, &'static str); 14] {
        [
            (self.fp16_tensor, "FP16 tensor-core products"),
            (self.integer_tensor, "integer tensor-core products"),
            (self.fp8_tensor, "FP8 tensor-core products"),
            (self.fp4_tensor, "block-scaled tensor-core products"),
            (self.thread_tensor, "BF16/TF32/FP64 tensor-core products"),
            (self.ldmatrix, "ldmatrix"),
            (self.movmatrix, "movmatrix"),
            (self.cp_async, "asynchronous copies"),
            (self.tanh, "tanh"),
            (self.cvt_packed, "conversions to packed floats"),
            (self.fp8_cvt, "FP8 conversions"),
            (self.nan_minmax, "min/max returning NaN"),
            (self.other, "other instructions of later GPUs"),
            (self.machine_code, "instructions added in translated machine code"),
        ]
    }

    /// Something is emulated.
    pub fn any(&self) -> bool {
        self.total() > 0
    }

    /// Number of emulated instructions.
    pub fn total(&self) -> usize {
        self.fields().iter().map(|(count, _)| count).sum()
    }

    pub fn add(&mut self, other: &EmulatedCounts) {
        self.fp16_tensor += other.fp16_tensor;
        self.integer_tensor += other.integer_tensor;
        self.fp8_tensor += other.fp8_tensor;
        self.fp4_tensor += other.fp4_tensor;
        self.thread_tensor += other.thread_tensor;
        self.ldmatrix += other.ldmatrix;
        self.movmatrix += other.movmatrix;
        self.cp_async += other.cp_async;
        self.tanh += other.tanh;
        self.cvt_packed += other.cvt_packed;
        self.fp8_cvt += other.fp8_cvt;
        self.nan_minmax += other.nan_minmax;
        self.other += other.other;
        self.machine_code += other.machine_code;
    }

    /// The counts that are not zero, for a message: "192 FP16 tensor-core products, 60 ldmatrix".
    pub fn summary(&self) -> String {
        let parts: Vec<String> = self
            .fields()
            .iter()
            .filter(|(count, _)| *count > 0)
            .map(|(count, what)| format!("{count} {what}"))
            .collect();
        parts.join(", ")
    }
}

/// A kernel that runs emulated instructions, in its own body or in the functions it calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Emulated {
    pub kernel: String,
    pub counts: EmulatedCounts,
}

/// The emulated instructions of a function body, as the program has it. Only meaningful for a body
/// that `lower_to_volta` could rewrite: a form it refuses is counted like the others.
pub fn count_emulated(body: &[u8]) -> EmulatedCounts {
    let mut counts = EmulatedCounts::default();
    let separator = |byte: &u8| byte.is_ascii_whitespace() || matches!(byte, b'{' | b'}' | b';');
    for opcode in body.split(separator).filter_map(|word| std::str::from_utf8(word).ok()) {
        let has = |part: &str| opcode.contains(part);
        let field = if let Some(form) = opcode.strip_prefix("mma.sync.aligned.") {
            if has("block_scale") {
                &mut counts.fp4_tensor
            } else if has(".e4m3") || has(".e5m2") {
                &mut counts.fp8_tensor
            } else if has(".bf16") || has(".tf32") || has(".f64") {
                &mut counts.thread_tensor
            } else if has(".s32") {
                &mut counts.integer_tensor
            } else if form.starts_with("m8n8k4.") {
                // Volta's own product.
                continue;
            } else {
                &mut counts.fp16_tensor
            }
        } else if opcode.starts_with("ldmatrix.") {
            &mut counts.ldmatrix
        } else if opcode.starts_with("movmatrix.") {
            &mut counts.movmatrix
        } else if opcode.starts_with("stmatrix.")
            || opcode.starts_with("mbarrier.")
            || opcode.starts_with("cp.async.mbarrier.")
        {
            &mut counts.other
        } else if opcode.starts_with("cp.async.") {
            // The group instructions wait for nothing once the copies are synchronous.
            if matches!(
                opcode,
                "cp.async.commit_group" | "cp.async.wait_group" | "cp.async.wait_all"
            ) {
                continue;
            }
            &mut counts.cp_async
        } else if opcode == "tanh.approx.f32" {
            &mut counts.tanh
        } else if opcode.starts_with("cvt.")
            && (has(".e4m3") || has(".e5m2") || has(".e2m1") || has(".ue8m0") || has(".ue4m3"))
        {
            &mut counts.fp8_cvt
        } else if opcode.starts_with("cvt.") && opcode.ends_with("x2.f32") {
            // The packed bf16 conversion is among the packed conversions.
            &mut counts.cvt_packed
        } else if has(".NaN.") {
            &mut counts.nan_minmax
        } else if has(".bf16")
            || opcode.starts_with("min.f16")
            || opcode.starts_with("max.f16")
            || has(".tf32")
            || opcode.ends_with(".f32x2")
            || opcode.starts_with("redux.sync.")
            || has(".relu.")
            || has(".v8.")
            || [".v4.f64", ".v4.b64", ".v4.u64", ".v4.s64"]
                .iter()
                .any(|wide| opcode.ends_with(wide))
        {
            // What is only dropped (hints, `griddepcontrol`) costs nothing and is not counted.
            &mut counts.other
        } else {
            continue;
        };
        *field += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_by_kind() {
        let body = b"{\n\
            ldmatrix.sync.aligned.m8n8.x4.b16 {%r1, %r2, %r3, %r4}, [%rd1];\n\
            ldmatrix.sync.aligned.m8n8.x4.b16 {%r5, %r6, %r7, %r8}, [%rd2];\n\
            mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%r9}, {%r1}, {%r5}, {%r9};\n\
            mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%r10}, {%r2}, {%r6}, {%r10};\n\
            tanh.approx.f32 %f1, %f2;\n\
            cp.async.ca.shared.global [%rd3], [%rd4], 16;\n\
            cp.async.commit_group;\n\
            cvt.rn.f16x2.f32 %r11, %f1, %f2;\n\
            min.NaN.f32 %f3, %f1, %f2;\n\
            ret;\n\
        }";
        let counts = count_emulated(body);
        assert_eq!(
            counts,
            EmulatedCounts {
                fp16_tensor: 2,
                ldmatrix: 2,
                cp_async: 1,
                tanh: 1,
                cvt_packed: 1,
                nan_minmax: 1,
                ..EmulatedCounts::default()
            }
        );
        assert!(counts.any());
        assert_eq!(counts.total(), 8);
        assert_eq!(
            counts.summary(),
            "2 FP16 tensor-core products, 2 ldmatrix, 1 asynchronous copies, 1 tanh, 1 conversions to packed floats, \
             1 min/max returning NaN"
        );
    }

    #[test]
    fn nothing_in_a_plain_body() {
        let counts = count_emulated(b"{\n\tret;\n}");
        assert!(!counts.any());
        assert_eq!(counts.summary(), "");
    }

    #[test]
    fn integer_products_and_the_plain_equivalents() {
        let body = b"{\n\
            mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%r1, %r2}, {%r3}, {%r4}, {%r1, %r2};\n\
            cvt.rn.bf16.f32 %h1, %f1;\n\
            cvt.f32.bf16 %f2, %h1;\n\
            cvt.rn.bf16x2.f32 %r5, %f1, %f2;\n\
            max.relu.s32 %r6, %r7, %r8;\n\
            redux.sync.add.s32 %r9, %r10, -1;\n\
            ld.global.v8.b32 {%r1, %r2, %r3, %r4, %r5, %r6, %r7, %r8}, [%rd1];\n\
            st.global.v4.b64 [%rd1], {%rd2, %rd3, %rd4, %rd5};\n\
            griddepcontrol.wait;\n\
            max.f16x2 %r11, %r12, %r13;\n\
            min.f16 %h2, %h3, %h4;\n\
            ret;\n\
        }";
        let counts = count_emulated(body);
        assert_eq!(counts.integer_tensor, 1);
        assert_eq!(counts.fp16_tensor, 0);
        assert_eq!(counts.cvt_packed, 1);
        assert_eq!(counts.other, 8);
    }

    #[test]
    fn integer_ampere_shapes_counted_as_integer() {
        let body = b"{\n\
            mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%r1, %r2, %r3, %r4}, {%r5, %r6}, {%r7}, {%r1, %r2, %r3, %r4};\n\
            mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%r1, %r2, %r3, %r4}, {%r5, %r6, %r7, %r8}, {%r9, %r10}, {%r1, %r2, %r3, %r4};\n\
            mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%r1, %r2, %r3, %r4}, {%r5, %r6, %r7, %r8}, {%r9, %r10}, {%r1, %r2, %r3, %r4};\n\
            ret;\n\
        }";
        let counts = count_emulated(body);
        assert_eq!(counts.integer_tensor, 2);
        assert_eq!(counts.fp16_tensor, 1);
        // Saturating, written both ways.
        let counts = count_emulated(
            b"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite {%r1}, {%r5}, {%r9}, {%r1};\n\
              mma.sync.aligned.m16n8k16.row.col.satfinite.s32.u8.u8.s32 {%r1}, {%r5}, {%r7}, {%r1};\n",
        );
        assert_eq!((counts.integer_tensor, counts.fp16_tensor), (2, 0));
    }

    #[test]
    fn the_products_and_conversions_of_other_element_types() {
        let counts = count_emulated(
            b"{\n\
            mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 {%r1}, {%r5}, {%r9}, {%r1};\n\
            mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%r1}, {%r5}, {%r9}, {%r1};\n\
            mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%r1}, {%r5}, {%r9}, {%r1};\n\
            mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%fd1}, {%fd5}, {%fd9}, {%fd1};\n\
            mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%r1}, {%r5}, {%r9}, {%r1};\n\
            {cvt.rn.f16x2.e4m3x2 %r2, %rs1;}\n\
            @%p1 cvt.rn.satfinite.e5m2x2.f32 %rs1, %f1, %f2;\n\
            cvt.rn.tf32.f32 %r1, %f1;\n\
            fma.rn.f32x2 %rd1, %rd2, %rd3, %rd1;\n\
            ld.global.L2::cache_hint.u32 %r1, [%rd1], %rd2;\n\
            cvt.rn.f16x2.f32 %r1, %f1, %f2;\n\
            ret;\n\
        }",
        );
        assert_eq!(
            counts,
            EmulatedCounts {
                fp8_tensor: 1,
                thread_tensor: 3,
                fp8_cvt: 2,
                cvt_packed: 1,
                other: 2,
                ..EmulatedCounts::default()
            }
        );
        assert_eq!(
            counts.summary(),
            "1 FP8 tensor-core products, 3 BF16/TF32/FP64 tensor-core products, \
             1 conversions to packed floats, 2 FP8 conversions, 2 other instructions of later GPUs"
        );
    }
}
