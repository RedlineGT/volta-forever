use super::{
    Frame, Program, Symbolic, Term, Value, Word, mma,
    program::Arg,
    values::{Affine, Axis},
};
const SIGNATURE: &str = "(.param .u64 a,.param .u32 stride,.param .u32 slots,.param .u64 weights,
.param .u32 groups,.param .u32 tiles,.param .u64 items,.param .u64 count,.param .u64 members,
.param .u64 out,.param .u32 width,.param .f32 limit) .maxntid 128,1,1";
#[test]
fn register_scope_and_scalar_width_are_checked() {
    let good = "{.reg .u32 %r<3>; mov.u32 %r0,-1; shl.b32 %r1,%r0,1; shr.u32 %r2,%r1,31; ret;}";
    let p = Program::parse(SIGNATURE, good).unwrap();
    let mut f = Frame::new(&p, 0);
    assert_eq!(
        f.advance(&p, usize::MAX, usize::MAX, &mut Symbolic::default()),
        Some(None)
    );
    let r = p.names.iter().position(|&name| name == "%r2").unwrap();
    assert_eq!(f.registers[r].scalar(), Some(1));
    for bad in [
        "{.reg .b32 %r<4>; mov.b32 %r01,1; ret;}",
        "{.reg .b32 %r; {.reg .b32 temporary; add.s32 %r,temporary,3;} ret;}",
        "{.reg .b32 %r; {.reg .b32 temporary; inside: mov.b32 temporary,1; mov.b32 %r,temporary;} ret;}",
    ] {
        assert!(Program::parse(SIGNATURE, bad).is_none(), "{bad}");
    }
    let inner =
        "{.reg .b32 %r; {.reg .b32 t; mov.b32 t,3; mov.b32 %r,t;} {.reg .b32 t; mov.b32 t,8; add.s32 %r,%r,t;} ret;}";
    let p = Program::parse(SIGNATURE, inner).unwrap();
    let mut f = Frame::new(&p, 0);
    assert_eq!(p.names.iter().filter(|&&name| name == "t").count(), 2);
    assert_eq!(
        f.advance(&p, usize::MAX, usize::MAX, &mut Symbolic::default()),
        Some(None)
    );
    assert_eq!(
        f.registers[p.names.iter().position(|&name| name == "%r").unwrap()].scalar(),
        Some(11)
    );
    let bad = Program::parse(SIGNATURE, "{.reg .u64 %ptr; ld.param.u64 %ptr,[slots]; ret;}").unwrap();
    assert!(
        Frame::new(&bad, 0)
            .advance(&bad, usize::MAX, usize::MAX, &mut Symbolic::default())
            .is_none()
    );
}

#[test]
fn affine_addresses_preserve_low_bits_and_reject_opaque_high_bits() {
    assert_eq!(Affine::num(-1).unsigned32(), Some(Affine::num(4294967295)));
    assert_eq!(Affine::num(4294967295).signed32(), Some(Affine::num(-1)));
    let wide = Affine::axis(Axis::Expert).times(1i64 << 28).unwrap();
    assert!(wide.signed32().is_none());
    assert!(Affine::axis(Axis::W).unsigned32().is_none());
    for lane in 0..32 {
        assert_eq!(Affine::tid(lane).mask(31), Some(Affine::num(lane as i64)));
        assert_eq!(Affine::tid(lane).shr_unsigned(5), Some(Affine::axis(Axis::Warp)));
    }
}
#[test]
fn every_nibble_is_mapped_to_the_original_column_and_k() {
    let mut math = Symbolic::default();
    for group in [0, 1, 31] {
        for table in 0..2 {
            for col in 0..32 {
                for k in 0..64 {
                    let quarter = k / 16;
                    let half = k % 16 / 8;
                    let physical = ((col / 8 * 2 + half) * 8 + col % 8) * 4 + quarter;
                    let offset = group * 2304
                        + table * 1152
                        + 4 * ((physical / 128) * 128 + (physical % 32) * 4 + (physical / 32) % 4);
                    let bits = math.weight(offset).unwrap();
                    let bit = (k % 8) / 2 * 4 + (k % 2) * 16;
                    let expected = Term::Raw {
                        group: group as u8,
                        table: table as u8,
                        column: col as u8,
                        k: k as u8,
                    };
                    assert_eq!(math.term(bits, bit, 4), Some(expected));
                }
            }
        }
    }
    assert!(math.weight(32 * 2304).is_none());
    assert!(math.input(4096).is_none());
}
fn tuple(x: &[usize]) -> Arg {
    Arg::Tuple(x.iter().map(|&r| Arg::Reg(r)).collect())
}
fn exercise_group(mutate: u8) -> bool {
    let mut math = Symbolic::default();
    let p = Program::parse(
        SIGNATURE,
        "{.reg .b32 v<10>; mov.b32 v0,v1; mov.b32 v2,v3; mov.b32 v4,v5; mov.b32 v6,v7; mov.b32 v8,v9; ret;}",
    )
    .unwrap();
    let mut lanes: Vec<_> = (0..32).map(|i| Frame::new(&p, i)).collect();
    let t: [usize; 10] = std::array::from_fn(|i| p.names.iter().position(|name| *name == format!("v{i}")).unwrap());
    for l in &mut lanes {
        for &r in &t {
            l.put(r, Value::Bits(Word::literal(0)))
        }
        for half in 0..2 {
            if l.lane < 4 {
                l.put(
                    t[2 * half],
                    Value::Bits(math.input((16 * l.lane + 2 * half) * 2).unwrap()),
                );
            }
            let mut word = Word::literal(0);
            for j in 0..2 {
                let term = Term::Nibble {
                    group: 0,
                    table: 0,
                    column: (l.lane / 4) as u8,
                    k: ((l.lane % 4) * 16 + 2 * half + j) as u8,
                };
                let src = math.word(term).unwrap();
                word.0[j * 16..j * 16 + 16].copy_from_slice(&src.0[..16]);
            }
            l.put(t[4 + half], Value::Bits(word));
        }
    }
    match mutate {
        1 => {
            lanes[5].put(t[3], Value::Bits(Word::literal(0x3f803f80)));
        }
        2 => {
            let wrong = lanes[0].registers[t[4]];
            lanes[1].put(t[4], wrong);
        }
        3 => {
            lanes[3].put(t[6], Value::Bits(Word::literal(0x80000000)));
        }
        _ => {}
    }
    let args = [tuple(&t[6..]), tuple(&t[..4]), tuple(&t[4..6]), tuple(&t[6..])];
    mma(&mut lanes, &args, &mut math, &mut [0; 256]).is_some()
}
#[test]
fn single_row_requires_all_four_pairs_from_the_same_k_and_a_zero_c() {
    assert!(exercise_group(0));
    for mutation in 1..=3 {
        assert!(!exercise_group(mutation));
    }
}

#[test]
fn guards_must_dominate_and_remain_immutable_on_both_sides_of_a_branch() {
    let prefix = "{ .reg .b32 r,v; .reg .b64 a; .reg .pred p; ";
    let load = "ld.global.nc.u32 r,[a]; ";
    let end = "target: mov.b32 v,r; ret; }";
    for (before, after, allowed) in [
        ("", "", true),
        ("@p bra target; ", "", false),
        ("", "@p mov.b32 r,1; ", false),
        ("", "@p bra target; mov.b32 r,1; ", false),
    ] {
        let text = format!("{prefix}{before}{load}{after}{end}");
        let p = Program::parse(SIGNATURE, &text).unwrap();
        let r = p.names.iter().position(|&n| n == "r").unwrap();
        let origin = p.statements.iter().position(|s| s.op == "ld.global.nc.u32").unwrap();
        assert_eq!(p.immutable_guard(r, origin, p.labels["target"], 0), allowed, "{text}");
    }
}
