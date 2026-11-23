# vf-tables

The instruction sets of NVIDIA's sm_70 (Volta) and sm_75 (Turing) architectures as Rust tables:
the fields and classes of the instructions, what each instruction reads and writes, the rules that
turn an instruction with uniform operands into an ordinary one, the templates of the instructions
that the translator builds, and the distances that native code keeps between a writer and a
reader; and the sm_80 (Ampere) instruction forms that the translator accepts (`src/ampere.rs`).
The code that reads them is in `crates/vf-sass`.

## Where the tables come from

Every file of `src/` is written by `reference/generate.py` (then formatted by
`cargo fmt -p vf-tables`) from:

- the instruction descriptions in the `data/` and `data12/` directories of the
  [denvdis](https://github.com/redplait/denvdis) project (`sm70_*.txt`, `sm75_*.txt` and `sm80_*.txt`), which that
  project extracted from NVIDIA's disassembler; the tables committed here were generated from its
  commit `3769cb8a628f`;
- `reference/ampere-forms.txt`, the instruction classes and operands whose encoding `reference/ampere_forms.py` checks against both the Turing and Ampere descriptions;
- `reference/rules-sm75.tsv` and `reference/distances-sm70.tsv`, this project's own tables, made by
  the scripts of `reference/` from machine code that NVIDIA's compilers produced.

The descriptions themselves are not in this repository: `scripts/fetch-all.sh src` fetches them.

## Licence

The denvdis project is distributed under the GNU Lesser General Public License, version 2.1. The
tables of this crate derive from its descriptions and are distributed under the same licence
(`LICENSE` in this directory; SPDX `LGPL-2.1-only`). The rest of the workspace is distributed under
the Apache License 2.0.

The layer's binaries (`libcuda.so.1`, `libnvJitLink.so.13`, `vf-translate`) contain these
tables. The complete source of the binaries is this repository, so anyone can change the tables
and build the binaries again (`scripts/build.sh`); `NOTICE` says so with each binary distribution.
