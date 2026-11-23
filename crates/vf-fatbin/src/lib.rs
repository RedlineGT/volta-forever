//! Fat binaries: the container in which a CUDA program carries its GPU code for several architectures.
//!
//! Layout (little endian), as written by NVIDIA's `fatbinary` 12.9 and 13.5 and read by the driver:
//!
//! ```text
//! container header, 16 bytes: u32 magic 0xBA55ED50, u16 version 1, u16 header size 16, u64 size of all entries
//! entry header, `header_size` bytes (0x40 to 0x80):
//!   0x00 u16 kind (1 PTX, 2 ELF, 8 LTO-IR)   0x02 u16 0x0101          0x04 u32 header_size
//!   0x08 u64 stored payload size (padded to 8 bytes)
//!   0x10 u32 compressed size (0: stored as is)                          0x14 u32 offset of the option string in the header
//!   0x18 u16 minor, 0x1a u16 major: PTX ISA version of a PTX entry, cubin ABI version of an ELF entry
//!   0x1c u32 SM architecture, decimal (70, 75, 100)                     0x20 u32, u32 object name offset and length
//!   0x28 u64 flags                                                     0x30 u64 0
//!   0x38 u64 uncompressed size (0: stored as is)
//! payload, padded with zero bytes to a multiple of 8
//! ```
//!
//! A program hands the driver either this container or a 24-byte wrapper that points to it
//! (`u32 magic 0x466243B1, u32 version, pointer to the container, pointer`).

use std::borrow::Cow;
use std::fmt;

pub const MAGIC: u32 = 0xBA55_ED50;
pub const WRAPPER_MAGIC: u32 = 0x4662_43B1;
pub const HEADER_SIZE: usize = 16;
const ENTRY_MIN_HEADER: usize = 0x40;

pub const KIND_PTX: u16 = 1;
pub const KIND_ELF: u16 = 2;
pub const KIND_LTO_IR: u16 = 8;

pub const FLAG_DEFLATE: u64 = 0x1000;
pub const FLAG_LZ4: u64 = 0x2000;
pub const FLAG_ZSTD: u64 = 0x8000;
const FLAGS_COMPRESSION: u64 = FLAG_DEFLATE | FLAG_LZ4 | FLAG_ZSTD;
/// `sm_90a`-style variant of the architecture.
pub const FLAG_VARIANT_A: u64 = 0x10_0000;
/// `sm_100f`-style (family) variant of the architecture.
pub const FLAG_VARIANT_F: u64 = 0x20_0000;

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    /// The bytes end before the container or one of its entries does.
    Truncated,
    /// Not a fat binary: wrong magic number or header fields.
    NotFatBinary,
    /// An entry header that cannot be right (offset of the entry in the container).
    BadEntry(usize),
    /// A compressed payload that does not decompress to the announced size.
    Compression(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Truncated => write!(f, "truncated fat binary"),
            Error::NotFatBinary => write!(f, "not a fat binary"),
            Error::BadEntry(offset) => write!(f, "bad fat binary entry at offset {offset}"),
            Error::Compression(what) => write!(f, "cannot decompress entry: {what}"),
        }
    }
}

impl std::error::Error for Error {}

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}
fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}
fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

/// One entry of a container: its header and its payload as stored.
#[derive(Clone, Copy)]
pub struct Entry<'a> {
    pub header: &'a [u8],
    pub stored: &'a [u8],
}

impl<'a> Entry<'a> {
    pub fn kind(&self) -> u16 {
        u16_at(self.header, 0x00)
    }
    /// (major, minor): PTX ISA version of a PTX entry, cubin ABI version of an ELF entry.
    pub fn content_version(&self) -> (u16, u16) {
        (u16_at(self.header, 0x1a), u16_at(self.header, 0x18))
    }
    pub fn sm(&self) -> u32 {
        u32_at(self.header, 0x1c)
    }
    pub fn flags(&self) -> u64 {
        u64_at(self.header, 0x28)
    }
    pub fn is_compressed(&self) -> bool {
        self.flags() & FLAGS_COMPRESSION != 0
    }
    /// `sm_75`, `sm_90a`, `sm_100f`.
    pub fn arch_name(&self) -> String {
        let suffix = match self.flags() & (FLAG_VARIANT_A | FLAG_VARIANT_F) {
            FLAG_VARIANT_A => "a",
            FLAG_VARIANT_F => "f",
            _ => "",
        };
        format!("sm_{}{suffix}", self.sm())
    }

    /// The payload as the compiler produced it: decompressed, and for PTX without the zero padding.
    pub fn payload(&self) -> Result<Cow<'a, [u8]>, Error> {
        let flags = self.flags();
        if flags & FLAGS_COMPRESSION == 0 {
            let stored = self.stored;
            if self.kind() == KIND_PTX {
                let end = stored.iter().position(|&b| b == 0).unwrap_or(stored.len());
                return Ok(Cow::Borrowed(&stored[..end]));
            }
            return Ok(Cow::Borrowed(stored));
        }
        let compressed_size = u32_at(self.header, 0x10) as usize;
        let size = u64_at(self.header, 0x38);
        let compressed = self
            .stored
            .get(..compressed_size)
            .ok_or(Error::Compression("compressed size exceeds the entry"))?;
        // The output is allocated at the announced size before anything is decompressed, and a
        // failed allocation ends the process (it is not a panic). A size that no entry has, or
        // more than LZ4 can make of the bytes there are (at most 255 for each), is not believed.
        // (The largest entry of PyTorch 2.14 and the CUDA 13.5 libraries: 136 MB.)
        if size > CUBIN_MAX || (flags & FLAG_ZSTD == 0 && size > 255 * compressed.len() as u64) {
            return Err(Error::Compression("announced size too large"));
        }
        let size = size as usize;
        let mut data = Vec::new();
        data.try_reserve_exact(size)
            .map_err(|_| Error::Compression("no memory for the announced size"))?;
        if flags & FLAG_ZSTD != 0 {
            use std::io::Read;
            let mut decoder = ruzstd::decoding::StreamingDecoder::new(compressed)
                .map_err(|_| Error::Compression("bad zstd frame"))?;
            // No more than announced: a frame can make far more than it holds, and the vector
            // would grow until an allocation fails. One byte more says that there is more.
            (&mut decoder)
                .take(size as u64)
                .read_to_end(&mut data)
                .map_err(|_| Error::Compression("bad zstd data"))?;
            let more = decoder
                .read(&mut [0])
                .map_err(|_| Error::Compression("bad zstd data"))?;
            if more != 0 {
                return Err(Error::Compression("unexpected uncompressed size"));
            }
        } else if flags & FLAG_LZ4 != 0 {
            data.resize(size, 0);
            let written = lz4_flex::block::decompress_into(compressed, &mut data)
                .map_err(|_| Error::Compression("bad LZ4 block"))?;
            data.truncate(written);
        } else {
            return Err(Error::Compression("deflate is not supported"));
        }
        if data.len() != size {
            return Err(Error::Compression("unexpected uncompressed size"));
        }
        Ok(Cow::Owned(data))
    }
}

/// A parsed container.
pub struct FatBinary<'a> {
    pub entries: Vec<Entry<'a>>,
    /// Bytes of the container, header included.
    pub len: usize,
}

/// Total size of the container that starts with these 16 header bytes.
pub fn container_len(header: &[u8]) -> Result<usize, Error> {
    if header.len() < HEADER_SIZE {
        return Err(Error::Truncated);
    }
    if u32_at(header, 0) != MAGIC || u16_at(header, 4) != 1 || u16_at(header, 6) as usize != HEADER_SIZE {
        return Err(Error::NotFatBinary);
    }
    usize::try_from(u64_at(header, 8))
        .ok()
        .and_then(|n| n.checked_add(HEADER_SIZE))
        .ok_or(Error::NotFatBinary)
}

/// Parse a container; `bytes` may extend past its end.
pub fn parse(bytes: &[u8]) -> Result<FatBinary<'_>, Error> {
    let len = container_len(bytes)?;
    let bytes = bytes.get(..len).ok_or(Error::Truncated)?;
    let mut entries = Vec::new();
    let mut offset = HEADER_SIZE;
    while offset < len {
        if len - offset < ENTRY_MIN_HEADER {
            return Err(Error::BadEntry(offset));
        }
        let header_size = u32_at(bytes, offset + 4) as usize;
        let stored_size = usize::try_from(u64_at(bytes, offset + 8)).map_err(|_| Error::BadEntry(offset))?;
        let end = offset.checked_add(header_size).and_then(|n| n.checked_add(stored_size));
        match end {
            Some(end) if header_size >= ENTRY_MIN_HEADER && end <= len => {
                entries.push(Entry {
                    header: &bytes[offset..offset + header_size],
                    stored: &bytes[offset + header_size..end],
                });
                offset = end;
            }
            _ => return Err(Error::BadEntry(offset)),
        }
    }
    Ok(FatBinary { entries, len })
}

/// Parse the container at `ptr`, whose length is only known from its own header.
///
/// # Safety
/// `ptr` must point to a complete fat binary that stays readable and unchanged for `'a`.
pub unsafe fn parse_at<'a>(ptr: *const u8) -> Result<FatBinary<'a>, Error> {
    let len = container_len(unsafe { std::slice::from_raw_parts(ptr, HEADER_SIZE) })?;
    parse(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// An entry of a container being built.
pub struct NewEntry {
    header: Vec<u8>,
    payload: Vec<u8>,
}

impl NewEntry {
    /// The same entry, byte for byte.
    pub fn copy_of(entry: &Entry<'_>) -> NewEntry {
        NewEntry {
            header: entry.header.to_vec(),
            payload: entry.stored.to_vec(),
        }
    }

    /// An ordinary uncompressed PTX entry, with the platform flags of its source cubin.
    /// fatbinary's PTX header includes a pointer to the empty compiler-options string.
    pub fn from_ptx(text: Vec<u8>, sm: u32, version: (u16, u16), source: &Entry<'_>) -> NewEntry {
        let mut header = [0u8; 80];
        header[..2].copy_from_slice(&KIND_PTX.to_le_bytes());
        header[2..4].copy_from_slice(&source.header[2..4]);
        header[4..8].copy_from_slice(&80u32.to_le_bytes());
        header[0x14..0x18].copy_from_slice(&64u32.to_le_bytes());
        header[0x1c..0x20].copy_from_slice(&source.sm().to_le_bytes());
        header[0x28..0x30].copy_from_slice(&source.flags().to_le_bytes());
        header[0x40..0x44].copy_from_slice(&72u32.to_le_bytes());
        Self::replace(
            &Entry {
                header: &header,
                stored: &[],
            },
            text,
            sm,
            version,
        )
    }

    /// `entry` with a new uncompressed payload, for architecture `sm`, with content version
    /// `(major, minor)`. Architecture-specific and family flags only apply to their original SM.
    pub fn replace(entry: &Entry<'_>, mut payload: Vec<u8>, sm: u32, version: (u16, u16)) -> NewEntry {
        let mut header = entry.header.to_vec();
        if entry.kind() == KIND_PTX {
            payload.push(0);
        }
        payload.resize(payload.len().next_multiple_of(8), 0);
        header[0x08..0x10].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        header[0x10..0x14].copy_from_slice(&0u32.to_le_bytes());
        header[0x18..0x1a].copy_from_slice(&version.1.to_le_bytes());
        header[0x1a..0x1c].copy_from_slice(&version.0.to_le_bytes());
        header[0x1c..0x20].copy_from_slice(&sm.to_le_bytes());
        let mut flags = entry.flags() & !FLAGS_COMPRESSION;
        if sm != entry.sm() {
            flags &= !(FLAG_VARIANT_A | FLAG_VARIANT_F);
        }
        header[0x28..0x30].copy_from_slice(&flags.to_le_bytes());
        header[0x38..0x40].copy_from_slice(&0u64.to_le_bytes());
        NewEntry { header, payload }
    }
}

/// Serialize a container.
pub fn build(entries: &[NewEntry]) -> Vec<u8> {
    let size: usize = entries.iter().map(|e| e.header.len() + e.payload.len()).sum();
    let mut out = Vec::with_capacity(HEADER_SIZE + size);
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&(HEADER_SIZE as u16).to_le_bytes());
    out.extend_from_slice(&(size as u64).to_le_bytes());
    for entry in entries {
        out.extend_from_slice(&entry.header);
        out.extend_from_slice(&entry.payload);
    }
    out
}

/// Offsets of the containers found anywhere in `bytes` (a shared library, an object file, a raw dump).
/// Libraries keep some containers outside `.nv_fatbin`, so the whole file is searched.
pub fn find_containers(bytes: &[u8]) -> Vec<usize> {
    let magic = MAGIC.to_le_bytes();
    let mut found = Vec::new();
    let mut offset = 0;
    while let Some(pos) = bytes[offset..].windows(4).position(|w| w == magic) {
        let at = offset + pos;
        match parse(&bytes[at..]) {
            Ok(fat) if !fat.entries.is_empty() => {
                found.push(at);
                offset = at + fat.len;
            }
            _ => offset = at + 4,
        }
    }
    found
}

/// True when `bytes` start like an ELF file (a cubin, when it comes from a CUDA program).
pub fn is_elf(bytes: &[u8]) -> bool {
    bytes.starts_with(b"\x7fELF")
}

/// SM architecture a cubin was built for. CUDA 12 cubins (ABI 7) keep it in byte 0 of `e_flags`, CUDA 13
/// cubins (ABI 8) in byte 1.
pub fn cubin_sm(bytes: &[u8]) -> Option<u32> {
    if !is_elf(bytes) || bytes.len() < 0x34 {
        return None;
    }
    let flags = u32_at(bytes, 0x30);
    Some(if bytes[8] >= 8 {
        (flags >> 8) & 0xff
    } else {
        flags & 0xff
    })
}

/// No cubin is that long (the largest of PyTorch is 22 MB): beyond it, a header is not believed.
const CUBIN_MAX: u64 = 1 << 31;

/// How many bytes from its start hold the header of a cubin and its tables of segments and sections,
/// as its 64-byte header says. `None` when `header` is not that of a 64-bit little-endian ELF file
/// with tables of the usual shape.
pub fn cubin_tables_len(header: &[u8]) -> Option<usize> {
    if !is_elf(header) || header.len() < 0x40 || header[4] != 2 || header[5] != 1 {
        return None;
    }
    let (segments, sections) = (u64_at(header, 0x20), u64_at(header, 0x28));
    let (segment_size, segment_count) = (u16_at(header, 0x36) as u64, u16_at(header, 0x38) as u64);
    let (section_size, section_count) = (u16_at(header, 0x3a) as u64, u16_at(header, 0x3c) as u64);
    if (section_count != 0 && section_size != 64) || (segment_count != 0 && segment_size != 56) {
        return None;
    }
    let end = 0x40
        .max(segments.checked_add(56 * segment_count)?)
        .max(sections.checked_add(64 * section_count)?);
    (end <= CUBIN_MAX).then_some(end as usize)
}

/// The length of a cubin: where the last of its tables and of the contents of its sections ends.
/// `bytes` must hold its header and tables (`cubin_tables_len`), and may hold no more: `None` when
/// they do not, or when a section is said to reach beyond what a cubin can be.
pub fn cubin_len(bytes: &[u8]) -> Option<usize> {
    let tables = cubin_tables_len(bytes)?;
    if bytes.len() < tables {
        return None;
    }
    let (sections, count) = (u64_at(bytes, 0x28) as usize, u16_at(bytes, 0x3c) as usize);
    let mut end = tables as u64;
    for section in 0..count {
        let at = sections + 64 * section;
        // A section without contents in the file: none, or only room in memory.
        if matches!(u32_at(bytes, at + 4), 0 | 8) {
            continue;
        }
        end = end.max(u64_at(bytes, at + 24).checked_add(u64_at(bytes, at + 32))?);
    }
    (end <= CUBIN_MAX).then_some(end as usize)
}

/// Section headers of a 64-bit little-endian CUDA ELF, borrowed from the input.
fn cubin_sections(bytes: &[u8]) -> Option<&[[u8; 64]]> {
    if bytes.len() < cubin_tables_len(bytes)? || u16_at(bytes, 18) != 190 {
        return None;
    }
    let start = u64_at(bytes, 40) as usize;
    let sections = bytes.get(start..start + 64 * usize::from(u16_at(bytes, 60)))?;
    Some(sections.as_chunks::<64>().0)
}

fn section_data<'a>(bytes: &'a [u8], section: &[u8; 64]) -> Option<&'a [u8]> {
    let start = usize::try_from(u64_at(section, 24)).ok()?;
    let size = usize::try_from(u64_at(section, 32)).ok()?;
    bytes.get(start..start.checked_add(size)?)
}

fn elf_string(table: &[u8], offset: u32) -> Option<&str> {
    let name = table.get(offset as usize..)?;
    std::str::from_utf8(&name[..name.iter().position(|&b| b == 0)?]).ok()
}

/// Source retained by ptxas with line information. Treat a linked or relocatable object with
/// additional externally visible symbols conservatively: kernel names alone cannot check its ABI.
/// Functions the object calls without defining them (`cubin_imports`) add nothing to its interface;
/// the caller (`embedded_ptx` in vf-adapt) checks that the source declares each of them.
pub fn cubin_debug_ptx(bytes: &[u8]) -> Option<&[u8]> {
    let sections = cubin_sections(bytes)?;
    if u16_at(bytes, 16) != 2 {
        return None;
    }
    let names = sections.get(usize::from(u16_at(bytes, 62)))?;
    if u32_at(names, 4) != 3 {
        return None;
    }
    let names = section_data(bytes, names)?;
    let mut source = None;
    for section in sections {
        if elf_string(names, u32_at(section, 0))? == ".nv_debug_ptx_txt" {
            if u32_at(section, 4) != 1 || source.is_some() {
                return None;
            }
            source = Some(section_data(bytes, section)?);
        }
    }
    let source = source?;
    let symbols = cubin_symbols(bytes, sections)?;
    for (index, symbol) in symbols.table.iter().enumerate() {
        if symbol[4] >> 4 != 0
            && !is_kernel(symbol)
            && !symbols.assembler_helper(index, sections)
            && !symbols.import(index)
        {
            return None;
        }
    }
    Some(source)
}

/// Functions a little-endian 64-bit CUDA ELF calls but does not define, which the driver resolves
/// when it loads the module: `__assertfail` and `vprintf` for device assertions and `printf`. In an
/// executable from ptxas only the functions it knows the driver provides can be undefined: it
/// refuses any other (`Unresolved extern function`); `-c` and `-ewp` objects are not executables.
pub fn cubin_imports(bytes: &[u8]) -> Option<Vec<&str>> {
    let symbols = cubin_symbols(bytes, cubin_sections(bytes)?)?;
    let mut imports = Vec::new();
    for index in (0..symbols.table.len()).filter(|&i| symbols.import(i)) {
        imports.push(elf_string(symbols.names, u32_at(&symbols.table[index], 0))?);
    }
    Some(imports)
}

fn is_kernel(symbol: &[u8; 24]) -> bool {
    symbol[4] & 0xf == 2 && symbol[5] & 0x10 != 0 && u16_at(symbol, 6) != 0
}

struct CubinSymbols<'a> {
    table: &'a [[u8; 24]],
    names: &'a [u8],
    first_global: usize,
}

impl CubinSymbols<'_> {
    /// An undefined global function with default visibility and no value: a call that the module
    /// leaves to the loader (PTX `.extern .func`).
    fn import(&self, index: usize) -> bool {
        let symbol = &self.table[index];
        index >= self.first_global
            && symbol[4] == 0x12
            && symbol[5] == 0
            && u16_at(symbol, 6) == 0
            && u64_at(symbol, 8) == 0
            && u64_at(symbol, 16) == 0
    }

    /// CUDA 12 ptxas marks its out-of-line routines (for instance, 64-bit integer division)
    /// as weak functions in the local portion of .symtab. Their machine code is inside the
    /// owning kernel. CUDA 13 emits local functions for the same PTX operations.
    fn assembler_helper(&self, index: usize, sections: &[[u8; 64]]) -> bool {
        let Some(symbol) = self.table.get(index) else {
            return false;
        };
        if index >= self.first_global || symbol[4] != 0x22 || symbol[5] != 0 {
            return false;
        }
        let Some((ordinal, routine)) = elf_string(self.names, u32_at(symbol, 0))
            .and_then(|s| s.strip_prefix("$__internal_"))
            .and_then(|s| s.split_once("_$__cuda_"))
        else {
            return false;
        };
        if ordinal.is_empty() || !ordinal.bytes().all(|b| b.is_ascii_digit()) || routine.is_empty() {
            return false;
        }
        let section_index = u16_at(symbol, 6);
        let Some(section) = sections.get(usize::from(section_index)) else {
            return false;
        };
        // In CUDA ELF, the low 24 bits of a text section's sh_info name the owning function.
        let owner_idx = (u32_at(section, 44) & 0x00ff_ffff) as usize;
        let Some(owner) = self.table.get(owner_idx) else {
            return false;
        };
        if u32_at(section, 4) != 1
            || u64_at(section, 8) & 4 == 0
            || !is_kernel(owner)
            || u16_at(owner, 6) != section_index
        {
            return false;
        }
        let start = u64_at(symbol, 8);
        let size = u64_at(symbol, 16);
        let owner_start = u64_at(owner, 8);
        let owner_size = u64_at(owner, 16);
        size != 0
            && start
                .checked_sub(owner_start)
                .and_then(|offset| offset.checked_add(size))
                .is_some_and(|end| end <= owner_size)
    }
}

fn cubin_symbols<'a>(bytes: &'a [u8], sections: &[[u8; 64]]) -> Option<CubinSymbols<'a>> {
    let symbols = sections.iter().find(|s| u32_at(*s, 4) == 2)?;
    if u64_at(symbols, 56) != 24 {
        return None;
    }
    let strings = sections.get(u32_at(symbols, 40) as usize)?;
    if u32_at(strings, 4) != 3 {
        return None;
    }
    let first_global = u32_at(symbols, 44) as usize;
    let (table, trailing) = section_data(bytes, symbols)?.as_chunks::<24>();
    (trailing.is_empty() && first_global <= table.len()).then_some(CubinSymbols {
        table,
        names: section_data(bytes, strings)?,
        first_global,
    })
}

/// Whether ptxas placed this weak routine in the local part of the kernel symbol table.
pub fn cubin_has_assembler_helper(bytes: &[u8], index: usize) -> bool {
    let Some(sections) = cubin_sections(bytes) else {
        return false;
    };
    cubin_symbols(bytes, sections).is_some_and(|symbols| symbols.assembler_helper(index, sections))
}

/// A kernel of a cubin that cannot run on the GPU it was made for: it is, or calls, `function`,
/// which traps, for `reason` (`cubin_with_refused`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefusedKernel {
    pub kernel: String,
    pub function: String,
    pub reason: String,
}

/// The section in which a cubin names the kernels that trap, and the name and type of its note.
pub const REFUSED_SECTION: &str = ".note.vf.refused";
const REFUSED_NOTE_NAME: &[u8] = b"volta-forever\0";
const REFUSED_NOTE_TYPE: u32 = 1;

/// `cubin` with a note section (`REFUSED_SECTION`) that names its kernels in `refused`, with
/// their functions and reasons: a cubin made from code in which they trap, so that whoever loads it
/// can refuse them (the layer, at the lookup of each). The cubin is kept as it is; the note, a new
/// table of section names and a new table of sections that has the note in addition are put after
/// it, and its header names the new tables. `None` when `cubin` is not a complete 64-bit cubin
/// with a table of section names, or already has the note.
pub fn cubin_with_refused(cubin: &[u8], refused: &[RefusedKernel]) -> Option<Vec<u8>> {
    let sections = cubin_sections(cubin)?;
    if cubin_len(cubin)? > cubin.len() || !cubin_refused(cubin).is_empty() {
        return None;
    }
    let names_index = usize::from(u16_at(cubin, 62));
    let names = section_data(cubin, sections.get(names_index)?)?;
    if u32_at(sections.get(names_index)?, 4) != 3 || sections.len() >= 0xff00 {
        return None;
    }
    let mut description = Vec::new();
    for kernel in refused {
        for text in [&kernel.kernel, &kernel.function, &kernel.reason] {
            if text.as_bytes().contains(&0) {
                return None;
            }
            description.extend_from_slice(text.as_bytes());
            description.push(0);
        }
    }
    let pad = |out: &mut Vec<u8>, to: usize| out.resize(out.len().next_multiple_of(to), 0);
    let mut out = cubin.to_vec();
    pad(&mut out, 4);
    let note = out.len();
    out.extend_from_slice(&(REFUSED_NOTE_NAME.len() as u32).to_le_bytes());
    out.extend_from_slice(&u32::try_from(description.len()).ok()?.to_le_bytes());
    out.extend_from_slice(&REFUSED_NOTE_TYPE.to_le_bytes());
    out.extend_from_slice(REFUSED_NOTE_NAME);
    pad(&mut out, 4);
    out.extend_from_slice(&description);
    pad(&mut out, 4);
    let note_size = out.len() - note;
    let new_names = out.len();
    out.extend_from_slice(names);
    let name = u32::try_from(names.len()).ok()?;
    out.extend_from_slice(REFUSED_SECTION.as_bytes());
    out.push(0);
    let new_names_size = out.len() - new_names;
    pad(&mut out, 8);
    let table = out.len();
    for (index, section) in sections.iter().enumerate() {
        let mut header = *section;
        if index == names_index {
            header[24..32].copy_from_slice(&(new_names as u64).to_le_bytes());
            header[32..40].copy_from_slice(&(new_names_size as u64).to_le_bytes());
        }
        out.extend_from_slice(&header);
    }
    let mut header = [0u8; 64];
    header[0..4].copy_from_slice(&name.to_le_bytes());
    header[4..8].copy_from_slice(&7u32.to_le_bytes()); // SHT_NOTE
    header[24..32].copy_from_slice(&(note as u64).to_le_bytes());
    header[32..40].copy_from_slice(&(note_size as u64).to_le_bytes());
    header[48..56].copy_from_slice(&4u64.to_le_bytes());
    out.extend_from_slice(&header);
    out[0x28..0x30].copy_from_slice(&(table as u64).to_le_bytes());
    out[0x3c..0x3e].copy_from_slice(&(sections.len() as u16 + 1).to_le_bytes());
    Some(out)
}

/// The kernels that the note of `cubin_with_refused` names in `cubin`; none when it has no such
/// note, or when the cubin or the note cannot be read whole.
pub fn cubin_refused(cubin: &[u8]) -> Vec<RefusedKernel> {
    let read = || -> Option<Vec<RefusedKernel>> {
        let sections = cubin_sections(cubin)?;
        let names = section_data(cubin, sections.get(usize::from(u16_at(cubin, 62)))?)?;
        let mut found = Vec::new();
        for section in sections.iter().filter(|s| u32_at(*s, 4) == 7) {
            if elf_string(names, u32_at(section, 0)) != Some(REFUSED_SECTION) {
                continue;
            }
            let note = section_data(cubin, section)?;
            let (name_size, size) = (u32_at(note.get(..12)?, 0) as usize, u32_at(note, 4) as usize);
            let start = 12usize.checked_add(name_size.next_multiple_of(4))?;
            if note.get(12..12usize.checked_add(name_size)?)? != REFUSED_NOTE_NAME
                || u32_at(note, 8) != REFUSED_NOTE_TYPE
            {
                continue;
            }
            let description = note.get(start..start.checked_add(size)?)?;
            let texts: Vec<&str> = description
                .strip_suffix(&[0])?
                .split(|&b| b == 0)
                .map(std::str::from_utf8)
                .collect::<Result<_, _>>()
                .ok()?;
            for kernel in texts.chunks(3) {
                let [kernel, function, reason] = kernel else {
                    return None;
                };
                found.push(RefusedKernel {
                    kernel: kernel.to_string(),
                    function: function.to_string(),
                    reason: reason.to_string(),
                });
            }
        }
        Some(found)
    };
    read().unwrap_or_default()
}

/// Entry points retained by the device linker in a little-endian 64-bit CUDA ELF.
pub fn cubin_entries(bytes: &[u8]) -> Option<Vec<&str>> {
    let symbols = cubin_symbols(bytes, cubin_sections(bytes)?)?;
    let mut entries = Vec::new();
    for symbol in symbols.table.iter().filter(|s| is_kernel(s)) {
        entries.push(elf_string(symbols.names, u32_at(symbol, 0))?);
    }
    Some(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ELF75_PTX75_ZSTD: &[u8] = include_bytes!("../tests/data/elf75-ptx75-zstd.fatbin");
    const ELF70_ELF75_PTX70_LZ4: &[u8] = include_bytes!("../tests/data/elf70-elf75-ptx70-lz4.fatbin");
    const ALL_ZSTD: &[u8] = include_bytes!("../tests/data/elf75-ptx75-allzstd.fatbin");

    fn summary(bytes: &[u8]) -> Vec<(u16, u32, bool, (u16, u16))> {
        parse(bytes)
            .unwrap()
            .entries
            .iter()
            .map(|e| (e.kind(), e.sm(), e.is_compressed(), e.content_version()))
            .collect()
    }

    #[test]
    fn parses_fatbinary_13_5_output() {
        assert_eq!(
            summary(ELF75_PTX75_ZSTD),
            [(KIND_ELF, 75, false, (1, 8)), (KIND_PTX, 75, true, (9, 5))]
        );
        assert_eq!(
            summary(ALL_ZSTD),
            [(KIND_ELF, 75, true, (1, 8)), (KIND_PTX, 75, true, (9, 5))]
        );
    }

    #[test]
    fn parses_fatbinary_12_9_output() {
        assert_eq!(
            summary(ELF70_ELF75_PTX70_LZ4),
            [
                (KIND_ELF, 70, false, (1, 7)),
                (KIND_ELF, 75, false, (1, 7)),
                (KIND_PTX, 70, true, (8, 7))
            ]
        );
    }

    #[test]
    fn decompresses_zstd_and_lz4() {
        let fat = parse(ELF75_PTX75_ZSTD).unwrap();
        let ptx = fat.entries[1].payload().unwrap();
        assert!(ptx.windows(13).any(|w| w == b".target sm_75"), "zstd PTX");
        let fat = parse(ELF70_ELF75_PTX70_LZ4).unwrap();
        let ptx = fat.entries[2].payload().unwrap();
        assert!(ptx.windows(13).any(|w| w == b".target sm_70"), "LZ4 PTX");
        let fat = parse(ALL_ZSTD).unwrap();
        let elf = fat.entries[0].payload().unwrap();
        assert_eq!(cubin_sm(&elf), Some(75));
        assert_eq!(elf.len(), 18592);
    }

    /// The header of `entry` with the uncompressed size it announces set to `size`.
    fn announcing(entry: &Entry<'_>, size: u64) -> Vec<u8> {
        let mut header = entry.header.to_vec();
        header[0x38..0x40].copy_from_slice(&size.to_le_bytes());
        header
    }

    #[test]
    fn an_announced_size_beyond_any_entry_is_refused_before_it_is_allocated() {
        // (Until 2026-11-22 the allocation of 1 TB failed, and that ended the process.)
        let zstd = parse(ALL_ZSTD).unwrap();
        let lz4 = parse(ELF70_ELF75_PTX70_LZ4).unwrap();
        for entry in [&zstd.entries[0], &zstd.entries[1], &lz4.entries[2]] {
            for size in [1 << 40, CUBIN_MAX + 1, u64::MAX >> 1] {
                let header = announcing(entry, size);
                let claimed = Entry {
                    header: &header,
                    stored: entry.stored,
                };
                assert_eq!(
                    claimed.payload().err(),
                    Some(Error::Compression("announced size too large"))
                );
            }
        }
        // LZ4 makes at most 255 bytes of each byte it reads.
        let entry = &lz4.entries[2];
        let most = 255 * u64::from(u32_at(entry.header, 0x10));
        for (size, error) in [
            (most + 1, "announced size too large"),
            (most, "unexpected uncompressed size"),
        ] {
            let header = announcing(entry, size);
            let claimed = Entry {
                header: &header,
                stored: entry.stored,
            };
            assert_eq!(claimed.payload().err(), Some(Error::Compression(error)));
        }
    }

    #[test]
    fn a_zstd_frame_is_read_no_further_than_the_announced_size() {
        // A frame can make far more than it holds. Here, with a window of 1 kB (the decoder hands
        // out what is beyond it): two blocks of 1 kB of the same value, then a block of the
        // reserved kind, announced as 16 bytes. The third block is not decoded. (Until 2026-11-22
        // the frame was decoded to its end whatever the entry announced, into a vector that grew
        // until an allocation was refused: 1 TB from 32 MB of such blocks.)
        let mut frame = vec![0x28, 0xb5, 0x2f, 0xfd, 0x00, 0x00];
        for _ in 0..2 {
            frame.extend_from_slice(&(1024u32 << 3 | 1 << 1).to_le_bytes()[..3]);
            frame.push(0x5a);
        }
        frame.extend_from_slice(&(1u32 << 3 | 3 << 1 | 1).to_le_bytes()[..3]);
        let entry = &parse(ALL_ZSTD).unwrap().entries[1];
        let mut header = announcing(entry, 16);
        header[0x10..0x14].copy_from_slice(&(frame.len() as u32).to_le_bytes());
        let claimed = Entry {
            header: &header,
            stored: &frame,
        };
        assert_eq!(
            claimed.payload().err(),
            Some(Error::Compression("unexpected uncompressed size"))
        );
        // Announced as all it makes: the 2 kB, then the block that cannot be read.
        let header = announcing(&claimed, 2048);
        let claimed = Entry {
            header: &header,
            stored: &frame,
        };
        assert_eq!(claimed.payload().err(), Some(Error::Compression("bad zstd data")));
    }

    #[test]
    fn reads_only_defined_cuda_entry_symbols() {
        let fat = parse(ELF75_PTX75_ZSTD).unwrap();
        let elf = fat.entries[0].payload().unwrap();
        let mut names = cubin_entries(&elf).unwrap();
        names.sort_unstable();
        assert_eq!(
            names,
            ["block_sum", "dbl", "half_dot", "histo", "mathk", "saxpy", "warp_sum"]
        );
        assert!(cubin_entries(&elf[..63]).is_none());
        let mut bad = elf.into_owned();
        bad[18] = 62; // x86-64, not a cubin.
        assert!(cubin_entries(&bad).is_none());
    }

    #[test]
    fn the_length_of_a_cubin_is_in_its_tables() {
        let fat = parse(ELF70_ELF75_PTX70_LZ4).unwrap();
        for entry in &fat.entries[..2] {
            let cubin = entry.payload().unwrap();
            let tables = cubin_tables_len(&cubin[..0x40]).unwrap();
            assert!(tables <= cubin.len());
            // From the tables alone, and from all of it: the same, and no more than there is (an
            // entry is padded to a multiple of 8).
            let length = cubin_len(&cubin[..tables]).unwrap();
            assert_eq!(cubin_len(&cubin), Some(length));
            assert!(
                length <= cubin.len() && cubin.len() - length < 8,
                "{length} of {}",
                cubin.len()
            );
            // Without its tables, or with a section that reaches past everything: not told.
            assert_eq!(cubin_len(&cubin[..tables - 1]), None);
            let mut wild = cubin.to_vec();
            let sections = u64_at(&wild, 0x28) as usize;
            wild[sections + 64 + 32..sections + 64 + 40].copy_from_slice(&u64::MAX.to_le_bytes());
            wild[sections + 64 + 4] = 1;
            assert_eq!(cubin_len(&wild), None);
        }
        assert_eq!(cubin_tables_len(b"\x7fELF"), None);
        assert_eq!(cubin_tables_len(&[0; 0x40]), None);
    }

    #[test]
    fn a_cubin_names_the_kernels_that_trap_in_a_note() {
        let fat = parse(ELF70_ELF75_PTX70_LZ4).unwrap();
        let cubin = fat.entries[0].payload().unwrap();
        let cubin = &cubin[..cubin_len(&cubin).unwrap()];
        let refused = [
            RefusedKernel {
                kernel: "warp_sum".into(),
                function: "warp_sum".into(),
                reason: "an instruction".into(),
            },
            RefusedKernel {
                kernel: "add_one".into(),
                function: "helper".into(),
                reason: String::new(),
            },
        ];
        assert_eq!(cubin_refused(cubin), []);
        let marked = cubin_with_refused(cubin, &refused).unwrap();
        assert_eq!(cubin_refused(&marked), refused);
        // The cubin is as it was, with a section more, and a length that its tables tell.
        assert_eq!(cubin_len(&marked), Some(marked.len()));
        assert_eq!(&marked[0x40..cubin.len()], &cubin[0x40..]);
        assert_eq!(cubin_entries(&marked), cubin_entries(cubin));
        assert_eq!(cubin_sm(&marked), Some(70));
        let (before, after) = (cubin_sections(cubin).unwrap(), cubin_sections(&marked).unwrap());
        assert_eq!(after.len(), before.len() + 1);
        let names = usize::from(u16_at(cubin, 62));
        for (index, (old, new)) in before.iter().zip(after).enumerate() {
            let (old, new) = (section_data(cubin, old).unwrap(), section_data(&marked, new).unwrap());
            if index == names {
                assert_eq!(new, [old, b".note.vf.refused\0"].concat());
            } else {
                assert_eq!(old, new);
            }
        }
        // Marked once; nothing of what cannot be read.
        assert_eq!(cubin_with_refused(&marked, &refused), None);
        assert_eq!(cubin_with_refused(&cubin[..cubin.len() - 1], &refused), None);
        let late = RefusedKernel {
            kernel: "a\0b".into(),
            ..refused[0].clone()
        };
        assert_eq!(cubin_with_refused(cubin, &[late]), None);
        for cut in [marked.len() - 64, marked.len() - 70] {
            assert_eq!(cubin_refused(&marked[..cut]), []);
        }
        let mut wrong_name = marked.clone();
        let at = wrong_name.windows(13).position(|w| w == b"volta-forever").unwrap();
        wrong_name[at] = b'V';
        assert_eq!(cubin_refused(&wrong_name), []);
        assert_eq!(cubin_refused(b"\x7fELF"), []);
    }

    #[test]
    fn cubin_sm_reads_both_abis() {
        let fat = parse(ELF70_ELF75_PTX70_LZ4).unwrap();
        assert_eq!(cubin_sm(fat.entries[0].stored), Some(70));
        assert_eq!(cubin_sm(fat.entries[1].stored), Some(75));
        let fat = parse(ELF75_PTX75_ZSTD).unwrap();
        assert_eq!(cubin_sm(fat.entries[0].stored), Some(75));
    }

    #[test]
    fn rebuild_is_identical() {
        for bytes in [ELF75_PTX75_ZSTD, ELF70_ELF75_PTX70_LZ4, ALL_ZSTD] {
            let fat = parse(bytes).unwrap();
            let copies: Vec<NewEntry> = fat.entries.iter().map(NewEntry::copy_of).collect();
            assert_eq!(build(&copies), bytes);
        }
    }

    #[test]
    fn replace_writes_a_plain_entry() {
        let fat = parse(ELF75_PTX75_ZSTD).unwrap();
        let entry = NewEntry::replace(&fat.entries[1], b".version 9.0\n".to_vec(), 70, (9, 0));
        let rebuilt = build(&[NewEntry::copy_of(&fat.entries[0]), entry]);
        let fat = parse(&rebuilt).unwrap();
        let ptx = &fat.entries[1];
        assert_eq!(
            (ptx.kind(), ptx.sm(), ptx.is_compressed(), ptx.content_version()),
            (KIND_PTX, 70, false, (9, 0))
        );
        assert_eq!(&*ptx.payload().unwrap(), b".version 9.0\n");
        assert_eq!(ptx.stored.len() % 8, 0);
    }

    #[test]
    fn retargeting_drops_only_the_source_architecture_variant() {
        let fat = parse(ELF75_PTX75_ZSTD).unwrap();
        let original = &fat.entries[1];
        for variant in [FLAG_VARIANT_A, FLAG_VARIANT_F] {
            let mut header = original.header.to_vec();
            let flags = original.flags() | variant;
            header[0x28..0x30].copy_from_slice(&flags.to_le_bytes());
            let source = Entry {
                header: &header,
                stored: original.stored,
            };
            for (target, expected) in [(source.sm(), flags), (70, flags & !variant)] {
                let bytes = build(&[NewEntry::replace(&source, Vec::new(), target, (9, 0))]);
                assert_eq!(parse(&bytes).unwrap().entries[0].flags(), expected & !FLAGS_COMPRESSION);
            }
        }
    }

    #[test]
    fn rejects_garbage() {
        assert_eq!(parse(b"not a fat binary at all").err(), Some(Error::NotFatBinary));
        assert_eq!(parse(&ELF75_PTX75_ZSTD[..100]).err(), Some(Error::Truncated));
        let mut broken = ELF75_PTX75_ZSTD.to_vec();
        broken[16 + 4] = 0; // entry header size 0x40 -> 0
        assert_eq!(parse(&broken).err(), Some(Error::BadEntry(16)));
    }

    #[test]
    fn finds_containers_inside_a_file() {
        let mut file = vec![0u8; 100];
        file.extend_from_slice(ELF75_PTX75_ZSTD);
        file.extend_from_slice(&[0xaa; 37]);
        file.extend_from_slice(ALL_ZSTD);
        assert_eq!(find_containers(&file), [100, 100 + ELF75_PTX75_ZSTD.len() + 37]);
    }
}
