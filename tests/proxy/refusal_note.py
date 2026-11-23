"""The note in which a cubin made by the layer's nvJitLink names its kernels that trap
(`vf_fatbin::cubin_with_refused`, section `.note.vf.refused`), read and written as the format says,
for the tests: `read(cubin)` gives (kernel, function, reason) for each; `mark(cubin, refused)`
gives the cubin with the note."""
import struct

SECTION = b".note.vf.refused"
NAME = b"volta-forever\0"
TYPE = 1


def sections(cubin):
    shoff, = struct.unpack_from("<Q", cubin, 0x28)
    count, names = struct.unpack_from("<HH", cubin, 0x3C)
    headers = [cubin[shoff + 64 * i:shoff + 64 * (i + 1)] for i in range(count)]
    _, _, _, _, offset, size = struct.unpack_from("<IIQQQQ", headers[names])
    return headers, names, cubin[offset:offset + size]


def read(cubin):
    headers, _, table = sections(cubin)
    found = []
    for header in headers:
        name, kind, _, _, offset, size = struct.unpack_from("<IIQQQQ", header)
        if kind != 7 or table[name:table.index(b"\0", name)] != SECTION:
            continue
        note = cubin[offset:offset + size]
        name_size, description_size, note_type = struct.unpack_from("<III", note)
        start = 12 + -(-name_size // 4) * 4
        if note[12:12 + name_size] != NAME or note_type != TYPE:
            continue
        texts = note[start:start + description_size].rstrip(b"\0").split(b"\0")
        found += [tuple(t.decode() for t in texts[i:i + 3]) for i in range(0, len(texts), 3)]
    return found


def mark(cubin, refused):
    headers, names, table = sections(cubin)
    pad = lambda data, to: data + bytes(-len(data) % to)
    description = b"".join(t.encode() + b"\0" for kernel in refused for t in kernel)
    out = pad(bytes(cubin), 4)
    note = len(out)
    out += struct.pack("<III", len(NAME), len(description), TYPE) + pad(NAME, 4) + pad(description, 4)
    note_size = len(out) - note
    new_names = len(out)
    out += table + SECTION + b"\0"
    out = pad(out, 8)
    shoff = len(out)
    for index, header in enumerate(headers):
        if index == names:
            header = bytearray(header)
            struct.pack_into("<QQ", header, 24, new_names, len(table) + len(SECTION) + 1)
        out += bytes(header)
    out += struct.pack("<IIQQQQIIQQ", len(table), 7, 0, 0, note, note_size, 0, 0, 4, 0)
    out = bytearray(out)
    struct.pack_into("<Q", out, 0x28, shoff)
    struct.pack_into("<H", out, 0x3C, len(headers) + 1)
    return bytes(out)
