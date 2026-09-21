"""Minimal but well-formed PE32/PE32+ images for the Windows release-gate tests.

The images carry one ``.rdata`` section holding a real import directory (with
lookup/address thunks and hint-name entries) and, optionally, a delay-load
directory, so ``scripts/pe_imports.py`` is exercised on the same structures a
linker emits.  Nothing here is executable; the entry point is zero.

Fixed layout, useful for tests that corrupt specific fields:

    0x000  DOS header (e_lfanew = 0x40 at offset 0x3C)
    0x040  "PE\\0\\0" + COFF header
    0x058  optional header (240 bytes PE32+, 224 bytes PE32)
    +40    one section header (.rdata, RVA 0x1000, raw offset 0x200)
    0x200  section data: import descriptors first (name RVA at +12 of each)
"""

from __future__ import annotations

import struct
from typing import Sequence


MACHINE_AMD64 = 0x8664
MACHINE_ARM64 = 0xAA64
MACHINE_I386 = 0x014C
SECTION_RVA = 0x1000
SECTION_RAW_OFFSET = 0x200
IMPORT_DESCRIPTOR_SIZE = 20
DELAY_DESCRIPTOR_SIZE = 32
FIRST_IMPORT_NAME_RVA_FIELD = SECTION_RAW_OFFSET + 12  # file offset of descriptor[0].Name


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def build_pe_image(
    *,
    imports: Sequence[str] = (),
    delay_imports: Sequence[str] = (),
    machine: int = MACHINE_AMD64,
    pe32_plus: bool = True,
    dll: bool = True,
    subsystem: int = 3,
    delay_rva_based: bool = True,
) -> bytes:
    """Return a PE image importing ``imports`` and delay-loading ``delay_imports``."""

    thunk_size = 8 if pe32_plus else 4
    thunk_format = "<Q" if pe32_plus else "<I"
    # A VA-based delay descriptor stores ImageBase + RVA in 32 bits, so that
    # legacy layout is only representable with a low image base.
    image_base = 0x180000000 if pe32_plus and delay_rva_based else 0x10000000

    # ---- section payload, laid out in one pass with RVA bookkeeping ----
    payload = bytearray()
    descriptors_offset = 0
    payload.extend(b"\0" * (IMPORT_DESCRIPTOR_SIZE * (len(imports) + 1)))
    delay_offset = len(payload)
    if delay_imports:
        payload.extend(b"\0" * (DELAY_DESCRIPTOR_SIZE * (len(delay_imports) + 1)))

    def rva(offset: int) -> int:
        return SECTION_RVA + offset

    def append(data: bytes, alignment: int = 1) -> int:
        while len(payload) % alignment:
            payload.append(0)
        offset = len(payload)
        payload.extend(data)
        return offset

    import_records: list[tuple[int, int, int]] = []  # (ilt, iat, name)
    for name in imports:
        hint_name = append(struct.pack("<H", 1) + b"Function\0", 2)
        ilt = append(struct.pack(thunk_format, rva(hint_name)) + b"\0" * thunk_size, thunk_size)
        iat = append(struct.pack(thunk_format, rva(hint_name)) + b"\0" * thunk_size, thunk_size)
        name_offset = append(name.encode("ascii") + b"\0")
        import_records.append((ilt, iat, name_offset))
    delay_records: list[tuple[int, int, int, int]] = []  # (name, handle, iat, int)
    for name in delay_imports:
        hint_name = append(struct.pack("<H", 1) + b"DelayFunction\0", 2)
        handle = append(b"\0" * thunk_size, thunk_size)
        int_table = append(struct.pack(thunk_format, rva(hint_name)) + b"\0" * thunk_size, thunk_size)
        iat = append(struct.pack(thunk_format, 0) + b"\0" * thunk_size, thunk_size)
        name_offset = append(name.encode("ascii") + b"\0")
        delay_records.append((name_offset, handle, iat, int_table))

    for index, (ilt, iat, name_offset) in enumerate(import_records):
        struct.pack_into(
            "<IIIII",
            payload,
            descriptors_offset + index * IMPORT_DESCRIPTOR_SIZE,
            rva(ilt),
            0,
            0,
            rva(name_offset),
            rva(iat),
        )
    for index, (name_offset, handle, iat, int_table) in enumerate(delay_records):
        address = rva(name_offset) if delay_rva_based else image_base + rva(name_offset)
        struct.pack_into(
            "<IIIIIIII",
            payload,
            delay_offset + index * DELAY_DESCRIPTOR_SIZE,
            1 if delay_rva_based else 0,
            address,
            rva(handle),
            rva(iat),
            rva(int_table),
            0,
            0,
            0,
        )
    virtual_size = len(payload)
    raw_size = _align(max(virtual_size, 1), 0x200)
    payload.extend(b"\0" * (raw_size - len(payload)))

    # ---- headers ----
    optional_size = 240 if pe32_plus else 224
    characteristics = 0x0022 | (0x2000 if dll else 0)  # EXECUTABLE_IMAGE | LARGE_ADDRESS_AWARE
    dos = bytearray(64)
    dos[:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 0x40)
    coff = struct.pack("<HHIIIHH", machine, 1, 0, 0, 0, optional_size, characteristics)
    size_of_headers = SECTION_RAW_OFFSET
    size_of_image = SECTION_RVA + _align(raw_size, 0x1000)
    directories = [(0, 0)] * 16
    directories[1] = (rva(descriptors_offset), IMPORT_DESCRIPTOR_SIZE * (len(imports) + 1))
    if delay_imports:
        directories[13] = (rva(delay_offset), DELAY_DESCRIPTOR_SIZE * (len(delay_imports) + 1))
    directory_blob = b"".join(struct.pack("<II", address, size) for address, size in directories)
    if pe32_plus:
        optional = struct.pack(
            "<HBBIIIIIQIIHHHHHHIIIIHHQQQQII",
            0x20B, 14, 0, raw_size, raw_size, 0, 0, SECTION_RVA,
            image_base, 0x1000, 0x200, 6, 0, 0, 0, 6, 0, 0,
            size_of_image, size_of_headers, 0, subsystem, 0x0160,
            0x100000, 0x1000, 0x100000, 0x1000, 0, 16,
        )
    else:
        optional = struct.pack(
            "<HBBIIIIIIIIIHHHHHHIIIIHHIIIIII",
            0x10B, 14, 0, raw_size, raw_size, 0, 0, SECTION_RVA, SECTION_RVA,
            image_base, 0x1000, 0x200, 6, 0, 0, 0, 6, 0, 0,
            size_of_image, size_of_headers, 0, subsystem, 0x0140,
            0x100000, 0x1000, 0x100000, 0x1000, 0, 16,
        )
    optional += directory_blob
    assert len(optional) == optional_size
    section = struct.pack(
        "<8sIIIIIIHHI",
        b".rdata\0\0",
        virtual_size,
        SECTION_RVA,
        raw_size,
        SECTION_RAW_OFFSET,
        0,
        0,
        0,
        0,
        0x40000040,  # INITIALIZED_DATA | READ
    )
    headers = bytes(dos) + b"PE\0\0" + coff + optional + section
    assert len(headers) <= size_of_headers
    headers += b"\0" * (size_of_headers - len(headers))
    return headers + bytes(payload)


__all__ = [
    "DELAY_DESCRIPTOR_SIZE",
    "FIRST_IMPORT_NAME_RVA_FIELD",
    "IMPORT_DESCRIPTOR_SIZE",
    "MACHINE_AMD64",
    "MACHINE_ARM64",
    "MACHINE_I386",
    "SECTION_RAW_OFFSET",
    "SECTION_RVA",
    "build_pe_image",
]
