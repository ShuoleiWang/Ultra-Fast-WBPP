#!/usr/bin/env python3
"""Read the import table of a Windows PE image without external tools.

The release gates need to know which DLLs a ``.dll``/``.pyd``/``.exe`` will
load before it ever runs: a native kernel library that links the dynamic
Visual C++ runtime only works on machines that happen to have the
redistributable, and a frozen engine whose extension modules import a DLL that
is neither part of Windows nor shipped in the tree fails on a clean machine.
``dumpbin`` needs a Visual Studio prompt and ``pefile`` is not a dependency, so
this module walks the on-disk structures directly:

    DOS header -> PE signature -> COFF header -> optional header (PE32/PE32+)
    -> data directories -> import directory / delay-load directory -> DLL names

Every offset is bounds-checked and malformed input raises ``PEFormatError``
instead of an ``IndexError``/``struct.error``.  Only the file-level facts the
gates use are exposed (machine, subsystem, DLL flag, imported DLL names); the
parser deliberately does not follow thunks, resolve function names, or trust
the ``Size`` field of a data directory where the format's own terminator is
authoritative.

    python scripts/pe_imports.py path/to/ufwbpp_native.dll
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import struct
import sys
from typing import Iterable, Sequence


class PEFormatError(ValueError):
    """The bytes are not a well-formed PE image that this parser can read."""


_DOS_MAGIC = b"MZ"
_PE_SIGNATURE = b"PE\0\0"
_OPTIONAL_MAGIC_PE32 = 0x10B
_OPTIONAL_MAGIC_PE32_PLUS = 0x20B
_IMAGE_FILE_DLL = 0x2000
_DIRECTORY_IMPORT = 1
_DIRECTORY_DELAY_IMPORT = 13
_IMPORT_DESCRIPTOR_SIZE = 20
_DELAY_DESCRIPTOR_SIZE = 32
_SECTION_HEADER_SIZE = 40
_MAXIMUM_DESCRIPTORS = 4096
_MAXIMUM_SECTIONS = 96  # the COFF limit
_MAXIMUM_NAME_LENGTH = 260

MACHINE_NAMES = {
    0x014C: "x86",
    0x01C4: "arm",
    0x8664: "x86_64",
    0xAA64: "aarch64",
}
SUBSYSTEM_NAMES = {
    1: "native",
    2: "windows-gui",
    3: "windows-cui",
}

# Visual C++ runtime DLLs that only exist on a machine with the matching
# redistributable installed. Matching is case-insensitive on the file name.
DYNAMIC_CRT_PATTERN = re.compile(
    r"^(?:msvcp|msvcr|vcruntime|concrt|vcomp|vcamp|mfc|mfcm|vccorlib)\d+.*\.dll$",
    re.IGNORECASE,
)
# Universal C runtime API-set forwarders and the UCRT itself. They are Windows
# components on the supported OS baseline (Windows 10+), so they are neither a
# distribution risk nor a sign of static linkage; they are reported separately.
UCRT_PATTERN = re.compile(r"^(?:api-ms-win-crt-.*|ucrtbase(?:d)?)\.dll$", re.IGNORECASE)
# API-set schemas resolved by the loader to system DLLs on Windows 10+.
_API_SET_PREFIXES = ("api-ms-", "ext-ms-")

# DLLs that ship with every supported Windows installation (System32). Names are
# lower-case; the loader resolves them case-insensitively. This is the closed
# allow-list for the worker attestation: anything else must be in the tree.
WINDOWS_SYSTEM_DLLS = frozenset(
    {
        "activeds.dll",
        "advapi32.dll",
        "advpack.dll",
        "apphelp.dll",
        "audioses.dll",
        "authz.dll",
        "avifil32.dll",
        "avrt.dll",
        "bcp47langs.dll",
        "bcp47mrm.dll",
        "bcrypt.dll",
        "bcryptprimitives.dll",
        "bluetoothapis.dll",
        "cabinet.dll",
        "cfgmgr32.dll",
        "clbcatq.dll",
        "combase.dll",
        "comctl32.dll",
        "comdlg32.dll",
        "coremessaging.dll",
        "credui.dll",
        "crypt32.dll",
        "cryptbase.dll",
        "cryptnet.dll",
        "cryptsp.dll",
        "cryptui.dll",
        "d2d1.dll",
        "d3d11.dll",
        "d3d12.dll",
        "d3d9.dll",
        "davclnt.dll",
        "dbgcore.dll",
        "dbgeng.dll",
        "dbghelp.dll",
        "dcomp.dll",
        "ddraw.dll",
        "dhcpcsvc.dll",
        "dhcpcsvc6.dll",
        "dinput8.dll",
        "dnsapi.dll",
        "dsound.dll",
        "dwmapi.dll",
        "dwrite.dll",
        "dxgi.dll",
        "dxva2.dll",
        "esent.dll",
        "faultrep.dll",
        "fltlib.dll",
        "fwpuclnt.dll",
        "gdi32.dll",
        "gdi32full.dll",
        "gdiplus.dll",
        "glu32.dll",
        "hid.dll",
        "httpapi.dll",
        "imagehlp.dll",
        "imm32.dll",
        "iphlpapi.dll",
        "kernel32.dll",
        "kernelbase.dll",
        "ktmw32.dll",
        "logoncli.dll",
        "mf.dll",
        "mfplat.dll",
        "mfreadwrite.dll",
        "mmdevapi.dll",
        "mpr.dll",
        "msacm32.dll",
        "mscms.dll",
        "mscoree.dll",
        "msctf.dll",
        "msi.dll",
        "msimg32.dll",
        "mstask.dll",
        "msvcirt.dll",
        "msvcp_win.dll",
        "msvcrt.dll",
        "msvfw32.dll",
        "mswsock.dll",
        "ncrypt.dll",
        "netapi32.dll",
        "netutils.dll",
        "newdev.dll",
        "normaliz.dll",
        "nsi.dll",
        "ntdll.dll",
        "ntdsapi.dll",
        "ntlanman.dll",
        "ntmarta.dll",
        "ole32.dll",
        "oleacc.dll",
        "oleaut32.dll",
        "olepro32.dll",
        "opengl32.dll",
        "pathcch.dll",
        "pdh.dll",
        "powrprof.dll",
        "prntvpt.dll",
        "profapi.dll",
        "propsys.dll",
        "psapi.dll",
        "qwave.dll",
        "rpcrt4.dll",
        "rsaenh.dll",
        "rstrtmgr.dll",
        "samcli.dll",
        "sechost.dll",
        "secur32.dll",
        "sensapi.dll",
        "setupapi.dll",
        "shcore.dll",
        "shell32.dll",
        "shfolder.dll",
        "shlwapi.dll",
        "srvcli.dll",
        "sspicli.dll",
        "synchronization.dll",
        "taskschd.dll",
        "tdh.dll",
        "textinputframework.dll",
        "twinapi.appcore.dll",
        "twinapi.dll",
        "ucrtbase.dll",
        "uiautomationcore.dll",
        "urlmon.dll",
        "user32.dll",
        "userenv.dll",
        "usp10.dll",
        "uxtheme.dll",
        "version.dll",
        "virtdisk.dll",
        "webservices.dll",
        "wecapi.dll",
        "wer.dll",
        "wevtapi.dll",
        "win32u.dll",
        "windows.storage.dll",
        "windowscodecs.dll",
        "winhttp.dll",
        "wininet.dll",
        "winmm.dll",
        "winnsi.dll",
        "winscard.dll",
        "winspool.drv",
        "winsta.dll",
        "wintrust.dll",
        "winusb.dll",
        "wkscli.dll",
        "wlanapi.dll",
        "wldap32.dll",
        "wldp.dll",
        "wmi.dll",
        "ws2_32.dll",
        "wsock32.dll",
        "wtsapi32.dll",
        "xaudio2_9.dll",
        "xinput1_4.dll",
        "xmllite.dll",
    }
)


@dataclass(frozen=True)
class PEImports:
    """File-level facts about one PE image and the DLL names it imports."""

    format: str
    machine: int
    machineName: str
    subsystem: int
    subsystemName: str
    isDll: bool
    imports: tuple[str, ...]
    delayImports: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["imports"] = list(self.imports)
        payload["delayImports"] = list(self.delayImports)
        return payload

    @property
    def all_imports(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.imports) | set(self.delayImports), key=str.casefold))


def _u16(data: bytes, offset: int, field: str) -> int:
    if offset < 0 or offset + 2 > len(data):
        raise PEFormatError(f"{field} lies outside the file")
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int, field: str) -> int:
    if offset < 0 or offset + 4 > len(data):
        raise PEFormatError(f"{field} lies outside the file")
    return struct.unpack_from("<I", data, offset)[0]


def _u64(data: bytes, offset: int, field: str) -> int:
    if offset < 0 or offset + 8 > len(data):
        raise PEFormatError(f"{field} lies outside the file")
    return struct.unpack_from("<Q", data, offset)[0]


@dataclass(frozen=True)
class _Section:
    virtual_address: int
    virtual_size: int
    raw_pointer: int
    raw_size: int


class _Image:
    def __init__(self, data: bytes) -> None:
        self.data = data
        if len(data) < 64 or data[:2] != _DOS_MAGIC:
            raise PEFormatError("missing MZ DOS header")
        pe_offset = _u32(data, 0x3C, "e_lfanew")
        if pe_offset < 64 or pe_offset + 4 + 20 > len(data):
            raise PEFormatError("e_lfanew does not point at a PE signature")
        if data[pe_offset : pe_offset + 4] != _PE_SIGNATURE:
            raise PEFormatError("PE signature is missing")
        coff = pe_offset + 4
        self.machine = _u16(data, coff, "Machine")
        section_count = _u16(data, coff + 2, "NumberOfSections")
        optional_size = _u16(data, coff + 16, "SizeOfOptionalHeader")
        self.characteristics = _u16(data, coff + 18, "Characteristics")
        if section_count > _MAXIMUM_SECTIONS:
            raise PEFormatError("NumberOfSections is outside the COFF range")
        optional = coff + 20
        magic = _u16(data, optional, "optional header magic")
        if magic == _OPTIONAL_MAGIC_PE32:
            self.format = "PE32"
            image_base_offset, directories_count_offset, minimum_size = 28, 92, 96
        elif magic == _OPTIONAL_MAGIC_PE32_PLUS:
            self.format = "PE32+"
            image_base_offset, directories_count_offset, minimum_size = 24, 108, 112
        else:
            raise PEFormatError(f"unknown optional header magic 0x{magic:x}")
        if optional_size < minimum_size or optional + optional_size > len(data):
            raise PEFormatError("optional header is truncated")
        if self.format == "PE32":
            self.image_base = _u32(data, optional + image_base_offset, "ImageBase")
        else:
            self.image_base = _u64(data, optional + image_base_offset, "ImageBase")
        self.subsystem = _u16(data, optional + 68, "Subsystem")
        self.size_of_headers = _u32(data, optional + 60, "SizeOfHeaders")
        directory_count = _u32(data, optional + directories_count_offset, "NumberOfRvaAndSizes")
        if directory_count > 16:
            raise PEFormatError("NumberOfRvaAndSizes exceeds the PE maximum")
        directories_offset = optional + minimum_size
        if directories_offset + directory_count * 8 > optional + optional_size:
            raise PEFormatError("data directories overflow the optional header")
        self.directories: list[tuple[int, int]] = []
        for index in range(directory_count):
            entry = directories_offset + index * 8
            self.directories.append(
                (_u32(data, entry, "directory RVA"), _u32(data, entry + 4, "directory size"))
            )
        sections_offset = optional + optional_size
        self.sections: list[_Section] = []
        for index in range(section_count):
            header = sections_offset + index * _SECTION_HEADER_SIZE
            if header + _SECTION_HEADER_SIZE > len(data):
                raise PEFormatError("section table is truncated")
            virtual_size = _u32(data, header + 8, "VirtualSize")
            virtual_address = _u32(data, header + 12, "VirtualAddress")
            raw_size = _u32(data, header + 16, "SizeOfRawData")
            raw_pointer = _u32(data, header + 20, "PointerToRawData")
            if raw_size and raw_pointer + raw_size > len(data):
                raise PEFormatError("section raw data extends outside the file")
            self.sections.append(_Section(virtual_address, virtual_size, raw_pointer, raw_size))

    def directory(self, index: int) -> tuple[int, int] | None:
        if index >= len(self.directories):
            return None
        rva, size = self.directories[index]
        if rva == 0:
            return None
        return rva, size

    def file_offset(self, rva: int, length: int, field: str) -> int:
        """Map an RVA to a file offset with ``length`` readable bytes."""

        if rva < 0 or length <= 0:
            raise PEFormatError(f"{field} has an invalid address")
        for section in self.sections:
            span = max(section.virtual_size, section.raw_size)
            if section.virtual_address <= rva < section.virtual_address + span:
                offset = rva - section.virtual_address
                if offset + length > section.raw_size:
                    raise PEFormatError(f"{field} points into uninitialised section data")
                return section.raw_pointer + offset
        # Data placed inside the headers is mapped one to one.
        if rva + length <= min(self.size_of_headers, len(self.data)):
            return rva
        raise PEFormatError(f"{field} RVA 0x{rva:x} is not backed by any section")

    def c_string(self, rva: int, field: str) -> str:
        start = self.file_offset(rva, 1, field)
        end = self.data.find(b"\0", start, start + _MAXIMUM_NAME_LENGTH + 1)
        if end < 0:
            raise PEFormatError(f"{field} is not NUL terminated within {_MAXIMUM_NAME_LENGTH} bytes")
        # Make sure the whole string is backed by initialised data, not just
        # its first byte.
        self.file_offset(rva, end - start + 1, field)
        raw = self.data[start:end]
        if not raw:
            raise PEFormatError(f"{field} is empty")
        try:
            value = raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise PEFormatError(f"{field} is not ASCII") from error
        if any(ord(character) < 0x20 or character in "\\/:*?\"<>|" for character in value):
            raise PEFormatError(f"{field} contains characters that are invalid in a DLL name")
        return value


def _import_names(image: _Image) -> tuple[str, ...]:
    directory = image.directory(_DIRECTORY_IMPORT)
    if directory is None:
        return ()
    rva, _size = directory
    names: list[str] = []
    for index in range(_MAXIMUM_DESCRIPTORS + 1):
        if index == _MAXIMUM_DESCRIPTORS:
            raise PEFormatError("import directory has no terminating descriptor")
        offset = image.file_offset(
            rva + index * _IMPORT_DESCRIPTOR_SIZE, _IMPORT_DESCRIPTOR_SIZE, "import descriptor"
        )
        descriptor = image.data[offset : offset + _IMPORT_DESCRIPTOR_SIZE]
        if descriptor == b"\0" * _IMPORT_DESCRIPTOR_SIZE:
            break
        name_rva = struct.unpack_from("<I", descriptor, 12)[0]
        if name_rva == 0:
            raise PEFormatError("import descriptor has no DLL name")
        names.append(image.c_string(name_rva, "imported DLL name"))
    return tuple(names)


def _delay_import_names(image: _Image) -> tuple[str, ...]:
    directory = image.directory(_DIRECTORY_DELAY_IMPORT)
    if directory is None:
        return ()
    rva, _size = directory
    names: list[str] = []
    for index in range(_MAXIMUM_DESCRIPTORS + 1):
        if index == _MAXIMUM_DESCRIPTORS:
            raise PEFormatError("delay-load directory has no terminating descriptor")
        offset = image.file_offset(
            rva + index * _DELAY_DESCRIPTOR_SIZE, _DELAY_DESCRIPTOR_SIZE, "delay-load descriptor"
        )
        descriptor = image.data[offset : offset + _DELAY_DESCRIPTOR_SIZE]
        if descriptor == b"\0" * _DELAY_DESCRIPTOR_SIZE:
            break
        attributes, name_address = struct.unpack_from("<II", descriptor, 0)
        if name_address == 0:
            raise PEFormatError("delay-load descriptor has no DLL name")
        # Attribute bit 0 says the fields are RVAs; the pre-VS2008 layout
        # stored virtual addresses that must be rebased first.
        if not attributes & 1:
            if name_address < image.image_base:
                raise PEFormatError("delay-load DLL name address precedes the image base")
            name_address -= image.image_base
        names.append(image.c_string(name_address, "delay-loaded DLL name"))
    return tuple(names)


def parse_pe_imports(data: bytes) -> PEImports:
    """Parse ``data`` as a PE32/PE32+ image and return its imported DLL names."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("PE image must be bytes")
    image = _Image(bytes(data))
    return PEImports(
        format=image.format,
        machine=image.machine,
        machineName=MACHINE_NAMES.get(image.machine, f"machine-0x{image.machine:04x}"),
        subsystem=image.subsystem,
        subsystemName=SUBSYSTEM_NAMES.get(image.subsystem, f"subsystem-{image.subsystem}"),
        isDll=bool(image.characteristics & _IMAGE_FILE_DLL),
        imports=tuple(sorted(set(_import_names(image)), key=str.casefold)),
        delayImports=tuple(sorted(set(_delay_import_names(image)), key=str.casefold)),
    )


def read_pe_imports(path: Path) -> PEImports:
    try:
        data = Path(path).read_bytes()
    except OSError as error:
        raise PEFormatError(f"cannot read PE image {path}: {error}") from error
    return parse_pe_imports(data)


def is_pe_image(path: Path) -> bool:
    """Cheap magic check used to select candidates before a full parse."""

    try:
        with Path(path).open("rb") as stream:
            header = stream.read(0x40)
    except OSError:
        return False
    if len(header) < 0x40 or header[:2] != _DOS_MAGIC:
        return False
    pe_offset = struct.unpack_from("<I", header, 0x3C)[0]
    try:
        with Path(path).open("rb") as stream:
            stream.seek(pe_offset)
            return stream.read(4) == _PE_SIGNATURE
    except (OSError, OverflowError, ValueError):
        return False


def dynamic_crt_imports(names: Iterable[str]) -> tuple[str, ...]:
    """Imported Visual C++ runtime DLLs that a static-CRT build must not have."""

    return tuple(sorted({name for name in names if DYNAMIC_CRT_PATTERN.match(name)}, key=str.casefold))


def ucrt_imports(names: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({name for name in names if UCRT_PATTERN.match(name)}, key=str.casefold))


def crt_linkage(names: Iterable[str]) -> str:
    """``static`` when no Visual C++ runtime DLL is imported, else ``dynamic``."""

    return "dynamic" if dynamic_crt_imports(names) else "static"


def is_windows_system_dll(name: str) -> bool:
    """True for DLLs every supported Windows installation provides."""

    lowered = name.casefold()
    if lowered.startswith(_API_SET_PREFIXES) and lowered.endswith(".dll"):
        return True
    return lowered in WINDOWS_SYSTEM_DLLS


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path)
    arguments = parser.parse_args(argv)
    report: dict[str, object] = {}
    failures = 0
    for path in arguments.paths:
        try:
            facts = read_pe_imports(path)
        except PEFormatError as error:
            report[str(path)] = {"error": str(error)}
            failures += 1
            continue
        entry = facts.as_dict()
        entry["crtLinkage"] = crt_linkage(facts.all_imports)
        entry["dynamicCrtImports"] = list(dynamic_crt_imports(facts.all_imports))
        entry["ucrtImports"] = list(ucrt_imports(facts.all_imports))
        entry["systemImports"] = [name for name in facts.all_imports if is_windows_system_dll(name)]
        report[str(path)] = entry
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "DYNAMIC_CRT_PATTERN",
    "MACHINE_NAMES",
    "PEFormatError",
    "PEImports",
    "UCRT_PATTERN",
    "WINDOWS_SYSTEM_DLLS",
    "crt_linkage",
    "dynamic_crt_imports",
    "is_pe_image",
    "is_windows_system_dll",
    "main",
    "parse_pe_imports",
    "read_pe_imports",
    "ucrt_imports",
]
