from __future__ import annotations

import json
from pathlib import Path
import struct
import sys

import pytest

from pe_fixtures import (
    FIRST_IMPORT_NAME_RVA_FIELD,
    MACHINE_ARM64,
    MACHINE_I386,
    SECTION_RAW_OFFSET,
    build_pe_image,
)
from scripts.pe_imports import (
    PEFormatError,
    crt_linkage,
    dynamic_crt_imports,
    is_pe_image,
    is_windows_system_dll,
    main,
    parse_pe_imports,
    read_pe_imports,
    ucrt_imports,
)


def test_parses_pe32_plus_import_table_naming_two_dlls() -> None:
    image = build_pe_image(imports=("KERNEL32.dll", "VCRUNTIME140.dll"))

    facts = parse_pe_imports(image)

    assert facts.format == "PE32+"
    assert facts.machineName == "x86_64"
    assert facts.isDll is True
    assert facts.subsystemName == "windows-cui"
    assert facts.imports == ("KERNEL32.dll", "VCRUNTIME140.dll")
    assert facts.delayImports == ()
    assert facts.all_imports == ("KERNEL32.dll", "VCRUNTIME140.dll")


def test_parses_pe32_executable_and_orders_names_case_insensitively() -> None:
    image = build_pe_image(
        imports=("python312.dll", "ADVAPI32.dll", "api-ms-win-crt-runtime-l1-1-0.dll"),
        pe32_plus=False,
        dll=False,
        machine=MACHINE_I386,
        subsystem=2,
    )

    facts = parse_pe_imports(image)

    assert facts.format == "PE32"
    assert facts.machineName == "x86"
    assert facts.isDll is False
    assert facts.subsystemName == "windows-gui"
    assert facts.imports == ("ADVAPI32.dll", "api-ms-win-crt-runtime-l1-1-0.dll", "python312.dll")


@pytest.mark.parametrize("pe32_plus", (True, False))
@pytest.mark.parametrize("rva_based", (True, False))
def test_reads_delay_load_directory_in_both_layouts(pe32_plus: bool, rva_based: bool) -> None:
    image = build_pe_image(
        imports=("KERNEL32.dll",),
        delay_imports=("SHELL32.dll", "ole32.dll"),
        pe32_plus=pe32_plus,
        delay_rva_based=rva_based,
    )

    facts = parse_pe_imports(image)

    assert facts.imports == ("KERNEL32.dll",)
    assert facts.delayImports == ("ole32.dll", "SHELL32.dll")
    assert facts.all_imports == ("KERNEL32.dll", "ole32.dll", "SHELL32.dll")


def test_reports_arm64_machine_and_empty_import_table() -> None:
    facts = parse_pe_imports(build_pe_image(imports=(), machine=MACHINE_ARM64))

    assert facts.machineName == "aarch64"
    assert facts.imports == ()


@pytest.mark.parametrize(
    "mutate, message",
    (
        (lambda image: b"", "MZ DOS header"),
        (lambda image: b"\xcf\xfa\xed\xfe" + image[4:], "MZ DOS header"),
        (lambda image: image[:0x3C] + struct.pack("<I", 0x7FFFFFF0) + image[0x40:], "e_lfanew"),
        (lambda image: image[:0x40] + b"XX\0\0" + image[0x44:], "PE signature"),
        (lambda image: image[:0x58] + struct.pack("<H", 0x107) + image[0x5A:], "optional header magic"),
        (lambda image: image[:0x120], "truncated"),
        (
            lambda image: image[:FIRST_IMPORT_NAME_RVA_FIELD]
            + struct.pack("<I", 0x7FFF0000)
            + image[FIRST_IMPORT_NAME_RVA_FIELD + 4 :],
            "not backed by any section",
        ),
        (
            lambda image: image[:FIRST_IMPORT_NAME_RVA_FIELD]
            + struct.pack("<I", 0)
            + image[FIRST_IMPORT_NAME_RVA_FIELD + 4 :],
            "no DLL name",
        ),
    ),
)
def test_malformed_images_raise_pe_format_error(mutate, message: str) -> None:
    image = build_pe_image(imports=("KERNEL32.dll",))

    with pytest.raises(PEFormatError, match=message):
        parse_pe_imports(mutate(image))


def test_import_directory_without_terminator_is_rejected() -> None:
    image = bytearray(build_pe_image(imports=("KERNEL32.dll",)))
    # Overwrite the terminating (all-zero) descriptor with a copy of the first
    # one so the walk never finds a terminator inside initialised data.
    first = bytes(image[SECTION_RAW_OFFSET : SECTION_RAW_OFFSET + 20])
    image[SECTION_RAW_OFFSET + 20 : SECTION_RAW_OFFSET + 40] = first
    section_end = len(image)
    # Every following 20-byte slot up to the section end is also a copy, so
    # the parser must fail on running out of section data, not loop forever.
    for offset in range(SECTION_RAW_OFFSET + 40, section_end - 20, 20):
        image[offset : offset + 20] = first

    with pytest.raises(PEFormatError):
        parse_pe_imports(bytes(image))


def test_non_ascii_or_control_names_are_rejected() -> None:
    image = bytearray(build_pe_image(imports=("KERNEL32.dll",)))
    marker = image.index(b"KERNEL32.dll\0")
    image[marker : marker + 12] = b"KERNEL\xff2.dll"

    with pytest.raises(PEFormatError, match="ASCII"):
        parse_pe_imports(bytes(image))

    image[marker : marker + 12] = b"KERN\\EL32.dl"
    with pytest.raises(PEFormatError, match="invalid in a DLL name"):
        parse_pe_imports(bytes(image))


def test_crt_classification_and_system_allow_list() -> None:
    names = (
        "KERNEL32.dll",
        "api-ms-win-crt-heap-l1-1-0.dll",
        "ucrtbase.dll",
        "VCRUNTIME140.dll",
        "VCRUNTIME140_1.dll",
        "MSVCP140.dll",
        "msvcp140-abcdef0123456789.dll",
        "python312.dll",
    )

    assert crt_linkage(names) == "dynamic"
    assert crt_linkage(("KERNEL32.dll", "USER32.dll")) == "static"
    assert crt_linkage(("KERNEL32.dll", "api-ms-win-crt-runtime-l1-1-0.dll")) == "static"
    assert dynamic_crt_imports(names) == (
        "msvcp140-abcdef0123456789.dll",
        "MSVCP140.dll",
        "VCRUNTIME140.dll",
        "VCRUNTIME140_1.dll",
    )
    assert ucrt_imports(names) == ("api-ms-win-crt-heap-l1-1-0.dll", "ucrtbase.dll")
    assert is_windows_system_dll("kernel32.dll")
    assert is_windows_system_dll("KERNEL32.DLL")
    assert is_windows_system_dll("api-ms-win-core-synch-l1-2-0.dll")
    assert is_windows_system_dll("ext-ms-win-ntuser-window-l1-1-0.dll")
    assert is_windows_system_dll("WS2_32.dll")
    assert not is_windows_system_dll("VCRUNTIME140.dll")
    assert not is_windows_system_dll("MSVCP140.dll")
    assert not is_windows_system_dll("python312.dll")
    assert not is_windows_system_dll("api-ms-win-crt-runtime-l1-1-0.txt")


def test_is_pe_image_uses_the_magic_not_the_suffix(tmp_path: Path) -> None:
    dll = tmp_path / "kernels.dll"
    dll.write_bytes(build_pe_image(imports=("KERNEL32.dll",)))
    fake = tmp_path / "fake.dll"
    fake.write_bytes(b"\xcf\xfa\xed\xfe" + b"\0" * 100)
    short = tmp_path / "short.pyd"
    short.write_bytes(b"MZ")
    dylib = tmp_path / "libufwbpp_native.dylib"
    dylib.write_bytes(b"\xcf\xfa\xed\xfe" + b"\0" * 100)

    assert is_pe_image(dll)
    assert not is_pe_image(fake)
    assert not is_pe_image(short)
    assert not is_pe_image(dylib)
    assert not is_pe_image(tmp_path / "missing.dll")
    assert read_pe_imports(dll).imports == ("KERNEL32.dll",)
    with pytest.raises(PEFormatError, match="cannot read"):
        read_pe_imports(tmp_path / "missing.dll")


def test_cli_reports_every_file_and_fails_on_invalid_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dll = tmp_path / "kernels.dll"
    dll.write_bytes(build_pe_image(imports=("KERNEL32.dll", "MSVCP140.dll")))
    broken = tmp_path / "broken.dll"
    broken.write_bytes(b"not a pe")

    assert main([str(dll)]) == 0
    report = json.loads(capsys.readouterr().out)
    entry = report[str(dll)]
    assert entry["crtLinkage"] == "dynamic"
    assert entry["dynamicCrtImports"] == ["MSVCP140.dll"]
    assert entry["systemImports"] == ["KERNEL32.dll"]

    assert main([str(dll), str(broken)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert "error" in report[str(broken)]


def test_parses_a_real_msvc_built_launcher_when_setuptools_ships_one() -> None:
    # setuptools ships prebuilt Windows console launchers on every platform;
    # they are genuine MSVC/PE32+ images with a dynamic VC runtime import.
    candidates = [
        Path(entry) / "setuptools" / "cli-64.exe"
        for entry in sys.path
        if (Path(entry) / "setuptools" / "cli-64.exe").is_file()
    ]
    if not candidates:
        pytest.skip("setuptools launcher executables are not installed")

    facts = read_pe_imports(candidates[0])

    assert facts.format == "PE32+"
    assert facts.machineName == "x86_64"
    assert facts.isDll is False
    assert "KERNEL32.dll" in facts.imports
    assert all(is_windows_system_dll(name) or dynamic_crt_imports((name,)) for name in facts.imports)
