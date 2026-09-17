"""Platform service layer: every platform's probes run on every host."""

from __future__ import annotations

from pathlib import Path
import struct

import pytest

from openastroflow_engine import platform as platform_services
from openastroflow_engine.platform import (
    CpuTopology,
    MemoryStatus,
    current,
    fallback_memory,
    fallback_topology,
    platform_id_for,
    services_for,
)
from openastroflow_engine.platform.darwin import parse_sysctl_topology
from openastroflow_engine.platform.linux import parse_proc_cpuinfo, parse_proc_meminfo_available
from openastroflow_engine.platform.posix import sysconf_memory
from openastroflow_engine.platform.windows import (
    global_memory_status,
    parse_processor_core_records,
    topology_from_core_records,
)


def _core_record(mask: int, *, smt: bool, efficiency_class: int) -> bytes:
    """One SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX RelationProcessorCore record."""

    relationship = 0
    flags = 1 if smt else 0
    body = struct.pack("<BB", flags, efficiency_class) + bytes(20) + struct.pack("<H", 1)
    group = struct.pack("<QH", mask, 0) + bytes(6)
    payload = body + group
    size = 8 + len(payload)
    return struct.pack("<II", relationship, size) + payload


def test_platform_id_mapping() -> None:
    assert platform_id_for("darwin") == "darwin"
    assert platform_id_for("win32") == "windows"
    assert platform_id_for("linux") == "linux"
    assert platform_id_for("freebsd14") == "linux"


def test_every_platform_service_is_constructible_on_this_host() -> None:
    names = {
        "darwin": "libopenastroflow_native.dylib",
        "windows": "openastroflow_native.dll",
        "linux": "libopenastroflow_native.so",
    }
    for identifier, filename in names.items():
        services = services_for(identifier)
        assert services.platform_id == identifier
        assert services.native_library_filename() == filename
        assert isinstance(services.scientific_execution_validated, bool)
    assert current() is current()
    assert current().platform_id == platform_id_for()


def test_host_probes_report_measured_values() -> None:
    memory = current().memory_status()
    assert memory.source != "fallback"
    assert memory.total_bytes > 1024**3
    topology = current().cpu_topology()
    assert topology.logical_cores >= 1
    assert topology.source != "unavailable"
    serialized = topology.serializable()
    assert serialized["logicalCores"] == topology.logical_cores
    assert memory.serializable()["source"] == memory.source


def test_windows_core_records_parse_physical_smt_and_hybrid_classes() -> None:
    # 8 physical cores, SMT (two logical each): 16 logical, one class.
    ryzen = b"".join(_core_record(0b11 << (2 * i), smt=True, efficiency_class=0) for i in range(8))
    physical, logical, smt, classes = parse_processor_core_records(ryzen)
    assert (physical, logical, smt) == (8, 16, True)
    assert classes == {0: 8}
    topology = topology_from_core_records(ryzen, "AMD Ryzen 7 5800H")
    assert topology.brand == "AMD Ryzen 7 5800H"
    assert (topology.physical_cores, topology.logical_cores, topology.smt) == (8, 16, True)
    assert (topology.performance_cores, topology.efficiency_cores) == (0, 0)
    assert topology.source == "GetLogicalProcessorInformationEx"

    # Intel hybrid: 6 P cores with SMT (class 1) + 8 E cores (class 0).
    hybrid = b"".join(_core_record(0b11 << (2 * i), smt=True, efficiency_class=1) for i in range(6))
    hybrid += b"".join(_core_record(1 << (12 + i), smt=False, efficiency_class=0) for i in range(8))
    topology = topology_from_core_records(hybrid, "12th Gen Intel(R) Core(TM) i7-12700H")
    assert (topology.physical_cores, topology.logical_cores) == (14, 20)
    assert (topology.performance_cores, topology.efficiency_cores) == (6, 8)
    assert topology.smt is True


def test_windows_core_records_ignore_other_relationships_and_truncation() -> None:
    cache_record = struct.pack("<II", 2, 40) + bytes(32)
    core = _core_record(0b1, smt=False, efficiency_class=0)
    physical, logical, smt, _classes = parse_processor_core_records(cache_record + core + b"\x01\x02")
    assert (physical, logical, smt) == (1, 1, False)
    assert parse_processor_core_records(b"") == (0, 0, False, {})
    fallback = topology_from_core_records(b"", "brand")
    assert fallback.source == "os.cpu_count"
    assert fallback.brand == "brand"


def test_windows_memory_probe_falls_back_off_windows() -> None:
    import sys

    status = global_memory_status()
    if sys.platform == "win32":
        assert status.source == "GlobalMemoryStatusEx"
        assert status.total_bytes > 0
        assert status.available_bytes is not None
    else:
        assert status == fallback_memory()


def test_sysctl_topology_parses_apple_silicon_and_intel_layouts() -> None:
    apple = (
        "machdep.cpu.brand_string: Apple M3 Pro\n"
        "hw.physicalcpu: 12\n"
        "hw.logicalcpu: 12\n"
        "hw.perflevel0.physicalcpu: 6\n"
        "hw.perflevel1.physicalcpu: 6\n"
    )
    topology = parse_sysctl_topology(apple, brand_fallback=lambda: "unused")
    assert topology.brand == "Apple M3 Pro"
    assert (topology.physical_cores, topology.logical_cores, topology.smt) == (12, 12, False)
    assert (topology.performance_cores, topology.efficiency_cores) == (6, 6)
    intel = "machdep.cpu.brand_string: Intel(R) Core(TM) i9\nhw.physicalcpu: 8\nhw.logicalcpu: 16\n"
    topology = parse_sysctl_topology(intel, brand_fallback=lambda: "unused")
    assert (topology.physical_cores, topology.logical_cores, topology.smt) == (8, 16, True)
    assert (topology.performance_cores, topology.efficiency_cores) == (0, 0)
    missing_brand = parse_sysctl_topology("hw.logicalcpu: 4\n", brand_fallback=lambda: "Apple M9")
    assert missing_brand.brand == "Apple M9"
    assert missing_brand.logical_cores == 4


def test_proc_cpuinfo_and_meminfo_parsers() -> None:
    cpuinfo = ""
    for processor in range(4):
        cpuinfo += (
            f"processor\t: {processor}\n"
            "model name\t: AMD Ryzen 5 5600\n"
            "physical id\t: 0\n"
            f"core id\t\t: {processor // 2}\n\n"
        )
    topology = parse_proc_cpuinfo(cpuinfo)
    assert topology.brand == "AMD Ryzen 5 5600"
    assert (topology.logical_cores, topology.physical_cores, topology.smt) == (4, 2, True)
    assert parse_proc_cpuinfo("").source == "os.cpu_count"
    assert parse_proc_meminfo_available("MemTotal: 10 kB\nMemAvailable: 2048 kB\n") == 2048 * 1024
    assert parse_proc_meminfo_available("MemTotal: 10 kB\n") is None


def test_sysconf_memory_reports_fallback_when_unavailable() -> None:
    def broken(name: str) -> int:
        raise OSError(name)

    assert sysconf_memory(broken) == fallback_memory()
    values = {"SC_PHYS_PAGES": 1000, "SC_PAGE_SIZE": 4096}
    status = sysconf_memory(lambda name: values[name], available_bytes=123)
    assert status == MemoryStatus(4096000, 123, "sysconf")


def test_fallbacks_are_explicit_never_silent() -> None:
    assert fallback_memory().source == "fallback"
    assert fallback_memory().total_bytes == platform_services.base.MEMORY_FALLBACK_BYTES
    topology = fallback_topology("brand")
    assert isinstance(topology, CpuTopology)
    assert topology.source == "os.cpu_count"
    assert topology.physical_cores == 0


@pytest.mark.parametrize("identifier", ["darwin", "windows", "linux"])
def test_services_expose_the_protocol_surface(identifier: str) -> None:
    services = services_for(identifier)
    for method in ("memory_status", "cpu_topology", "gpu_adapters", "native_library_filename"):
        assert callable(getattr(services, method))
    assert services.gpu_adapters() == ()


# ---------------------------------------------------------------------------
# File-system, process and environment primitives
# ---------------------------------------------------------------------------

import os
import subprocess
import sys

from openastroflow_engine.platform import ChildProcessOptions, NoReplaceError, VolumeCapabilities
from openastroflow_engine.platform.darwin import parse_mount_output
from openastroflow_engine.platform.linux import parse_proc_mounts
from openastroflow_engine.platform.posix import (
    hardlink_support,
    rename_directory_no_replace_with,
    volume_from_mount_table,
)
from openastroflow_engine.platform import windows as windows_platform


def test_host_rename_directory_no_replace_is_create_only(tmp_path) -> None:
    services = current()
    source = tmp_path / "staging"
    source.mkdir()
    (source / "file.txt").write_text("x", encoding="utf-8")
    destination = tmp_path / "final"
    services.rename_directory_no_replace(source, destination)
    assert (destination / "file.txt").read_text(encoding="utf-8") == "x"
    assert not source.exists()
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(NoReplaceError) as exists:
        services.rename_directory_no_replace(other, destination)
    assert exists.value.code == "OUTPUT_EXISTS"
    assert exists.value.precheck is True
    assert exists.value.path == str(destination)
    assert other.exists()


def test_posix_rename_reports_a_race_and_an_unsupported_platform(tmp_path) -> None:
    source = tmp_path / "s"
    source.mkdir()
    destination = tmp_path / "d"

    def racing_rename(src: bytes, dst: bytes) -> int:
        import ctypes
        import errno

        ctypes.set_errno(errno.EEXIST)
        return -1

    with pytest.raises(NoReplaceError) as raced:
        rename_directory_no_replace_with(source, destination, racing_rename)
    assert raced.value.code == "OUTPUT_EXISTS"
    assert raced.value.precheck is False
    with pytest.raises(NoReplaceError) as unsupported:
        rename_directory_no_replace_with(source, destination, None)
    assert unsupported.value.code == "ATOMIC_DIRECTORY_PUBLISH_UNSUPPORTED"

    def failing_rename(src: bytes, dst: bytes) -> int:
        import ctypes
        import errno

        ctypes.set_errno(errno.EXDEV)
        return -1

    with pytest.raises(OSError) as failed:
        rename_directory_no_replace_with(source, destination, failing_rename)
    assert failed.value.errno == 18


def test_windows_rename_and_publish_semantics_with_injected_calls(tmp_path) -> None:
    source = tmp_path / "s"
    source.mkdir()
    destination = tmp_path / "d"
    calls: list[tuple[str, str]] = []

    def rename(src, dst) -> None:
        calls.append(("rename", str(dst)))

    windows_platform.rename_directory_no_replace_with(source, destination, rename)
    assert calls == [("rename", str(destination))]

    def racing(src, dst) -> None:
        raise FileExistsError(183, "exists")

    with pytest.raises(NoReplaceError) as raced:
        windows_platform.rename_directory_no_replace_with(source, destination, racing)
    assert raced.value.code == "OUTPUT_EXISTS" and raced.value.precheck is False
    destination.mkdir()
    with pytest.raises(NoReplaceError) as pre:
        windows_platform.rename_directory_no_replace_with(source, destination, rename)
    assert pre.value.precheck is True

    temporary = tmp_path / "t.tmp"
    temporary.write_text("v", encoding="utf-8")
    final = tmp_path / "final.fits"

    def linking(src, dst) -> None:
        Path(dst).write_bytes(Path(src).read_bytes())

    assert windows_platform.publish_file_no_replace_with(temporary, final, link=linking, rename=rename) == "hardlink"
    assert final.read_text(encoding="utf-8") == "v" and not temporary.exists()

    temporary.write_text("w", encoding="utf-8")
    moved = tmp_path / "moved.fits"

    def no_hardlinks(src, dst) -> None:
        raise OSError(1, "Incorrect function")

    def move(src, dst) -> None:
        os.rename(src, dst)

    assert windows_platform.publish_file_no_replace_with(temporary, moved, link=no_hardlinks, rename=move) == "rename"
    assert moved.read_text(encoding="utf-8") == "w" and not temporary.exists()

    temporary.write_text("z", encoding="utf-8")

    def existing(src, dst) -> None:
        raise FileExistsError(80, "exists")

    with pytest.raises(FileExistsError):
        windows_platform.publish_file_no_replace_with(temporary, final, link=existing, rename=move)
    assert temporary.exists()


def test_host_publish_file_no_replace(tmp_path) -> None:
    services = current()
    temporary = tmp_path / "a.tmp"
    temporary.write_text("a", encoding="utf-8")
    final = tmp_path / "a.fits"
    mode = services.publish_file_no_replace(temporary, final)
    assert mode in {"hardlink", "rename"}
    assert final.read_text(encoding="utf-8") == "a" and not temporary.exists()
    temporary.write_text("b", encoding="utf-8")
    with pytest.raises(FileExistsError):
        services.publish_file_no_replace(temporary, final)
    assert final.read_text(encoding="utf-8") == "a"


def test_host_fsync_directory_and_volume_facts(tmp_path) -> None:
    services = current()
    result = services.fsync_directory(tmp_path)
    assert isinstance(result, bool)
    volume = services.volume_capabilities(tmp_path)
    assert isinstance(volume, VolumeCapabilities)
    assert volume.filesystem
    if sys.platform == "darwin":
        assert volume.source == "mount"
        assert volume.filesystem in {"apfs", "hfs"}
        assert volume.hardlinks is True
    serialized = volume.serializable()
    assert serialized["filesystem"] == volume.filesystem


def test_mount_table_parsers_and_hardlink_knowledge() -> None:
    darwin = parse_mount_output(
        "/dev/disk3s5 on / (apfs, sealed, local, read-only, journaled)\n"
        "/dev/disk4s1 on /Volumes/USB DISK (exfat, local, nodev, nosuid, noowners)\n"
        "map auto_home on /System/Volumes/Data/home (autofs, automounted, nobrowse)\n"
    )
    assert darwin[0] == ("/", "apfs")
    assert darwin[1] == ("/Volumes/USB DISK", "exfat")
    usb = volume_from_mount_table(Path("/Volumes/USB DISK/out"), darwin, source="mount")
    assert (usb.filesystem, usb.hardlinks) == ("exfat", False)
    root = volume_from_mount_table(Path("/Users/example"), darwin, source="mount")
    assert (root.filesystem, root.hardlinks) == ("apfs", True)
    linux = parse_proc_mounts(
        "/dev/nvme0n1p2 / ext4 rw,relatime 0 0\n"
        "/dev/sdb1 /media/user/My\\040Disk vfat rw 0 0\n"
        "bad line\n"
    )
    assert linux == [("/", "ext4"), ("/media/user/My Disk", "vfat")]
    assert volume_from_mount_table(Path("/media/user/My Disk/a"), linux, source="/proc/mounts").hardlinks is False
    assert hardlink_support("ntfs") is True and hardlink_support("smbfs") is False
    assert hardlink_support("weirdfs") is None
    unknown = volume_from_mount_table(Path("/x"), [], source="mount")
    assert unknown.filesystem == "unknown" and unknown.hardlinks is None


def test_windows_volume_flags_are_interpreted() -> None:
    ntfs = windows_platform.volume_capabilities_from("NTFS", 0x03E706FF)
    assert (ntfs.filesystem, ntfs.hardlinks, ntfs.case_sensitive) == ("NTFS", True, True)
    exfat = windows_platform.volume_capabilities_from("exFAT", 0x00000000)
    assert (exfat.hardlinks, exfat.case_sensitive) == (False, False)
    assert windows_platform.volume_capabilities_from("", 0).filesystem == "unknown"
    if sys.platform != "win32":
        assert windows_platform.volume_information(Path(".")).source == "unavailable"
        assert windows_platform.flush_directory(Path(".")) is False


def test_roots_follow_each_platform_convention(tmp_path) -> None:
    darwin = services_for("darwin")
    assert darwin.cache_root() == Path.home() / "Library" / "Caches"
    assert darwin.data_root(environment={}, home=tmp_path) == tmp_path / ".openastroflow"
    assert darwin.data_root(environment={"OPENASTROFLOW_DATA_DIR": str(tmp_path / "d")}) == tmp_path / "d"
    windows = services_for("windows")
    assert windows.data_root(environment={"LOCALAPPDATA": r"C:\Users\example\AppData\Local"}) == Path(r"C:\Users\example\AppData\Local") / "OpenAstroFlow"
    assert windows.data_root(environment={}, home=tmp_path) == tmp_path / ".openastroflow"
    linux = services_for("linux")
    assert linux.data_root(environment={}, home=tmp_path) == tmp_path / ".openastroflow"


def test_child_process_options_and_executable_tables() -> None:
    posix = ChildProcessOptions(start_new_session=True)
    assert posix.popen_kwargs() == {"start_new_session": True}
    assert ChildProcessOptions().popen_kwargs() == {}
    assert services_for("darwin").child_process_options().start_new_session is True
    windows = services_for("windows").child_process_options()
    if sys.platform == "win32":
        assert windows.creationflags & subprocess.CREATE_NEW_PROCESS_GROUP
        assert windows.creationflags & subprocess.CREATE_NO_WINDOW
    assert services_for("darwin").well_known_executables("astap")[0] == Path("/Applications/ASTAP.app/Contents/MacOS/astap")
    assert services_for("linux").well_known_executables("siril-cli") == (Path("/usr/bin/siril-cli"), Path("/usr/local/bin/siril-cli"))
    environment = {"ProgramFiles": r"C:\Program Files", "LOCALAPPDATA": r"C:\Users\example\AppData\Local"}
    candidates = services_for("windows").well_known_executables("solve-field", environment=environment)
    assert candidates[0] == Path(r"C:\Program Files") / r"Astrometry.net\bin\solve-field.exe"
    assert len(candidates) == 4
    assert services_for("windows").well_known_executables("unknown-tool", environment=environment) == ()
    assert services_for("darwin").well_known_executables("unknown-tool") == ()


def test_keep_awake_is_a_no_op_context_everywhere() -> None:
    for identifier in ("darwin", "windows", "linux"):
        services = services_for(identifier)
        if identifier != platform_id_for():
            # Foreign services must not touch the host (no caffeinate on
            # Windows, no SetThreadExecutionState on macOS).
            if identifier == "darwin" and sys.platform != "darwin":
                continue
        with services.keep_awake():
            pass


def test_kill_process_tree_terminates_a_live_child() -> None:
    services = current()
    options = services.child_process_options().popen_kwargs()
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **options,
    )
    try:
        services.kill_process_tree(process)
        assert process.wait(timeout=10) != 0
    finally:
        if process.poll() is None:
            process.kill()
    services.kill_process_tree(process)
