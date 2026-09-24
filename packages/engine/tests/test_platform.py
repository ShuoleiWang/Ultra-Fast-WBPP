"""Platform service layer: every platform's probes run on every host."""

from __future__ import annotations

from pathlib import Path
import struct

import pytest

from ufwbpp import platform as platform_services
from ufwbpp.platform import (
    CpuTopology,
    MemoryStatus,
    current,
    fallback_memory,
    fallback_topology,
    platform_id_for,
    services_for,
)
from ufwbpp.platform.darwin import parse_sysctl_topology
from ufwbpp.platform.linux import parse_proc_cpuinfo, parse_proc_meminfo_available
from ufwbpp.platform.posix import sysconf_memory
from ufwbpp.platform.windows import (
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
        "darwin": "libufwbpp_native.dylib",
        "windows": "ufwbpp_native.dll",
        "linux": "libufwbpp_native.so",
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
import shutil
import subprocess
import sys

from ufwbpp.platform import ChildProcessOptions, NoReplaceError, VolumeCapabilities
from ufwbpp.platform.darwin import parse_mount_output
from ufwbpp.platform.linux import parse_proc_mounts
from ufwbpp.platform.posix import (
    hardlink_support,
    rename_directory_no_replace_with,
    volume_from_mount_table,
)
from ufwbpp.platform import windows as windows_platform


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
    assert darwin.data_root(environment={}, home=tmp_path) == tmp_path / ".ultra-fast-wbpp"
    assert darwin.data_root(environment={"UFWBPP_DATA_DIR": str(tmp_path / "d")}) == tmp_path / "d"
    windows = services_for("windows")
    assert windows.data_root(environment={"LOCALAPPDATA": r"C:\Users\example\AppData\Local"}) == Path(r"C:\Users\example\AppData\Local") / "Ultra-Fast-WBPP"
    assert windows.data_root(environment={}, home=tmp_path) == tmp_path / ".ultra-fast-wbpp"
    linux = services_for("linux")
    assert linux.data_root(environment={}, home=tmp_path) == tmp_path / ".ultra-fast-wbpp"


def test_an_existing_legacy_data_root_is_used_in_place(tmp_path) -> None:
    # Installed catalogs record their absolute location, so a root created
    # under the former project name is kept instead of moved.
    (tmp_path / ".openastroflow").mkdir()
    assert services_for("darwin").data_root(environment={}, home=tmp_path) == tmp_path / ".openastroflow"
    local = tmp_path / "Local"
    (local / "OpenAstroFlow").mkdir(parents=True)
    windows = services_for("windows")
    assert windows.data_root(environment={"LOCALAPPDATA": str(local)}) == local / "OpenAstroFlow"
    # Once the new root exists it wins.
    (tmp_path / ".ultra-fast-wbpp").mkdir()
    (local / "Ultra-Fast-WBPP").mkdir()
    assert services_for("darwin").data_root(environment={}, home=tmp_path) == tmp_path / ".ultra-fast-wbpp"
    assert windows.data_root(environment={"LOCALAPPDATA": str(local)}) == local / "Ultra-Fast-WBPP"


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
    assert Path("/usr/bin/solve-field") in services_for("linux").well_known_executables("solve-field")
    environment = {"ProgramFiles": r"C:\Program Files", "LOCALAPPDATA": r"C:\Users\example\AppData\Local"}
    candidates = services_for("windows").well_known_executables("solve-field", environment=environment)
    assert candidates[0] == Path(r"C:\Program Files").joinpath("Astrometry.net", "bin", "solve-field.exe")
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


def test_environment_view_resolves_windows_names_regardless_of_case() -> None:
    from ufwbpp.platform import EnvironmentView, environment_view, merged_environment

    # A merged copy of ``os.environ`` on Windows carries upper-cased keys.
    upper = {"PROGRAMFILES": r"C:\Program Files", "PATH": r"C:\Windows", "ProgramFiles(x86)": r"C:\Program Files (x86)"}
    view = environment_view(upper, platform_id="windows")
    assert isinstance(view, EnvironmentView) and view.case_insensitive
    assert view.get("ProgramFiles") == r"C:\Program Files"
    assert view["programfiles(X86)"] == r"C:\Program Files (x86)"
    assert "path" in view and "Missing" not in view and view.get("Missing") is None
    assert list(view) == list(upper) and len(view) == 3
    with pytest.raises(KeyError):
        view["nothing"]
    # POSIX resolves names exactly; the view is transparent.
    posix = environment_view(upper, platform_id="darwin")
    assert posix.get("ProgramFiles") is None and posix.get("PROGRAMFILES") == r"C:\Program Files"
    # Overrides replace the existing spelling on Windows instead of adding a second key.
    merged = merged_environment(upper, {"Path": r"D:\tools", "TEMP": r"C:\t"}, platform_id="windows")
    assert merged == {"PROGRAMFILES": r"C:\Program Files", "ProgramFiles(x86)": r"C:\Program Files (x86)", "Path": r"D:\tools", "TEMP": r"C:\t"}
    assert merged_environment(upper, {"Path": "x"}, platform_id="linux") == {**upper, "Path": "x"}
    assert merged_environment(upper, None, platform_id="windows") == upper


def test_windows_well_known_executables_prefer_astap_cli_and_ignore_key_case(tmp_path: Path) -> None:
    program_files = tmp_path / "Program Files"
    (program_files / "astap").mkdir(parents=True)
    cli = program_files / "astap" / "astap_cli.exe"
    gui = program_files / "astap" / "astap.exe"
    cli.write_bytes(b"MZ"); gui.write_bytes(b"MZ")
    services = platform_services.services_for("windows")
    candidates = services.well_known_executables("astap", environment={"PROGRAMFILES": str(program_files), "LOCALAPPDATA": str(tmp_path / "Local")})
    assert candidates[0] == cli and candidates[1] == gui
    # Both spellings of the folder are listed for case-sensitive volumes; the
    # per-user root follows the machine-wide ones.
    assert [candidate.relative_to(program_files).as_posix().lower() for candidate in candidates[:4]] == [
        "astap/astap_cli.exe", "astap/astap.exe", "astap/astap_cli.exe", "astap/astap.exe",
    ]
    assert candidates[4].is_relative_to(tmp_path / "Local")
    assert services.data_root(environment={"LOCALAPPDATA": str(tmp_path / "Local")}) == tmp_path / "Local" / "Ultra-Fast-WBPP"
    assert services.data_root(environment={"localappdata": str(tmp_path / "Local")}) == tmp_path / "Local" / "Ultra-Fast-WBPP"


# ---------------------------------------------------------------------------
# Path limits and file lifecycle helpers
# ---------------------------------------------------------------------------

import threading
import time

from ufwbpp.platform import PathLimit, remove_file, remove_tree, rename_with_retry
from ufwbpp.platform.base import (
    RETRIED_WINERRORS,
    RETRY_BUDGET_SECONDS,
    RETRY_INITIAL_SECONDS,
    RETRY_MAXIMUM_SECONDS,
    retry_file_operation,
)


class _SharingViolation(OSError):
    """``ERROR_SHARING_VIOLATION`` as CPython raises it on Windows."""

    winerror = 32


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def __call__(self) -> float:
        return self.now


def test_path_limit_shape_on_every_platform() -> None:
    for identifier in ("darwin", "linux"):
        limit = services_for(identifier).path_limit()
        assert limit == PathLimit(None, None, "unlimited")
        assert limit.serializable() == {"maxCharacters": None, "longPathsEnabled": None, "source": "unlimited"}
    windows = services_for("windows").path_limit()
    assert windows.max_characters in {259, 32767}
    if sys.platform == "win32":
        assert isinstance(windows.long_paths_enabled, bool)
        if windows.source == windows_platform.LONG_PATHS_REGISTRY_KEY:
            assert windows.max_characters == (32767 if windows.long_paths_enabled else 259)
        else:
            # The policy is on but this interpreter's manifest is not
            # longPathAware: the probe decided, and the source says so.
            assert windows.long_paths_enabled is True
            assert windows.max_characters == 259
            assert "longPathAware" in windows.source
    else:
        # Off Windows the registry cannot be read: the short limit applies.
        assert windows.max_characters == 259 and windows.long_paths_enabled is None
        assert "unavailable" in windows.source
    host = current().path_limit()
    assert isinstance(host, PathLimit)
    assert (host.max_characters is None) == (sys.platform != "win32")


def test_long_paths_policy_is_read_from_the_registry_value(tmp_path: Path) -> None:
    def missing(key: str, value: str) -> object:
        raise FileNotFoundError(2, "The system cannot find the file specified", key + "\\" + value)

    def denied(key: str, value: str) -> object:
        raise PermissionError(13, "Access is denied")

    read = {"key": None}

    def enabled(key: str, value: str) -> object:
        read["key"] = (key, value)
        return 1

    assert windows_platform.long_paths_enabled(enabled) is True
    assert read["key"] == (r"SYSTEM\CurrentControlSet\Control\FileSystem", "LongPathsEnabled")
    assert windows_platform.long_paths_enabled(lambda key, value: 0) is False
    assert windows_platform.long_paths_enabled(lambda key, value: "1") is True
    assert windows_platform.long_paths_enabled(lambda key, value: "junk") is False
    assert windows_platform.long_paths_enabled(missing) is False
    assert windows_platform.long_paths_enabled(denied) is None
    assert windows_platform.path_limit(True) == PathLimit(32767, True, windows_platform.LONG_PATHS_REGISTRY_KEY)
    assert windows_platform.path_limit(False) == PathLimit(259, False, windows_platform.LONG_PATHS_REGISTRY_KEY)
    # The policy alone does not make long paths usable: the executable's
    # manifest must opt in, which only a real attempt can confirm.
    on = lambda key, value: 1  # noqa: E731
    assert windows_platform.path_limit(read_value=on, probe=lambda: True) == PathLimit(32767, True, windows_platform.LONG_PATHS_REGISTRY_KEY)
    assert windows_platform.path_limit(read_value=on, probe=lambda: False) == PathLimit(259, True, "MAX_PATH (policy on, process manifest not longPathAware)")
    assert windows_platform.path_limit(read_value=lambda key, value: 0, probe=lambda: True) == PathLimit(259, False, windows_platform.LONG_PATHS_REGISTRY_KEY)
    unknown = windows_platform.path_limit(read_value=denied)
    assert unknown.max_characters == 259 and unknown.long_paths_enabled is None


def test_long_path_probe_exceeds_max_path_through_nested_components(tmp_path: Path) -> None:
    # NTFS caps a single component at 255 characters whatever the policy
    # says, so the probe reaches past MAX_PATH by nesting, never by one name.
    component = windows_platform.LONG_PATH_PROBE_COMPONENT
    assert len(component) < 255
    assert 3 * (len(component) + 1) > windows_platform.MAX_PATH_CHARACTERS
    result = windows_platform.long_paths_effective(tmp_path)
    assert result in (True, False)
    # Probe directories never survive, whichever way the attempt went.
    assert not (tmp_path / component).exists()
    if sys.platform != "win32":
        assert result is True
    elif windows_platform.long_paths_enabled() is True:
        # Policy on: the probe is what decides the effective limit.
        assert result == (windows_platform.path_limit().max_characters == windows_platform.LONG_PATH_CHARACTERS)
    else:
        # Policy off (or unreadable): nothing can create a path past MAX_PATH.
        assert result is False


def test_retry_waits_out_windows_sharing_violations_with_bounded_backoff() -> None:
    clock = _FakeClock()
    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] <= 3:
            raise _SharingViolation(13, "sharing violation")
        return "done"

    assert retry_file_operation(flaky, platform_id="windows", sleep=clock.sleep, clock=clock) == "done"
    assert attempts["count"] == 4
    assert clock.sleeps == [RETRY_INITIAL_SECONDS, RETRY_INITIAL_SECONDS * 2, RETRY_INITIAL_SECONDS * 4]

    # PermissionError counts as transient too; ERROR_DIR_NOT_EMPTY as well.
    assert 145 in RETRIED_WINERRORS and 5 in RETRIED_WINERRORS
    attempts["count"] = 0

    def denied_once() -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise PermissionError(13, "denied")

    retry_file_operation(denied_once, platform_id="windows", sleep=clock.sleep, clock=clock)
    assert attempts["count"] == 2

    # Errors that waiting cannot fix are raised at once, without sleeping.
    before = list(clock.sleeps)

    def missing() -> None:
        raise FileNotFoundError(2, "missing")

    with pytest.raises(FileNotFoundError):
        retry_file_operation(missing, platform_id="windows", sleep=clock.sleep, clock=clock)
    assert clock.sleeps == before

    # The budget is bounded: the delay saturates and the error propagates.
    clock = _FakeClock()

    def stuck() -> None:
        raise _SharingViolation(13, "still open")

    with pytest.raises(_SharingViolation):
        retry_file_operation(stuck, platform_id="windows", sleep=clock.sleep, clock=clock)
    assert max(clock.sleeps) == RETRY_MAXIMUM_SECONDS
    assert RETRY_BUDGET_SECONDS <= sum(clock.sleeps) <= RETRY_BUDGET_SECONDS + RETRY_MAXIMUM_SECONDS

    # POSIX never retries: the operation runs once and its error propagates.
    clock = _FakeClock()
    attempts["count"] = 0
    with pytest.raises(_SharingViolation):
        retry_file_operation(stuck, platform_id="darwin", sleep=clock.sleep, clock=clock)
    assert clock.sleeps == []


def test_remove_helpers_are_idempotent_and_report_what_they_removed(tmp_path, monkeypatch) -> None:
    target = tmp_path / "a.partial"
    assert remove_file(target) is False
    with pytest.raises(FileNotFoundError):
        remove_file(target, missing_ok=False)
    target.write_bytes(b"x")
    assert remove_file(target) is True and not target.exists()

    tree = tmp_path / "staging"
    assert remove_tree(tree) is False
    with pytest.raises(FileNotFoundError):
        remove_tree(tree, missing_ok=False)
    (tree / "nested").mkdir(parents=True)
    (tree / "nested" / "file").write_bytes(b"y")
    assert remove_tree(tree) is True and not tree.exists()

    # A tree another process still holds is retried as a whole on Windows and
    # given up quietly only when the caller asked for best effort.
    clock = _FakeClock()
    calls = {"count": 0}
    real_rmtree = shutil.rmtree

    def rmtree_locked(path, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise _SharingViolation(13, "directory in use")
        real_rmtree(path, *args, **kwargs)

    tree.mkdir()
    monkeypatch.setattr(shutil, "rmtree", rmtree_locked)
    assert remove_tree(tree, platform_id="windows", sleep=clock.sleep, clock=clock) is True
    assert calls["count"] == 3 and len(clock.sleeps) == 2 and not tree.exists()

    def rmtree_stuck(path, *args, **kwargs):
        raise _SharingViolation(13, "directory in use")

    tree.mkdir()
    monkeypatch.setattr(shutil, "rmtree", rmtree_stuck)
    clock = _FakeClock()
    assert remove_tree(tree, ignore_errors=True, platform_id="windows", sleep=clock.sleep, clock=clock) is False
    with pytest.raises(_SharingViolation):
        remove_tree(tree, platform_id="windows", sleep=clock.sleep, clock=clock)
    monkeypatch.setattr(shutil, "rmtree", real_rmtree)

    source = tmp_path / "receipt.json.tmp"
    destination = tmp_path / "receipt.json"
    destination.write_text("old", encoding="utf-8")
    source.write_text("new", encoding="utf-8")
    rename_with_retry(source, destination, replace=True)
    assert destination.read_text(encoding="utf-8") == "new" and not source.exists()
    source.write_text("moved", encoding="utf-8")
    rename_with_retry(source, tmp_path / "fresh.json")
    assert (tmp_path / "fresh.json").read_text(encoding="utf-8") == "moved"


def test_remove_file_outlives_a_handle_another_thread_holds(tmp_path) -> None:
    """A file that is open elsewhere is removed as soon as the handle closes.

    POSIX unlinks the name immediately; Windows refuses while the handle is
    open (sharing violation), so the helper must wait.  Both hosts end with
    the file gone; only Windows is expected to have slept.
    """

    target = tmp_path / "held.partial"
    target.write_bytes(b"held")
    opened = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with target.open("rb"):
            opened.set()
            release.wait(5.0)

    holder = threading.Thread(target=hold)
    holder.start()
    assert opened.wait(5.0)
    sleeps: list[float] = []

    def sleeping(seconds: float) -> None:
        sleeps.append(seconds)
        # Release the handle after the first wait so the retry can succeed.
        release.set()
        time.sleep(seconds)

    try:
        removed = remove_file(target, sleep=sleeping)
    finally:
        release.set()
        holder.join(5.0)
    assert removed is True
    assert not target.exists()
    if sys.platform == "win32":
        assert sleeps, "Windows must have waited for the holder to close the file"
    else:
        assert sleeps == []
