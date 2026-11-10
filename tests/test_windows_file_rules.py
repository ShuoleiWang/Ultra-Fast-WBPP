"""The repository conftest holds macOS and Linux runs to Windows' file rules."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

from astropy.io import fits
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Windows enforces these rules itself")


def _frame(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(path, np.arange(64, dtype=np.float32).reshape(8, 8))
    return path


def test_open_file_blocks_rename_delete_and_directory_moves(tmp_path: Path) -> None:
    source = _frame(tmp_path / "run" / "light.fits")
    with source.open("rb"):
        with pytest.raises(PermissionError, match="still open"):
            source.rename(tmp_path / "moved.fits")
        with pytest.raises(PermissionError, match="still open"):
            source.unlink()
        with pytest.raises(PermissionError, match="still open"):
            os.replace(_frame(tmp_path / "other.fits"), source)
        with pytest.raises(PermissionError, match="still open"):
            os.rename(tmp_path / "run", tmp_path / "published")
        with pytest.raises(PermissionError, match="still open"):
            shutil.rmtree(tmp_path / "run")
    os.rename(tmp_path / "run", tmp_path / "published")
    assert (tmp_path / "published" / "light.fits").is_file()


def test_memory_mapped_pixels_block_until_they_are_released(tmp_path: Path) -> None:
    source = _frame(tmp_path / "light.fits")
    pixels = fits.getdata(source)  # memory-mapped by default; the file itself is closed
    with pytest.raises(PermissionError, match="memory-mapped at tests/test_windows_file_rules.py"):
        source.unlink()
    with pytest.raises(PermissionError, match="memory-mapped"):
        source.open("wb")
    del pixels
    source.unlink()

    # A temporary file in the same directory is no truncation of the mapped
    # file: Linux creates it by opening the directory itself for writing.
    mapped = fits.getdata(_frame(tmp_path / "mapped.fits"))
    with tempfile.TemporaryFile(dir=tmp_path) as scratch:
        scratch.write(b"scratch")
    del mapped

    copied = fits.getdata(_frame(tmp_path / "copied.fits"), memmap=False)
    (tmp_path / "copied.fits").unlink()
    assert copied.shape == (8, 8)


def test_engine_directory_publication_is_held_to_the_same_rule(tmp_path: Path) -> None:
    from ufwbpp import platform as platform_services

    staging = tmp_path / "result.staging"
    master = _frame(staging / "L.fits")
    with fits.open(master) as hdul:
        with pytest.raises(PermissionError, match="still open"):
            platform_services.current().rename_directory_no_replace(staging, tmp_path / "result")
        assert hdul[0].header["NAXIS"] == 2
    platform_services.current().rename_directory_no_replace(staging, tmp_path / "result")
    assert (tmp_path / "result" / "L.fits").is_file()


def test_rename_onto_an_existing_path_fails_as_on_windows(tmp_path: Path) -> None:
    first, second = _frame(tmp_path / "a.fits"), _frame(tmp_path / "b.fits")
    with pytest.raises(FileExistsError):
        os.rename(first, second)
    os.replace(first, second)
    assert not first.exists()
