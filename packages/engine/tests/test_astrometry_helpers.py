from __future__ import annotations

from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

from ufwbpp.solvers.astrometry_helpers import main


def _xylist(path: Path, x: list[float], y: list[float]) -> fits.HDUList:
    primary = fits.PrimaryHDU()
    primary.header["SRCEXT"] = 1
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="X", format="E", array=np.asarray(x, dtype=np.float32)),
            fits.Column(name="Y", format="E", array=np.asarray(y, dtype=np.float32)),
            fits.Column(name="FLUX", format="E", array=np.arange(len(x), dtype=np.float32)),
            fits.Column(name="BACKGROUND", format="E", array=np.zeros(len(x), dtype=np.float32)),
        ]
    )
    table.header["IMAGEW"] = 1000
    table.header["IMAGEH"] = 800
    hdus = fits.HDUList([primary, table])
    hdus.writeto(path)
    return fits.open(path)


def test_removelines_drops_a_column_of_sources_and_keeps_rows_and_headers(tmp_path: Path) -> None:
    generator = np.random.default_rng(3)
    x = list(generator.uniform(0, 1000, 200)) + [500.0] * 60
    y = list(generator.uniform(0, 800, 200)) + list(np.linspace(0, 800, 60))
    source = _xylist(tmp_path / "in.xyls", x, y)
    assert main(["removelines", str(tmp_path / "in.xyls"), str(tmp_path / "out.xyls")]) == 0
    with fits.open(tmp_path / "out.xyls") as output:
        kept = source[1].data["X"] != np.float32(500.0)
        assert output[1].data.tobytes() == source[1].data[kept].tobytes()
        assert output[1].header["NAXIS2"] == int(kept.sum()) == 200
        assert output[1].header["IMAGEW"] == 1000 and output[0].header["SRCEXT"] == 1


def test_uniformize_orders_rows_by_bin_round_and_drops_non_finite_positions(tmp_path: Path) -> None:
    x = [0, 1, 10, 0, 10, 9, np.nan]
    y = [0, 1, 0, 10, 10, 9, 5]
    source = _xylist(tmp_path / "in.xyls", x, y)
    assert main(["uniformize", "-n", "4", str(tmp_path / "in.xyls"), str(tmp_path / "out.xyls")]) == 0
    with fits.open(tmp_path / "out.xyls") as output:
        # 2x2 bins [0, 1], [2], [3], [4, 5]: first each bin's first source, then the rest.
        assert output[1].data.tobytes() == source[1].data[[0, 2, 3, 4, 1, 5]].tobytes()


def test_helper_rejects_a_table_it_cannot_filter_exactly(tmp_path: Path) -> None:
    fits.HDUList(
        [fits.PrimaryHDU(), fits.BinTableHDU.from_columns([fits.Column(name="X", format="2E", array=np.zeros((3, 2)))])]
    ).writeto(tmp_path / "in.xyls")
    assert main(["uniformize", str(tmp_path / "in.xyls"), str(tmp_path / "out.xyls")]) == 1
    assert main(["unknown"]) == 2
    with pytest.raises(SystemExit):
        main(["removelines", str(tmp_path / "in.xyls")])
