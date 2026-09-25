"""``removelines`` and ``uniformize`` for a ``solve-field`` without Python.

Astrometry.net's ``solve-field`` 0.97 filters the image2xy source list through
two helpers that upstream ships as Python scripts (numpy + fitsio).  A
self-contained desktop build ships ``solve-field`` without a Python
installation, so wrapper scripts beside it call the frozen engine instead
(``__astrometry-helper-v1``, dispatched by the launcher before any engine
import).  The numerics are upstream's ``astrometry.util.removelines`` and
``astrometry.util.uniformize`` line for line; only the table I/O differs:
rows are selected as raw bytes and the input headers are kept with
``NAXIS2`` updated, so the sources ``solve-field`` reads back are
bit-identical to upstream's.

The ported functions are Copyright (c) 2006-2015, Astrometry.net Developers,
under the 3-clause BSD license in LICENSES/astrometry-net-BSD-3-Clause.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import numpy as np


_BLOCK = 2880
_CARD = 80
_TFORM = re.compile(r"^\s*(\d*)([LXBIJKAEDCMPQ])")
_WIDTHS = {"L": 1, "B": 1, "I": 2, "J": 4, "K": 8, "A": 1, "E": 4, "D": 8, "C": 8, "M": 16, "P": 8, "Q": 16}
_NUMERIC = {"B": "u1", "I": ">i2", "J": ">i4", "K": ">i8", "E": ">f4", "D": ">f8"}


class HelperError(RuntimeError):
    """The input is not a plain binary-table xylist this helper can filter."""


def _padded(size: int) -> int:
    return -(-size // _BLOCK) * _BLOCK


def _header(data: bytes, offset: int) -> tuple[list[bytes], int]:
    cards: list[bytes] = []
    while True:
        block = data[offset : offset + _BLOCK]
        if len(block) != _BLOCK:
            raise HelperError("truncated FITS header")
        offset += _BLOCK
        for start in range(0, _BLOCK, _CARD):
            card = block[start : start + _CARD]
            cards.append(card)
            if card[:8] == b"END     ":
                return cards, offset


def _values(cards: Sequence[bytes]) -> dict[str, str]:
    values: dict[str, str] = {}
    for card in cards:
        keyword = card[:8].decode("ascii", "replace").strip()
        if card[8:10] != b"= " or keyword in values:
            continue
        text = card[10:].decode("ascii", "replace")
        quoted = re.match(r"\s*'((?:[^']|'')*)'", text)
        values[keyword] = quoted.group(1).replace("''", "'").rstrip() if quoted else text.split("/", 1)[0].strip()
    return values


def _data_size(values: dict[str, str]) -> int:
    axes = int(values.get("NAXIS", "0"))
    if axes == 0:
        return 0
    count = 1
    for axis in range(1, axes + 1):
        count *= int(values[f"NAXIS{axis}"])
    groups = int(values.get("GCOUNT", "1"))
    heap = int(values.get("PCOUNT", "0"))
    return abs(int(values["BITPIX"])) // 8 * groups * (heap + count)


class _Table:
    """The first extension of an xylist, filtered row by row as raw bytes."""

    def __init__(self, path: str | Path) -> None:
        data = Path(path).read_bytes()
        cards, offset = _header(data, 0)
        offset += _padded(_data_size(_values(cards)))
        self._primary = data[:offset]
        start = offset
        cards, offset = _header(data, offset)
        values = _values(cards)
        if values.get("XTENSION") != "BINTABLE":
            raise HelperError("HDU 1 is not a binary table")
        if int(values.get("PCOUNT", "0")) != 0:
            raise HelperError("tables with a heap are not supported")
        self._header = data[start:offset]
        self._rows_card = next(index for index, card in enumerate(cards) if card[:8] == b"NAXIS2  ")
        self._width = int(values["NAXIS1"])
        self._values = values
        self._columns: dict[str, tuple[int, str, int, int]] = {}
        position = 0
        for index in range(1, int(values["TFIELDS"]) + 1):
            form = _TFORM.match(values.get(f"TFORM{index}", ""))
            if form is None:
                raise HelperError(f"unsupported TFORM{index}")
            repeat = int(form.group(1) or 1)
            code = form.group(2)
            self._columns[values.get(f"TTYPE{index}", "")] = (position, code, repeat, index)
            position += (repeat + 7) // 8 if code == "X" else repeat * _WIDTHS[code]
        if position != self._width:
            raise HelperError("table columns do not fill a row")
        rows = int(values["NAXIS2"])
        body = data[offset : offset + self._width * rows]
        if len(body) != self._width * rows:
            raise HelperError("truncated table data")
        self.rows = np.frombuffer(body, dtype=np.dtype((np.void, self._width)))

    def __len__(self) -> int:
        return len(self.rows)

    def get(self, name: str) -> np.ndarray:
        column = self._columns.get(name)
        if column is None:
            matches = [value for key, value in self._columns.items() if key.lower() == name.lower()]
            column = matches[0] if len(matches) == 1 else None
        if column is None:
            raise HelperError(f"no column {name!r}")
        position, code, repeat, index = column
        if code not in _NUMERIC or repeat != 1 or f"TSCAL{index}" in self._values or f"TZERO{index}" in self._values:
            raise HelperError(f"column {name!r} is not an unscaled numeric scalar")
        stored = np.dtype(_NUMERIC[code])
        view = np.dtype({"names": ["value"], "formats": [stored], "offsets": [position], "itemsize": self._width})
        # fitsio hands upstream a native-endian copy of the column.
        return self.rows.view(view)["value"].astype(stored.newbyteorder("="))

    def cut(self, index: Any) -> None:
        self.rows = self.rows[index]

    def writeto(self, path: str | Path) -> None:
        start = self._rows_card * _CARD
        card = self._header[start : start + _CARD]
        if not card[10:30].strip().isdigit() or card[30:31] not in (b" ", b"/"):
            raise HelperError("NAXIS2 is not a fixed-format integer card")
        header = (
            self._header[: start + 10]
            + f"{len(self.rows):>20d}".encode("ascii")
            + self._header[start + 30 :]
        )
        body = self.rows.tobytes()
        with open(path, "wb") as stream:
            stream.write(self._primary)
            stream.write(header)
            stream.write(body)
            stream.write(b"\0" * (_padded(len(body)) - len(body)))


# Upstream astrometry.util.removelines, unchanged apart from the table I/O.
def hist_remove_lines(x, binwidth, binoffset, logcut):
    bins = -binoffset + np.arange(0, max(x) + binwidth + 1, binwidth)
    (counts, thebins) = np.histogram(x, bins)

    # We're ignoring empty bins.
    occupied = np.nonzero(counts > 1)[0]
    noccupied = len(occupied)
    if noccupied == 0:
        return np.array([True] * len(x))
    k = counts[occupied] - 1
    mean = sum(k) / float(noccupied)
    logpoisson = k * np.log(mean) - mean - np.array([sum(np.arange(kk)) for kk in k])
    badbins = occupied[logpoisson < logcut]
    if len(badbins) == 0:
        return np.array([True] * len(x))

    badleft = bins[badbins]
    badright = badleft + binwidth

    badpoints = sum(np.array([(x >= L) * (x < R) for (L, R) in zip(badleft, badright)]), 0)
    return badpoints == 0


def removelines(infile, outfile, xcol="X", ycol="Y", cut=None):
    if cut is None:
        cut = 100
    T = _Table(infile)
    if len(T) == 0:
        print("removelines.py: Input file contains no sources.")
        T.writeto(outfile)
        return 0

    ix = hist_remove_lines(T.get(xcol), 1, 0.5, logcut=-cut)
    iy = hist_remove_lines(T.get(ycol), 1, 0.5, logcut=-cut)
    Norig = len(T)
    T.cut(ix * iy)
    print("removelines.py: Removed %i sources" % (Norig - len(T)))
    T.writeto(outfile)
    return 0


# Upstream astrometry.util.uniformize, unchanged apart from the table I/O.
def uniformize(infile, outfile, n, xcol="X", ycol="Y"):
    T = _Table(infile)
    if len(T) == 0:
        print("No sources")
        T.writeto(outfile)
        return 0
    x = T.get(xcol)
    y = T.get(ycol)
    I = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not all(I):
        print("%i source positions are not finite." % np.sum(np.logical_not(I)))
        x = x[I]
        y = y[I]
        T.cut(I)

    W = max(x) - min(x)
    H = max(y) - min(y)
    if W == 0 or H == 0:
        print("Area of the rectangle enclosing all image sources: %i x %i" % (W, H))
        T.writeto(outfile)
        return 0
    NX = int(max(1, np.round(W / np.sqrt(W * H / float(n)))))
    NY = int(max(1, np.round(n / float(NX))))
    print("Uniformizing into %i x %i bins" % (NX, NY))
    print("Image bounds: x [%g,%g], y [%g,%g]" % (min(x), max(x), min(y), max(y)))

    ix = (np.clip(np.floor((x - min(x)) / float(W) * NX), 0, NX - 1)).astype(int)
    iy = (np.clip(np.floor((y - min(y)) / float(H) * NY), 0, NY - 1)).astype(int)
    I = iy * NX + ix
    if not (np.all(ix >= 0) and np.all(ix < NX) and np.all(iy >= 0) and np.all(iy < NY)):
        raise HelperError("source bin outside the grid")
    bins = [[] for i in range(NX * NY)]
    for j, i in enumerate(I):
        bins[int(i)].append(j)
    maxlen = max([len(b) for b in bins])
    J = []
    for i in range(maxlen):
        thisrow = []
        for b in bins:
            if i >= len(b):
                continue
            thisrow.append(b[i])
        thisrow.sort()
        J += thisrow
    J = np.array(J)
    T.cut(J)
    T.writeto(outfile)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in {"removelines", "uniformize"}:
        sys.stderr.write("usage: __astrometry-helper-v1 {removelines|uniformize} [options] <in-xylist> <out-xylist>\n")
        return 2
    tool = arguments[0]
    parser = argparse.ArgumentParser(prog=tool)
    parser.add_argument("-X", dest="xcol", default="X")
    parser.add_argument("-Y", dest="ycol", default="Y")
    # Upstream accepts an extension number and always reads HDU 1.
    parser.add_argument("-e", dest="ext", type=int, default=1)
    if tool == "removelines":
        parser.add_argument("-s", dest="cut", type=float, default=None)
    else:
        parser.add_argument("-n", dest="n", type=int, default=10)
    parser.add_argument("infile")
    parser.add_argument("outfile")
    options = parser.parse_args(arguments[1:])
    try:
        if tool == "removelines":
            return removelines(options.infile, options.outfile, xcol=options.xcol, ycol=options.ycol, cut=options.cut)
        return uniformize(options.infile, options.outfile, options.n, xcol=options.xcol, ycol=options.ycol)
    except (HelperError, OSError, KeyError, ValueError, StopIteration) as error:
        sys.stderr.write(f"{tool}: {error}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
