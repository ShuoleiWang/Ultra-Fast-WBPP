"""Entry point of the frozen engine: the ``ultra-fast-wbpp`` command line."""

import json
import multiprocessing
import platform
import sys


def _utf8_stdio() -> None:
    # The desktop shell exchanges UTF-8 JSON with the engine in both directions; a
    # Windows console (code page 936 on a Chinese system) would otherwise
    # encode a degree sign or a CJK path as mojibake or fail outright.  The
    # engine's ``platform`` layer does the same for the source CLI; this copy
    # runs before any engine import so the runtime smoke stays import-free.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (LookupError, OSError, ValueError):
            pass


def main() -> int:
    _utf8_stdio()
    # Quality-gate measurement spawns worker processes from this same frozen
    # executable; a child carries ``--multiprocessing-fork`` and must run its
    # task loop here instead of the command line.
    multiprocessing.freeze_support()
    if sys.argv[1:] == ["__packaging-runtime-smoke-v1"]:
        import ctypes
        import decimal
        import ssl

        if ctypes.sizeof(ctypes.c_void_p) != 8:
            raise RuntimeError("packaged runtime does not expose a 64-bit pointer ABI")
        print(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "python": platform.python_version(),
                    "openssl": ssl.OPENSSL_VERSION,
                    "libmpdec": decimal.__libmpdec_version__,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    if sys.argv[1:2] == ["__astrometry-helper-v1"]:
        # solve-field's removelines/uniformize in a self-contained build
        # (scripts/stage_astrometry_runtime.py); imports numpy only.
        from ufwbpp.solvers.astrometry_helpers import main as helper_main

        return helper_main(sys.argv[2:])
    from ufwbpp.cli import main as cli_main

    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
