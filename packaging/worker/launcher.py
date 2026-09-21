"""Entry point for the frozen worker and one-shot controller helpers."""

import json
import multiprocessing
import platform
import sys


def _utf8_stdio() -> None:
    # The protocol with the desktop shell is UTF-8 in both directions; a
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
    # A no-argument launch retains the long-running externalBin behavior.  GUI
    # plan construction uses the same signed/frozen binary with
    # ``controller-plan ...`` and therefore cannot drift from worker semantics.
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
    from openastroflow_engine.cli import main as cli_main

    if len(sys.argv) == 1:
        from openastroflow_engine.worker import main as worker_main

        return worker_main()
    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
