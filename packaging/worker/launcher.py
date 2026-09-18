"""Entry point for the frozen worker and one-shot controller helpers."""

import json
import multiprocessing
import platform
import sys


def main() -> int:
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
