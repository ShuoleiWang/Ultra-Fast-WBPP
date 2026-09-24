from .cli import main

# The guard matters: a spawned measurement worker re-imports this module as
# ``__mp_main__`` and must not run the command line again.
if __name__ == "__main__":
    raise SystemExit(main())
