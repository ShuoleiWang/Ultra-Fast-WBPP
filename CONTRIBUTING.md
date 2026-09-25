# Contributing

Bug reports, reproducible scientific comparisons, documentation and focused fixes are welcome.

[`AGENTS.md`](AGENTS.md) is the full working guide for this repository (repository map, every command, the rules that are not negotiable, the gotchas); coding agents such as OpenAI Codex read it automatically and Claude Code reads the shorter [`CLAUDE.md`](CLAUDE.md). The documentation map is [`docs/README.md`](docs/README.md) and the feature and evidence overview is [`docs/features.md`](docs/features.md).

## Development setup

Use Python 3.11 or 3.12, Node.js 22, Rust 1.88 or newer, and CMake 3.28 or newer. Desktop development also needs the platform's [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/). On macOS, install Xcode Command Line Tools; Metal builds require the Metal tools provided by Xcode.

From a fresh checkout on macOS or Linux:

```bash
make bootstrap
make desktop-dev
```

`make demo` opens the browser UI preview without the native engine. On Windows, create `.venv` with `py -3.12 -m venv .venv`, then use `.venv\Scripts\python.exe` for the pip and pytest commands in the Makefile. See the [Windows support limits](docs/windows.md) before working on that target. Development dependency ranges are intentionally resolved by pip; they are not a reproducible binary-release lock.

## Before opening a change

- Discuss new scientific algorithms or protocol changes in an issue first.
- Never commit personal acquisition data, absolute home-directory paths, API keys, signing material, proprietary application code, or third-party catalogs.
- Tiny fixtures must be synthetic or explicitly redistributable and live under a `tests/fixtures` directory with provenance.
- Keep source data read-only. Tests that write must use a new temporary output directory and verify no-replace behavior.

## Scientific change requirements

Every scientific change needs a stated mathematical contract, CPU reference behavior, boundary/error tests, deterministic fixture evidence, and a declared numerical tolerance. Accelerator changes also need a differential comparison against the CPU reference. Performance results never substitute for scientific validation.

Plate-solving changes must preserve the distinction between `SEED` and `SOLVED`. Drizzle changes must test science, weight, context/coverage, null-pixel, and WCS behavior.

## Local checks

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
.venv/bin/python -m pytest -q packages/light-frame-qc/tests packages/registration/tests packages/engine/tests tests
npm --prefix apps/desktop test
npm --prefix apps/desktop run build
cmake -S native -B build/native -DUFWBPP_BUILD_TESTS=ON
cmake --build build/native --parallel
ctest --test-dir build/native --output-on-failure
make source-check
```

Run focused tests while developing; `make check` runs the full local gate before review. The `build/native` configuration above is the unoptimized test build; for real runs install the Release kernels with `make native-release-install` and never install `build/native`. Hardware-dependent skips are not hardware acceptance. For numerical changes, report fixture size, backend, tolerance, and wall time; keep benchmarks separate from correctness tests. Do not attach private receipts with acquisition paths or source identities to a public pull request.

Use Conventional Commit-style subjects when practical. By contributing, you agree that your contribution is licensed under MIT.
