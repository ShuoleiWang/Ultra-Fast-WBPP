# Ultra-Fast WBPP control-plane protocol

This directory is the checked-in contract between the desktop controller and processing workers. Version 1 uses newline-delimited JSON (NDJSON), one UTF-8 JSON object per line. The Rust source of truth is `crates/app-core`; the files under `schema/` are generated snapshots.

## Transport and direction

A local worker uses two independent ordered streams:

- controller to worker (normally worker stdin): `handshake`, `plan`, `execute`;
- worker to controller (normally worker stdout): `handshake`, `progress`, `artifact`, `error`.

Each direction starts with a `handshake` at sequence `0`. Subsequent messages use contiguous sequence numbers and the same `sessionId`. Logs and diagnostics must go to stderr; stdout contains protocol records only. A record is limited to 8 MiB. JSON text containing a physical embedded newline, multiple records passed to the single-record decoder, unknown fields, unsupported protocol versions, non-finite numeric values, and invalid message invariants fail closed.

The framing examples are executable fixtures:

- `examples/controller-to-worker-v1.ndjson`
- `examples/worker-to-controller-v1.ndjson`

## Message lifecycle

```text
controller                                      worker
    |---- handshake(protocol versions) ----------->|
    |<--- handshake(capabilities + profiles) -------|
    |---- plan(project + recipe + input digest) --->|
    |---- execute(new output directory) ----------->|
    |<--- progress(stage + bounded fraction) --------|
    |<--- artifact(stage + identity receipt) --------|
    |<--- error(stable code + retryability), if any -|
```

`plan` is immutable and identity-bound by `planId` plus `inputManifestSha256`. `execute` names a new output directory; it does not authorize replacement of an existing path. `artifact` binds a content-addressed artifact receipt to the successful stage that created it. `error` is structured and machine-readable; user-facing detail belongs in `message` and optional `details`.

## Versioning

- `protocolVersion` changes only for wire-incompatible changes. A v1 decoder accepts exactly v1.
- Persisted Project, Recipe, capability, receipt, and publication documents each carry their own `schemaVersion`.
- Enum wire names and error/check codes are stable lowercase kebab-case identifiers.
- A sender advertises every protocol version it can speak during the handshake. No peer may silently downgrade a plan or receipt.
- Additive changes require a new protocol version because v1 deliberately rejects unknown fields. This makes an old GUI fail visibly instead of silently ignoring a scientific setting.

## Result safety gate

Process exit code zero does not make a run publishable. `RequiredResultGate` checks artifacts designated `final-master` against the recipe:

- a required solver result needs a successful `astrometric-solve` stage and a valid astrometry receipt embedded in every final master;
- the receipt must meet the recipe's projection, minimum matched-star count, and maximum RMS;
- required drizzle needs a successful `drizzle` stage and matching scale, drop-shrink, kernel, dimensions, and input-frame provenance on every final master;
- a missing final master, malformed artifact identity, missing WCS, or missing required drizzle provenance blocks publication.

The Rust gate is a structural receipt gate, not a FITS parser. The trusted Ultra-Fast WBPP controller must independently reopen the staged FITS, recompute file/WCS identities and scientific metrics, and only then construct the receipts passed to `NewDirectoryPublication::authorize`. Authorization additionally binds every successful enabled stage and every artifact path/hash/size to one exact publication plan; `publish` rejects a missing or stale authorization. Optional external solvers remain untrusted child processes and cannot authorize publication directly.

The default solver setting is `required`, so an end-to-end product recipe cannot report success while leaving the user with an unsolved master. `best-effort` produces visible failed checks but does not block; it must be an explicit recipe choice.

## Hardware profiles

Workers advertise supported profiles and features independently:

- `portable-cpu`: non-Windows portable CPU path, used by x86-64 Linux development CI;
- `generic-arm64-cpu`: AArch64 CPU path, including every Apple M-series chip;
- `generic-apple-metal`: portable Metal path for every Apple M-series chip;
- `m3-pro-tuned`: M3 Pro-only tuned scheduling/kernels, selected only after positive chip detection;
- `windows-cpu`: Windows CPU contract and the extension point for the Windows worker.

Unknown or future Apple chips fall back to generic Metal, then generic Arm64 CPU. They never enter the M3 Pro tuned path based on architecture alone.

One handshake reports only profiles executable on that host: Windows workers advertise `windows-cpu`, Apple arm64 workers advertise `generic-arm64-cpu` plus only positively probed Metal profiles, and Linux x86-64 workers advertise `portable-cpu`. A missing dylib, unavailable Metal device, or failed executor construction therefore produces a valid CPU-only handshake rather than a false accelerator claim or a failed CPU probe.

## Filesystem publication contract

Serialized artifact paths use forward-slash `SafeRelativePath` values. Absolute paths, `..`, empty components, backslashes, control characters, Windows device names, trailing dots/spaces, and cross-platform-invalid characters are rejected.

`NewDirectoryPublication` performs identity validation before claiming a destination with create-new directory semantics. It never truncates or replaces an existing path. Files are copied with create-new semantics and re-hashed. The final `.openastroflow-complete.json` receipt is written last. If anything fails after the destination claim, the incomplete directory is preserved for audit and is not consumable because it lacks the completion receipt.

The pure-Rust standard adapter protects against accidental collisions and cooperating concurrent workers. It rejects staging symlinks and requires immutable staging, but it is not presented as a security boundary against a hostile local process racing filesystem operations. Platform packages may provide stronger `PlatformFs` implementations without changing the control-plane contract.

## Regenerating schemas

From the repository root:

```bash
cargo run --manifest-path crates/app-core/Cargo.toml \
  --bin export-openastroflow-schemas -- protocol/schema
```

`cargo test --manifest-path crates/app-core/Cargo.toml` verifies the protocol examples and that every checked-in schema exactly matches the Rust model.
