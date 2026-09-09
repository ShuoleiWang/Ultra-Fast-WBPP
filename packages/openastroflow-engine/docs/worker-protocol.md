# Canonical NDJSON worker protocol v1

The Python sidecar implements the Rust `app-core` schema in
`protocol/schema/worker-envelope-v1.schema.json`; it has no legacy request
format. stdin is the controller stream (`handshake`, `plan`, `execute`) and
stdout is the worker stream (`handshake`, `progress`, `artifact`, `error`). Each
stream starts at sequence `0`, stays in one `sessionId`, and increments without
gaps. Logs never appear on stdout.

## Product lifecycle

1. Start `openastroflow-worker` (or `openastroflow-engine worker`).
2. Send a controller handshake at sequence `0`; read the worker capability
   handshake at sequence `0`.
3. Generate a role-bound plan with the same binary:

   ```bash
   openastroflow-engine controller-plan \
     /data/nina/lights /data/calibration \
     --mode ordinary --session-id session-1 --output plan.json
   ```

4. Send that plan as controller sequence `1`. Successful storage is silent;
   invalid role ownership, a changed inventory digest, an unavailable hardware
   profile/backend, or an unmappable recipe emits `error`.
5. Send `execute` at sequence `2` with the same request/plan IDs, a new run ID,
   an existing output parent, and a portable directory name.
6. Consume progress and artifacts until the run succeeds or emits an error.

The plan's `inputManifestSha256` is the raw 64-hex digest returned by
`inventory_manifest_sha256()`. It binds the controller scan to the worker's
header/stat inventory. E2E execution additionally hashes all source content and
rechecks identities before atomic publication.

## Artifacts and stage receipts

On success the worker emits real diagnostic artifacts for enabled QC,
calibration, registration, integration, and (when requested) Drizzle stages.
Each artifact message carries its successful `StageReceipt`. Final output is one
`final-master` per actual filter—not a fabricated RGB product—with:

- the final FITS SHA-256 and size;
- a freshly recomputed canonical WCS digest;
- catalog correspondence count, RMS, parity, catalog/index identities;
- Drizzle scale, pixfrac/drop-shrink, kernel, input count, and geometry when
  Drizzle was required.

The worker validates each final artifact against the canonical recipe before it
encodes the protocol record. Missing solver evidence, missing Drizzle evidence,
unsupported recipe stages (including RGB/mosaic in this executor), source drift,
or an existing destination fails closed.

The authoritative framing examples are
`protocol/examples/controller-to-worker-v1.ndjson` and
`protocol/examples/worker-to-controller-v1.ndjson`.
