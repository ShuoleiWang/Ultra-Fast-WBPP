# Ultra-Fast WBPP app core

`openastroflow-app-core` is the compatibility-named OS-neutral control plane shared by the Ultra-Fast WBPP desktop GUI and processing workers. It contains no GUI toolkit, image kernels, catalog downloader, or subprocess policy.

Its stable responsibilities are:

- versioned controller/worker NDJSON messages;
- Project and Recipe manifests plus Project, Recipe, Stage, Artifact, and Run receipts;
- backend capability negotiation and hardware-profile selection;
- required astrometric-solver and drizzle final-result gates;
- portable safe relative paths and create-new directory publication;
- generated JSON Schema snapshots under `../../protocol/schema`.

The crate is `#![forbid(unsafe_code)]`. Platform-specific optimization stays behind worker capabilities. `portable-cpu` supplies the non-Windows portable baseline used by Linux x86-64 CI; Apple M-series hosts select `generic-arm64-cpu` or a positively probed `generic-apple-metal`, and only a positively identified M3 Pro with that executor selects `m3-pro-tuned`. Windows remains on the first-class `windows-cpu` profile. Filesystem operations use the `PlatformFs` trait so a future handle-based Windows adapter can strengthen platform behavior without changing persisted data.

## Typical controller flow

```rust
use openastroflow_app_core::{
    HardwareProfile, HostPlatform, RequiredResultGate, Validate,
};

# fn choose(
#   host: &HostPlatform,
#   capabilities: &openastroflow_app_core::BackendCapabilities,
#   recipe: &openastroflow_app_core::Recipe,
# ) -> Result<(), Box<dyn std::error::Error>> {
recipe.validate()?;
let profile = HardwareProfile::select(host, &capabilities.hardware_profiles)
    .ok_or("no compatible hardware profile")?;
capabilities.check_dispatch(recipe, profile, host)?;

// After execution, do not publish merely because the process exited zero.
# let stages = Vec::new();
# let artifacts = Vec::new();
let gate = RequiredResultGate::evaluate(recipe, &stages, &artifacts);
if !gate.is_ready() {
    return Err("required scientific result is missing".into());
}
# Ok(())
# }
```

The solver default is `required`. A final master therefore needs a successful solver stage and a validated WCS receipt that meets the recipe limits. If drizzle is required, its settings and provenance must also be present on every final master.

## Validation

```bash
cargo test -p openastroflow-app-core
cargo clippy -p openastroflow-app-core --all-targets -- -D warnings
cargo run -p openastroflow-app-core \
  --bin export-openastroflow-schemas -- protocol/schema
```

The test suite checks schema drift, both protocol-direction examples, message sequencing, profile fallback, result-gate failures, path traversal and Windows reserved names, source symlinks, identity drift, no-replace behavior, and completion-marker ordering.
