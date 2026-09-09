//! Ultra-Fast WBPP desktop controller.
//!
//! The webview never performs scientific work and never fabricates completion.
//! A signed sidecar owns inventory and pixel execution; this controller owns
//! protocol ordering, process lifetime, filesystem identity checks, and the
//! app-core final-result gate.

use std::sync::Arc;

use tauri::{AppHandle, Manager, RunEvent, Runtime, State};

mod catalog;
mod platform;
mod project;
mod sidecar;

use sidecar::{InspectRequest, InspectResponse, PipelineRegistry, RunReceipt, RunRequest};

#[tauri::command]
async fn catalog_list(app: AppHandle) -> Result<serde_json::Value, String> {
    tauri::async_runtime::spawn_blocking(move || catalog::list(&app))
        .await
        .map_err(|error| format!("catalog list task failed: {error}"))?
}

#[tauri::command]
async fn catalog_doctor(app: AppHandle) -> Result<serde_json::Value, String> {
    tauri::async_runtime::spawn_blocking(move || catalog::doctor(&app))
        .await
        .map_err(|error| format!("catalog doctor task failed: {error}"))?
}

#[tauri::command]
async fn solver_doctor(app: AppHandle) -> Result<serde_json::Value, String> {
    tauri::async_runtime::spawn_blocking(move || catalog::solver_doctor(&app))
        .await
        .map_err(|error| format!("solver doctor task failed: {error}"))?
}

#[tauri::command]
async fn catalog_verify(
    app: AppHandle,
    catalog_id: String,
    configure: bool,
) -> Result<serde_json::Value, String> {
    tauri::async_runtime::spawn_blocking(move || catalog::verify(&app, &catalog_id, configure))
        .await
        .map_err(|error| format!("catalog verify task failed: {error}"))?
}

#[tauri::command]
async fn start_catalog_install(
    app: AppHandle,
    registry: State<'_, Arc<catalog::CatalogRegistry>>,
    request: catalog::CatalogInstallRequest,
) -> Result<catalog::CatalogJobReceipt, String> {
    let registry = registry.inner().clone();
    tauri::async_runtime::spawn_blocking(move || catalog::start_install(app, registry, request))
        .await
        .map_err(|error| format!("catalog install launch task failed: {error}"))?
}

#[tauri::command]
async fn cancel_catalog_install(
    registry: State<'_, Arc<catalog::CatalogRegistry>>,
    job_id: String,
) -> Result<(), String> {
    let registry = registry.inner().clone();
    tauri::async_runtime::spawn_blocking(move || catalog::cancel_install(&registry, &job_id))
        .await
        .map_err(|error| format!("catalog cancellation task failed: {error}"))?
}

#[tauri::command]
async fn start_project(
    app: AppHandle,
    registry: State<'_, Arc<project::ProjectRegistry>>,
    request: project::ProjectRunRequest,
) -> Result<project::ProjectRunReceipt, String> {
    let registry = registry.inner().clone();
    tauri::async_runtime::spawn_blocking(move || project::start(app, registry, request))
        .await
        .map_err(|error| format!("project launch task failed: {error}"))?
}

#[tauri::command]
async fn cancel_project(
    registry: State<'_, Arc<project::ProjectRegistry>>,
    job_id: String,
) -> Result<(), String> {
    let registry = registry.inner().clone();
    tauri::async_runtime::spawn_blocking(move || project::cancel(&registry, &job_id))
        .await
        .map_err(|error| format!("project cancellation task failed: {error}"))?
}

#[tauri::command]
async fn get_capabilities(app: AppHandle) -> Result<sidecar::RuntimeCapabilities, String> {
    tauri::async_runtime::spawn_blocking(move || sidecar::get_capabilities(&app))
        .await
        .map_err(|error| format!("capability probe task failed: {error}"))
}

#[tauri::command]
async fn inspect_paths(app: AppHandle, request: InspectRequest) -> Result<InspectResponse, String> {
    tauri::async_runtime::spawn_blocking(move || sidecar::inspect_paths(&app, request))
        .await
        .map_err(|error| format!("inventory task failed: {error}"))?
}

#[tauri::command]
async fn inspect_calibration(
    app: AppHandle,
    request: sidecar::InspectCalibrationRequest,
) -> Result<sidecar::CalibrationInspection, String> {
    tauri::async_runtime::spawn_blocking(move || sidecar::inspect_calibration(&app, request))
        .await
        .map_err(|error| format!("calibration inspection task failed: {error}"))?
}

#[tauri::command]
async fn inspect_quality(
    app: AppHandle,
    request: sidecar::InspectQualityRequest,
) -> Result<sidecar::QualityInspection, String> {
    tauri::async_runtime::spawn_blocking(move || sidecar::inspect_quality(&app, request))
        .await
        .map_err(|error| format!("quality inspection task failed: {error}"))?
}

#[tauri::command]
async fn hash_sources(
    request: sidecar::HashSourcesRequest,
) -> Result<sidecar::HashSourcesResponse, String> {
    tauri::async_runtime::spawn_blocking(move || sidecar::hash_sources(request))
        .await
        .map_err(|error| format!("source hashing task failed: {error}"))?
}

#[tauri::command]
async fn start_pipeline(
    app: AppHandle,
    registry: State<'_, Arc<PipelineRegistry>>,
    request: RunRequest,
) -> Result<RunReceipt, String> {
    let registry = registry.inner().clone();
    tauri::async_runtime::spawn_blocking(move || sidecar::start_pipeline(app, registry, request))
        .await
        .map_err(|error| format!("pipeline launch task failed: {error}"))?
}

#[tauri::command]
async fn cancel_pipeline(
    registry: State<'_, Arc<PipelineRegistry>>,
    job_id: String,
) -> Result<(), String> {
    let registry = registry.inner().clone();
    tauri::async_runtime::spawn_blocking(move || sidecar::cancel_pipeline(&registry, &job_id))
        .await
        .map_err(|error| format!("pipeline cancellation task failed: {error}"))?
}

fn terminate_managed_children<R: Runtime>(app: &AppHandle<R>) -> Result<(), String> {
    let mut failures = Vec::new();
    if let Err(error) = project::terminate_all(
        app.state::<Arc<project::ProjectRegistry>>()
            .inner()
            .as_ref(),
    ) {
        failures.push(error);
    }
    if let Err(error) = catalog::terminate_all(
        app.state::<Arc<catalog::CatalogRegistry>>()
            .inner()
            .as_ref(),
    ) {
        failures.push(error);
    }
    if let Err(error) =
        sidecar::terminate_all(app.state::<Arc<PipelineRegistry>>().inner().as_ref())
    {
        failures.push(error);
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; "))
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .manage(Arc::new(PipelineRegistry::default()))
        .manage(Arc::new(catalog::CatalogRegistry::default()))
        .manage(Arc::new(project::ProjectRegistry::default()))
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            get_capabilities,
            catalog_list,
            catalog_doctor,
            solver_doctor,
            catalog_verify,
            start_catalog_install,
            cancel_catalog_install,
            start_project,
            cancel_project,
            inspect_paths,
            inspect_calibration,
            inspect_quality,
            hash_sources,
            start_pipeline,
            cancel_pipeline
        ])
        .build(tauri::generate_context!())
        .expect("error while building Ultra-Fast WBPP desktop application");
    let mut children_terminated = false;
    app.run(move |app_handle, event| {
        if !children_terminated && matches!(event, RunEvent::ExitRequested { .. } | RunEvent::Exit)
        {
            match terminate_managed_children(app_handle) {
                Ok(()) => children_terminated = true,
                Err(error) => eprintln!("failed to terminate one or more child processes: {error}"),
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn managed_shutdown_is_idempotent_when_no_jobs_are_running() {
        let app = tauri::test::mock_app();
        assert!(app.manage(Arc::new(PipelineRegistry::default())));
        assert!(app.manage(Arc::new(catalog::CatalogRegistry::default())));
        assert!(app.manage(Arc::new(project::ProjectRegistry::default())));
        terminate_managed_children(app.handle()).expect("first shutdown");
        terminate_managed_children(app.handle()).expect("second shutdown");
    }

    #[test]
    fn provider_url_capability_is_https_and_host_scoped() {
        let capability: serde_json::Value =
            serde_json::from_str(include_str!("../capabilities/default.json"))
                .expect("desktop capability JSON");
        let permissions = capability["permissions"]
            .as_array()
            .expect("permissions array");
        let opener = permissions
            .iter()
            .find(|value| value["identifier"] == "opener:allow-open-url")
            .expect("scoped opener permission");
        let urls = opener["allow"]
            .as_array()
            .expect("opener allow scope")
            .iter()
            .map(|value| value["url"].as_str().expect("URL pattern"))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            urls,
            std::collections::BTreeSet::from([
                "https://astrometry.net/*",
                "https://www.hnsky.org/*",
            ])
        );
    }
}
