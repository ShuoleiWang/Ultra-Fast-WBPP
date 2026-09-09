fn main() {
    // Release bundles must contain the attested onedir worker resource declared
    // in tauri.conf.json. Debug builds intentionally use the controlled .venv
    // discovery seam, so source tests do not need a frozen scientific runtime.
    if std::env::var("PROFILE").as_deref() != Ok("release")
        && std::env::var_os("TAURI_CONFIG").is_none()
    {
        std::env::set_var("TAURI_CONFIG", r#"{"bundle":{"resources":[]}}"#);
    }
    tauri_build::build()
}
