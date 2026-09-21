// Release builds on Windows are GUI-subsystem executables: without this a
// console window opens next to the app. Debug builds keep the console so
// `tauri dev` shows the log output; macOS and Linux ignore the attribute.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    openastroflow_desktop_lib::run();
}
