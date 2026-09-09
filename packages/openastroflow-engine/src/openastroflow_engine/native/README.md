# Optional native runtime

Release builds install `libopenastroflow_native.dylib` (macOS),
`libopenastroflow_native.so` (Linux), or `openastroflow_native.dll` (Windows)
into this directory before building the Python wheel/sidecar. The Python
adapter validates ABI version and required symbols before use. No native
binary is checked into source control.

For a source build:

```bash
cmake -S engine/native -B build/native -DOAF_ENABLE_METAL=ON
cmake --build build/native --parallel
cmake --install build/native \
  --prefix packages/openastroflow-engine/src
```
