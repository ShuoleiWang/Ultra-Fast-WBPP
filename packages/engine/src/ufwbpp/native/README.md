# Optional native runtime

Release builds install `libufwbpp_native.dylib` (macOS),
`libufwbpp_native.so` (Linux), or `ufwbpp_native.dll` (Windows)
into this directory before building the Python wheel/sidecar. The Python
adapter validates ABI version and required symbols before use. No native
binary is checked into source control.

For a source build:

```bash
python scripts/build_native_runtime.py --build-dir build/native-release
# = Release configure, build, ctest and install into this directory
#   (the Makefile's `native-release-install` runs the same chain).
# Never install the unoptimized `build/native` test configuration.
```
