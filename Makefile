PYTHON ?= $(firstword $(wildcard .venv/bin/python .venv/Scripts/python.exe) python3)
CMAKE ?= cmake
NATIVE_BUILD ?= build/native
NATIVE_RELEASE_BUILD := build/native-release
NATIVE_INSTALL_PREFIX := packages/openastroflow-engine/src
NATIVE_RUNTIME_DIR := $(NATIVE_INSTALL_PREFIX)/openastroflow_engine/native
MACOS14_BOTTLE_ROOT ?= build/macos14-runtime-libraries
HOST_SYSTEM := $(shell uname -s 2>/dev/null)

ifeq ($(HOST_SYSTEM),Darwin)
DESKTOP_BUILD_SCRIPT := tauri:build:macos-prerelease
DESKTOP_SIDECAR_PLATFORM_DEPS := macos14-runtime-libraries
DESKTOP_SIDECAR_PLATFORM_ARGS := --macos14-bottle-root $(MACOS14_BOTTLE_ROOT)
else
DESKTOP_BUILD_SCRIPT := tauri:build
DESKTOP_SIDECAR_PLATFORM_DEPS :=
DESKTOP_SIDECAR_PLATFORM_ARGS :=
endif

.PHONY: bootstrap demo desktop-dev test python-test rust-test frontend-test native-configure native-build native-test native-release-configure native-release-build native-release-test native-release-install macos14-runtime-libraries desktop-sidecar desktop-build desktop-build-macos-prerelease source-check check doctor clean

bootstrap:
	python3 -m venv .venv
	.venv/bin/python -m pip install -e './packages/light-frame-qc[test]' -e './engine/native/python[test]' -e './packages/openastroflow-engine[test,all]' -r packaging/worker/requirements-build.txt
	.venv/bin/python -m pip check
	npm --prefix apps/desktop ci
	$(MAKE) native-build

demo:
	npm --prefix apps/desktop run demo

desktop-dev: native-build
	npm --prefix apps/desktop run tauri -- dev

test: python-test rust-test frontend-test native-test

python-test:
	$(PYTHON) -m pytest -q \
		packages/light-frame-qc/tests \
		engine/native/python/tests \
		packages/openastroflow-engine/tests \
		tests

rust-test:
	cargo test --workspace --locked

frontend-test:
	npm --prefix apps/desktop test
	npm --prefix apps/desktop run build

native-configure:
	$(CMAKE) -S engine/native -B $(NATIVE_BUILD) \
		-DOAF_BUILD_TESTS=ON -DOAF_ENABLE_METAL=ON

native-build: native-configure
	$(CMAKE) --build $(NATIVE_BUILD) --parallel

native-test: native-build
	ctest --test-dir $(NATIVE_BUILD) --output-on-failure

native-release-configure:
	$(CMAKE) -S engine/native -B $(NATIVE_RELEASE_BUILD) \
		-DOAF_BUILD_TESTS=ON -DOAF_ENABLE_METAL=ON \
		-DCMAKE_BUILD_TYPE=Release

native-release-build: native-release-configure
	$(CMAKE) --build $(NATIVE_RELEASE_BUILD) --config Release --parallel

native-release-test: native-release-build
	ctest --test-dir $(NATIVE_RELEASE_BUILD) -C Release --output-on-failure

native-release-install: native-release-test
	$(CMAKE) -E rm -f \
		$(NATIVE_RUNTIME_DIR)/libopenastroflow_native.dylib \
		$(NATIVE_RUNTIME_DIR)/libopenastroflow_native.so \
		$(NATIVE_RUNTIME_DIR)/openastroflow_native.dll
	$(CMAKE) --install $(NATIVE_RELEASE_BUILD) --config Release \
		--prefix $(NATIVE_INSTALL_PREFIX)

$(MACOS14_BOTTLE_ROOT): packaging/worker/macos14-runtime-libraries-v1.json scripts/fetch_macos14_runtime_libraries.py scripts/build_worker_sidecar.py
	$(CMAKE) -E make_directory $(dir $(MACOS14_BOTTLE_ROOT))
	$(PYTHON) scripts/fetch_macos14_runtime_libraries.py \
		--output $(MACOS14_BOTTLE_ROOT)

macos14-runtime-libraries: $(MACOS14_BOTTLE_ROOT)

desktop-sidecar: native-release-install $(DESKTOP_SIDECAR_PLATFORM_DEPS)
	$(PYTHON) scripts/build_worker_sidecar.py --output-dir build/sidecars $(DESKTOP_SIDECAR_PLATFORM_ARGS)
	$(PYTHON) scripts/stage_tauri_sidecar.py \
		build/sidecars/openastroflow-worker-$$($(PYTHON) -c 'from scripts.build_worker_sidecar import detect_host_target_triple; print(detect_host_target_triple())').manifest.json

desktop-build: desktop-sidecar
	npm --prefix apps/desktop run $(DESKTOP_BUILD_SCRIPT)

desktop-build-macos-prerelease: desktop-sidecar
	npm --prefix apps/desktop run tauri:build:macos-prerelease

source-check:
	$(PYTHON) scripts/check_public_tree.py .
	$(PYTHON) scripts/check_local_links.py .

check: source-check
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets --locked -- -D warnings
	$(MAKE) test

doctor:
	openastroflow-engine doctor

clean:
	$(CMAKE) -E rm -rf build
	$(CMAKE) -E rm -f \
		$(NATIVE_RUNTIME_DIR)/libopenastroflow_native.dylib \
		$(NATIVE_RUNTIME_DIR)/libopenastroflow_native.so \
		$(NATIVE_RUNTIME_DIR)/openastroflow_native.dll
