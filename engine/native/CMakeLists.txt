cmake_minimum_required(VERSION 3.28)

if(APPLE AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS version")
endif()

project(UfwbppNative VERSION 0.1.0 LANGUAGES CXX)

if(MSVC)
  # Link the Visual C++ runtime statically (/MT, /MTd for Debug). The kernel
  # DLL ships inside the frozen engine and must load on a clean Windows
  # machine without the VC++ redistributable; MSVCP140/VCRUNTIME140 imports
  # are therefore a release-gate failure (scripts/build_native_runtime.py
  # --require-static-crt). Set project-wide (CMP0091 NEW, CMake >= 3.15) so
  # the ctest executables that link the same objects do not mix runtimes.
  # The C ABI only exchanges caller-owned buffers, so no CRT allocation
  # crosses the DLL boundary.
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

option(UFWBPP_BUILD_TESTS "Build native scientific contract tests" ON)
option(UFWBPP_ENABLE_METAL "Build the Apple Metal backend when available" ON)
option(UFWBPP_BUILD_BENCHMARKS "Build opt-in native performance benchmarks" OFF)

find_package(Threads REQUIRED)

add_library(ufwbpp_native_math STATIC
  src/FusedIntegration.cpp
  src/CpuFeatures.cpp
  src/DebayerKernel.cpp
  src/DrizzleKernel.cpp
  src/ImageTile.cpp
  src/Lanczos3Table.cpp
  src/OffsetGridKernel.cpp
  src/PortableKernels.cpp
)
add_library(Ufwbpp::NativeMath ALIAS ufwbpp_native_math)
target_include_directories(ufwbpp_native_math PUBLIC include)
target_compile_features(ufwbpp_native_math PUBLIC cxx_std_20)
target_link_libraries(ufwbpp_native_math PUBLIC Threads::Threads)
set_target_properties(ufwbpp_native_math PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

if(MSVC)
  target_compile_options(ufwbpp_native_math PRIVATE /W4 /WX /fp:strict)
else()
  target_compile_options(ufwbpp_native_math PRIVATE
    -Wall -Wextra -Wpedantic -Werror -fno-fast-math -ffp-contract=off
  )
endif()

add_library(ufwbpp_native_abi SHARED src/c_api.cpp)
add_library(Ufwbpp::NativeAbi ALIAS ufwbpp_native_abi)
target_include_directories(ufwbpp_native_abi PUBLIC include)
target_compile_features(ufwbpp_native_abi PUBLIC cxx_std_20)
target_compile_definitions(ufwbpp_native_abi PRIVATE UFWBPP_NATIVE_BUILDING_DLL=1)
target_link_libraries(ufwbpp_native_abi PRIVATE Ufwbpp::NativeMath)
set_target_properties(ufwbpp_native_abi PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  OUTPUT_NAME ufwbpp_native
)
install(TARGETS ufwbpp_native_abi
  LIBRARY DESTINATION ufwbpp/native
  RUNTIME DESTINATION ufwbpp/native
)
if(MSVC)
  target_compile_options(ufwbpp_native_abi PRIVATE /W4 /WX /fp:strict)
else()
  target_compile_options(ufwbpp_native_abi PRIVATE
    -Wall -Wextra -Wpedantic -Werror -fno-fast-math -ffp-contract=off
  )
endif()

if(APPLE AND UFWBPP_ENABLE_METAL)
  enable_language(OBJCXX)
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/metal/FusedIntegration.metal"
    UFWBPP_EMBEDDED_METAL_SOURCE)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/EmbeddedMetalSource.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/EmbeddedMetalSource.h"
    @ONLY
  )
  add_library(ufwbpp_native_metal STATIC src/MetalFusedIntegration.mm)
  add_library(Ufwbpp::NativeMetal ALIAS ufwbpp_native_metal)
  target_include_directories(ufwbpp_native_metal
    PUBLIC include
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated"
  )
  target_compile_features(ufwbpp_native_metal PUBLIC cxx_std_20)
  target_compile_options(ufwbpp_native_metal PRIVATE
    -fobjc-arc -Wall -Wextra -Wpedantic -Werror
    -fno-fast-math -ffp-contract=off
  )
  target_link_libraries(ufwbpp_native_metal
    PUBLIC Ufwbpp::NativeMath
    PRIVATE "-framework Foundation" "-framework Metal"
  )
  target_compile_definitions(ufwbpp_native_abi PRIVATE UFWBPP_WITH_METAL_ABI=1)
  target_link_libraries(ufwbpp_native_abi
    PRIVATE Ufwbpp::NativeMetal
            "-framework Foundation" "-framework Metal"
  )
  set_property(TARGET ufwbpp_native_abi PROPERTY LINKER_LANGUAGE OBJCXX)
endif()

if(APPLE AND UFWBPP_ENABLE_METAL AND UFWBPP_BUILD_BENCHMARKS)
  if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(FATAL_ERROR
      "UFWBPP_BUILD_BENCHMARKS requires -DCMAKE_BUILD_TYPE=Release"
    )
  endif()
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/metal/FusedIntegration.metal"
    UFWBPP_BENCH_METAL_SOURCE_SHA256)
  set(UFWBPP_BENCH_SOURCE_COMMIT "unknown")
  set(UFWBPP_BENCH_SOURCE_DIRTY 1)
  set(UFWBPP_BENCH_TRACKED_DIFF_SHA256 "unknown")
  set(UFWBPP_BENCH_UNTRACKED_TREE_SHA256 "unknown")
  set(UFWBPP_BENCH_UNTRACKED_FILE_COUNT 0)
  find_package(Git QUIET)
  if(Git_FOUND)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" rev-parse --show-toplevel
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      RESULT_VARIABLE UFWBPP_BENCH_GIT_ROOT_RESULT
      OUTPUT_VARIABLE UFWBPP_BENCH_GIT_ROOT
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      RESULT_VARIABLE UFWBPP_BENCH_GIT_REV_RESULT
      OUTPUT_VARIABLE UFWBPP_BENCH_GIT_REV
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(UFWBPP_BENCH_GIT_REV_RESULT EQUAL 0)
      set(UFWBPP_BENCH_SOURCE_COMMIT "${UFWBPP_BENCH_GIT_REV}")
    endif()
    if(UFWBPP_BENCH_GIT_ROOT_RESULT EQUAL 0)
      execute_process(
        COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=normal
        WORKING_DIRECTORY "${UFWBPP_BENCH_GIT_ROOT}"
        RESULT_VARIABLE UFWBPP_BENCH_GIT_STATUS_RESULT
        OUTPUT_VARIABLE UFWBPP_BENCH_GIT_STATUS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      if(UFWBPP_BENCH_GIT_STATUS_RESULT EQUAL 0 AND UFWBPP_BENCH_GIT_STATUS STREQUAL "")
        set(UFWBPP_BENCH_SOURCE_DIRTY 0)
      endif()
      execute_process(
        COMMAND "${GIT_EXECUTABLE}" diff --binary HEAD
        WORKING_DIRECTORY "${UFWBPP_BENCH_GIT_ROOT}"
        RESULT_VARIABLE UFWBPP_BENCH_GIT_DIFF_RESULT
        OUTPUT_VARIABLE UFWBPP_BENCH_GIT_DIFF
        ERROR_QUIET
      )
      if(UFWBPP_BENCH_GIT_DIFF_RESULT EQUAL 0)
        string(SHA256 UFWBPP_BENCH_TRACKED_DIFF_SHA256 "${UFWBPP_BENCH_GIT_DIFF}")
        set(UFWBPP_BENCH_TRACKED_DIFF_SHA256
          "sha256:${UFWBPP_BENCH_TRACKED_DIFF_SHA256}")
      endif()
      execute_process(
        COMMAND "${GIT_EXECUTABLE}" ls-files --others --exclude-standard
        WORKING_DIRECTORY "${UFWBPP_BENCH_GIT_ROOT}"
        RESULT_VARIABLE UFWBPP_BENCH_UNTRACKED_RESULT
        OUTPUT_VARIABLE UFWBPP_BENCH_UNTRACKED_FILES
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      if(UFWBPP_BENCH_UNTRACKED_RESULT EQUAL 0)
        set(UFWBPP_BENCH_UNTRACKED_MANIFEST "")
        if(NOT UFWBPP_BENCH_UNTRACKED_FILES STREQUAL "")
          string(REPLACE "\n" ";" UFWBPP_BENCH_UNTRACKED_LIST
            "${UFWBPP_BENCH_UNTRACKED_FILES}")
          foreach(UFWBPP_BENCH_UNTRACKED_FILE IN LISTS UFWBPP_BENCH_UNTRACKED_LIST)
            set(UFWBPP_BENCH_UNTRACKED_PATH
              "${UFWBPP_BENCH_GIT_ROOT}/${UFWBPP_BENCH_UNTRACKED_FILE}")
            if(EXISTS "${UFWBPP_BENCH_UNTRACKED_PATH}"
               AND NOT IS_DIRECTORY "${UFWBPP_BENCH_UNTRACKED_PATH}")
              file(SHA256 "${UFWBPP_BENCH_UNTRACKED_PATH}"
                UFWBPP_BENCH_UNTRACKED_FILE_SHA256)
              string(APPEND UFWBPP_BENCH_UNTRACKED_MANIFEST
                "${UFWBPP_BENCH_UNTRACKED_FILE}\n${UFWBPP_BENCH_UNTRACKED_FILE_SHA256}\n")
              math(EXPR UFWBPP_BENCH_UNTRACKED_FILE_COUNT
                "${UFWBPP_BENCH_UNTRACKED_FILE_COUNT} + 1")
            endif()
          endforeach()
        endif()
        string(SHA256 UFWBPP_BENCH_UNTRACKED_TREE_SHA256
          "${UFWBPP_BENCH_UNTRACKED_MANIFEST}")
        set(UFWBPP_BENCH_UNTRACKED_TREE_SHA256
          "sha256:${UFWBPP_BENCH_UNTRACKED_TREE_SHA256}")
      endif()
    endif()
  endif()
  add_executable(ufwbpp_integration_benchmark
    benchmarks/IntegrationBenchmark.cpp
  )
  target_compile_features(ufwbpp_integration_benchmark PRIVATE cxx_std_20)
  target_compile_options(ufwbpp_integration_benchmark PRIVATE
    -Wall -Wextra -Wpedantic -Werror -fno-fast-math -ffp-contract=off
  )
  target_compile_definitions(ufwbpp_integration_benchmark PRIVATE
    UFWBPP_BENCH_BUILD_TYPE="$<CONFIG>"
    UFWBPP_BENCH_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
    UFWBPP_BENCH_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
    UFWBPP_BENCH_SOURCE_COMMIT="${UFWBPP_BENCH_SOURCE_COMMIT}"
    UFWBPP_BENCH_SOURCE_DIRTY=${UFWBPP_BENCH_SOURCE_DIRTY}
    UFWBPP_BENCH_TRACKED_DIFF_SHA256="${UFWBPP_BENCH_TRACKED_DIFF_SHA256}"
    UFWBPP_BENCH_UNTRACKED_TREE_SHA256="${UFWBPP_BENCH_UNTRACKED_TREE_SHA256}"
    UFWBPP_BENCH_UNTRACKED_FILE_COUNT=${UFWBPP_BENCH_UNTRACKED_FILE_COUNT}
    UFWBPP_BENCH_METAL_SOURCE_SHA256="${UFWBPP_BENCH_METAL_SOURCE_SHA256}"
  )
  target_link_libraries(ufwbpp_integration_benchmark
    PRIVATE Ufwbpp::NativeMetal
  )
endif()

if(UFWBPP_BUILD_TESTS)
  include(CTest)
  enable_testing()

  add_executable(ufwbpp_native_core_tests tests/NativeCoreTests.cpp)
  target_link_libraries(ufwbpp_native_core_tests PRIVATE Ufwbpp::NativeMath)
  target_compile_features(ufwbpp_native_core_tests PRIVATE cxx_std_20)

  if(APPLE AND UFWBPP_ENABLE_METAL)
    target_compile_definitions(ufwbpp_native_core_tests PRIVATE UFWBPP_WITH_METAL=1)
    target_link_libraries(ufwbpp_native_core_tests PRIVATE Ufwbpp::NativeMetal)
    add_test(
      NAME native-core-metal-differential
      COMMAND ufwbpp_native_core_tests
              ${CMAKE_CURRENT_SOURCE_DIR}/metal/FusedIntegration.metal
    )
    set_tests_properties(native-core-metal-differential PROPERTIES
      SKIP_RETURN_CODE 77
      LABELS "metal;differential"
    )
  else()
    add_test(NAME native-core-cpu COMMAND ufwbpp_native_core_tests)
  endif()

  add_executable(ufwbpp_native_c_api_tests tests/NativeCapiTests.cpp)
  target_link_libraries(ufwbpp_native_c_api_tests
    PRIVATE Ufwbpp::NativeAbi
  )
  target_compile_features(ufwbpp_native_c_api_tests PRIVATE cxx_std_20)
  add_test(NAME native-c-api COMMAND ufwbpp_native_c_api_tests)

  add_executable(ufwbpp_portable_kernel_tests tests/PortableKernelTests.cpp)
  target_link_libraries(ufwbpp_portable_kernel_tests
    PRIVATE Ufwbpp::NativeMath Ufwbpp::NativeAbi
  )
  # The weight-table test reads the internal Lanczos3Table.h.
  target_include_directories(ufwbpp_portable_kernel_tests PRIVATE src)
  target_compile_features(ufwbpp_portable_kernel_tests PRIVATE cxx_std_20)
  if(NOT MSVC)
    target_compile_options(ufwbpp_portable_kernel_tests PRIVATE
      -Wall -Wextra -Wpedantic -Werror -fno-fast-math -ffp-contract=off
    )
  endif()
  add_test(NAME native-portable-kernels COMMAND ufwbpp_portable_kernel_tests)
endif()
