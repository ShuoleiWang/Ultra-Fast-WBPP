cmake_minimum_required(VERSION 3.28)

if(APPLE AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS version")
endif()

project(OpenAstroFlowNative VERSION 0.1.0 LANGUAGES CXX)

if(MSVC)
  # Link the Visual C++ runtime statically (/MT, /MTd for Debug). The kernel
  # DLL ships inside the frozen worker and must load on a clean Windows
  # machine without the VC++ redistributable; MSVCP140/VCRUNTIME140 imports
  # are therefore a release-gate failure (scripts/build_native_runtime.py
  # --require-static-crt). Set project-wide (CMP0091 NEW, CMake >= 3.15) so
  # the ctest executables that link the same objects do not mix runtimes.
  # The C ABI only exchanges caller-owned buffers, so no CRT allocation
  # crosses the DLL boundary.
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

option(OAF_BUILD_TESTS "Build native scientific contract tests" ON)
option(OAF_ENABLE_METAL "Build the Apple Metal backend when available" ON)
option(OAF_BUILD_BENCHMARKS "Build opt-in native performance benchmarks" OFF)

find_package(Threads REQUIRED)

add_library(openastroflow_native_math STATIC
  src/FusedLnIntegration.cpp
  src/CpuFeatures.cpp
  src/DebayerKernel.cpp
  src/DrizzleKernel.cpp
  src/ImageTile.cpp
  src/Lanczos3Table.cpp
  src/PortableKernels.cpp
)
add_library(OpenAstroFlow::NativeMath ALIAS openastroflow_native_math)
target_include_directories(openastroflow_native_math PUBLIC include)
target_compile_features(openastroflow_native_math PUBLIC cxx_std_20)
target_link_libraries(openastroflow_native_math PUBLIC Threads::Threads)
set_target_properties(openastroflow_native_math PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

if(MSVC)
  target_compile_options(openastroflow_native_math PRIVATE /W4 /WX /fp:strict)
else()
  target_compile_options(openastroflow_native_math PRIVATE
    -Wall -Wextra -Wpedantic -Werror -fno-fast-math -ffp-contract=off
  )
endif()

add_library(openastroflow_native_abi SHARED src/c_api.cpp)
add_library(OpenAstroFlow::NativeAbi ALIAS openastroflow_native_abi)
target_include_directories(openastroflow_native_abi PUBLIC include)
target_compile_features(openastroflow_native_abi PUBLIC cxx_std_20)
target_compile_definitions(openastroflow_native_abi PRIVATE OAF_NATIVE_BUILDING_DLL=1)
target_link_libraries(openastroflow_native_abi PRIVATE OpenAstroFlow::NativeMath)
set_target_properties(openastroflow_native_abi PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  OUTPUT_NAME openastroflow_native
)
install(TARGETS openastroflow_native_abi
  LIBRARY DESTINATION openastroflow_engine/native
  RUNTIME DESTINATION openastroflow_engine/native
)
if(MSVC)
  target_compile_options(openastroflow_native_abi PRIVATE /W4 /WX /fp:strict)
else()
  target_compile_options(openastroflow_native_abi PRIVATE
    -Wall -Wextra -Wpedantic -Werror -fno-fast-math -ffp-contract=off
  )
endif()

if(APPLE AND OAF_ENABLE_METAL)
  enable_language(OBJCXX)
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/metal/FusedLnIntegration.metal"
    OAF_EMBEDDED_METAL_SOURCE)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/EmbeddedMetalSource.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/EmbeddedMetalSource.h"
    @ONLY
  )
  add_library(openastroflow_native_metal STATIC src/MetalFusedLnIntegration.mm)
  add_library(OpenAstroFlow::NativeMetal ALIAS openastroflow_native_metal)
  target_include_directories(openastroflow_native_metal
    PUBLIC include
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated"
  )
  target_compile_features(openastroflow_native_metal PUBLIC cxx_std_20)
  target_compile_options(openastroflow_native_metal PRIVATE
    -fobjc-arc -Wall -Wextra -Wpedantic -Werror
    -fno-fast-math -ffp-contract=off
  )
  target_link_libraries(openastroflow_native_metal
    PUBLIC OpenAstroFlow::NativeMath
    PRIVATE "-framework Foundation" "-framework Metal"
  )
  target_compile_definitions(openastroflow_native_abi PRIVATE OAF_WITH_METAL_ABI=1)
  target_link_libraries(openastroflow_native_abi
    PRIVATE OpenAstroFlow::NativeMetal
            "-framework Foundation" "-framework Metal"
  )
  set_property(TARGET openastroflow_native_abi PROPERTY LINKER_LANGUAGE OBJCXX)
endif()

if(APPLE AND OAF_ENABLE_METAL AND OAF_BUILD_BENCHMARKS)
  if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(FATAL_ERROR
      "OAF_BUILD_BENCHMARKS requires -DCMAKE_BUILD_TYPE=Release"
    )
  endif()
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/metal/FusedLnIntegration.metal"
    OAF_BENCH_METAL_SOURCE_SHA256)
  set(OAF_BENCH_SOURCE_COMMIT "unknown")
  set(OAF_BENCH_SOURCE_DIRTY 1)
  set(OAF_BENCH_TRACKED_DIFF_SHA256 "unknown")
  set(OAF_BENCH_UNTRACKED_TREE_SHA256 "unknown")
  set(OAF_BENCH_UNTRACKED_FILE_COUNT 0)
  find_package(Git QUIET)
  if(Git_FOUND)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" rev-parse --show-toplevel
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      RESULT_VARIABLE OAF_BENCH_GIT_ROOT_RESULT
      OUTPUT_VARIABLE OAF_BENCH_GIT_ROOT
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      RESULT_VARIABLE OAF_BENCH_GIT_REV_RESULT
      OUTPUT_VARIABLE OAF_BENCH_GIT_REV
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(OAF_BENCH_GIT_REV_RESULT EQUAL 0)
      set(OAF_BENCH_SOURCE_COMMIT "${OAF_BENCH_GIT_REV}")
    endif()
    if(OAF_BENCH_GIT_ROOT_RESULT EQUAL 0)
      execute_process(
        COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=normal
        WORKING_DIRECTORY "${OAF_BENCH_GIT_ROOT}"
        RESULT_VARIABLE OAF_BENCH_GIT_STATUS_RESULT
        OUTPUT_VARIABLE OAF_BENCH_GIT_STATUS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      if(OAF_BENCH_GIT_STATUS_RESULT EQUAL 0 AND OAF_BENCH_GIT_STATUS STREQUAL "")
        set(OAF_BENCH_SOURCE_DIRTY 0)
      endif()
      execute_process(
        COMMAND "${GIT_EXECUTABLE}" diff --binary HEAD
        WORKING_DIRECTORY "${OAF_BENCH_GIT_ROOT}"
        RESULT_VARIABLE OAF_BENCH_GIT_DIFF_RESULT
        OUTPUT_VARIABLE OAF_BENCH_GIT_DIFF
        ERROR_QUIET
      )
      if(OAF_BENCH_GIT_DIFF_RESULT EQUAL 0)
        string(SHA256 OAF_BENCH_TRACKED_DIFF_SHA256 "${OAF_BENCH_GIT_DIFF}")
        set(OAF_BENCH_TRACKED_DIFF_SHA256
          "sha256:${OAF_BENCH_TRACKED_DIFF_SHA256}")
      endif()
      execute_process(
        COMMAND "${GIT_EXECUTABLE}" ls-files --others --exclude-standard
        WORKING_DIRECTORY "${OAF_BENCH_GIT_ROOT}"
        RESULT_VARIABLE OAF_BENCH_UNTRACKED_RESULT
        OUTPUT_VARIABLE OAF_BENCH_UNTRACKED_FILES
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      if(OAF_BENCH_UNTRACKED_RESULT EQUAL 0)
        set(OAF_BENCH_UNTRACKED_MANIFEST "")
        if(NOT OAF_BENCH_UNTRACKED_FILES STREQUAL "")
          string(REPLACE "\n" ";" OAF_BENCH_UNTRACKED_LIST
            "${OAF_BENCH_UNTRACKED_FILES}")
          foreach(OAF_BENCH_UNTRACKED_FILE IN LISTS OAF_BENCH_UNTRACKED_LIST)
            set(OAF_BENCH_UNTRACKED_PATH
              "${OAF_BENCH_GIT_ROOT}/${OAF_BENCH_UNTRACKED_FILE}")
            if(EXISTS "${OAF_BENCH_UNTRACKED_PATH}"
               AND NOT IS_DIRECTORY "${OAF_BENCH_UNTRACKED_PATH}")
              file(SHA256 "${OAF_BENCH_UNTRACKED_PATH}"
                OAF_BENCH_UNTRACKED_FILE_SHA256)
              string(APPEND OAF_BENCH_UNTRACKED_MANIFEST
                "${OAF_BENCH_UNTRACKED_FILE}\n${OAF_BENCH_UNTRACKED_FILE_SHA256}\n")
              math(EXPR OAF_BENCH_UNTRACKED_FILE_COUNT
                "${OAF_BENCH_UNTRACKED_FILE_COUNT} + 1")
            endif()
          endforeach()
        endif()
        string(SHA256 OAF_BENCH_UNTRACKED_TREE_SHA256
          "${OAF_BENCH_UNTRACKED_MANIFEST}")
        set(OAF_BENCH_UNTRACKED_TREE_SHA256
          "sha256:${OAF_BENCH_UNTRACKED_TREE_SHA256}")
      endif()
    endif()
  endif()
  add_executable(openastroflow_integration_benchmark
    benchmarks/IntegrationBenchmark.cpp
  )
  target_compile_features(openastroflow_integration_benchmark PRIVATE cxx_std_20)
  target_compile_options(openastroflow_integration_benchmark PRIVATE
    -Wall -Wextra -Wpedantic -Werror -fno-fast-math -ffp-contract=off
  )
  target_compile_definitions(openastroflow_integration_benchmark PRIVATE
    OAF_BENCH_BUILD_TYPE="$<CONFIG>"
    OAF_BENCH_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
    OAF_BENCH_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
    OAF_BENCH_SOURCE_COMMIT="${OAF_BENCH_SOURCE_COMMIT}"
    OAF_BENCH_SOURCE_DIRTY=${OAF_BENCH_SOURCE_DIRTY}
    OAF_BENCH_TRACKED_DIFF_SHA256="${OAF_BENCH_TRACKED_DIFF_SHA256}"
    OAF_BENCH_UNTRACKED_TREE_SHA256="${OAF_BENCH_UNTRACKED_TREE_SHA256}"
    OAF_BENCH_UNTRACKED_FILE_COUNT=${OAF_BENCH_UNTRACKED_FILE_COUNT}
    OAF_BENCH_METAL_SOURCE_SHA256="${OAF_BENCH_METAL_SOURCE_SHA256}"
  )
  target_link_libraries(openastroflow_integration_benchmark
    PRIVATE OpenAstroFlow::NativeMetal
  )
endif()

if(OAF_BUILD_TESTS)
  include(CTest)
  enable_testing()

  add_executable(openastroflow_native_core_tests tests/NativeCoreTests.cpp)
  target_link_libraries(openastroflow_native_core_tests PRIVATE OpenAstroFlow::NativeMath)
  target_compile_features(openastroflow_native_core_tests PRIVATE cxx_std_20)

  if(APPLE AND OAF_ENABLE_METAL)
    target_compile_definitions(openastroflow_native_core_tests PRIVATE OAF_WITH_METAL=1)
    target_link_libraries(openastroflow_native_core_tests PRIVATE OpenAstroFlow::NativeMetal)
    add_test(
      NAME native-core-metal-differential
      COMMAND openastroflow_native_core_tests
              ${CMAKE_CURRENT_SOURCE_DIR}/metal/FusedLnIntegration.metal
    )
    set_tests_properties(native-core-metal-differential PROPERTIES
      SKIP_RETURN_CODE 77
      LABELS "metal;differential"
    )
  else()
    add_test(NAME native-core-cpu COMMAND openastroflow_native_core_tests)
  endif()

  add_executable(openastroflow_native_c_api_tests tests/NativeCapiTests.cpp)
  target_link_libraries(openastroflow_native_c_api_tests
    PRIVATE OpenAstroFlow::NativeAbi
  )
  target_compile_features(openastroflow_native_c_api_tests PRIVATE cxx_std_20)
  add_test(NAME native-c-api COMMAND openastroflow_native_c_api_tests)

  add_executable(openastroflow_portable_kernel_tests tests/PortableKernelTests.cpp)
  target_link_libraries(openastroflow_portable_kernel_tests
    PRIVATE OpenAstroFlow::NativeMath OpenAstroFlow::NativeAbi
  )
  # The weight-table test reads the internal Lanczos3Table.h.
  target_include_directories(openastroflow_portable_kernel_tests PRIVATE src)
  target_compile_features(openastroflow_portable_kernel_tests PRIVATE cxx_std_20)
  if(NOT MSVC)
    target_compile_options(openastroflow_portable_kernel_tests PRIVATE
      -Wall -Wextra -Wpedantic -Werror -fno-fast-math -ffp-contract=off
    )
  endif()
  add_test(NAME native-portable-kernels COMMAND openastroflow_portable_kernel_tests)
endif()
