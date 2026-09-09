#ifndef OPENASTROFLOW_NATIVE_C_API_H
#define OPENASTROFLOW_NATIVE_C_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(OAF_NATIVE_BUILDING_DLL)
#    define OAF_NATIVE_API __declspec(dllexport)
#  else
#    define OAF_NATIVE_API __declspec(dllimport)
#  endif
#else
#  define OAF_NATIVE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum
{
   OAF_NATIVE_ABI_VERSION = 1,
   OAF_NATIVE_OK = 0,
   OAF_NATIVE_INVALID_ARGUMENT = 1,
   OAF_NATIVE_BUFFER_TOO_SMALL = 2,
   OAF_NATIVE_EXECUTION_FAILED = 3,
   OAF_NATIVE_BACKEND_UNAVAILABLE = 4
};

enum
{
   OAF_NATIVE_MAX_DEVICE_NAME_BYTES = 128
};

typedef struct OafNativeIntegrationRequestV1
{
   uint32_t struct_size;
   uint32_t width;
   uint32_t image_height;
   uint32_t first_row;
   uint32_t row_count;
   uint32_t frame_count;
   uint32_t grid_width;
   uint32_t grid_height;
   const float* frame_major_samples;
   size_t sample_count;
   const float* frame_major_scale_grid;
   size_t scale_grid_count;
   const float* frame_major_zero_offset_grid;
   size_t zero_offset_grid_count;
   const float* frame_weights;
   size_t weight_count;
   float range_low;
   float low_tolerance;
   float high_tolerance;
   uint32_t fit_bisection_iterations;
   uint32_t rejection_iterations;
   float output_scale;
   float output_offset;
} OafNativeIntegrationRequestV1;

typedef struct OafNativeMaskedIntegrationRequestV1
{
   uint32_t struct_size;
   uint32_t width;
   uint32_t image_height;
   uint32_t first_row;
   uint32_t row_count;
   uint32_t frame_count;
   uint32_t grid_width;
   uint32_t grid_height;
   const float* frame_major_samples;
   size_t sample_count;
   const uint8_t* frame_major_rejection_mask;
   size_t rejection_mask_count;
   const float* frame_major_scale_grid;
   size_t scale_grid_count;
   const float* frame_major_zero_offset_grid;
   size_t zero_offset_grid_count;
   const float* frame_weights;
   size_t weight_count;
   uint32_t rejection_bits;
   float output_scale;
   float output_offset;
} OafNativeMaskedIntegrationRequestV1;

typedef struct OafNativeIntegrationOutputV1
{
   uint32_t struct_size;
   float* integrated;
   uint16_t* accepted_samples;
   uint16_t* rejected_samples;
   size_t pixel_capacity;
} OafNativeIntegrationOutputV1;

typedef struct OafNativeExecutionStatsV1
{
   uint32_t struct_size;
   uint32_t executed_on_gpu;
   double wall_seconds;
   double gpu_seconds;
   uint64_t submitted_buffer_bytes;
   uint64_t recommended_working_set_bytes;
   uint64_t maximum_buffer_bytes;
   char device_name[OAF_NATIVE_MAX_DEVICE_NAME_BYTES];
} OafNativeExecutionStatsV1;

// Opaque, reusable Metal state. Callers never dereference this type.
typedef struct OafNativeMetalExecutorV1 OafNativeMetalExecutorV1;

OAF_NATIVE_API uint32_t oaf_native_abi_version(void);

// Runs the strict portable CPU Linear Fit integration implementation. The
// caller owns every buffer and must provide width*row_count output capacity.
// Errors are copied as UTF-8 into error_message when that buffer is nonempty.
OAF_NATIVE_API int oaf_native_cpu_linear_fit_v1(
   const OafNativeIntegrationRequestV1* request,
   OafNativeIntegrationOutputV1* output,
   char* error_message,
   size_t error_message_capacity );

// Exact full-stack weighted integration using a caller-generated per-sample
// rejection mask. This path supports up to 65535 frames and never batches
// partial means.
OAF_NATIVE_API int oaf_native_cpu_masked_weighted_v1(
   const OafNativeMaskedIntegrationRequestV1* request,
   OafNativeIntegrationOutputV1* output,
   char* error_message,
   size_t error_message_capacity );

// Capability probing never creates output artifacts. `available` is always
// written as zero or one when it is non-null.
OAF_NATIVE_API int oaf_native_metal_available_v1(
   uint32_t* available,
   char* error_message,
   size_t error_message_capacity );

// `metal_source_path` may be null/empty to use the shader source embedded at
// native-library build time. A nonempty path must be canonical and absolute.
OAF_NATIVE_API int oaf_native_metal_executor_create_v1(
   const char* metal_source_path,
   OafNativeMetalExecutorV1** executor,
   char* error_message,
   size_t error_message_capacity );

OAF_NATIVE_API void oaf_native_metal_executor_destroy_v1(
   OafNativeMetalExecutorV1* executor );

OAF_NATIVE_API int oaf_native_metal_linear_fit_v1(
   OafNativeMetalExecutorV1* executor,
   const OafNativeIntegrationRequestV1* request,
   OafNativeIntegrationOutputV1* output,
   OafNativeExecutionStatsV1* stats,
   char* error_message,
   size_t error_message_capacity );

OAF_NATIVE_API int oaf_native_metal_masked_weighted_v1(
   OafNativeMetalExecutorV1* executor,
   const OafNativeMaskedIntegrationRequestV1* request,
   OafNativeIntegrationOutputV1* output,
   OafNativeExecutionStatsV1* stats,
   char* error_message,
   size_t error_message_capacity );

#ifdef __cplusplus
}
#endif

#endif
