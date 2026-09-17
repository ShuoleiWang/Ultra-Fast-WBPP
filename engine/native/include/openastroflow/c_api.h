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


// Portable multithreaded CPU kernels (see PortableKernels.h). Every kernel
// reproduces the Python engine's NumPy reference arithmetic value for value.

typedef struct OafNativeWarpLanczos3RequestV1
{
   uint32_t struct_size;
   uint32_t source_width;
   uint32_t source_height;
   uint32_t output_width;
   uint32_t first_row;
   uint32_t row_count;
   uint32_t threads;
   uint32_t reserved;
   // Native-endian Float32 physical source samples, source_height*source_width.
   const float* source_samples;
   size_t source_sample_count;
   // Row-major 2x3 output-to-input affine map: m00 m01 m02 m10 m11 m12.
   double inverse[6];
   float domain_scale;
   float reserved_scale;
} OafNativeWarpLanczos3RequestV1;

// Writes row_count*output_width Float32 samples (NaN outside the valid
// support). destination_capacity counts samples, not bytes.
OAF_NATIVE_API int oaf_native_cpu_warp_lanczos3_v1(
   const OafNativeWarpLanczos3RequestV1* request,
   float* destination,
   size_t destination_capacity,
   char* error_message,
   size_t error_message_capacity );

// Version 2 carries the complete row-major 3x3 homogeneous output-to-input
// map (m00 m01 m02 m10 m11 m12 m20 m21 m22). A last row of 0 0 1 is the
// affine map of version 1 with identical arithmetic; any other last row is a
// projective map whose coordinates are divided by w = m20*x + m21*y + m22.
typedef struct OafNativeWarpLanczos3RequestV2
{
   uint32_t struct_size;
   uint32_t source_width;
   uint32_t source_height;
   uint32_t output_width;
   uint32_t first_row;
   uint32_t row_count;
   uint32_t threads;
   uint32_t reserved;
   const float* source_samples;
   size_t source_sample_count;
   double inverse[9];
   float domain_scale;
   float reserved_scale;
} OafNativeWarpLanczos3RequestV2;

OAF_NATIVE_API int oaf_native_cpu_warp_lanczos3_v2(
   const OafNativeWarpLanczos3RequestV2* request,
   float* destination,
   size_t destination_capacity,
   char* error_message,
   size_t error_message_capacity );

typedef struct OafNativeMadRejectionRequestV1
{
   uint32_t struct_size;
   uint32_t frame_count;
   uint32_t row_count;
   uint32_t width;
   uint32_t minimum_rejection_frames;
   uint32_t threads;
   const float* frame_major_samples;
   size_t sample_count;
   float sigma_clip;
   float group_sigma_floor;
   float absolute_floor;
   float epsilon_floor;
} OafNativeMadRejectionRequestV1;

// accepted receives one byte per sample (frame-major, 1 = accepted); center
// receives row_count*width Float32 per-pixel centres. Capacities count
// elements.
OAF_NATIVE_API int oaf_native_cpu_mad_rejection_v1(
   const OafNativeMadRejectionRequestV1* request,
   uint8_t* accepted,
   size_t accepted_capacity,
   float* center,
   size_t center_capacity,
   char* error_message,
   size_t error_message_capacity );

typedef struct OafNativeMaskedMeanRequestV1
{
   uint32_t struct_size;
   uint32_t frame_count;
   uint32_t row_count;
   uint32_t width;
   uint32_t threads;
   uint32_t reserved;
   const float* frame_major_samples;
   size_t sample_count;
   const uint8_t* frame_major_accepted;
   size_t accepted_count;
   const double* frame_weights;
   size_t weight_count;
} OafNativeMaskedMeanRequestV1;

typedef struct OafNativeMaskedMeanOutputV1
{
   uint32_t struct_size;
   uint32_t reserved;
   float* integrated;
   uint16_t* accepted_samples;
   uint16_t* rejected_samples;
   size_t pixel_capacity;
} OafNativeMaskedMeanOutputV1;

OAF_NATIVE_API int oaf_native_cpu_masked_mean_v1(
   const OafNativeMaskedMeanRequestV1* request,
   OafNativeMaskedMeanOutputV1* output,
   char* error_message,
   size_t error_message_capacity );

typedef struct OafNativeTileOffsetRequestV1
{
   uint32_t struct_size;
   uint32_t tile_count;
   uint32_t minimum_samples;
   uint32_t threads;
   const double* target;
   const double* reference;
   size_t sample_count;
   const uint64_t* boundaries;
   size_t boundary_count;
   double scale;
   double lower_quantile;
   double upper_quantile;
   double residual_clip_sigma;
} OafNativeTileOffsetRequestV1;

typedef struct OafNativeTileOffsetOutputV1
{
   uint32_t struct_size;
   uint32_t reserved;
   double* offset;
   uint32_t* count;
   double* residual_mad;
   uint8_t* valid;
   size_t tile_capacity;
} OafNativeTileOffsetOutputV1;

// Per-tile additive offsets for global normalization (see PortableKernels.h).
OAF_NATIVE_API int oaf_native_cpu_tile_offsets_v1(
   const OafNativeTileOffsetRequestV1* request,
   OafNativeTileOffsetOutputV1* output,
   char* error_message,
   size_t error_message_capacity );

// Hardware concurrency clamped to [1, 64].
OAF_NATIVE_API uint32_t oaf_native_default_kernel_threads_v1(void);

#ifdef __cplusplus
}
#endif

#endif
