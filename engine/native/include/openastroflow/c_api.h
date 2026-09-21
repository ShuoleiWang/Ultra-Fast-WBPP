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

// v2 adds the rejection scale model: frame_scales points at frame_count
// finite positive Float32 factors (NULL with frame_scale_count 0 means every
// factor is 1) and pool_half_width is the half width of the same-row window
// whose per-pixel MADs are pooled (0: none). NULL scales and 0 reproduce the
// v1 decisions exactly.
typedef struct OafNativeMadRejectionRequestV2
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
   const float* frame_scales;
   size_t frame_scale_count;
   uint32_t pool_half_width;
   uint32_t reserved;
} OafNativeMadRejectionRequestV2;

OAF_NATIVE_API int oaf_native_cpu_mad_rejection_v2(
   const OafNativeMadRejectionRequestV2* request,
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

/* V2 adds optional frame-major per-sample weights (Float32, the layout of the
   samples; null = every sample weighs 1). The effective sample weight is the
   Float64 product frame_weight*sample_weight. */
typedef struct OafNativeMaskedMeanRequestV2
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
   const float* frame_major_sample_weights;
   size_t sample_weight_count;
} OafNativeMaskedMeanRequestV2;

OAF_NATIVE_API int oaf_native_cpu_masked_mean_v2(
   const OafNativeMaskedMeanRequestV2* request,
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

typedef struct OafNativeRadonPeakRequestV1
{
   uint32_t struct_size;
   uint32_t width;
   uint32_t height;
   uint32_t size;
   uint32_t minimum_rows;
   uint32_t minimum_scale_samples;
   uint32_t threads;
   float detection_z;
   // Row-major Float32 samples and 0/1 weights, height*width each.
   const float* image;
   size_t image_count;
   const uint8_t* weight;
   size_t weight_count;
   double minimum_coverage;
   double minimum_count;
} OafNativeRadonPeakRequestV1;

typedef struct OafNativeRadonPeakV1
{
   uint32_t level;
   uint32_t block;
   uint32_t shift_index;
   uint32_t column;
   float z;
   uint32_t reserved;
} OafNativeRadonPeakV1;

typedef struct OafNativeRadonPeakOutputV1
{
   uint32_t struct_size;
   uint32_t reserved;
   OafNativeRadonPeakV1* peaks;
   size_t peak_capacity;
   // Always written: the number of peaks found. When it exceeds
   // peak_capacity the call returns OAF_NATIVE_BUFFER_TOO_SMALL and the
   // caller retries with that capacity.
   size_t peak_count;
} OafNativeRadonPeakOutputV1;

// Multi-scale fast-Radon line peaks of one frame orientation (see
// PortableKernels.h RadonLinePeaks).
OAF_NATIVE_API int oaf_native_cpu_radon_peaks_v1(
   const OafNativeRadonPeakRequestV1* request,
   OafNativeRadonPeakOutputV1* output,
   char* error_message,
   size_t error_message_capacity );

enum
{
   OAF_NATIVE_DRIZZLE_KERNEL_SQUARE = 0,
   OAF_NATIVE_DRIZZLE_KERNEL_CIRCULAR = 1,
   OAF_NATIVE_DRIZZLE_KERNEL_GAUSSIAN = 2,
   OAF_NATIVE_DRIZZLE_KERNEL_POINT = 3
};

typedef struct OafNativeDrizzleRequestV1
{
   uint32_t struct_size;
   uint32_t source_width;
   uint32_t source_rows;
   uint32_t source_row0;
   uint32_t scale;
   uint32_t kernel;
   uint32_t output_width;
   uint32_t output_rows;
   uint32_t output_row0;
   uint32_t mask_width;
   uint32_t mask_height;
   uint32_t threads;
   uint8_t cfa_pattern[4];
   uint8_t channel;
   uint8_t reserved[3];
   float normalization_scale;
   float normalization_offset;
   float frame_weight;
   float reserved_float;
   double pixfrac;
   // Row-major 3x3 input-to-output pixel-centre map.
   double forward[9];
   const float* source;
   size_t source_count;
   const double* grid;
   size_t grid_count;
   const double* grid_x_nodes;
   size_t grid_x_count;
   const double* grid_y_nodes;
   size_t grid_y_count;
   const double* weight_grid;
   size_t weight_grid_count;
   const double* weight_grid_x_nodes;
   size_t weight_grid_x_count;
   const double* weight_grid_y_nodes;
   size_t weight_grid_y_count;
   const uint8_t* mask;
   size_t mask_count;
   // Accumulators of the output band, added to in place.
   double* output_sum;
   double* output_weight;
   size_t output_count;
   // Optional output_count touch flags set to 1 where the frame contributed
   // positive weight (NULL: not recorded).
   uint8_t* output_touched;
} OafNativeDrizzleRequestV1;

typedef struct OafNativeDebayerRequestV1
{
   uint32_t struct_size;
   uint32_t width;
   uint32_t height;
   uint32_t threads;
   uint8_t pattern[4];
   const float* mosaic;
   size_t mosaic_count;
   // Three row-major planes R, G, B of width*height Float32 each.
   float* planes;
   size_t plane_count;
} OafNativeDebayerRequestV1;

// Copies the deterministic Lanczos-3 weight table (node_count rows of six
// Float64 normalized tap weights; see Lanczos3Table.h) into `values`, which
// holds `capacity` doubles; writes the row count to `node_count`.
OAF_NATIVE_API int oaf_native_lanczos3_table_v1(
   double* values,
   size_t capacity,
   uint32_t* node_count,
   char* error_message,
   size_t error_message_capacity );

// Bilinear demosaic of a Bayer mosaic (see PortableKernels.h DebayerBilinear).
OAF_NATIVE_API int oaf_native_cpu_debayer_bilinear_v1(
   const OafNativeDebayerRequestV1* request,
   char* error_message,
   size_t error_message_capacity );

// Drizzles one frame band onto one output band (see PortableKernels.h
// DrizzleBand).
OAF_NATIVE_API int oaf_native_cpu_drizzle_v1(
   const OafNativeDrizzleRequestV1* request,
   char* error_message,
   size_t error_message_capacity );

// Hardware concurrency clamped to [1, 64].
OAF_NATIVE_API uint32_t oaf_native_default_kernel_threads_v1(void);

enum
{
   OAF_NATIVE_CPU_ARCHITECTURE_UNKNOWN = 0,
   OAF_NATIVE_CPU_ARCHITECTURE_X86_64 = 1,
   OAF_NATIVE_CPU_ARCHITECTURE_ARM64 = 2
};

// Report-only processor facts (see CpuFeatures.h): the architecture the
// library was compiled for, the comma-separated lowercase instruction-set
// extensions usable by the running OS (cpuid + XSAVE state on x86-64), and
// the cpuid brand string on x86-64. Receipts record them; no kernel selects
// code by them.
typedef struct OafNativeCpuFeaturesV1
{
   uint32_t struct_size;
   uint32_t architecture;
   char features[256];
   char brand[64];
} OafNativeCpuFeaturesV1;

OAF_NATIVE_API int oaf_native_cpu_features_v1(
   OafNativeCpuFeaturesV1* features,
   char* error_message,
   size_t error_message_capacity );

#ifdef __cplusplus
}
#endif

#endif
