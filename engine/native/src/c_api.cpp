#include "openastroflow/c_api.h"

#include "openastroflow/FusedLnIntegration.h"
#include "openastroflow/CpuFeatures.h"
#include "openastroflow/PortableKernels.h"
#if defined(OAF_WITH_METAL_ABI)
#include "openastroflow/MetalFusedLnIntegration.h"
#endif

#include <algorithm>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

struct OafNativeMetalExecutorV1
{
#if defined(OAF_WITH_METAL_ABI)
   std::unique_ptr<openastroflow::native::MetalFusedLnIntegrationExecutor>
      implementation;
#endif
};

namespace
{

void CopyError( char* destination, std::size_t capacity,
                std::string_view message ) noexcept
{
   if ( destination == nullptr || capacity == 0 )
      return;
   const std::size_t count = std::min( capacity - 1, message.size() );
   std::memcpy( destination, message.data(), count );
   destination[count] = '\0';
}

void CopyFixed( char* destination, std::size_t capacity,
                std::string_view value ) noexcept
{
   if ( destination == nullptr || capacity == 0 )
      return;
   std::memset( destination, 0, capacity );
   const std::size_t count = std::min( capacity - 1, value.size() );
   std::memcpy( destination, value.data(), count );
}

openastroflow::native::NativeLinearFitIntegrationRequest MakeRequest(
   const OafNativeIntegrationRequestV1& input )
{
   if ( input.struct_size != sizeof( OafNativeIntegrationRequestV1 )
     || input.frame_major_samples == nullptr
     || input.frame_major_scale_grid == nullptr
     || input.frame_major_zero_offset_grid == nullptr
     || input.frame_weights == nullptr )
      throw std::invalid_argument(
         "C ABI structure version or input buffer is invalid" );
   using namespace openastroflow::native;
   return {
      TileRegion{
         ImageGeometry{ input.width, input.image_height, 1 },
         input.first_row,
         input.row_count
      },
      input.frame_count,
      input.grid_width,
      input.grid_height,
      std::span<const float>( input.frame_major_samples, input.sample_count ),
      std::span<const float>(
         input.frame_major_scale_grid, input.scale_grid_count ),
      std::span<const float>(
         input.frame_major_zero_offset_grid, input.zero_offset_grid_count ),
      std::span<const float>( input.frame_weights, input.weight_count ),
      input.range_low,
      input.low_tolerance,
      input.high_tolerance,
      input.fit_bisection_iterations,
      input.rejection_iterations,
      input.output_scale,
      input.output_offset
   };
}

openastroflow::native::FusedLnIntegrationRequest MakeMaskedRequest(
   const OafNativeMaskedIntegrationRequestV1& input )
{
   if ( input.struct_size != sizeof( OafNativeMaskedIntegrationRequestV1 )
     || input.frame_major_samples == nullptr
     || input.frame_major_rejection_mask == nullptr
     || input.frame_major_scale_grid == nullptr
     || input.frame_major_zero_offset_grid == nullptr
     || input.frame_weights == nullptr
     || input.rejection_bits == 0 || input.rejection_bits > 0xffU )
      throw std::invalid_argument(
         "masked C ABI structure version or input buffer is invalid" );
   using namespace openastroflow::native;
   return {
      TileRegion{
         ImageGeometry{ input.width, input.image_height, 1 },
         input.first_row,
         input.row_count
      },
      input.frame_count,
      input.grid_width,
      input.grid_height,
      std::span<const float>( input.frame_major_samples, input.sample_count ),
      std::span<const std::uint8_t>(
         input.frame_major_rejection_mask, input.rejection_mask_count ),
      std::span<const float>(
         input.frame_major_scale_grid, input.scale_grid_count ),
      std::span<const float>(
         input.frame_major_zero_offset_grid, input.zero_offset_grid_count ),
      std::span<const float>( input.frame_weights, input.weight_count ),
      static_cast<std::uint8_t>( input.rejection_bits ),
      input.output_scale,
      input.output_offset
   };
}

template <class Request>
int ValidateOutput( const Request& input,
                    const OafNativeIntegrationOutputV1& output )
{
   if ( output.struct_size != sizeof( OafNativeIntegrationOutputV1 ) )
      return OAF_NATIVE_INVALID_ARGUMENT;
   const std::size_t pixels =
      static_cast<std::size_t>( input.width )*input.row_count;
   if ( pixels == 0 || output.pixel_capacity < pixels
     || output.integrated == nullptr
     || output.accepted_samples == nullptr
     || output.rejected_samples == nullptr )
      return OAF_NATIVE_BUFFER_TOO_SMALL;
   return OAF_NATIVE_OK;
}

void CopyResult(
   const openastroflow::native::FusedLnIntegrationResult& result,
   OafNativeIntegrationOutputV1& output )
{
   std::copy( result.integrated.begin(), result.integrated.end(),
              output.integrated );
   std::copy( result.acceptedSamples.begin(), result.acceptedSamples.end(),
              output.accepted_samples );
   std::copy( result.rejectedSamples.begin(), result.rejectedSamples.end(),
              output.rejected_samples );
}

int ExecuteLinearFit( const OafNativeIntegrationRequestV1& input,
                      OafNativeIntegrationOutputV1& output )
{
   const int outputStatus = ValidateOutput( input, output );
   if ( outputStatus != OAF_NATIVE_OK )
      return outputStatus;
   const auto request = MakeRequest( input );
   const auto result =
      openastroflow::native::RunCpuOracleNativeLinearFitIntegration( request );
   CopyResult( result, output );
   return OAF_NATIVE_OK;
}

int ExecuteMaskedWeighted(
   const OafNativeMaskedIntegrationRequestV1& input,
   OafNativeIntegrationOutputV1& output )
{
   const int outputStatus = ValidateOutput( input, output );
   if ( outputStatus != OAF_NATIVE_OK )
      return outputStatus;
   const auto request = MakeMaskedRequest( input );
   const auto result =
      openastroflow::native::RunCpuOracleFusedLnIntegration( request );
   CopyResult( result, output );
   return OAF_NATIVE_OK;
}

#if defined(OAF_WITH_METAL_ABI)
void CopyStats( const openastroflow::native::MetalExecutionStats& source,
                OafNativeExecutionStatsV1& destination ) noexcept
{
   destination.executed_on_gpu = 1;
   destination.wall_seconds = source.wallSeconds;
   destination.gpu_seconds = source.gpuSeconds;
   destination.submitted_buffer_bytes = source.submittedBufferBytes;
   destination.recommended_working_set_bytes =
      source.recommendedWorkingSetBytes;
   destination.maximum_buffer_bytes = source.maximumBufferBytes;
   CopyFixed( destination.device_name, sizeof( destination.device_name ),
              source.deviceName );
}
#endif

} // namespace

extern "C" uint32_t oaf_native_abi_version(void)
{
   return OAF_NATIVE_ABI_VERSION;
}

extern "C" int oaf_native_cpu_linear_fit_v1(
   const OafNativeIntegrationRequestV1* request,
   OafNativeIntegrationOutputV1* output,
   char* error_message,
   size_t error_message_capacity )
{
   if ( request == nullptr || output == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "request and output are required" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   try
   {
      const int status = ExecuteLinearFit( *request, *output );
      if ( status == OAF_NATIVE_INVALID_ARGUMENT )
         CopyError( error_message, error_message_capacity,
                    "C ABI structure version or request is invalid" );
      else if ( status == OAF_NATIVE_BUFFER_TOO_SMALL )
         CopyError( error_message, error_message_capacity,
                    "output capacity is smaller than the requested tile" );
      else
         CopyError( error_message, error_message_capacity, {} );
      return status;
   }
   catch ( const std::invalid_argument& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   catch ( const std::exception& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return OAF_NATIVE_EXECUTION_FAILED;
   }
   catch ( ... )
   {
      CopyError( error_message, error_message_capacity,
                 "unknown native execution failure" );
      return OAF_NATIVE_EXECUTION_FAILED;
   }
}

extern "C" int oaf_native_cpu_masked_weighted_v1(
   const OafNativeMaskedIntegrationRequestV1* request,
   OafNativeIntegrationOutputV1* output,
   char* error_message,
   size_t error_message_capacity )
{
   if ( request == nullptr || output == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "request and output are required" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   try
   {
      const int status = ExecuteMaskedWeighted( *request, *output );
      if ( status == OAF_NATIVE_BUFFER_TOO_SMALL )
         CopyError( error_message, error_message_capacity,
                    "output capacity is smaller than the requested tile" );
      else
         CopyError( error_message, error_message_capacity, {} );
      return status;
   }
   catch ( const std::invalid_argument& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   catch ( const std::exception& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return OAF_NATIVE_EXECUTION_FAILED;
   }
   catch ( ... )
   {
      CopyError( error_message, error_message_capacity,
                 "unknown native masked execution failure" );
      return OAF_NATIVE_EXECUTION_FAILED;
   }
}

extern "C" int oaf_native_metal_available_v1(
   uint32_t* available,
   char* error_message,
   size_t error_message_capacity )
{
   if ( available == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "available output is required" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
#if defined(OAF_WITH_METAL_ABI)
   *available = openastroflow::native::MetalFusedLnIntegrationAvailable()
      ? 1U : 0U;
#else
   *available = 0;
#endif
   CopyError( error_message, error_message_capacity, {} );
   return OAF_NATIVE_OK;
}

extern "C" int oaf_native_metal_executor_create_v1(
   const char* metal_source_path,
   OafNativeMetalExecutorV1** executor,
   char* error_message,
   size_t error_message_capacity )
{
   if ( executor == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "executor output is required" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   *executor = nullptr;
#if defined(OAF_WITH_METAL_ABI)
   try
   {
      auto result = std::make_unique<OafNativeMetalExecutorV1>();
      const std::filesystem::path source =
         metal_source_path == nullptr ? std::filesystem::path()
         : std::filesystem::path( metal_source_path );
      result->implementation = std::make_unique<
         openastroflow::native::MetalFusedLnIntegrationExecutor>( source );
      *executor = result.release();
      CopyError( error_message, error_message_capacity, {} );
      return OAF_NATIVE_OK;
   }
   catch ( const std::invalid_argument& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   catch ( const std::exception& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return OAF_NATIVE_EXECUTION_FAILED;
   }
   catch ( ... )
   {
      CopyError( error_message, error_message_capacity,
                 "unknown Metal executor creation failure" );
      return OAF_NATIVE_EXECUTION_FAILED;
   }
#else
   (void)metal_source_path;
   CopyError( error_message, error_message_capacity,
              "Metal backend was not compiled into this native library" );
   return OAF_NATIVE_BACKEND_UNAVAILABLE;
#endif
}

extern "C" void oaf_native_metal_executor_destroy_v1(
   OafNativeMetalExecutorV1* executor )
{
   delete executor;
}

extern "C" int oaf_native_metal_linear_fit_v1(
   OafNativeMetalExecutorV1* executor,
   const OafNativeIntegrationRequestV1* request,
   OafNativeIntegrationOutputV1* output,
   OafNativeExecutionStatsV1* stats,
   char* error_message,
   size_t error_message_capacity )
{
   if ( executor == nullptr || request == nullptr || output == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "executor, request, and output are required" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   if ( stats != nullptr
     && stats->struct_size != sizeof( OafNativeExecutionStatsV1 ) )
   {
      CopyError( error_message, error_message_capacity,
                 "execution stats structure version is invalid" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
#if defined(OAF_WITH_METAL_ABI)
   if ( !executor->implementation )
   {
      CopyError( error_message, error_message_capacity,
                 "Metal executor is not initialized" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   try
   {
      const int outputStatus = ValidateOutput( *request, *output );
      if ( outputStatus != OAF_NATIVE_OK )
         return outputStatus;
      const auto nativeRequest = MakeRequest( *request );
      openastroflow::native::MetalExecutionStats nativeStats;
      const auto result = executor->implementation->RunNativeLinearFitRejection(
         nativeRequest, stats == nullptr ? nullptr : &nativeStats );
      CopyResult( result, *output );
      if ( stats != nullptr )
         CopyStats( nativeStats, *stats );
      CopyError( error_message, error_message_capacity, {} );
      return OAF_NATIVE_OK;
   }
   catch ( const std::invalid_argument& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   catch ( const std::exception& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return OAF_NATIVE_EXECUTION_FAILED;
   }
   catch ( ... )
   {
      CopyError( error_message, error_message_capacity,
                 "unknown Metal execution failure" );
      return OAF_NATIVE_EXECUTION_FAILED;
   }
#else
   (void)stats;
   CopyError( error_message, error_message_capacity,
              "Metal backend was not compiled into this native library" );
   return OAF_NATIVE_BACKEND_UNAVAILABLE;
#endif
}

extern "C" int oaf_native_metal_masked_weighted_v1(
   OafNativeMetalExecutorV1* executor,
   const OafNativeMaskedIntegrationRequestV1* request,
   OafNativeIntegrationOutputV1* output,
   OafNativeExecutionStatsV1* stats,
   char* error_message,
   size_t error_message_capacity )
{
   if ( executor == nullptr || request == nullptr || output == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "executor, request, and output are required" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   if ( stats != nullptr
     && stats->struct_size != sizeof( OafNativeExecutionStatsV1 ) )
   {
      CopyError( error_message, error_message_capacity,
                 "execution stats structure version is invalid" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
#if defined(OAF_WITH_METAL_ABI)
   if ( !executor->implementation )
   {
      CopyError( error_message, error_message_capacity,
                 "Metal executor is not initialized" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   try
   {
      const int outputStatus = ValidateOutput( *request, *output );
      if ( outputStatus != OAF_NATIVE_OK )
         return outputStatus;
      const auto nativeRequest = MakeMaskedRequest( *request );
      openastroflow::native::MetalExecutionStats nativeStats;
      const auto result = executor->implementation->Run(
         nativeRequest, stats == nullptr ? nullptr : &nativeStats );
      CopyResult( result, *output );
      if ( stats != nullptr )
         CopyStats( nativeStats, *stats );
      CopyError( error_message, error_message_capacity, {} );
      return OAF_NATIVE_OK;
   }
   catch ( const std::invalid_argument& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   catch ( const std::exception& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return OAF_NATIVE_EXECUTION_FAILED;
   }
   catch ( ... )
   {
      CopyError( error_message, error_message_capacity,
                 "unknown masked Metal execution failure" );
      return OAF_NATIVE_EXECUTION_FAILED;
   }
#else
   (void)stats;
   CopyError( error_message, error_message_capacity,
              "Metal backend was not compiled into this native library" );
   return OAF_NATIVE_BACKEND_UNAVAILABLE;
#endif
}

namespace
{

template <class Function>
int GuardedKernelCall( Function&& function,
                       char* error_message,
                       size_t error_message_capacity,
                       const char* unknown_failure )
{
   try
   {
      function();
      CopyError( error_message, error_message_capacity, {} );
      return OAF_NATIVE_OK;
   }
   catch ( const std::invalid_argument& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   catch ( const std::exception& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return OAF_NATIVE_EXECUTION_FAILED;
   }
   catch ( ... )
   {
      CopyError( error_message, error_message_capacity, unknown_failure );
      return OAF_NATIVE_EXECUTION_FAILED;
   }
}

} // namespace

extern "C" int oaf_native_cpu_warp_lanczos3_v1(
   const OafNativeWarpLanczos3RequestV1* request,
   float* destination,
   size_t destination_capacity,
   char* error_message,
   size_t error_message_capacity )
{
   if ( request == nullptr || destination == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "request and destination are required" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   if ( request->struct_size != sizeof( OafNativeWarpLanczos3RequestV1 )
     || request->source_samples == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "warp C ABI structure version or input buffer is invalid" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   const size_t pixels =
      static_cast<size_t>( request->output_width )*request->row_count;
   if ( pixels == 0 || destination_capacity < pixels )
   {
      CopyError( error_message, error_message_capacity,
                 "destination capacity is smaller than the requested band" );
      return OAF_NATIVE_BUFFER_TOO_SMALL;
   }
   return GuardedKernelCall(
      [&]()
      {
         using namespace openastroflow::native;
         WarpLanczos3Request native;
         native.source = std::span<const float>(
            request->source_samples, request->source_sample_count );
         native.sourceWidth = request->source_width;
         native.sourceHeight = request->source_height;
         native.inverse = AffineInverse{
            request->inverse[0], request->inverse[1], request->inverse[2],
            request->inverse[3], request->inverse[4], request->inverse[5] };
         native.outputWidth = request->output_width;
         native.firstRow = request->first_row;
         native.rowCount = request->row_count;
         native.domainScale = request->domain_scale;
         native.threads = request->threads;
         WarpLanczos3Clamped(
            native, std::span<float>( destination, destination_capacity ) );
      },
      error_message, error_message_capacity,
      "unknown native warp failure" );
}

extern "C" int oaf_native_cpu_warp_lanczos3_v2(
   const OafNativeWarpLanczos3RequestV2* request,
   float* destination,
   size_t destination_capacity,
   char* error_message,
   size_t error_message_capacity )
{
   if ( request == nullptr || destination == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "request and destination are required" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   if ( request->struct_size != sizeof( OafNativeWarpLanczos3RequestV2 )
     || request->source_samples == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "warp C ABI structure version or input buffer is invalid" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   const size_t pixels =
      static_cast<size_t>( request->output_width )*request->row_count;
   if ( pixels == 0 || destination_capacity < pixels )
   {
      CopyError( error_message, error_message_capacity,
                 "destination capacity is smaller than the requested band" );
      return OAF_NATIVE_BUFFER_TOO_SMALL;
   }
   return GuardedKernelCall(
      [&]()
      {
         using namespace openastroflow::native;
         WarpLanczos3Request native;
         native.source = std::span<const float>(
            request->source_samples, request->source_sample_count );
         native.sourceWidth = request->source_width;
         native.sourceHeight = request->source_height;
         native.inverse = AffineInverse{
            request->inverse[0], request->inverse[1], request->inverse[2],
            request->inverse[3], request->inverse[4], request->inverse[5],
            request->inverse[6], request->inverse[7], request->inverse[8] };
         native.outputWidth = request->output_width;
         native.firstRow = request->first_row;
         native.rowCount = request->row_count;
         native.domainScale = request->domain_scale;
         native.threads = request->threads;
         WarpLanczos3Clamped(
            native, std::span<float>( destination, destination_capacity ) );
      },
      error_message, error_message_capacity,
      "unknown native warp failure" );
}

extern "C" int oaf_native_cpu_mad_rejection_v1(
   const OafNativeMadRejectionRequestV1* request,
   uint8_t* accepted,
   size_t accepted_capacity,
   float* center,
   size_t center_capacity,
   char* error_message,
   size_t error_message_capacity )
{
   if ( request == nullptr || accepted == nullptr || center == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "request, accepted, and center buffers are required" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   if ( request->struct_size != sizeof( OafNativeMadRejectionRequestV1 )
     || request->frame_major_samples == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "MAD C ABI structure version or input buffer is invalid" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   const size_t pixels =
      static_cast<size_t>( request->row_count )*request->width;
   const size_t samples = pixels*request->frame_count;
   if ( pixels == 0 || accepted_capacity < samples || center_capacity < pixels )
   {
      CopyError( error_message, error_message_capacity,
                 "MAD output capacity is smaller than the requested tile" );
      return OAF_NATIVE_BUFFER_TOO_SMALL;
   }
   return GuardedKernelCall(
      [&]()
      {
         using namespace openastroflow::native;
         MadRejectionRequest native;
         native.frameMajorSamples = std::span<const float>(
            request->frame_major_samples, request->sample_count );
         native.frameCount = request->frame_count;
         native.rowCount = request->row_count;
         native.width = request->width;
         native.sigmaClip = request->sigma_clip;
         native.minimumRejectionFrames = request->minimum_rejection_frames;
         native.groupSigmaFloor = request->group_sigma_floor;
         native.absoluteFloor = request->absolute_floor;
         native.epsilonFloor = request->epsilon_floor;
         native.threads = request->threads;
         MadRejectionMask(
            native,
            std::span<uint8_t>( accepted, accepted_capacity ),
            std::span<float>( center, center_capacity ) );
      },
      error_message, error_message_capacity,
      "unknown native MAD rejection failure" );
}

extern "C" int oaf_native_cpu_mad_rejection_v2(
   const OafNativeMadRejectionRequestV2* request,
   uint8_t* accepted,
   size_t accepted_capacity,
   float* center,
   size_t center_capacity,
   char* error_message,
   size_t error_message_capacity )
{
   if ( request == nullptr || accepted == nullptr || center == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "request, accepted, and center buffers are required" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   if ( request->struct_size != sizeof( OafNativeMadRejectionRequestV2 )
     || request->frame_major_samples == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "MAD v2 C ABI structure version or input buffer is invalid" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   if ( (request->frame_scales == nullptr) != (request->frame_scale_count == 0) )
   {
      CopyError( error_message, error_message_capacity,
                 "MAD v2 frame scales pointer and count disagree" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   const size_t pixels =
      static_cast<size_t>( request->row_count )*request->width;
   const size_t samples = pixels*request->frame_count;
   if ( pixels == 0 || accepted_capacity < samples || center_capacity < pixels )
   {
      CopyError( error_message, error_message_capacity,
                 "MAD output capacity is smaller than the requested tile" );
      return OAF_NATIVE_BUFFER_TOO_SMALL;
   }
   return GuardedKernelCall(
      [&]()
      {
         using namespace openastroflow::native;
         MadRejectionRequest native;
         native.frameMajorSamples = std::span<const float>(
            request->frame_major_samples, request->sample_count );
         native.frameCount = request->frame_count;
         native.rowCount = request->row_count;
         native.width = request->width;
         native.sigmaClip = request->sigma_clip;
         native.minimumRejectionFrames = request->minimum_rejection_frames;
         native.groupSigmaFloor = request->group_sigma_floor;
         native.absoluteFloor = request->absolute_floor;
         native.epsilonFloor = request->epsilon_floor;
         native.threads = request->threads;
         if ( request->frame_scales != nullptr )
            native.frameScales = std::span<const float>(
               request->frame_scales, request->frame_scale_count );
         native.poolHalfWidth = request->pool_half_width;
         MadRejectionMask(
            native,
            std::span<uint8_t>( accepted, accepted_capacity ),
            std::span<float>( center, center_capacity ) );
      },
      error_message, error_message_capacity,
      "unknown native MAD rejection failure" );
}

extern "C" int oaf_native_cpu_masked_mean_v1(
   const OafNativeMaskedMeanRequestV1* request,
   OafNativeMaskedMeanOutputV1* output,
   char* error_message,
   size_t error_message_capacity )
{
   if ( request == nullptr || output == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "request and output are required" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   if ( request->struct_size != sizeof( OafNativeMaskedMeanRequestV1 )
     || output->struct_size != sizeof( OafNativeMaskedMeanOutputV1 )
     || request->frame_major_samples == nullptr
     || request->frame_major_accepted == nullptr
     || request->frame_weights == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "masked mean C ABI structure version or input buffer is invalid" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   const size_t pixels =
      static_cast<size_t>( request->row_count )*request->width;
   if ( pixels == 0 || output->pixel_capacity < pixels
     || output->integrated == nullptr
     || output->accepted_samples == nullptr
     || output->rejected_samples == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "masked mean output capacity is smaller than the requested tile" );
      return OAF_NATIVE_BUFFER_TOO_SMALL;
   }
   return GuardedKernelCall(
      [&]()
      {
         using namespace openastroflow::native;
         MaskedMeanRequest native;
         native.frameMajorSamples = std::span<const float>(
            request->frame_major_samples, request->sample_count );
         native.frameMajorAccepted = std::span<const uint8_t>(
            request->frame_major_accepted, request->accepted_count );
         native.frameWeights = std::span<const double>(
            request->frame_weights, request->weight_count );
         native.frameCount = request->frame_count;
         native.rowCount = request->row_count;
         native.width = request->width;
         native.threads = request->threads;
         const MaskedMeanOutput destination{
            std::span<float>( output->integrated, output->pixel_capacity ),
            std::span<uint16_t>(
               output->accepted_samples, output->pixel_capacity ),
            std::span<uint16_t>(
               output->rejected_samples, output->pixel_capacity ) };
         MaskedWeightedMean( native, destination );
      },
      error_message, error_message_capacity,
      "unknown native masked mean failure" );
}

extern "C" int oaf_native_cpu_tile_offsets_v1(
   const OafNativeTileOffsetRequestV1* request,
   OafNativeTileOffsetOutputV1* output,
   char* error_message,
   size_t error_message_capacity )
{
   if ( request == nullptr || output == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "request and output are required" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   if ( request->struct_size != sizeof( OafNativeTileOffsetRequestV1 )
     || output->struct_size != sizeof( OafNativeTileOffsetOutputV1 )
     || request->target == nullptr || request->reference == nullptr
     || request->boundaries == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "tile offset C ABI structure version or input buffer is invalid" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   if ( request->tile_count == 0 || output->tile_capacity < request->tile_count
     || output->offset == nullptr || output->count == nullptr
     || output->residual_mad == nullptr || output->valid == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "tile offset output capacity is smaller than the request" );
      return OAF_NATIVE_BUFFER_TOO_SMALL;
   }
   return GuardedKernelCall(
      [&]()
      {
         using namespace openastroflow::native;
         TileOffsetRequest native;
         native.target = std::span<const double>( request->target, request->sample_count );
         native.reference = std::span<const double>( request->reference, request->sample_count );
         native.boundaries = std::span<const std::uint64_t>(
            request->boundaries, request->boundary_count );
         native.tileCount = request->tile_count;
         native.scale = request->scale;
         native.lowerQuantile = request->lower_quantile;
         native.upperQuantile = request->upper_quantile;
         native.minimumSamples = request->minimum_samples;
         native.residualClipSigma = request->residual_clip_sigma;
         native.threads = request->threads;
         const TileOffsetOutput destination{
            std::span<double>( output->offset, output->tile_capacity ),
            std::span<uint32_t>( output->count, output->tile_capacity ),
            std::span<double>( output->residual_mad, output->tile_capacity ),
            std::span<uint8_t>( output->valid, output->tile_capacity ) };
         TileOffsets( native, destination );
      },
      error_message, error_message_capacity,
      "unknown native tile offset failure" );
}

extern "C" uint32_t oaf_native_default_kernel_threads_v1(void)
{
   return openastroflow::native::DefaultKernelThreads();
}

extern "C" int oaf_native_cpu_features_v1(
   OafNativeCpuFeaturesV1* features,
   char* error_message,
   size_t error_message_capacity )
{
   if ( features == nullptr || features->struct_size != sizeof( OafNativeCpuFeaturesV1 ) )
   {
      CopyError( error_message, error_message_capacity,
                 "cpu features struct size mismatch" );
      return OAF_NATIVE_INVALID_ARGUMENT;
   }
   try
   {
      const openastroflow::native::CpuFeatures detected =
         openastroflow::native::DetectCpuFeatures();
      features->architecture = OAF_NATIVE_CPU_ARCHITECTURE_UNKNOWN;
      if ( detected.architecture == "x86-64" )
         features->architecture = OAF_NATIVE_CPU_ARCHITECTURE_X86_64;
      else if ( detected.architecture == "arm64" )
         features->architecture = OAF_NATIVE_CPU_ARCHITECTURE_ARM64;
      CopyFixed( features->features, sizeof( features->features ),
                 openastroflow::native::JoinFeatures( detected ) );
      CopyFixed( features->brand, sizeof( features->brand ), detected.brand );
      CopyError( error_message, error_message_capacity, {} );
      return OAF_NATIVE_OK;
   }
   catch ( const std::exception& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return OAF_NATIVE_EXECUTION_FAILED;
   }
   catch ( ... )
   {
      CopyError( error_message, error_message_capacity,
                 "unknown cpu features failure" );
      return OAF_NATIVE_EXECUTION_FAILED;
   }
}
