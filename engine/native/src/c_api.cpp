#include "ufwbpp/c_api.h"
#include "Lanczos3Table.h"

#include "ufwbpp/FusedIntegration.h"
#include "ufwbpp/CpuFeatures.h"
#include "ufwbpp/PortableKernels.h"
#if defined(UFWBPP_WITH_METAL_ABI)
#include "ufwbpp/MetalFusedIntegration.h"
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
#include <vector>

struct UfwbppNativeMetalExecutorV1
{
#if defined(UFWBPP_WITH_METAL_ABI)
   std::unique_ptr<ufwbpp::native::MetalFusedIntegrationExecutor>
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

ufwbpp::native::NativeLinearFitIntegrationRequest MakeRequest(
   const UfwbppNativeIntegrationRequestV1& input )
{
   if ( input.struct_size != sizeof( UfwbppNativeIntegrationRequestV1 )
     || input.frame_major_samples == nullptr
     || input.frame_major_scale_grid == nullptr
     || input.frame_major_zero_offset_grid == nullptr
     || input.frame_weights == nullptr )
      throw std::invalid_argument(
         "C ABI structure version or input buffer is invalid" );
   using namespace ufwbpp::native;
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

ufwbpp::native::FusedIntegrationRequest MakeMaskedRequest(
   const UfwbppNativeMaskedIntegrationRequestV1& input )
{
   if ( input.struct_size != sizeof( UfwbppNativeMaskedIntegrationRequestV1 )
     || input.frame_major_samples == nullptr
     || input.frame_major_rejection_mask == nullptr
     || input.frame_major_scale_grid == nullptr
     || input.frame_major_zero_offset_grid == nullptr
     || input.frame_weights == nullptr
     || input.rejection_bits == 0 || input.rejection_bits > 0xffU )
      throw std::invalid_argument(
         "masked C ABI structure version or input buffer is invalid" );
   using namespace ufwbpp::native;
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
                    const UfwbppNativeIntegrationOutputV1& output )
{
   if ( output.struct_size != sizeof( UfwbppNativeIntegrationOutputV1 ) )
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   const std::size_t pixels =
      static_cast<std::size_t>( input.width )*input.row_count;
   if ( pixels == 0 || output.pixel_capacity < pixels
     || output.integrated == nullptr
     || output.accepted_samples == nullptr
     || output.rejected_samples == nullptr )
      return UFWBPP_NATIVE_BUFFER_TOO_SMALL;
   return UFWBPP_NATIVE_OK;
}

void CopyResult(
   const ufwbpp::native::FusedIntegrationResult& result,
   UfwbppNativeIntegrationOutputV1& output )
{
   std::copy( result.integrated.begin(), result.integrated.end(),
              output.integrated );
   std::copy( result.acceptedSamples.begin(), result.acceptedSamples.end(),
              output.accepted_samples );
   std::copy( result.rejectedSamples.begin(), result.rejectedSamples.end(),
              output.rejected_samples );
}

int ExecuteLinearFit( const UfwbppNativeIntegrationRequestV1& input,
                      UfwbppNativeIntegrationOutputV1& output )
{
   const int outputStatus = ValidateOutput( input, output );
   if ( outputStatus != UFWBPP_NATIVE_OK )
      return outputStatus;
   const auto request = MakeRequest( input );
   const auto result =
      ufwbpp::native::RunCpuOracleNativeLinearFitIntegration( request );
   CopyResult( result, output );
   return UFWBPP_NATIVE_OK;
}

int ExecuteMaskedWeighted(
   const UfwbppNativeMaskedIntegrationRequestV1& input,
   UfwbppNativeIntegrationOutputV1& output )
{
   const int outputStatus = ValidateOutput( input, output );
   if ( outputStatus != UFWBPP_NATIVE_OK )
      return outputStatus;
   const auto request = MakeMaskedRequest( input );
   const auto result =
      ufwbpp::native::RunCpuOracleFusedIntegration( request );
   CopyResult( result, output );
   return UFWBPP_NATIVE_OK;
}

#if defined(UFWBPP_WITH_METAL_ABI)
void CopyStats( const ufwbpp::native::MetalExecutionStats& source,
                UfwbppNativeExecutionStatsV1& destination ) noexcept
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

extern "C" uint32_t ufwbpp_native_abi_version(void)
{
   return UFWBPP_NATIVE_ABI_VERSION;
}

extern "C" int ufwbpp_native_cpu_linear_fit_v1(
   const UfwbppNativeIntegrationRequestV1* request,
   UfwbppNativeIntegrationOutputV1* output,
   char* error_message,
   size_t error_message_capacity )
{
   if ( request == nullptr || output == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "request and output are required" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   try
   {
      const int status = ExecuteLinearFit( *request, *output );
      if ( status == UFWBPP_NATIVE_INVALID_ARGUMENT )
         CopyError( error_message, error_message_capacity,
                    "C ABI structure version or request is invalid" );
      else if ( status == UFWBPP_NATIVE_BUFFER_TOO_SMALL )
         CopyError( error_message, error_message_capacity,
                    "output capacity is smaller than the requested tile" );
      else
         CopyError( error_message, error_message_capacity, {} );
      return status;
   }
   catch ( const std::invalid_argument& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   catch ( const std::exception& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return UFWBPP_NATIVE_EXECUTION_FAILED;
   }
   catch ( ... )
   {
      CopyError( error_message, error_message_capacity,
                 "unknown native execution failure" );
      return UFWBPP_NATIVE_EXECUTION_FAILED;
   }
}

extern "C" int ufwbpp_native_cpu_masked_weighted_v1(
   const UfwbppNativeMaskedIntegrationRequestV1* request,
   UfwbppNativeIntegrationOutputV1* output,
   char* error_message,
   size_t error_message_capacity )
{
   if ( request == nullptr || output == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "request and output are required" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   try
   {
      const int status = ExecuteMaskedWeighted( *request, *output );
      if ( status == UFWBPP_NATIVE_BUFFER_TOO_SMALL )
         CopyError( error_message, error_message_capacity,
                    "output capacity is smaller than the requested tile" );
      else
         CopyError( error_message, error_message_capacity, {} );
      return status;
   }
   catch ( const std::invalid_argument& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   catch ( const std::exception& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return UFWBPP_NATIVE_EXECUTION_FAILED;
   }
   catch ( ... )
   {
      CopyError( error_message, error_message_capacity,
                 "unknown native masked execution failure" );
      return UFWBPP_NATIVE_EXECUTION_FAILED;
   }
}

extern "C" int ufwbpp_native_metal_available_v1(
   uint32_t* available,
   char* error_message,
   size_t error_message_capacity )
{
   if ( available == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "available output is required" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
#if defined(UFWBPP_WITH_METAL_ABI)
   *available = ufwbpp::native::MetalFusedIntegrationAvailable()
      ? 1U : 0U;
#else
   *available = 0;
#endif
   CopyError( error_message, error_message_capacity, {} );
   return UFWBPP_NATIVE_OK;
}

extern "C" int ufwbpp_native_metal_executor_create_v1(
   const char* metal_source_path,
   UfwbppNativeMetalExecutorV1** executor,
   char* error_message,
   size_t error_message_capacity )
{
   if ( executor == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "executor output is required" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   *executor = nullptr;
#if defined(UFWBPP_WITH_METAL_ABI)
   try
   {
      auto result = std::make_unique<UfwbppNativeMetalExecutorV1>();
      const std::filesystem::path source =
         metal_source_path == nullptr ? std::filesystem::path()
         : std::filesystem::path( metal_source_path );
      result->implementation = std::make_unique<
         ufwbpp::native::MetalFusedIntegrationExecutor>( source );
      *executor = result.release();
      CopyError( error_message, error_message_capacity, {} );
      return UFWBPP_NATIVE_OK;
   }
   catch ( const std::invalid_argument& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   catch ( const std::exception& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return UFWBPP_NATIVE_EXECUTION_FAILED;
   }
   catch ( ... )
   {
      CopyError( error_message, error_message_capacity,
                 "unknown Metal executor creation failure" );
      return UFWBPP_NATIVE_EXECUTION_FAILED;
   }
#else
   (void)metal_source_path;
   CopyError( error_message, error_message_capacity,
              "Metal backend was not compiled into this native library" );
   return UFWBPP_NATIVE_BACKEND_UNAVAILABLE;
#endif
}

extern "C" void ufwbpp_native_metal_executor_destroy_v1(
   UfwbppNativeMetalExecutorV1* executor )
{
   delete executor;
}

extern "C" int ufwbpp_native_metal_linear_fit_v1(
   UfwbppNativeMetalExecutorV1* executor,
   const UfwbppNativeIntegrationRequestV1* request,
   UfwbppNativeIntegrationOutputV1* output,
   UfwbppNativeExecutionStatsV1* stats,
   char* error_message,
   size_t error_message_capacity )
{
   if ( executor == nullptr || request == nullptr || output == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "executor, request, and output are required" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   if ( stats != nullptr
     && stats->struct_size != sizeof( UfwbppNativeExecutionStatsV1 ) )
   {
      CopyError( error_message, error_message_capacity,
                 "execution stats structure version is invalid" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
#if defined(UFWBPP_WITH_METAL_ABI)
   if ( !executor->implementation )
   {
      CopyError( error_message, error_message_capacity,
                 "Metal executor is not initialized" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   try
   {
      const int outputStatus = ValidateOutput( *request, *output );
      if ( outputStatus != UFWBPP_NATIVE_OK )
         return outputStatus;
      const auto nativeRequest = MakeRequest( *request );
      ufwbpp::native::MetalExecutionStats nativeStats;
      const auto result = executor->implementation->RunNativeLinearFitRejection(
         nativeRequest, stats == nullptr ? nullptr : &nativeStats );
      CopyResult( result, *output );
      if ( stats != nullptr )
         CopyStats( nativeStats, *stats );
      CopyError( error_message, error_message_capacity, {} );
      return UFWBPP_NATIVE_OK;
   }
   catch ( const std::invalid_argument& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   catch ( const std::exception& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return UFWBPP_NATIVE_EXECUTION_FAILED;
   }
   catch ( ... )
   {
      CopyError( error_message, error_message_capacity,
                 "unknown Metal execution failure" );
      return UFWBPP_NATIVE_EXECUTION_FAILED;
   }
#else
   (void)stats;
   CopyError( error_message, error_message_capacity,
              "Metal backend was not compiled into this native library" );
   return UFWBPP_NATIVE_BACKEND_UNAVAILABLE;
#endif
}

extern "C" int ufwbpp_native_metal_masked_weighted_v1(
   UfwbppNativeMetalExecutorV1* executor,
   const UfwbppNativeMaskedIntegrationRequestV1* request,
   UfwbppNativeIntegrationOutputV1* output,
   UfwbppNativeExecutionStatsV1* stats,
   char* error_message,
   size_t error_message_capacity )
{
   if ( executor == nullptr || request == nullptr || output == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "executor, request, and output are required" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   if ( stats != nullptr
     && stats->struct_size != sizeof( UfwbppNativeExecutionStatsV1 ) )
   {
      CopyError( error_message, error_message_capacity,
                 "execution stats structure version is invalid" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
#if defined(UFWBPP_WITH_METAL_ABI)
   if ( !executor->implementation )
   {
      CopyError( error_message, error_message_capacity,
                 "Metal executor is not initialized" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   try
   {
      const int outputStatus = ValidateOutput( *request, *output );
      if ( outputStatus != UFWBPP_NATIVE_OK )
         return outputStatus;
      const auto nativeRequest = MakeMaskedRequest( *request );
      ufwbpp::native::MetalExecutionStats nativeStats;
      const auto result = executor->implementation->Run(
         nativeRequest, stats == nullptr ? nullptr : &nativeStats );
      CopyResult( result, *output );
      if ( stats != nullptr )
         CopyStats( nativeStats, *stats );
      CopyError( error_message, error_message_capacity, {} );
      return UFWBPP_NATIVE_OK;
   }
   catch ( const std::invalid_argument& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   catch ( const std::exception& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return UFWBPP_NATIVE_EXECUTION_FAILED;
   }
   catch ( ... )
   {
      CopyError( error_message, error_message_capacity,
                 "unknown masked Metal execution failure" );
      return UFWBPP_NATIVE_EXECUTION_FAILED;
   }
#else
   (void)stats;
   CopyError( error_message, error_message_capacity,
              "Metal backend was not compiled into this native library" );
   return UFWBPP_NATIVE_BACKEND_UNAVAILABLE;
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
      return UFWBPP_NATIVE_OK;
   }
   catch ( const std::invalid_argument& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   catch ( const std::exception& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return UFWBPP_NATIVE_EXECUTION_FAILED;
   }
   catch ( ... )
   {
      CopyError( error_message, error_message_capacity, unknown_failure );
      return UFWBPP_NATIVE_EXECUTION_FAILED;
   }
}

} // namespace

extern "C" int ufwbpp_native_cpu_warp_lanczos3_v1(
   const UfwbppNativeWarpLanczos3RequestV1* request,
   float* destination,
   size_t destination_capacity,
   char* error_message,
   size_t error_message_capacity )
{
   if ( request == nullptr || destination == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "request and destination are required" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   if ( request->struct_size != sizeof( UfwbppNativeWarpLanczos3RequestV1 )
     || request->source_samples == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "warp C ABI structure version or input buffer is invalid" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   const size_t pixels =
      static_cast<size_t>( request->output_width )*request->row_count;
   if ( pixels == 0 || destination_capacity < pixels )
   {
      CopyError( error_message, error_message_capacity,
                 "destination capacity is smaller than the requested band" );
      return UFWBPP_NATIVE_BUFFER_TOO_SMALL;
   }
   return GuardedKernelCall(
      [&]()
      {
         using namespace ufwbpp::native;
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

extern "C" int ufwbpp_native_cpu_warp_lanczos3_v2(
   const UfwbppNativeWarpLanczos3RequestV2* request,
   float* destination,
   size_t destination_capacity,
   char* error_message,
   size_t error_message_capacity )
{
   if ( request == nullptr || destination == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "request and destination are required" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   if ( request->struct_size != sizeof( UfwbppNativeWarpLanczos3RequestV2 )
     || request->source_samples == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "warp C ABI structure version or input buffer is invalid" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   const size_t pixels =
      static_cast<size_t>( request->output_width )*request->row_count;
   if ( pixels == 0 || destination_capacity < pixels )
   {
      CopyError( error_message, error_message_capacity,
                 "destination capacity is smaller than the requested band" );
      return UFWBPP_NATIVE_BUFFER_TOO_SMALL;
   }
   return GuardedKernelCall(
      [&]()
      {
         using namespace ufwbpp::native;
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

extern "C" int ufwbpp_native_cpu_mad_rejection_v1(
   const UfwbppNativeMadRejectionRequestV1* request,
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
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   if ( request->struct_size != sizeof( UfwbppNativeMadRejectionRequestV1 )
     || request->frame_major_samples == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "MAD C ABI structure version or input buffer is invalid" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   const size_t pixels =
      static_cast<size_t>( request->row_count )*request->width;
   const size_t samples = pixels*request->frame_count;
   if ( pixels == 0 || accepted_capacity < samples || center_capacity < pixels )
   {
      CopyError( error_message, error_message_capacity,
                 "MAD output capacity is smaller than the requested tile" );
      return UFWBPP_NATIVE_BUFFER_TOO_SMALL;
   }
   return GuardedKernelCall(
      [&]()
      {
         using namespace ufwbpp::native;
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

extern "C" int ufwbpp_native_cpu_mad_rejection_v2(
   const UfwbppNativeMadRejectionRequestV2* request,
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
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   if ( request->struct_size != sizeof( UfwbppNativeMadRejectionRequestV2 )
     || request->frame_major_samples == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "MAD v2 C ABI structure version or input buffer is invalid" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   if ( (request->frame_scales == nullptr) != (request->frame_scale_count == 0) )
   {
      CopyError( error_message, error_message_capacity,
                 "MAD v2 frame scales pointer and count disagree" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   const size_t pixels =
      static_cast<size_t>( request->row_count )*request->width;
   const size_t samples = pixels*request->frame_count;
   if ( pixels == 0 || accepted_capacity < samples || center_capacity < pixels )
   {
      CopyError( error_message, error_message_capacity,
                 "MAD output capacity is smaller than the requested tile" );
      return UFWBPP_NATIVE_BUFFER_TOO_SMALL;
   }
   return GuardedKernelCall(
      [&]()
      {
         using namespace ufwbpp::native;
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

extern "C" int ufwbpp_native_cpu_masked_mean_v1(
   const UfwbppNativeMaskedMeanRequestV1* request,
   UfwbppNativeMaskedMeanOutputV1* output,
   char* error_message,
   size_t error_message_capacity )
{
   if ( request == nullptr || output == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "request and output are required" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   if ( request->struct_size != sizeof( UfwbppNativeMaskedMeanRequestV1 )
     || output->struct_size != sizeof( UfwbppNativeMaskedMeanOutputV1 )
     || request->frame_major_samples == nullptr
     || request->frame_major_accepted == nullptr
     || request->frame_weights == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "masked mean C ABI structure version or input buffer is invalid" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
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
      return UFWBPP_NATIVE_BUFFER_TOO_SMALL;
   }
   return GuardedKernelCall(
      [&]()
      {
         using namespace ufwbpp::native;
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

extern "C" int ufwbpp_native_cpu_masked_mean_v2(
   const UfwbppNativeMaskedMeanRequestV2* request,
   UfwbppNativeMaskedMeanOutputV1* output,
   char* error_message,
   size_t error_message_capacity )
{
   if ( request == nullptr || output == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "request and output are required" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   if ( request->struct_size != sizeof( UfwbppNativeMaskedMeanRequestV2 )
     || output->struct_size != sizeof( UfwbppNativeMaskedMeanOutputV1 )
     || request->frame_major_samples == nullptr
     || request->frame_major_accepted == nullptr
     || request->frame_weights == nullptr
     || (request->frame_major_sample_weights == nullptr
         && request->sample_weight_count != 0) )
   {
      CopyError( error_message, error_message_capacity,
                 "masked mean V2 C ABI structure version or input buffer is invalid" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
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
      return UFWBPP_NATIVE_BUFFER_TOO_SMALL;
   }
   return GuardedKernelCall(
      [&]()
      {
         using namespace ufwbpp::native;
         MaskedMeanRequest native;
         native.frameMajorSamples = std::span<const float>(
            request->frame_major_samples, request->sample_count );
         native.frameMajorAccepted = std::span<const uint8_t>(
            request->frame_major_accepted, request->accepted_count );
         native.frameWeights = std::span<const double>(
            request->frame_weights, request->weight_count );
         if ( request->frame_major_sample_weights != nullptr )
            native.frameMajorSampleWeights = std::span<const float>(
               request->frame_major_sample_weights,
               request->sample_weight_count );
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

extern "C" int ufwbpp_native_cpu_tile_offsets_v1(
   const UfwbppNativeTileOffsetRequestV1* request,
   UfwbppNativeTileOffsetOutputV1* output,
   char* error_message,
   size_t error_message_capacity )
{
   if ( request == nullptr || output == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "request and output are required" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   if ( request->struct_size != sizeof( UfwbppNativeTileOffsetRequestV1 )
     || output->struct_size != sizeof( UfwbppNativeTileOffsetOutputV1 )
     || request->target == nullptr || request->reference == nullptr
     || request->boundaries == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "tile offset C ABI structure version or input buffer is invalid" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   if ( request->tile_count == 0 || output->tile_capacity < request->tile_count
     || output->offset == nullptr || output->count == nullptr
     || output->residual_mad == nullptr || output->valid == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "tile offset output capacity is smaller than the request" );
      return UFWBPP_NATIVE_BUFFER_TOO_SMALL;
   }
   return GuardedKernelCall(
      [&]()
      {
         using namespace ufwbpp::native;
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

extern "C" int ufwbpp_native_cpu_radon_peaks_v1(
   const UfwbppNativeRadonPeakRequestV1* request,
   UfwbppNativeRadonPeakOutputV1* output,
   char* error_message,
   size_t error_message_capacity )
{
   if ( request == nullptr || output == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "request and output are required" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   if ( request->struct_size != sizeof( UfwbppNativeRadonPeakRequestV1 )
     || output->struct_size != sizeof( UfwbppNativeRadonPeakOutputV1 )
     || request->image == nullptr || request->weight == nullptr
     || (output->peaks == nullptr && output->peak_capacity != 0) )
   {
      CopyError( error_message, error_message_capacity,
                 "radon peak C ABI structure version or input buffer is invalid" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   output->peak_count = 0;
   std::vector<ufwbpp::native::RadonPeak> peaks;
   const int status = GuardedKernelCall(
      [&]()
      {
         using namespace ufwbpp::native;
         RadonPeakRequest native;
         native.image = std::span<const float>( request->image, request->image_count );
         native.weight = std::span<const uint8_t>( request->weight, request->weight_count );
         native.width = request->width;
         native.height = request->height;
         native.size = request->size;
         native.minimumRows = request->minimum_rows;
         native.detectionZ = request->detection_z;
         native.minimumCoverage = request->minimum_coverage;
         native.minimumCount = request->minimum_count;
         native.minimumScaleSamples = request->minimum_scale_samples;
         native.threads = request->threads;
         RadonLinePeaks( native, peaks );
      },
      error_message, error_message_capacity,
      "unknown native radon peak failure" );
   if ( status != UFWBPP_NATIVE_OK )
      return status;
   output->peak_count = peaks.size();
   if ( peaks.size() > output->peak_capacity )
   {
      CopyError( error_message, error_message_capacity,
                 "radon peak output capacity is smaller than the number of peaks" );
      return UFWBPP_NATIVE_BUFFER_TOO_SMALL;
   }
   for ( size_t i = 0; i < peaks.size(); ++i )
   {
      UfwbppNativeRadonPeakV1& out = output->peaks[i];
      out.level = peaks[i].level;
      out.block = peaks[i].block;
      out.shift_index = peaks[i].shiftIndex;
      out.column = peaks[i].column;
      out.z = peaks[i].z;
      out.reserved = 0;
   }
   return UFWBPP_NATIVE_OK;
}

extern "C" int ufwbpp_native_cpu_drizzle_v1(
   const UfwbppNativeDrizzleRequestV1* request,
   char* error_message,
   size_t error_message_capacity )
{
   if ( request == nullptr )
   {
      CopyError( error_message, error_message_capacity, "request is required" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   if ( request->struct_size != sizeof( UfwbppNativeDrizzleRequestV1 )
     || request->source == nullptr || request->output_sum == nullptr
     || request->output_weight == nullptr
     || (request->grid == nullptr && request->grid_count != 0)
     || (request->weight_grid == nullptr && request->weight_grid_count != 0)
     || (request->mask == nullptr && request->mask_count != 0) )
   {
      CopyError( error_message, error_message_capacity,
                 "drizzle C ABI structure version or input buffer is invalid" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   return GuardedKernelCall(
      [&]()
      {
         using namespace ufwbpp::native;
         DrizzleRequest native;
         native.source = std::span<const float>( request->source, request->source_count );
         native.sourceWidth = request->source_width;
         native.sourceRows = request->source_rows;
         native.sourceRow0 = request->source_row0;
         std::copy( request->forward, request->forward + 9, native.forward );
         native.scale = request->scale;
         native.pixfrac = request->pixfrac;
         native.kernel = static_cast<DrizzleKernel>( request->kernel );
         native.normalizationScale = request->normalization_scale;
         native.normalizationOffset = request->normalization_offset;
         if ( request->grid_count != 0 )
         {
            native.grid = std::span<const double>( request->grid, request->grid_count );
            native.gridXNodes = std::span<const double>( request->grid_x_nodes, request->grid_x_count );
            native.gridYNodes = std::span<const double>( request->grid_y_nodes, request->grid_y_count );
         }
         if ( request->weight_grid_count != 0 )
         {
            native.weightGrid = std::span<const double>( request->weight_grid, request->weight_grid_count );
            native.weightGridXNodes = std::span<const double>( request->weight_grid_x_nodes, request->weight_grid_x_count );
            native.weightGridYNodes = std::span<const double>( request->weight_grid_y_nodes, request->weight_grid_y_count );
         }
         if ( request->mask_count != 0 )
         {
            native.mask = std::span<const uint8_t>( request->mask, request->mask_count );
            native.maskWidth = request->mask_width;
            native.maskHeight = request->mask_height;
         }
         std::copy( request->cfa_pattern, request->cfa_pattern + 4, native.cfaPattern );
         native.channel = request->channel;
         native.frameWeight = request->frame_weight;
         native.outputWidth = request->output_width;
         native.outputRows = request->output_rows;
         native.outputRow0 = request->output_row0;
         native.outputSum = std::span<double>( request->output_sum, request->output_count );
         native.outputWeight = std::span<double>( request->output_weight, request->output_count );
         if ( request->output_touched != nullptr )
            native.outputTouched = std::span<uint8_t>( request->output_touched, request->output_count );
         native.threads = request->threads;
         DrizzleBand( native );
      },
      error_message, error_message_capacity,
      "unknown native drizzle failure" );
}

extern "C" int ufwbpp_native_lanczos3_table_v1(
   double* values,
   size_t capacity,
   uint32_t* node_count,
   char* error_message,
   size_t error_message_capacity )
{
   using ufwbpp::native::detail::Lanczos3TableNodeValues;
   using ufwbpp::native::detail::Lanczos3TableNodes;
   if ( node_count == nullptr )
   {
      CopyError( error_message, error_message_capacity, "node_count is required" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   *node_count = Lanczos3TableNodes;
   const size_t required = static_cast<size_t>( Lanczos3TableNodes )*6;
   if ( values == nullptr || capacity < required )
   {
      CopyError( error_message, error_message_capacity,
                 "Lanczos-3 table buffer is smaller than the table" );
      return UFWBPP_NATIVE_BUFFER_TOO_SMALL;
   }
   return GuardedKernelCall(
      [&]()
      {
         const double* table = Lanczos3TableNodeValues();
         std::copy( table, table + required, values );
      },
      error_message, error_message_capacity,
      "unknown native Lanczos-3 table failure" );
}

extern "C" int ufwbpp_native_cpu_debayer_bilinear_v1(
   const UfwbppNativeDebayerRequestV1* request,
   char* error_message,
   size_t error_message_capacity )
{
   if ( request == nullptr )
   {
      CopyError( error_message, error_message_capacity, "request is required" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   if ( request->struct_size != sizeof( UfwbppNativeDebayerRequestV1 )
     || request->mosaic == nullptr || request->planes == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "debayer C ABI structure version or input buffer is invalid" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   return GuardedKernelCall(
      [&]()
      {
         using namespace ufwbpp::native;
         DebayerRequest native;
         native.mosaic = std::span<const float>( request->mosaic, request->mosaic_count );
         native.width = request->width;
         native.height = request->height;
         std::copy( request->pattern, request->pattern + 4, native.pattern );
         native.planes = std::span<float>( request->planes, request->plane_count );
         native.threads = request->threads;
         DebayerBilinear( native );
      },
      error_message, error_message_capacity,
      "unknown native debayer failure" );
}

extern "C" int ufwbpp_native_cpu_add_offset_grid_v1(
   const UfwbppNativeOffsetGridRequestV1* request,
   char* error_message,
   size_t error_message_capacity )
{
   if ( request == nullptr )
   {
      CopyError( error_message, error_message_capacity, "request is required" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   if ( request->struct_size != sizeof( UfwbppNativeOffsetGridRequestV1 )
     || request->values == nullptr || request->rows == nullptr || request->grid == nullptr
     || request->x_nodes == nullptr || request->y_nodes == nullptr )
   {
      CopyError( error_message, error_message_capacity,
                 "offset grid C ABI structure version or input buffer is invalid" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   return GuardedKernelCall(
      [&]()
      {
         using namespace ufwbpp::native;
         OffsetGridRequest native;
         native.values = std::span<float>( request->values, request->value_count );
         native.width = request->width;
         native.rows = std::span<const std::int64_t>( request->rows, request->row_count );
         native.grid = std::span<const double>( request->grid, request->grid_count );
         native.xNodes = std::span<const double>( request->x_nodes, request->x_node_count );
         native.yNodes = std::span<const double>( request->y_nodes, request->y_node_count );
         native.threads = request->threads;
         AddOffsetGrid( native );
      },
      error_message, error_message_capacity,
      "unknown native offset grid failure" );
}

extern "C" uint32_t ufwbpp_native_default_kernel_threads_v1(void)
{
   return ufwbpp::native::DefaultKernelThreads();
}

extern "C" int ufwbpp_native_cpu_features_v1(
   UfwbppNativeCpuFeaturesV1* features,
   char* error_message,
   size_t error_message_capacity )
{
   if ( features == nullptr || features->struct_size != sizeof( UfwbppNativeCpuFeaturesV1 ) )
   {
      CopyError( error_message, error_message_capacity,
                 "cpu features struct size mismatch" );
      return UFWBPP_NATIVE_INVALID_ARGUMENT;
   }
   try
   {
      const ufwbpp::native::CpuFeatures detected =
         ufwbpp::native::DetectCpuFeatures();
      features->architecture = UFWBPP_NATIVE_CPU_ARCHITECTURE_UNKNOWN;
      if ( detected.architecture == "x86-64" )
         features->architecture = UFWBPP_NATIVE_CPU_ARCHITECTURE_X86_64;
      else if ( detected.architecture == "arm64" )
         features->architecture = UFWBPP_NATIVE_CPU_ARCHITECTURE_ARM64;
      CopyFixed( features->features, sizeof( features->features ),
                 ufwbpp::native::JoinFeatures( detected ) );
      CopyFixed( features->brand, sizeof( features->brand ), detected.brand );
      CopyError( error_message, error_message_capacity, {} );
      return UFWBPP_NATIVE_OK;
   }
   catch ( const std::exception& error )
   {
      CopyError( error_message, error_message_capacity, error.what() );
      return UFWBPP_NATIVE_EXECUTION_FAILED;
   }
   catch ( ... )
   {
      CopyError( error_message, error_message_capacity,
                 "unknown cpu features failure" );
      return UFWBPP_NATIVE_EXECUTION_FAILED;
   }
}
