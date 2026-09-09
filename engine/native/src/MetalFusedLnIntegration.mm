#include "openastroflow/MetalFusedLnIntegration.h"
#include "EmbeddedMetalSource.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace openastroflow::native
{

namespace
{

struct FusedParameters
{
   std::uint32_t width;
   std::uint32_t referenceHeight;
   std::uint32_t firstRow;
   std::uint32_t rowCount;
   std::uint32_t frameCount;
   std::uint32_t gridWidth;
   std::uint32_t gridHeight;
   std::uint32_t rejectionBits;
   float outputScale;
   float outputOffset;
};

struct RangeParameters
{
   std::uint32_t sampleCount;
};

struct OutputNormalizationParameters
{
   std::uint32_t sampleCount;
   float scale;
   float offset;
};

struct RobustParameters
{
   std::uint32_t width;
   std::uint32_t referenceHeight;
   std::uint32_t firstRow;
   std::uint32_t rowCount;
   std::uint32_t frameCount;
   std::uint32_t gridWidth;
   std::uint32_t gridHeight;
   std::uint32_t winsorIterations;
   float rangeLow;
   float lowSigma;
   float highSigma;
   float winsorSigma;
   float outputScale;
   float outputOffset;
};

struct LinearFitParameters
{
   std::uint32_t width;
   std::uint32_t referenceHeight;
   std::uint32_t firstRow;
   std::uint32_t rowCount;
   std::uint32_t frameCount;
   std::uint32_t gridWidth;
   std::uint32_t gridHeight;
   std::uint32_t fitBisectionIterations;
   std::uint32_t rejectionIterations;
   float rangeLow;
   float lowTolerance;
   float highTolerance;
   float outputScale;
   float outputOffset;
};

std::uint32_t OrderedFloatBits( float value ) noexcept
{
   std::uint32_t bits = 0;
   std::memcpy( &bits, &value, sizeof( bits ) );
   return (bits & 0x80000000U) != 0 ? ~bits : bits ^ 0x80000000U;
}

float FloatFromOrderedBits( std::uint32_t ordered ) noexcept
{
   const std::uint32_t bits = (ordered & 0x80000000U) != 0
      ? ordered ^ 0x80000000U : ~ordered;
   float value = 0;
   std::memcpy( &value, &bits, sizeof( value ) );
   return value;
}

std::string Utf8( NSString* value )
{
   return value == nil ? std::string() : std::string( value.UTF8String );
}

id<MTLBuffer> BufferWithBytes(
   id<MTLDevice> device,
   const void* data,
   std::size_t size,
   const char* role )
{
   if ( size == 0 || size > std::numeric_limits<NSUInteger>::max() )
      throw std::invalid_argument(
         std::string( "Ultra-Fast WBPP native invalid Metal buffer size: " ) + role );
   id<MTLBuffer> buffer = [device
      newBufferWithBytes:data
      length:static_cast<NSUInteger>( size )
      options:MTLResourceStorageModeShared];
   if ( buffer == nil )
      throw std::runtime_error(
         std::string( "Ultra-Fast WBPP native unable to allocate Metal buffer: " ) + role );
   return buffer;
}

id<MTLBuffer> EmptyBuffer(
   id<MTLDevice> device,
   std::size_t size,
   const char* role )
{
   if ( size == 0 || size > std::numeric_limits<NSUInteger>::max() )
      throw std::invalid_argument(
         std::string( "Ultra-Fast WBPP native invalid Metal output size: " ) + role );
   id<MTLBuffer> buffer = [device
      newBufferWithLength:static_cast<NSUInteger>( size )
      options:MTLResourceStorageModeShared];
   if ( buffer == nil )
      throw std::runtime_error(
         std::string( "Ultra-Fast WBPP native unable to allocate Metal output: " ) + role );
   return buffer;
}

} // namespace

bool MetalFusedLnIntegrationAvailable() noexcept
{
   @autoreleasepool
   {
      return MTLCreateSystemDefaultDevice() != nil;
   }
}

struct MetalFusedLnIntegrationExecutor::Impl
{
   id<MTLDevice> device = nil;
   id<MTLComputePipelineState> pipeline = nil;
   id<MTLComputePipelineState> rangePipeline = nil;
   id<MTLComputePipelineState> normalizationPipeline = nil;
   id<MTLComputePipelineState> robustPipeline = nil;
   id<MTLComputePipelineState> linearFitPipeline = nil;
   id<MTLCommandQueue> queue = nil;

   explicit Impl( const std::filesystem::path& metalSourcePath )
   {
      if ( !metalSourcePath.empty()
        && (!metalSourcePath.is_absolute()
          || std::filesystem::canonical( metalSourcePath ) != metalSourcePath) )
         throw std::invalid_argument(
            "Ultra-Fast WBPP native Metal source path must be canonical and absolute" );
      @autoreleasepool
      {
         device = MTLCreateSystemDefaultDevice();
         if ( device == nil )
            throw std::runtime_error(
               "Ultra-Fast WBPP native Metal device is unavailable" );
         NSError* error = nil;
         NSString* source = metalSourcePath.empty()
            ? [NSString stringWithUTF8String:EmbeddedMetalSource]
            : [NSString
               stringWithContentsOfFile:
                  [NSString stringWithUTF8String:metalSourcePath.c_str()]
               encoding:NSUTF8StringEncoding
               error:&error];
         if ( source == nil )
            throw std::runtime_error(
               "Ultra-Fast WBPP native unable to read Metal source: "
               + Utf8( error.localizedDescription ) );
         MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
         // This property is available in the oldest SDK supported by the
         // project.  NO is the strict equivalent of newer MTLMathModeSafe and
         // keeps the same no-fast-math contract on Xcode 15 and newer.
         options.fastMathEnabled = NO;
#pragma clang diagnostic pop
         id<MTLLibrary> library = [device
            newLibraryWithSource:source options:options error:&error];
         if ( library == nil )
            throw std::runtime_error(
               "Ultra-Fast WBPP native runtime Metal compilation failed: "
               + Utf8( error.localizedDescription ) );
         id<MTLFunction> function = [library
            newFunctionWithName:@"fused_ln_masked_weighted_integration"];
         if ( function == nil )
            throw std::runtime_error(
               "Ultra-Fast WBPP native fused Metal function is unavailable" );
         pipeline = [device
            newComputePipelineStateWithFunction:function error:&error];
         if ( pipeline == nil )
            throw std::runtime_error(
               "Ultra-Fast WBPP native unable to create Metal pipeline: "
               + Utf8( error.localizedDescription ) );
         id<MTLFunction> rangeFunction = [library
            newFunctionWithName:@"reduce_finite_output_range"];
         id<MTLFunction> normalizationFunction = [library
            newFunctionWithName:@"normalize_output_range_in_place"];
         if ( rangeFunction == nil || normalizationFunction == nil )
            throw std::runtime_error(
               "Ultra-Fast WBPP native output-range Metal functions are unavailable" );
         rangePipeline = [device
            newComputePipelineStateWithFunction:rangeFunction error:&error];
         if ( rangePipeline == nil )
            throw std::runtime_error(
               "Ultra-Fast WBPP native unable to create output-range reduction pipeline: "
               + Utf8( error.localizedDescription ) );
         normalizationPipeline = [device
            newComputePipelineStateWithFunction:normalizationFunction
            error:&error];
         if ( normalizationPipeline == nil )
            throw std::runtime_error(
               "Ultra-Fast WBPP native unable to create output normalization pipeline: "
               + Utf8( error.localizedDescription ) );
         id<MTLFunction> robustFunction = [library
            newFunctionWithName:@"fused_ln_native_robust_integration"];
         if ( robustFunction == nil )
            throw std::runtime_error(
               "Ultra-Fast WBPP native robust rejection Metal function is unavailable" );
         robustPipeline = [device
            newComputePipelineStateWithFunction:robustFunction error:&error];
         if ( robustPipeline == nil )
            throw std::runtime_error(
               "Ultra-Fast WBPP native unable to create robust rejection pipeline: "
               + Utf8( error.localizedDescription ) );
         id<MTLFunction> linearFitFunction = [library
            newFunctionWithName:@"fused_ln_native_linear_fit_integration"];
         if ( linearFitFunction == nil )
            throw std::runtime_error(
               "Ultra-Fast WBPP native linear-fit Metal function is unavailable" );
         linearFitPipeline = [device
            newComputePipelineStateWithFunction:linearFitFunction
            error:&error];
         if ( linearFitPipeline == nil )
            throw std::runtime_error(
               "Ultra-Fast WBPP native unable to create linear-fit pipeline: "
               + Utf8( error.localizedDescription ) );
         queue = [device newCommandQueue];
         if ( queue == nil )
            throw std::runtime_error(
               "Ultra-Fast WBPP native unable to create Metal command queue" );
      }
   }
};

MetalFusedLnIntegrationExecutor::MetalFusedLnIntegrationExecutor(
   const std::filesystem::path& metalSourcePath )
   : m_impl( std::make_unique<Impl>( metalSourcePath ) )
{
}

MetalFusedLnIntegrationExecutor::~MetalFusedLnIntegrationExecutor() = default;

MetalFusedLnIntegrationExecutor::MetalFusedLnIntegrationExecutor(
   MetalFusedLnIntegrationExecutor&& ) noexcept = default;

MetalFusedLnIntegrationExecutor&
MetalFusedLnIntegrationExecutor::operator =(
   MetalFusedLnIntegrationExecutor&& ) noexcept = default;

FusedLnIntegrationResult MetalFusedLnIntegrationExecutor::Run(
   const FusedLnIntegrationRequest& request,
   MetalExecutionStats* stats )
{
   request.Validate();
   if ( !m_impl )
      throw std::logic_error(
         "Ultra-Fast WBPP native Metal executor was moved from" );

   @autoreleasepool
   {
      id<MTLDevice> device = m_impl->device;
      id<MTLComputePipelineState> pipeline = m_impl->pipeline;
      id<MTLCommandQueue> queue = m_impl->queue;

      const std::size_t pixels = request.TilePixels();
      const std::size_t sampleBytes =
         request.frameMajorSamples.size_bytes();
      const std::size_t maskBytes =
         request.frameMajorRejectionMask.size_bytes();
      const std::size_t scaleBytes =
         request.frameMajorScaleGrid.size_bytes();
      const std::size_t offsetBytes =
         request.frameMajorZeroOffsetGrid.size_bytes();
      const std::size_t weightBytes = request.frameWeights.size_bytes();
      const std::size_t integratedBytes = pixels*sizeof( float );
      const std::size_t countBytes = pixels*sizeof( std::uint16_t );
      const std::uint64_t submittedBytes =
         sampleBytes + maskBytes + scaleBytes + offsetBytes + weightBytes
         + integratedBytes + 2*countBytes + sizeof( FusedParameters );
      if ( submittedBytes > device.recommendedMaxWorkingSetSize )
         throw std::runtime_error(
            "Ultra-Fast WBPP native tile exceeds the Metal recommended working set" );
      for ( std::size_t size : { sampleBytes, maskBytes, scaleBytes,
                                offsetBytes, weightBytes, integratedBytes,
                                countBytes } )
         if ( size > device.maxBufferLength )
            throw std::runtime_error(
               "Ultra-Fast WBPP native tile exceeds the Metal maximum buffer length" );

      id<MTLBuffer> samples = BufferWithBytes(
         device, request.frameMajorSamples.data(), sampleBytes, "samples" );
      id<MTLBuffer> mask = BufferWithBytes(
         device, request.frameMajorRejectionMask.data(), maskBytes, "mask" );
      id<MTLBuffer> scale = BufferWithBytes(
         device, request.frameMajorScaleGrid.data(), scaleBytes, "scale grids" );
      id<MTLBuffer> offset = BufferWithBytes(
         device, request.frameMajorZeroOffsetGrid.data(), offsetBytes,
         "zero-offset grids" );
      id<MTLBuffer> weights = BufferWithBytes(
         device, request.frameWeights.data(), weightBytes, "weights" );
      id<MTLBuffer> integrated = EmptyBuffer(
         device, integratedBytes, "integrated output" );
      id<MTLBuffer> accepted = EmptyBuffer(
         device, countBytes, "accepted counts" );
      id<MTLBuffer> rejected = EmptyBuffer(
         device, countBytes, "rejected counts" );
      const FusedParameters parameters{
         request.tile.image.width,
         request.tile.image.height,
         request.tile.firstRow,
         request.tile.rowCount,
         request.frameCount,
         request.gridWidth,
         request.gridHeight,
         request.rejectionBits,
         request.outputScale,
         request.outputOffset
      };
      id<MTLBuffer> parameterBuffer = BufferWithBytes(
         device, &parameters, sizeof( parameters ), "parameters" );

      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      if ( command == nil || encoder == nil )
         throw std::runtime_error(
            "Ultra-Fast WBPP native unable to create Metal command objects" );
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:samples offset:0 atIndex:0];
      [encoder setBuffer:mask offset:0 atIndex:1];
      [encoder setBuffer:scale offset:0 atIndex:2];
      [encoder setBuffer:offset offset:0 atIndex:3];
      [encoder setBuffer:weights offset:0 atIndex:4];
      [encoder setBuffer:integrated offset:0 atIndex:5];
      [encoder setBuffer:accepted offset:0 atIndex:6];
      [encoder setBuffer:rejected offset:0 atIndex:7];
      [encoder setBuffer:parameterBuffer offset:0 atIndex:8];
      const NSUInteger threads = std::min<NSUInteger>(
         256, pipeline.maxTotalThreadsPerThreadgroup );
      [encoder dispatchThreads:MTLSizeMake( pixels, 1, 1 )
          threadsPerThreadgroup:MTLSizeMake( threads, 1, 1 )];
      [encoder endEncoding];

      const auto started = std::chrono::steady_clock::now();
      [command commit];
      [command waitUntilCompleted];
      const auto finished = std::chrono::steady_clock::now();
      if ( command.status == MTLCommandBufferStatusError )
         throw std::runtime_error(
            "Ultra-Fast WBPP native Metal execution failed: "
            + Utf8( command.error.localizedDescription ) );

      FusedLnIntegrationResult result;
      result.integrated.resize( pixels );
      result.acceptedSamples.resize( pixels );
      result.rejectedSamples.resize( pixels );
      std::memcpy( result.integrated.data(), integrated.contents,
                   integratedBytes );
      std::memcpy( result.acceptedSamples.data(), accepted.contents,
                   countBytes );
      std::memcpy( result.rejectedSamples.data(), rejected.contents,
                   countBytes );

      if ( stats != nullptr )
      {
         stats->deviceName = Utf8( device.name );
         stats->wallSeconds =
            std::chrono::duration<double>( finished - started ).count();
         stats->gpuSeconds = command.GPUEndTime > command.GPUStartTime
            ? command.GPUEndTime - command.GPUStartTime : 0;
         stats->submittedBufferBytes = submittedBytes;
         stats->recommendedWorkingSetBytes =
            device.recommendedMaxWorkingSetSize;
         stats->maximumBufferBytes = device.maxBufferLength;
      }
      return result;
   }
}

FusedLnIntegrationResult
MetalFusedLnIntegrationExecutor::RunNativeRobustRejection(
   const NativeRobustIntegrationRequest& request,
   MetalExecutionStats* stats )
{
   request.Validate();
   if ( request.frameCount > MaximumMetalRejectionFrames )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native Metal robust rejection supports at most 64 "
         "frames per exact request; use the CPU oracle for larger stacks" );
   if ( !m_impl )
      throw std::logic_error(
         "Ultra-Fast WBPP native Metal executor was moved from" );

   @autoreleasepool
   {
      id<MTLDevice> device = m_impl->device;
      const std::size_t pixels = request.TilePixels();
      const std::size_t sampleBytes = request.frameMajorSamples.size_bytes();
      const std::size_t scaleBytes = request.frameMajorScaleGrid.size_bytes();
      const std::size_t offsetBytes =
         request.frameMajorZeroOffsetGrid.size_bytes();
      const std::size_t weightBytes = request.frameWeights.size_bytes();
      const std::size_t integratedBytes = pixels*sizeof( float );
      const std::size_t countBytes = pixels*sizeof( std::uint16_t );
      const std::uint64_t submittedBytes = sampleBytes + scaleBytes
         + offsetBytes + weightBytes + integratedBytes + 2*countBytes
         + sizeof( RobustParameters );
      if ( submittedBytes > device.recommendedMaxWorkingSetSize )
         throw std::runtime_error(
            "Ultra-Fast WBPP native robust tile exceeds the Metal working set" );
      for ( std::size_t size : { sampleBytes, scaleBytes, offsetBytes,
                                weightBytes, integratedBytes, countBytes } )
         if ( size > device.maxBufferLength )
            throw std::runtime_error(
               "Ultra-Fast WBPP native robust tile exceeds the Metal buffer limit" );

      id<MTLBuffer> samples = BufferWithBytes(
         device, request.frameMajorSamples.data(), sampleBytes, "samples" );
      id<MTLBuffer> scale = BufferWithBytes(
         device, request.frameMajorScaleGrid.data(), scaleBytes,
         "scale grids" );
      id<MTLBuffer> offset = BufferWithBytes(
         device, request.frameMajorZeroOffsetGrid.data(), offsetBytes,
         "zero-offset grids" );
      id<MTLBuffer> weights = BufferWithBytes(
         device, request.frameWeights.data(), weightBytes, "weights" );
      id<MTLBuffer> integrated = EmptyBuffer(
         device, integratedBytes, "robust integrated output" );
      id<MTLBuffer> accepted = EmptyBuffer(
         device, countBytes, "robust accepted counts" );
      id<MTLBuffer> rejected = EmptyBuffer(
         device, countBytes, "robust rejected counts" );
      const RobustParameters parameters{
         request.tile.image.width,
         request.tile.image.height,
         request.tile.firstRow,
         request.tile.rowCount,
         request.frameCount,
         request.gridWidth,
         request.gridHeight,
         request.winsorIterations,
         request.rangeLow,
         request.lowSigma,
         request.highSigma,
         request.winsorSigma,
         request.outputScale,
         request.outputOffset
      };
      id<MTLBuffer> parameterBuffer = BufferWithBytes(
         device, &parameters, sizeof( parameters ), "robust parameters" );
      id<MTLCommandBuffer> command = [m_impl->queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      if ( command == nil || encoder == nil )
         throw std::runtime_error(
            "Ultra-Fast WBPP native unable to create robust Metal command" );
      [encoder setComputePipelineState:m_impl->robustPipeline];
      [encoder setBuffer:samples offset:0 atIndex:0];
      [encoder setBuffer:scale offset:0 atIndex:1];
      [encoder setBuffer:offset offset:0 atIndex:2];
      [encoder setBuffer:weights offset:0 atIndex:3];
      [encoder setBuffer:integrated offset:0 atIndex:4];
      [encoder setBuffer:accepted offset:0 atIndex:5];
      [encoder setBuffer:rejected offset:0 atIndex:6];
      [encoder setBuffer:parameterBuffer offset:0 atIndex:7];
      const NSUInteger threads = std::min<NSUInteger>(
         256, m_impl->robustPipeline.maxTotalThreadsPerThreadgroup );
      [encoder dispatchThreads:MTLSizeMake( pixels, 1, 1 )
          threadsPerThreadgroup:MTLSizeMake( threads, 1, 1 )];
      [encoder endEncoding];
      const auto started = std::chrono::steady_clock::now();
      [command commit];
      [command waitUntilCompleted];
      const auto finished = std::chrono::steady_clock::now();
      if ( command.status == MTLCommandBufferStatusError )
         throw std::runtime_error(
            "Ultra-Fast WBPP native robust Metal execution failed: "
            + Utf8( command.error.localizedDescription ) );

      FusedLnIntegrationResult result;
      result.integrated.resize( pixels );
      result.acceptedSamples.resize( pixels );
      result.rejectedSamples.resize( pixels );
      std::memcpy( result.integrated.data(), integrated.contents,
                   integratedBytes );
      std::memcpy( result.acceptedSamples.data(), accepted.contents,
                   countBytes );
      std::memcpy( result.rejectedSamples.data(), rejected.contents,
                   countBytes );
      if ( stats != nullptr )
      {
         stats->deviceName = Utf8( device.name );
         stats->wallSeconds =
            std::chrono::duration<double>( finished - started ).count();
         stats->gpuSeconds = command.GPUEndTime > command.GPUStartTime
            ? command.GPUEndTime - command.GPUStartTime : 0;
         stats->submittedBufferBytes = submittedBytes;
         stats->recommendedWorkingSetBytes =
            device.recommendedMaxWorkingSetSize;
         stats->maximumBufferBytes = device.maxBufferLength;
      }
      return result;
   }
}

FusedLnIntegrationResult
MetalFusedLnIntegrationExecutor::RunNativeLinearFitRejection(
   const NativeLinearFitIntegrationRequest& request,
   MetalExecutionStats* stats )
{
   request.Validate();
   if ( request.frameCount > MaximumMetalRejectionFrames )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native Metal linear-fit rejection supports at most 64 "
         "frames per exact request; use the CPU oracle for larger stacks" );
   if ( !m_impl )
      throw std::logic_error(
         "Ultra-Fast WBPP native Metal executor was moved from" );
   @autoreleasepool
   {
      id<MTLDevice> device = m_impl->device;
      const std::size_t pixels = request.TilePixels();
      const std::size_t sampleBytes = request.frameMajorSamples.size_bytes();
      const std::size_t scaleBytes = request.frameMajorScaleGrid.size_bytes();
      const std::size_t offsetBytes =
         request.frameMajorZeroOffsetGrid.size_bytes();
      const std::size_t weightBytes = request.frameWeights.size_bytes();
      const std::size_t integratedBytes = pixels*sizeof( float );
      const std::size_t countBytes = pixels*sizeof( std::uint16_t );
      const std::uint64_t submittedBytes = sampleBytes + scaleBytes
         + offsetBytes + weightBytes + integratedBytes + 2*countBytes
         + sizeof( LinearFitParameters );
      if ( submittedBytes > device.recommendedMaxWorkingSetSize )
         throw std::runtime_error(
            "Ultra-Fast WBPP native linear-fit tile exceeds the Metal working set" );
      for ( std::size_t size : { sampleBytes, scaleBytes, offsetBytes,
                                weightBytes, integratedBytes, countBytes } )
         if ( size > device.maxBufferLength )
            throw std::runtime_error(
               "Ultra-Fast WBPP native linear-fit tile exceeds the Metal buffer limit" );
      id<MTLBuffer> samples = BufferWithBytes(
         device, request.frameMajorSamples.data(), sampleBytes, "samples" );
      id<MTLBuffer> scale = BufferWithBytes(
         device, request.frameMajorScaleGrid.data(), scaleBytes,
         "scale grids" );
      id<MTLBuffer> offset = BufferWithBytes(
         device, request.frameMajorZeroOffsetGrid.data(), offsetBytes,
         "zero-offset grids" );
      id<MTLBuffer> weights = BufferWithBytes(
         device, request.frameWeights.data(), weightBytes, "weights" );
      id<MTLBuffer> integrated = EmptyBuffer(
         device, integratedBytes, "linear-fit integrated output" );
      id<MTLBuffer> accepted = EmptyBuffer(
         device, countBytes, "linear-fit accepted counts" );
      id<MTLBuffer> rejected = EmptyBuffer(
         device, countBytes, "linear-fit rejected counts" );
      const LinearFitParameters parameters{
         request.tile.image.width,
         request.tile.image.height,
         request.tile.firstRow,
         request.tile.rowCount,
         request.frameCount,
         request.gridWidth,
         request.gridHeight,
         request.fitBisectionIterations,
         request.rejectionIterations,
         request.rangeLow,
         request.lowTolerance,
         request.highTolerance,
         request.outputScale,
         request.outputOffset
      };
      id<MTLBuffer> parameterBuffer = BufferWithBytes(
         device, &parameters, sizeof( parameters ),
         "linear-fit parameters" );
      id<MTLCommandBuffer> command = [m_impl->queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      if ( command == nil || encoder == nil )
         throw std::runtime_error(
            "Ultra-Fast WBPP native unable to create linear-fit Metal command" );
      [encoder setComputePipelineState:m_impl->linearFitPipeline];
      [encoder setBuffer:samples offset:0 atIndex:0];
      [encoder setBuffer:scale offset:0 atIndex:1];
      [encoder setBuffer:offset offset:0 atIndex:2];
      [encoder setBuffer:weights offset:0 atIndex:3];
      [encoder setBuffer:integrated offset:0 atIndex:4];
      [encoder setBuffer:accepted offset:0 atIndex:5];
      [encoder setBuffer:rejected offset:0 atIndex:6];
      [encoder setBuffer:parameterBuffer offset:0 atIndex:7];
      const NSUInteger threads = std::min<NSUInteger>(
         256, m_impl->linearFitPipeline.maxTotalThreadsPerThreadgroup );
      [encoder dispatchThreads:MTLSizeMake( pixels, 1, 1 )
          threadsPerThreadgroup:MTLSizeMake( threads, 1, 1 )];
      [encoder endEncoding];
      const auto started = std::chrono::steady_clock::now();
      [command commit];
      [command waitUntilCompleted];
      const auto finished = std::chrono::steady_clock::now();
      if ( command.status == MTLCommandBufferStatusError )
         throw std::runtime_error(
            "Ultra-Fast WBPP native linear-fit Metal execution failed: "
            + Utf8( command.error.localizedDescription ) );
      FusedLnIntegrationResult result;
      result.integrated.resize( pixels );
      result.acceptedSamples.resize( pixels );
      result.rejectedSamples.resize( pixels );
      std::memcpy( result.integrated.data(), integrated.contents,
                   integratedBytes );
      std::memcpy( result.acceptedSamples.data(), accepted.contents,
                   countBytes );
      std::memcpy( result.rejectedSamples.data(), rejected.contents,
                   countBytes );
      if ( stats != nullptr )
      {
         stats->deviceName = Utf8( device.name );
         stats->wallSeconds =
            std::chrono::duration<double>( finished - started ).count();
         stats->gpuSeconds = command.GPUEndTime > command.GPUStartTime
            ? command.GPUEndTime - command.GPUStartTime : 0;
         stats->submittedBufferBytes = submittedBytes;
         stats->recommendedWorkingSetBytes =
            device.recommendedMaxWorkingSetSize;
         stats->maximumBufferBytes = device.maxBufferLength;
      }
      return result;
   }
}

OutputRangeNormalization
MetalFusedLnIntegrationExecutor::NormalizeOutputRangeInPlace(
   std::span<float> samples,
   MetalExecutionStats* stats )
{
   if ( !m_impl )
      throw std::logic_error(
         "Ultra-Fast WBPP native Metal executor was moved from" );
   if ( samples.empty()
     || samples.size() > std::numeric_limits<std::uint32_t>::max() )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native output range sample count is invalid" );

   @autoreleasepool
   {
      id<MTLDevice> device = m_impl->device;
      const std::size_t sampleBytes = samples.size_bytes();
      if ( sampleBytes > device.maxBufferLength
        || sampleBytes + 3*sizeof( std::uint32_t )
              > device.recommendedMaxWorkingSetSize )
         throw std::runtime_error(
            "Ultra-Fast WBPP native output range exceeds Metal memory limits" );
      id<MTLBuffer> pixelBuffer = BufferWithBytes(
         device, samples.data(), sampleBytes, "output-range samples" );
      const std::uint32_t initialRange[3]{
         OrderedFloatBits( std::numeric_limits<float>::infinity() ),
         OrderedFloatBits( -std::numeric_limits<float>::infinity() ),
         0
      };
      id<MTLBuffer> rangeBuffer = BufferWithBytes(
         device, initialRange, sizeof( initialRange ), "output range" );
      const RangeParameters rangeParameters{
         static_cast<std::uint32_t>( samples.size() )
      };
      id<MTLBuffer> rangeParameterBuffer = BufferWithBytes(
         device, &rangeParameters, sizeof( rangeParameters ),
         "output-range parameters" );

      id<MTLCommandBuffer> reductionCommand = [m_impl->queue commandBuffer];
      id<MTLComputeCommandEncoder> reductionEncoder =
         [reductionCommand computeCommandEncoder];
      if ( reductionCommand == nil || reductionEncoder == nil )
         throw std::runtime_error(
            "Ultra-Fast WBPP native unable to create range reduction command" );
      [reductionEncoder setComputePipelineState:m_impl->rangePipeline];
      [reductionEncoder setBuffer:pixelBuffer offset:0 atIndex:0];
      [reductionEncoder setBuffer:rangeBuffer offset:0 atIndex:1];
      [reductionEncoder setBuffer:rangeParameterBuffer offset:0 atIndex:2];
      constexpr NSUInteger threads = 256;
      const NSUInteger groups =
         (static_cast<NSUInteger>( samples.size() ) + threads - 1)/threads;
      [reductionEncoder dispatchThreadgroups:MTLSizeMake( groups, 1, 1 )
          threadsPerThreadgroup:MTLSizeMake( threads, 1, 1 )];
      [reductionEncoder endEncoding];

      const auto started = std::chrono::steady_clock::now();
      [reductionCommand commit];
      [reductionCommand waitUntilCompleted];
      if ( reductionCommand.status == MTLCommandBufferStatusError )
         throw std::runtime_error(
            "Ultra-Fast WBPP native output-range reduction failed: "
            + Utf8( reductionCommand.error.localizedDescription ) );

      const auto* reduced = static_cast<const std::uint32_t*>(
         rangeBuffer.contents );
      OutputRangeNormalization result;
      result.finiteSamples = reduced[2];
      if ( result.finiteSamples == 0 )
         throw std::runtime_error(
            "Ultra-Fast WBPP native output range has no finite samples" );
      result.finiteMinimum = FloatFromOrderedBits( reduced[0] );
      result.finiteMaximum = FloatFromOrderedBits( reduced[1] );
      result.effectiveMinimum = std::min( 0.0F, result.finiteMinimum );
      result.effectiveMaximum = std::max( 1.0F, result.finiteMaximum );
      const float span = result.effectiveMaximum - result.effectiveMinimum;
      if ( !std::isfinite( span ) || span <= 0 )
         throw std::runtime_error(
            "Ultra-Fast WBPP native output range is invalid" );
      result.applied = result.effectiveMinimum < 0
                    || result.effectiveMaximum > 1;
      result.scale = result.applied ? 1.0F/span : 1.0F;
      result.offset = result.applied
         ? -result.effectiveMinimum*result.scale : 0.0F;

      double gpuSeconds = reductionCommand.GPUEndTime
                        > reductionCommand.GPUStartTime
         ? reductionCommand.GPUEndTime - reductionCommand.GPUStartTime : 0;
      std::uint64_t submittedBytes = sampleBytes + sizeof( initialRange )
                                   + sizeof( rangeParameters );
      if ( result.applied )
      {
         const OutputNormalizationParameters normalizationParameters{
            static_cast<std::uint32_t>( samples.size() ),
            result.scale, result.offset
         };
         id<MTLBuffer> normalizationParameterBuffer = BufferWithBytes(
            device, &normalizationParameters,
            sizeof( normalizationParameters ),
            "output normalization parameters" );
         id<MTLCommandBuffer> normalizationCommand =
            [m_impl->queue commandBuffer];
         id<MTLComputeCommandEncoder> normalizationEncoder =
            [normalizationCommand computeCommandEncoder];
         if ( normalizationCommand == nil || normalizationEncoder == nil )
            throw std::runtime_error(
               "Ultra-Fast WBPP native unable to create normalization command" );
         [normalizationEncoder
            setComputePipelineState:m_impl->normalizationPipeline];
         [normalizationEncoder setBuffer:pixelBuffer offset:0 atIndex:0];
         [normalizationEncoder
            setBuffer:normalizationParameterBuffer offset:0 atIndex:1];
         [normalizationEncoder
            dispatchThreads:MTLSizeMake( samples.size(), 1, 1 )
            threadsPerThreadgroup:MTLSizeMake( threads, 1, 1 )];
         [normalizationEncoder endEncoding];
         [normalizationCommand commit];
         [normalizationCommand waitUntilCompleted];
         if ( normalizationCommand.status == MTLCommandBufferStatusError )
            throw std::runtime_error(
               "Ultra-Fast WBPP native output normalization failed: "
               + Utf8( normalizationCommand.error.localizedDescription ) );
         if ( normalizationCommand.GPUEndTime
                > normalizationCommand.GPUStartTime )
            gpuSeconds += normalizationCommand.GPUEndTime
                        - normalizationCommand.GPUStartTime;
         submittedBytes += sizeof( normalizationParameters );
      }
      std::memcpy( samples.data(), pixelBuffer.contents, sampleBytes );
      const auto finished = std::chrono::steady_clock::now();
      if ( stats != nullptr )
      {
         stats->deviceName = Utf8( device.name );
         stats->wallSeconds =
            std::chrono::duration<double>( finished - started ).count();
         stats->gpuSeconds = gpuSeconds;
         stats->submittedBufferBytes = submittedBytes;
         stats->recommendedWorkingSetBytes =
            device.recommendedMaxWorkingSetSize;
         stats->maximumBufferBytes = device.maxBufferLength;
      }
      return result;
   }
}

} // namespace openastroflow::native
