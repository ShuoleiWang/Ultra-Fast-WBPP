#include "openastroflow/FusedLnIntegration.h"
#include "openastroflow/ImageTile.h"
#include "openastroflow/MetalFusedLnIntegration.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{

using namespace openastroflow::native;

struct Options
{
   std::filesystem::path metalSource;
   std::filesystem::path output;
   std::uint32_t width = 6252;
   std::uint32_t height = 4176;
   std::uint32_t frames = 40;
   std::uint32_t tileRows = 144;
   std::uint32_t repetitions = 5;
   std::uint32_t warmupRuns = 1;
};

std::uint32_t Unsigned( std::string_view value, std::string_view role )
{
   std::size_t parsed = 0;
   const unsigned long result = std::stoul( std::string( value ), &parsed );
   if ( parsed != value.size() || result == 0
     || result > std::numeric_limits<std::uint32_t>::max() )
      throw std::invalid_argument( std::string( role ) + " is invalid" );
   return static_cast<std::uint32_t>( result );
}

Options Parse( int argc, char** argv )
{
   Options options;
   for ( int i = 1; i < argc; ++i )
   {
      const std::string_view argument( argv[i] );
      if ( i + 1 >= argc )
         throw std::invalid_argument( "Ultra-Fast WBPP native benchmark option has no value" );
      const std::string_view value( argv[++i] );
      if ( argument == "--metal-source" ) options.metalSource = value;
      else if ( argument == "--output" ) options.output = value;
      else if ( argument == "--width" ) options.width = Unsigned( value, "width" );
      else if ( argument == "--height" ) options.height = Unsigned( value, "height" );
      else if ( argument == "--frames" ) options.frames = Unsigned( value, "frames" );
      else if ( argument == "--tile-rows" ) options.tileRows = Unsigned( value, "tile rows" );
      else if ( argument == "--repetitions" )
         options.repetitions = Unsigned( value, "repetitions" );
      else if ( argument == "--warmup-runs" )
         options.warmupRuns = Unsigned( value, "warmup runs" );
      else throw std::invalid_argument( "unknown Ultra-Fast WBPP native benchmark option" );
   }
   if ( options.metalSource.empty() || options.output.empty()
     || !options.metalSource.is_absolute() || !options.output.is_absolute() )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native benchmark requires absolute --metal-source and --output" );
   return options;
}

double Median( std::vector<double> values )
{
   if ( values.empty() )
      throw std::invalid_argument( "benchmark sample set is empty" );
   std::sort( values.begin(), values.end() );
   const std::size_t middle = values.size()/2;
   return (values.size() & 1U) != 0 ? values[middle]
      : 0.5*(values[middle - 1] + values[middle]);
}

std::uint64_t Fnv1a64( const std::vector<float>& values ) noexcept
{
   constexpr std::uint64_t Offset = 14695981039346656037ULL;
   constexpr std::uint64_t Prime = 1099511628211ULL;
   const auto* bytes = reinterpret_cast<const unsigned char*>( values.data() );
   const std::size_t size = values.size()*sizeof( float );
   std::uint64_t result = Offset;
   for ( std::size_t index = 0; index < size; ++index )
   {
      result ^= bytes[index];
      result *= Prime;
   }
   return result;
}

void WriteSamples( std::ostream& output, const std::vector<double>& values )
{
   output << '[';
   for ( std::size_t index = 0; index < values.size(); ++index )
   {
      if ( index != 0 ) output << ", ";
      output << values[index];
   }
   output << ']';
}

} // namespace

int main( int argc, char** argv )
{
   try
   {
      const Options options = Parse( argc, argv );
      const ImageGeometry geometry{ options.width, options.height, 1 };
      const std::vector<TileRegion> tiles =
         PartitionRows( geometry, options.tileRows );
      const std::uint32_t gridWidth = 49;
      const std::uint32_t gridHeight = 33;
      const std::size_t gridSamples =
         static_cast<std::size_t>( gridWidth )*gridHeight;
      std::vector<float> scale( options.frames*gridSamples );
      std::vector<float> offset( options.frames*gridSamples );
      std::vector<float> weights( options.frames );
      for ( std::uint32_t frame = 0; frame < options.frames; ++frame )
      {
         weights[frame] = 0.25F + 0.75F*(frame + 1)/options.frames;
         for ( std::uint32_t y = 0; y < gridHeight; ++y )
            for ( std::uint32_t x = 0; x < gridWidth; ++x )
            {
               const std::size_t index =
                  static_cast<std::size_t>( frame )*gridSamples
                  + static_cast<std::size_t>( y )*gridWidth + x;
               scale[index] = 0.86F + 0.002F*(frame % 11)
                            + 0.00003F*x - 0.00002F*y;
               offset[index] = -0.0014F + 0.00001F*(frame % 7)
                             + 0.0000004F*(x + y);
            }
      }

#ifndef NDEBUG
      throw std::runtime_error(
         "Ultra-Fast WBPP performance benchmarks require a Release binary" );
#endif
      if ( std::filesystem::exists( options.output ) )
         throw std::runtime_error(
            "Ultra-Fast WBPP native benchmark refuses to replace output" );

      std::vector<double> wallSamples;
      std::vector<double> preparationSamples;
      std::vector<double> metalCallSamples;
      std::vector<double> gpuSamples;
      std::vector<double> normalizationWallSamples;
      std::vector<double> normalizationGpuSamples;
      wallSamples.reserve( options.repetitions );
      preparationSamples.reserve( options.repetitions );
      metalCallSamples.reserve( options.repetitions );
      gpuSamples.reserve( options.repetitions );
      normalizationWallSamples.reserve( options.repetitions );
      normalizationGpuSamples.reserve( options.repetitions );

      const std::uint64_t processedPixelFrames =
         static_cast<std::uint64_t>( geometry.SampleCount() )*options.frames;
      std::string deviceName;
      std::uint64_t recommendedWorkingSet = 0;
      std::uint64_t submittedBytes = 0;
      std::uint64_t outputHash = 0;
      bool outputHashInitialized = false;
      OutputRangeNormalization outputRange;
      MetalFusedLnIntegrationExecutor executor( options.metalSource );
      const std::uint64_t totalRuns =
         static_cast<std::uint64_t>( options.warmupRuns ) + options.repetitions;
      for ( std::uint64_t run = 0; run < totalRuns; ++run )
      {
         double gpuSeconds = 0;
         double syntheticPreparationSeconds = 0;
         double metalCallSeconds = 0;
         std::uint64_t runSubmittedBytes = 0;
         std::uint64_t accountedPixelFrames = 0;
         std::vector<float> finalImage( geometry.SampleCount() );
         const auto started = std::chrono::steady_clock::now();
         for ( const TileRegion& tile : tiles )
         {
            const auto preparationStarted = std::chrono::steady_clock::now();
            const std::size_t pixels = tile.PixelCount();
            std::vector<float> samples( options.frames*pixels );
            std::vector<std::uint8_t> mask( options.frames*pixels, 0 );
            for ( std::uint32_t frame = 0; frame < options.frames; ++frame )
               for ( std::size_t pixel = 0; pixel < pixels; ++pixel )
               {
                  samples[static_cast<std::size_t>( frame )*pixels + pixel] =
                     0.001F + 0.0002F*(frame % 13)
                     + 0.01F*float( pixel % options.width )/options.width;
                  if ( (pixel + 97*frame) % 8191 == 0 )
                     mask[static_cast<std::size_t>( frame )*pixels + pixel] =
                        (frame & 1) ? 0x01 : 0x02;
               }
            syntheticPreparationSeconds += std::chrono::duration<double>(
               std::chrono::steady_clock::now() - preparationStarted ).count();
            const FusedLnIntegrationRequest request{
               tile, options.frames, gridWidth, gridHeight,
               samples, mask, scale, offset, weights,
               DefaultOracleRejectionBits
            };
            MetalExecutionStats stats;
            const auto metalStarted = std::chrono::steady_clock::now();
            const FusedLnIntegrationResult result = executor.Run( request, &stats );
            metalCallSeconds += std::chrono::duration<double>(
               std::chrono::steady_clock::now() - metalStarted ).count();
            if ( result.integrated.size() != pixels
              || result.acceptedSamples.size() != pixels
              || result.rejectedSamples.size() != pixels )
               throw std::runtime_error(
                  "Ultra-Fast WBPP native benchmark output cardinality differs" );
            for ( std::size_t pixel = 0; pixel < pixels; ++pixel )
               accountedPixelFrames += result.acceptedSamples[pixel]
                                      + result.rejectedSamples[pixel];
            std::copy( result.integrated.begin(), result.integrated.end(),
                       finalImage.begin()
                          + static_cast<std::size_t>( tile.firstRow )
                              *options.width );
            if ( deviceName.empty() ) deviceName = stats.deviceName;
            else if ( deviceName != stats.deviceName )
               throw std::runtime_error(
                  "Ultra-Fast WBPP benchmark Metal device changed during a run" );
            gpuSeconds += stats.gpuSeconds;
            runSubmittedBytes += stats.submittedBufferBytes;
            recommendedWorkingSet = stats.recommendedWorkingSetBytes;
         }
         if ( accountedPixelFrames != processedPixelFrames )
            throw std::runtime_error(
               "Ultra-Fast WBPP benchmark sample accounting is incomplete" );
         MetalExecutionStats normalizationStats;
         outputRange = executor.NormalizeOutputRangeInPlace(
            finalImage, &normalizationStats );
         gpuSeconds += normalizationStats.gpuSeconds;
         metalCallSeconds += normalizationStats.wallSeconds;
         runSubmittedBytes += normalizationStats.submittedBufferBytes;
         const double wallSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started ).count();
         const std::uint64_t currentHash = Fnv1a64( finalImage );
         if ( outputHashInitialized && currentHash != outputHash )
            throw std::runtime_error(
               "Ultra-Fast WBPP benchmark output changed across repetitions" );
         outputHash = currentHash;
         outputHashInitialized = true;
         if ( run >= options.warmupRuns )
         {
            wallSamples.push_back( wallSeconds );
            preparationSamples.push_back( syntheticPreparationSeconds );
            metalCallSamples.push_back( metalCallSeconds );
            gpuSamples.push_back( gpuSeconds );
            normalizationWallSamples.push_back( normalizationStats.wallSeconds );
            normalizationGpuSamples.push_back( normalizationStats.gpuSeconds );
            if ( submittedBytes == 0 ) submittedBytes = runSubmittedBytes;
            else if ( submittedBytes != runSubmittedBytes )
               throw std::runtime_error(
                  "Ultra-Fast WBPP submitted-byte count changed across repetitions" );
         }
      }
      const double wallSeconds = Median( wallSamples );
      const double syntheticPreparationSeconds = Median( preparationSamples );
      const double metalCallSeconds = Median( metalCallSamples );
      const double gpuSeconds = Median( gpuSamples );
      const double normalizationWallSeconds = Median( normalizationWallSamples );
      const double normalizationGpuSeconds = Median( normalizationGpuSamples );
      const double megapixelFramesPerSecond =
         processedPixelFrames/1.0e6/wallSeconds;
      const double linearProjected96FrameSeconds =
         wallSeconds*96.0/options.frames;
      const bool kernelFeasibilityGate =
         linearProjected96FrameSeconds <= 105.0;

      std::filesystem::create_directories( options.output.parent_path() );
      std::ofstream output( options.output, std::ios::binary | std::ios::trunc );
      if ( !output )
         throw std::runtime_error(
            "Ultra-Fast WBPP native benchmark output cannot be created" );
      output << std::setprecision( 17 )
         << "{\n"
         << "  \"schemaVersion\": 2,\n"
         << "  \"kind\": \"ultra-fast-wbpp-metal-integration-benchmark-v2\",\n"
         << "  \"build\": {\"configuration\": \"" OAF_BENCH_BUILD_TYPE
            "\", \"compilerId\": \"" OAF_BENCH_COMPILER_ID
            "\", \"compilerVersion\": \"" OAF_BENCH_COMPILER_VERSION
            "\", \"sourceCommit\": \"" OAF_BENCH_SOURCE_COMMIT
            "\", \"sourceDirty\": "
         << (OAF_BENCH_SOURCE_DIRTY ? "true" : "false")
         << ", \"trackedDiffSha256\": \""
            OAF_BENCH_TRACKED_DIFF_SHA256
            "\", \"untrackedTreeSha256\": \""
            OAF_BENCH_UNTRACKED_TREE_SHA256
            "\", \"untrackedFileCount\": " << OAF_BENCH_UNTRACKED_FILE_COUNT
         << ", \"metalSourceSha256\": \"sha256:"
            OAF_BENCH_METAL_SOURCE_SHA256 "\"},\n"
         << "  \"device\": \"" << deviceName << "\",\n"
         << "  \"width\": " << options.width << ",\n"
         << "  \"height\": " << options.height << ",\n"
         << "  \"frameCount\": " << options.frames << ",\n"
         << "  \"tileRows\": " << options.tileRows << ",\n"
         << "  \"tileCount\": " << tiles.size() << ",\n"
         << "  \"processedPixelFrames\": " << processedPixelFrames << ",\n"
         << "  \"warmupRuns\": " << options.warmupRuns << ",\n"
         << "  \"repetitions\": " << options.repetitions << ",\n"
         << "  \"wallSeconds\": " << wallSeconds << ",\n"
         << "  \"wallSecondsStatistic\": \"median\",\n"
         << "  \"wallSecondsMinimum\": "
         << *std::min_element( wallSamples.begin(), wallSamples.end() ) << ",\n"
         << "  \"wallSecondsMaximum\": "
         << *std::max_element( wallSamples.begin(), wallSamples.end() ) << ",\n"
         << "  \"wallSecondsSamples\": ";
      WriteSamples( output, wallSamples );
      output << ",\n"
         << "  \"syntheticPreparationSeconds\": "
         << syntheticPreparationSeconds << ",\n"
         << "  \"metalCallSeconds\": " << metalCallSeconds << ",\n"
         << "  \"gpuSeconds\": " << gpuSeconds << ",\n"
         << "  \"megapixelFramesPerSecond\": "
         << megapixelFramesPerSecond << ",\n"
         << "  \"linearProjected96FrameSeconds\": "
         << linearProjected96FrameSeconds << ",\n"
         << "  \"submittedBufferBytes\": " << submittedBytes << ",\n"
         << "  \"outputNormalizationWallSeconds\": "
         << normalizationWallSeconds << ",\n"
         << "  \"outputNormalizationGpuSeconds\": "
         << normalizationGpuSeconds << ",\n"
         << "  \"outputFiniteMinimum\": "
         << outputRange.finiteMinimum << ",\n"
         << "  \"outputFiniteMaximum\": "
         << outputRange.finiteMaximum << ",\n"
         << "  \"outputScale\": " << outputRange.scale << ",\n"
         << "  \"outputFnv1a64\": \"" << std::hex << std::setw( 16 )
         << std::setfill( '0' ) << outputHash << std::dec << std::setfill( ' ' )
         << "\",\n"
         << "  \"outputStableAcrossRuns\": true,\n"
         << "  \"automaticOutputNormalization\": true,\n"
         << "  \"recommendedWorkingSetBytes\": "
         << recommendedWorkingSet << ",\n"
         << "  \"oracleAssisted\": true,\n"
         << "  \"includesXnmlGeneration\": false,\n"
         << "  \"includesRejectionGeneration\": false,\n"
         << "  \"includesImageIo\": false,\n"
         << "  \"timedRuntimeShaderCompilationIncluded\": false,\n"
         << "  \"kernelTargetSeconds\": 105,\n"
         << "  \"kernelFeasibilityGate\": "
         << (kernelFeasibilityGate ? "true" : "false") << "\n"
         << "}\n";
      output.close();
      if ( !output )
         throw std::runtime_error(
            "Ultra-Fast WBPP native benchmark output write failed" );
      std::cout << options.output << '\n';
      return 0;
   }
   catch ( const std::exception& error )
   {
      std::cerr << "UltraFastWBPPBenchmark failed: " << error.what() << '\n';
      return 1;
   }
}
