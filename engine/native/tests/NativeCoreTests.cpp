#include "openastroflow/FusedLnIntegration.h"
#include "openastroflow/ImageTile.h"
#if defined(OAF_WITH_METAL)
#include "openastroflow/MetalFusedLnIntegration.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using namespace openastroflow::native;

void Require( bool condition, const std::string& message )
{
   if ( !condition )
      throw std::runtime_error( message );
}

template <class Exception, class Function>
void RequireThrows( Function&& function, const std::string& message )
{
   bool threw = false;
   try
   {
      function();
   }
   catch ( const Exception& )
   {
      threw = true;
   }
   Require( threw, message );
}

struct Fixture
{
   ImageGeometry geometry{ 37, 29, 1 };
   TileRegion tile{ geometry, 0, geometry.height };
   std::uint32_t frames = 5;
   std::uint32_t gridWidth = 7;
   std::uint32_t gridHeight = 5;
   std::vector<float> samples;
   std::vector<std::uint8_t> mask;
   std::vector<float> scale;
   std::vector<float> offset;
   std::vector<float> weights;

   Fixture()
   {
      const std::size_t pixels = tile.PixelCount();
      const std::size_t grids =
         static_cast<std::size_t>( gridWidth )*gridHeight;
      samples.resize( frames*pixels );
      mask.assign( frames*pixels, 0 );
      scale.resize( frames*grids );
      offset.resize( frames*grids );
      weights.resize( frames );
      std::mt19937 generator( 0x10f00dU );
      std::uniform_real_distribution<float> sampleDistribution(
         -0.01F, 0.08F );
      for ( float& sample : samples )
         sample = sampleDistribution( generator );
      for ( std::uint32_t frame = 0; frame < frames; ++frame )
      {
         weights[frame] = 0.5F + 0.125F*frame;
         for ( std::uint32_t y = 0; y < gridHeight; ++y )
            for ( std::uint32_t x = 0; x < gridWidth; ++x )
            {
               const std::size_t index =
                  static_cast<std::size_t>( frame )*grids
                  + static_cast<std::size_t>( y )*gridWidth + x;
               scale[index] = 0.88F + 0.015F*frame
                            + 0.0007F*x - 0.0004F*y;
               offset[index] = -0.0012F + 0.00003F*frame
                             + 0.000002F*(x + 2*y);
            }
      }
      for ( std::size_t pixel = 0; pixel < pixels; pixel += 113 )
         mask[pixels + pixel] = 0x01;
      for ( std::size_t pixel = 17; pixel < pixels; pixel += 127 )
         mask[3*pixels + pixel] = 0x02;
      samples[4*pixels + 9] = std::numeric_limits<float>::quiet_NaN();
   }

   FusedLnIntegrationRequest Request() const
   {
      return {
         tile, frames, gridWidth, gridHeight,
         samples, mask, scale, offset, weights,
         DefaultOracleRejectionBits
      };
   }
};

void TestTilePartition()
{
   const ImageGeometry geometry{ 6252, 4176, 1 };
   const std::vector<TileRegion> tiles = PartitionRows( geometry, 144 );
   Require( tiles.size() == 29,
            "full-size image did not partition into 29 exact tiles" );
   std::uint32_t next = 0;
   std::size_t samples = 0;
   for ( const TileRegion& tile : tiles )
   {
      tile.Validate();
      Require( tile.firstRow == next && tile.rowCount == 144,
               "tile partition has a gap or unexpected row count" );
      next += tile.rowCount;
      samples += tile.PixelCount();
   }
   Require( next == geometry.height && samples == geometry.SampleCount(),
            "tile partition does not cover the image exactly" );
   RequireThrows<std::invalid_argument>(
      [&] { (void)PartitionRows( geometry, 0 ); },
      "zero-row partition was accepted" );
}

void TestConstantBicubicGrid()
{
   const std::vector<float> grid( 7*5, 0.9035055F );
   for ( const auto [x, y] : {
         std::pair{ 0.0F, 0.0F }, std::pair{ 0.01F, 0.02F },
         std::pair{ 3.5F, 2.25F }, std::pair{ 6.99F, 4.99F } } )
      Require( std::abs( BicubicBSplineSample( grid, 7, 5, x, y )
                         - 0.9035055F ) < 2.0e-6F,
               "B-spline did not preserve a constant grid at an edge" );
}

void TestCpuOracleMaskAndWeights()
{
   ImageGeometry geometry{ 2, 2, 1 };
   TileRegion tile{ geometry, 0, 2 };
   const std::vector<float> samples{ 1, 2, 0, 0, 3, 4, 0, 0 };
   const std::vector<std::uint8_t> mask{ 0, 0, 0, 0, 1, 0, 0, 0 };
   const std::vector<float> scale( 2*4, 2.0F );
   const std::vector<float> offset( 2*4, -1.0F );
   const std::vector<float> weights{ 1, 3 };
   const FusedLnIntegrationRequest request{
      tile, 2, 2, 2, samples, mask, scale, offset, weights, 0x3f
   };
   const FusedLnIntegrationResult result =
      RunCpuOracleFusedLnIntegration( request );
   Require( result.acceptedSamples
               == std::vector<std::uint16_t>{ 1, 2, 2, 2 }
         && result.rejectedSamples
               == std::vector<std::uint16_t>{ 1, 0, 0, 0 },
            "CPU oracle rejection counts differ" );
   Require( std::abs( result.integrated[0] - 1.0F ) < 1.0e-6F
         && std::abs( result.integrated[1] - 6.0F ) < 1.0e-6F,
            "CPU oracle LN/weighted integration differs" );
}

void TestCpuLinearFitSupportsFiveHundredTwelveFrames()
{
   constexpr std::uint32_t frames = 512;
   const ImageGeometry geometry{ 2, 2, 1 };
   const TileRegion tile{ geometry, 0, 2 };
   const std::size_t pixels = tile.PixelCount();
   std::vector<float> samples( frames*pixels, 2.0F );
   for ( std::uint32_t frame = 0; frame < 64; ++frame )
      std::fill_n( samples.begin() + frame*pixels, pixels, 1.0F );
   const std::vector<float> scales( frames*4, 1.0F );
   const std::vector<float> offsets( frames*4, 0.0F );
   const std::vector<float> weights( frames, 1.0F );
   const NativeLinearFitIntegrationRequest request{
      tile, frames, 2, 2, samples, scales, offsets, weights,
      -1.0F, 5.0F, 5.0F
   };
   const FusedLnIntegrationResult result =
      RunCpuOracleNativeLinearFitIntegration( request );
   for ( std::size_t pixel = 0; pixel < pixels; ++pixel )
   {
      Require( static_cast<std::uint32_t>( result.acceptedSamples[pixel] )
                  + result.rejectedSamples[pixel] == frames,
               "512-frame CPU native accounting differs" );
      Require( result.integrated[pixel] > 1.5F,
               "512-frame CPU native path truncated after frame 64" );
   }
}

#if defined(OAF_WITH_METAL)
void TestMetalMatchesCpu( const std::filesystem::path& metalSource )
{
   Fixture fixture;
   const FusedLnIntegrationRequest request = fixture.Request();
   const FusedLnIntegrationResult cpu =
      RunCpuOracleFusedLnIntegration( request );
   MetalExecutionStats stats;
   MetalFusedLnIntegrationExecutor executor( metalSource );
   const FusedLnIntegrationResult gpu = executor.Run( request, &stats );
   Require( gpu.acceptedSamples == cpu.acceptedSamples
         && gpu.rejectedSamples == cpu.rejectedSamples,
            "Metal and CPU oracle sample counts differ" );
   float maximum = 0;
   double squareSum = 0;
   std::size_t finite = 0;
   for ( std::size_t i = 0; i < cpu.integrated.size(); ++i )
   {
      const bool finiteCpu = std::isfinite( cpu.integrated[i] );
      const bool finiteGpu = std::isfinite( gpu.integrated[i] );
      Require( finiteCpu == finiteGpu,
               "Metal and CPU oracle finite masks differ" );
      if ( finiteCpu )
      {
         const float difference =
            std::abs( cpu.integrated[i] - gpu.integrated[i] );
         maximum = std::max( maximum, difference );
         squareSum += static_cast<double>( difference )*difference;
         ++finite;
      }
   }
   const double rmse = finite == 0 ? 0 : std::sqrt( squareSum/finite );
   Require( maximum <= 2.0e-6F && rmse <= 2.0e-7,
            "Metal LN oracle exceeds the numerical gate: max="
            + std::to_string( maximum )
            + " rmse=" + std::to_string( rmse ) );
   Require( !stats.deviceName.empty() && stats.wallSeconds >= 0
         && stats.submittedBufferBytes > 0,
            "Metal execution statistics are incomplete" );
}

void TestMetalOutputRangeNormalization(
   const std::filesystem::path& metalSource )
{
   std::vector<float> samples{
      -0.25F, 0.0F, 0.5F, 1.5F,
      std::numeric_limits<float>::quiet_NaN()
   };
   MetalFusedLnIntegrationExecutor executor( metalSource );
   MetalExecutionStats stats;
   const OutputRangeNormalization range =
      executor.NormalizeOutputRangeInPlace( samples, &stats );
   Require( range.finiteSamples == 4 && range.applied
         && std::abs( range.finiteMinimum + 0.25F ) < 1.0e-7F
         && std::abs( range.finiteMaximum - 1.5F ) < 1.0e-7F
         && std::abs( range.scale - 1.0F/1.75F ) < 1.0e-7F
         && std::abs( range.offset - 0.25F/1.75F ) < 1.0e-7F,
            "Metal output-range parameters differ" );
   Require( std::abs( samples[0] ) < 1.0e-7F
         && std::abs( samples[3] - 1.0F ) < 1.0e-7F
         && std::isnan( samples[4] ),
            "Metal output-range normalization differs" );

   std::vector<float> positive{ 0.001F, 0.25F, 1.10127938F };
   const OutputRangeNormalization positiveRange =
      executor.NormalizeOutputRangeInPlace( positive );
   Require( std::abs( positiveRange.effectiveMinimum ) < 1.0e-8F
         && std::abs( positiveRange.scale
                       - 1.0F/1.10127938F ) < 1.0e-7F
         && std::abs( positive.front()
                       - 0.001F/1.10127938F ) < 1.0e-7F,
            "positive integration minimum was incorrectly subtracted" );
}

void TestMetalNativeRobustRejection(
   const std::filesystem::path& metalSource )
{
   Fixture fixture;
   const std::size_t pixels = fixture.tile.PixelCount();
   for ( std::size_t pixel = 23; pixel < pixels; pixel += 89 )
      fixture.samples[pixel] += 0.8F;
   for ( std::size_t pixel = 47; pixel < pixels; pixel += 131 )
      fixture.samples[2*pixels + pixel] -= 0.8F;
   const NativeRobustIntegrationRequest request{
      fixture.tile, fixture.frames, fixture.gridWidth, fixture.gridHeight,
      fixture.samples, fixture.scale, fixture.offset, fixture.weights
   };
   const FusedLnIntegrationResult cpu =
      RunCpuOracleNativeRobustIntegration( request );
   MetalFusedLnIntegrationExecutor executor( metalSource );
   const FusedLnIntegrationResult gpu =
      executor.RunNativeRobustRejection( request );
   Require( gpu.acceptedSamples == cpu.acceptedSamples
         && gpu.rejectedSamples == cpu.rejectedSamples,
            "Metal and CPU native rejection counts differ" );
   Require( std::any_of( gpu.rejectedSamples.begin(),
                         gpu.rejectedSamples.end(),
                         []( std::uint16_t value ) { return value != 0; } ),
            "native robust rejection did not reject injected outliers" );
   float maximum = 0;
   double squareSum = 0;
   std::size_t finite = 0;
   for ( std::size_t i = 0; i < cpu.integrated.size(); ++i )
   {
      Require( std::isfinite( cpu.integrated[i] )
                  == std::isfinite( gpu.integrated[i] ),
               "native robust CPU/GPU finite masks differ" );
      if ( std::isfinite( cpu.integrated[i] ) )
      {
         const float difference =
            std::abs( cpu.integrated[i] - gpu.integrated[i] );
         maximum = std::max( maximum, difference );
         squareSum += static_cast<double>( difference )*difference;
         ++finite;
      }
   }
   const double rmse = std::sqrt( squareSum/finite );
   Require( maximum <= 2.0e-6F && rmse <= 2.0e-7,
            "native robust Metal integration exceeds numerical gate" );
}

void TestMetalNativeLinearFitRejection(
   const std::filesystem::path& metalSource )
{
   Fixture fixture;
   const std::size_t pixels = fixture.tile.PixelCount();
   for ( std::size_t pixel = 11; pixel < pixels; pixel += 61 )
   {
      fixture.samples[pixel] += 0.55F;
      fixture.samples[pixels + pixel] += 0.35F;
   }
   for ( std::size_t pixel = 29; pixel < pixels; pixel += 97 )
      fixture.samples[2*pixels + pixel] -= 0.55F;
   const NativeLinearFitIntegrationRequest request{
      fixture.tile, fixture.frames, fixture.gridWidth, fixture.gridHeight,
      fixture.samples, fixture.scale, fixture.offset, fixture.weights
   };
   const FusedLnIntegrationResult cpu =
      RunCpuOracleNativeLinearFitIntegration( request );
   MetalFusedLnIntegrationExecutor executor( metalSource );
   const FusedLnIntegrationResult gpu =
      executor.RunNativeLinearFitRejection( request );
   Require( gpu.acceptedSamples == cpu.acceptedSamples
         && gpu.rejectedSamples == cpu.rejectedSamples,
            "Metal and CPU linear-fit rejection counts differ" );
   float maximum = 0;
   double squareSum = 0;
   std::size_t finite = 0;
   for ( std::size_t i = 0; i < cpu.integrated.size(); ++i )
   {
      Require( std::isfinite( cpu.integrated[i] )
                  == std::isfinite( gpu.integrated[i] ),
               "linear-fit CPU/GPU finite masks differ" );
      if ( std::isfinite( cpu.integrated[i] ) )
      {
         const float difference =
            std::abs( cpu.integrated[i] - gpu.integrated[i] );
         maximum = std::max( maximum, difference );
         squareSum += static_cast<double>( difference )*difference;
         ++finite;
      }
   }
   Require( maximum <= 2.0e-6F
         && std::sqrt( squareSum/finite ) <= 2.0e-7,
            "linear-fit Metal integration exceeds numerical gate" );
}

void TestMetalMaskedFiveHundredTwelveFrames(
   const std::filesystem::path& metalSource )
{
   constexpr std::uint32_t frames = 512;
   const ImageGeometry geometry{ 2, 2, 1 };
   const TileRegion tile{ geometry, 0, 2 };
   const std::size_t pixels = tile.PixelCount();
   std::vector<float> samples( frames*pixels );
   std::vector<std::uint8_t> mask( frames*pixels, 0 );
   for ( std::uint32_t frame = 0; frame < frames; ++frame )
      for ( std::size_t pixel = 0; pixel < pixels; ++pixel )
         samples[static_cast<std::size_t>( frame )*pixels + pixel] =
            0.1F + 0.00001F*frame + 0.000001F*pixel;
   for ( std::uint32_t frame = 0; frame < frames; frame += 11 )
      mask[static_cast<std::size_t>( frame )*pixels + frame%pixels] = 1;
   const std::vector<float> scales( frames*4, 1.0F );
   const std::vector<float> offsets( frames*4, 0.0F );
   const std::vector<float> weights( frames, 1.0F );
   const FusedLnIntegrationRequest request{
      tile, frames, 2, 2, samples, mask, scales, offsets, weights, 1
   };
   const FusedLnIntegrationResult cpu =
      RunCpuOracleFusedLnIntegration( request );
   MetalFusedLnIntegrationExecutor executor( metalSource );
   const FusedLnIntegrationResult gpu = executor.Run( request );
   Require( gpu.acceptedSamples == cpu.acceptedSamples
         && gpu.rejectedSamples == cpu.rejectedSamples,
            "512-frame masked Metal rejection counts differ" );
   float maximum = 0;
   double squareSum = 0;
   for ( std::size_t pixel = 0; pixel < pixels; ++pixel )
   {
      Require( static_cast<std::uint32_t>( gpu.acceptedSamples[pixel] )
                  + gpu.rejectedSamples[pixel] == frames,
               "512-frame masked Metal truncated the full stack" );
      Require( std::isfinite( cpu.integrated[pixel] )
                  == std::isfinite( gpu.integrated[pixel] ),
               "512-frame masked Metal finite mask differs" );
      const float difference = std::abs(
         cpu.integrated[pixel] - gpu.integrated[pixel] );
      maximum = std::max( maximum, difference );
      squareSum += static_cast<double>( difference )*difference;
   }
   Require( maximum <= 2.0e-6F
         && std::sqrt( squareSum/pixels ) <= 2.0e-7,
            "512-frame masked Metal numerical gate failed" );
}
#endif

} // namespace

int main( int argc, char** argv )
{
   try
   {
      TestTilePartition();
      TestConstantBicubicGrid();
      TestCpuOracleMaskAndWeights();
      TestCpuLinearFitSupportsFiveHundredTwelveFrames();
#if defined(OAF_WITH_METAL)
      if ( argc != 2 )
         throw std::invalid_argument(
            "usage: OpenAstroFlowNativeCoreTests /absolute/path/to/source.metal" );
      const std::filesystem::path metalSource =
         std::filesystem::canonical( argv[1] );
      if ( !MetalFusedLnIntegrationAvailable() )
      {
         std::cout << "OpenAstroFlowNativeCoreTests skipped: Metal is unavailable\n";
         return 77;
      }
      TestMetalMatchesCpu( metalSource );
      TestMetalOutputRangeNormalization( metalSource );
      TestMetalNativeRobustRejection( metalSource );
      TestMetalNativeLinearFitRejection( metalSource );
      TestMetalMaskedFiveHundredTwelveFrames( metalSource );
#else
      if ( argc != 1 )
         throw std::invalid_argument(
            "OpenAstroFlowNativeCoreTests takes no arguments without Metal" );
#endif
      std::cout << "OpenAstroFlowNativeCoreTests passed\n";
      return 0;
   }
   catch ( const std::exception& error )
   {
      std::cerr << "OpenAstroFlowNativeCoreTests failed: " << error.what() << '\n';
      return 1;
   }
}
