#include "openastroflow/FusedLnIntegration.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace openastroflow::native
{

namespace
{

std::size_t CheckedMultiply( std::size_t left,
                             std::size_t right,
                             const char* role )
{
   if ( left != 0 && right > std::numeric_limits<std::size_t>::max()/left )
      throw std::overflow_error( role );
   return left*right;
}

int ReflectedIndex( int index, int length ) noexcept
{
   if ( index < 0 )
      return -index - 1;
   if ( index >= length )
      return 2*length - index - 1;
   return index;
}

float BSplineWeight( float x ) noexcept
{
   const auto positiveCube = []( float value ) noexcept
   {
      return value > 0 ? value*value*value : 0.0F;
   };
   return (positiveCube( x + 2 ) - 4*positiveCube( x + 1 )
         + 6*positiveCube( x ) - 4*positiveCube( x - 1 ))/6.0F;
}

float MedianInPlace( std::span<float> values, std::uint32_t count )
{
   const std::uint32_t middle = count/2;
   std::nth_element( values.begin(), values.begin() + middle,
                     values.begin() + count );
   const float upper = values[middle];
   if ( (count & 1U) != 0 )
      return upper;
   return 0.5F*(upper + *std::max_element(
      values.begin(), values.begin() + middle ));
}

struct LadLine
{
   float a = 0;
   float b = 0;
   float averageDeviation = 0;
};

LadLine FitSortedLadLine(
   std::span<const float> values,
   std::uint32_t count,
   std::uint32_t iterations )
{
   float low = 0;
   float high = std::max(
      1.0e-12F, 4.0F*(values[count - 1] - values[0])/(count - 1) );
   std::vector<float> scratch( count );
   float intercept = 0;
   for ( std::uint32_t iteration = 0; iteration < iterations; ++iteration )
   {
      const float slope = 0.5F*(low + high);
      for ( std::uint32_t i = 0; i < count; ++i )
         scratch[i] = values[i] - slope*i;
      intercept = MedianInPlace( scratch, count );
      float score = 0;
      for ( std::uint32_t i = 0; i < count; ++i )
      {
         const float residual = values[i] - (intercept + slope*i);
         score += residual > 0 ? static_cast<float>( i )
                : residual < 0 ? -static_cast<float>( i ) : 0.0F;
      }
      if ( score > 0 ) low = slope;
      else high = slope;
   }
   LadLine line;
   line.b = 0.5F*(low + high);
   for ( std::uint32_t i = 0; i < count; ++i )
      scratch[i] = values[i] - line.b*i;
   line.a = MedianInPlace( scratch, count );
   for ( std::uint32_t i = 0; i < count; ++i )
      line.averageDeviation += std::abs(
         values[i] - (line.a + line.b*i) );
   line.averageDeviation /= count;
   return line;
}

} // namespace

std::size_t FusedLnIntegrationRequest::TilePixels() const
{
   return tile.PixelCount();
}

std::size_t FusedLnIntegrationRequest::GridSamplesPerFrame() const
{
   return CheckedMultiply(
      gridWidth, gridHeight, "Ultra-Fast WBPP native LN grid size overflow" );
}

void FusedLnIntegrationRequest::Validate() const
{
   tile.Validate();
   if ( tile.image.channels != 1 )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native fused LN prototype currently accepts mono images only" );
   if ( frameCount == 0 || frameCount > 65535 )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native fused LN frame count is invalid" );
   if ( gridWidth < 2 || gridHeight < 2
     || gridWidth > tile.image.width || gridHeight > tile.image.height )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native fused LN grid geometry is invalid" );

   const std::size_t pixels = TilePixels();
   const std::size_t framePixels = CheckedMultiply(
      frameCount, pixels, "Ultra-Fast WBPP native frame tile size overflow" );
   const std::size_t gridSamples = CheckedMultiply(
      frameCount, GridSamplesPerFrame(),
      "Ultra-Fast WBPP native frame grid size overflow" );
   if ( frameMajorSamples.size() != framePixels
     || frameMajorRejectionMask.size() != framePixels
     || frameMajorScaleGrid.size() != gridSamples
     || frameMajorZeroOffsetGrid.size() != gridSamples
     || frameWeights.size() != frameCount )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native fused LN request cardinality differs" );
   if ( rejectionBits == 0 )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native fused LN rejection bit mask is empty" );
   if ( !std::isfinite( outputScale ) || outputScale <= 0
     || !std::isfinite( outputOffset ) )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native fused LN output normalization is invalid" );
   for ( float weight : frameWeights )
      if ( !std::isfinite( weight ) || weight < 0 )
         throw std::invalid_argument(
            "Ultra-Fast WBPP native fused LN frame weight is invalid" );
}

std::size_t NativeRobustIntegrationRequest::TilePixels() const
{
   return tile.PixelCount();
}

std::size_t NativeRobustIntegrationRequest::GridSamplesPerFrame() const
{
   return CheckedMultiply(
      gridWidth, gridHeight, "Ultra-Fast WBPP native native rejection grid overflow" );
}

void NativeRobustIntegrationRequest::Validate() const
{
   tile.Validate();
   if ( tile.image.channels != 1 || frameCount < 3
     || frameCount > MaximumNativeRejectionFrames )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native robust integration geometry/frame count is invalid" );
   if ( gridWidth < 2 || gridHeight < 2
     || gridWidth > tile.image.width || gridHeight > tile.image.height )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native robust integration grid geometry is invalid" );
   const std::size_t pixels = TilePixels();
   const std::size_t framePixels = CheckedMultiply(
      frameCount, pixels, "Ultra-Fast WBPP native robust frame tile overflow" );
   const std::size_t gridSamples = CheckedMultiply(
      frameCount, GridSamplesPerFrame(),
      "Ultra-Fast WBPP native robust frame grid overflow" );
   if ( frameMajorSamples.size() != framePixels
     || frameMajorScaleGrid.size() != gridSamples
     || frameMajorZeroOffsetGrid.size() != gridSamples
     || frameWeights.size() != frameCount )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native robust integration cardinality differs" );
   if ( !std::isfinite( rangeLow ) || !std::isfinite( lowSigma )
     || !std::isfinite( highSigma ) || !std::isfinite( winsorSigma )
     || lowSigma <= 0 || highSigma <= 0 || winsorSigma <= 0
     || winsorIterations == 0 || winsorIterations > 8
     || !std::isfinite( outputScale ) || outputScale <= 0
     || !std::isfinite( outputOffset ) )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native robust integration parameters are invalid" );
   for ( float weight : frameWeights )
      if ( !std::isfinite( weight ) || weight < 0 )
         throw std::invalid_argument(
            "Ultra-Fast WBPP native robust integration weight is invalid" );
}

std::size_t NativeLinearFitIntegrationRequest::TilePixels() const
{
   return tile.PixelCount();
}

std::size_t NativeLinearFitIntegrationRequest::GridSamplesPerFrame() const
{
   return CheckedMultiply(
      gridWidth, gridHeight, "Ultra-Fast WBPP native linear-fit grid overflow" );
}

void NativeLinearFitIntegrationRequest::Validate() const
{
   tile.Validate();
   if ( tile.image.channels != 1 || frameCount < 5
     || frameCount > MaximumNativeRejectionFrames )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native linear-fit geometry/frame count is invalid" );
   if ( gridWidth < 2 || gridHeight < 2
     || gridWidth > tile.image.width || gridHeight > tile.image.height )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native linear-fit grid geometry is invalid" );
   const std::size_t framePixels = CheckedMultiply(
      frameCount, TilePixels(), "Ultra-Fast WBPP native linear-fit tile overflow" );
   const std::size_t gridSamples = CheckedMultiply(
      frameCount, GridSamplesPerFrame(),
      "Ultra-Fast WBPP native linear-fit frame grid overflow" );
   if ( frameMajorSamples.size() != framePixels
     || frameMajorScaleGrid.size() != gridSamples
     || frameMajorZeroOffsetGrid.size() != gridSamples
     || frameWeights.size() != frameCount )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native linear-fit cardinality differs" );
   if ( !std::isfinite( rangeLow ) || !std::isfinite( lowTolerance )
     || !std::isfinite( highTolerance ) || lowTolerance <= 0
     || highTolerance <= 0 || fitBisectionIterations < 4
     || fitBisectionIterations > 24 || rejectionIterations == 0
     || rejectionIterations > 16 || !std::isfinite( outputScale )
     || outputScale <= 0 || !std::isfinite( outputOffset ) )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native linear-fit parameters are invalid" );
   for ( float weight : frameWeights )
      if ( !std::isfinite( weight ) || weight < 0 )
         throw std::invalid_argument(
            "Ultra-Fast WBPP native linear-fit weight is invalid" );
}

float BicubicBSplineSample(
   std::span<const float> grid,
   std::uint32_t width,
   std::uint32_t height,
   float x,
   float y )
{
   if ( width < 2 || height < 2
     || grid.size() != static_cast<std::size_t>( width )*height
     || !std::isfinite( x ) || !std::isfinite( y )
     || x < 0 || y < 0 || x >= width || y >= height )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native B-spline interpolation request is invalid" );

   const int x1 = static_cast<int>( std::floor( x ) );
   const int y1 = static_cast<int>( std::floor( y ) );
   const float dx = x - x1;
   const float dy = y - y1;
   float result = 0;
   for ( int row = 0; row < 4; ++row )
   {
      const int gy = ReflectedIndex( y1 + row - 1,
                                     static_cast<int>( height ) );
      const float wy = BSplineWeight( static_cast<float>( row - 1 ) - dy );
      for ( int column = 0; column < 4; ++column )
      {
         const int gx = ReflectedIndex( x1 + column - 1,
                                        static_cast<int>( width ) );
         const float wx = BSplineWeight(
            static_cast<float>( column - 1 ) - dx );
         result += grid[static_cast<std::size_t>( gy )*width + gx]*wx*wy;
      }
   }
   return result;
}

FusedLnIntegrationResult RunCpuOracleFusedLnIntegration(
   const FusedLnIntegrationRequest& request )
{
   request.Validate();
   const std::size_t pixels = request.TilePixels();
   const std::size_t gridSamples = request.GridSamplesPerFrame();
   FusedLnIntegrationResult result;
   result.integrated.resize( pixels );
   result.acceptedSamples.resize( pixels );
   result.rejectedSamples.resize( pixels );

   std::vector<std::uint8_t> identityNormalization( request.frameCount, 0 );
   if ( request.gridWidth == 2 && request.gridHeight == 2 )
      for ( std::uint32_t frame = 0; frame < request.frameCount; ++frame )
      {
         const std::size_t offset = static_cast<std::size_t>( frame )*4;
         identityNormalization[frame] =
            std::all_of(
               request.frameMajorScaleGrid.begin() + offset,
               request.frameMajorScaleGrid.begin() + offset + 4,
               []( float value ) { return value == 1.0F; } )
         && std::all_of(
               request.frameMajorZeroOffsetGrid.begin() + offset,
               request.frameMajorZeroOffsetGrid.begin() + offset + 4,
               []( float value ) { return value == 0.0F; } );
      }
   const float sx = static_cast<float>( request.gridWidth )
                  / request.tile.image.width;
   const float sy = static_cast<float>( request.gridHeight )
                  / request.tile.image.height;
   for ( std::size_t pixel = 0; pixel < pixels; ++pixel )
   {
      const std::uint32_t x = static_cast<std::uint32_t>(
         pixel % request.tile.image.width );
      const std::uint32_t localY = static_cast<std::uint32_t>(
         pixel / request.tile.image.width );
      const std::uint32_t y = request.tile.firstRow + localY;
      double weighted = 0;
      double weightSum = 0;
      std::uint16_t accepted = 0;
      std::uint16_t rejected = 0;
      for ( std::uint32_t frame = 0; frame < request.frameCount; ++frame )
      {
         const std::size_t sampleIndex =
            static_cast<std::size_t>( frame )*pixels + pixel;
         if ( (request.frameMajorRejectionMask[sampleIndex]
                & request.rejectionBits) != 0 )
         {
            ++rejected;
            continue;
         }
         const float input = request.frameMajorSamples[sampleIndex];
         const float weight = request.frameWeights[frame];
         if ( !std::isfinite( input ) || weight <= 0 )
         {
            ++rejected;
            continue;
         }
         const std::size_t gridOffset =
            static_cast<std::size_t>( frame )*gridSamples;
         const bool identity = identityNormalization[frame] != 0;
         const float a = identity ? 1.0F : BicubicBSplineSample(
            request.frameMajorScaleGrid.subspan( gridOffset, gridSamples ),
            request.gridWidth, request.gridHeight, sx*x, sy*y );
         const float b = identity ? 0.0F : BicubicBSplineSample(
            request.frameMajorZeroOffsetGrid.subspan(
               gridOffset, gridSamples ),
            request.gridWidth, request.gridHeight, sx*x, sy*y );
         const float normalized = identity ? input : a*input + b;
         if ( !std::isfinite( normalized ) )
         {
            ++rejected;
            continue;
         }
         weighted += static_cast<double>( weight )*normalized;
         weightSum += weight;
         ++accepted;
      }
      result.integrated[pixel] = weightSum > 0
         ? request.outputScale*static_cast<float>( weighted/weightSum )
             + request.outputOffset
         : 0.0F;
      result.acceptedSamples[pixel] = accepted;
      result.rejectedSamples[pixel] = rejected;
   }
   return result;
}

FusedLnIntegrationResult RunCpuOracleNativeRobustIntegration(
   const NativeRobustIntegrationRequest& request )
{
   request.Validate();
   const std::size_t pixels = request.TilePixels();
   const std::size_t gridSamples = request.GridSamplesPerFrame();
   FusedLnIntegrationResult result;
   result.integrated.resize( pixels );
   result.acceptedSamples.resize( pixels );
   result.rejectedSamples.resize( pixels );
   const float sx = static_cast<float>( request.gridWidth )
                  / request.tile.image.width;
   const float sy = static_cast<float>( request.gridHeight )
                  / request.tile.image.height;
   std::vector<float> values( request.frameCount );
   std::vector<std::uint8_t> valid( request.frameCount );
   for ( std::size_t pixel = 0; pixel < pixels; ++pixel )
   {
      const std::uint32_t x = static_cast<std::uint32_t>(
         pixel % request.tile.image.width );
      const std::uint32_t y = request.tile.firstRow
         + static_cast<std::uint32_t>( pixel/request.tile.image.width );
      double sum = 0, squareSum = 0;
      std::uint32_t validCount = 0;
      for ( std::uint32_t frame = 0; frame < request.frameCount; ++frame )
      {
         const std::size_t sampleIndex =
            static_cast<std::size_t>( frame )*pixels + pixel;
         const std::size_t gridOffset =
            static_cast<std::size_t>( frame )*gridSamples;
         const float a = BicubicBSplineSample(
            request.frameMajorScaleGrid.subspan( gridOffset, gridSamples ),
            request.gridWidth, request.gridHeight, sx*x, sy*y );
         const float b = BicubicBSplineSample(
            request.frameMajorZeroOffsetGrid.subspan(
               gridOffset, gridSamples ),
            request.gridWidth, request.gridHeight, sx*x, sy*y );
         const float input = request.frameMajorSamples[sampleIndex];
         const float value = a*input + b;
         values[frame] = value;
         valid[frame] = std::isfinite( input ) && input > request.rangeLow
                     && std::isfinite( value )
                     && request.frameWeights[frame] > 0;
         if ( valid[frame] )
         {
            sum += value;
            squareSum += static_cast<double>( value )*value;
            ++validCount;
         }
      }
      if ( validCount == 0 )
      {
         result.integrated[pixel] = 0;
         result.acceptedSamples[pixel] = 0;
         result.rejectedSamples[pixel] =
            static_cast<std::uint16_t>( request.frameCount );
         continue;
      }
      double mean = sum/validCount;
      double variance = std::max( 0.0, squareSum/validCount - mean*mean );
      double deviation = std::sqrt( variance );
      for ( std::uint32_t iteration = 0;
            iteration < request.winsorIterations && deviation > 0;
            ++iteration )
      {
         const double low = mean - request.winsorSigma*deviation;
         const double high = mean + request.winsorSigma*deviation;
         sum = squareSum = 0;
         for ( std::uint32_t frame = 0; frame < request.frameCount; ++frame )
            if ( valid[frame] )
            {
               const double value = std::clamp<double>(
                  values[frame], low, high );
               sum += value;
               squareSum += value*value;
            }
         mean = sum/validCount;
         variance = std::max(
            0.0, squareSum/validCount - mean*mean );
         deviation = std::sqrt( variance );
      }
      const double low = mean - request.lowSigma*deviation;
      const double high = mean + request.highSigma*deviation;
      double weighted = 0, weightSum = 0;
      std::uint16_t accepted = 0, rejected = 0;
      for ( std::uint32_t frame = 0; frame < request.frameCount; ++frame )
      {
         if ( !valid[frame] || values[frame] < low || values[frame] > high )
         {
            ++rejected;
            continue;
         }
         weighted += static_cast<double>( request.frameWeights[frame] )
                   * values[frame];
         weightSum += request.frameWeights[frame];
         ++accepted;
      }
      result.integrated[pixel] = weightSum > 0
         ? request.outputScale*static_cast<float>( weighted/weightSum )
             + request.outputOffset
         : 0.0F;
      result.acceptedSamples[pixel] = accepted;
      result.rejectedSamples[pixel] = rejected;
   }
   return result;
}

FusedLnIntegrationResult RunCpuOracleNativeLinearFitIntegration(
   const NativeLinearFitIntegrationRequest& request )
{
   request.Validate();
   const std::size_t pixels = request.TilePixels();
   const std::size_t gridSamples = request.GridSamplesPerFrame();
   FusedLnIntegrationResult result;
   result.integrated.resize( pixels );
   result.acceptedSamples.resize( pixels );
   result.rejectedSamples.resize( pixels );
   const float sx = static_cast<float>( request.gridWidth )
                  / request.tile.image.width;
   const float sy = static_cast<float>( request.gridHeight )
                  / request.tile.image.height;
   for ( std::size_t pixel = 0; pixel < pixels; ++pixel )
   {
      const std::uint32_t x = static_cast<std::uint32_t>(
         pixel % request.tile.image.width );
      const std::uint32_t y = request.tile.firstRow
         + static_cast<std::uint32_t>( pixel/request.tile.image.width );
      std::vector<float> values( request.frameCount );
      std::vector<std::uint16_t> frames( request.frameCount );
      std::uint32_t count = 0;
      for ( std::uint32_t frame = 0; frame < request.frameCount; ++frame )
      {
         const std::size_t sampleIndex =
            static_cast<std::size_t>( frame )*pixels + pixel;
         const std::size_t gridOffset =
            static_cast<std::size_t>( frame )*gridSamples;
         const float a = BicubicBSplineSample(
            request.frameMajorScaleGrid.subspan( gridOffset, gridSamples ),
            request.gridWidth, request.gridHeight, sx*x, sy*y );
         const float b = BicubicBSplineSample(
            request.frameMajorZeroOffsetGrid.subspan(
               gridOffset, gridSamples ),
            request.gridWidth, request.gridHeight, sx*x, sy*y );
         const float input = request.frameMajorSamples[sampleIndex];
         const float value = a*input + b;
         if ( !std::isfinite( input ) || input <= request.rangeLow
           || !std::isfinite( value )
           || request.frameWeights[frame] <= 0 )
            continue;
         std::uint32_t position = count;
         while ( position > 0 && values[position - 1] > value )
         {
            values[position] = values[position - 1];
            frames[position] = frames[position - 1];
            --position;
         }
         values[position] = value;
         frames[position] = static_cast<std::uint16_t>( frame );
         ++count;
      }
      for ( std::uint32_t round = 0;
            round < request.rejectionIterations && count >= 5; ++round )
      {
         const LadLine line = FitSortedLadLine(
            values, count, request.fitBisectionIterations );
         const float scale = 2*line.averageDeviation
                           * std::sqrt( 1 + line.b*line.b );
         if ( !(scale > 0) || 1 + scale == 1 )
            break;
         std::uint32_t kept = 0;
         for ( std::uint32_t i = 0; i < count; ++i )
         {
            const float distance =
               (values[i] - (line.a + line.b*i))/scale;
            if ( distance < -request.lowTolerance
              || distance > request.highTolerance )
               continue;
            values[kept] = values[i];
            frames[kept] = frames[i];
            ++kept;
         }
         if ( kept == count )
            break;
         count = kept;
      }
      double weighted = 0, weightSum = 0;
      for ( std::uint32_t i = 0; i < count; ++i )
      {
         const float weight = request.frameWeights[frames[i]];
         weighted += static_cast<double>( weight )*values[i];
         weightSum += weight;
      }
      result.integrated[pixel] = weightSum > 0
         ? request.outputScale*static_cast<float>( weighted/weightSum )
             + request.outputOffset
         : 0.0F;
      result.acceptedSamples[pixel] = static_cast<std::uint16_t>( count );
      result.rejectedSamples[pixel] = static_cast<std::uint16_t>(
         request.frameCount - count );
   }
   return result;
}

} // namespace openastroflow::native
