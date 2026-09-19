#include "openastroflow/PortableKernels.h"
#include "openastroflow/c_api.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using namespace openastroflow::native;

constexpr float Nan = std::numeric_limits<float>::quiet_NaN();

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

bool SameOrBothNan( float left, float right )
{
   if ( std::isnan( left ) || std::isnan( right ) )
      return std::isnan( left ) && std::isnan( right );
   return left == right;
}

std::vector<float> Warp( const std::vector<float>& source,
                         std::uint32_t width,
                         std::uint32_t height,
                         const AffineInverse& inverse,
                         std::uint32_t threads,
                         float domainScale = 1.0F )
{
   WarpLanczos3Request request;
   request.source = source;
   request.sourceWidth = width;
   request.sourceHeight = height;
   request.inverse = inverse;
   request.outputWidth = width;
   request.firstRow = 0;
   request.rowCount = height;
   request.domainScale = domainScale;
   request.threads = threads;
   std::vector<float> destination( request.OutputPixels() );
   WarpLanczos3Clamped( request, destination );
   return destination;
}

void TestWarpIntegerTranslationCopiesPixelsAndMasksTheMargin()
{
   const std::uint32_t width = 17;
   const std::uint32_t height = 13;
   std::vector<float> source( width*height );
   for ( std::uint32_t y = 0; y < height; ++y )
      for ( std::uint32_t x = 0; x < width; ++x )
         source[y*width + x] = 0.25F*static_cast<float>( x ) - 0.5F*static_cast<float>( y );
   source[5*width + 7] = Nan;
   // Output (x, y) reads input (x - 3, y + 2).
   const AffineInverse inverse{ 1.0, 0.0, -3.0, 0.0, 1.0, 2.0 };
   const std::vector<float> warped = Warp( source, width, height, inverse, 3 );
   for ( std::uint32_t y = 0; y < height; ++y )
      for ( std::uint32_t x = 0; x < width; ++x )
      {
         const double inputX = static_cast<double>( x ) - 3.0;
         const double inputY = static_cast<double>( y ) + 2.0;
         const float actual = warped[y*width + x];
         const bool valid = inputX >= 2.0 && inputX <= width - 3.0
                         && inputY >= 2.0 && inputY <= height - 3.0;
         if ( !valid )
         {
            Require( std::isnan( actual ), "margin sample must be NaN" );
            continue;
         }
         const float expected = source[static_cast<std::size_t>( inputY )*width
                                       + static_cast<std::size_t>( inputX )];
         Require( SameOrBothNan( actual, expected ),
                  "integer translation must copy source pixels exactly" );
      }
}

void TestWarpConstantFieldIsPreservedAndThreadInvariant()
{
   const std::uint32_t width = 40;
   const std::uint32_t height = 31;
   std::vector<float> source( width*height, 0.125F );
   const double angle = 0.37;
   const AffineInverse inverse{
      std::cos( angle ), -std::sin( angle ), 1.37,
      std::sin( angle ), std::cos( angle ), -2.11 };
   const std::vector<float> single = Warp( source, width, height, inverse, 1 );
   const std::vector<float> multi = Warp( source, width, height, inverse, 5 );
   std::size_t finite = 0;
   for ( std::size_t index = 0; index < single.size(); ++index )
   {
      Require( SameOrBothNan( single[index], multi[index] ),
               "thread count must not change warp results" );
      if ( std::isfinite( single[index] ) )
      {
         ++finite;
         Require( std::fabs( single[index] - 0.125F ) <= 2.0e-7F,
                  "constant field must be preserved by normalized weights" );
      }
   }
   Require( finite > 0, "rotated warp must produce valid samples" );
}

void TestWarpNanSupportAndDomainClamp()
{
   const std::uint32_t width = 24;
   const std::uint32_t height = 20;
   std::vector<float> source( width*height, 0.2F );
   source[9*width + 11] = Nan;
   // A bright isolated pixel above the declared domain scale and a negative
   // pixel below zero must survive as input extrema while interpolation
   // overshoot is clamped to the union of support and domain.
   source[4*width + 4] = 3.0F;
   source[15*width + 18] = -0.75F;
   const AffineInverse inverse{ 1.0, 0.0, 0.5, 0.0, 1.0, 0.5 };
   const std::vector<float> warped = Warp( source, width, height, inverse, 2 );
   // Samples whose active 6x6 support touches the NaN are NaN; samples
   // three or more pixels away remain finite.
   Require( std::isnan( warped[9*width + 11] ), "NaN support must invalidate" );
   Require( std::isnan( warped[7*width + 9] ), "NaN support must invalidate" );
   Require( std::isfinite( warped[9*width + 15] ),
            "samples outside the NaN support must stay finite" );
   for ( float value : warped )
      if ( std::isfinite( value ) )
         Require( value >= -0.75F && value <= 3.0F,
                  "clamp must bound samples by the union of support and domain" );
   Require( warped[4*width + 4] > 0.2F,
            "a bright source pixel must still contribute above the background" );
}

void TestWarpValidationRejectsBadGeometry()
{
   std::vector<float> source( 6*6, 0.0F );
   WarpLanczos3Request request;
   request.source = source;
   request.sourceWidth = 6;
   request.sourceHeight = 6;
   request.outputWidth = 6;
   request.rowCount = 6;
   request.domainScale = 1.0F;
   std::vector<float> destination( 36 );
   request.sourceWidth = 7;
   RequireThrows<std::invalid_argument>(
      [&]() { WarpLanczos3Clamped( request, destination ); },
      "source sample count mismatch must be rejected" );
   request.sourceWidth = 6;
   request.domainScale = 0.0F;
   RequireThrows<std::invalid_argument>(
      [&]() { WarpLanczos3Clamped( request, destination ); },
      "nonpositive domain scale must be rejected" );
   request.domainScale = 1.0F;
   std::vector<float> small( 35 );
   RequireThrows<std::invalid_argument>(
      [&]() { WarpLanczos3Clamped( request, small ); },
      "undersized destination must be rejected" );
}

float ReferenceMedian( std::vector<float> values )
{
   std::sort( values.begin(), values.end() );
   const std::size_t count = values.size();
   if ( (count & 1U) != 0 )
      return values[count/2];
   const float sum = values[count/2 - 1] + values[count/2];
   return sum/2.0F;
}

void TestMadRejectionMatchesReferenceSemantics()
{
   const std::uint32_t frames = 9;
   const std::uint32_t width = 6;
   const std::uint32_t rows = 4;
   const std::size_t pixels = static_cast<std::size_t>( width )*rows;
   std::vector<float> samples( frames*pixels );
   std::mt19937 generator( 42U );
   std::normal_distribution<float> noise( 100.0F, 2.0F );
   for ( float& value : samples )
      value = noise( generator );
   // Column 0: a single large outlier in frame 8.
   samples[8*pixels + 0] += 1000.0F;
   // Column 1: too few finite samples to reject anything.
   for ( std::uint32_t frame = 0; frame < frames; ++frame )
      samples[frame*pixels + 1] = frame < 2 ? 100.0F + frame : Nan;
   // Column 2: nonfinite values must be unavailable, not rejected.
   samples[3*pixels + 2] = std::numeric_limits<float>::infinity();
   samples[4*pixels + 2] = Nan;
   // Column 3: identical values (zero MAD) with one tiny deviation.
   for ( std::uint32_t frame = 0; frame < frames; ++frame )
      samples[frame*pixels + 3] = 50.0F;
   samples[5*pixels + 3] = 50.0F + 0.01F;
   // Column 4: entirely unavailable.
   for ( std::uint32_t frame = 0; frame < frames; ++frame )
      samples[frame*pixels + 4] = Nan;

   MadRejectionRequest request;
   request.frameMajorSamples = samples;
   request.frameCount = frames;
   request.rowCount = rows;
   request.width = width;
   request.sigmaClip = 4.0F;
   request.minimumRejectionFrames = 3;
   request.groupSigmaFloor = 1.0e-7F;
   request.threads = 3;
   std::vector<std::uint8_t> accepted( samples.size(), 7 );
   std::vector<float> center( pixels, 0.0F );
   MadRejectionMask( request, accepted, center );

   std::vector<std::uint8_t> serial( samples.size(), 7 );
   std::vector<float> serialCenter( pixels, 0.0F );
   request.threads = 1;
   MadRejectionMask( request, serial, serialCenter );
   Require( accepted == serial, "thread count must not change decisions" );
   for ( std::size_t pixel = 0; pixel < pixels; ++pixel )
      Require( SameOrBothNan( center[pixel], serialCenter[pixel] ),
               "thread count must not change centres" );

   for ( std::size_t pixel = 0; pixel < pixels; ++pixel )
   {
      std::vector<float> finite;
      for ( std::uint32_t frame = 0; frame < frames; ++frame )
      {
         const float value = samples[frame*pixels + pixel];
         if ( std::isfinite( value ) )
            finite.push_back( value );
      }
      if ( finite.empty() )
      {
         Require( std::isnan( center[pixel] ), "empty pixel centre must be NaN" );
         for ( std::uint32_t frame = 0; frame < frames; ++frame )
            Require( accepted[frame*pixels + pixel] == 0,
                     "unavailable samples are never accepted" );
         continue;
      }
      const float expectedCenter = ReferenceMedian( finite );
      Require( center[pixel] == expectedCenter,
               "centre must equal the NumPy nanmedian semantics" );
      if ( finite.size() < 3 )
      {
         for ( std::uint32_t frame = 0; frame < frames; ++frame )
            Require( (accepted[frame*pixels + pixel] != 0)
                        == std::isfinite( samples[frame*pixels + pixel] ),
                     "sparse pixels accept every finite sample" );
         continue;
      }
      std::vector<float> deviations;
      for ( float value : finite )
         deviations.push_back( std::fabs( value - expectedCenter ) );
      const float mad = ReferenceMedian( deviations );
      const float robust = 1.4826F*mad;
      const float numerical = std::max(
         1.0e-7F, request.epsilonFloor*std::max( 1.0F, std::fabs( expectedCenter ) ) );
      const float threshold =
         4.0F*std::max( std::max( robust, 1.0e-7F ), numerical );
      for ( std::uint32_t frame = 0; frame < frames; ++frame )
      {
         const float value = samples[frame*pixels + pixel];
         const bool expected = std::isfinite( value )
            && std::fabs( value - expectedCenter ) <= threshold;
         Require( (accepted[frame*pixels + pixel] != 0) == expected,
                  "per-sample decision differs from the reference rule" );
      }
   }
   Require( accepted[8*pixels + 0] == 0, "large outlier must be rejected" );
   Require( accepted[5*pixels + 3] == 0,
            "tiny deviation must be rejected against the numerical floor" );
   Require( accepted[3*pixels + 2] == 0 && accepted[4*pixels + 2] == 0,
            "nonfinite samples are unavailable" );
   Require( accepted[0*pixels + 2] == 1, "finite inliers stay accepted" );
}

void TestMadRejectionScaleModelMatchesReferenceRule()
{
   // One row of 41 pixels, 9 frames of Gaussian noise. Pixel 20 has an
   // artificially narrow stack (per-pixel MAD far below the row's noise)
   // holding one sample at 2.5 true sigma: the v1 per-pixel rule rejects it,
   // the pooled rule keeps it. Pixel 30 is a "star core" whose stack is much
   // wider than the row: its excess variance must be kept per pixel.
   const std::uint32_t frames = 9;
   const std::uint32_t width = 41;
   const std::uint32_t rows = 1;
   const std::size_t pixels = width;
   std::vector<float> samples( frames*pixels );
   std::mt19937 generator( 7U );
   std::normal_distribution<float> noise( 100.0F, 2.0F );
   for ( float& value : samples )
      value = noise( generator );
   for ( std::uint32_t frame = 0; frame < frames; ++frame )
      samples[frame*pixels + 20] = 100.0F + 0.05F*static_cast<float>( frame );
   samples[4*pixels + 20] = 105.0F;
   for ( std::uint32_t frame = 0; frame < frames; ++frame )
      samples[frame*pixels + 30] = 500.0F + 40.0F*static_cast<float>( frame );
   samples[0*pixels + 30] = 500.0F + 40.0F*frames + 30.0F;
   // Pixel 5 has too few finite samples: it must not pool into neighbours.
   for ( std::uint32_t frame = 2; frame < frames; ++frame )
      samples[frame*pixels + 5] = Nan;

   MadRejectionRequest request;
   request.frameMajorSamples = samples;
   request.frameCount = frames;
   request.rowCount = rows;
   request.width = width;
   request.sigmaClip = 4.0F;
   request.minimumRejectionFrames = 3;
   request.groupSigmaFloor = 1.0e-7F;
   request.poolHalfWidth = 12;
   std::vector<float> scales( frames, 1.0F );
   scales[4] = 1.25F;
   scales[8] = 0.8F;
   request.frameScales = scales;
   request.threads = 3;
   std::vector<std::uint8_t> accepted( samples.size(), 7 );
   std::vector<float> center( pixels, 0.0F );
   MadRejectionMask( request, accepted, center );

   std::vector<std::uint8_t> serial( samples.size(), 7 );
   std::vector<float> serialCenter( pixels, 0.0F );
   request.threads = 1;
   MadRejectionMask( request, serial, serialCenter );
   Require( accepted == serial, "scale model: thread count must not change decisions" );

   // Reference rule, evaluated independently.
   std::vector<float> madMap( pixels, Nan );
   std::vector<float> centers( pixels, Nan );
   for ( std::size_t pixel = 0; pixel < pixels; ++pixel )
   {
      std::vector<float> finite;
      for ( std::uint32_t frame = 0; frame < frames; ++frame )
         if ( std::isfinite( samples[frame*pixels + pixel] ) )
            finite.push_back( samples[frame*pixels + pixel] );
      if ( finite.empty() )
         continue;
      centers[pixel] = ReferenceMedian( finite );
      Require( center[pixel] == centers[pixel], "scale model keeps the nanmedian centre" );
      if ( finite.size() < 3 )
         continue;
      std::vector<float> deviations;
      for ( float value : finite )
         deviations.push_back( std::fabs( value - centers[pixel] ) );
      madMap[pixel] = ReferenceMedian( deviations );
   }
   for ( std::size_t pixel = 0; pixel < pixels; ++pixel )
   {
      if ( std::isnan( madMap[pixel] ) )
      {
         for ( std::uint32_t frame = 0; frame < frames; ++frame )
            Require( (accepted[frame*pixels + pixel] != 0)
                        == std::isfinite( samples[frame*pixels + pixel] ),
                     "sparse pixels accept every finite sample" );
         continue;
      }
      std::vector<float> window;
      const std::size_t first = pixel > 12 ? pixel - 12 : 0;
      const std::size_t last = std::min<std::size_t>( width - 1, pixel + 12 );
      for ( std::size_t column = first; column <= last; ++column )
         if ( !std::isnan( madMap[column] ) )
            window.push_back( madMap[column] );
      const float pooled = ReferenceMedian( window );
      const float robust = 1.4826F*madMap[pixel];
      const float sigmaPool = 1.4826F*pooled;
      const float excess = std::max( robust*robust - sigmaPool*sigmaPool, 0.0F );
      const float numerical = std::max(
         1.0e-7F, request.epsilonFloor*std::max( 1.0F, std::fabs( centers[pixel] ) ) );
      for ( std::uint32_t frame = 0; frame < frames; ++frame )
      {
         const float value = samples[frame*pixels + pixel];
         const float scaled = scales[frame]*sigmaPool;
         const float sigmaFrame = std::sqrt( scaled*scaled + excess );
         const float threshold =
            4.0F*std::max( std::max( sigmaFrame, 1.0e-7F ), numerical );
         const bool expected = std::isfinite( value )
            && std::fabs( value - centers[pixel] ) <= threshold;
         Require( (accepted[frame*pixels + pixel] != 0) == expected,
                  "scale model decision differs from the reference rule" );
      }
   }
   Require( accepted[4*pixels + 20] == 1,
            "a 2.5 sigma sample survives when the row noise is pooled" );
   Require( accepted[0*pixels + 30] == 1,
            "a wide star-core stack keeps its own per-pixel scale" );

   // Legacy request (no scales, no pooling) through the same entry point
   // must reproduce the v1 decisions bit for bit.
   MadRejectionRequest legacy = request;
   legacy.frameScales = {};
   legacy.poolHalfWidth = 0;
   std::vector<std::uint8_t> legacyAccepted( samples.size(), 7 );
   std::vector<float> legacyCenter( pixels, 0.0F );
   MadRejectionMask( legacy, legacyAccepted, legacyCenter );
   MadRejectionRequest unitScales = request;
   std::vector<float> ones( frames, 1.0F );
   unitScales.frameScales = ones;
   unitScales.poolHalfWidth = 0;
   std::vector<std::uint8_t> unitAccepted( samples.size(), 7 );
   std::vector<float> unitCenter( pixels, 0.0F );
   MadRejectionMask( unitScales, unitAccepted, unitCenter );
   Require( legacyAccepted == unitAccepted,
            "unit scales without pooling equal the legacy decisions" );
   Require( legacyAccepted[4*pixels + 20] == 0,
            "the legacy per-pixel rule rejects the narrow-stack sample" );

   MadRejectionRequest badScales = request;
   std::vector<float> wrong( frames - 1, 1.0F );
   badScales.frameScales = wrong;
   RequireThrows<std::invalid_argument>(
      [&]() { MadRejectionMask( badScales, accepted, center ); },
      "frame scale count must match the frame count" );
}

void TestMaskedMeanAccumulatesInFrameOrder()
{
   const std::uint32_t frames = 5;
   const std::uint32_t width = 3;
   const std::uint32_t rows = 1;
   const std::size_t pixels = width*rows;
   const std::array<float, frames*pixels> samples{
      1.0F, 10.0F, Nan,
      2.0F, 20.0F, 5.0F,
      3.0F, 30.0F, 6.0F,
      4.0F, 1000.0F, Nan,
      5.0F, 50.0F, 7.0F };
   const std::array<std::uint8_t, frames*pixels> accepted{
      1, 1, 0,
      1, 1, 0,
      1, 1, 0,
      1, 0, 0,
      1, 1, 0 };
   const std::array<double, frames> weights{ 0.1, 0.2, 0.3, 0.25, 0.15 };
   MaskedMeanRequest request;
   request.frameMajorSamples = samples;
   request.frameMajorAccepted = accepted;
   request.frameWeights = weights;
   request.frameCount = frames;
   request.rowCount = rows;
   request.width = width;
   request.threads = 2;
   std::vector<float> integrated( pixels );
   std::vector<std::uint16_t> acceptedCount( pixels );
   std::vector<std::uint16_t> rejectedCount( pixels );
   MaskedWeightedMean(
      request, MaskedMeanOutput{ integrated, acceptedCount, rejectedCount } );

   double numerator = 0.0;
   double denominator = 0.0;
   for ( std::uint32_t frame = 0; frame < frames; ++frame )
   {
      numerator = numerator + static_cast<double>( samples[frame*pixels] )*weights[frame];
      denominator = denominator + weights[frame];
   }
   Require( integrated[0] == static_cast<float>( numerator/denominator ),
            "weighted mean must use Float64 frame-order accumulation" );
   Require( acceptedCount[0] == 5 && rejectedCount[0] == 0,
            "fully accepted pixel counts" );
   Require( acceptedCount[1] == 4 && rejectedCount[1] == 1,
            "finite rejected sample must be counted" );
   Require( std::isnan( integrated[2] ) && acceptedCount[2] == 0
            && rejectedCount[2] == 3,
            "pixel without accepted samples is NaN; NaN inputs are not rejections" );
}

void TestTileOffsetsMatchReferenceStatistics()
{
   // Two tiles: a clean linear relation with an outlier, and one that is too
   // sparse after nonfinite removal.
   std::vector<double> target;
   std::vector<double> reference;
   std::mt19937 generator( 7U );
   std::normal_distribution<double> noise( 0.0, 2.0 );
   const double scale = 1.25;
   for ( int i = 0; i < 1000; ++i )
   {
      const double t = 100.0 + 0.1*i;
      target.push_back( t );
      reference.push_back( scale*t + 40.0 + noise( generator ) );
   }
   reference[500] += 5000.0;  // clipped by the residual sigma gate
   target.push_back( std::numeric_limits<double>::quiet_NaN() );
   reference.push_back( 1.0 );
   const std::uint64_t firstTileEnd = target.size();
   for ( int i = 0; i < 20; ++i )
   {
      target.push_back( 10.0 + i );
      reference.push_back( 20.0 + i );
   }
   const std::vector<std::uint64_t> boundaries{
      0, firstTileEnd, static_cast<std::uint64_t>( target.size() ) };
   TileOffsetRequest request;
   request.target = target;
   request.reference = reference;
   request.boundaries = boundaries;
   request.tileCount = 2;
   request.scale = scale;
   request.lowerQuantile = 0.05;
   request.upperQuantile = 0.95;
   request.minimumSamples = 64;
   request.residualClipSigma = 3.0;
   request.threads = 2;
   std::vector<double> offset( 2 ), mad( 2 );
   std::vector<std::uint32_t> count( 2 );
   std::vector<std::uint8_t> valid( 2 );
   TileOffsets( request, TileOffsetOutput{ offset, count, mad, valid } );
   Require( valid[0] == 1 && valid[1] == 0, "tile validity must follow the sample minimum" );
   Require( std::fabs( offset[0] - 40.0 ) < 0.5, "offset must recover the additive term" );
   Require( mad[0] > 0.5 && mad[0] < 5.0, "residual MAD must reflect the noise" );
   Require( count[0] >= 800 && count[0] < 1000, "outlier and quantile tails are excluded" );
   Require( std::isnan( offset[1] ) && std::isnan( mad[1] ), "invalid tiles report NaN" );
   std::vector<double> serialOffset( 2 ), serialMad( 2 );
   std::vector<std::uint32_t> serialCount( 2 );
   std::vector<std::uint8_t> serialValid( 2 );
   request.threads = 1;
   TileOffsets( request, TileOffsetOutput{ serialOffset, serialCount, serialMad, serialValid } );
   Require( serialOffset[0] == offset[0] && serialMad[0] == mad[0]
            && serialCount == count && serialValid == valid,
            "thread count must not change tile statistics" );
}

std::vector<RadonPeak> RadonPeaksOf( const std::vector<float>& image,
                                     const std::vector<std::uint8_t>& weight,
                                     std::uint32_t width,
                                     std::uint32_t height,
                                     std::uint32_t size,
                                     std::uint32_t threads )
{
   RadonPeakRequest request;
   request.image = image;
   request.weight = weight;
   request.width = width;
   request.height = height;
   request.size = size;
   request.minimumRows = 16;
   request.threads = threads;
   std::vector<RadonPeak> peaks;
   RadonLinePeaks( request, peaks );
   return peaks;
}

void TestRadonPeaksFindTheDyadicLineAtEveryLevel()
{
   // A vertical line of value 2 at column 10 of a 40x64 frame on a 64-row
   // canvas.  Every block whose rows reach into the frame with at least
   // 60% coverage sums 2 per row, so the line is the single peak of the
   // block at shift 0; blocks 2 and 3 of level 16 and block 1 of level 32
   // cover 8 frame rows only and are invalid.  Most valid lines are zero,
   // so the median absolute deviation vanishes and z stays unscaled.
   const std::uint32_t width = 64;
   const std::uint32_t height = 40;
   const std::uint32_t size = 64;
   std::vector<float> image( height*width, 0.0F );
   std::vector<std::uint8_t> weight( height*width, 1 );
   for ( std::uint32_t row = 0; row < height; ++row )
      image[row*width + 10] = 2.0F;
   const std::vector<RadonPeak> peaks = RadonPeaksOf( image, weight, width, height, size, 1 );
   Require( peaks.size() == 4U, "radon peaks: expected one peak per covered block" );
   const std::uint32_t expectedLevels[4] = { 16, 16, 32, 64 };
   const std::uint32_t expectedBlocks[4] = { 0, 1, 0, 0 };
   const float expectedRows[4] = { 16.0F, 16.0F, 32.0F, 40.0F };
   for ( std::size_t i = 0; i < peaks.size(); ++i )
   {
      Require( peaks[i].level == expectedLevels[i], "radon peaks: level order" );
      Require( peaks[i].block == expectedBlocks[i], "radon peaks: block order" );
      Require( peaks[i].shiftIndex == peaks[i].level - 1, "radon peaks: vertical line has shift 0" );
      Require( peaks[i].column == size + 10U, "radon peaks: padded column of the line" );
      const float sum = 2.0F*expectedRows[i];
      const float expectedZ = sum/std::sqrt( expectedRows[i] );
      Require( peaks[i].z == expectedZ, "radon peaks: z is Float32 sum/sqrt(count)" );
   }
   const std::vector<RadonPeak> threaded = RadonPeaksOf( image, weight, width, height, size, 4 );
   Require( threaded.size() == peaks.size(), "radon peaks: thread-invariant count" );
   for ( std::size_t i = 0; i < peaks.size(); ++i )
      Require( threaded[i].level == peaks[i].level && threaded[i].block == peaks[i].block
            && threaded[i].shiftIndex == peaks[i].shiftIndex
            && threaded[i].column == peaks[i].column && threaded[i].z == peaks[i].z,
               "radon peaks: thread-invariant peaks" );

   // The dyadic line with shift +1 over the 64 rows (column 10 for the top
   // half, column 11 for the bottom half) is the shift-1 peak of the top
   // level: the vertical lines through either half sum less.
   std::fill( image.begin(), image.end(), 0.0F );
   for ( std::uint32_t row = 0; row < height; ++row )
      image[row*width + (row < 32 ? 10U : 11U)] = 2.0F;
   const std::vector<RadonPeak> oblique = RadonPeaksOf( image, weight, width, height, size, 2 );
   const RadonPeak* strongest = nullptr;
   for ( const RadonPeak& peak : oblique )
      if ( peak.level == 64 && (strongest == nullptr || peak.z > strongest->z) )
         strongest = &peak;
   Require( strongest != nullptr, "radon peaks: oblique top-level peak found" );
   Require( strongest->shiftIndex == 63U + 1U && strongest->column == size + 10U,
            "radon peaks: oblique line is the shift-1 top-level peak" );
   Require( strongest->z == 80.0F/std::sqrt( 40.0F ), "radon peaks: oblique z" );

   // Validation: non-power-of-two canvas, canvas smaller than the frame.
   RadonPeakRequest bad;
   bad.image = image;
   bad.weight = weight;
   bad.width = width;
   bad.height = height;
   bad.size = 48;
   bad.minimumRows = 16;
   std::vector<RadonPeak> sink;
   RequireThrows<std::invalid_argument>( [&]() { RadonLinePeaks( bad, sink ); },
                                         "radon peaks: canvas must be a power of two" );
   bad.size = 32;
   RequireThrows<std::invalid_argument>( [&]() { RadonLinePeaks( bad, sink ); },
                                         "radon peaks: canvas must hold the frame" );
}

void TestCAbiRoundTrip()
{
   const std::uint32_t width = 12;
   const std::uint32_t height = 10;
   std::vector<float> source( width*height );
   for ( std::size_t index = 0; index < source.size(); ++index )
      source[index] = static_cast<float>( index % 7 )*0.1F;
   OafNativeWarpLanczos3RequestV1 warp{};
   warp.struct_size = sizeof( warp );
   warp.source_width = width;
   warp.source_height = height;
   warp.output_width = width;
   warp.first_row = 2;
   warp.row_count = 3;
   warp.threads = 2;
   warp.source_samples = source.data();
   warp.source_sample_count = source.size();
   const double inverse[6] = { 1.0, 0.0, 0.25, 0.0, 1.0, -0.5 };
   std::memcpy( warp.inverse, inverse, sizeof( inverse ) );
   warp.domain_scale = 1.0F;
   std::vector<float> band( width*3 );
   std::array<char, 256> error{};
   Require( oaf_native_cpu_warp_lanczos3_v1(
               &warp, band.data(), band.size(), error.data(), error.size() )
               == OAF_NATIVE_OK,
            error.data() );
   AffineInverse nativeInverse{ 1.0, 0.0, 0.25, 0.0, 1.0, -0.5 };
   const std::vector<float> full = Warp( source, width, height, nativeInverse, 1 );
   for ( std::size_t index = 0; index < band.size(); ++index )
      Require( SameOrBothNan( band[index], full[2*width + index] ),
               "C ABI band must match the direct kernel band" );
   Require( oaf_native_cpu_warp_lanczos3_v1(
               &warp, band.data(), band.size() - 1, error.data(), error.size() )
               == OAF_NATIVE_BUFFER_TOO_SMALL,
            "C ABI must reject an undersized destination" );

   const std::uint32_t frames = 4;
   std::vector<float> samples( frames*width*height, 3.0F );
   samples[2*width*height + 5] = 900.0F;
   OafNativeMadRejectionRequestV1 mad{};
   mad.struct_size = sizeof( mad );
   mad.frame_count = frames;
   mad.row_count = height;
   mad.width = width;
   mad.minimum_rejection_frames = 3;
   mad.threads = 3;
   mad.frame_major_samples = samples.data();
   mad.sample_count = samples.size();
   mad.sigma_clip = 4.0F;
   mad.group_sigma_floor = 1.0e-7F;
   mad.absolute_floor = 1.0e-7F;
   mad.epsilon_floor = 16.0F*1.1920928955078125e-07F;
   std::vector<std::uint8_t> accepted( samples.size(), 9 );
   std::vector<float> center( width*height );
   Require( oaf_native_cpu_mad_rejection_v1(
               &mad, accepted.data(), accepted.size(), center.data(),
               center.size(), error.data(), error.size() ) == OAF_NATIVE_OK,
            error.data() );
   Require( accepted[2*width*height + 5] == 0 && accepted[5] == 1,
            "C ABI MAD rejection must flag the outlier only" );
   Require( center[5] == 3.0F, "C ABI centre must be the median" );

   OafNativeMadRejectionRequestV2 madV2{};
   madV2.struct_size = sizeof( madV2 );
   madV2.frame_count = frames;
   madV2.row_count = height;
   madV2.width = width;
   madV2.minimum_rejection_frames = 3;
   madV2.threads = 3;
   madV2.frame_major_samples = samples.data();
   madV2.sample_count = samples.size();
   madV2.sigma_clip = 4.0F;
   madV2.group_sigma_floor = 1.0e-7F;
   madV2.absolute_floor = 1.0e-7F;
   madV2.epsilon_floor = 16.0F*1.1920928955078125e-07F;
   std::vector<float> scales( frames, 1.0F );
   madV2.frame_scales = scales.data();
   madV2.frame_scale_count = scales.size();
   madV2.pool_half_width = 12;
   std::vector<std::uint8_t> acceptedV2( samples.size(), 9 );
   std::vector<float> centerV2( width*height );
   Require( oaf_native_cpu_mad_rejection_v2(
               &madV2, acceptedV2.data(), acceptedV2.size(), centerV2.data(),
               centerV2.size(), error.data(), error.size() ) == OAF_NATIVE_OK,
            error.data() );
   Require( acceptedV2[2*width*height + 5] == 0 && acceptedV2[5] == 1,
            "C ABI MAD v2 rejection must flag the outlier only" );
   Require( centerV2 == center, "C ABI MAD v2 centres equal v1" );
   madV2.frame_scales = nullptr;
   Require( oaf_native_cpu_mad_rejection_v2(
               &madV2, acceptedV2.data(), acceptedV2.size(), centerV2.data(),
               centerV2.size(), error.data(), error.size() )
               == OAF_NATIVE_INVALID_ARGUMENT,
            "C ABI MAD v2 must reject a scale count without a pointer" );
   madV2.frame_scale_count = 0;
   madV2.pool_half_width = 0;
   Require( oaf_native_cpu_mad_rejection_v2(
               &madV2, acceptedV2.data(), acceptedV2.size(), centerV2.data(),
               centerV2.size(), error.data(), error.size() ) == OAF_NATIVE_OK,
            error.data() );
   Require( acceptedV2 == accepted, "C ABI MAD v2 without the scale model equals v1" );

   std::vector<double> weights( frames, 0.25 );
   OafNativeMaskedMeanRequestV1 mean{};
   mean.struct_size = sizeof( mean );
   mean.frame_count = frames;
   mean.row_count = height;
   mean.width = width;
   mean.threads = 2;
   mean.frame_major_samples = samples.data();
   mean.sample_count = samples.size();
   mean.frame_major_accepted = accepted.data();
   mean.accepted_count = accepted.size();
   mean.frame_weights = weights.data();
   mean.weight_count = weights.size();
   std::vector<float> integrated( width*height );
   std::vector<std::uint16_t> acceptedCount( width*height );
   std::vector<std::uint16_t> rejectedCount( width*height );
   OafNativeMaskedMeanOutputV1 output{};
   output.struct_size = sizeof( output );
   output.integrated = integrated.data();
   output.accepted_samples = acceptedCount.data();
   output.rejected_samples = rejectedCount.data();
   output.pixel_capacity = integrated.size();
   Require( oaf_native_cpu_masked_mean_v1(
               &mean, &output, error.data(), error.size() ) == OAF_NATIVE_OK,
            error.data() );
   Require( integrated[5] == 3.0F && acceptedCount[5] == 3 && rejectedCount[5] == 1,
            "C ABI masked mean must exclude the rejected outlier" );
   Require( oaf_native_default_kernel_threads_v1() >= 1,
            "default kernel thread count must be positive" );

   OafNativeCpuFeaturesV1 features{};
   features.struct_size = sizeof( features );
   Require( oaf_native_cpu_features_v1( &features, error.data(), error.size() )
               == OAF_NATIVE_OK,
            error.data() );
   Require( features.architecture == OAF_NATIVE_CPU_ARCHITECTURE_X86_64
         || features.architecture == OAF_NATIVE_CPU_ARCHITECTURE_ARM64,
            "cpu features must name the compiled architecture" );
   const std::string names( features.features );
   Require( features.architecture != OAF_NATIVE_CPU_ARCHITECTURE_X86_64
         || names.find( "sse4.2" ) != std::string::npos,
            "every x86-64 host running this test supports SSE4.2" );
   Require( features.architecture != OAF_NATIVE_CPU_ARCHITECTURE_ARM64
         || names == "neon",
            "arm64 reports neon" );
   Require( names.find( ' ' ) == std::string::npos && names.find( ",," ) == std::string::npos,
            "feature names are a comma-separated lowercase list" );
   Require( std::strlen( features.brand ) < sizeof( features.brand ),
            "brand string is NUL-terminated" );
   OafNativeCpuFeaturesV1 wrongSize{};
   wrongSize.struct_size = 1;
   Require( oaf_native_cpu_features_v1( &wrongSize, error.data(), error.size() )
               == OAF_NATIVE_INVALID_ARGUMENT,
            "cpu features must reject an unexpected struct size" );
}

// Dynamic chunking: results of every kernel must not depend on how the
// range is split among threads. Row/pixel counts that are not multiples
// of the chunk grains exercise the last, partial chunk on several threads.
void TestDynamicChunkingIsThreadAndGrainInvariant()
{
   const std::uint32_t width = 131;
   const std::uint32_t height = 37;
   std::vector<float> source( width*height );
   std::mt19937 generator( 77 );
   std::uniform_real_distribution<float> values( 0.0F, 1.0F );
   for ( float& value : source )
      value = values( generator );
   WarpLanczos3Request warp;
   warp.source = source;
   warp.sourceWidth = width;
   warp.sourceHeight = height;
   warp.inverse.m00 = 0.999;
   warp.inverse.m01 = 0.013;
   warp.inverse.m02 = 0.37;
   warp.inverse.m10 = -0.011;
   warp.inverse.m11 = 1.001;
   warp.inverse.m12 = -0.21;
   warp.outputWidth = width;
   warp.rowCount = height;
   warp.domainScale = 1.0F;
   std::vector<float> serial( width*height );
   warp.threads = 1;
   WarpLanczos3Clamped( warp, serial );
   for ( std::uint32_t threads : { 2U, 3U, 7U, 64U } )
   {
      std::vector<float> parallel( width*height );
      warp.threads = threads;
      WarpLanczos3Clamped( warp, parallel );
      for ( std::size_t i = 0; i < serial.size(); ++i )
         Require( SameOrBothNan( serial[i], parallel[i] ),
                  "warp output must not depend on the thread count" );
   }

   const std::uint32_t frames = 9;
   const std::uint32_t pixels = 4096*2 + 517;
   std::vector<float> stack( static_cast<std::size_t>( frames )*pixels );
   std::normal_distribution<float> noise( 100.0F, 3.0F );
   for ( float& value : stack )
      value = noise( generator );
   stack[3*pixels + 4100] = 1.0e6F;
   stack[5*pixels + 8000] = Nan;
   MadRejectionRequest mad;
   mad.frameMajorSamples = stack;
   mad.frameCount = frames;
   mad.rowCount = 1;
   mad.width = pixels;
   std::vector<std::uint8_t> serialAccepted( stack.size() );
   std::vector<float> serialCenter( pixels );
   mad.threads = 1;
   MadRejectionMask( mad, serialAccepted, serialCenter );
   std::vector<double> weights( frames, 0.5 );
   MaskedMeanRequest mean;
   mean.frameMajorSamples = stack;
   mean.frameMajorAccepted = serialAccepted;
   mean.frameWeights = weights;
   mean.frameCount = frames;
   mean.rowCount = 1;
   mean.width = pixels;
   std::vector<float> serialIntegrated( pixels );
   std::vector<std::uint16_t> serialAcceptedCount( pixels );
   std::vector<std::uint16_t> serialRejectedCount( pixels );
   mean.threads = 1;
   MaskedWeightedMean( mean, { serialIntegrated, serialAcceptedCount, serialRejectedCount } );
   for ( std::uint32_t threads : { 2U, 5U, 64U } )
   {
      std::vector<std::uint8_t> accepted( stack.size() );
      std::vector<float> center( pixels );
      mad.threads = threads;
      MadRejectionMask( mad, accepted, center );
      Require( accepted == serialAccepted, "chunked rejection decisions differ" );
      for ( std::size_t i = 0; i < pixels; ++i )
         Require( SameOrBothNan( center[i], serialCenter[i] ), "chunked centres differ" );
      std::vector<float> integrated( pixels );
      std::vector<std::uint16_t> acceptedCount( pixels );
      std::vector<std::uint16_t> rejectedCount( pixels );
      mean.threads = threads;
      MaskedWeightedMean( mean, { integrated, acceptedCount, rejectedCount } );
      for ( std::size_t i = 0; i < pixels; ++i )
         Require( SameOrBothNan( integrated[i], serialIntegrated[i] )
               && acceptedCount[i] == serialAcceptedCount[i]
               && rejectedCount[i] == serialRejectedCount[i],
                  "chunked weighted mean differs" );
   }
}

} // namespace

int main()
{
   try
   {
      TestWarpIntegerTranslationCopiesPixelsAndMasksTheMargin();
      TestWarpConstantFieldIsPreservedAndThreadInvariant();
      TestWarpNanSupportAndDomainClamp();
      TestWarpValidationRejectsBadGeometry();
      TestMadRejectionMatchesReferenceSemantics();
      TestMadRejectionScaleModelMatchesReferenceRule();
      TestMaskedMeanAccumulatesInFrameOrder();
      TestTileOffsetsMatchReferenceStatistics();
      TestRadonPeaksFindTheDyadicLineAtEveryLevel();
      TestCAbiRoundTrip();
      TestDynamicChunkingIsThreadAndGrainInvariant();
      std::cout << "OpenAstroFlowPortableKernelTests passed\n";
      return 0;
   }
   catch ( const std::exception& error )
   {
      std::cerr << "OpenAstroFlowPortableKernelTests failed: " << error.what() << '\n';
      return 1;
   }
}
