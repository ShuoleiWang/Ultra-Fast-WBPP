#include "openastroflow/PortableKernels.h"
#include "openastroflow/c_api.h"
#include "Lanczos3Table.h"

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

void TestMaskedMeanSampleWeightsScaleContributions()
{
   constexpr std::uint32_t frames = 2;
   constexpr std::uint32_t width = 2;
   constexpr std::uint32_t rows = 1;
   constexpr std::size_t pixels = width*rows;
   const std::array<float, frames*pixels> samples{ 1.0F, 3.0F, 5.0F, 7.0F };
   const std::array<std::uint8_t, frames*pixels> accepted{ 1, 1, 1, 1 };
   const std::array<double, frames> weights{ 0.5, 0.5 };
   const std::array<float, frames*pixels> sampleWeights{ 1.0F, 0.0F, 1.0F, 1.0F };
   MaskedMeanRequest request;
   request.frameMajorSamples = samples;
   request.frameMajorAccepted = accepted;
   request.frameWeights = weights;
   request.frameCount = frames;
   request.rowCount = rows;
   request.width = width;
   request.threads = 1;
   std::vector<float> plain( pixels );
   std::vector<std::uint16_t> acceptedCount( pixels );
   std::vector<std::uint16_t> rejectedCount( pixels );
   MaskedWeightedMean( request, MaskedMeanOutput{ plain, acceptedCount, rejectedCount } );
   request.frameMajorSampleWeights = sampleWeights;
   std::vector<float> weighted( pixels );
   MaskedWeightedMean( request, MaskedMeanOutput{ weighted, acceptedCount, rejectedCount } );
   Require( plain[0] == 3.0F && plain[1] == 5.0F, "unweighted means" );
   Require( weighted[0] == 3.0F,
            "a sample weight of one leaves the contribution unchanged" );
   Require( weighted[1] == 7.0F,
            "a zero sample weight removes the sample from the mean" );
   Require( acceptedCount[1] == 2,
            "sample weights do not change the accepted-sample count" );
   std::array<float, frames*pixels> zero{};
   request.frameMajorSampleWeights = zero;
   MaskedWeightedMean( request, MaskedMeanOutput{ weighted, acceptedCount, rejectedCount } );
   Require( std::isnan( weighted[0] ) && std::isnan( weighted[1] ),
            "pixels without positive effective weight are NaN" );
   std::array<float, 3> wrong{ 1.0F, 1.0F, 1.0F };
   request.frameMajorSampleWeights = wrong;
   bool rejected = false;
   try
   {
      MaskedWeightedMean( request, MaskedMeanOutput{ weighted, acceptedCount, rejectedCount } );
   }
   catch ( const std::invalid_argument& )
   {
      rejected = true;
   }
   Require( rejected, "sample weight geometry mismatch must be rejected" );
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

struct DrizzleOutput
{
   std::vector<double> sum;
   std::vector<double> weight;
   std::vector<std::uint8_t> touched;
};

DrizzleOutput DrizzleWhole( DrizzleRequest request, std::uint32_t outputWidth,
                            std::uint32_t outputHeight, std::uint32_t threads,
                            std::uint32_t bandRows = 0 )
{
   DrizzleOutput output;
   output.sum.assign( static_cast<std::size_t>( outputWidth )*outputHeight, 0.0 );
   output.weight.assign( output.sum.size(), 0.0 );
   output.touched.assign( output.sum.size(), 0 );
   request.outputWidth = outputWidth;
   request.threads = threads;
   const std::uint32_t rows = bandRows == 0 ? outputHeight : bandRows;
   for ( std::uint32_t row0 = 0; row0 < outputHeight; row0 += rows )
   {
      const std::uint32_t count = std::min( rows, outputHeight - row0 );
      const std::size_t offset = static_cast<std::size_t>( row0 )*outputWidth;
      const std::size_t length = static_cast<std::size_t>( count )*outputWidth;
      request.outputRow0 = row0;
      request.outputRows = count;
      request.outputSum = std::span<double>( output.sum.data() + offset, length );
      request.outputWeight = std::span<double>( output.weight.data() + offset, length );
      request.outputTouched = std::span<std::uint8_t>( output.touched.data() + offset, length );
      DrizzleBand( request );
   }
   return output;
}

void TestDrizzleBandDropsExactAreasAndIsBandAndThreadInvariant()
{
   const std::uint32_t width = 23;
   const std::uint32_t height = 17;
   std::vector<float> source( width*height );
   std::mt19937 generator( 11 );
   std::uniform_real_distribution<float> values( 10.0F, 20.0F );
   for ( float& value : source )
      value = values( generator );
   source[5*width + 7] = Nan;

   // Identity map, unit drops at 1x: every output pixel is a copy of its
   // input pixel with weight exactly 1, and the NaN sample leaves a hole.
   DrizzleRequest request;
   request.source = source;
   request.sourceWidth = width;
   request.sourceRows = height;
   request.scale = 1;
   request.pixfrac = 1.0;
   request.kernel = DrizzleKernel::Square;
   request.frameWeight = 1.0F;
   const DrizzleOutput identity = DrizzleWhole( request, width, height, 1 );
   for ( std::uint32_t y = 0; y < height; ++y )
      for ( std::uint32_t x = 0; x < width; ++x )
      {
         const std::size_t index = static_cast<std::size_t>( y )*width + x;
         if ( std::isnan( source[index] ) )
         {
            Require( identity.weight[index] == 0.0 && identity.touched[index] == 0,
                     "drizzle: a NaN sample carries no weight" );
            continue;
         }
         Require( std::fabs( identity.weight[index] - 1.0 ) < 1.0e-12
               && std::fabs( identity.sum[index] - source[index] ) < 1.0e-9
               && identity.touched[index] == 1,
                  "drizzle: identity 1x square drop copies the pixel" );
      }

   // 2x with a rotated, translated map: the drop areas are exact, so the
   // total dropped weight equals the number of finite pixels that land
   // fully inside the output (interior pixels), scaled by pixfrac^2.
   const double angle = 0.31;
   const double c = std::cos( angle ), s = std::sin( angle );
   const double scale = 2.0;
   const double forward[9] = { scale*c, -scale*s, scale*(8.4), scale*s, scale*c, scale*(2.6), 0.0, 0.0, 1.0 };
   std::copy( forward, forward + 9, request.forward );
   request.scale = 2;
   request.pixfrac = 0.7;
   const std::uint32_t outputWidth = 2*(width + 12);
   const std::uint32_t outputHeight = 2*(height + 14);
   const DrizzleOutput single = DrizzleWhole( request, outputWidth, outputHeight, 1 );
   double totalWeight = 0.0, totalSum = 0.0, expectedWeight = 0.0, expectedSum = 0.0;
   for ( std::size_t i = 0; i < single.weight.size(); ++i )
   {
      totalWeight += single.weight[i];
      totalSum += single.sum[i];
   }
   for ( std::uint32_t y = 0; y < height; ++y )
      for ( std::uint32_t x = 0; x < width; ++x )
      {
         const float value = source[y*width + x];
         if ( !std::isfinite( value ) )
            continue;
         const double u = forward[0]*x + forward[1]*y + forward[2];
         const double v = forward[3]*x + forward[4]*y + forward[5];
         Require( u > 2.0 && v > 2.0 && u < outputWidth - 3.0 && v < outputHeight - 3.0,
                  "drizzle test geometry keeps every drop inside the output" );
         const double area = (0.7*scale)*(0.7*scale);
         expectedWeight += area;
         expectedSum += area*value;
      }
   Require( std::fabs( totalWeight - expectedWeight ) < 1.0e-6*expectedWeight,
            "drizzle: exact square drops conserve the dropped weight" );
   Require( std::fabs( totalSum - expectedSum ) < 1.0e-6*expectedSum,
            "drizzle: exact square drops conserve the dropped flux" );

   for ( std::uint32_t threads : { 2U, 3U, 5U, 64U } )
      for ( std::uint32_t bandRows : { 0U, 7U, 16U } )
      {
         const DrizzleOutput parallel = DrizzleWhole( request, outputWidth, outputHeight, threads, bandRows );
         Require( parallel.sum == single.sum && parallel.weight == single.weight
               && parallel.touched == single.touched,
                  "drizzle: output must not depend on the thread count or the band split" );
      }
   for ( DrizzleKernel kernel : { DrizzleKernel::Circular, DrizzleKernel::Gaussian, DrizzleKernel::Point } )
   {
      request.kernel = kernel;
      const DrizzleOutput serial = DrizzleWhole( request, outputWidth, outputHeight, 1 );
      const DrizzleOutput threaded = DrizzleWhole( request, outputWidth, outputHeight, 4, 5 );
      Require( serial.sum == threaded.sum && serial.weight == threaded.weight,
               "drizzle: every kernel is thread and band invariant" );
      double kernelWeight = 0.0;
      for ( double w : serial.weight )
         kernelWeight += w;
      Require( kernelWeight > 0.0, "drizzle: every kernel drops weight" );
      if ( kernel == DrizzleKernel::Point )
         Require( std::fabs( kernelWeight - (width*height - 1.0) ) < 1.0e-9,
                  "drizzle: point drops carry unit weight per finite pixel" );
      if ( kernel == DrizzleKernel::Gaussian )
         Require( std::fabs( kernelWeight - expectedWeight ) < 1.0e-6*expectedWeight,
                  "drizzle: Gaussian drops are normalized to the square drop's area" );
      if ( kernel == DrizzleKernel::Circular )
         Require( std::fabs( kernelWeight - expectedWeight*3.141592653589793/4.0 ) < 1.0e-6*expectedWeight,
                  "drizzle: circular drops carry the exact disc area" );
   }
   request.kernel = DrizzleKernel::Square;

   // The rejection mask lives on the reference grid and is sampled at the
   // rounded reference position of each dropped pixel; the normalization is
   // applied in Float32 exactly as the integration does; the CFA channel
   // selection keeps only the pattern's pixels.
   const std::uint32_t maskWidth = width + 12, maskHeight = height + 14;
   std::vector<std::uint8_t> mask( static_cast<std::size_t>( maskWidth )*maskHeight, 1 );
   const double maskedX = std::round( (forward[0]*4 + forward[1]*3 + forward[2])/scale );
   const double maskedY = std::round( (forward[3]*4 + forward[4]*3 + forward[5])/scale );
   mask[static_cast<std::size_t>( maskedY )*maskWidth + static_cast<std::size_t>( maskedX )] = 0;
   request.mask = mask;
   request.maskWidth = maskWidth;
   request.maskHeight = maskHeight;
   request.normalizationScale = 1.5F;
   request.normalizationOffset = -2.25F;
   request.frameWeight = 0.5F;
   const DrizzleOutput masked = DrizzleWhole( request, outputWidth, outputHeight, 3 );
   double maskedWeight = 0.0, maskedSum = 0.0, expectedMaskedSum = 0.0, expectedMaskedWeight = 0.0;
   std::uint32_t rejected = 0;
   for ( std::size_t i = 0; i < masked.weight.size(); ++i )
   {
      maskedWeight += masked.weight[i];
      maskedSum += masked.sum[i];
   }
   for ( std::uint32_t y = 0; y < height; ++y )
      for ( std::uint32_t x = 0; x < width; ++x )
      {
         const float value = source[y*width + x];
         if ( !std::isfinite( value ) )
            continue;
         const double rx = std::round( (forward[0]*x + forward[1]*y + forward[2])/scale );
         const double ry = std::round( (forward[3]*x + forward[4]*y + forward[5])/scale );
         if ( mask[static_cast<std::size_t>( ry )*maskWidth + static_cast<std::size_t>( rx )] == 0 )
         {
            ++rejected;
            continue;
         }
         const float normalized = (value*1.5F) + (-2.25F);
         expectedMaskedWeight += 0.5*(0.7*scale)*(0.7*scale);
         expectedMaskedSum += 0.5*(0.7*scale)*(0.7*scale)*static_cast<double>( normalized );
      }
   Require( rejected == 1, "drizzle test: exactly one input pixel lands on the masked reference pixel" );
   Require( std::fabs( maskedWeight - expectedMaskedWeight ) < 1.0e-6*expectedMaskedWeight,
            "drizzle: the rejected pixel drops nothing and the frame weight scales the rest" );
   Require( std::fabs( maskedSum - expectedMaskedSum ) < 1.0e-6*std::fabs( expectedMaskedSum ),
            "drizzle: normalization is applied before dropping" );

   request.mask = {};
   request.maskWidth = request.maskHeight = 0;
   request.normalizationScale = 1.0F;
   request.normalizationOffset = 0.0F;
   request.frameWeight = 1.0F;
   const std::uint8_t pattern[4] = { 0, 1, 1, 2 };
   std::copy( pattern, pattern + 4, request.cfaPattern );
   double channelWeight[3] = { 0.0, 0.0, 0.0 };
   for ( std::uint8_t channel = 0; channel < 3; ++channel )
   {
      request.channel = channel;
      const DrizzleOutput plane = DrizzleWhole( request, outputWidth, outputHeight, 2 );
      for ( double w : plane.weight )
         channelWeight[channel] += w;
   }
   Require( std::fabs( channelWeight[0] + channelWeight[1] + channelWeight[2] - expectedWeight )
               < 1.0e-6*expectedWeight,
            "drizzle: the three CFA planes partition the mosaic" );
   Require( channelWeight[1] > channelWeight[0] && channelWeight[1] > channelWeight[2],
            "drizzle: the green plane holds half of the Bayer pixels" );
   request.channel = 255;

   // C ABI round trip against the direct kernel.
   OafNativeDrizzleRequestV1 abi{};
   abi.struct_size = sizeof( abi );
   abi.source_width = width;
   abi.source_rows = height;
   abi.scale = 2;
   abi.kernel = OAF_NATIVE_DRIZZLE_KERNEL_SQUARE;
   abi.output_width = outputWidth;
   abi.output_rows = outputHeight;
   abi.threads = 3;
   std::copy( pattern, pattern + 4, abi.cfa_pattern );
   abi.channel = 255;
   abi.normalization_scale = 1.0F;
   abi.frame_weight = 1.0F;
   abi.pixfrac = 0.7;
   std::copy( forward, forward + 9, abi.forward );
   abi.source = source.data();
   abi.source_count = source.size();
   std::vector<double> abiSum( single.sum.size(), 0.0 ), abiWeight( single.sum.size(), 0.0 );
   std::vector<std::uint8_t> abiTouched( single.sum.size(), 0 );
   abi.output_sum = abiSum.data();
   abi.output_weight = abiWeight.data();
   abi.output_count = abiSum.size();
   abi.output_touched = abiTouched.data();
   std::array<char, 256> error{};
   Require( oaf_native_cpu_drizzle_v1( &abi, error.data(), error.size() ) == OAF_NATIVE_OK,
            error.data() );
   Require( abiSum == single.sum && abiWeight == single.weight && abiTouched == single.touched,
            "drizzle: C ABI equals the direct kernel" );
   abi.scale = 9;
   Require( oaf_native_cpu_drizzle_v1( &abi, error.data(), error.size() ) == OAF_NATIVE_INVALID_ARGUMENT,
            "drizzle: C ABI rejects an unsupported scale" );
   abi.scale = 2;
   abi.output_count = abiSum.size() - 1;
   Require( oaf_native_cpu_drizzle_v1( &abi, error.data(), error.size() ) == OAF_NATIVE_INVALID_ARGUMENT,
            "drizzle: C ABI rejects mismatched accumulators" );

   request.pixfrac = 0.0;
   RequireThrows<std::invalid_argument>( [&]() { DrizzleWhole( request, outputWidth, outputHeight, 1 ); },
                                         "drizzle: pixfrac must be positive" );
   request.pixfrac = 0.7;
   request.channel = 3;
   RequireThrows<std::invalid_argument>( [&]() { DrizzleWhole( request, outputWidth, outputHeight, 1 ); },
                                         "drizzle: channel must be 0, 1, 2 or 255" );
}

void TestDebayerBilinearMatchesTheReferenceRules()
{
   // A 4x5 RGGB mosaic with one missing (NaN) green sample.
   const std::uint32_t width = 5, height = 4;
   std::vector<float> mosaic( width*height );
   for ( std::uint32_t y = 0; y < height; ++y )
      for ( std::uint32_t x = 0; x < width; ++x )
         mosaic[y*width + x] = static_cast<float>( 10*y + x );
   mosaic[1*width + 2] = Nan; // (1,2) is green in RGGB (row odd, column even)
   DebayerRequest request;
   request.mosaic = mosaic;
   request.width = width;
   request.height = height;
   const std::uint8_t pattern[4] = { 0, 1, 1, 2 };
   std::copy( pattern, pattern + 4, request.pattern );
   std::vector<float> planes( 3*width*height );
   request.planes = planes;
   request.threads = 1;
   DebayerBilinear( request );
   auto plane = [&]( int channel, std::uint32_t y, std::uint32_t x ) { return planes[channel*width*height + y*width + x]; };
   // Known samples are copied.
   Require( plane( 0, 0, 0 ) == 0.0F && plane( 1, 0, 1 ) == 1.0F && plane( 2, 1, 1 ) == 11.0F,
            "debayer: known samples are copied" );
   // Red at (0,1): left/right red neighbours (0,0) and (0,2) -> (0 + 2)/2.
   Require( plane( 0, 0, 1 ) == 1.0F, "debayer: row neighbours interpolate" );
   // Red at (1,0): up/down red neighbours (0,0) and (2,0) -> (0 + 20)/2.
   Require( plane( 0, 1, 0 ) == 10.0F, "debayer: column neighbours interpolate" );
   // Red at (1,1): no 4-neighbour is red; diagonals (0,0),(0,2),(2,0),(2,2) -> 11.
   Require( plane( 0, 1, 1 ) == 11.0F, "debayer: diagonal neighbours interpolate" );
   // Green at (1,1) (a blue position): neighbours (1,0) green, (1,2) green but NaN,
   // (0,1) green, (2,1) green -> (10 + 1 + 21)/3 = 32/3 rounded to Float32.
   Require( plane( 1, 1, 1 ) == static_cast<float>( (10.0 + 0.0 + (1.0 + 21.0))/3.0 ),
            "debayer: a NaN sample is left out of the mean" );
   // The NaN green sample's own pixel has no green 4-neighbour (its
   // neighbours are red and blue positions) and takes its green diagonals
   // (0,1), (0,3), (2,1), (2,3) -> (1 + 3 + 21 + 23)/4.
   Require( plane( 1, 1, 2 ) == 12.0F, "debayer: a missing checkerboard sample is filled from its diagonals" );
   // Edge clamping: blue at (0,0) has no blue 4-neighbour (clamped up/left are
   // itself); its clamped diagonals are (0,1) [row neighbour], (1,1) and (1,1)
   // -> mean of the known blue ones = (11 + 11)/2 with (0,1) not blue.
   Require( plane( 2, 0, 0 ) == 11.0F, "debayer: edge diagonals clamp like NumPy's edge padding" );
   for ( std::uint32_t threads : { 2U, 3U, 8U } )
   {
      std::vector<float> threaded( planes.size() );
      request.planes = threaded;
      request.threads = threads;
      DebayerBilinear( request );
      for ( std::size_t i = 0; i < planes.size(); ++i )
         Require( SameOrBothNan( threaded[i], planes[i] ), "debayer: thread-invariant" );
   }
   request.planes = planes;
   request.pattern[0] = 3;
   RequireThrows<std::invalid_argument>( [&]() { DebayerBilinear( request ); },
                                         "debayer: pattern channels must be 0..2" );
}

void TestLanczos3TableIsDeterministicAndAccurate()
{
   using namespace openastroflow::native::detail;
   // The series agrees with libm to a few ulps and is exact at integers.
   for ( int i = -40; i <= 40; ++i )
   {
      const double v = static_cast<double>( i )*0.1;
      Require( std::fabs( DeterministicSinPi( v ) - std::sin( 3.141592653589793*v ) ) < 4.0e-15,
               "lanczos table: deterministic sin(pi v) matches libm" );
   }
   Require( DeterministicSinPi( 3.0 ) == 0.0 && DeterministicSinPi( -2.0 ) == 0.0,
            "lanczos table: sin(pi n) is exactly zero" );
   const double* table = Lanczos3TableNodeValues();
   // Row 1 is f = 0: the centre tap alone; row N+1 is f = 1: the next tap.
   Require( table[6 + 2] == 1.0 && table[6 + 0] == 0.0 && table[6 + 1] == 0.0 && table[6 + 3] == 0.0,
            "lanczos table: f = 0 selects the centre tap exactly" );
   Require( table[(Lanczos3TableIntervals + 1)*6 + 3] == 1.0,
            "lanczos table: f = 1 selects the next tap exactly" );
   for ( std::uint32_t node = 0; node < Lanczos3TableNodes; ++node )
   {
      double total = 0.0;
      for ( int tap = 0; tap < 6; ++tap )
         total += table[node*6 + tap];
      Require( std::fabs( total - 1.0 ) < 1.0e-15, "lanczos table: every node is normalized" );
   }
   // Interpolated weights against the exact sinc product, far below Float32.
   float weights[6];
   for ( int i = 0; i < 2000; ++i )
   {
      const double f = (static_cast<double>( i ) + 0.37)/2000.0;
      Lanczos3TableWeights( f, weights );
      double exact[6];
      double total = 0.0;
      for ( int tap = 0; tap < 6; ++tap )
      {
         const double d = f - static_cast<double>( tap - 2 );
         const double x = 3.141592653589793*d;
         exact[tap] = (std::sin( x )/x)*(std::sin( x/3.0 )/(x/3.0));
         total += exact[tap];
      }
      float sum = 0.0F;
      for ( int tap = 0; tap < 6; ++tap )
      {
         Require( std::fabs( static_cast<double>( weights[tap] ) - exact[tap]/total ) < 1.0e-7,
                  "lanczos table: interpolated weights match the exact weights within Float32" );
         sum += weights[tap];
      }
      Require( std::fabs( sum - 1.0F ) < 4.0e-7F, "lanczos table: weights sum to one" );
   }
   Lanczos3TableWeights( 0.0, weights );
   Require( weights[2] == 1.0F && weights[0] == 0.0F && weights[5] == 0.0F,
            "lanczos table: a zero fraction is an exact copy" );
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
      TestMaskedMeanSampleWeightsScaleContributions();
      TestTileOffsetsMatchReferenceStatistics();
      TestRadonPeaksFindTheDyadicLineAtEveryLevel();
      TestCAbiRoundTrip();
      TestDrizzleBandDropsExactAreasAndIsBandAndThreadInvariant();
      TestDebayerBilinearMatchesTheReferenceRules();
      TestLanczos3TableIsDeterministicAndAccurate();
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
