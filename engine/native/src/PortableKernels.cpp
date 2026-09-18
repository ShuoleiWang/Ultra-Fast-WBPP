#include "openastroflow/PortableKernels.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace openastroflow::native
{

namespace
{

// np.pi as stored by NumPy: the nearest Float64 to pi.
constexpr double Pi = 3.141592653589793;

std::size_t CheckedMultiply( std::size_t left,
                             std::size_t right,
                             const char* role )
{
   if ( left != 0 && right > std::numeric_limits<std::size_t>::max()/left )
      throw std::overflow_error( role );
   return left*right;
}

// Runs function(begin, end) over [0, count) on up to `threads` threads that
// claim contiguous chunks of `grain` items from a shared atomic counter, so a
// slow core (efficiency core, SMT sibling, throttled core) never holds the
// tail of the range while the others idle.  Every item is processed exactly
// once and the per-item results never depend on which thread or chunk ran
// it; the calling thread drains chunks too.  Exceptions stop further claims,
// are collected, and are rethrown after every worker has joined so no thread
// outlives the call.
template <class Function>
void ParallelRange( std::size_t count,
                    std::uint32_t threads,
                    std::size_t grain,
                    Function&& function )
{
   if ( count == 0 )
      return;
   grain = std::max<std::size_t>( 1, grain );
   const std::size_t chunks = (count + grain - 1)/grain;
   const std::size_t workers = std::max<std::size_t>(
      1, std::min<std::size_t>( threads, chunks ) );
   if ( workers == 1 )
   {
      function( std::size_t{ 0 }, count );
      return;
   }
   std::atomic<std::size_t> next{ 0 };
   std::vector<std::exception_ptr> errors( workers );
   auto drain = [&]( std::size_t worker )
   {
      try
      {
         for ( ;; )
         {
            const std::size_t begin =
               next.fetch_add( grain, std::memory_order_relaxed );
            if ( begin >= count )
               return;
            function( begin, std::min( count, begin + grain ) );
         }
      }
      catch ( ... )
      {
         errors[worker] = std::current_exception();
         next.store( count, std::memory_order_relaxed );
      }
   };
   std::vector<std::thread> pool;
   pool.reserve( workers - 1 );
   for ( std::size_t index = 1; index < workers; ++index )
      pool.emplace_back( [&drain, index]() { drain( index ); } );
   drain( 0 );
   for ( std::thread& worker : pool )
      worker.join();
   for ( const std::exception_ptr& error : errors )
      if ( error )
         std::rethrow_exception( error );
}

// Chunk sizes: a few milliseconds of work per claim on one core, so the
// atomic counter costs nothing and the load balances at the end of a range.
constexpr std::size_t WarpRowGrain = 8;
constexpr std::size_t RejectionPixelGrain = 4096;
constexpr std::size_t MeanPixelGrain = 8192;
constexpr std::size_t TileGrain = 1;

// Lanczos-3 tap constants for offsets k = -2..3 (see calibration.py):
//   sin(pi(f-k))   = (-1)^k sin(pi f)
//   sin(pi(f-k)/3) = sin(pi f/3) cos(k pi/3) - cos(pi f/3) sin(k pi/3)
constexpr double Sqrt3Half = 0.8660254037844386;
constexpr double TapSigns[6] = { 1.0, -1.0, 1.0, -1.0, 1.0, -1.0 };
constexpr double TapCosines[6] = { -0.5, 0.5, 1.0, 0.5, -0.5, -1.0 };
constexpr double TapSines[6] =
   { -Sqrt3Half, -Sqrt3Half, 0.0, Sqrt3Half, Sqrt3Half, 0.0 };

// Reproduces _sample_lanczos3_clamped_from's weight evaluation: three
// Float64 transcendental calls per axis feed all six sinc products through
// exact identities, each raw weight is stored as Float32, and the normalized
// weight is Float32 of the Float64 quotient by the Float64 tap total.
void EvaluateLanczos3Weights( double fraction, float weights[6] )
{
   double total = 0.0;
   float stored[6];
   const double primarySine = std::sin( Pi*fraction );
   const double reduced = (Pi*fraction)/3.0;
   const double reducedSine = std::sin( reduced );
   const double reducedCosine = std::cos( reduced );
   for ( int tap = 0; tap < 6; ++tap )
   {
      const double distance = fraction - static_cast<double>( tap - 2 );
      const double absolute = std::fabs( distance );
      double weight = 0.0;
      const bool atOrigin = absolute <= 1.0e-14;
      if ( atOrigin )
         weight = 1.0;
      else if ( absolute < 3.0 )
      {
         const double phase = Pi*distance;
         const double primary = (TapSigns[tap]*primarySine)/phase;
         const double left = reducedSine*TapCosines[tap];
         const double right = reducedCosine*TapSines[tap];
         const double secondary = (left - right)/(phase/3.0);
         weight = primary*secondary;
      }
      // Exact integer offsets other than zero are mathematical zeros.
      if ( absolute > 1.0e-14
        && std::fabs( distance - std::nearbyint( distance ) ) <= 1.0e-14 )
         weight = 0.0;
      total += weight;
      stored[tap] = static_cast<float>( weight );
   }
   if ( !std::isfinite( total ) || std::fabs( total ) < 1.0e-12 )
      throw std::runtime_error(
         "Lanczos-3 weight normalization is singular" );
   for ( int tap = 0; tap < 6; ++tap )
      weights[tap] = static_cast<float>(
         static_cast<double>( stored[tap] )/total );
}

// np.nanmedian over the finite values of one pixel: the middle sorted value
// for an odd count, Float32(low + high)/2 for an even count.
float MedianOfFinite( float* values, std::uint32_t count )
{
   const std::uint32_t half = count/2;
   std::nth_element( values, values + half, values + count );
   const float upper = values[half];
   if ( (count & 1U) != 0 )
      return upper;
   const float lower = *std::max_element( values, values + half );
   const float sum = lower + upper;
   return sum/2.0F;
}

} // namespace

void WarpLanczos3Request::Validate() const
{
   if ( sourceWidth < 6 || sourceHeight < 6 )
      throw std::invalid_argument(
         "Lanczos-3 warp requires a source of at least 6x6 pixels" );
   if ( outputWidth == 0 || rowCount == 0 )
      throw std::invalid_argument(
         "Lanczos-3 warp requires a nonempty output band" );
   if ( firstRow > std::numeric_limits<std::uint32_t>::max() - rowCount )
      throw std::invalid_argument( "Lanczos-3 warp row range overflows" );
   const std::size_t expected = CheckedMultiply(
      sourceWidth, sourceHeight, "Lanczos-3 warp source size overflow" );
   if ( source.size() != expected )
      throw std::invalid_argument(
         "Lanczos-3 warp source sample count differs from its geometry" );
   if ( !std::isfinite( domainScale ) || domainScale <= 0.0F )
      throw std::invalid_argument(
         "Lanczos-3 warp requires a finite positive numeric domain scale" );
   const double coefficients[] = {
      inverse.m00, inverse.m01, inverse.m02,
      inverse.m10, inverse.m11, inverse.m12,
      inverse.m20, inverse.m21, inverse.m22 };
   for ( double value : coefficients )
      if ( !std::isfinite( value ) )
         throw std::invalid_argument(
            "Lanczos-3 warp inverse matrix must be finite" );
}

std::size_t WarpLanczos3Request::OutputPixels() const
{
   return CheckedMultiply(
      outputWidth, rowCount, "Lanczos-3 warp output size overflow" );
}

void WarpLanczos3Clamped( const WarpLanczos3Request& request,
                          std::span<float> destination )
{
   request.Validate();
   const std::size_t outputPixels = request.OutputPixels();
   if ( destination.size() < outputPixels )
      throw std::invalid_argument(
         "Lanczos-3 warp destination is smaller than the output band" );

   const AffineInverse inverse = request.inverse;
   const bool projective = !inverse.IsAffine();
   const std::uint32_t width = request.sourceWidth;
   const std::uint32_t height = request.sourceHeight;
   // Python: width - 3.0 (exact Float64 for every practical image size).
   const double xLimit = static_cast<double>( width ) - 3.0;
   const double yLimit = static_cast<double>( height ) - 3.0;
   const float* source = request.source.data();
   const float domainScale = request.domainScale;
   const std::size_t outputWidth = request.outputWidth;
   const float nan = std::numeric_limits<float>::quiet_NaN();
   float* output = destination.data();

   ParallelRange( request.rowCount, request.threads, WarpRowGrain,
      [&]( std::size_t rowBegin, std::size_t rowEnd )
      {
         float xWeights[6] = {};
         float yWeights[6] = {};
         // Fractions lie in [0, 1); -1 never matches so the first pixel
         // always evaluates its weights.
         double lastXFraction = -1.0;
         double lastYFraction = -1.0;
         for ( std::size_t localRow = rowBegin; localRow < rowEnd; ++localRow )
         {
            const double outputY = static_cast<double>(
               request.firstRow + localRow );
            const double xRowTerm = inverse.m01*outputY;
            const double yRowTerm = inverse.m11*outputY;
            const double wRowTerm = inverse.m21*outputY;
            float* row = output + localRow*outputWidth;
            for ( std::size_t column = 0; column < outputWidth; ++column )
            {
               const double outputX = static_cast<double>( column );
               double inputX = inverse.m00*outputX;
               inputX = inputX + xRowTerm;
               inputX = inputX + inverse.m02;
               double inputY = inverse.m10*outputX;
               inputY = inputY + yRowTerm;
               inputY = inputY + inverse.m12;
               if ( projective )
               {
                  double w = inverse.m20*outputX;
                  w = w + wRowTerm;
                  w = w + inverse.m22;
                  inputX = inputX/w;
                  inputY = inputY/w;
               }
               if ( !( std::isfinite( inputX ) && std::isfinite( inputY )
                    && inputX >= 2.0 && inputX <= xLimit
                    && inputY >= 2.0 && inputY <= yLimit ) )
               {
                  row[column] = nan;
                  continue;
               }
               const double xFloorValue = std::floor( inputX );
               const double yFloorValue = std::floor( inputY );
               const std::int64_t xFloor =
                  static_cast<std::int64_t>( xFloorValue );
               const std::int64_t yFloor =
                  static_cast<std::int64_t>( yFloorValue );
               const double xFraction = inputX - xFloorValue;
               const double yFraction = inputY - yFloorValue;
               if ( xFraction != lastXFraction )
               {
                  EvaluateLanczos3Weights( xFraction, xWeights );
                  lastXFraction = xFraction;
               }
               if ( yFraction != lastYFraction )
               {
                  EvaluateLanczos3Weights( yFraction, yWeights );
                  lastYFraction = yFraction;
               }
               float samples = 0.0F;
               bool valid = true;
               float supportMinimum = std::numeric_limits<float>::infinity();
               float supportMaximum = -std::numeric_limits<float>::infinity();
               for ( int j = 0; j < 6; ++j )
               {
                  std::int64_t yIndex = yFloor + (j - 2);
                  if ( j == 5 )
                     yIndex = std::min<std::int64_t>( yIndex, height - 1 );
                  const float* sourceRow =
                     source + static_cast<std::size_t>( yIndex )*width;
                  const float yWeight = yWeights[j];
                  for ( int i = 0; i < 6; ++i )
                  {
                     std::int64_t xIndex = xFloor + (i - 2);
                     if ( i == 5 )
                        xIndex = std::min<std::int64_t>( xIndex, width - 1 );
                     const float combined = yWeight*xWeights[i];
                     const float neighbor =
                        sourceRow[static_cast<std::size_t>( xIndex )];
                     const bool active = combined != 0.0F;
                     const bool finite = std::isfinite( neighbor );
                     if ( active && finite )
                     {
                        supportMinimum = std::min( supportMinimum, neighbor );
                        supportMaximum = std::max( supportMaximum, neighbor );
                        const float product = neighbor*combined;
                        samples = samples + product;
                     }
                     else
                     {
                        if ( active )
                           valid = false;
                        samples = samples + 0.0F;
                     }
                  }
               }
               if ( !valid )
               {
                  row[column] = nan;
                  continue;
               }
               const float lower = std::min( supportMinimum, 0.0F );
               const float upper = std::max( supportMaximum, domainScale );
               row[column] = std::min( std::max( samples, lower ), upper );
            }
         }
      } );
}

void MadRejectionRequest::Validate() const
{
   if ( frameCount == 0 || frameCount > 65535 )
      throw std::invalid_argument(
         "MAD rejection frame count must be in [1, 65535]" );
   if ( rowCount == 0 || width == 0 )
      throw std::invalid_argument( "MAD rejection tile must be nonempty" );
   const std::size_t samples = CheckedMultiply(
      frameCount, TilePixels(), "MAD rejection sample count overflow" );
   if ( frameMajorSamples.size() != samples )
      throw std::invalid_argument(
         "MAD rejection sample count differs from its geometry" );
   if ( !std::isfinite( sigmaClip ) || sigmaClip <= 0.0F )
      throw std::invalid_argument(
         "MAD rejection sigma clip must be finite and positive" );
   if ( minimumRejectionFrames < 3 )
      throw std::invalid_argument(
         "MAD rejection requires at least three frames per decision" );
   for ( float floor : { groupSigmaFloor, absoluteFloor, epsilonFloor } )
      if ( !std::isfinite( floor ) || floor < 0.0F )
         throw std::invalid_argument(
            "MAD rejection floors must be finite and nonnegative" );
   if ( !frameScales.empty() )
   {
      if ( frameScales.size() != frameCount )
         throw std::invalid_argument(
            "MAD rejection frame scale count differs from the frame count" );
      for ( float scale : frameScales )
         if ( !std::isfinite( scale ) || scale <= 0.0F )
            throw std::invalid_argument(
               "MAD rejection frame scales must be finite and positive" );
   }
   if ( poolHalfWidth > 65535U )
      throw std::invalid_argument( "MAD rejection pool half width is too large" );
}

std::size_t MadRejectionRequest::TilePixels() const
{
   return CheckedMultiply( rowCount, width, "MAD rejection tile overflow" );
}

bool MadRejectionRequest::UsesScaleModel() const noexcept
{
   if ( poolHalfWidth > 0 )
      return true;
   for ( float scale : frameScales )
      if ( scale != 1.0F )
         return true;
   return false;
}

namespace
{

// v2 scale model, phase 2: per-frame thresholds from the pooled row MADs.
// Phase 1 (MadRejectionMask) has written the centre of every pixel and its
// MAD (NaN when the pixel has too few finite samples for a decision).
void ScaledRejectionDecisions( const MadRejectionRequest& request,
                               const std::vector<float>& madMap,
                               std::span<const float> centerData,
                               std::span<std::uint8_t> accepted )
{
   const std::size_t pixels = request.TilePixels();
   const std::uint32_t frames = request.frameCount;
   const std::uint32_t width = request.width;
   const std::size_t halfWidth = request.poolHalfWidth;
   const float* samples = request.frameMajorSamples.data();
   std::uint8_t* acceptedData = accepted.data();
   std::vector<float> scales( frames, 1.0F );
   if ( !request.frameScales.empty() )
      std::copy( request.frameScales.begin(), request.frameScales.end(),
                 scales.begin() );

   ParallelRange( pixels, request.threads, RejectionPixelGrain,
      [&]( std::size_t begin, std::size_t end )
      {
         constexpr std::size_t Block = 64;
         std::vector<float> transposed( Block*frames );
         std::vector<float> window( 2*halfWidth + 1 );
         std::vector<std::uint8_t> decisions( Block*frames );
         for ( std::size_t blockStart = begin; blockStart < end;
               blockStart += Block )
         {
            const std::size_t count = std::min( Block, end - blockStart );
            for ( std::uint32_t frame = 0; frame < frames; ++frame )
            {
               const float* row = samples
                  + static_cast<std::size_t>( frame )*pixels + blockStart;
               for ( std::size_t i = 0; i < count; ++i )
                  transposed[i*frames + frame] = row[i];
            }
            for ( std::size_t i = 0; i < count; ++i )
            {
               const std::size_t pixel = blockStart + i;
               const float* values = transposed.data() + i*frames;
               std::uint8_t* flags = decisions.data() + i*frames;
               const float mad = madMap[pixel];
               if ( std::isnan( mad ) )
               {
                  // Too few usable samples: never infer outliers.
                  for ( std::uint32_t frame = 0; frame < frames; ++frame )
                     flags[frame] = std::isfinite( values[frame] ) ? 1 : 0;
                  continue;
               }
               const std::size_t x = pixel % width;
               const std::size_t rowStart = pixel - x;
               const std::size_t first = x > halfWidth ? x - halfWidth : 0;
               const std::size_t last = std::min<std::size_t>( width - 1, x + halfWidth );
               std::uint32_t windowCount = 0;
               for ( std::size_t column = first; column <= last; ++column )
               {
                  const float neighbour = madMap[rowStart + column];
                  if ( !std::isnan( neighbour ) )
                     window[windowCount++] = neighbour;
               }
               const float pooledMad = MedianOfFinite( window.data(), windowCount );
               const float pixelCenter = centerData[pixel];
               const float robustSigma = 1.4826F*mad;
               const float sigmaPool = 1.4826F*pooledMad;
               const float excess = std::max(
                  robustSigma*robustSigma - sigmaPool*sigmaPool, 0.0F );
               const float numericalFloor = std::max(
                  request.absoluteFloor,
                  request.epsilonFloor
                     *std::max( 1.0F, std::fabs( pixelCenter ) ) );
               for ( std::uint32_t frame = 0; frame < frames; ++frame )
               {
                  const float value = values[frame];
                  bool keep = false;
                  if ( std::isfinite( value ) )
                  {
                     const float scaled = scales[frame]*sigmaPool;
                     const float sigmaFrame = std::sqrt( scaled*scaled + excess );
                     const float effectiveSigma = std::max(
                        std::max( sigmaFrame, request.groupSigmaFloor ),
                        numericalFloor );
                     const float threshold = request.sigmaClip*effectiveSigma;
                     const float deviation = value - pixelCenter;
                     keep = std::fabs( deviation ) <= threshold;
                  }
                  flags[frame] = keep ? 1 : 0;
               }
            }
            for ( std::uint32_t frame = 0; frame < frames; ++frame )
            {
               std::uint8_t* row = acceptedData
                  + static_cast<std::size_t>( frame )*pixels + blockStart;
               for ( std::size_t i = 0; i < count; ++i )
                  row[i] = decisions[i*frames + frame];
            }
         }
      } );
}

} // namespace

void MadRejectionMask( const MadRejectionRequest& request,
                       std::span<std::uint8_t> accepted,
                       std::span<float> center )
{
   request.Validate();
   const std::size_t pixels = request.TilePixels();
   const std::uint32_t frames = request.frameCount;
   if ( accepted.size() < request.frameMajorSamples.size()
     || center.size() < pixels )
      throw std::invalid_argument(
         "MAD rejection outputs are smaller than the request" );
   const float* samples = request.frameMajorSamples.data();
   const bool applyRejection = frames >= request.minimumRejectionFrames;
   const float nan = std::numeric_limits<float>::quiet_NaN();
   std::uint8_t* acceptedData = accepted.data();
   float* centerData = center.data();
   // Scale model: phase 1 below records every pixel's MAD, phase 2 pools
   // them along the rows and decides; the legacy path decides in phase 1.
   const bool scaleModel = applyRejection && request.UsesScaleModel();
   std::vector<float> madMap( scaleModel ? pixels : 0 );

   ParallelRange( pixels, request.threads, RejectionPixelGrain,
      [&]( std::size_t begin, std::size_t end )
      {
         constexpr std::size_t Block = 64;
         std::vector<float> transposed( Block*frames );
         std::vector<float> scratch( frames );
         std::vector<std::uint8_t> decisions( Block*frames );
         for ( std::size_t blockStart = begin; blockStart < end;
               blockStart += Block )
         {
            const std::size_t count = std::min( Block, end - blockStart );
            for ( std::uint32_t frame = 0; frame < frames; ++frame )
            {
               const float* row = samples
                  + static_cast<std::size_t>( frame )*pixels + blockStart;
               for ( std::size_t i = 0; i < count; ++i )
                  transposed[i*frames + frame] = row[i];
            }
            for ( std::size_t i = 0; i < count; ++i )
            {
               const float* values = transposed.data() + i*frames;
               std::uint8_t* flags = decisions.data() + i*frames;
               std::uint32_t finiteCount = 0;
               for ( std::uint32_t frame = 0; frame < frames; ++frame )
                  if ( std::isfinite( values[frame] ) )
                     scratch[finiteCount++] = values[frame];
               float pixelCenter = nan;
               if ( finiteCount > 0 )
                  pixelCenter = MedianOfFinite( scratch.data(), finiteCount );
               centerData[blockStart + i] = pixelCenter;
               if ( !applyRejection
                 || finiteCount < request.minimumRejectionFrames )
               {
                  // Too few usable samples: never infer outliers.
                  if ( scaleModel )
                     madMap[blockStart + i] = nan;
                  for ( std::uint32_t frame = 0; frame < frames; ++frame )
                     flags[frame] = std::isfinite( values[frame] ) ? 1 : 0;
                  continue;
               }
               std::uint32_t deviationCount = 0;
               for ( std::uint32_t frame = 0; frame < frames; ++frame )
                  if ( std::isfinite( values[frame] ) )
                  {
                     const float deviation = values[frame] - pixelCenter;
                     scratch[deviationCount++] = std::fabs( deviation );
                  }
               const float mad = MedianOfFinite( scratch.data(), deviationCount );
               if ( scaleModel )
               {
                  madMap[blockStart + i] = mad;
                  continue;
               }
               const float robustSigma = 1.4826F*mad;
               const float numericalFloor = std::max(
                  request.absoluteFloor,
                  request.epsilonFloor
                     *std::max( 1.0F, std::fabs( pixelCenter ) ) );
               const float effectiveSigma = std::max(
                  std::max( robustSigma, request.groupSigmaFloor ),
                  numericalFloor );
               const float threshold = request.sigmaClip*effectiveSigma;
               for ( std::uint32_t frame = 0; frame < frames; ++frame )
               {
                  const float value = values[frame];
                  bool keep = false;
                  if ( std::isfinite( value ) )
                  {
                     const float deviation = value - pixelCenter;
                     keep = std::fabs( deviation ) <= threshold;
                  }
                  flags[frame] = keep ? 1 : 0;
               }
            }
            for ( std::uint32_t frame = 0; frame < frames; ++frame )
            {
               std::uint8_t* row = acceptedData
                  + static_cast<std::size_t>( frame )*pixels + blockStart;
               for ( std::size_t i = 0; i < count; ++i )
                  row[i] = decisions[i*frames + frame];
            }
         }
      } );
   if ( scaleModel )
      ScaledRejectionDecisions(
         request, madMap, std::span<const float>( centerData, pixels ), accepted );
}

void MaskedMeanRequest::Validate() const
{
   if ( frameCount == 0 || frameCount > 65535 )
      throw std::invalid_argument(
         "masked mean frame count must be in [1, 65535]" );
   if ( rowCount == 0 || width == 0 )
      throw std::invalid_argument( "masked mean tile must be nonempty" );
   const std::size_t samples = CheckedMultiply(
      frameCount, TilePixels(), "masked mean sample count overflow" );
   if ( frameMajorSamples.size() != samples
     || frameMajorAccepted.size() != samples )
      throw std::invalid_argument(
         "masked mean sample and mask counts differ from the geometry" );
   if ( frameWeights.size() != frameCount )
      throw std::invalid_argument(
         "masked mean weight count differs from the frame count" );
   for ( double weight : frameWeights )
      if ( !std::isfinite( weight ) )
         throw std::invalid_argument( "masked mean weights must be finite" );
}

std::size_t MaskedMeanRequest::TilePixels() const
{
   return CheckedMultiply( rowCount, width, "masked mean tile overflow" );
}

void MaskedWeightedMean( const MaskedMeanRequest& request,
                         const MaskedMeanOutput& output )
{
   request.Validate();
   const std::size_t pixels = request.TilePixels();
   if ( output.integrated.size() < pixels
     || output.acceptedSamples.size() < pixels
     || output.rejectedSamples.size() < pixels )
      throw std::invalid_argument(
         "masked mean outputs are smaller than the request" );
   const float* samples = request.frameMajorSamples.data();
   const std::uint8_t* accepted = request.frameMajorAccepted.data();
   const double* weights = request.frameWeights.data();
   const std::uint32_t frames = request.frameCount;
   const float nan = std::numeric_limits<float>::quiet_NaN();
   float* integrated = output.integrated.data();
   std::uint16_t* acceptedSamples = output.acceptedSamples.data();
   std::uint16_t* rejectedSamples = output.rejectedSamples.data();

   ParallelRange( pixels, request.threads, MeanPixelGrain,
      [&]( std::size_t begin, std::size_t end )
      {
         constexpr std::size_t Block = 256;
         double numerator[Block];
         double denominator[Block];
         std::uint16_t acceptedCount[Block];
         std::uint16_t rejectedCount[Block];
         for ( std::size_t blockStart = begin; blockStart < end;
               blockStart += Block )
         {
            const std::size_t count = std::min( Block, end - blockStart );
            std::fill( numerator, numerator + count, 0.0 );
            std::fill( denominator, denominator + count, 0.0 );
            std::fill( acceptedCount, acceptedCount + count, std::uint16_t{ 0 } );
            std::fill( rejectedCount, rejectedCount + count, std::uint16_t{ 0 } );
            for ( std::uint32_t frame = 0; frame < frames; ++frame )
            {
               const std::size_t offset =
                  static_cast<std::size_t>( frame )*pixels + blockStart;
               const float* row = samples + offset;
               const std::uint8_t* mask = accepted + offset;
               const double weight = weights[frame];
               for ( std::size_t i = 0; i < count; ++i )
               {
                  const float value = row[i];
                  const bool isAccepted = mask[i] != 0;
                  const double term = isAccepted
                     ? static_cast<double>( value )*weight : 0.0;
                  numerator[i] = numerator[i] + term;
                  denominator[i] = denominator[i] + (isAccepted ? weight : 0.0);
                  if ( isAccepted )
                     ++acceptedCount[i];
                  else if ( std::isfinite( value ) )
                     ++rejectedCount[i];
               }
            }
            for ( std::size_t i = 0; i < count; ++i )
            {
               integrated[blockStart + i] = denominator[i] > 0.0
                  ? static_cast<float>( numerator[i]/denominator[i] )
                  : nan;
               acceptedSamples[blockStart + i] = acceptedCount[i];
               rejectedSamples[blockStart + i] = rejectedCount[i];
            }
         }
      } );
}

namespace
{

// np.median of an ascending Float64 sample: the middle value, or the mean of
// the two middle values (Float64 sum then division by two).
double SortedMedian( const std::vector<double>& sorted )
{
   const std::size_t count = sorted.size();
   const std::size_t half = count/2;
   if ( (count & 1U) != 0 )
      return sorted[half];
   const double sum = sorted[half - 1] + sorted[half];
   return sum/2.0;
}

double MedianOf( std::vector<double> values )
{
   std::sort( values.begin(), values.end() );
   return SortedMedian( values );
}

// np.quantile(..., method="linear") on an ascending Float64 sample:
// virtual index (n-1)*q, floor/next neighbours, gamma-weighted lerp with
// NumPy's symmetric rounding branch at gamma >= 0.5.
double SortedLinearQuantile( const std::vector<double>& sorted, double quantile )
{
   const std::size_t count = sorted.size();
   const double virtualIndex =
      static_cast<double>( count - 1 )*quantile;
   if ( virtualIndex >= static_cast<double>( count - 1 ) )
      return sorted[count - 1];
   if ( virtualIndex < 0.0 )
      return sorted[0];
   const double previous = std::floor( virtualIndex );
   const std::size_t previousIndex = static_cast<std::size_t>( previous );
   const double gamma = virtualIndex - previous;
   const double a = sorted[previousIndex];
   const double b = sorted[previousIndex + 1];
   const double difference = b - a;
   double result = a + difference*gamma;
   if ( gamma >= 0.5 )
   {
      const double complement = 1.0 - gamma;
      result = b - difference*complement;
   }
   return result;
}

// Reproduces global_normalization._location_and_mad.
std::pair<double, double> LocationAndMad( const std::vector<double>& values )
{
   const double location = MedianOf( values );
   std::vector<double> deviations( values.size() );
   for ( std::size_t i = 0; i < values.size(); ++i )
      deviations[i] = std::fabs( values[i] - location );
   const double dispersion = 1.4826*MedianOf( deviations );
   return { location, dispersion };
}

} // namespace

void TileOffsetRequest::Validate() const
{
   if ( tileCount == 0 )
      throw std::invalid_argument( "tile offsets require at least one tile" );
   if ( boundaries.size() != static_cast<std::size_t>( tileCount ) + 1 )
      throw std::invalid_argument(
         "tile offsets require tileCount + 1 boundaries" );
   if ( boundaries[0] != 0 || boundaries[tileCount] != target.size()
     || target.size() != reference.size() )
      throw std::invalid_argument(
         "tile offset boundaries do not cover the paired samples" );
   for ( std::uint32_t tile = 0; tile < tileCount; ++tile )
      if ( boundaries[tile] > boundaries[tile + 1] )
         throw std::invalid_argument( "tile offset boundaries must ascend" );
   if ( !std::isfinite( scale ) || !std::isfinite( residualClipSigma )
     || !std::isfinite( lowerQuantile ) || !std::isfinite( upperQuantile )
     || lowerQuantile < 0.0 || upperQuantile > 1.0
     || lowerQuantile > upperQuantile || minimumSamples < 1 )
      throw std::invalid_argument( "tile offset parameters are invalid" );
}

void TileOffsets( const TileOffsetRequest& request, const TileOffsetOutput& output )
{
   request.Validate();
   const std::size_t tiles = request.tileCount;
   if ( output.offset.size() < tiles || output.count.size() < tiles
     || output.residualMad.size() < tiles || output.valid.size() < tiles )
      throw std::invalid_argument( "tile offset outputs are smaller than the request" );
   const double nan = std::numeric_limits<double>::quiet_NaN();
   const double epsilon = std::numeric_limits<double>::epsilon();

   ParallelRange( tiles, request.threads, TileGrain,
      [&]( std::size_t begin, std::size_t end )
      {
         std::vector<double> x;
         std::vector<double> y;
         std::vector<double> sortedX;
         std::vector<double> sortedY;
         std::vector<double> residual;
         std::vector<double> kept;
         std::vector<double> centered;
         for ( std::size_t tile = begin; tile < end; ++tile )
         {
            output.offset[tile] = nan;
            output.count[tile] = 0;
            output.residualMad[tile] = nan;
            output.valid[tile] = 0;
            const std::size_t first = request.boundaries[tile];
            const std::size_t last = request.boundaries[tile + 1];
            x.clear();
            y.clear();
            for ( std::size_t i = first; i < last; ++i )
            {
               const double t = request.target[i];
               const double r = request.reference[i];
               if ( std::isfinite( t ) && std::isfinite( r ) )
               {
                  x.push_back( t );
                  y.push_back( r );
               }
            }
            if ( x.size() < request.minimumSamples )
               continue;
            sortedX = x;
            sortedY = y;
            std::sort( sortedX.begin(), sortedX.end() );
            std::sort( sortedY.begin(), sortedY.end() );
            const double xLow = SortedLinearQuantile( sortedX, request.lowerQuantile );
            const double xHigh = SortedLinearQuantile( sortedX, request.upperQuantile );
            const double yLow = SortedLinearQuantile( sortedY, request.lowerQuantile );
            const double yHigh = SortedLinearQuantile( sortedY, request.upperQuantile );
            residual.clear();
            for ( std::size_t i = 0; i < x.size(); ++i )
               if ( x[i] >= xLow && x[i] <= xHigh && y[i] >= yLow && y[i] <= yHigh )
               {
                  const double scaled = request.scale*x[i];
                  residual.push_back( y[i] - scaled );
               }
            if ( residual.size() < request.minimumSamples )
               continue;
            const auto [center, sigma] = LocationAndMad( residual );
            if ( sigma > epsilon )
            {
               const double limit = request.residualClipSigma*sigma;
               kept.clear();
               for ( double value : residual )
                  if ( std::fabs( value - center ) <= limit )
                     kept.push_back( value );
               if ( kept.size() >= request.minimumSamples )
                  residual.swap( kept );
            }
            const double offset = MedianOf( residual );
            centered.resize( residual.size() );
            for ( std::size_t i = 0; i < residual.size(); ++i )
               centered[i] = residual[i] - offset;
            const double residualMad = LocationAndMad( centered ).second;
            if ( !std::isfinite( offset ) || !std::isfinite( residualMad ) )
               continue;
            output.offset[tile] = offset;
            output.count[tile] = static_cast<std::uint32_t>( residual.size() );
            output.residualMad[tile] = residualMad;
            output.valid[tile] = 1;
         }
      } );
}

std::uint32_t DefaultKernelThreads() noexcept
{
   const unsigned concurrency = std::thread::hardware_concurrency();
   if ( concurrency == 0 )
      return 1;
   return static_cast<std::uint32_t>( std::min( 64U, concurrency ) );
}

} // namespace openastroflow::native
