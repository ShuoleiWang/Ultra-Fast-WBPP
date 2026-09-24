#include "ufwbpp/PortableKernels.h"
#include "Lanczos3Table.h"
#include "ParallelRange.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define UFWBPP_WARP_NEON 1
#endif

namespace ufwbpp::native
{

using detail::ParallelRange;

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

// Chunk sizes: a few milliseconds of work per claim on one core, so the
// atomic counter costs nothing and the load balances at the end of a range.
constexpr std::size_t WarpRowGrain = 8;
constexpr std::size_t RejectionPixelGrain = 4096;
constexpr std::size_t MeanPixelGrain = 8192;
constexpr std::size_t TileGrain = 1;

// Lanczos-3 tap weights come from the deterministic table (Lanczos3Table.h):
// the same values on every platform and library version, and the same
// values the Python reference computes.  Kept as a thin wrapper so the warp
// loop reads as before.
void EvaluateLanczos3Weights( double fraction, float weights[6] )
{
   detail::Lanczos3TableWeights( fraction, weights );
}

// The reference arithmetic of one warp output pixel: domain test, table
// weights (cached while a fraction repeats), the 36 products summed in
// row-major tap order in Float32, and the clamp to the support's range.
struct WarpPixelContext
{
   const float* source;
   std::int64_t width;
   std::int64_t height;
   double xLimit;
   double yLimit;
   float domainScale;
   float xWeights[6] = {};
   float yWeights[6] = {};
   // Fractions lie in [0, 1); -1 never matches so the first pixel always
   // evaluates its weights.
   double lastXFraction = -1.0;
   double lastYFraction = -1.0;

   float Evaluate( double inputX, double inputY )
   {
      if ( !( std::isfinite( inputX ) && std::isfinite( inputY )
           && inputX >= 2.0 && inputX <= xLimit
           && inputY >= 2.0 && inputY <= yLimit ) )
         return std::numeric_limits<float>::quiet_NaN();
      const double xFloorValue = std::floor( inputX );
      const double yFloorValue = std::floor( inputY );
      const std::int64_t xFloor = static_cast<std::int64_t>( xFloorValue );
      const std::int64_t yFloor = static_cast<std::int64_t>( yFloorValue );
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
         const float* sourceRow = source + static_cast<std::size_t>( yIndex*width );
         const float yWeight = yWeights[j];
         for ( int i = 0; i < 6; ++i )
         {
            std::int64_t xIndex = xFloor + (i - 2);
            if ( i == 5 )
               xIndex = std::min<std::int64_t>( xIndex, width - 1 );
            const float combined = yWeight*xWeights[i];
            const float neighbor = sourceRow[static_cast<std::size_t>( xIndex )];
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
         return std::numeric_limits<float>::quiet_NaN();
      const float lower = std::min( supportMinimum, 0.0F );
      const float upper = std::max( supportMaximum, domainScale );
      return std::min( std::max( samples, lower ), upper );
   }
};

#if UFWBPP_WARP_NEON
// Output pixels the NEON warp evaluates side by side, one per Float32 lane.
constexpr std::size_t WarpLanes = 4;

// detail::Lanczos3TableWeights for four fractions, lanes (0,1) and (2,3) in
// two Float64x2 vectors: every lane performs the scalar operations in the
// scalar order (x/2 as the exact x*0.5), so each lane's weights are the
// scalar weights bit for bit.  weights[tap] holds the tap's four lanes.
inline void Lanczos3WeightsNeon( const double* fraction, float32x4_t weights[6] )
{
   const double* table = detail::Lanczos3TableNodeValues();
   const float64x2_t zero = vdupq_n_f64( 0.0 );
   const float64x2_t one = vdupq_n_f64( 1.0 );
   const float64x2_t two = vdupq_n_f64( 2.0 );
   const float64x2_t six = vdupq_n_f64( 6.0 );
   float64x2_t quotients[2][6];
   for ( int pair = 0; pair < 2; ++pair )
   {
      const float64x2_t u = vmulq_n_f64( vld1q_f64( fraction + 2*pair ),
                                         static_cast<double>( detail::Lanczos3TableIntervals ) );
      const float64x2_t floorU = vrndmq_f64( u );
      const float64x2_t t = vsubq_f64( u, floorU );
      const float64x2_t a = vaddq_f64( t, one );
      const float64x2_t b = vsubq_f64( t, one );
      const float64x2_t c = vsubq_f64( t, two );
      const float64x2_t at = vmulq_f64( a, t );
      const float64x2_t b0 = vdivq_f64( vnegq_f64( vmulq_f64( vmulq_f64( t, b ), c ) ), six );
      const float64x2_t b1 = vmulq_n_f64( vmulq_f64( vmulq_f64( a, b ), c ), 0.5 );
      const float64x2_t b2 = vmulq_n_f64( vnegq_f64( vmulq_f64( at, c ) ), 0.5 );
      const float64x2_t b3 = vdivq_f64( vmulq_f64( at, b ), six );
      const double* rows0 = table + static_cast<std::size_t>( vgetq_lane_f64( floorU, 0 ) )*6;
      const double* rows1 = table + static_cast<std::size_t>( vgetq_lane_f64( floorU, 1 ) )*6;
      float64x2_t values[6];
      for ( int taps = 0; taps < 6; taps += 2 )
      {
         float64x2_t even[4], odd[4];
         for ( int node = 0; node < 4; ++node )
         {
            const float64x2_t lane0 = vld1q_f64( rows0 + 6*node + taps );
            const float64x2_t lane1 = vld1q_f64( rows1 + 6*node + taps );
            even[node] = vzip1q_f64( lane0, lane1 );
            odd[node] = vzip2q_f64( lane0, lane1 );
         }
         // ((b0*p0 + b1*p1) + b2*p2) + b3*p3, as the scalar reference.
         float64x2_t value = vaddq_f64( vmulq_f64( b0, even[0] ), vmulq_f64( b1, even[1] ) );
         value = vaddq_f64( value, vmulq_f64( b2, even[2] ) );
         values[taps] = vaddq_f64( value, vmulq_f64( b3, even[3] ) );
         value = vaddq_f64( vmulq_f64( b0, odd[0] ), vmulq_f64( b1, odd[1] ) );
         value = vaddq_f64( value, vmulq_f64( b2, odd[2] ) );
         values[taps + 1] = vaddq_f64( value, vmulq_f64( b3, odd[3] ) );
      }
      float64x2_t total = zero;
      for ( int tap = 0; tap < 6; ++tap )
         total = vaddq_f64( total, values[tap] );
      for ( int tap = 0; tap < 6; ++tap )
         quotients[pair][tap] = vdivq_f64( values[tap], total );
   }
   for ( int tap = 0; tap < 6; ++tap )
      weights[tap] = vcvt_high_f32_f64( vcvt_f32_f64( quotients[0][tap] ), quotients[1][tap] );
}

inline bool AllLanes( uint32x4_t mask )
{
   return vminvq_u32( mask ) != 0;
}

// WarpLanes interior output pixels (6x6 windows inside the source, no edge
// clamp applies): their window origins and table weights.
struct WarpGroupNeon
{
   const float* origin[WarpLanes];
   // Four horizontally adjacent windows on one source row: each tap's four
   // samples are one vector load.
   bool contiguous = false;
   float32x4_t xWeights[6];
   float32x4_t yWeights[6];

   void Prepare( const float* source, std::int64_t width, const std::int64_t* xFloor,
                 const std::int64_t* yFloor, const double* xFraction, const double* yFraction )
   {
      Lanczos3WeightsNeon( xFraction, xWeights );
      Lanczos3WeightsNeon( yFraction, yWeights );
      contiguous = yFloor[1] == yFloor[0] && yFloor[2] == yFloor[0] && yFloor[3] == yFloor[0]
                && xFloor[1] == xFloor[0] + 1 && xFloor[2] == xFloor[0] + 2
                && xFloor[3] == xFloor[0] + 3;
      for ( std::size_t lane = 0; lane < WarpLanes; ++lane )
         origin[lane] = source + static_cast<std::size_t>( (yFloor[lane] - 2)*width + (xFloor[lane] - 2) );
   }

   float32x4_t Neighbors( std::size_t offset ) const
   {
      if ( contiguous )
         return vld1q_f32( origin[0] + offset );
      float32x4_t value = vld1q_dup_f32( origin[0] + offset );
      value = vld1q_lane_f32( origin[1] + offset, value, 1 );
      value = vld1q_lane_f32( origin[2] + offset, value, 2 );
      return vld1q_lane_f32( origin[3] + offset, value, 3 );
   }

   // Weights of at least 2^-60 make every combined weight a normal nonzero
   // Float32, so every tap is active.
   bool AllActive() const
   {
      float32x4_t smallest = vabsq_f32( xWeights[0] );
      for ( int tap = 0; tap < 6; ++tap )
         smallest = vminq_f32( vminq_f32( smallest, vabsq_f32( xWeights[tap] ) ),
                               vabsq_f32( yWeights[tap] ) );
      return AllLanes( vcgeq_f32( smallest, vdupq_n_f32( 0x1p-60F ) ) );
   }
};

// The reference clamp: lower = std::min(minimum, 0), upper =
// std::max(maximum, scale), value = std::min(std::max(samples, lower),
// upper); invalid lanes are NaN.
inline void StoreWarpLanesNeon( float32x4_t samples, float32x4_t supportMinimum,
                                float32x4_t supportMaximum, uint32x4_t invalid,
                                float domainScale, float* output )
{
   const float32x4_t zero = vdupq_n_f32( 0.0F );
   const float32x4_t scale = vdupq_n_f32( domainScale );
   const float32x4_t lower = vbslq_f32( vcltq_f32( zero, supportMinimum ), zero, supportMinimum );
   const float32x4_t upper = vbslq_f32( vcltq_f32( supportMaximum, scale ), scale, supportMaximum );
   float32x4_t value = vbslq_f32( vcltq_f32( samples, lower ), lower, samples );
   value = vbslq_f32( vcltq_f32( upper, value ), upper, value );
   value = vbslq_f32( invalid, vdupq_n_f32( std::numeric_limits<float>::quiet_NaN() ), value );
   vst1q_f32( output, value );
}

// WarpPixelContext::Evaluate for one group: lane p runs pixel p's scalar
// operations in the scalar order -- the same products, the same Float32
// running sum (+0 for skipped taps), the support bounds as the reference's
// std::min/std::max selects and the same clamp -- so every output is the
// scalar output bit for bit.
inline void WarpExactNeon( const WarpGroupNeon& group, std::int64_t width, float domainScale,
                           float* output )
{
   const float32x4_t zero = vdupq_n_f32( 0.0F );
   const float32x4_t largest = vdupq_n_f32( std::numeric_limits<float>::max() );
   const uint32x4_t magnitude = vdupq_n_u32( 0x7fffffffU );
   float32x4_t samples = zero;
   float32x4_t supportMinimum = vdupq_n_f32( std::numeric_limits<float>::infinity() );
   float32x4_t supportMaximum = vdupq_n_f32( -std::numeric_limits<float>::infinity() );
   uint32x4_t invalid = vdupq_n_u32( 0 );
   for ( int j = 0; j < 6; ++j )
      for ( int i = 0; i < 6; ++i )
      {
         const float32x4_t combined = vmulq_f32( group.yWeights[j], group.xWeights[i] );
         const float32x4_t neighbor = group.Neighbors( static_cast<std::size_t>( j*width + i ) );
         const uint32x4_t active = vtstq_u32( vreinterpretq_u32_f32( combined ), magnitude );
         const uint32x4_t finite = vcaleq_f32( neighbor, largest );
         const uint32x4_t taken = vandq_u32( active, finite );
         supportMinimum = vbslq_f32( vandq_u32( taken, vcltq_f32( neighbor, supportMinimum ) ),
                                     neighbor, supportMinimum );
         supportMaximum = vbslq_f32( vandq_u32( taken, vcltq_f32( supportMaximum, neighbor ) ),
                                     neighbor, supportMaximum );
         samples = vaddq_f32( samples, vbslq_f32( taken, vmulq_f32( neighbor, combined ), zero ) );
         invalid = vorrq_u32( invalid, vbicq_u32( active, finite ) );
      }
   StoreWarpLanesNeon( samples, supportMinimum, supportMaximum, invalid, domainScale, output );
}

// The common case of WarpExactNeon for Groups groups at once (independent
// dependency chains side by side), with the same result: when every tap is
// active, a non-finite sample makes the running sum non-finite, so a finite
// sum proves every tap was taken; the sum is then the reference sum, and
// vminq/vmaxq give the reference bounds whenever the bounds are not zeros
// (a zero bound could be +0 or -0 depending on which came first, which only
// the select chain reproduces).  A group without that proof reruns the
// exact route.  Every group must be AllActive().
template <std::size_t Groups>
inline void WarpCommonNeon( const WarpGroupNeon* groups, std::int64_t width, float domainScale,
                            float* output )
{
   float32x4_t samples[Groups];
   float32x4_t supportMinimum[Groups];
   float32x4_t supportMaximum[Groups];
   for ( std::size_t g = 0; g < Groups; ++g )
   {
      samples[g] = vdupq_n_f32( 0.0F );
      supportMinimum[g] = vdupq_n_f32( std::numeric_limits<float>::infinity() );
      supportMaximum[g] = vdupq_n_f32( -std::numeric_limits<float>::infinity() );
   }
   for ( int j = 0; j < 6; ++j )
      for ( int i = 0; i < 6; ++i )
      {
         const std::size_t offset = static_cast<std::size_t>( j*width + i );
         for ( std::size_t g = 0; g < Groups; ++g )
         {
            const float32x4_t combined = vmulq_f32( groups[g].yWeights[j], groups[g].xWeights[i] );
            const float32x4_t neighbor = groups[g].Neighbors( offset );
            samples[g] = vaddq_f32( samples[g], vmulq_f32( neighbor, combined ) );
            supportMinimum[g] = vminq_f32( supportMinimum[g], neighbor );
            supportMaximum[g] = vmaxq_f32( supportMaximum[g], neighbor );
         }
      }
   const float32x4_t largest = vdupq_n_f32( std::numeric_limits<float>::max() );
   const uint32x4_t magnitude = vdupq_n_u32( 0x7fffffffU );
   for ( std::size_t g = 0; g < Groups; ++g )
   {
      const uint32x4_t certain = vandq_u32(
         vcaleq_f32( samples[g], largest ),
         vandq_u32( vtstq_u32( vreinterpretq_u32_f32( supportMinimum[g] ), magnitude ),
                    vtstq_u32( vreinterpretq_u32_f32( supportMaximum[g] ), magnitude ) ) );
      float* groupOutput = output + g*WarpLanes;
      if ( AllLanes( certain ) )
         StoreWarpLanesNeon( samples[g], supportMinimum[g], supportMaximum[g], vdupq_n_u32( 0 ),
                             domainScale, groupOutput );
      else
         WarpExactNeon( groups[g], width, domainScale, groupOutput );
   }
}

// Output pixels a row chunk prepares before any of them is warped: the
// geometry, then the weights, then the taps of the whole chunk run as three
// passes of independent iterations, which the core overlaps far better than
// one long per-pixel dependency chain.
constexpr std::size_t WarpChunkGroups = 32;
#endif

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

// Sorts a short sample in place (insertion sort beats selection for the
// frame counts of a stack) and returns the same median as MedianOfFinite.
constexpr std::uint32_t SortedMedianMaximumCount = 64;

float SortedMedianOfFinite( float* values, std::uint32_t count )
{
   if ( count > SortedMedianMaximumCount )
      return MedianOfFinite( values, count );
   for ( std::uint32_t i = 1; i < count; ++i )
   {
      const float value = values[i];
      std::uint32_t j = i;
      while ( j > 0 && values[j - 1] > value )
      {
         values[j] = values[j - 1];
         --j;
      }
      values[j] = value;
   }
   const std::uint32_t half = count/2;
   const float upper = values[half];
   if ( (count & 1U) != 0 )
      return upper;
   const float sum = values[half - 1] + upper;
   return sum/2.0F;
}

// Median absolute deviation of an ascending sample about ``center``: the
// deviations Float32(|value - center|) grow away from the centre on either
// side, so their k-th smallest is found by merging the two runs, the same
// order statistic (and the same Float32(low + high)/2 for an even count)
// MedianOfFinite returns from the deviation sample.
float SortedMadOfFinite( const float* sorted, std::uint32_t count, float center )
{
   if ( count > SortedMedianMaximumCount )
   {
      // Not reached for sorted samples (they are short); kept for safety.
      std::vector<float> deviations( sorted, sorted + count );
      for ( float& value : deviations )
      {
         const float deviation = value - center;
         value = std::fabs( deviation );
      }
      return MedianOfFinite( deviations.data(), count );
   }
   // Elements below the centre, walked downward from ``left``; elements at
   // or above it, walked upward from ``right``.
   std::int32_t left = -1;
   std::uint32_t right = 0;
   while ( right < count && sorted[right] < center )
      ++right;
   left = static_cast<std::int32_t>( right ) - 1;
   auto deviationAt = [&]( std::uint32_t index )
   {
      const float deviation = sorted[index] - center;
      return std::fabs( deviation );
   };
   const std::uint32_t half = count/2;
   float previous = 0.0F;
   float current = 0.0F;
   for ( std::uint32_t taken = 0; taken <= half; ++taken )
   {
      float next;
      if ( left >= 0
        && (right >= count || deviationAt( static_cast<std::uint32_t>( left ) ) <= deviationAt( right )) )
      {
         next = deviationAt( static_cast<std::uint32_t>( left ) );
         --left;
      }
      else
      {
         next = deviationAt( right );
         ++right;
      }
      previous = current;
      current = next;
   }
   if ( (count & 1U) != 0 )
      return current;
   const float sum = previous + current;
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
   float* output = destination.data();

   ParallelRange( request.rowCount, request.threads, WarpRowGrain,
      [&]( std::size_t rowBegin, std::size_t rowEnd )
      {
         WarpPixelContext pixel{ source, static_cast<std::int64_t>( width ),
                                 static_cast<std::int64_t>( height ), xLimit, yLimit, domainScale };
         for ( std::size_t localRow = rowBegin; localRow < rowEnd; ++localRow )
         {
            const double outputY = static_cast<double>(
               request.firstRow + localRow );
            const double xRowTerm = inverse.m01*outputY;
            const double yRowTerm = inverse.m11*outputY;
            const double wRowTerm = inverse.m21*outputY;
            float* row = output + localRow*outputWidth;
            const auto inputCoordinate = [&]( std::size_t column, double& inputX, double& inputY )
            {
               const double outputX = static_cast<double>( column );
               inputX = inverse.m00*outputX;
               inputX = inputX + xRowTerm;
               inputX = inputX + inverse.m02;
               inputY = inverse.m10*outputX;
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
            };
            std::size_t column = 0;
#if UFWBPP_WARP_NEON
            // Groups of WarpLanes pixels whose windows need no edge clamp run
            // in the NEON lanes; any other pixel runs the scalar reference.
            const auto scalar = [&]( std::size_t first, std::size_t count )
            {
               for ( std::size_t lane = 0; lane < count; ++lane )
               {
                  double x, y;
                  inputCoordinate( first + lane, x, y );
                  row[first + lane] = pixel.Evaluate( x, y );
               }
            };
            WarpGroupNeon groups[WarpChunkGroups];
            bool interior[WarpChunkGroups];
            while ( column + WarpLanes <= outputWidth )
            {
               const std::size_t chunk = std::min( WarpChunkGroups, (outputWidth - column)/WarpLanes );
               // Geometry and weights of every group of the chunk.
               for ( std::size_t g = 0; g < chunk; ++g )
               {
                  std::int64_t xFloor[WarpLanes], yFloor[WarpLanes];
                  double xFraction[WarpLanes], yFraction[WarpLanes];
                  bool inside = true;
                  for ( std::size_t lane = 0; lane < WarpLanes && inside; ++lane )
                  {
                     double inputX, inputY;
                     inputCoordinate( column + g*WarpLanes + lane, inputX, inputY );
                     inside = std::isfinite( inputX ) && std::isfinite( inputY )
                           && inputX >= 2.0 && inputX <= xLimit
                           && inputY >= 2.0 && inputY <= yLimit;
                     if ( !inside )
                        break;
                     const double xFloorValue = std::floor( inputX );
                     const double yFloorValue = std::floor( inputY );
                     xFloor[lane] = static_cast<std::int64_t>( xFloorValue );
                     yFloor[lane] = static_cast<std::int64_t>( yFloorValue );
                     xFraction[lane] = inputX - xFloorValue;
                     yFraction[lane] = inputY - yFloorValue;
                     inside = xFloor[lane] + 3 <= static_cast<std::int64_t>( width ) - 1
                           && yFloor[lane] + 3 <= static_cast<std::int64_t>( height ) - 1;
                  }
                  interior[g] = inside;
                  if ( inside )
                     groups[g].Prepare( source, width, xFloor, yFloor, xFraction, yFraction );
               }
               // Taps: pairs of common groups side by side.
               for ( std::size_t g = 0; g < chunk; )
               {
                  float* groupOutput = row + column + g*WarpLanes;
                  if ( !interior[g] )
                  {
                     scalar( column + g*WarpLanes, WarpLanes );
                     ++g;
                     continue;
                  }
                  const bool common = groups[g].AllActive();
                  if ( common && g + 1 < chunk && interior[g + 1] && groups[g + 1].AllActive() )
                  {
                     WarpCommonNeon<2>( groups + g, width, domainScale, groupOutput );
                     g += 2;
                     continue;
                  }
                  if ( common )
                     WarpCommonNeon<1>( groups + g, width, domainScale, groupOutput );
                  else
                     WarpExactNeon( groups[g], width, domainScale, groupOutput );
                  ++g;
               }
               column += chunk*WarpLanes;
            }
#endif
            for ( ; column < outputWidth; ++column )
            {
               double x, y;
               inputCoordinate( column, x, y );
               row[column] = pixel.Evaluate( x, y );
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

// Median of the finite per-pixel MADs in the same-row window
// [x - halfWidth, x + halfWidth] of consecutive pixels: the window's values
// are kept sorted and each step removes the column that leaves and inserts
// the one that enters, so the median is read from the middle instead of
// being selected anew.  The value is the same order statistic MedianOfFinite
// returns (Float32(low + high)/2 for an even count).
class PooledMadWindow
{
public:
   PooledMadWindow( const std::vector<float>& madMap,
                    std::uint32_t width,
                    std::size_t halfWidth )
      : m_madMap( madMap ), m_width( width ), m_halfWidth( halfWidth )
   {
      m_sorted.reserve( 2*halfWidth + 2 );
   }

   float MedianAt( std::size_t pixel )
   {
      const std::size_t x = pixel % m_width;
      const std::size_t rowStart = pixel - x;
      if ( !m_valid || rowStart != m_rowStart || x < m_x || x - m_x > m_halfWidth )
         Rebuild( rowStart, x );
      else
         while ( m_x < x )
            Step();
      const std::size_t count = m_sorted.size();
      const std::size_t half = count/2;
      const float upper = m_sorted[half];
      if ( (count & 1U) != 0 )
         return upper;
      const float sum = m_sorted[half - 1] + upper;
      return sum/2.0F;
   }

private:
   void Insert( float value )
   {
      if ( std::isnan( value ) )
         return;
      m_sorted.insert( std::upper_bound( m_sorted.begin(), m_sorted.end(), value ), value );
   }

   void Remove( float value )
   {
      if ( std::isnan( value ) )
         return;
      m_sorted.erase( std::lower_bound( m_sorted.begin(), m_sorted.end(), value ) );
   }

   void Rebuild( std::size_t rowStart, std::size_t x )
   {
      m_sorted.clear();
      m_rowStart = rowStart;
      m_x = x;
      const std::size_t first = x > m_halfWidth ? x - m_halfWidth : 0;
      const std::size_t last = std::min<std::size_t>( m_width - 1, x + m_halfWidth );
      for ( std::size_t column = first; column <= last; ++column )
         Insert( m_madMap[rowStart + column] );
      m_valid = true;
   }

   // Move the window from column m_x to m_x + 1.
   void Step()
   {
      if ( m_x >= m_halfWidth )
         Remove( m_madMap[m_rowStart + m_x - m_halfWidth] );
      ++m_x;
      if ( m_x + m_halfWidth < m_width )
         Insert( m_madMap[m_rowStart + m_x + m_halfWidth] );
   }

   const std::vector<float>& m_madMap;
   std::uint32_t m_width;
   std::size_t m_halfWidth;
   std::vector<float> m_sorted;
   std::size_t m_rowStart = 0;
   std::size_t m_x = 0;
   bool m_valid = false;
};

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
         PooledMadWindow window( madMap, width, halfWidth );
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
               const float pooledMad = window.MedianAt( pixel );
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
                  pixelCenter = SortedMedianOfFinite( scratch.data(), finiteCount );
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
               float mad;
               if ( finiteCount <= SortedMedianMaximumCount )
                  // ``scratch`` holds the ascending finite sample.
                  mad = SortedMadOfFinite( scratch.data(), finiteCount, pixelCenter );
               else
               {
                  std::uint32_t deviationCount = 0;
                  for ( std::uint32_t frame = 0; frame < frames; ++frame )
                     if ( std::isfinite( values[frame] ) )
                     {
                        const float deviation = values[frame] - pixelCenter;
                        scratch[deviationCount++] = std::fabs( deviation );
                     }
                  mad = MedianOfFinite( scratch.data(), deviationCount );
               }
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
   if ( !frameMajorSampleWeights.empty()
     && frameMajorSampleWeights.size() != samples )
      throw std::invalid_argument(
         "masked mean sample weight count differs from the geometry" );
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
   const float* sampleWeights = request.frameMajorSampleWeights.empty()
      ? nullptr : request.frameMajorSampleWeights.data();
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
               const float* rowWeights =
                  sampleWeights == nullptr ? nullptr : sampleWeights + offset;
               for ( std::size_t i = 0; i < count; ++i )
               {
                  const float value = row[i];
                  const bool isAccepted = mask[i] != 0;
                  // Without sample weights this is exactly the previous
                  // arithmetic (effective == weight).
                  const double effective = rowWeights == nullptr
                     ? weight : weight*static_cast<double>( rowWeights[i] );
                  const double term = isAccepted
                     ? static_cast<double>( value )*effective : 0.0;
                  numerator[i] = numerator[i] + term;
                  denominator[i] = denominator[i] + (isAccepted ? effective : 0.0);
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

// np.median of an unordered Float64 sample: the middle order statistic, or
// the mean of the two middle ones, selected without sorting the sample.
double MedianOf( std::vector<double> values )
{
   const std::size_t count = values.size();
   const std::size_t half = count/2;
   const auto middle = values.begin() + static_cast<std::ptrdiff_t>( half );
   std::nth_element( values.begin(), middle, values.end() );
   const double upper = *middle;
   if ( (count & 1U) != 0 )
      return upper;
   const double lower = *std::max_element( values.begin(), middle );
   const double sum = lower + upper;
   return sum/2.0;
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

void RadonPeakRequest::Validate() const
{
   if ( width == 0 || height == 0 )
      throw std::invalid_argument( "radon peaks require a non-empty image" );
   if ( size < height || size < 2 || (size & (size - 1)) != 0 )
      throw std::invalid_argument(
         "radon canvas size must be a power of two not smaller than the image height" );
   if ( size > 32768 || width > 32768 )
      throw std::invalid_argument( "radon canvas is larger than the kernel supports" );
   if ( minimumRows < 2 || minimumRows > size || (minimumRows & (minimumRows - 1)) != 0 )
      throw std::invalid_argument(
         "radon minimum rows must be a power of two between 2 and the canvas size" );
   const std::size_t samples = CheckedMultiply( height, width, "radon image" );
   if ( image.size() != samples || weight.size() != samples )
      throw std::invalid_argument( "radon image and weight sizes do not match the geometry" );
   if ( !std::isfinite( detectionZ ) || !std::isfinite( minimumCoverage )
     || !std::isfinite( minimumCount ) || minimumCoverage < 0.0 || minimumCount < 0.0 )
      throw std::invalid_argument( "radon detection parameters are invalid" );
}

namespace
{

// One dyadic level of the fast Radon transform restricted to the lines that
// can touch the image: blocks whose rows reach into the image and padded
// columns [lo, hi).  Every other entry of the reference's dense
// (blocks, 2n-1, width + 2 size) array is an exact zero (its line never
// meets the image), and reading such an entry yields zero here.
struct RadonLevel
{
   std::uint32_t n = 1;
   std::uint32_t activeBlocks = 0;
   std::uint32_t shifts = 1;
   std::uint32_t lo = 0;
   std::uint32_t hi = 0;
   std::vector<float> sums;
   std::vector<std::uint16_t> counts;

   std::size_t RowLength() const noexcept { return hi - lo; }
   std::size_t Rows() const noexcept
   {
      return static_cast<std::size_t>( activeBlocks )*shifts;
   }
};

// np.median of a Float32 sample: the middle value, or Float32(low + high)/2.
// The sample is reordered.
float Float32Median( std::vector<float>& values )
{
   const std::size_t count = values.size();
   const std::size_t half = count/2;
   std::nth_element( values.begin(), values.begin() + static_cast<std::ptrdiff_t>( half ),
                     values.end() );
   const float upper = values[half];
   if ( (count & 1U) != 0 )
      return upper;
   const float lower = *std::max_element(
      values.begin(), values.begin() + static_cast<std::ptrdiff_t>( half ) );
   const float sum = lower + upper;
   return sum/2.0F;
}

constexpr std::size_t RadonRowGrain = 4;

// F_n[b, s, x] = F_h[2b, t, x] + F_h[2b+1, t, x + (s - t)] with h = n/2 and
// t = trunc(s/2), the reference recursion, on the restricted layout.
void NextRadonLevel( const RadonLevel& previous, RadonLevel& level,
                     std::uint32_t height, std::uint32_t width, std::uint32_t size,
                     std::uint32_t threads )
{
   const std::uint32_t n = previous.n*2;
   const std::uint32_t half = previous.n;
   level.n = n;
   level.activeBlocks = (height + n - 1)/n;
   level.shifts = 2*n - 1;
   level.lo = size - (n - 1);
   level.hi = std::min( size + width + n - 1, width + 2*size );
   level.sums.assign( level.Rows()*level.RowLength(), 0.0F );
   level.counts.assign( level.Rows()*level.RowLength(), 0 );
   const std::size_t rowLength = level.RowLength();
   const std::size_t previousLength = previous.RowLength();
   const std::int64_t previousLo = previous.lo;
   const std::int64_t previousHi = previous.hi;

   ParallelRange( level.Rows(), threads, RadonRowGrain,
      [&]( std::size_t begin, std::size_t end )
      {
         for ( std::size_t row = begin; row < end; ++row )
         {
            const std::uint32_t block = static_cast<std::uint32_t>( row/level.shifts );
            const std::uint32_t shiftIndex = static_cast<std::uint32_t>( row % level.shifts );
            const std::int64_t shift =
               static_cast<std::int64_t>( shiftIndex ) - static_cast<std::int64_t>( n - 1 );
            const std::int64_t trunc = shift/2; // C++ division truncates toward zero
            const std::int64_t delta = shift - trunc;
            const std::size_t previousRow =
               static_cast<std::size_t>( trunc + static_cast<std::int64_t>( half ) - 1 );
            const std::uint32_t topBlock = 2*block;
            const std::uint32_t bottomBlock = 2*block + 1;
            const float* top = nullptr;
            const std::uint16_t* topCount = nullptr;
            if ( topBlock < previous.activeBlocks )
            {
               const std::size_t offset =
                  (static_cast<std::size_t>( topBlock )*previous.shifts + previousRow)
                  *previousLength;
               top = previous.sums.data() + offset;
               topCount = previous.counts.data() + offset;
            }
            const float* bottom = nullptr;
            const std::uint16_t* bottomCount = nullptr;
            if ( bottomBlock < previous.activeBlocks )
            {
               const std::size_t offset =
                  (static_cast<std::size_t>( bottomBlock )*previous.shifts + previousRow)
                  *previousLength;
               bottom = previous.sums.data() + offset;
               bottomCount = previous.counts.data() + offset;
            }
            float* out = level.sums.data() + row*rowLength;
            std::uint16_t* outCount = level.counts.data() + row*rowLength;
            for ( std::size_t i = 0; i < rowLength; ++i )
            {
               const std::int64_t x = static_cast<std::int64_t>( level.lo + i );
               float topValue = 0.0F;
               std::uint16_t topN = 0;
               if ( top != nullptr && x >= previousLo && x < previousHi )
               {
                  const std::size_t j = static_cast<std::size_t>( x - previousLo );
                  topValue = top[j];
                  topN = topCount[j];
               }
               const std::int64_t xb = x + delta;
               float bottomValue = 0.0F;
               std::uint16_t bottomN = 0;
               if ( bottom != nullptr && xb >= previousLo && xb < previousHi )
               {
                  const std::size_t j = static_cast<std::size_t>( xb - previousLo );
                  bottomValue = bottom[j];
                  bottomN = bottomCount[j];
               }
               out[i] = topValue + bottomValue;
               outCount[i] = static_cast<std::uint16_t>( topN + bottomN );
            }
         }
      } );
}

// Standardised z of one level and its peaks (see RadonLinePeaks).
void CollectLevelPeaks( const RadonLevel& level, const RadonPeakRequest& request,
                        std::vector<float>& z, std::vector<float>& sample,
                        std::vector<float>& deviations, std::vector<RadonPeak>& peaks )
{
   const std::size_t entries = level.sums.size();
   const std::size_t rowLength = level.RowLength();
   z.assign( entries, 0.0F );
   // counts >= max(8, coverage*n): the reference compares Float32 counts with
   // a Python float, which NumPy casts to Float32.
   const double minimumCount = std::max(
      request.minimumCount, request.minimumCoverage*static_cast<double>( level.n ) );
   const float threshold = static_cast<float>( minimumCount );
   ParallelRange( level.Rows(), request.threads, RadonRowGrain,
      [&]( std::size_t begin, std::size_t end )
      {
         for ( std::size_t row = begin; row < end; ++row )
         {
            const float* sums = level.sums.data() + row*rowLength;
            const std::uint16_t* counts = level.counts.data() + row*rowLength;
            float* out = z.data() + row*rowLength;
            for ( std::size_t i = 0; i < rowLength; ++i )
            {
               const float count = static_cast<float>( counts[i] );
               if ( count >= threshold )
               {
                  const float root = std::sqrt( std::max( count, 1.0F ) );
                  out[i] = sums[i]/root;
               }
            }
         }
      } );
   sample.clear();
   for ( std::size_t i = 0; i < entries; ++i )
      if ( static_cast<float>( level.counts[i] ) >= threshold )
         sample.push_back( z[i] );
   if ( sample.size() >= request.minimumScaleSamples )
   {
      const float location = Float32Median( sample );
      deviations.resize( sample.size() );
      for ( std::size_t i = 0; i < sample.size(); ++i )
      {
         const float centered = sample[i] - location;
         deviations[i] = std::fabs( centered );
      }
      const double scale = 1.4826*static_cast<double>( Float32Median( deviations ) );
      if ( std::isfinite( scale ) && scale > 0.0 )
      {
         const float divisor = static_cast<float>( scale );
         for ( float& value : z )
            value = value/divisor;
      }
   }
   const std::size_t shifts = level.shifts;
   for ( std::size_t row = 0; row < level.Rows(); ++row )
   {
      const std::size_t block = row/shifts;
      const std::size_t shiftIndex = row % shifts;
      const float* values = z.data() + row*rowLength;
      for ( std::size_t i = 0; i < rowLength; ++i )
      {
         const float value = values[i];
         if ( !(value >= request.detectionZ) )
            continue;
         // Maximum of the (1, 5, 7) window in the same block; scipy's
         // reflect border mode never introduces values from outside the
         // clipped window, and entries outside the restricted layout are
         // zero, so clipping to the stored rows and columns is exact.
         const std::size_t firstShift = shiftIndex >= 2 ? shiftIndex - 2 : 0;
         const std::size_t lastShift = std::min( shifts - 1, shiftIndex + 2 );
         const std::size_t firstColumn = i >= 3 ? i - 3 : 0;
         const std::size_t lastColumn = std::min( rowLength - 1, i + 3 );
         bool peak = true;
         for ( std::size_t s = firstShift; s <= lastShift && peak; ++s )
         {
            const float* neighbours = z.data() + (block*shifts + s)*rowLength;
            for ( std::size_t c = firstColumn; c <= lastColumn; ++c )
               if ( neighbours[c] > value )
               {
                  peak = false;
                  break;
               }
         }
         if ( peak )
            peaks.push_back( RadonPeak{ level.n, static_cast<std::uint32_t>( block ),
                                        static_cast<std::uint32_t>( shiftIndex ),
                                        static_cast<std::uint32_t>( level.lo + i ), value } );
      }
   }
}

} // namespace

void RadonLinePeaks( const RadonPeakRequest& request, std::vector<RadonPeak>& peaks )
{
   request.Validate();
   const std::uint32_t threads = std::max( 1U, request.threads );
   RadonLevel previous;
   previous.n = 1;
   previous.activeBlocks = request.height;
   previous.shifts = 1;
   previous.lo = request.size;
   previous.hi = request.size + request.width;
   previous.sums.assign( request.image.begin(), request.image.end() );
   previous.counts.resize( request.weight.size() );
   for ( std::size_t i = 0; i < request.weight.size(); ++i )
      previous.counts[i] = request.weight[i] != 0 ? 1 : 0;
   RadonLevel level;
   std::vector<float> z;
   std::vector<float> sample;
   std::vector<float> deviations;
   while ( previous.n < request.size )
   {
      NextRadonLevel( previous, level, request.height, request.width, request.size, threads );
      if ( level.n >= request.minimumRows )
         CollectLevelPeaks( level, request, z, sample, deviations, peaks );
      std::swap( previous, level );
   }
}

std::uint32_t DefaultKernelThreads() noexcept
{
   const unsigned concurrency = std::thread::hardware_concurrency();
   if ( concurrency == 0 )
      return 1;
   return static_cast<std::uint32_t>( std::min( 64U, concurrency ) );
}

} // namespace ufwbpp::native
