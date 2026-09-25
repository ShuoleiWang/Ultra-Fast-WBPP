#include "ufwbpp/PortableKernels.h"
#include "ParallelRange.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ufwbpp::native
{

using detail::ParallelRange;

namespace
{

constexpr std::size_t DebayerRowGrain = 16;

} // namespace

void DebayerRequest::Validate() const
{
   if ( width == 0 || height == 0 )
      throw std::invalid_argument( "debayer requires a nonempty mosaic" );
   const std::size_t pixels = static_cast<std::size_t>( width )*height;
   if ( mosaic.size() != pixels )
      throw std::invalid_argument( "debayer mosaic sample count differs from its geometry" );
   if ( planes.size() != 3*pixels )
      throw std::invalid_argument( "debayer output must hold three planes of the mosaic's size" );
   for ( std::uint8_t value : pattern )
      if ( value > 2 )
         throw std::invalid_argument( "debayer pattern channels must be 0, 1 or 2" );
   if ( threads < 1 )
      throw std::invalid_argument( "debayer threads must be positive" );
}

// Mirrors lightframeqc.cfa.bilinear_debayer: an unknown pixel of a colour
// plane is the Float64 mean of its known 4-neighbours, summed as
// (left + right) + (up + down); a pixel with no known 4-neighbour takes the
// mean of its known diagonal neighbours, summed as (ul + ur) + (dl + dr); a
// pixel with none is NaN.  Neighbour coordinates clamp to the frame (NumPy's
// edge padding), so at an edge a "diagonal" can be a row neighbour, exactly
// as in the reference.  Non-finite samples are unknown.  Known samples are
// copied.  Every value is rounded to Float32 once, at the end.
void DebayerBilinear( const DebayerRequest& request )
{
   request.Validate();
   const std::int64_t width = request.width;
   const std::int64_t height = request.height;
   const std::size_t stride = request.width;
   const float* mosaic = request.mosaic.data();
   const std::size_t planeSize = stride*request.height;
   const float nan = std::numeric_limits<float>::quiet_NaN();

   auto channelAt = [&]( std::int64_t y, std::int64_t x ) -> std::uint8_t
   {
      return request.pattern[((y & 1) << 1) | (x & 1)];
   };
   auto known = [&]( std::int64_t y, std::int64_t x, std::uint8_t channel, double& value ) -> bool
   {
      const float sample = mosaic[static_cast<std::size_t>( y )*stride + static_cast<std::size_t>( x )];
      if ( channelAt( y, x ) != channel || !std::isfinite( sample ) )
      {
         value = 0.0;
         return false;
      }
      value = static_cast<double>( sample );
      return true;
   };

   ParallelRange( static_cast<std::size_t>( height ), request.threads, DebayerRowGrain,
      [&]( std::size_t rowBegin, std::size_t rowEnd )
      {
         for ( std::size_t row = rowBegin; row < rowEnd; ++row )
         {
            const std::int64_t y = static_cast<std::int64_t>( row );
            const std::int64_t up = std::max<std::int64_t>( y - 1, 0 );
            const std::int64_t down = std::min<std::int64_t>( y + 1, height - 1 );
            for ( std::int64_t x = 0; x < width; ++x )
            {
               const std::int64_t left = std::max<std::int64_t>( x - 1, 0 );
               const std::int64_t right = std::min<std::int64_t>( x + 1, width - 1 );
               const std::size_t index = static_cast<std::size_t>( y )*stride + static_cast<std::size_t>( x );
               for ( std::uint8_t channel = 0; channel < 3; ++channel )
               {
                  float* plane = request.planes.data() + channel*planeSize;
                  double own;
                  if ( known( y, x, channel, own ) )
                  {
                     plane[index] = static_cast<float>( own );
                     continue;
                  }
                  double l, r, u, d;
                  const int count = static_cast<int>( known( y, left, channel, l ) )
                                  + static_cast<int>( known( y, right, channel, r ) )
                                  + static_cast<int>( known( up, x, channel, u ) )
                                  + static_cast<int>( known( down, x, channel, d ) );
                  if ( count > 0 )
                  {
                     const double total = (l + r) + (u + d);
                     plane[index] = static_cast<float>( total/count );
                     continue;
                  }
                  double ul, ur, dl, dr;
                  const int corners = static_cast<int>( known( up, left, channel, ul ) )
                                    + static_cast<int>( known( up, right, channel, ur ) )
                                    + static_cast<int>( known( down, left, channel, dl ) )
                                    + static_cast<int>( known( down, right, channel, dr ) );
                  if ( corners > 0 )
                  {
                     const double total = (ul + ur) + (dl + dr);
                     plane[index] = static_cast<float>( total/corners );
                  }
                  else
                     plane[index] = nan;
               }
            }
         }
      } );
}

} // namespace ufwbpp::native
