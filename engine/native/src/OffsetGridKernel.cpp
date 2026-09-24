#include "ufwbpp/PortableKernels.h"
#include "ParallelRange.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ufwbpp::native
{

using detail::ParallelRange;

namespace
{

constexpr std::size_t OffsetGridRowGrain = 16;

// np.clip, then np.searchsorted(side="right") clipped to [1, n - 1]: the
// enclosing node pair and the Float64 weight of the upper node.
void Locate( std::span<const double> nodes, double value, std::size_t& lower,
             std::size_t& upper, double& weight )
{
   const double clipped = std::min( std::max( value, nodes.front() ), nodes.back() );
   std::size_t index = static_cast<std::size_t>(
      std::upper_bound( nodes.begin(), nodes.end(), clipped ) - nodes.begin() );
   index = std::min( std::max<std::size_t>( index, 1 ), nodes.size() - 1 );
   lower = index - 1;
   upper = index;
   weight = (clipped - nodes[lower])/(nodes[upper] - nodes[lower]);
}

} // namespace

void OffsetGridRequest::Validate() const
{
   if ( width == 0 )
      throw std::invalid_argument( "offset grid rows must be nonempty" );
   if ( values.size() != rows.size()*static_cast<std::size_t>( width ) )
      throw std::invalid_argument( "offset grid values do not match rows x width" );
   if ( xNodes.size() < 2 || yNodes.size() < 2 || grid.size() != xNodes.size()*yNodes.size() )
      throw std::invalid_argument( "offset grid geometry is invalid" );
   for ( std::span<const double> nodes : { xNodes, yNodes } )
      for ( std::size_t index = 0; index < nodes.size(); ++index )
         if ( !std::isfinite( nodes[index] ) || (index > 0 && !(nodes[index] > nodes[index - 1])) )
            throw std::invalid_argument( "offset grid nodes must be finite and increasing" );
   for ( double value : grid )
      if ( !std::isfinite( value ) )
         throw std::invalid_argument( "offset grid values must be finite" );
   if ( threads < 1 )
      throw std::invalid_argument( "offset grid threads must be positive" );
}

void AddOffsetGrid( const OffsetGridRequest& request )
{
   request.Validate();
   const std::size_t width = request.width;
   const std::size_t columns = request.xNodes.size();
   // The horizontal plan of every column (calibration._offset_grid_x_plan).
   std::vector<std::size_t> xLower( width ), xUpper( width );
   std::vector<double> xWeight( width ), xComplement( width );
   for ( std::size_t x = 0; x < width; ++x )
   {
      Locate( request.xNodes, static_cast<double>( x ), xLower[x], xUpper[x], xWeight[x] );
      xComplement[x] = 1.0 - xWeight[x];
   }
   const double* grid = request.grid.data();
   ParallelRange( request.rows.size(), request.threads, OffsetGridRowGrain,
      [&]( std::size_t rowBegin, std::size_t rowEnd )
      {
         // Horizontal interpolations of the last two node rows used.
         std::vector<double> top( width ), bottom( width );
         std::size_t topNode = request.yNodes.size(), bottomNode = request.yNodes.size();
         const auto horizontal = [&]( std::size_t node, std::vector<double>& target )
         {
            const double* values = grid + node*columns;
            for ( std::size_t x = 0; x < width; ++x )
               target[x] = values[xLower[x]]*xComplement[x] + values[xUpper[x]]*xWeight[x];
         };
         for ( std::size_t row = rowBegin; row < rowEnd; ++row )
         {
            std::size_t lower, upper;
            double wy;
            Locate( request.yNodes, static_cast<double>( request.rows[row] ), lower, upper, wy );
            if ( lower != topNode )
            {
               // Moving down one node interval reuses the previous bottom row.
               if ( lower == bottomNode )
               {
                  top.swap( bottom );
                  std::swap( topNode, bottomNode );
               }
               else
               {
                  horizontal( lower, top );
                  topNode = lower;
               }
            }
            if ( upper != bottomNode )
            {
               horizontal( upper, bottom );
               bottomNode = upper;
            }
            const double complement = 1.0 - wy;
            float* values = request.values.data() + row*width;
            for ( std::size_t x = 0; x < width; ++x )
               values[x] = values[x] + static_cast<float>( top[x]*complement + bottom[x]*wy );
         }
      } );
}

} // namespace ufwbpp::native
