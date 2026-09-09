#include "openastroflow/ImageTile.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

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

} // namespace

void ImageGeometry::Validate() const
{
   if ( width == 0 || height == 0 || channels == 0 )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native image geometry must be positive" );
   (void)SampleCount();
}

std::size_t ImageGeometry::SampleCount() const
{
   return CheckedMultiply(
      CheckedMultiply( width, height, "Ultra-Fast WBPP native image area overflow" ),
      channels, "Ultra-Fast WBPP native image sample-count overflow" );
}

void TileRegion::Validate() const
{
   image.Validate();
   if ( rowCount == 0 || firstRow >= image.height
     || rowCount > image.height - firstRow )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native tile is outside its image" );
   (void)PixelCount();
}

std::size_t TileRegion::PixelCount() const
{
   return CheckedMultiply(
      CheckedMultiply( image.width, rowCount,
                       "Ultra-Fast WBPP native tile area overflow" ),
      image.channels, "Ultra-Fast WBPP native tile sample-count overflow" );
}

std::vector<TileRegion> PartitionRows(
   const ImageGeometry& image,
   std::uint32_t maximumRows )
{
   image.Validate();
   if ( maximumRows == 0 )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native maximum tile rows must be positive" );

   std::vector<TileRegion> result;
   for ( std::uint32_t first = 0; first < image.height; )
   {
      const std::uint32_t rows =
         std::min( maximumRows, image.height - first );
      result.push_back( { image, first, rows } );
      first += rows;
   }
   return result;
}

} // namespace openastroflow::native
