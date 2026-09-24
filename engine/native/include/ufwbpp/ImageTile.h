#ifndef UFWBPP_NATIVE_IMAGETILE_H
#define UFWBPP_NATIVE_IMAGETILE_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ufwbpp::native
{

struct ImageGeometry
{
   std::uint32_t width = 0;
   std::uint32_t height = 0;
   std::uint32_t channels = 1;

   void Validate() const;
   std::size_t SampleCount() const;
};

struct TileRegion
{
   ImageGeometry image;
   std::uint32_t firstRow = 0;
   std::uint32_t rowCount = 0;

   void Validate() const;
   std::size_t PixelCount() const;
};

std::vector<TileRegion> PartitionRows(
   const ImageGeometry& image,
   std::uint32_t maximumRows );

} // namespace ufwbpp::native

#endif // UFWBPP_NATIVE_IMAGETILE_H
