#ifndef OPENASTROFLOW_NATIVE_IMAGETILE_H
#define OPENASTROFLOW_NATIVE_IMAGETILE_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace openastroflow::native
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

} // namespace openastroflow::native

#endif // OPENASTROFLOW_NATIVE_IMAGETILE_H
