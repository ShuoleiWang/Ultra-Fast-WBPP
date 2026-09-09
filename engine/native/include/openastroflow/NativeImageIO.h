#ifndef OPENASTROFLOW_NATIVE_NATIVEIMAGEIO_H
#define OPENASTROFLOW_NATIVE_NATIVEIMAGEIO_H

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace openastroflow::native
{

struct FitsHeaderCard
{
   std::string keyword;
   std::string rawValue;
   std::string comment;
};

struct FitsImageInfo
{
   std::uint32_t width = 0;
   std::uint32_t height = 0;
   int bitpix = 0;
   double bscale = 1;
   double bzero = 0;
   std::uint64_t dataOffset = 0;
   std::uint64_t dataBytes = 0;
   std::vector<FitsHeaderCard> cards;

   const FitsHeaderCard* FindCard( std::string_view keyword ) const;
   std::string HeaderString( std::string_view keyword ) const;
};

// The returned sample is:
// ((stored * BSCALE) + BZERO) * scale + offset.
struct FitsReadTransform
{
   double scale = 1;
   double offset = 0;

   static FitsReadTransform Unsigned16ToUnit();
};

// A read-only memory mapping keeps row-tile reads zero-copy before the required
// big-endian-to-native float conversion. Separate instances can be opened by
// independent pipeline workers; ReadRows itself is safe for concurrent calls.
class FitsMonoReader
{
public:
   explicit FitsMonoReader( const std::filesystem::path& path );
   ~FitsMonoReader();

   FitsMonoReader( const FitsMonoReader& ) = delete;
   FitsMonoReader& operator =( const FitsMonoReader& ) = delete;
   FitsMonoReader( FitsMonoReader&& ) noexcept;
   FitsMonoReader& operator =( FitsMonoReader&& ) noexcept;

   const FitsImageInfo& Info() const noexcept;

   void ReadRows( std::uint32_t firstRow,
                  std::uint32_t rowCount,
                  std::span<float> destination,
                  FitsReadTransform transform = {} ) const;

   std::vector<float> ReadRows(
      std::uint32_t firstRow,
      std::uint32_t rowCount,
      FitsReadTransform transform = {} ) const;

   std::vector<float> ReadAll(
      FitsReadTransform transform = {} ) const;

private:
   struct Impl;
   std::unique_ptr<Impl> impl_;
   FitsImageInfo info_;
};

struct XisfImageInfo
{
   std::uint32_t width = 0;
   std::uint32_t height = 0;
   std::uint64_t dataOffset = 0;
   std::uint64_t dataBytes = 0;
   bool littleEndian = true;
   std::string id;
   std::string imageType;
   std::string colorSpace;
};

// Minimal XISF path for portable intermediates and master calibration frames:
// first 2-D Gray Float32 image, uncompressed attachment storage.
// Unsupported compression is rejected explicitly instead of being misread.
class XisfFloat32MonoReader
{
public:
   explicit XisfFloat32MonoReader( const std::filesystem::path& path );
   ~XisfFloat32MonoReader();

   XisfFloat32MonoReader( const XisfFloat32MonoReader& ) = delete;
   XisfFloat32MonoReader& operator =( const XisfFloat32MonoReader& ) = delete;
   XisfFloat32MonoReader( XisfFloat32MonoReader&& ) noexcept;
   XisfFloat32MonoReader& operator =(
      XisfFloat32MonoReader&& ) noexcept;

   const XisfImageInfo& Info() const noexcept;

   void ReadRows( std::uint32_t firstRow,
                  std::uint32_t rowCount,
                  std::span<float> destination ) const;

   std::vector<float> ReadRows( std::uint32_t firstRow,
                                std::uint32_t rowCount ) const;
   std::vector<float> ReadAll() const;

private:
   struct Impl;
   std::unique_ptr<Impl> impl_;
   XisfImageInfo info_;
};

using FitsKeywordValue =
   std::variant<bool, std::int64_t, double, std::string>;

struct FitsKeyword
{
   std::string keyword;
   FitsKeywordValue value;
   std::string comment;
};

struct FitsWriteOptions
{
   std::vector<FitsKeyword> keywords;
   bool overwrite = false;
   bool syncToDisk = false;
};

// Writes a standards-compliant primary 2-D BITPIX=-32 HDU. Publication is
// atomic: bytes are completed in a sibling temporary file before the final
// path appears.
void WriteFitsFloat32Mono( const std::filesystem::path& path,
                           std::uint32_t width,
                           std::uint32_t height,
                           std::span<const float> samples,
                           const FitsWriteOptions& options = {} );

} // namespace openastroflow::native

#endif // OPENASTROFLOW_NATIVE_NATIVEIMAGEIO_H
