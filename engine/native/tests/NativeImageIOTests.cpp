#include "openastroflow/NativeImageIO.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace
{

using openastroflow::native::FitsKeyword;
using openastroflow::native::FitsMonoReader;
using openastroflow::native::FitsReadTransform;
using openastroflow::native::FitsWriteOptions;
using openastroflow::native::WriteFitsFloat32Mono;
using openastroflow::native::XisfFloat32MonoReader;

void Expect( bool condition, const std::string& message )
{
   if ( !condition )
      throw std::runtime_error( message );
}

void ExpectNear( float actual,
                 float expected,
                 float tolerance,
                 const std::string& role )
{
   if ( !std::isfinite( actual )
     || std::abs( actual - expected ) > tolerance )
      throw std::runtime_error( role + " mismatch" );
}

std::string Card( const std::string& text )
{
   if ( text.size() > 80 )
      throw std::runtime_error( "test FITS card is too long" );
   std::string result = text;
   result.resize( 80, ' ' );
   return result;
}

void WriteUnsigned16Fixture( const std::filesystem::path& path )
{
   std::string header;
   header += Card( "SIMPLE  =                    T" );
   header += Card( "BITPIX  =                   16" );
   header += Card( "NAXIS   =                    2" );
   header += Card( "NAXIS1  =                    3" );
   header += Card( "NAXIS2  =                    2" );
   header += Card( "BSCALE  =                    1" );
   header += Card( "BZERO   =                32768" );
   header += Card( "IMAGETYP= 'LIGHT' / deterministic fixture" );
   header += Card( "FILTER  = 'B'" );
   header += Card( "END" );
   header.resize( 2880, ' ' );

   const std::array<std::uint16_t, 6> physical = {
      0, 1, 32767, 32768, 65534, 65535
   };
   std::vector<std::byte> payload( 2880 );
   for ( std::size_t i = 0; i < physical.size(); ++i )
   {
      const std::int32_t stored =
         static_cast<std::int32_t>( physical[i] ) - 32768;
      const std::uint16_t bits = static_cast<std::uint16_t>( stored );
      payload[2*i] = static_cast<std::byte>( bits >> 8 );
      payload[2*i + 1] = static_cast<std::byte>( bits & 0xff );
   }

   std::ofstream stream( path, std::ios::binary | std::ios::trunc );
   stream.write( header.data(), static_cast<std::streamsize>( header.size() ) );
   stream.write( reinterpret_cast<const char*>( payload.data() ),
                 static_cast<std::streamsize>( payload.size() ) );
   if ( !stream )
      throw std::runtime_error( "failed to write deterministic FITS fixture" );
}

void TestUnsigned16Read( const std::filesystem::path& directory )
{
   const std::filesystem::path path = directory/"unsigned16.fits";
   WriteUnsigned16Fixture( path );
   FitsMonoReader reader( path );
   const auto& info = reader.Info();
   Expect( info.width == 3 && info.height == 2,
           "16-bit fixture geometry" );
   Expect( info.bitpix == 16, "16-bit fixture BITPIX" );
   Expect( info.bscale == 1 && info.bzero == 32768,
           "16-bit fixture scale" );
   Expect( info.HeaderString( "imagetyp" ) == "LIGHT",
           "case-insensitive header lookup" );

   const std::vector<float> physical = reader.ReadAll();
   const std::array<float, 6> expected = {
      0, 1, 32767, 32768, 65534, 65535
   };
   Expect( std::equal( physical.begin(), physical.end(), expected.begin() ),
           "16-bit physical sample decode" );

   const std::vector<float> secondRow = reader.ReadRows(
      1, 1, FitsReadTransform::Unsigned16ToUnit() );
   ExpectNear( secondRow[0], 32768.0f/65535.0f, 1e-7f,
               "normalized first sample" );
   ExpectNear( secondRow[1], 65534.0f/65535.0f, 1e-7f,
               "normalized second sample" );
   ExpectNear( secondRow[2], 1, 0, "normalized final sample" );
}

void TestFloat32WriteRead( const std::filesystem::path& directory )
{
   const std::filesystem::path path = directory/"float32.fits";
   const std::vector<float> samples = {
      -2.5f, -0.0f, 0.125f, 1.0f, 1024.25f,
      std::numeric_limits<float>::quiet_NaN()
   };
   FitsWriteOptions options;
   options.keywords = {
      FitsKeyword{ "IMAGETYP", std::string( "MASTER LIGHT" ),
                   "native integration output" },
      FitsKeyword{ "FILTER", std::string( "B" ), {} },
      FitsKeyword{ "OBJECT", std::string( "M42's core" ), {} },
      FitsKeyword{ "EXPTIME", 300.0, "seconds" },
      FitsKeyword{ "NCOMBINE", std::int64_t( 40 ), {} }
   };
   WriteFitsFloat32Mono( path, 3, 2, samples, options );

   {
      std::ifstream raw( path, std::ios::binary );
      raw.seekg( 2880 );
      std::array<unsigned char, 4> first = {};
      raw.read( reinterpret_cast<char*>( first.data() ), first.size() );
      Expect( raw.good()
           && first == std::array<unsigned char, 4>{ 0xc0, 0x20, 0, 0 },
              "float32 FITS payload is not independently big-endian" );
   }

   FitsMonoReader reader( path );
   const auto& info = reader.Info();
   Expect( info.width == 3 && info.height == 2 && info.bitpix == -32,
           "float32 output geometry/BITPIX" );
   Expect( info.HeaderString( "OBJECT" ) == "M42's core",
           "quoted FITS string round trip" );
   Expect( info.HeaderString( "FILTER" ) == "B",
           "metadata round trip" );
   const std::vector<float> decoded = reader.ReadAll();
   Expect( decoded.size() == samples.size(), "float32 sample count" );
   for ( std::size_t i = 0; i + 1 < samples.size(); ++i )
      Expect( std::bit_cast<std::uint32_t>( decoded[i] )
                 == std::bit_cast<std::uint32_t>( samples[i] ),
              "float32 bit-exact round trip at sample "
                 + std::to_string( i ) );
   Expect( std::isnan( decoded.back() ), "float32 NaN round trip" );

   bool refusedOverwrite = false;
   try
   {
      WriteFitsFloat32Mono( path, 3, 2, samples, options );
   }
   catch ( const std::exception& )
   {
      refusedOverwrite = true;
   }
   Expect( refusedOverwrite, "default writer must refuse overwrite" );

   options.overwrite = true;
   WriteFitsFloat32Mono( path, 3, 2, samples, options );
   FitsMonoReader overwritten( path );
   Expect( overwritten.Info().bitpix == -32,
           "atomic overwrite result" );
}

void WriteFloat32XisfFixture( const std::filesystem::path& path,
                              const std::vector<float>& samples )
{
   constexpr std::size_t attachmentOffset = 4096;
   const std::string xml =
      "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
      "<xisf version=\"1.0\">"
      "<Image id=\"integration\" geometry=\"3:2:1\" "
      "sampleFormat=\"Float32\" bounds=\"0:1\" "
      "imageType=\"MasterBias\" colorSpace=\"Gray\" "
      "location=\"attachment:4096:24\"/>"
      "</xisf>";
   Expect( 16 + xml.size() <= attachmentOffset,
           "synthetic XISF header fits before attachment" );

   std::vector<std::byte> bytes( attachmentOffset + samples.size()*4 );
   const std::string signature = "XISF0100";
   std::copy( signature.begin(), signature.end(),
              reinterpret_cast<char*>( bytes.data() ) );
   const std::uint64_t headerBytes = xml.size();
   for ( unsigned i = 0; i < 8; ++i )
      bytes[8 + i] = static_cast<std::byte>( headerBytes >> (8*i) );
   std::copy( xml.begin(), xml.end(),
              reinterpret_cast<char*>( bytes.data() + 16 ) );
   for ( std::size_t i = 0; i < samples.size(); ++i )
   {
      const std::uint32_t bits =
         std::bit_cast<std::uint32_t>( samples[i] );
      for ( unsigned byte = 0; byte < 4; ++byte )
         bytes[attachmentOffset + 4*i + byte] =
            static_cast<std::byte>( bits >> (8*byte) );
   }
   std::ofstream stream( path, std::ios::binary | std::ios::trunc );
   stream.write( reinterpret_cast<const char*>( bytes.data() ),
                 static_cast<std::streamsize>( bytes.size() ) );
   if ( !stream )
      throw std::runtime_error( "failed to write deterministic XISF fixture" );
}

void TestFloat32XisfRead( const std::filesystem::path& directory )
{
   const std::filesystem::path path = directory/"float32.xisf";
   const std::vector<float> samples = {
      -0.0f, 0.125f, 0.5f, 1.0f, 16.25f,
      std::numeric_limits<float>::quiet_NaN()
   };
   WriteFloat32XisfFixture( path, samples );
   XisfFloat32MonoReader reader( path );
   const auto& info = reader.Info();
   Expect( info.width == 3 && info.height == 2,
           "XISF fixture geometry" );
   Expect( info.dataOffset == 4096 && info.dataBytes == 24,
           "XISF attachment bounds" );
   Expect( info.littleEndian && info.id == "integration"
        && info.imageType == "MasterBias" && info.colorSpace == "Gray",
           "XISF image attributes" );
   const std::vector<float> decoded = reader.ReadAll();
   Expect( decoded.size() == samples.size(), "XISF sample count" );
   for ( std::size_t i = 0; i + 1 < samples.size(); ++i )
      Expect( std::bit_cast<std::uint32_t>( decoded[i] )
                 == std::bit_cast<std::uint32_t>( samples[i] ),
              "XISF bit-exact decode at sample " + std::to_string( i ) );
   Expect( std::isnan( decoded.back() ), "XISF NaN decode" );
   const std::vector<float> secondRow = reader.ReadRows( 1, 1 );
   Expect( secondRow.size() == 3 && secondRow.front() == 1.0f,
           "XISF row-tile decode" );
}

std::string JsonEscape( const std::string& text )
{
   std::string result;
   for ( unsigned char c : text )
   {
      switch ( c )
      {
      case '\\': result += "\\\\"; break;
      case '"': result += "\\\""; break;
      case '\n': result += "\\n"; break;
      case '\r': result += "\\r"; break;
      case '\t': result += "\\t"; break;
      default:
         if ( c < 0x20 )
         {
            const char* digits = "0123456789abcdef";
            result += "\\u00";
            result.push_back( digits[c >> 4] );
            result.push_back( digits[c & 0xf] );
         }
         else
            result.push_back( static_cast<char>( c ) );
      }
   }
   return result;
}

void VerifyRealLight( const std::filesystem::path& path )
{
   FitsMonoReader reader( path );
   const auto& info = reader.Info();
   Expect( info.width == 6252 && info.height == 4176,
           "real light geometry is not the expected QHY268M frame" );
   Expect( info.bitpix == 16, "real light is not BITPIX=16" );
   Expect( info.bscale == 1 && info.bzero == 32768,
           "real light does not use the expected unsigned-16 convention" );
   Expect( info.HeaderString( "IMAGETYP" ) == "LIGHT",
           "real frame is not tagged LIGHT" );
   Expect( !info.HeaderString( "FILTER" ).empty(),
           "real light has no FILTER" );

   const auto started = std::chrono::steady_clock::now();
   double decodeSeconds = 0;
   constexpr std::uint32_t rowsPerTile = 128;
   std::vector<float> tile;
   float minimum = std::numeric_limits<float>::infinity();
   float maximum = -std::numeric_limits<float>::infinity();
   double sum = 0;
   std::uint64_t count = 0;
   for ( std::uint32_t firstRow = 0; firstRow < info.height; )
   {
      const std::uint32_t rows = std::min(
         rowsPerTile, info.height - firstRow );
      tile.resize( static_cast<std::size_t>( info.width )*rows );
      const auto decodeStarted = std::chrono::steady_clock::now();
      reader.ReadRows( firstRow, rows, tile,
                       FitsReadTransform::Unsigned16ToUnit() );
      decodeSeconds += std::chrono::duration<double>(
         std::chrono::steady_clock::now() - decodeStarted ).count();
      for ( float sample : tile )
      {
         Expect( std::isfinite( sample ), "nonfinite real FITS sample" );
         minimum = std::min( minimum, sample );
         maximum = std::max( maximum, sample );
         sum += sample;
      }
      count += tile.size();
      firstRow += rows;
   }
   const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started ).count();
   Expect( minimum >= 0 && maximum <= 1,
           "real FITS normalized samples are outside [0,1]" );
   Expect( count == static_cast<std::uint64_t>( info.width )*info.height,
           "real FITS scan sample count" );

   std::cout << std::setprecision( 17 )
      << "{\n"
      << "  \"path\": \"" << JsonEscape( path.string() ) << "\",\n"
      << "  \"width\": " << info.width << ",\n"
      << "  \"height\": " << info.height << ",\n"
      << "  \"bitpix\": " << info.bitpix << ",\n"
      << "  \"bscale\": " << info.bscale << ",\n"
      << "  \"bzero\": " << info.bzero << ",\n"
      << "  \"imageType\": \""
      << JsonEscape( info.HeaderString( "IMAGETYP" ) ) << "\",\n"
      << "  \"filter\": \""
      << JsonEscape( info.HeaderString( "FILTER" ) ) << "\",\n"
      << "  \"object\": \""
      << JsonEscape( info.HeaderString( "OBJECT" ) ) << "\",\n"
      << "  \"exposure\": \""
      << JsonEscape( info.HeaderString( "EXPTIME" ) ) << "\",\n"
      << "  \"normalizedMinimum\": " << minimum << ",\n"
      << "  \"normalizedMaximum\": " << maximum << ",\n"
      << "  \"normalizedMean\": "
      << (sum/count) << ",\n"
      << "  \"scanSeconds\": " << seconds << ",\n"
      << "  \"scanMegapixelsPerSecond\": "
      << (static_cast<double>( count )/1e6)/seconds << ",\n"
      << "  \"decodeSeconds\": " << decodeSeconds << ",\n"
      << "  \"decodeMegapixelsPerSecond\": "
      << (static_cast<double>( count )/1e6)/decodeSeconds << ",\n"
      << "  \"gate\": true\n"
      << "}\n";
}

void VerifyRealXisf( const std::filesystem::path& path )
{
   XisfFloat32MonoReader reader( path );
   const auto& info = reader.Info();
   Expect( info.width == 6252 && info.height == 4176,
           "real XISF geometry is not the expected QHY268M frame" );
   const std::vector<float> tile = reader.ReadRows( 2048, 64 );
   const auto [minimum, maximum] = std::minmax_element(
      tile.begin(), tile.end() );
   Expect( minimum != tile.end() && std::isfinite( *minimum )
        && std::isfinite( *maximum ), "real XISF has nonfinite tile bounds" );
   std::cerr << std::setprecision( 9 )
      << "real XISF gate passed: " << info.width << 'x' << info.height
      << ", id=" << info.id << ", imageType=" << info.imageType
      << ", tileRange=[" << *minimum << ',' << *maximum << "]\n";
}

} // namespace

int main( int argc, char** argv )
{
   try
   {
      const std::filesystem::path directory =
         std::filesystem::temp_directory_path()/(
            "OpenAstroFlow native-image-io-tests-" + std::to_string( ::getpid() ) );
      std::filesystem::remove_all( directory );
      std::filesystem::create_directories( directory );
      try
      {
         TestUnsigned16Read( directory );
         TestFloat32WriteRead( directory );
         TestFloat32XisfRead( directory );
         if ( argc >= 2 )
            VerifyRealLight( argv[1] );
         else
            std::cout << "OpenAstroFlow native image I/O unit tests passed\n";
         if ( argc >= 3 )
            VerifyRealXisf( argv[2] );
      }
      catch ( ... )
      {
         std::filesystem::remove_all( directory );
         throw;
      }
      std::filesystem::remove_all( directory );
      return 0;
   }
   catch ( const std::exception& error )
   {
      std::cerr << error.what() << '\n';
      return 1;
   }
}
