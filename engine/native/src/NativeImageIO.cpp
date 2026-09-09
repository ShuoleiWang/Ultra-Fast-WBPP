#include "openastroflow/NativeImageIO.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace openastroflow::native
{

namespace
{

constexpr std::uint64_t kFitsBlockBytes = 2880;
constexpr std::uint64_t kFitsCardBytes = 80;
constexpr std::uint64_t kMaximumHeaderBytes = 64*1024*1024;

std::string PathText( const std::filesystem::path& path )
{
   return path.string();
}

[[noreturn]] void Fail( const std::filesystem::path& path,
                        const std::string& message )
{
   throw std::runtime_error(
      "Ultra-Fast WBPP native FITS '" + PathText( path ) + "': " + message );
}

std::uint64_t CheckedMultiply( std::uint64_t left,
                               std::uint64_t right,
                               const std::filesystem::path& path,
                               const char* role )
{
   if ( left != 0
     && right > std::numeric_limits<std::uint64_t>::max()/left )
      Fail( path, role );
   return left*right;
}

std::string_view TrimView( std::string_view text )
{
   while ( !text.empty() && text.front() == ' ' )
      text.remove_prefix( 1 );
   while ( !text.empty() && text.back() == ' ' )
      text.remove_suffix( 1 );
   return text;
}

std::string UpperKeyword( std::string_view text )
{
   text = TrimView( text );
   std::string result;
   result.reserve( text.size() );
   for ( unsigned char c : text )
   {
      if ( c >= 'a' && c <= 'z' )
         c = static_cast<unsigned char>( c - 'a' + 'A' );
      result.push_back( static_cast<char>( c ) );
   }
   return result;
}

std::pair<std::string, std::string> SplitValueAndComment(
   std::string_view text )
{
   bool inString = false;
   for ( std::size_t i = 0; i < text.size(); ++i )
   {
      if ( text[i] == '\'' )
      {
         if ( inString && i + 1 < text.size() && text[i + 1] == '\'' )
         {
            ++i;
            continue;
         }
         inString = !inString;
      }
      else if ( text[i] == '/' && !inString )
      {
         return {
            std::string( TrimView( text.substr( 0, i ) ) ),
            std::string( TrimView( text.substr( i + 1 ) ) )
         };
      }
   }
   return { std::string( TrimView( text ) ), {} };
}

std::int64_t ParseInteger( const FitsImageInfo& info,
                           std::string_view keyword,
                           const std::filesystem::path& path )
{
   const FitsHeaderCard* card = info.FindCard( keyword );
   if ( card == nullptr )
      Fail( path, "missing required " + std::string( keyword ) + " card" );
   const std::string_view value = TrimView( card->rawValue );
   std::int64_t result = 0;
   const auto parsed = std::from_chars(
      value.data(), value.data() + value.size(), result );
   if ( parsed.ec != std::errc() || parsed.ptr != value.data() + value.size() )
      Fail( path, "invalid integer in " + std::string( keyword ) );
   return result;
}

double ParseDouble( const FitsHeaderCard* card,
                    double defaultValue,
                    std::string_view keyword,
                    const std::filesystem::path& path )
{
   if ( card == nullptr )
      return defaultValue;
   std::string value( TrimView( card->rawValue ) );
   std::replace( value.begin(), value.end(), 'D', 'E' );
   std::replace( value.begin(), value.end(), 'd', 'e' );
   char* end = nullptr;
   errno = 0;
   const double result = std::strtod( value.c_str(), &end );
   if ( errno != 0 || end == value.c_str() || *end != '\0'
     || !std::isfinite( result ) )
      Fail( path, "invalid number in " + std::string( keyword ) );
   return result;
}

std::uint64_t RoundUpFitsBlock( std::uint64_t size,
                                const std::filesystem::path& path )
{
   if ( size > std::numeric_limits<std::uint64_t>::max()
                 - (kFitsBlockBytes - 1) )
      Fail( path, "size overflow" );
   return ((size + kFitsBlockBytes - 1)/kFitsBlockBytes)*kFitsBlockBytes;
}

FitsImageInfo ParsePrimaryImage( const std::byte* bytes,
                                 std::size_t fileSize,
                                 const std::filesystem::path& path )
{
   if ( fileSize < kFitsBlockBytes )
      Fail( path, "file is smaller than one FITS block" );

   FitsImageInfo info;
   bool foundEnd = false;
   std::uint64_t endOffset = 0;
   const std::uint64_t parseLimit = std::min<std::uint64_t>(
      fileSize, kMaximumHeaderBytes );

   for ( std::uint64_t offset = 0;
         offset + kFitsCardBytes <= parseLimit;
         offset += kFitsCardBytes )
   {
      const char* cardBytes =
         reinterpret_cast<const char*>( bytes + offset );
      const std::string keyword = UpperKeyword(
         std::string_view( cardBytes, 8 ) );

      FitsHeaderCard card;
      card.keyword = keyword;
      if ( cardBytes[8] == '=' )
      {
         auto [value, comment] = SplitValueAndComment(
            std::string_view( cardBytes + 10, 70 ) );
         card.rawValue = std::move( value );
         card.comment = std::move( comment );
      }
      else if ( !keyword.empty() )
      {
         card.comment = std::string(
            TrimView( std::string_view( cardBytes + 8, 72 ) ) );
      }
      if ( !keyword.empty() )
         info.cards.push_back( std::move( card ) );

      if ( keyword == "END" )
      {
         foundEnd = true;
         endOffset = offset + kFitsCardBytes;
         break;
      }
   }

   if ( !foundEnd )
      Fail( path, "END card not found in bounded primary header" );
   const FitsHeaderCard* simple = info.FindCard( "SIMPLE" );
   if ( simple == nullptr || UpperKeyword( simple->rawValue ) != "T" )
      Fail( path, "primary HDU is not SIMPLE = T" );

   const std::int64_t naxis = ParseInteger( info, "NAXIS", path );
   const std::int64_t width = ParseInteger( info, "NAXIS1", path );
   const std::int64_t height = ParseInteger( info, "NAXIS2", path );
   const std::int64_t bitpix = ParseInteger( info, "BITPIX", path );
   if ( naxis != 2 )
      Fail( path, "only 2-D primary images are supported" );
   if ( width <= 0 || height <= 0
     || width > std::numeric_limits<std::uint32_t>::max()
     || height > std::numeric_limits<std::uint32_t>::max() )
      Fail( path, "invalid image dimensions" );
   if ( bitpix != 16 && bitpix != -32 )
      Fail( path, "supported BITPIX values are 16 and -32" );

   info.width = static_cast<std::uint32_t>( width );
   info.height = static_cast<std::uint32_t>( height );
   info.bitpix = static_cast<int>( bitpix );
   info.bscale = ParseDouble(
      info.FindCard( "BSCALE" ), 1, "BSCALE", path );
   info.bzero = ParseDouble(
      info.FindCard( "BZERO" ), 0, "BZERO", path );
   info.dataOffset = RoundUpFitsBlock( endOffset, path );

   const std::uint64_t bytesPerSample = bitpix == 16 ? 2 : 4;
   const std::uint64_t sampleCount = CheckedMultiply(
      static_cast<std::uint64_t>( width ),
      static_cast<std::uint64_t>( height ), path,
      "image sample-count overflow" );
   info.dataBytes = CheckedMultiply(
      sampleCount, bytesPerSample, path, "image byte-count overflow" );
   if ( info.dataOffset > fileSize
     || info.dataBytes > fileSize - info.dataOffset )
      Fail( path, "pixel payload is truncated" );
   return info;
}

bool IsPrintableAscii( std::string_view text )
{
   return std::all_of( text.begin(), text.end(), []( unsigned char c ) {
      return c >= 0x20 && c <= 0x7e;
   } );
}

std::string ValidateKeyword( std::string_view keyword )
{
   std::string result = UpperKeyword( keyword );
   if ( result.empty() || result.size() > 8 )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native FITS keyword must contain 1..8 characters" );
   for ( unsigned char c : result )
      if ( !(c >= 'A' && c <= 'Z')
        && !(c >= '0' && c <= '9') && c != '-' && c != '_' )
         throw std::invalid_argument(
            "Ultra-Fast WBPP native FITS keyword contains a nonstandard character" );
   return result;
}

std::string FormatStringValue( std::string_view value )
{
   if ( !IsPrintableAscii( value ) )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native FITS string values must be printable ASCII" );
   std::string result = "'";
   for ( char c : value )
   {
      result.push_back( c );
      if ( c == '\'' )
         result.push_back( '\'' );
   }
   result.push_back( '\'' );
   return result;
}

std::string FormatKeywordValue( const FitsKeywordValue& value )
{
   return std::visit( []( const auto& typed ) -> std::string {
      using Type = std::decay_t<decltype( typed )>;
      if constexpr ( std::is_same_v<Type, bool> )
         return typed ? "T" : "F";
      else if constexpr ( std::is_same_v<Type, std::int64_t> )
         return std::to_string( typed );
      else if constexpr ( std::is_same_v<Type, double> )
      {
         if ( !std::isfinite( typed ) )
            throw std::invalid_argument(
               "Ultra-Fast WBPP native FITS numeric header values must be finite" );
         std::ostringstream stream;
         stream << std::uppercase << std::setprecision( 17 ) << typed;
         return stream.str();
      }
      else
         return FormatStringValue( typed );
   }, value );
}

std::string FormatCard( std::string keyword,
                        const FitsKeywordValue& value,
                        std::string_view comment )
{
   keyword = ValidateKeyword( keyword );
   if ( !IsPrintableAscii( comment ) )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native FITS comments must be printable ASCII" );

   const std::string formattedValue = FormatKeywordValue( value );
   std::string card = keyword;
   card.resize( 8, ' ' );
   card += "= ";
   if ( !formattedValue.empty() && formattedValue.front() == '\'' )
      card += formattedValue;
   else
   {
      if ( formattedValue.size() < 20 )
         card.append( 20 - formattedValue.size(), ' ' );
      card += formattedValue;
   }
   if ( !comment.empty() )
      card += " / " + std::string( comment );
   if ( card.size() > kFitsCardBytes )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native FITS header card exceeds 80 bytes: " + keyword );
   card.resize( kFitsCardBytes, ' ' );
   return card;
}

void AppendCard( std::string& header,
                 const std::string& keyword,
                 FitsKeywordValue value,
                 const std::string& comment = {} )
{
   header += FormatCard( keyword, value, comment );
}

void WriteAll( int fd,
               const void* bytes,
               std::size_t size,
               const std::filesystem::path& path )
{
   const auto* cursor = static_cast<const std::byte*>( bytes );
   while ( size != 0 )
   {
      const ssize_t written = ::write( fd, cursor, size );
      if ( written < 0 )
      {
         if ( errno == EINTR )
            continue;
         Fail( path, "write failed: " +
                     std::string( std::strerror( errno ) ) );
      }
      if ( written == 0 )
         Fail( path, "write made no progress" );
      cursor += written;
      size -= static_cast<std::size_t>( written );
   }
}

struct TemporaryOutput
{
   int fd = -1;
   std::filesystem::path path;
   bool published = false;

   TemporaryOutput() = default;
   TemporaryOutput( int descriptor, std::filesystem::path temporaryPath )
      : fd( descriptor ), path( std::move( temporaryPath ) )
   {
   }

   TemporaryOutput( const TemporaryOutput& ) = delete;
   TemporaryOutput& operator =( const TemporaryOutput& ) = delete;

   TemporaryOutput( TemporaryOutput&& other ) noexcept
      : fd( other.fd )
      , path( std::move( other.path ) )
      , published( other.published )
   {
      other.fd = -1;
      other.published = true;
   }

   TemporaryOutput& operator =( TemporaryOutput&& ) = delete;

   ~TemporaryOutput()
   {
      if ( fd >= 0 )
         ::close( fd );
      if ( !published && !path.empty() )
         ::unlink( path.c_str() );
   }
};

TemporaryOutput CreateTemporaryOutput( const std::filesystem::path& target )
{
   static std::atomic<std::uint64_t> counter = 0;
   for ( unsigned attempt = 0; attempt < 100; ++attempt )
   {
      const std::filesystem::path path = target.string()
         + ".OpenAstroFlow native-tmp-" + std::to_string( ::getpid() )
         + "-" + std::to_string( counter.fetch_add( 1 ) + 1 );
      const int fd = ::open(
         path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644 );
      if ( fd >= 0 )
         return TemporaryOutput( fd, path );
      if ( errno != EEXIST )
         Fail( target, "cannot create temporary output: " +
                       std::string( std::strerror( errno ) ) );
   }
   Fail( target, "cannot allocate a unique temporary output name" );
}

std::uint32_t ToBigEndianBits( float sample )
{
   std::uint32_t bits = std::bit_cast<std::uint32_t>( sample );
   if constexpr ( std::endian::native == std::endian::little )
      bits = __builtin_bswap32( bits );
   return bits;
}

std::uint64_t ReadLittleUInt64( const std::byte* bytes )
{
   std::uint64_t result = 0;
   for ( unsigned i = 0; i < 8; ++i )
      result |= static_cast<std::uint64_t>( bytes[i] ) << (8*i);
   return result;
}

std::uint64_t ParseUnsignedText( std::string_view text,
                                 const std::filesystem::path& path,
                                 const char* role )
{
   text = TrimView( text );
   std::uint64_t result = 0;
   const auto parsed = std::from_chars(
      text.data(), text.data() + text.size(), result );
   if ( parsed.ec != std::errc()
     || parsed.ptr != text.data() + text.size() )
      Fail( path, std::string( "invalid XISF " ) + role );
   return result;
}

std::string XmlUnescape( std::string_view value,
                         const std::filesystem::path& path )
{
   std::string result;
   for ( std::size_t i = 0; i < value.size(); )
   {
      if ( value[i] != '&' )
      {
         result.push_back( value[i++] );
         continue;
      }
      const std::size_t end = value.find( ';', i + 1 );
      if ( end == std::string_view::npos )
         Fail( path, "unterminated XML entity in XISF Image attribute" );
      const std::string_view entity = value.substr( i, end - i + 1 );
      if ( entity == "&amp;" ) result.push_back( '&' );
      else if ( entity == "&quot;" ) result.push_back( '"' );
      else if ( entity == "&apos;" ) result.push_back( '\'' );
      else if ( entity == "&lt;" ) result.push_back( '<' );
      else if ( entity == "&gt;" ) result.push_back( '>' );
      else
         Fail( path, "unsupported XML entity in XISF Image attribute" );
      i = end + 1;
   }
   return result;
}

using XmlAttributes = std::vector<std::pair<std::string, std::string>>;

XmlAttributes ParseXmlAttributes( std::string_view tag,
                                  const std::filesystem::path& path )
{
   XmlAttributes result;
   std::size_t cursor = tag.find_first_of( " \t\r\n" );
   if ( cursor == std::string_view::npos )
      return result;
   while ( cursor < tag.size() )
   {
      while ( cursor < tag.size()
           && (tag[cursor] == ' ' || tag[cursor] == '\t'
            || tag[cursor] == '\r' || tag[cursor] == '\n') )
         ++cursor;
      if ( cursor >= tag.size() || tag[cursor] == '>'
        || tag[cursor] == '/' )
         break;
      const std::size_t nameStart = cursor;
      while ( cursor < tag.size()
           && tag[cursor] != '=' && tag[cursor] != ' '
           && tag[cursor] != '\t' && tag[cursor] != '\r'
           && tag[cursor] != '\n' )
         ++cursor;
      const std::string name( tag.substr( nameStart, cursor - nameStart ) );
      while ( cursor < tag.size()
           && (tag[cursor] == ' ' || tag[cursor] == '\t'
            || tag[cursor] == '\r' || tag[cursor] == '\n') )
         ++cursor;
      if ( name.empty() || cursor >= tag.size() || tag[cursor] != '=' )
         Fail( path, "malformed XISF Image attribute" );
      ++cursor;
      while ( cursor < tag.size()
           && (tag[cursor] == ' ' || tag[cursor] == '\t'
            || tag[cursor] == '\r' || tag[cursor] == '\n') )
         ++cursor;
      if ( cursor >= tag.size()
        || (tag[cursor] != '"' && tag[cursor] != '\'') )
         Fail( path, "unquoted XISF Image attribute" );
      const char quote = tag[cursor++];
      const std::size_t valueStart = cursor;
      const std::size_t valueEnd = tag.find( quote, cursor );
      if ( valueEnd == std::string_view::npos )
         Fail( path, "unterminated XISF Image attribute" );
      result.emplace_back(
         name, XmlUnescape(
            tag.substr( valueStart, valueEnd - valueStart ), path ) );
      cursor = valueEnd + 1;
   }
   return result;
}

const std::string* FindXmlAttribute( const XmlAttributes& attributes,
                                     std::string_view name )
{
   const auto iterator = std::find_if(
      attributes.begin(), attributes.end(), [&]( const auto& attribute ) {
         return attribute.first == name;
      } );
   return iterator == attributes.end() ? nullptr : &iterator->second;
}

std::string_view FindFirstImageTag( std::string_view xml,
                                    const std::filesystem::path& path )
{
   for ( std::size_t start = xml.find( "<Image" );
         start != std::string_view::npos;
         start = xml.find( "<Image", start + 6 ) )
   {
      if ( start + 6 < xml.size()
        && xml[start + 6] != ' ' && xml[start + 6] != '\t'
        && xml[start + 6] != '\r' && xml[start + 6] != '\n'
        && xml[start + 6] != '>' )
         continue;
      bool inQuote = false;
      char quote = 0;
      for ( std::size_t end = start + 6; end < xml.size(); ++end )
      {
         if ( inQuote )
         {
            if ( xml[end] == quote )
               inQuote = false;
         }
         else if ( xml[end] == '"' || xml[end] == '\'' )
         {
            inQuote = true;
            quote = xml[end];
         }
         else if ( xml[end] == '>' )
            return xml.substr( start, end - start + 1 );
      }
      Fail( path, "unterminated XISF Image element" );
   }
   Fail( path, "XISF header has no Image element" );
}

XisfImageInfo ParsePrimaryXisfImage( const std::byte* bytes,
                                     std::size_t fileSize,
                                     const std::filesystem::path& path )
{
   constexpr std::string_view signature = "XISF0100";
   if ( fileSize < 16
     || std::memcmp( bytes, signature.data(), signature.size() ) != 0 )
      Fail( path, "invalid XISF 1.0 signature" );
   const std::uint64_t headerBytes = ReadLittleUInt64( bytes + 8 );
   if ( headerBytes == 0 || headerBytes > kMaximumHeaderBytes
     || headerBytes > fileSize - 16 )
      Fail( path, "invalid XISF XML header length" );
   const std::string_view xml(
      reinterpret_cast<const char*>( bytes + 16 ),
      static_cast<std::size_t>( headerBytes ) );
   const XmlAttributes attributes = ParseXmlAttributes(
      FindFirstImageTag( xml, path ), path );
   const auto require = [&]( std::string_view name ) -> const std::string& {
      const std::string* value = FindXmlAttribute( attributes, name );
      if ( value == nullptr || value->empty() )
         Fail( path, "XISF Image lacks " + std::string( name ) );
      return *value;
   };

   if ( require( "sampleFormat" ) != "Float32" )
      Fail( path, "only XISF Float32 images are supported" );
   if ( const std::string* compression =
           FindXmlAttribute( attributes, "compression" );
        compression != nullptr && !compression->empty() )
      Fail( path, "compressed XISF attachments are not implemented" );
   if ( const std::string* color = FindXmlAttribute( attributes, "colorSpace" );
        color != nullptr && *color != "Gray" )
      Fail( path, "only XISF Gray images are supported" );

   const std::string& geometry = require( "geometry" );
   const std::size_t firstColon = geometry.find( ':' );
   const std::size_t secondColon = geometry.find(
      ':', firstColon == std::string::npos ? 0 : firstColon + 1 );
   if ( firstColon == std::string::npos || secondColon == std::string::npos
     || geometry.find( ':', secondColon + 1 ) != std::string::npos )
      Fail( path, "invalid XISF image geometry" );
   const std::uint64_t width = ParseUnsignedText(
      std::string_view( geometry ).substr( 0, firstColon ), path,
      "image width" );
   const std::uint64_t height = ParseUnsignedText(
      std::string_view( geometry ).substr(
         firstColon + 1, secondColon - firstColon - 1 ), path,
      "image height" );
   const std::uint64_t channels = ParseUnsignedText(
      std::string_view( geometry ).substr( secondColon + 1 ), path,
      "channel count" );
   if ( width == 0 || height == 0 || channels != 1
     || width > std::numeric_limits<std::uint32_t>::max()
     || height > std::numeric_limits<std::uint32_t>::max() )
      Fail( path, "unsupported XISF image geometry" );

   const std::string& location = require( "location" );
   constexpr std::string_view attachmentPrefix = "attachment:";
   if ( !std::string_view( location ).starts_with( attachmentPrefix ) )
      Fail( path, "only XISF attachment storage is supported" );
   const std::string_view locationData =
      std::string_view( location ).substr( attachmentPrefix.size() );
   const std::size_t locationColon = locationData.find( ':' );
   if ( locationColon == std::string_view::npos
     || locationData.find( ':', locationColon + 1 ) != std::string_view::npos )
      Fail( path, "invalid XISF attachment location" );

   XisfImageInfo info;
   info.width = static_cast<std::uint32_t>( width );
   info.height = static_cast<std::uint32_t>( height );
   info.dataOffset = ParseUnsignedText(
      locationData.substr( 0, locationColon ), path,
      "attachment offset" );
   info.dataBytes = ParseUnsignedText(
      locationData.substr( locationColon + 1 ), path,
      "attachment size" );
   const std::uint64_t expectedBytes = CheckedMultiply(
      CheckedMultiply( width, height, path,
                       "XISF sample-count overflow" ),
      sizeof( float ), path, "XISF byte-count overflow" );
   if ( info.dataBytes != expectedBytes )
      Fail( path, "XISF attachment size does not match geometry" );
   if ( info.dataOffset > fileSize
     || info.dataBytes > fileSize - info.dataOffset )
      Fail( path, "XISF attachment is truncated" );
   if ( const std::string* byteOrder =
           FindXmlAttribute( attributes, "byteOrder" );
        byteOrder != nullptr && !byteOrder->empty() )
   {
      if ( *byteOrder == "little" )
         info.littleEndian = true;
      else if ( *byteOrder == "big" )
         info.littleEndian = false;
      else
         Fail( path, "invalid XISF byteOrder" );
   }
   if ( const std::string* id = FindXmlAttribute( attributes, "id" ) )
      info.id = *id;
   if ( const std::string* imageType =
           FindXmlAttribute( attributes, "imageType" ) )
      info.imageType = *imageType;
   if ( const std::string* colorSpace =
           FindXmlAttribute( attributes, "colorSpace" ) )
      info.colorSpace = *colorSpace;
   return info;
}

} // namespace

struct FitsMonoReader::Impl
{
   std::filesystem::path path;
   int fd = -1;
   void* mapping = MAP_FAILED;
   std::size_t mappingBytes = 0;

   explicit Impl( const std::filesystem::path& source )
      : path( source )
   {
      try
      {
         fd = ::open( source.c_str(), O_RDONLY | O_CLOEXEC );
         if ( fd < 0 )
            Fail( source, "open failed: " +
                          std::string( std::strerror( errno ) ) );
         struct stat status = {};
         if ( ::fstat( fd, &status ) != 0 )
            Fail( source, "fstat failed: " +
                          std::string( std::strerror( errno ) ) );
         if ( status.st_size <= 0
           || static_cast<std::uint64_t>( status.st_size )
                 > std::numeric_limits<std::size_t>::max() )
            Fail( source, "invalid file size" );
         mappingBytes = static_cast<std::size_t>( status.st_size );
         mapping = ::mmap(
            nullptr, mappingBytes, PROT_READ, MAP_PRIVATE, fd, 0 );
         if ( mapping == MAP_FAILED )
            Fail( source, "mmap failed: " +
                          std::string( std::strerror( errno ) ) );
      }
      catch ( ... )
      {
         if ( mapping != MAP_FAILED )
            ::munmap( mapping, mappingBytes );
         if ( fd >= 0 )
            ::close( fd );
         throw;
      }
   }

   ~Impl()
   {
      if ( mapping != MAP_FAILED )
         ::munmap( mapping, mappingBytes );
      if ( fd >= 0 )
         ::close( fd );
   }
};

struct XisfFloat32MonoReader::Impl
{
   std::filesystem::path path;
   int fd = -1;
   void* mapping = MAP_FAILED;
   std::size_t mappingBytes = 0;

   explicit Impl( const std::filesystem::path& source )
      : path( source )
   {
      try
      {
         fd = ::open( source.c_str(), O_RDONLY | O_CLOEXEC );
         if ( fd < 0 )
            Fail( source, "open failed: " +
                          std::string( std::strerror( errno ) ) );
         struct stat status = {};
         if ( ::fstat( fd, &status ) != 0 )
            Fail( source, "fstat failed: " +
                          std::string( std::strerror( errno ) ) );
         if ( status.st_size <= 0
           || static_cast<std::uint64_t>( status.st_size )
                 > std::numeric_limits<std::size_t>::max() )
            Fail( source, "invalid file size" );
         mappingBytes = static_cast<std::size_t>( status.st_size );
         mapping = ::mmap(
            nullptr, mappingBytes, PROT_READ, MAP_PRIVATE, fd, 0 );
         if ( mapping == MAP_FAILED )
            Fail( source, "mmap failed: " +
                          std::string( std::strerror( errno ) ) );
      }
      catch ( ... )
      {
         if ( mapping != MAP_FAILED )
            ::munmap( mapping, mappingBytes );
         if ( fd >= 0 )
            ::close( fd );
         throw;
      }
   }

   ~Impl()
   {
      if ( mapping != MAP_FAILED )
         ::munmap( mapping, mappingBytes );
      if ( fd >= 0 )
         ::close( fd );
   }
};

const FitsHeaderCard* FitsImageInfo::FindCard(
   std::string_view keyword ) const
{
   const std::string normalized = UpperKeyword( keyword );
   const auto iterator = std::find_if(
      cards.begin(), cards.end(), [&]( const FitsHeaderCard& card ) {
         return card.keyword == normalized;
      } );
   return iterator == cards.end() ? nullptr : &*iterator;
}

std::string FitsImageInfo::HeaderString( std::string_view keyword ) const
{
   const FitsHeaderCard* card = FindCard( keyword );
   if ( card == nullptr )
      return {};
   const std::string_view raw = TrimView( card->rawValue );
   if ( raw.size() < 2 || raw.front() != '\'' )
      return std::string( raw );

   std::string result;
   for ( std::size_t i = 1; i < raw.size(); ++i )
   {
      if ( raw[i] != '\'' )
      {
         result.push_back( raw[i] );
         continue;
      }
      if ( i + 1 < raw.size() && raw[i + 1] == '\'' )
      {
         result.push_back( '\'' );
         ++i;
         continue;
      }
      return result;
   }
   return {};
}

FitsReadTransform FitsReadTransform::Unsigned16ToUnit()
{
   return { 1.0/65535.0, 0 };
}

FitsMonoReader::FitsMonoReader( const std::filesystem::path& path )
   : impl_( std::make_unique<Impl>( path ) )
   , info_( ParsePrimaryImage(
        static_cast<const std::byte*>( impl_->mapping ),
        impl_->mappingBytes, path ) )
{
}

FitsMonoReader::~FitsMonoReader() = default;
FitsMonoReader::FitsMonoReader( FitsMonoReader&& ) noexcept = default;
FitsMonoReader& FitsMonoReader::operator =( FitsMonoReader&& ) noexcept = default;

const FitsImageInfo& FitsMonoReader::Info() const noexcept
{
   return info_;
}

void FitsMonoReader::ReadRows( std::uint32_t firstRow,
                               std::uint32_t rowCount,
                               std::span<float> destination,
                               FitsReadTransform transform ) const
{
   if ( impl_ == nullptr )
      throw std::logic_error( "Ultra-Fast WBPP native FITS reader has been moved from" );
   if ( rowCount == 0 || firstRow >= info_.height
     || rowCount > info_.height - firstRow )
      throw std::out_of_range( "Ultra-Fast WBPP native FITS row range is invalid" );
   if ( !std::isfinite( transform.scale )
     || !std::isfinite( transform.offset ) )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native FITS read transform must be finite" );

   const std::uint64_t sampleCount64 =
      static_cast<std::uint64_t>( info_.width )*rowCount;
   if ( sampleCount64 > std::numeric_limits<std::size_t>::max()
     || destination.size() != static_cast<std::size_t>( sampleCount64 ) )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native FITS destination has the wrong sample count" );

   const std::uint64_t bytesPerSample = info_.bitpix == 16 ? 2 : 4;
   const std::uint64_t sampleOffset =
      static_cast<std::uint64_t>( firstRow )*info_.width;
   const auto* source = static_cast<const std::byte*>( impl_->mapping )
      + info_.dataOffset + sampleOffset*bytesPerSample;
   const float scale = static_cast<float>(
      info_.bscale*transform.scale );
   const float offset = static_cast<float>(
      info_.bzero*transform.scale + transform.offset );

   if ( info_.bitpix == 16 )
   {
      for ( std::size_t i = 0; i < destination.size(); ++i )
      {
         const std::uint16_t bits =
            (static_cast<std::uint16_t>( source[2*i] ) << 8)
            | static_cast<std::uint16_t>( source[2*i + 1] );
         const std::int32_t stored = bits < 0x8000u
            ? static_cast<std::int32_t>( bits )
            : static_cast<std::int32_t>( bits ) - 0x10000;
         destination[i] = static_cast<float>( stored )*scale + offset;
      }
   }
   else
   {
      const bool identity = scale == 1 && offset == 0;
      for ( std::size_t i = 0; i < destination.size(); ++i )
      {
         const std::uint32_t bits =
            (static_cast<std::uint32_t>( source[4*i] ) << 24)
          | (static_cast<std::uint32_t>( source[4*i + 1] ) << 16)
          | (static_cast<std::uint32_t>( source[4*i + 2] ) << 8)
          | static_cast<std::uint32_t>( source[4*i + 3] );
         const float stored = std::bit_cast<float>( bits );
         destination[i] = identity ? stored : stored*scale + offset;
      }
   }
}

std::vector<float> FitsMonoReader::ReadRows(
   std::uint32_t firstRow,
   std::uint32_t rowCount,
   FitsReadTransform transform ) const
{
   const std::uint64_t count =
      static_cast<std::uint64_t>( info_.width )*rowCount;
   if ( count > std::numeric_limits<std::size_t>::max() )
      throw std::overflow_error( "Ultra-Fast WBPP native FITS tile is too large" );
   std::vector<float> result( static_cast<std::size_t>( count ) );
   ReadRows( firstRow, rowCount, result, transform );
   return result;
}

std::vector<float> FitsMonoReader::ReadAll(
   FitsReadTransform transform ) const
{
   return ReadRows( 0, info_.height, transform );
}

XisfFloat32MonoReader::XisfFloat32MonoReader(
   const std::filesystem::path& path )
   : impl_( std::make_unique<Impl>( path ) )
   , info_( ParsePrimaryXisfImage(
        static_cast<const std::byte*>( impl_->mapping ),
        impl_->mappingBytes, path ) )
{
}

XisfFloat32MonoReader::~XisfFloat32MonoReader() = default;
XisfFloat32MonoReader::XisfFloat32MonoReader(
   XisfFloat32MonoReader&& ) noexcept = default;
XisfFloat32MonoReader& XisfFloat32MonoReader::operator =(
   XisfFloat32MonoReader&& ) noexcept = default;

const XisfImageInfo& XisfFloat32MonoReader::Info() const noexcept
{
   return info_;
}

void XisfFloat32MonoReader::ReadRows(
   std::uint32_t firstRow,
   std::uint32_t rowCount,
   std::span<float> destination ) const
{
   if ( impl_ == nullptr )
      throw std::logic_error( "Ultra-Fast WBPP native XISF reader has been moved from" );
   if ( rowCount == 0 || firstRow >= info_.height
     || rowCount > info_.height - firstRow )
      throw std::out_of_range( "Ultra-Fast WBPP native XISF row range is invalid" );
   const std::uint64_t sampleCount64 =
      static_cast<std::uint64_t>( info_.width )*rowCount;
   if ( sampleCount64 > std::numeric_limits<std::size_t>::max()
     || destination.size() != static_cast<std::size_t>( sampleCount64 ) )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native XISF destination has the wrong sample count" );

   const std::uint64_t sampleOffset =
      static_cast<std::uint64_t>( firstRow )*info_.width;
   const auto* source = static_cast<const std::byte*>( impl_->mapping )
      + info_.dataOffset + sampleOffset*sizeof( float );
   for ( std::size_t i = 0; i < destination.size(); ++i )
   {
      std::uint32_t bits = 0;
      if ( info_.littleEndian )
      {
         bits = static_cast<std::uint32_t>( source[4*i] )
          | (static_cast<std::uint32_t>( source[4*i + 1] ) << 8)
          | (static_cast<std::uint32_t>( source[4*i + 2] ) << 16)
          | (static_cast<std::uint32_t>( source[4*i + 3] ) << 24);
      }
      else
      {
         bits = (static_cast<std::uint32_t>( source[4*i] ) << 24)
          | (static_cast<std::uint32_t>( source[4*i + 1] ) << 16)
          | (static_cast<std::uint32_t>( source[4*i + 2] ) << 8)
          | static_cast<std::uint32_t>( source[4*i + 3] );
      }
      destination[i] = std::bit_cast<float>( bits );
   }
}

std::vector<float> XisfFloat32MonoReader::ReadRows(
   std::uint32_t firstRow,
   std::uint32_t rowCount ) const
{
   const std::uint64_t count =
      static_cast<std::uint64_t>( info_.width )*rowCount;
   if ( count > std::numeric_limits<std::size_t>::max() )
      throw std::overflow_error( "Ultra-Fast WBPP native XISF tile is too large" );
   std::vector<float> result( static_cast<std::size_t>( count ) );
   ReadRows( firstRow, rowCount, result );
   return result;
}

std::vector<float> XisfFloat32MonoReader::ReadAll() const
{
   return ReadRows( 0, info_.height );
}

void WriteFitsFloat32Mono( const std::filesystem::path& path,
                           std::uint32_t width,
                           std::uint32_t height,
                           std::span<const float> samples,
                           const FitsWriteOptions& options )
{
   if ( path.empty() )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native FITS output path must not be empty" );
   if ( width == 0 || height == 0 )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native FITS output dimensions must be positive" );
   const std::uint64_t expected =
      static_cast<std::uint64_t>( width )*height;
   if ( expected > std::numeric_limits<std::size_t>::max()
     || samples.size() != static_cast<std::size_t>( expected ) )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native FITS output has the wrong sample count" );

   const std::array<std::string_view, 9> reserved = {
      "SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2",
      "EXTEND", "ORIGIN", "ROWORDER", "END"
   };
   std::vector<std::string> seen;
   for ( const FitsKeyword& keyword : options.keywords )
   {
      const std::string normalized = ValidateKeyword( keyword.keyword );
      if ( std::find( reserved.begin(), reserved.end(), normalized )
             != reserved.end()
        || std::find( seen.begin(), seen.end(), normalized ) != seen.end() )
         throw std::invalid_argument(
            "Ultra-Fast WBPP native FITS duplicate or reserved keyword: " + normalized );
      seen.push_back( normalized );
   }

   std::string header;
   AppendCard( header, "SIMPLE", true,
               "conforms to FITS standard" );
   AppendCard( header, "BITPIX", std::int64_t( -32 ) );
   AppendCard( header, "NAXIS", std::int64_t( 2 ) );
   AppendCard( header, "NAXIS1", std::int64_t( width ) );
   AppendCard( header, "NAXIS2", std::int64_t( height ) );
   AppendCard( header, "EXTEND", true );
   AppendCard( header, "ORIGIN", std::string( "Ultra-Fast WBPP" ) );
   AppendCard( header, "ROWORDER", std::string( "TOP-DOWN" ) );
   for ( const FitsKeyword& keyword : options.keywords )
      header += FormatCard(
         keyword.keyword, keyword.value, keyword.comment );
   std::string end = "END";
   end.resize( kFitsCardBytes, ' ' );
   header += end;
   header.resize( static_cast<std::size_t>(
      RoundUpFitsBlock( header.size(), path ) ), ' ' );

   TemporaryOutput output = CreateTemporaryOutput( path );
   WriteAll( output.fd, header.data(), header.size(), output.path );

   constexpr std::size_t kChunkSamples = 1024*1024;
   std::vector<std::uint32_t> encoded(
      std::min<std::size_t>( kChunkSamples, samples.size() ) );
   for ( std::size_t first = 0; first < samples.size(); )
   {
      const std::size_t count = std::min(
         encoded.size(), samples.size() - first );
      for ( std::size_t i = 0; i < count; ++i )
         encoded[i] = ToBigEndianBits( samples[first + i] );
      WriteAll( output.fd, encoded.data(), count*sizeof( encoded[0] ),
                output.path );
      first += count;
   }

   const std::uint64_t dataBytes = expected*sizeof( float );
   const std::uint64_t paddedDataBytes =
      RoundUpFitsBlock( dataBytes, path );
   const std::array<std::byte, kFitsBlockBytes> zeroes = {};
   if ( paddedDataBytes != dataBytes )
      WriteAll( output.fd, zeroes.data(),
                static_cast<std::size_t>( paddedDataBytes - dataBytes ),
                output.path );
   if ( options.syncToDisk && ::fsync( output.fd ) != 0 )
      Fail( output.path, "fsync failed: " +
                         std::string( std::strerror( errno ) ) );
   if ( ::close( output.fd ) != 0 )
      Fail( output.path, "close failed: " +
                         std::string( std::strerror( errno ) ) );
   output.fd = -1;

   int publishResult = 0;
   if ( options.overwrite )
      publishResult = ::rename( output.path.c_str(), path.c_str() );
   else
      publishResult = ::link( output.path.c_str(), path.c_str() );
   if ( publishResult != 0 )
      Fail( path, "atomic publication failed: " +
                  std::string( std::strerror( errno ) ) );
   if ( !options.overwrite && ::unlink( output.path.c_str() ) != 0 )
      Fail( path, "published output but failed to remove temporary link: " +
                  std::string( std::strerror( errno ) ) );
   output.published = true;
}

} // namespace openastroflow::native
