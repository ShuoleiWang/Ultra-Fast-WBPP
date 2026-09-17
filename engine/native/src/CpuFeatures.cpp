#include "openastroflow/CpuFeatures.h"

#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#  define OAF_CPU_X86_64 1
#  if defined(_MSC_VER)
#     include <intrin.h>
#  else
#     include <cpuid.h>
#  endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#  define OAF_CPU_ARM64 1
#endif

namespace openastroflow::native
{

namespace
{

#if defined(OAF_CPU_X86_64)

struct CpuidRegisters
{
   std::uint32_t eax = 0;
   std::uint32_t ebx = 0;
   std::uint32_t ecx = 0;
   std::uint32_t edx = 0;
};

CpuidRegisters Cpuid( std::uint32_t leaf, std::uint32_t subleaf )
{
   CpuidRegisters registers;
#  if defined(_MSC_VER)
   int values[4] = { 0, 0, 0, 0 };
   __cpuidex( values, static_cast<int>( leaf ), static_cast<int>( subleaf ) );
   registers.eax = static_cast<std::uint32_t>( values[0] );
   registers.ebx = static_cast<std::uint32_t>( values[1] );
   registers.ecx = static_cast<std::uint32_t>( values[2] );
   registers.edx = static_cast<std::uint32_t>( values[3] );
#  else
   unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
   __cpuid_count( leaf, subleaf, eax, ebx, ecx, edx );
   registers.eax = eax;
   registers.ebx = ebx;
   registers.ecx = ecx;
   registers.edx = edx;
#  endif
   return registers;
}

std::uint64_t ExtendedControlRegister0()
{
#  if defined(_MSC_VER)
   return static_cast<std::uint64_t>( _xgetbv( 0 ) );
#  else
   std::uint32_t eax = 0, edx = 0;
   __asm__ __volatile__( "xgetbv" : "=a"( eax ), "=d"( edx ) : "c"( 0 ) );
   return (static_cast<std::uint64_t>( edx ) << 32) | eax;
#  endif
}

bool Bit( std::uint32_t value, unsigned bit )
{
   return ((value >> bit) & 1U) != 0;
}

CpuFeatures DetectX86()
{
   CpuFeatures result;
   result.architecture = "x86-64";
   const CpuidRegisters leaf0 = Cpuid( 0, 0 );
   const std::uint32_t maximumLeaf = leaf0.eax;
   if ( maximumLeaf < 1 )
      return result;
   const CpuidRegisters leaf1 = Cpuid( 1, 0 );
   const bool osxsave = Bit( leaf1.ecx, 27 );
   const std::uint64_t xcr0 = osxsave ? ExtendedControlRegister0() : 0;
   const bool avxState = (xcr0 & 0x6U) == 0x6U;          // XMM + YMM
   const bool avx512State = avxState && (xcr0 & 0xE0U) == 0xE0U; // opmask + ZMM
   CpuidRegisters leaf7;
   if ( maximumLeaf >= 7 )
      leaf7 = Cpuid( 7, 0 );
   if ( Bit( leaf1.ecx, 20 ) )
      result.features.push_back( "sse4.2" );
   if ( Bit( leaf1.ecx, 23 ) )
      result.features.push_back( "popcnt" );
   if ( avxState && Bit( leaf1.ecx, 28 ) )
      result.features.push_back( "avx" );
   if ( avxState && Bit( leaf1.ecx, 12 ) )
      result.features.push_back( "fma" );
   if ( avxState && Bit( leaf1.ecx, 29 ) )
      result.features.push_back( "f16c" );
   if ( Bit( leaf7.ebx, 3 ) )
      result.features.push_back( "bmi1" );
   if ( Bit( leaf7.ebx, 8 ) )
      result.features.push_back( "bmi2" );
   if ( avxState && Bit( leaf7.ebx, 5 ) )
      result.features.push_back( "avx2" );
   if ( avx512State && Bit( leaf7.ebx, 16 ) )
      result.features.push_back( "avx512f" );
   if ( avx512State && Bit( leaf7.ebx, 17 ) )
      result.features.push_back( "avx512dq" );
   if ( avx512State && Bit( leaf7.ebx, 30 ) )
      result.features.push_back( "avx512bw" );
   if ( avx512State && Bit( leaf7.ebx, 31 ) )
      result.features.push_back( "avx512vl" );

   const CpuidRegisters extended = Cpuid( 0x80000000U, 0 );
   if ( extended.eax >= 0x80000004U )
   {
      char brand[49] = {};
      for ( std::uint32_t leaf = 0; leaf < 3; ++leaf )
      {
         const CpuidRegisters part = Cpuid( 0x80000002U + leaf, 0 );
         const std::uint32_t words[4] = { part.eax, part.ebx, part.ecx, part.edx };
         std::memcpy( brand + leaf*16, words, 16 );
      }
      std::string text( brand );
      const std::size_t first = text.find_first_not_of( " \t" );
      const std::size_t last = text.find_last_not_of( " \t" );
      if ( first != std::string::npos )
         result.brand = text.substr( first, last - first + 1 );
   }
   return result;
}

#endif // OAF_CPU_X86_64

} // namespace

CpuFeatures DetectCpuFeatures()
{
#if defined(OAF_CPU_X86_64)
   return DetectX86();
#elif defined(OAF_CPU_ARM64)
   CpuFeatures result;
   result.architecture = "arm64";
   result.features.push_back( "neon" );
   return result;
#else
   CpuFeatures result;
   result.architecture = "unknown";
   return result;
#endif
}

std::string JoinFeatures( const CpuFeatures& features )
{
   std::string joined;
   for ( const std::string& feature : features.features )
   {
      if ( !joined.empty() )
         joined += ',';
      joined += feature;
   }
   return joined;
}

} // namespace openastroflow::native
