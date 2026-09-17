#ifndef OPENASTROFLOW_NATIVE_CPUFEATURES_H
#define OPENASTROFLOW_NATIVE_CPUFEATURES_H

#include <cstdint>
#include <string>
#include <vector>

namespace openastroflow::native
{

// Instruction-set facts of the running processor, read with cpuid on x86-64
// (including the OS XSAVE state check that AVX and AVX-512 require) and from
// the compile-time architecture elsewhere. Report-only: the kernels are
// compiled once for the baseline ISA and no code path depends on these bits.
struct CpuFeatures
{
   // "x86-64", "arm64" or "unknown".
   std::string architecture;
   // Lowercase feature names in a fixed order, e.g. "sse4.2", "avx2", "fma".
   std::vector<std::string> features;
   // cpuid brand string on x86-64 (trimmed); empty when unavailable.
   std::string brand;
};

CpuFeatures DetectCpuFeatures();

// Comma-separated `features`, in order.
std::string JoinFeatures( const CpuFeatures& features );

} // namespace openastroflow::native

#endif
