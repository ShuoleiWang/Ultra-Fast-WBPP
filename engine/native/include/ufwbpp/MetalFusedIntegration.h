#ifndef UFWBPP_NATIVE_METALFUSEDINTEGRATION_H
#define UFWBPP_NATIVE_METALFUSEDINTEGRATION_H

#include "ufwbpp/FusedIntegration.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

namespace ufwbpp::native
{

struct MetalExecutionStats
{
   std::string deviceName;
   double wallSeconds = 0;
   double gpuSeconds = 0;
   std::uint64_t submittedBufferBytes = 0;
   std::uint64_t recommendedWorkingSetBytes = 0;
   std::uint64_t maximumBufferBytes = 0;
};

struct OutputRangeNormalization
{
   float finiteMinimum = 0;
   float finiteMaximum = 0;
   float effectiveMinimum = 0;
   float effectiveMaximum = 1;
   float scale = 1;
   float offset = 0;
   std::uint64_t finiteSamples = 0;
   bool applied = false;
};

bool MetalFusedIntegrationAvailable() noexcept;

class MetalFusedIntegrationExecutor final
{
public:
   explicit MetalFusedIntegrationExecutor(
      const std::filesystem::path& metalSourcePath );
   ~MetalFusedIntegrationExecutor();

   MetalFusedIntegrationExecutor(
      const MetalFusedIntegrationExecutor& ) = delete;
   MetalFusedIntegrationExecutor& operator =(
      const MetalFusedIntegrationExecutor& ) = delete;
   MetalFusedIntegrationExecutor(
      MetalFusedIntegrationExecutor&& ) noexcept;
   MetalFusedIntegrationExecutor& operator =(
      MetalFusedIntegrationExecutor&& ) noexcept;

   FusedIntegrationResult Run(
      const FusedIntegrationRequest& request,
      MetalExecutionStats* stats = nullptr );

   FusedIntegrationResult RunNativeRobustRejection(
      const NativeRobustIntegrationRequest& request,
      MetalExecutionStats* stats = nullptr );

   FusedIntegrationResult RunNativeLinearFitRejection(
      const NativeLinearFitIntegrationRequest& request,
      MetalExecutionStats* stats = nullptr );

   // Matches ImageIntegration's nominal [0,1] output-domain normalization:
   // positive minima are preserved, while negative minima and values above
   // one expand the effective range. Nonfinite samples are ignored.
   OutputRangeNormalization NormalizeOutputRangeInPlace(
      std::span<float> samples,
      MetalExecutionStats* stats = nullptr );

private:
   struct Impl;
   std::unique_ptr<Impl> m_impl;
};

} // namespace ufwbpp::native

#endif // UFWBPP_NATIVE_METALFUSEDINTEGRATION_H
