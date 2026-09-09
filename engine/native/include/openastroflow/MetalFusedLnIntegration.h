#ifndef OPENASTROFLOW_NATIVE_METALFUSEDLNINTEGRATION_H
#define OPENASTROFLOW_NATIVE_METALFUSEDLNINTEGRATION_H

#include "openastroflow/FusedLnIntegration.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

namespace openastroflow::native
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

bool MetalFusedLnIntegrationAvailable() noexcept;

class MetalFusedLnIntegrationExecutor final
{
public:
   explicit MetalFusedLnIntegrationExecutor(
      const std::filesystem::path& metalSourcePath );
   ~MetalFusedLnIntegrationExecutor();

   MetalFusedLnIntegrationExecutor(
      const MetalFusedLnIntegrationExecutor& ) = delete;
   MetalFusedLnIntegrationExecutor& operator =(
      const MetalFusedLnIntegrationExecutor& ) = delete;
   MetalFusedLnIntegrationExecutor(
      MetalFusedLnIntegrationExecutor&& ) noexcept;
   MetalFusedLnIntegrationExecutor& operator =(
      MetalFusedLnIntegrationExecutor&& ) noexcept;

   FusedLnIntegrationResult Run(
      const FusedLnIntegrationRequest& request,
      MetalExecutionStats* stats = nullptr );

   FusedLnIntegrationResult RunNativeRobustRejection(
      const NativeRobustIntegrationRequest& request,
      MetalExecutionStats* stats = nullptr );

   FusedLnIntegrationResult RunNativeLinearFitRejection(
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

} // namespace openastroflow::native

#endif // OPENASTROFLOW_NATIVE_METALFUSEDLNINTEGRATION_H
