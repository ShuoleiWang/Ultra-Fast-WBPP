#ifndef UFWBPP_NATIVE_FUSEDINTEGRATION_H
#define UFWBPP_NATIVE_FUSEDINTEGRATION_H

#include "ufwbpp/ImageTile.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ufwbpp::native
{

inline constexpr std::uint8_t DefaultOracleRejectionBits = 0x3f;
inline constexpr std::uint32_t MaximumNativeRejectionFrames = 512;
inline constexpr std::uint32_t MaximumMetalRejectionFrames = 64;

struct FusedIntegrationRequest
{
   TileRegion tile;
   std::uint32_t frameCount = 0;
   std::uint32_t gridWidth = 0;
   std::uint32_t gridHeight = 0;
   std::span<const float> frameMajorSamples;
   std::span<const std::uint8_t> frameMajorRejectionMask;
   std::span<const float> frameMajorScaleGrid;
   std::span<const float> frameMajorZeroOffsetGrid;
   std::span<const float> frameWeights;
   std::uint8_t rejectionBits = DefaultOracleRejectionBits;
   float outputScale = 1.0F;
   float outputOffset = 0.0F;

   void Validate() const;
   std::size_t TilePixels() const;
   std::size_t GridSamplesPerFrame() const;
};

struct FusedIntegrationResult
{
   std::vector<float> integrated;
   std::vector<std::uint16_t> acceptedSamples;
   std::vector<std::uint16_t> rejectedSamples;
};

struct NativeRobustIntegrationRequest
{
   TileRegion tile;
   std::uint32_t frameCount = 0;
   std::uint32_t gridWidth = 0;
   std::uint32_t gridHeight = 0;
   std::span<const float> frameMajorSamples;
   std::span<const float> frameMajorScaleGrid;
   std::span<const float> frameMajorZeroOffsetGrid;
   std::span<const float> frameWeights;
   float rangeLow = 0;
   float lowSigma = 5.0F;
   float highSigma = 3.5F;
   float winsorSigma = 1.5F;
   std::uint32_t winsorIterations = 2;
   float outputScale = 1.0F;
   float outputOffset = 0.0F;

   void Validate() const;
   std::size_t TilePixels() const;
   std::size_t GridSamplesPerFrame() const;
};

struct NativeLinearFitIntegrationRequest
{
   TileRegion tile;
   std::uint32_t frameCount = 0;
   std::uint32_t gridWidth = 0;
   std::uint32_t gridHeight = 0;
   std::span<const float> frameMajorSamples;
   std::span<const float> frameMajorScaleGrid;
   std::span<const float> frameMajorZeroOffsetGrid;
   std::span<const float> frameWeights;
   float rangeLow = 0;
   float lowTolerance = 5.0F;
   float highTolerance = 3.5F;
   std::uint32_t fitBisectionIterations = 10;
   std::uint32_t rejectionIterations = 8;
   float outputScale = 1.0F;
   float outputOffset = 0.0F;

   void Validate() const;
   std::size_t TilePixels() const;
   std::size_t GridSamplesPerFrame() const;
};

float BicubicBSplineSample(
   std::span<const float> grid,
   std::uint32_t width,
   std::uint32_t height,
   float x,
   float y );

FusedIntegrationResult RunCpuOracleFusedIntegration(
   const FusedIntegrationRequest& request );

FusedIntegrationResult RunCpuOracleNativeRobustIntegration(
   const NativeRobustIntegrationRequest& request );

FusedIntegrationResult RunCpuOracleNativeLinearFitIntegration(
   const NativeLinearFitIntegrationRequest& request );

} // namespace ufwbpp::native

#endif // UFWBPP_NATIVE_FUSEDINTEGRATION_H
