#ifndef OPENASTROFLOW_NATIVE_CALIBRATION_H
#define OPENASTROFLOW_NATIVE_CALIBRATION_H

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace openastroflow::native
{

enum class MasterDarkBiasModel
{
   // The master dark contains the camera bias pedestal. At equal exposure the
   // complete master dark replaces a separate bias subtraction.
   IncludesBias,

   // The master dark has already had its master bias removed.
   BiasSubtracted
};

enum class NegativeValuePolicy
{
   Preserve,
   ClampToZero
};

struct CalibrationStats
{
   std::uint64_t negativeSamples = 0;
   std::uint64_t invalidInputSamples = 0;
   std::uint64_t invalidFlatSamples = 0;
};

struct FlatCalibrationOptions
{
   // Pedestals use normalized sample units. A FITS PEDESTAL expressed in ADU
   // must be divided by 65535 before it is supplied here.
   float inputPedestal = 0;
   NegativeValuePolicy negativePolicy = NegativeValuePolicy::Preserve;
};

struct LightCalibrationOptions
{
   MasterDarkBiasModel darkBiasModel =
      MasterDarkBiasModel::IncludesBias;
   float darkScale = 1;
   float inputPedestal = 0;

   // A native master flat normalized to median one uses 1. An externally
   // produced master can use its reported or independently measured location.
   float masterFlatNormalization = 1;
   float minimumFlatResponse = 0.05f;

   // Added after dark/bias subtraction and flat division. Auto pedestal is a
   // deliberate two-pass operation via ComputeAutoOutputPedestal().
   float outputPedestal = 0;
   NegativeValuePolicy negativePolicy = NegativeValuePolicy::Preserve;
};

struct RobustLocationOptions
{
   // A deterministic spatially uniform sample bounds location-estimation cost.
   // Set at least as large as the input size for an exact median.
   std::size_t maximumSamples = 262144;
};

struct WinsorizedFlatOptions
{
   float lowSigma = 4;
   float highSigma = 3;
   std::uint32_t iterations = 2;
   float minimumSigma = 1e-7f;
};

struct MasterFlatIntegrationStats
{
   std::uint64_t outputSamples = 0;
   std::uint64_t invalidFrameSamples = 0;
   std::uint64_t lowWinsorizationOperations = 0;
   std::uint64_t highWinsorizationOperations = 0;
};

void ConvertUnsigned16ToUnit( std::span<const std::uint16_t> source,
                              std::span<float> destination );

CalibrationStats CalibrateFlat( std::span<const float> rawFlat,
                                std::span<const float> masterBias,
                                std::span<float> destination,
                                const FlatCalibrationOptions& options = {} );

float RobustMedian( std::span<const float> samples,
                    const RobustLocationOptions& options = {} );

float NormalizeMasterFlat( std::span<float> masterFlat,
                           const RobustLocationOptions& options = {} );

// Each input frame is already bias-calibrated. frameNormalizations contains
// one robust positive location per frame. Samples are divided by their frame
// location before iterative sigma winsorization and averaging.
MasterFlatIntegrationStats IntegrateMasterFlat(
   std::span<const std::span<const float>> calibratedFrames,
   std::span<const float> frameNormalizations,
   std::span<float> destination,
   const WinsorizedFlatOptions& options = {} );

// Correct bias/dark behavior:
// IncludesBias, scale=1: raw - dark.
// IncludesBias, scale!=1: raw - bias - scale*(dark-bias), bias required.
// BiasSubtracted: raw - bias - scale*dark, bias required.
// The result is then multiplied by masterFlatNormalization/masterFlat.
CalibrationStats CalibrateLight(
   std::span<const float> rawLight,
   std::span<const float> masterBias,
   std::span<const float> masterDark,
   std::span<const float> masterFlat,
   std::span<float> destination,
   const LightCalibrationOptions& options = {} );

float ComputeAutoOutputPedestal( std::span<const float> samples,
                                 float maximumPedestal = 0.0001f );

CalibrationStats ApplyOutputPedestal(
   std::span<float> samples,
   float pedestal,
   NegativeValuePolicy negativePolicy = NegativeValuePolicy::Preserve );

} // namespace openastroflow::native

#endif // OPENASTROFLOW_NATIVE_CALIBRATION_H
