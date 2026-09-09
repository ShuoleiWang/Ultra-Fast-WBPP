#include "openastroflow/Calibration.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace openastroflow::native
{

namespace
{

void RequireSameSize( std::size_t expected,
                      std::size_t actual,
                      const char* role )
{
   if ( expected != actual )
      throw std::invalid_argument(
         std::string( "Ultra-Fast WBPP native calibration " ) + role
         + " sample count mismatch" );
}

void ValidatePedestal( float pedestal, const char* role )
{
   if ( !std::isfinite( pedestal ) || pedestal < 0 )
      throw std::invalid_argument(
         std::string( "Ultra-Fast WBPP native " ) + role
         + " pedestal must be finite and nonnegative" );
}

float MedianInPlace( std::vector<float>& values )
{
   if ( values.empty() )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native robust median has no finite samples" );
   const std::size_t middle = values.size()/2;
   std::nth_element(
      values.begin(), values.begin() + middle, values.end() );
   const float upper = values[middle];
   if ( values.size() % 2 != 0 )
      return upper;
   const float lower = *std::max_element(
      values.begin(), values.begin() + middle );
   return static_cast<float>(
      (static_cast<double>( lower ) + upper)*0.5 );
}

} // namespace

void ConvertUnsigned16ToUnit( std::span<const std::uint16_t> source,
                              std::span<float> destination )
{
   RequireSameSize( source.size(), destination.size(),
                    "raw16 destination" );
   constexpr float scale = 1.0f/65535.0f;
   for ( std::size_t i = 0; i < source.size(); ++i )
      destination[i] = static_cast<float>( source[i] )*scale;
}

CalibrationStats CalibrateFlat(
   std::span<const float> rawFlat,
   std::span<const float> masterBias,
   std::span<float> destination,
   const FlatCalibrationOptions& options )
{
   RequireSameSize( rawFlat.size(), masterBias.size(), "master bias" );
   RequireSameSize( rawFlat.size(), destination.size(),
                    "flat destination" );
   ValidatePedestal( options.inputPedestal, "input" );

   CalibrationStats stats;
   for ( std::size_t i = 0; i < rawFlat.size(); ++i )
   {
      if ( !std::isfinite( rawFlat[i] )
        || !std::isfinite( masterBias[i] ) )
      {
         destination[i] = std::numeric_limits<float>::quiet_NaN();
         ++stats.invalidInputSamples;
         continue;
      }
      float sample = rawFlat[i] - options.inputPedestal - masterBias[i];
      if ( sample < 0 )
      {
         ++stats.negativeSamples;
         if ( options.negativePolicy == NegativeValuePolicy::ClampToZero )
            sample = 0;
      }
      destination[i] = sample;
   }
   return stats;
}

float RobustMedian( std::span<const float> samples,
                    const RobustLocationOptions& options )
{
   if ( samples.empty() )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native robust median input must not be empty" );
   if ( options.maximumSamples == 0 )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native robust median maximumSamples must be positive" );

   const std::size_t target = std::min(
      samples.size(), options.maximumSamples );
   std::vector<float> finite;
   finite.reserve( target );
   if ( target == 1 )
   {
      if ( std::isfinite( samples.front() ) )
         finite.push_back( samples.front() );
   }
   else
   {
      for ( std::size_t i = 0; i < target; ++i )
      {
         const std::size_t index = static_cast<std::size_t>(
            (static_cast<unsigned long long>( i )*(samples.size() - 1))
            /(target - 1) );
         if ( std::isfinite( samples[index] ) )
            finite.push_back( samples[index] );
      }
   }
   return MedianInPlace( finite );
}

float NormalizeMasterFlat( std::span<float> masterFlat,
                           const RobustLocationOptions& options )
{
   const float location = RobustMedian( masterFlat, options );
   if ( !std::isfinite( location )
     || location <= std::numeric_limits<float>::epsilon() )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native master-flat robust location must be positive" );
   const float inverse = 1/location;
   for ( float& sample : masterFlat )
      if ( std::isfinite( sample ) )
         sample *= inverse;
   return location;
}

MasterFlatIntegrationStats IntegrateMasterFlat(
   std::span<const std::span<const float>> calibratedFrames,
   std::span<const float> frameNormalizations,
   std::span<float> destination,
   const WinsorizedFlatOptions& options )
{
   if ( calibratedFrames.empty() )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native master-flat integration needs at least one frame" );
   if ( destination.empty() )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native master-flat destination must not be empty" );
   RequireSameSize( calibratedFrames.size(), frameNormalizations.size(),
                    "flat normalization" );
   for ( std::span<const float> frame : calibratedFrames )
      RequireSameSize( destination.size(), frame.size(), "flat frame" );
   if ( !std::isfinite( options.lowSigma ) || options.lowSigma <= 0
     || !std::isfinite( options.highSigma ) || options.highSigma <= 0
     || !std::isfinite( options.minimumSigma )
     || options.minimumSigma < 0 )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native master-flat winsorization options are invalid" );
   for ( float normalization : frameNormalizations )
      if ( !std::isfinite( normalization )
        || normalization <= std::numeric_limits<float>::epsilon() )
         throw std::invalid_argument(
            "Ultra-Fast WBPP native flat normalization must be finite and positive" );

   MasterFlatIntegrationStats stats;
   stats.outputSamples = destination.size();
   std::vector<float> stack( calibratedFrames.size() );
   for ( std::size_t pixel = 0; pixel < destination.size(); ++pixel )
   {
      std::size_t count = 0;
      double sum = 0;
      double squareSum = 0;
      for ( std::size_t frame = 0; frame < calibratedFrames.size(); ++frame )
      {
         const float sample =
            calibratedFrames[frame][pixel]/frameNormalizations[frame];
         if ( !std::isfinite( sample ) )
         {
            ++stats.invalidFrameSamples;
            continue;
         }
         stack[count++] = sample;
         sum += sample;
         squareSum += static_cast<double>( sample )*sample;
      }
      if ( count == 0 )
      {
         destination[pixel] = std::numeric_limits<float>::quiet_NaN();
         continue;
      }

      double mean = sum/count;
      double variance = std::max( 0.0, squareSum/count - mean*mean );
      double sigma = std::sqrt( variance );
      for ( std::uint32_t iteration = 0;
            iteration < options.iterations
            && sigma > options.minimumSigma;
            ++iteration )
      {
         const double low = mean - options.lowSigma*sigma;
         const double high = mean + options.highSigma*sigma;
         sum = 0;
         squareSum = 0;
         for ( std::size_t i = 0; i < count; ++i )
         {
            if ( stack[i] < low )
            {
               stack[i] = static_cast<float>( low );
               ++stats.lowWinsorizationOperations;
            }
            else if ( stack[i] > high )
            {
               stack[i] = static_cast<float>( high );
               ++stats.highWinsorizationOperations;
            }
            sum += stack[i];
            squareSum += static_cast<double>( stack[i] )*stack[i];
         }
         mean = sum/count;
         variance = std::max( 0.0, squareSum/count - mean*mean );
         sigma = std::sqrt( variance );
      }
      destination[pixel] = static_cast<float>( mean );
   }
   return stats;
}

CalibrationStats CalibrateLight(
   std::span<const float> rawLight,
   std::span<const float> masterBias,
   std::span<const float> masterDark,
   std::span<const float> masterFlat,
   std::span<float> destination,
   const LightCalibrationOptions& options )
{
   RequireSameSize( rawLight.size(), masterDark.size(), "master dark" );
   RequireSameSize( rawLight.size(), masterFlat.size(), "master flat" );
   RequireSameSize( rawLight.size(), destination.size(),
                    "light destination" );
   ValidatePedestal( options.inputPedestal, "input" );
   ValidatePedestal( options.outputPedestal, "output" );
   if ( !std::isfinite( options.darkScale ) || options.darkScale < 0 )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native dark scale must be finite and nonnegative" );
   if ( !std::isfinite( options.masterFlatNormalization )
     || options.masterFlatNormalization <= 0 )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native master-flat normalization must be positive" );
   if ( !std::isfinite( options.minimumFlatResponse )
     || options.minimumFlatResponse <= 0 )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native minimum flat response must be positive" );

   const bool needsBias =
      options.darkBiasModel == MasterDarkBiasModel::BiasSubtracted
      || options.darkScale != 1;
   if ( needsBias )
      RequireSameSize( rawLight.size(), masterBias.size(), "master bias" );
   else if ( !masterBias.empty() )
      RequireSameSize( rawLight.size(), masterBias.size(), "master bias" );

   CalibrationStats stats;
   for ( std::size_t i = 0; i < rawLight.size(); ++i )
   {
      const bool biasFinite = !needsBias || std::isfinite( masterBias[i] );
      if ( !std::isfinite( rawLight[i] )
        || !std::isfinite( masterDark[i] ) || !biasFinite )
      {
         destination[i] = std::numeric_limits<float>::quiet_NaN();
         ++stats.invalidInputSamples;
         continue;
      }
      if ( !std::isfinite( masterFlat[i] )
        || masterFlat[i] <= options.minimumFlatResponse )
      {
         destination[i] = std::numeric_limits<float>::quiet_NaN();
         ++stats.invalidFlatSamples;
         continue;
      }

      float signal = rawLight[i] - options.inputPedestal;
      if ( options.darkBiasModel == MasterDarkBiasModel::IncludesBias )
      {
         if ( options.darkScale == 1 )
            signal -= masterDark[i];
         else
            signal = signal - masterBias[i]
                   - options.darkScale*(masterDark[i] - masterBias[i]);
      }
      else
         signal = signal - masterBias[i]
                - options.darkScale*masterDark[i];

      float sample = signal*options.masterFlatNormalization/masterFlat[i]
                   + options.outputPedestal;
      if ( sample < 0 )
      {
         ++stats.negativeSamples;
         if ( options.negativePolicy == NegativeValuePolicy::ClampToZero )
            sample = 0;
      }
      destination[i] = sample;
   }
   return stats;
}

float ComputeAutoOutputPedestal( std::span<const float> samples,
                                 float maximumPedestal )
{
   ValidatePedestal( maximumPedestal, "maximum auto output" );
   float minimum = std::numeric_limits<float>::infinity();
   for ( float sample : samples )
      if ( std::isfinite( sample ) )
         minimum = std::min( minimum, sample );
   if ( !std::isfinite( minimum ) )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native auto pedestal has no finite samples" );
   if ( minimum >= 0 )
      return 0;
   return std::min( -minimum, maximumPedestal );
}

CalibrationStats ApplyOutputPedestal(
   std::span<float> samples,
   float pedestal,
   NegativeValuePolicy negativePolicy )
{
   ValidatePedestal( pedestal, "output" );
   CalibrationStats stats;
   for ( float& sample : samples )
   {
      if ( !std::isfinite( sample ) )
      {
         ++stats.invalidInputSamples;
         continue;
      }
      sample += pedestal;
      if ( sample < 0 )
      {
         ++stats.negativeSamples;
         if ( negativePolicy == NegativeValuePolicy::ClampToZero )
            sample = 0;
      }
   }
   return stats;
}

} // namespace openastroflow::native
