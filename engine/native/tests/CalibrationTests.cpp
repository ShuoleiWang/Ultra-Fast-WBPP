#include "openastroflow/Calibration.h"
#include "openastroflow/NativeImageIO.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using namespace openastroflow::native;

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

void TestRaw16AndFlatCalibration()
{
   const std::vector<std::uint16_t> raw16 = {
      0, 1, 32768, 65534, 65535
   };
   std::vector<float> converted( raw16.size() );
   ConvertUnsigned16ToUnit( raw16, converted );
   ExpectNear( converted.front(), 0, 0, "raw16 zero" );
   ExpectNear( converted[2], 32768.0f/65535.0f, 1e-7f,
               "raw16 midpoint" );
   ExpectNear( converted.back(), 1, 0, "raw16 maximum" );

   const std::vector<float> raw = {
      0.1f, 0.2f, std::numeric_limits<float>::quiet_NaN()
   };
   const std::vector<float> bias = { 0.01f, 0.25f, 0.01f };
   std::vector<float> calibrated( raw.size() );
   FlatCalibrationOptions options;
   options.inputPedestal = 0.005f;
   CalibrationStats stats = CalibrateFlat(
      raw, bias, calibrated, options );
   ExpectNear( calibrated[0], 0.085f, 2e-8f,
               "flat bias/pedestal subtraction" );
   ExpectNear( calibrated[1], -0.055f, 2e-8f,
               "flat negative preservation" );
   Expect( std::isnan( calibrated[2] ), "flat nonfinite propagation" );
   Expect( stats.negativeSamples == 1 && stats.invalidInputSamples == 1,
           "flat calibration stats" );

   options.negativePolicy = NegativeValuePolicy::ClampToZero;
   stats = CalibrateFlat( raw, bias, calibrated, options );
   ExpectNear( calibrated[1], 0, 0, "flat negative clamp" );
}

void TestRobustFlatIntegration()
{
   std::vector<float> medianInput = {
      1, 5, 2, std::numeric_limits<float>::quiet_NaN(), 100, 3, 4
   };
   RobustLocationOptions exact;
   exact.maximumSamples = medianInput.size();
   ExpectNear( RobustMedian( medianInput, exact ), 3.5f, 0,
               "robust median" );

   constexpr std::size_t frameCount = 20;
   const std::vector<float> response = { 0.8f, 1.0f, 1.2f, 1.0f };
   std::vector<std::vector<float>> storage(
      frameCount, std::vector<float>( response.size() ) );
   std::vector<float> normalizations( frameCount );
   std::vector<std::span<const float>> frames;
   for ( std::size_t frame = 0; frame < frameCount; ++frame )
   {
      normalizations[frame] = 0.45f + 0.01f*frame;
      for ( std::size_t pixel = 0; pixel < response.size(); ++pixel )
         storage[frame][pixel] = response[pixel]*normalizations[frame];
      frames.push_back( storage[frame] );
   }
   storage[3][1] *= 12;
   storage[4][2] = 0;
   std::vector<float> integrated( response.size() );
   WinsorizedFlatOptions winsor;
   winsor.lowSigma = 2;
   winsor.highSigma = 2;
   winsor.iterations = 4;
   const MasterFlatIntegrationStats stats = IntegrateMasterFlat(
      frames, normalizations, integrated, winsor );
   Expect( stats.highWinsorizationOperations != 0
        && stats.lowWinsorizationOperations != 0,
           "master-flat winsorization counters" );
   for ( std::size_t i = 0; i < response.size(); ++i )
      ExpectNear( integrated[i], response[i], 0.04f,
                  "winsorized master-flat response" );

   const float location = NormalizeMasterFlat( integrated, exact );
   ExpectNear( location, 1, 0.02f, "master-flat output location" );
   ExpectNear( RobustMedian( integrated, exact ), 1, 1e-6f,
               "normalized master-flat median" );
}

void TestLightDarkBiasAndPedestalSemantics()
{
   const std::vector<float> raw = { 0.5f, 0.04f, 0.5f };
   const std::vector<float> bias = { 0.05f, 0.05f, 0.05f };
   const std::vector<float> darkIncludesBias = { 0.1f, 0.1f, 0.1f };
   const std::vector<float> darkBiasSubtracted = { 0.05f, 0.05f, 0.05f };
   const std::vector<float> flat = { 2, 2, 0.01f };
   std::vector<float> output( raw.size() );

   LightCalibrationOptions options;
   CalibrationStats stats = CalibrateLight(
      raw, {}, darkIncludesBias, flat, output, options );
   ExpectNear( output[0], 0.2f, 2e-8f,
               "dark includes bias equal-exposure formula" );
   ExpectNear( output[1], -0.03f, 2e-8f,
               "negative light preservation" );
   Expect( std::isnan( output[2] ), "invalid flat emits NaN" );
   Expect( stats.negativeSamples == 1 && stats.invalidFlatSamples == 1,
           "light calibration stats" );

   options.darkScale = 0.5f;
   bool requiredBias = false;
   try
   {
      CalibrateLight(
         raw, {}, darkIncludesBias, flat, output, options );
   }
   catch ( const std::invalid_argument& )
   {
      requiredBias = true;
   }
   Expect( requiredBias,
           "scaled bias-bearing dark must require master bias" );
   stats = CalibrateLight(
      raw, bias, darkIncludesBias, flat, output, options );
   ExpectNear( output[0], 0.2125f, 3e-8f,
               "scaled bias-bearing dark formula" );

   options.darkScale = 1;
   options.darkBiasModel = MasterDarkBiasModel::BiasSubtracted;
   stats = CalibrateLight(
      raw, bias, darkBiasSubtracted, flat, output, options );
   ExpectNear( output[0], 0.2f, 3e-8f,
               "bias-subtracted dark formula" );

   options.darkBiasModel = MasterDarkBiasModel::IncludesBias;
   options.inputPedestal = 0.01f;
   options.outputPedestal = 0.001f;
   options.negativePolicy = NegativeValuePolicy::ClampToZero;
   stats = CalibrateLight(
      raw, {}, darkIncludesBias, flat, output, options );
   ExpectNear( output[0], 0.196f, 3e-8f,
               "explicit input/output pedestal" );
   ExpectNear( output[1], 0, 0, "negative light clamp" );

   std::vector<float> pedestalSamples = { -0.001f, -0.00003f, 0.2f };
   const float pedestal = ComputeAutoOutputPedestal(
      pedestalSamples, 0.0001f );
   ExpectNear( pedestal, 0.0001f, 0, "limited auto pedestal" );
   stats = ApplyOutputPedestal(
      pedestalSamples, pedestal, NegativeValuePolicy::ClampToZero );
   ExpectNear( pedestalSamples[0], 0, 0, "post-pedestal clamp" );
   ExpectNear( pedestalSamples[1], 0.00007f, 1e-9f,
               "post-pedestal preserved signal" );
   Expect( stats.negativeSamples == 1, "post-pedestal negative stats" );
}

std::vector<std::filesystem::path> FitsFiles(
   const std::filesystem::path& directory )
{
   std::vector<std::filesystem::path> result;
   for ( const auto& entry : std::filesystem::directory_iterator( directory ) )
      if ( entry.is_regular_file()
        && (entry.path().extension() == ".fits"
         || entry.path().extension() == ".fit") )
         result.push_back( entry.path() );
   std::sort( result.begin(), result.end() );
   return result;
}

float EstimateCalibratedFlatMedian( FitsMonoReader& raw,
                                    XisfFloat32MonoReader& bias )
{
   const auto& info = raw.Info();
   std::vector<float> rawRow( info.width );
   std::vector<float> biasRow( info.width );
   std::vector<float> calibrated( info.width );
   std::vector<float> samples;
   samples.reserve( ((info.height + 31)/32)*((info.width + 15)/16) );
   for ( std::uint32_t row = 0; row < info.height; row += 32 )
   {
      raw.ReadRows( row, 1, rawRow,
                    FitsReadTransform::Unsigned16ToUnit() );
      bias.ReadRows( row, 1, biasRow );
      CalibrateFlat( rawRow, biasRow, calibrated );
      for ( std::uint32_t column = 0; column < info.width; column += 16 )
         samples.push_back( calibrated[column] );
   }
   RobustLocationOptions exact;
   exact.maximumSamples = samples.size();
   return RobustMedian( samples, exact );
}

float EstimateXisfMean( XisfFloat32MonoReader& reader )
{
   std::vector<float> samples = reader.ReadAll();
   long double sum = 0;
   std::uint64_t count = 0;
   for ( float sample : samples )
      if ( std::isfinite( sample ) && sample > 0 )
      {
         sum += sample;
         ++count;
      }
   Expect( count != 0, "XISF mean has no finite positive samples" );
   return static_cast<float>( sum/count );
}

struct Metrics
{
   std::uint64_t count = 0;
   double meanAbsolute = 0;
   double maximumAbsolute = 0;
   double rmse = 0;
   double nrmse = 0;
   double pearson = 0;
   double p99Absolute = 0;
   double fittedScale = 1;
   double fittedOffset = 0;
   double fittedRmse = 0;
   double fittedNrmse = 0;
};

Metrics Compare( std::span<const float> candidate,
                 std::span<const float> reference )
{
   Expect( candidate.size() == reference.size(), "metric sample count" );
   long double sumCandidate = 0;
   long double sumReference = 0;
   long double sumCandidate2 = 0;
   long double sumReference2 = 0;
   long double cross = 0;
   long double squareError = 0;
   long double absoluteError = 0;
   float minimumReference = std::numeric_limits<float>::infinity();
   float maximumReference = -std::numeric_limits<float>::infinity();
   std::vector<float> absoluteErrors;
   absoluteErrors.reserve( candidate.size() );
   Metrics result;
   for ( std::size_t i = 0; i < candidate.size(); ++i )
   {
      if ( !std::isfinite( candidate[i] ) || !std::isfinite( reference[i] ) )
         continue;
      const double error = static_cast<double>( candidate[i] ) - reference[i];
      const double absolute = std::abs( error );
      absoluteErrors.push_back( static_cast<float>( absolute ) );
      absoluteError += absolute;
      squareError += error*error;
      result.maximumAbsolute = std::max( result.maximumAbsolute, absolute );
      sumCandidate += candidate[i];
      sumReference += reference[i];
      sumCandidate2 += static_cast<double>( candidate[i] )*candidate[i];
      sumReference2 += static_cast<double>( reference[i] )*reference[i];
      cross += static_cast<double>( candidate[i] )*reference[i];
      minimumReference = std::min( minimumReference, reference[i] );
      maximumReference = std::max( maximumReference, reference[i] );
      ++result.count;
   }
   Expect( result.count != 0, "metric has no finite pairs" );
   const long double count = result.count;
   result.meanAbsolute = static_cast<double>( absoluteError/count );
   result.rmse = std::sqrt( static_cast<double>( squareError/count ) );
   const double range = maximumReference - minimumReference;
   result.nrmse = range > 0 ? result.rmse/range : result.rmse;
   const long double covariance = cross - sumCandidate*sumReference/count;
   const long double candidateVariance =
      sumCandidate2 - sumCandidate*sumCandidate/count;
   const long double referenceVariance =
      sumReference2 - sumReference*sumReference/count;
   result.pearson = covariance/std::sqrt(
      candidateVariance*referenceVariance );
   result.fittedScale = static_cast<double>( covariance/candidateVariance );
   result.fittedOffset = static_cast<double>(
      (sumReference - result.fittedScale*sumCandidate)/count );
   long double fittedSquareError = 0;
   for ( std::size_t i = 0; i < candidate.size(); ++i )
      if ( std::isfinite( candidate[i] ) && std::isfinite( reference[i] ) )
      {
         const double error = result.fittedScale*candidate[i]
                            + result.fittedOffset - reference[i];
         fittedSquareError += error*error;
      }
   result.fittedRmse = std::sqrt(
      static_cast<double>( fittedSquareError/count ) );
   result.fittedNrmse = range > 0
      ? result.fittedRmse/range : result.fittedRmse;
   const std::size_t p99 = static_cast<std::size_t>(
      0.99*(absoluteErrors.size() - 1) );
   std::nth_element(
      absoluteErrors.begin(), absoluteErrors.begin() + p99,
      absoluteErrors.end() );
   result.p99Absolute = absoluteErrors[p99];
   return result;
}

void PrintMetrics( const char* name, const Metrics& metrics, bool comma )
{
   std::cout << "  \"" << name << "\": {"
      << "\"count\":" << metrics.count
      << ",\"meanAbs\":" << metrics.meanAbsolute
      << ",\"maxAbs\":" << metrics.maximumAbsolute
      << ",\"rmse\":" << metrics.rmse
      << ",\"nrmse\":" << metrics.nrmse
      << ",\"pearson\":" << metrics.pearson
      << ",\"p99Abs\":" << metrics.p99Absolute
      << ",\"fittedScale\":" << metrics.fittedScale
      << ",\"fittedOffset\":" << metrics.fittedOffset
      << ",\"fittedRmse\":" << metrics.fittedRmse
      << ",\"fittedNrmse\":" << metrics.fittedNrmse << "}"
      << (comma ? "," : "") << '\n';
}

void RunRealSmoke( int argc, char** argv )
{
   Expect( argc == 8,
      "real smoke args: RAW_FLAT_DIR MASTER_BIAS PI_CAL_FLAT_DIR "
      "PI_MASTER_FLAT RAW_LIGHT MASTER_DARK PI_CAL_LIGHT" );
   const std::vector<std::filesystem::path> rawFlatPaths = FitsFiles( argv[1] );
   Expect( rawFlatPaths.size() == 20,
           "real master-flat smoke requires exactly 20 raw flats" );
   XisfFloat32MonoReader biasReader( argv[2] );
   XisfFloat32MonoReader piMasterFlatReader( argv[4] );
   constexpr std::uint32_t firstRow = 2048;
   constexpr std::uint32_t rowCount = 64;
   const std::size_t tileSamples =
      static_cast<std::size_t>( biasReader.Info().width )*rowCount;
   const std::vector<float> biasTile =
      biasReader.ReadRows( firstRow, rowCount );

   std::vector<std::vector<float>> calibratedFlatStorage;
   std::vector<float> flatNormalizations;
   calibratedFlatStorage.reserve( rawFlatPaths.size() );
   flatNormalizations.reserve( rawFlatPaths.size() );
   for ( const std::filesystem::path& path : rawFlatPaths )
   {
      FitsMonoReader rawReader( path );
      flatNormalizations.push_back(
         EstimateCalibratedFlatMedian( rawReader, biasReader ) );
      const std::vector<float> rawTile = rawReader.ReadRows(
         firstRow, rowCount, FitsReadTransform::Unsigned16ToUnit() );
      calibratedFlatStorage.emplace_back( tileSamples );
      CalibrateFlat( rawTile, biasTile, calibratedFlatStorage.back() );
   }
   std::vector<std::span<const float>> calibratedFlatSpans;
   for ( const auto& frame : calibratedFlatStorage )
      calibratedFlatSpans.push_back( frame );
   std::vector<float> nativeMasterFlat( tileSamples );
   WinsorizedFlatOptions winsor;
   winsor.lowSigma = 4;
   winsor.highSigma = 3;
   winsor.iterations = 2;
   const auto integrationStarted = std::chrono::steady_clock::now();
   const MasterFlatIntegrationStats integrationStats = IntegrateMasterFlat(
      calibratedFlatSpans, flatNormalizations, nativeMasterFlat, winsor );
   const double integrationSeconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - integrationStarted ).count();
   const std::vector<float> nativeMasterFlatForLight = nativeMasterFlat;
   RobustLocationOptions exactTile;
   exactTile.maximumSamples = tileSamples;
   NormalizeMasterFlat( nativeMasterFlat, exactTile );

   std::vector<float> piMasterFlat = piMasterFlatReader.ReadRows(
      firstRow, rowCount );
   NormalizeMasterFlat( piMasterFlat, exactTile );
   const Metrics masterFlatMetrics = Compare(
      nativeMasterFlat, piMasterFlat );

   FitsMonoReader firstRawFlat( rawFlatPaths.front() );
   const std::vector<float> firstRawFlatTile = firstRawFlat.ReadRows(
      firstRow, rowCount, FitsReadTransform::Unsigned16ToUnit() );
   std::vector<float> nativeCalibratedFlat( tileSamples );
   CalibrateFlat(
      firstRawFlatTile, biasTile, nativeCalibratedFlat );
   const std::filesystem::path piCalibratedFlatPath =
      std::filesystem::path( argv[3] )/
      (rawFlatPaths.front().stem().string() + "_c.xisf");
   XisfFloat32MonoReader piCalibratedFlatReader( piCalibratedFlatPath );
   const std::vector<float> piCalibratedFlat =
      piCalibratedFlatReader.ReadRows( firstRow, rowCount );
   const Metrics calibratedFlatMetrics = Compare(
      nativeCalibratedFlat, piCalibratedFlat );

   FitsMonoReader rawLightReader( argv[5] );
   XisfFloat32MonoReader darkReader( argv[6] );
   XisfFloat32MonoReader piCalibratedLightReader( argv[7] );
   const std::vector<float> rawLight = rawLightReader.ReadRows(
      firstRow, rowCount, FitsReadTransform::Unsigned16ToUnit() );
   const std::vector<float> dark = darkReader.ReadRows(
      firstRow, rowCount );
   const std::vector<float> piFlatForLight =
      piMasterFlatReader.ReadRows( firstRow, rowCount );
   const std::vector<float> piCalibratedLight =
      piCalibratedLightReader.ReadRows( firstRow, rowCount );
   std::vector<float> nativeCalibratedLight( tileSamples );
   LightCalibrationOptions lightOptions;
   lightOptions.darkBiasModel = MasterDarkBiasModel::IncludesBias;
   lightOptions.masterFlatNormalization =
      EstimateXisfMean( piMasterFlatReader );
   CalibrateLight( rawLight, {}, dark, piFlatForLight,
                   nativeCalibratedLight, lightOptions );
   const Metrics calibratedLightMetrics = Compare(
      nativeCalibratedLight, piCalibratedLight );
   std::vector<float> nativeChainLight( tileSamples );
   LightCalibrationOptions nativeChainOptions;
   nativeChainOptions.darkBiasModel = MasterDarkBiasModel::IncludesBias;
   nativeChainOptions.masterFlatNormalization = 1;
   const auto lightStarted = std::chrono::steady_clock::now();
   CalibrateLight( rawLight, {}, dark, nativeMasterFlatForLight,
                   nativeChainLight, nativeChainOptions );
   const double lightSeconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - lightStarted ).count();
   const Metrics nativeChainLightMetrics = Compare(
      nativeChainLight, piCalibratedLight );

   Expect( calibratedFlatMetrics.maximumAbsolute <= 2e-6,
           "real flat calibration parity gate" );
   Expect( masterFlatMetrics.pearson >= 0.999,
           "real master-flat shape gate" );
   Expect( calibratedLightMetrics.pearson >= 0.99,
           "real light calibration smoke gate" );
   Expect( nativeChainLightMetrics.pearson >= 0.99,
           "real native master-flat plus light smoke gate" );

   std::cout << std::setprecision( 17 ) << "{\n"
      << "  \"kind\":\"OpenAstroFlow native-real-calibration-smoke-v1\",\n"
      << "  \"flatCount\":" << rawFlatPaths.size() << ",\n"
      << "  \"masterFlatLowWinsorOps\":"
      << integrationStats.lowWinsorizationOperations << ",\n"
      << "  \"masterFlatHighWinsorOps\":"
      << integrationStats.highWinsorizationOperations << ",\n"
      << "  \"masterFlatIntegrationSeconds\":"
      << integrationSeconds << ",\n"
      << "  \"masterFlatMegapixelFramesPerSecond\":"
      << (tileSamples*rawFlatPaths.size()/1e6)/integrationSeconds << ",\n"
      << "  \"lightCalibrationSeconds\":" << lightSeconds << ",\n"
      << "  \"lightCalibrationMegapixelsPerSecond\":"
      << (tileSamples/1e6)/lightSeconds << ",\n"
      << "  \"piMasterFlatNormalization\":"
      << lightOptions.masterFlatNormalization << ",\n";
   PrintMetrics( "calibratedFlat", calibratedFlatMetrics, true );
   PrintMetrics( "masterFlatShape", masterFlatMetrics, true );
   PrintMetrics( "calibratedLight", calibratedLightMetrics, true );
   PrintMetrics( "nativeMasterFlatAndLight", nativeChainLightMetrics, true );
   std::cout << "  \"gate\":true\n}\n";
}

} // namespace

int main( int argc, char** argv )
{
   try
   {
      TestRaw16AndFlatCalibration();
      TestRobustFlatIntegration();
      TestLightDarkBiasAndPedestalSemantics();
      if ( argc == 1 )
         std::cout << "OpenAstroFlow native calibration unit tests passed\n";
      else
         RunRealSmoke( argc, argv );
      return 0;
   }
   catch ( const std::exception& error )
   {
      std::cerr << error.what() << '\n';
      return 1;
   }
}
