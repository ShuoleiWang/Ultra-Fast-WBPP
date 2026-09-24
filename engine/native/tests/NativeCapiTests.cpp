#include "ufwbpp/c_api.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

void Require( bool condition, const char* message )
{
   if ( !condition )
      throw std::runtime_error( message );
}

void TestAbiAndLinearFit()
{
   Require( ufwbpp_native_abi_version() == UFWBPP_NATIVE_ABI_VERSION,
            "native ABI version differs" );
   constexpr std::uint32_t frames = 5;
   constexpr std::uint32_t pixels = 4;
   std::array<float, frames*pixels> samples{
      0.10F, 0.20F, 0.30F, 0.40F,
      0.11F, 0.19F, 0.31F, 0.39F,
      0.09F, 0.21F, 0.29F, 0.41F,
      0.10F, 0.20F, 0.30F, 0.40F,
      0.80F, 0.20F, 0.30F, 0.40F
   };
   std::array<float, frames*4> scales;
   std::array<float, frames*4> offsets{};
   std::array<float, frames> weights;
   scales.fill( 1.0F );
   weights.fill( 1.0F );
   std::array<float, pixels> integrated{};
   std::array<std::uint16_t, pixels> accepted{};
   std::array<std::uint16_t, pixels> rejected{};
   std::array<char, 256> error{};

   const UfwbppNativeIntegrationRequestV1 request{
      sizeof( UfwbppNativeIntegrationRequestV1 ),
      2, 2, 0, 2, frames, 2, 2,
      samples.data(), samples.size(),
      scales.data(), scales.size(),
      offsets.data(), offsets.size(),
      weights.data(), weights.size(),
      0.0F, 5.0F, 3.5F, 10, 8, 1.0F, 0.0F
   };
   UfwbppNativeIntegrationOutputV1 output{
      sizeof( UfwbppNativeIntegrationOutputV1 ),
      integrated.data(), accepted.data(), rejected.data(), integrated.size()
   };
   const int status = ufwbpp_native_cpu_linear_fit_v1(
      &request, &output, error.data(), error.size() );
   Require( status == UFWBPP_NATIVE_OK, error.data() );
   for ( float value : integrated )
      Require( std::isfinite( value ), "C ABI returned a nonfinite sample" );
   for ( std::uint16_t value : accepted )
      Require( value >= 3 && value <= frames,
               "C ABI returned an invalid accepted-sample count" );
}

void TestCapacityGate()
{
   const UfwbppNativeIntegrationRequestV1 request{
      sizeof( UfwbppNativeIntegrationRequestV1 ), 2, 2, 0, 2, 5, 2, 2
   };
   UfwbppNativeIntegrationOutputV1 output{
      sizeof( UfwbppNativeIntegrationOutputV1 ), nullptr, nullptr, nullptr, 0
   };
   std::array<char, 128> error{};
   Require( ufwbpp_native_cpu_linear_fit_v1(
               &request, &output, error.data(), error.size() )
               == UFWBPP_NATIVE_BUFFER_TOO_SMALL,
            "C ABI accepted missing output buffers" );
}

void TestFiveHundredTwelveFramesAreNotTruncated()
{
   constexpr std::uint32_t frames = 512;
   constexpr std::uint32_t pixels = 4;
   std::vector<float> samples( frames*pixels, 2.0F );
   for ( std::uint32_t frame = 0; frame < 64; ++frame )
      for ( std::uint32_t pixel = 0; pixel < pixels; ++pixel )
         samples[static_cast<std::size_t>( frame )*pixels + pixel] = 1.0F;
   std::vector<float> scales( frames*4, 1.0F );
   std::vector<float> offsets( frames*4, 0.0F );
   std::vector<float> weights( frames, 1.0F );
   std::array<float, pixels> integrated{};
   std::array<std::uint16_t, pixels> accepted{};
   std::array<std::uint16_t, pixels> rejected{};
   std::array<char, 256> error{};
   const UfwbppNativeIntegrationRequestV1 request{
      sizeof( UfwbppNativeIntegrationRequestV1 ),
      2, 2, 0, 2, frames, 2, 2,
      samples.data(), samples.size(), scales.data(), scales.size(),
      offsets.data(), offsets.size(), weights.data(), weights.size(),
      -1.0F, 5.0F, 5.0F, 10, 8, 1.0F, 0.0F
   };
   UfwbppNativeIntegrationOutputV1 output{
      sizeof( UfwbppNativeIntegrationOutputV1 ),
      integrated.data(), accepted.data(), rejected.data(), integrated.size()
   };
   Require( ufwbpp_native_cpu_linear_fit_v1(
               &request, &output, error.data(), error.size() ) == UFWBPP_NATIVE_OK,
            error.data() );
   for ( std::size_t pixel = 0; pixel < pixels; ++pixel )
   {
      Require( static_cast<std::uint32_t>( accepted[pixel] ) + rejected[pixel]
                  == frames,
               "512-frame C ABI accounting differs" );
      Require( integrated[pixel] > 1.5F,
               "512-frame C ABI silently truncated samples after frame 64" );
   }
}

void TestOpaqueMetalExecutor()
{
   std::array<char, 1024> error{};
   std::uint32_t available = 0;
   Require( ufwbpp_native_metal_available_v1(
               &available, error.data(), error.size() ) == UFWBPP_NATIVE_OK,
            error.data() );
   if ( available == 0 )
      return;
   UfwbppNativeMetalExecutorV1* executor = nullptr;
   Require( ufwbpp_native_metal_executor_create_v1(
               nullptr, &executor, error.data(), error.size() ) == UFWBPP_NATIVE_OK
         && executor != nullptr,
            error.data() );
   constexpr std::uint32_t frames = 5;
   constexpr std::uint32_t pixels = 4;
   std::array<float, frames*pixels> samples{
      0.10F, 0.20F, 0.30F, 0.40F,
      0.11F, 0.19F, 0.31F, 0.39F,
      0.09F, 0.21F, 0.29F, 0.41F,
      0.10F, 0.20F, 0.30F, 0.40F,
      0.80F, 0.20F, 0.30F, 0.40F
   };
   std::array<float, frames*4> scales;
   std::array<float, frames*4> offsets{};
   std::array<float, frames> weights;
   scales.fill( 1.0F );
   weights.fill( 1.0F );
   std::array<float, pixels> integrated{};
   std::array<std::uint16_t, pixels> accepted{};
   std::array<std::uint16_t, pixels> rejected{};
   const UfwbppNativeIntegrationRequestV1 request{
      sizeof( UfwbppNativeIntegrationRequestV1 ),
      2, 2, 0, 2, frames, 2, 2,
      samples.data(), samples.size(), scales.data(), scales.size(),
      offsets.data(), offsets.size(), weights.data(), weights.size(),
      0.0F, 5.0F, 3.5F, 10, 8, 1.0F, 0.0F
   };
   UfwbppNativeIntegrationOutputV1 output{
      sizeof( UfwbppNativeIntegrationOutputV1 ),
      integrated.data(), accepted.data(), rejected.data(), integrated.size()
   };
   UfwbppNativeExecutionStatsV1 stats{};
   stats.struct_size = sizeof( stats );
   Require( ufwbpp_native_metal_linear_fit_v1(
               executor, &request, &output, &stats,
               error.data(), error.size() ) == UFWBPP_NATIVE_OK,
            error.data() );
   Require( stats.executed_on_gpu == 1 && stats.device_name[0] != '\0'
         && stats.submitted_buffer_bytes > 0,
            "opaque Metal C ABI did not report GPU evidence" );
   ufwbpp_native_metal_executor_destroy_v1( executor );
}

void TestOpaqueMetalMaskedNinetySixFrames()
{
   std::array<char, 1024> error{};
   std::uint32_t available = 0;
   Require( ufwbpp_native_metal_available_v1(
               &available, error.data(), error.size() ) == UFWBPP_NATIVE_OK,
            error.data() );
   if ( available == 0 )
      return;
   UfwbppNativeMetalExecutorV1* executor = nullptr;
   Require( ufwbpp_native_metal_executor_create_v1(
               nullptr, &executor, error.data(), error.size() ) == UFWBPP_NATIVE_OK
         && executor != nullptr,
            error.data() );
   constexpr std::uint32_t frames = 96;
   constexpr std::uint32_t width = 4;
   constexpr std::uint32_t height = 3;
   constexpr std::uint32_t pixels = width*height;
   std::vector<float> samples( frames*pixels );
   std::vector<std::uint8_t> mask( frames*pixels, 0 );
   for ( std::uint32_t frame = 0; frame < frames; ++frame )
      for ( std::uint32_t pixel = 0; pixel < pixels; ++pixel )
         samples[static_cast<std::size_t>( frame )*pixels + pixel] =
            0.2F + 0.0001F*frame + 0.00001F*pixel;
   for ( std::uint32_t frame = 0; frame < frames; frame += 7 )
      mask[static_cast<std::size_t>( frame )*pixels + frame%pixels] = 1;
   samples[95*pixels + 1] = std::numeric_limits<float>::quiet_NaN();
   std::vector<float> scales( frames*4, 1.0F );
   std::vector<float> offsets( frames*4, 0.0F );
   std::vector<float> weights( frames, 1.0F );
   std::array<float, pixels> cpuIntegrated{}, gpuIntegrated{};
   std::array<std::uint16_t, pixels> cpuAccepted{}, gpuAccepted{};
   std::array<std::uint16_t, pixels> cpuRejected{}, gpuRejected{};
   const UfwbppNativeMaskedIntegrationRequestV1 request{
      sizeof( UfwbppNativeMaskedIntegrationRequestV1 ),
      width, height, 0, height, frames, 2, 2,
      samples.data(), samples.size(), mask.data(), mask.size(),
      scales.data(), scales.size(), offsets.data(), offsets.size(),
      weights.data(), weights.size(), 1, 1.0F, 0.0F
   };
   UfwbppNativeIntegrationOutputV1 cpuOutput{
      sizeof( UfwbppNativeIntegrationOutputV1 ), cpuIntegrated.data(),
      cpuAccepted.data(), cpuRejected.data(), cpuIntegrated.size()
   };
   UfwbppNativeIntegrationOutputV1 gpuOutput{
      sizeof( UfwbppNativeIntegrationOutputV1 ), gpuIntegrated.data(),
      gpuAccepted.data(), gpuRejected.data(), gpuIntegrated.size()
   };
   Require( ufwbpp_native_cpu_masked_weighted_v1(
               &request, &cpuOutput, error.data(), error.size() )
               == UFWBPP_NATIVE_OK,
            error.data() );
   UfwbppNativeExecutionStatsV1 stats{};
   stats.struct_size = sizeof( stats );
   Require( ufwbpp_native_metal_masked_weighted_v1(
               executor, &request, &gpuOutput, &stats,
               error.data(), error.size() ) == UFWBPP_NATIVE_OK,
            error.data() );
   Require( cpuAccepted == gpuAccepted && cpuRejected == gpuRejected,
            "96-frame masked CPU/Metal rejection counts differ" );
   double squareSum = 0;
   float maximum = 0;
   for ( std::size_t pixel = 0; pixel < pixels; ++pixel )
   {
      Require( std::isfinite( cpuIntegrated[pixel] )
                  == std::isfinite( gpuIntegrated[pixel] ),
               "96-frame masked CPU/Metal finite masks differ" );
      const float difference = std::abs(
         cpuIntegrated[pixel] - gpuIntegrated[pixel] );
      maximum = std::max( maximum, difference );
      squareSum += static_cast<double>( difference )*difference;
      Require( static_cast<std::uint32_t>( gpuAccepted[pixel] )
                  + gpuRejected[pixel] == frames,
               "96-frame masked Metal accounting truncated the stack" );
   }
   Require( maximum <= 2.0e-6F
         && std::sqrt( squareSum/pixels ) <= 2.0e-7,
            "96-frame masked Metal numerical gate failed" );
   Require( stats.executed_on_gpu == 1 && stats.gpu_seconds >= 0,
            "96-frame masked C ABI did not report GPU execution" );
   ufwbpp_native_metal_executor_destroy_v1( executor );
}

} // namespace

int main()
{
   try
   {
      TestAbiAndLinearFit();
      TestCapacityGate();
      TestFiveHundredTwelveFramesAreNotTruncated();
      TestOpaqueMetalExecutor();
      TestOpaqueMetalMaskedNinetySixFrames();
      std::cout << "Ultra-Fast WBPP native C ABI tests passed\n";
      return 0;
   }
   catch ( const std::exception& error )
   {
      std::cerr << error.what() << '\n';
      return 1;
   }
}
