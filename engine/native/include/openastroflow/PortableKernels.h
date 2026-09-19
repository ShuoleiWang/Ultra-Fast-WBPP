#ifndef OPENASTROFLOW_NATIVE_PORTABLEKERNELS_H
#define OPENASTROFLOW_NATIVE_PORTABLEKERNELS_H

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace openastroflow::native
{

// Multithreaded portable CPU kernels for the ordinary mono pipeline.
//
// Every kernel reproduces the arithmetic of the NumPy reference implementation
// in the Python engine operation for operation: the same Float32/Float64
// intermediate types, the same evaluation order, and the same NaN policy. A
// differential test in the Python package therefore expects value-identical
// output (the sign of an exact zero is the only permitted difference). The
// kernels never change scientific parameters; they only change where the
// work runs.

// Output-to-input homogeneous map in pixel coordinates. The parenthesized
// evaluation order is part of the contract because it reproduces the NumPy
// broadcast expression used by the reference resampler:
//   xIn = ((m00*xOut) + (m01*yOut)) + m02
//   yIn = ((m10*xOut) + (m11*yOut)) + m12
// The map is affine when m20 = m21 = 0 and m22 = 1 (the default). Otherwise
// it is projective and both coordinates are divided by the same denominator,
// again in the reference order:
//   w   = ((m20*xOut) + (m21*yOut)) + m22
//   xIn = xIn/w, yIn = yIn/w
struct AffineInverse
{
   double m00 = 1;
   double m01 = 0;
   double m02 = 0;
   double m10 = 0;
   double m11 = 1;
   double m12 = 0;
   double m20 = 0;
   double m21 = 0;
   double m22 = 1;

   bool IsAffine() const noexcept
   {
      return m20 == 0.0 && m21 == 0.0 && m22 == 1.0;
   }
};

struct WarpLanczos3Request
{
   // Native-endian Float32 physical source values, row-major,
   // sourceHeight*sourceWidth samples. Nonfinite samples are permitted and
   // invalidate every output sample whose nonzero support touches them.
   std::span<const float> source;
   std::uint32_t sourceWidth = 0;
   std::uint32_t sourceHeight = 0;
   AffineInverse inverse;
   std::uint32_t outputWidth = 0;
   std::uint32_t firstRow = 0;
   std::uint32_t rowCount = 0;
   // Declared normalized-unit scale of the numeric domain; the upper clamp is
   // max(support maximum, domainScale), the lower clamp min(support minimum, 0).
   float domainScale = 0;
   std::uint32_t threads = 1;

   void Validate() const;
   std::size_t OutputPixels() const;
};

// Normalized, domain-bounded 6x6 Lanczos-3 resampling of one band of output
// rows. Writes rowCount*outputWidth Float32 samples; NaN marks samples whose
// coordinates fall outside the two-pixel interpolation margin or whose active
// support contains a nonfinite source pixel.
void WarpLanczos3Clamped( const WarpLanczos3Request& request,
                          std::span<float> destination );

struct MadRejectionRequest
{
   // Frame-major Float32 samples: frameCount*rowCount*width.
   std::span<const float> frameMajorSamples;
   std::uint32_t frameCount = 0;
   std::uint32_t rowCount = 0;
   std::uint32_t width = 0;
   float sigmaClip = 4.0F;
   std::uint32_t minimumRejectionFrames = 3;
   float groupSigmaFloor = 1.0e-7F;
   float absoluteFloor = 1.0e-7F;
   // Already multiplied: epsilonFactor*float32 epsilon.
   float epsilonFloor = 16.0F*1.1920928955078125e-07F;
   std::uint32_t threads = 1;
   // v2 scale model. frameScales holds one finite positive Float32 factor per
   // frame (empty: every factor is 1) that turns the pooled mixture sigma into
   // the frame's own noise; poolHalfWidth is the half width of the same-row
   // window whose per-pixel MADs are pooled (0: no pooling). Empty scales and
   // half width 0 reproduce the v1 per-pixel decisions exactly.
   std::span<const float> frameScales;
   std::uint32_t poolHalfWidth = 0;

   void Validate() const;
   std::size_t TilePixels() const;
   bool UsesScaleModel() const noexcept;
};

// Per-pixel median/MAD sigma clipping over the complete frame stack. Writes
// one accepted flag per sample (frame-major, 1 accepted / 0 rejected or
// unavailable) and the per-pixel Float32 centre (NaN without finite samples).
//
// With the v2 scale model the threshold of frame j at a pixel is
//   sigmaClip * max( sqrt( (s_j*sigmaPool)^2 + max(sigmaPix^2 - sigmaPool^2, 0) ),
//                    groupSigmaFloor, numericalFloor )
// where sigmaPix = 1.4826*MAD of the pixel, sigmaPool = 1.4826*nanmedian of
// the MADs of the pixels [x-h, x+h] of the same row (pixels with too few
// finite samples excluded), and s_j the frame's factor. Float32 throughout,
// in exactly this evaluation order (the NumPy reference does the same).
void MadRejectionMask( const MadRejectionRequest& request,
                       std::span<std::uint8_t> accepted,
                       std::span<float> center );

struct MaskedMeanRequest
{
   std::span<const float> frameMajorSamples;
   std::span<const std::uint8_t> frameMajorAccepted;
   std::span<const double> frameWeights;
   // Optional frame-major per-sample weights with the layout of the samples
   // (empty = every sample weighs 1). The effective weight of a sample is the
   // Float64 product frameWeight*sampleWeight; region weight maps of the
   // unattended selection use this to blend out occluded or cloudy areas.
   std::span<const float> frameMajorSampleWeights;
   std::uint32_t frameCount = 0;
   std::uint32_t rowCount = 0;
   std::uint32_t width = 0;
   std::uint32_t threads = 1;

   void Validate() const;
   std::size_t TilePixels() const;
};

struct MaskedMeanOutput
{
   std::span<float> integrated;
   std::span<std::uint16_t> acceptedSamples;
   std::span<std::uint16_t> rejectedSamples;
};

// Exact full-stack weighted mean with Float64 accumulation in frame order.
// Pixels without accepted samples (or without positive effective weight) are
// NaN. rejectedSamples counts finite samples that were not accepted; nonfinite
// samples count in neither total. Without sample weights the arithmetic is
// value-identical to the NumPy reference reduction; with them the reference is
// numerator = sum(where(accepted, value, 0)*(frameWeight*sampleWeight)).
void MaskedWeightedMean( const MaskedMeanRequest& request,
                         const MaskedMeanOutput& output );

struct TileOffsetRequest
{
   // Paired Float64 samples for every tile, concatenated; tile t occupies
   // [boundaries[t], boundaries[t+1]).  Nonfinite pairs are ignored.
   std::span<const double> target;
   std::span<const double> reference;
   std::span<const std::uint64_t> boundaries;
   std::uint32_t tileCount = 0;
   double scale = 1.0;
   double lowerQuantile = 0.05;
   double upperQuantile = 0.95;
   std::uint32_t minimumSamples = 512;
   double residualClipSigma = 3.0;
   std::uint32_t threads = 1;

   void Validate() const;
};

struct TileOffsetOutput
{
   std::span<double> offset;          // NaN when the tile is invalid
   std::span<std::uint32_t> count;    // residual samples that formed offset
   std::span<double> residualMad;     // NaN when the tile is invalid
   std::span<std::uint8_t> valid;     // 1 valid, 0 invalid
};

// Per-tile additive offset between reference and scale*target, reproducing
// global_normalization._tile_offset: joint 5-95% quantile selection with
// NumPy's linear interpolation, median/MAD clipping at residualClipSigma,
// and the median residual plus its robust dispersion.  Tiles are independent
// and run concurrently.
void TileOffsets( const TileOffsetRequest& request, const TileOffsetOutput& output );

struct RadonPeakRequest
{
   // Row-major Float32 residual samples and 0/1 line weights of one frame in
   // one orientation, height*width each.  The frame is embedded in a dyadic
   // canvas of `size` rows (a power of two, rows height..size-1 zero) whose
   // columns are padded by `size` zeros on both sides, exactly as
   // transient_rejection.fast_radon_levels lays it out.
   std::span<const float> image;
   std::span<const std::uint8_t> weight;
   std::uint32_t width = 0;
   std::uint32_t height = 0;
   std::uint32_t size = 0;
   // First reported block length: a power of two in [2, size].
   std::uint32_t minimumRows = 0;
   float detectionZ = 6.5F;
   // A line is valid when its weight count reaches max(minimumCount,
   // minimumCoverage*n) at block length n.
   double minimumCoverage = 0.6;
   double minimumCount = 8.0;
   // Levels with fewer valid lines keep their unstandardised z.
   std::uint32_t minimumScaleSamples = 64;
   std::uint32_t threads = 1;

   void Validate() const;
};

struct RadonPeak
{
   std::uint32_t level = 0;      // block length n
   std::uint32_t block = 0;      // dyadic block b: rows b*n .. b*n+n-1
   std::uint32_t shiftIndex = 0; // s + n - 1 for the column shift s
   std::uint32_t column = 0;     // padded column x (image column x - size)
   float z = 0;
};

// Multi-scale line peaks of transient_rejection._candidate_lines for one
// orientation: the dyadic fast Radon transform of the samples and of the
// weights (Float32 sums in the reference recursion order), per level the
// standardised z of every valid line (Float32 sum/sqrt(count) divided by
// 1.4826 times the Float32 median absolute deviation about the Float32
// median, when at least minimumScaleSamples lines are valid), and every
// line whose z reaches detectionZ and is the maximum of its (5 shifts x 7
// columns) neighbourhood within the block.  Peaks are appended in level
// order and, within a level, in (block, shiftIndex, column) row-major
// order, which is the NumPy nonzero order of the reference.
void RadonLinePeaks( const RadonPeakRequest& request, std::vector<RadonPeak>& peaks );

// Hardware concurrency clamped to [1, 64]; never zero.
std::uint32_t DefaultKernelThreads() noexcept;

} // namespace openastroflow::native

#endif // OPENASTROFLOW_NATIVE_PORTABLEKERNELS_H
