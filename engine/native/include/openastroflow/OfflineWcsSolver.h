#ifndef OPENASTROFLOW_NATIVE_OFFLINEWCSSOLVER_H
#define OPENASTROFLOW_NATIVE_OFFLINEWCSSOLVER_H

#include <cstdint>
#include <string>
#include <vector>

namespace openastroflow::native
{

struct DetectedStar
{
   double x = 0; // zero-based image coordinate
   double y = 0;
   float flux = 0;
};

struct CatalogSkyStar
{
   double ra = 0;
   double dec = 0;
   float magnitude = 0;
};

struct WcsCorrespondence
{
   DetectedStar image;
   CatalogSkyStar sky;
};

struct FitsWcsCard
{
   std::string keyword;
   std::string value;
   std::string comment;
};

struct TanWcs
{
   double referenceRa = 0;
   double referenceDec = 0;
   double referencePixelX = 0; // zero-based internally; CRPIX is +1
   double referencePixelY = 0;
   double cd11 = 0; // degrees per pixel
   double cd12 = 0;
   double cd21 = 0;
   double cd22 = 0;

   bool IsValid() const;
   DetectedStar SkyToPixel( double ra, double dec ) const;
   CatalogSkyStar PixelToSky( double x, double y ) const;
   std::vector<FitsWcsCard> FitsCards() const;
};

struct WcsFitOptions
{
   double tangentRa = 0;
   double tangentDec = 0;
   double rejectionPixels = 3;
   std::uint32_t rejectionIterations = 3;
   std::uint32_t minimumMatches = 6;
};

struct WcsMatchOptions
{
   double approximateRa = 0;
   double approximateDec = 0;
   std::uint32_t imageWidth = 0;
   std::uint32_t imageHeight = 0;
   double focalLengthMm = 0;
   double pixelSizeMicrons = 0;

   double scaleRelativeTolerance = 0.04;
   double hypothesisMatchPixels = 5;
   double finalMatchPixels = 3;
   std::uint32_t hypothesisImageStars = 30;
   std::uint32_t hypothesisCatalogStars = 55;
   std::uint32_t maximumImageStars = 100;
   std::uint32_t maximumCatalogStars = 160;
   std::uint32_t minimumMatches = 8;
   bool allowReflection = true;
};

struct WcsSolveResult
{
   TanWcs wcs;
   std::vector<WcsCorrespondence> matches;
   double rmsPixels = 0;
   double medianPixels = 0;
   double maximumPixels = 0;
   std::uint64_t hypothesesTested = 0;
};

// Robust linear TAN fit from known image/catalog correspondences.
WcsSolveResult FitTanWcs(
   const std::vector<WcsCorrespondence>& correspondences,
   const WcsFitOptions& options );

// Seed-scale pair matching solves unknown rotation and optional reflection,
// then refines a least-squares TAN WCS with unique nearest-neighbour matches.
WcsSolveResult SolveTanWcs(
   const std::vector<DetectedStar>& detections,
   const std::vector<CatalogSkyStar>& catalog,
   const WcsMatchOptions& options );

double PixelScaleArcseconds( double focalLengthMm,
                             double pixelSizeMicrons );

} // namespace openastroflow::native

#endif // OPENASTROFLOW_NATIVE_OFFLINEWCSSOLVER_H
