#include "openastroflow/OfflineWcsSolver.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace openastroflow::native
{

namespace
{

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kDegreesToRadians = kPi/180;

struct Point
{
   double x = 0;
   double y = 0;
};

Point ProjectTan( double ra, double dec, double ra0, double dec0 )
{
   const double alpha = ra*kDegreesToRadians;
   const double delta = dec*kDegreesToRadians;
   const double alpha0 = ra0*kDegreesToRadians;
   const double delta0 = dec0*kDegreesToRadians;
   const double difference = alpha - alpha0;
   const double denominator = std::sin( delta )*std::sin( delta0 )
      + std::cos( delta )*std::cos( delta0 )*std::cos( difference );
   if ( denominator <= 0 )
      return { std::numeric_limits<double>::quiet_NaN(),
               std::numeric_limits<double>::quiet_NaN() };
   return {
      std::cos( delta )*std::sin( difference )/denominator
         / kDegreesToRadians,
      (std::sin( delta )*std::cos( delta0 )
       - std::cos( delta )*std::sin( delta0 )*std::cos( difference ))
         / denominator/kDegreesToRadians
   };
}

CatalogSkyStar UnprojectTan( double xi, double eta,
                             double ra0, double dec0 )
{
   xi *= kDegreesToRadians;
   eta *= kDegreesToRadians;
   const double alpha0 = ra0*kDegreesToRadians;
   const double delta0 = dec0*kDegreesToRadians;
   const double denominator = std::cos( delta0 )
                            - eta*std::sin( delta0 );
   double ra = alpha0 + std::atan2( xi, denominator );
   const double dec = std::atan2(
      std::sin( delta0 ) + eta*std::cos( delta0 ),
      std::sqrt( denominator*denominator + xi*xi ) );
   ra /= kDegreesToRadians;
   while ( ra < 0 ) ra += 360;
   while ( ra >= 360 ) ra -= 360;
   return { ra, dec/kDegreesToRadians, 0 };
}

double Determinant( double a, double b, double c, double d )
{
   return a*d - b*c;
}

std::array<double, 3> Solve3x3(
   std::array<std::array<double, 3>, 3> matrix,
   std::array<double, 3> right )
{
   for ( unsigned column = 0; column < 3; ++column )
   {
      unsigned pivot = column;
      for ( unsigned row = column + 1; row < 3; ++row )
         if ( std::abs( matrix[row][column] )
            > std::abs( matrix[pivot][column] ) )
            pivot = row;
      if ( std::abs( matrix[pivot][column] ) < 1e-18 )
         throw std::runtime_error(
            "Ultra-Fast WBPP native WCS fit has a singular normal matrix" );
      std::swap( matrix[pivot], matrix[column] );
      std::swap( right[pivot], right[column] );
      const double divisor = matrix[column][column];
      for ( unsigned item = column; item < 3; ++item )
         matrix[column][item] /= divisor;
      right[column] /= divisor;
      for ( unsigned row = 0; row < 3; ++row )
         if ( row != column )
         {
            const double factor = matrix[row][column];
            for ( unsigned item = column; item < 3; ++item )
               matrix[row][item] -= factor*matrix[column][item];
            right[row] -= factor*right[column];
         }
   }
   return right;
}

TanWcs FitAffine( const std::vector<WcsCorrespondence>& matches,
                  double tangentRa,
                  double tangentDec )
{
   if ( matches.size() < 3 )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native TAN fit needs at least three correspondences" );
   double meanX = 0;
   double meanY = 0;
   for ( const WcsCorrespondence& match : matches )
   {
      meanX += match.image.x;
      meanY += match.image.y;
   }
   meanX /= matches.size();
   meanY /= matches.size();

   std::array<std::array<double, 3>, 3> normal = {};
   std::array<double, 3> rightXi = {};
   std::array<double, 3> rightEta = {};
   for ( const WcsCorrespondence& match : matches )
   {
      const Point plane = ProjectTan(
         match.sky.ra, match.sky.dec, tangentRa, tangentDec );
      if ( !std::isfinite( plane.x ) || !std::isfinite( plane.y ) )
         throw std::invalid_argument(
            "Ultra-Fast WBPP native TAN fit has an unprojectable catalog point" );
      const std::array<double, 3> row = {
         match.image.x - meanX, match.image.y - meanY, 1
      };
      for ( unsigned i = 0; i < 3; ++i )
      {
         rightXi[i] += row[i]*plane.x;
         rightEta[i] += row[i]*plane.y;
         for ( unsigned j = 0; j < 3; ++j )
            normal[i][j] += row[i]*row[j];
      }
   }
   const std::array<double, 3> xi = Solve3x3( normal, rightXi );
   const std::array<double, 3> eta = Solve3x3( normal, rightEta );
   const double determinant = Determinant(
      xi[0], xi[1], eta[0], eta[1] );
   if ( std::abs( determinant ) < 1e-15 )
      throw std::runtime_error( "Ultra-Fast WBPP native TAN fit is degenerate" );

   TanWcs wcs;
   wcs.referenceRa = tangentRa;
   wcs.referenceDec = tangentDec;
   wcs.cd11 = xi[0];
   wcs.cd12 = xi[1];
   wcs.cd21 = eta[0];
   wcs.cd22 = eta[1];
   wcs.referencePixelX = meanX
      + (-wcs.cd22*xi[2] + wcs.cd12*eta[2])/determinant;
   wcs.referencePixelY = meanY
      + ( wcs.cd21*xi[2] - wcs.cd11*eta[2])/determinant;
   return wcs;
}

double Residual( const TanWcs& wcs, const WcsCorrespondence& match )
{
   const DetectedStar predicted = wcs.SkyToPixel(
      match.sky.ra, match.sky.dec );
   return std::hypot(
      predicted.x - match.image.x, predicted.y - match.image.y );
}

WcsSolveResult Summarize( TanWcs wcs,
                          std::vector<WcsCorrespondence> matches )
{
   std::vector<double> residuals;
   residuals.reserve( matches.size() );
   long double squareSum = 0;
   double maximum = 0;
   for ( const WcsCorrespondence& match : matches )
   {
      const double residual = Residual( wcs, match );
      residuals.push_back( residual );
      squareSum += residual*residual;
      maximum = std::max( maximum, residual );
   }
   std::sort( residuals.begin(), residuals.end() );
   WcsSolveResult result;
   result.wcs = wcs;
   result.matches = std::move( matches );
   result.rmsPixels = std::sqrt(
      static_cast<double>( squareSum/residuals.size() ) );
   result.medianPixels = residuals.size() % 2
      ? residuals[residuals.size()/2]
      : 0.5*(residuals[residuals.size()/2 - 1]
           + residuals[residuals.size()/2]);
   result.maximumPixels = maximum;
   return result;
}

struct Similarity
{
   double m11 = 0;
   double m12 = 0;
   double m21 = 0;
   double m22 = 0;
   double tx = 0;
   double ty = 0;

   Point Apply( const Point& point ) const
   {
      return { m11*point.x + m12*point.y + tx,
               m21*point.x + m22*point.y + ty };
   }
};

Similarity PairTransform( const Point& world1,
                          const Point& world2,
                          const DetectedStar& image1,
                          const DetectedStar& image2,
                          bool reflection )
{
   const double wx = world2.x - world1.x;
   const double wy = world2.y - world1.y;
   const double ix = image2.x - image1.x;
   const double iy = image2.y - image1.y;
   const double denominator = wx*wx + wy*wy;
   Similarity result;
   if ( reflection )
   {
      const double a = (ix*wx - iy*wy)/denominator;
      const double b = (ix*wy + iy*wx)/denominator;
      result.m11 = a; result.m12 = b;
      result.m21 = b; result.m22 = -a;
   }
   else
   {
      const double a = (ix*wx + iy*wy)/denominator;
      const double b = (iy*wx - ix*wy)/denominator;
      result.m11 = a; result.m12 = -b;
      result.m21 = b; result.m22 = a;
   }
   result.tx = image1.x - result.m11*world1.x
                           - result.m12*world1.y;
   result.ty = image1.y - result.m21*world1.x
                           - result.m22*world1.y;
   return result;
}

std::int64_t CellKey( int x, int y )
{
   const std::uint64_t key =
      (static_cast<std::uint64_t>( static_cast<std::uint32_t>( x ) ) << 32)
      | static_cast<std::uint32_t>( y );
   return std::bit_cast<std::int64_t>( key );
}

class DetectionGrid
{
public:
   DetectionGrid( const std::vector<DetectedStar>& stars, double cellSize )
      : stars_( stars ), cellSize_( cellSize )
   {
      for ( std::size_t i = 0; i < stars.size(); ++i )
         cells_[CellKey( Cell( stars[i].x ), Cell( stars[i].y ) )]
            .push_back( i );
   }

   std::pair<std::size_t, double> Nearest( double x, double y,
                                           double maximumDistance ) const
   {
      std::size_t best = std::numeric_limits<std::size_t>::max();
      double bestSquare = maximumDistance*maximumDistance;
      const int cx = Cell( x );
      const int cy = Cell( y );
      for ( int dy = -1; dy <= 1; ++dy )
         for ( int dx = -1; dx <= 1; ++dx )
         {
            const auto iterator = cells_.find( CellKey( cx + dx, cy + dy ) );
            if ( iterator == cells_.end() )
               continue;
            for ( std::size_t index : iterator->second )
            {
               const double square =
                  (stars_[index].x - x)*(stars_[index].x - x)
                + (stars_[index].y - y)*(stars_[index].y - y);
               if ( square <= bestSquare )
               {
                  bestSquare = square;
                  best = index;
               }
            }
         }
      return { best, bestSquare };
   }

private:
   int Cell( double coordinate ) const
   {
      return static_cast<int>( std::floor( coordinate/cellSize_ ) );
   }

   const std::vector<DetectedStar>& stars_;
   double cellSize_;
   std::unordered_map<std::int64_t, std::vector<std::size_t>> cells_;
};

struct WorkCatalogStar
{
   CatalogSkyStar sky;
   Point plane;
};

std::vector<WcsCorrespondence> UniqueMatches(
   const std::vector<DetectedStar>& detections,
   const std::vector<WorkCatalogStar>& catalog,
   const DetectionGrid& grid,
   const Similarity* similarity,
   const TanWcs* wcs,
   double tolerance,
   std::uint32_t width,
   std::uint32_t height )
{
   struct Candidate
   {
      std::size_t detection = 0;
      std::size_t catalog = 0;
      double square = 0;
   };
   std::vector<Candidate> candidates;
   for ( std::size_t i = 0; i < catalog.size(); ++i )
   {
      Point predicted;
      if ( similarity != nullptr )
         predicted = similarity->Apply( catalog[i].plane );
      else
      {
         const DetectedStar image = wcs->SkyToPixel(
            catalog[i].sky.ra, catalog[i].sky.dec );
         predicted = { image.x, image.y };
      }
      if ( predicted.x < -tolerance || predicted.y < -tolerance
        || predicted.x > width - 1 + tolerance
        || predicted.y > height - 1 + tolerance )
         continue;
      const auto [detection, square] = grid.Nearest(
         predicted.x, predicted.y, tolerance );
      if ( detection != std::numeric_limits<std::size_t>::max() )
         candidates.push_back( { detection, i, square } );
   }
   std::sort( candidates.begin(), candidates.end(),
      []( const Candidate& left, const Candidate& right ) {
         return left.square < right.square;
      } );
   std::unordered_set<std::size_t> usedDetections;
   std::unordered_set<std::size_t> usedCatalog;
   std::vector<WcsCorrespondence> result;
   for ( const Candidate& candidate : candidates )
      if ( usedDetections.insert( candidate.detection ).second
        && usedCatalog.insert( candidate.catalog ).second )
         result.push_back( {
            detections[candidate.detection],
            catalog[candidate.catalog].sky
         } );
   return result;
}

std::string Numeric( double value )
{
   std::ostringstream stream;
   stream << std::setprecision( 17 ) << value;
   return stream.str();
}

} // namespace

bool TanWcs::IsValid() const
{
   return std::isfinite( referenceRa ) && referenceRa >= 0 && referenceRa < 360
      && std::isfinite( referenceDec ) && std::abs( referenceDec ) <= 90
      && std::isfinite( referencePixelX )
      && std::isfinite( referencePixelY )
      && std::isfinite( cd11 ) && std::isfinite( cd12 )
      && std::isfinite( cd21 ) && std::isfinite( cd22 )
      && std::abs( Determinant( cd11, cd12, cd21, cd22 ) ) > 1e-15;
}

DetectedStar TanWcs::SkyToPixel( double ra, double dec ) const
{
   if ( !IsValid() )
      throw std::logic_error( "Ultra-Fast WBPP native TAN WCS is invalid" );
   const Point plane = ProjectTan( ra, dec, referenceRa, referenceDec );
   const double determinant = Determinant( cd11, cd12, cd21, cd22 );
   return {
      referencePixelX + (cd22*plane.x - cd12*plane.y)/determinant,
      referencePixelY + (-cd21*plane.x + cd11*plane.y)/determinant,
      0
   };
}

CatalogSkyStar TanWcs::PixelToSky( double x, double y ) const
{
   if ( !IsValid() )
      throw std::logic_error( "Ultra-Fast WBPP native TAN WCS is invalid" );
   const double dx = x - referencePixelX;
   const double dy = y - referencePixelY;
   return UnprojectTan(
      cd11*dx + cd12*dy, cd21*dx + cd22*dy,
      referenceRa, referenceDec );
}

std::vector<FitsWcsCard> TanWcs::FitsCards() const
{
   if ( !IsValid() )
      throw std::logic_error( "Ultra-Fast WBPP native TAN WCS is invalid" );
   return {
      { "WCSAXES", "2", "number of WCS axes" },
      { "CTYPE1", "'RA---TAN'", "gnomonic right ascension" },
      { "CTYPE2", "'DEC--TAN'", "gnomonic declination" },
      { "CUNIT1", "'deg'", "axis unit" },
      { "CUNIT2", "'deg'", "axis unit" },
      { "CRVAL1", Numeric( referenceRa ), "reference RA" },
      { "CRVAL2", Numeric( referenceDec ), "reference Dec" },
      { "CRPIX1", Numeric( referencePixelX + 1 ), "one-based reference pixel" },
      { "CRPIX2", Numeric( referencePixelY + 1 ), "one-based reference pixel" },
      { "CD1_1", Numeric( cd11 ), "degrees per pixel" },
      { "CD1_2", Numeric( cd12 ), "degrees per pixel" },
      { "CD2_1", Numeric( cd21 ), "degrees per pixel" },
      { "CD2_2", Numeric( cd22 ), "degrees per pixel" },
      { "RADESYS", "'ICRS'", "celestial reference frame" },
      { "EQUINOX", "2000.0", "ICRS equinox" }
   };
}

WcsSolveResult FitTanWcs(
   const std::vector<WcsCorrespondence>& correspondences,
   const WcsFitOptions& options )
{
   if ( !std::isfinite( options.tangentRa ) || options.tangentRa < 0
     || options.tangentRa >= 360 || !std::isfinite( options.tangentDec )
     || std::abs( options.tangentDec ) >= 89
     || !std::isfinite( options.rejectionPixels )
     || options.rejectionPixels <= 0 || options.minimumMatches < 3
     || correspondences.size() < options.minimumMatches )
      throw std::invalid_argument( "Ultra-Fast WBPP native TAN fit options are invalid" );

   std::vector<WcsCorrespondence> active = correspondences;
   TanWcs wcs;
   for ( std::uint32_t iteration = 0;
         iteration <= options.rejectionIterations;
         ++iteration )
   {
      wcs = FitAffine( active, options.tangentRa, options.tangentDec );
      if ( iteration == options.rejectionIterations )
         break;
      std::vector<double> residuals;
      residuals.reserve( active.size() );
      for ( const WcsCorrespondence& match : active )
         residuals.push_back( Residual( wcs, match ) );
      std::vector<double> ordered = residuals;
      const std::size_t middle = ordered.size()/2;
      std::nth_element(
         ordered.begin(), ordered.begin() + middle, ordered.end() );
      const double median = ordered[middle];
      for ( double& residual : ordered )
         residual = std::abs( residual - median );
      std::nth_element(
         ordered.begin(), ordered.begin() + middle, ordered.end() );
      const double robustSigma = 1.4826*ordered[middle];
      const double threshold = std::max(
         options.rejectionPixels, median + 4*robustSigma );
      std::vector<WcsCorrespondence> kept;
      kept.reserve( active.size() );
      for ( std::size_t i = 0; i < active.size(); ++i )
         if ( residuals[i] <= threshold )
            kept.push_back( active[i] );
      if ( kept.size() < options.minimumMatches || kept.size() == active.size() )
         break;
      active = std::move( kept );
   }
   return Summarize( wcs, std::move( active ) );
}

WcsSolveResult SolveTanWcs(
   const std::vector<DetectedStar>& detections,
   const std::vector<CatalogSkyStar>& catalog,
   const WcsMatchOptions& options )
{
   if ( options.imageWidth == 0 || options.imageHeight == 0
     || !std::isfinite( options.approximateRa )
     || options.approximateRa < 0 || options.approximateRa >= 360
     || !std::isfinite( options.approximateDec )
     || std::abs( options.approximateDec ) >= 75
     || options.focalLengthMm <= 0 || options.pixelSizeMicrons <= 0
     || options.scaleRelativeTolerance <= 0
     || options.hypothesisMatchPixels <= 0
     || options.finalMatchPixels <= 0
     || options.minimumMatches < 4
     || detections.size() < options.minimumMatches
     || catalog.size() < options.minimumMatches )
      throw std::invalid_argument( "Ultra-Fast WBPP native WCS match options are invalid" );

   std::vector<DetectedStar> imageStars = detections;
   std::sort( imageStars.begin(), imageStars.end(),
      []( const DetectedStar& left, const DetectedStar& right ) {
         return left.flux > right.flux;
      } );
   if ( imageStars.size() > options.maximumImageStars )
      imageStars.resize( options.maximumImageStars );

   std::vector<WorkCatalogStar> skyStars;
   skyStars.reserve( catalog.size() );
   for ( const CatalogSkyStar& sky : catalog )
   {
      const Point plane = ProjectTan(
         sky.ra, sky.dec, options.approximateRa, options.approximateDec );
      if ( std::isfinite( plane.x ) && std::isfinite( plane.y ) )
         skyStars.push_back( { sky, plane } );
   }
   std::sort( skyStars.begin(), skyStars.end(),
      []( const WorkCatalogStar& left, const WorkCatalogStar& right ) {
         return left.sky.magnitude < right.sky.magnitude;
      } );
   if ( skyStars.size() > options.maximumCatalogStars )
      skyStars.resize( options.maximumCatalogStars );

   const std::size_t hypothesisImageCount = std::min<std::size_t>(
      imageStars.size(), options.hypothesisImageStars );
   const std::size_t hypothesisCatalogCount = std::min<std::size_t>(
      skyStars.size(), options.hypothesisCatalogStars );
   const double degreesPerPixel = PixelScaleArcseconds(
      options.focalLengthMm, options.pixelSizeMicrons )/3600;
   const double minimumPairPixels = 0.08*std::min(
      options.imageWidth, options.imageHeight );
   const double maximumPairPixels = std::hypot(
      options.imageWidth, options.imageHeight )*1.1;
   const DetectionGrid grid( imageStars, options.hypothesisMatchPixels );

   Similarity best;
   std::size_t bestCount = 0;
   double bestSquare = std::numeric_limits<double>::infinity();
   std::uint64_t hypotheses = 0;
   for ( std::size_t image1 = 0; image1 < hypothesisImageCount; ++image1 )
      for ( std::size_t image2 = image1 + 1;
            image2 < hypothesisImageCount; ++image2 )
      {
         const double imageDistance = std::hypot(
            imageStars[image2].x - imageStars[image1].x,
            imageStars[image2].y - imageStars[image1].y );
         if ( imageDistance < minimumPairPixels )
            continue;
         for ( std::size_t sky1 = 0; sky1 < hypothesisCatalogCount; ++sky1 )
            for ( std::size_t sky2 = sky1 + 1;
                  sky2 < hypothesisCatalogCount; ++sky2 )
            {
               const double skyDistancePixels = std::hypot(
                  skyStars[sky2].plane.x - skyStars[sky1].plane.x,
                  skyStars[sky2].plane.y - skyStars[sky1].plane.y )
                  / degreesPerPixel;
               if ( skyDistancePixels < minimumPairPixels
                 || skyDistancePixels > maximumPairPixels
                 || std::abs( skyDistancePixels - imageDistance )
                       > options.scaleRelativeTolerance*imageDistance )
                  continue;
               for ( unsigned swap = 0; swap < 2; ++swap )
                  for ( unsigned reflected = 0;
                        reflected <= (options.allowReflection ? 1u : 0u);
                        ++reflected )
                  {
                     const DetectedStar& first = swap
                        ? imageStars[image2] : imageStars[image1];
                     const DetectedStar& second = swap
                        ? imageStars[image1] : imageStars[image2];
                     const Similarity hypothesis = PairTransform(
                        skyStars[sky1].plane, skyStars[sky2].plane,
                        first, second, reflected != 0 );
                     ++hypotheses;
                     std::size_t count = 0;
                     double square = 0;
                     for ( std::size_t sky = 0;
                           sky < hypothesisCatalogCount; ++sky )
                     {
                        const Point predicted = hypothesis.Apply(
                           skyStars[sky].plane );
                        const auto [nearest, distanceSquare] = grid.Nearest(
                           predicted.x, predicted.y,
                           options.hypothesisMatchPixels );
                        if ( nearest != std::numeric_limits<std::size_t>::max() )
                        {
                           ++count;
                           square += distanceSquare;
                        }
                     }
                     if ( count > bestCount
                       || (count == bestCount && square < bestSquare) )
                     {
                        best = hypothesis;
                        bestCount = count;
                        bestSquare = square;
                     }
                  }
            }
      }
   if ( bestCount < options.minimumMatches )
      throw std::runtime_error(
         "Ultra-Fast WBPP native WCS pair matching found too few seed matches" );

   std::vector<WcsCorrespondence> matches = UniqueMatches(
      imageStars, skyStars, grid, &best, nullptr,
      options.hypothesisMatchPixels,
      options.imageWidth, options.imageHeight );
   WcsFitOptions fitOptions;
   fitOptions.tangentRa = options.approximateRa;
   fitOptions.tangentDec = options.approximateDec;
   fitOptions.rejectionPixels = options.finalMatchPixels;
   fitOptions.minimumMatches = options.minimumMatches;
   WcsSolveResult result = FitTanWcs( matches, fitOptions );

   for ( unsigned iteration = 0; iteration < 3; ++iteration )
   {
      const CatalogSkyStar center = result.wcs.PixelToSky(
         0.5*(options.imageWidth - 1),
         0.5*(options.imageHeight - 1) );
      fitOptions.tangentRa = center.ra;
      fitOptions.tangentDec = center.dec;
      matches = UniqueMatches(
         imageStars, skyStars, grid, nullptr, &result.wcs,
         options.finalMatchPixels,
         options.imageWidth, options.imageHeight );
      if ( matches.size() < options.minimumMatches )
         break;
      result = FitTanWcs( matches, fitOptions );
   }
   result.hypothesesTested = hypotheses;
   return result;
}

double PixelScaleArcseconds( double focalLengthMm,
                             double pixelSizeMicrons )
{
   if ( !std::isfinite( focalLengthMm ) || focalLengthMm <= 0
     || !std::isfinite( pixelSizeMicrons ) || pixelSizeMicrons <= 0 )
      throw std::invalid_argument(
         "Ultra-Fast WBPP native pixel-scale inputs must be positive" );
   return std::atan( pixelSizeMicrons/1000/focalLengthMm )
      / kDegreesToRadians*3600;
}

} // namespace openastroflow::native
