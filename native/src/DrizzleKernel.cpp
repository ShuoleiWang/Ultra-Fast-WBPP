#include "ufwbpp/PortableKernels.h"
#include "ParallelRange.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// The per-drop helpers below are called from both bands; with one caller
// the compiler inlined them, with two it keeps them out of line, which
// costs the reference-grid band 6 % on a 26 MP frame.
#if defined(_MSC_VER)
#define UFWBPP_DRIZZLE_INLINE __forceinline
#else
#define UFWBPP_DRIZZLE_INLINE inline __attribute__((always_inline))
#endif

namespace ufwbpp::native
{

namespace
{

using detail::ParallelRange;

constexpr std::size_t DrizzleRowGrain = 16;
// A lattice map has no closed inverse: every source row's mapped row is
// sampled each RowSampleStep pixels, and a run of columns is visited when
// its samples, widened by RowSlack output pixels, reach a thread's rows.
constexpr std::uint32_t RowSampleStep = 8;
constexpr double RowSlack = 1.0;

struct Point
{
   double x;
   double y;
};

// Homogeneous 3x3 map applied to pixel-centre coordinates.
struct Homography
{
   double m[9];

   Point Map( double x, double y ) const
   {
      const double w = m[6]*x + m[7]*y + m[8];
      return { (m[0]*x + m[1]*y + m[2])/w, (m[3]*x + m[4]*y + m[5])/w };
   }

   bool IsAffine() const
   {
      return m[6] == 0.0 && m[7] == 0.0 && m[8] == 1.0;
   }

   Homography Inverse() const
   {
      const double a = m[0], b = m[1], c = m[2];
      const double d = m[3], e = m[4], f = m[5];
      const double g = m[6], h = m[7], i = m[8];
      const double det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g);
      if ( !std::isfinite( det ) || det == 0.0 )
         throw std::invalid_argument( "drizzle forward map is singular" );
      Homography inverse;
      inverse.m[0] = (e*i - f*h)/det;
      inverse.m[1] = (c*h - b*i)/det;
      inverse.m[2] = (b*f - c*e)/det;
      inverse.m[3] = (f*g - d*i)/det;
      inverse.m[4] = (a*i - c*g)/det;
      inverse.m[5] = (c*d - a*f)/det;
      inverse.m[6] = (d*h - e*g)/det;
      inverse.m[7] = (b*g - a*h)/det;
      inverse.m[8] = (a*e - b*d)/det;
      return inverse;
   }
};

// Bilinear node lattice of output coordinates over the forward map's
// domain (DrizzleRequest): points beyond it extrapolate its edge cell.
struct LatticeMap
{
   const double* u = nullptr;
   const double* v = nullptr;
   double x0 = 0.0;
   double y0 = 0.0;
   double spacing = 1.0;
   std::int64_t columns = 0;
   std::int64_t rows = 0;

   UFWBPP_DRIZZLE_INLINE Point Map( const Point& p ) const
   {
      const double gx = (p.x - x0)/spacing;
      const double gy = (p.y - y0)/spacing;
      if ( !std::isfinite( gx ) || !std::isfinite( gy ) )
         return { std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN() };
      const double cellX = std::floor( std::min( std::max( gx, 0.0 ), static_cast<double>( columns - 2 ) ) );
      const double cellY = std::floor( std::min( std::max( gy, 0.0 ), static_cast<double>( rows - 2 ) ) );
      const double fx = gx - cellX;
      const double fy = gy - cellY;
      const std::size_t n00 = static_cast<std::size_t>( cellY )*static_cast<std::size_t>( columns )
                            + static_cast<std::size_t>( cellX );
      const std::size_t n10 = n00 + 1;
      const std::size_t n01 = n00 + static_cast<std::size_t>( columns );
      const std::size_t n11 = n01 + 1;
      const auto blend = [&]( const double* nodes )
      {
         const double top = nodes[n00] + fx*(nodes[n10] - nodes[n00]);
         const double bottom = nodes[n01] + fx*(nodes[n11] - nodes[n01]);
         return top + fy*(bottom - top);
      };
      return { blend( u ), blend( v ) };
   }

   // The four corners of one drop, which almost always lie in one cell:
   // its nodes are loaded once, and every point is blended with the
   // arithmetic of Map() (identical results).  Corners in different cells,
   // or outside the finite plane, go through Map() one by one.
   UFWBPP_DRIZZLE_INLINE void MapQuad( const Point* in, Point* out ) const
   {
      std::size_t n00[4];
      double fx[4], fy[4];
      bool shared = true;
      for ( int i = 0; i < 4 && shared; ++i )
      {
         const double gx = (in[i].x - x0)/spacing;
         const double gy = (in[i].y - y0)/spacing;
         if ( !std::isfinite( gx ) || !std::isfinite( gy ) )
         {
            shared = false;
            break;
         }
         const double cellX = std::floor( std::min( std::max( gx, 0.0 ), static_cast<double>( columns - 2 ) ) );
         const double cellY = std::floor( std::min( std::max( gy, 0.0 ), static_cast<double>( rows - 2 ) ) );
         fx[i] = gx - cellX;
         fy[i] = gy - cellY;
         n00[i] = static_cast<std::size_t>( cellY )*static_cast<std::size_t>( columns )
                + static_cast<std::size_t>( cellX );
         shared = n00[i] == n00[0];
      }
      if ( !shared )
      {
         for ( int i = 0; i < 4; ++i )
            out[i] = Map( in[i] );
         return;
      }
      const std::size_t n10 = n00[0] + 1;
      const std::size_t n01 = n00[0] + static_cast<std::size_t>( columns );
      const std::size_t n11 = n01 + 1;
      const double u00 = u[n00[0]], u10 = u[n10], u01 = u[n01], u11 = u[n11];
      const double v00 = v[n00[0]], v10 = v[n10], v01 = v[n01], v11 = v[n11];
      for ( int i = 0; i < 4; ++i )
      {
         const double uTop = u00 + fx[i]*(u10 - u00);
         const double uBottom = u01 + fx[i]*(u11 - u01);
         const double vTop = v00 + fx[i]*(v10 - v00);
         const double vBottom = v01 + fx[i]*(v11 - v01);
         out[i] = { uTop + fy[i]*(uBottom - uTop), vTop + fy[i]*(vBottom - vTop) };
      }
   }
};

// Input pixel centres to output pixel centres through the registration
// homography and then the lattice of a canvas window.  MapQuad maps the
// corners of the square drop of pixel (x, y), counter-clockwise from
// (x - half, y - half).
struct LatticeForwardMap
{
   Homography homography;
   LatticeMap lattice;

   Point Map( double x, double y ) const
   {
      return lattice.Map( homography.Map( x, y ) );
   }

   std::array<Point, 4> MapQuad( double x, double y, double half ) const
   {
      const Point corners[4] = {
         homography.Map( x - half, y - half ), homography.Map( x + half, y - half ),
         homography.Map( x + half, y + half ), homography.Map( x - half, y + half ) };
      std::array<Point, 4> quad;
      lattice.MapQuad( corners, quad.data() );
      return quad;
   }
};

// One source row of a chunk and the columns to visit on it.
struct SourceRun
{
   std::int64_t y;
   std::int64_t x0;
   std::int64_t x1;
};

// Area of a convex polygon (shoelace, orientation-independent).
UFWBPP_DRIZZLE_INLINE double PolygonArea( const Point* vertices, int count )
{
   double twice = 0.0;
   for ( int i = 0; i < count; ++i )
   {
      const Point& p = vertices[i];
      const Point& q = vertices[(i + 1) % count];
      twice += p.x*q.y - q.x*p.y;
   }
   return std::fabs( twice )*0.5;
}

// Sutherland-Hodgman clip of a convex polygon against the half-plane
// keep(point); writes the result to `out` and returns its vertex count.
template <class Keep, class Intersect>
UFWBPP_DRIZZLE_INLINE int ClipEdge( const Point* in, int count, Point* out, Keep keep, Intersect intersect )
{
   int written = 0;
   for ( int i = 0; i < count; ++i )
   {
      const Point& current = in[i];
      const Point& previous = in[(i + count - 1) % count];
      const bool currentIn = keep( current );
      const bool previousIn = keep( previous );
      if ( currentIn )
      {
         if ( !previousIn )
            out[written++] = intersect( previous, current );
         out[written++] = current;
      }
      else if ( previousIn )
         out[written++] = intersect( previous, current );
   }
   return written;
}

// Exact area of the quadrilateral `quad` inside the output pixel square
// [px - 0.5, px + 0.5] x [py - 0.5, py + 0.5].
UFWBPP_DRIZZLE_INLINE double QuadPixelOverlap( const Point quad[4], double px, double py )
{
   const double x0 = px - 0.5, x1 = px + 0.5, y0 = py - 0.5, y1 = py + 0.5;
   Point a[12], b[12];
   int count = 4;
   for ( int i = 0; i < 4; ++i )
      a[i] = quad[i];
   auto lerpX = []( const Point& p, const Point& q, double x )
   {
      const double t = (x - p.x)/(q.x - p.x);
      return Point{ x, p.y + t*(q.y - p.y) };
   };
   auto lerpY = []( const Point& p, const Point& q, double y )
   {
      const double t = (y - p.y)/(q.y - p.y);
      return Point{ p.x + t*(q.x - p.x), y };
   };
   count = ClipEdge( a, count, b, [&]( const Point& p ) { return p.x >= x0; },
                     [&]( const Point& p, const Point& q ) { return lerpX( p, q, x0 ); } );
   if ( count < 3 ) return 0.0;
   count = ClipEdge( b, count, a, [&]( const Point& p ) { return p.x <= x1; },
                     [&]( const Point& p, const Point& q ) { return lerpX( p, q, x1 ); } );
   if ( count < 3 ) return 0.0;
   count = ClipEdge( a, count, b, [&]( const Point& p ) { return p.y >= y0; },
                     [&]( const Point& p, const Point& q ) { return lerpY( p, q, y0 ); } );
   if ( count < 3 ) return 0.0;
   count = ClipEdge( b, count, a, [&]( const Point& p ) { return p.y <= y1; },
                     [&]( const Point& p, const Point& q ) { return lerpY( p, q, y1 ); } );
   if ( count < 3 ) return 0.0;
   return PolygonArea( a, count );
}

// Area of the disc of radius r centred on the origin inside the corner
// region [0, x] x [0, y], x, y >= 0.
UFWBPP_DRIZZLE_INLINE double DiscCornerArea( double x, double y, double r )
{
   x = std::min( x, r );
   y = std::min( y, r );
   if ( x <= 0.0 || y <= 0.0 )
      return 0.0;
   if ( x*x + y*y <= r*r )
      return x*y;
   const double xy = std::sqrt( std::max( r*r - y*y, 0.0 ) ); // where the circle crosses v = y
   const double yx = std::sqrt( std::max( r*r - x*x, 0.0 ) ); // where the circle crosses u = x
   return 0.5*xy*y + 0.5*x*yx + 0.5*r*r*(std::asin( x/r ) - std::asin( xy/r ));
}

UFWBPP_DRIZZLE_INLINE double DiscQuadrantArea( double x, double y, double r )
{
   const double sign = ((x < 0.0) != (y < 0.0)) ? -1.0 : 1.0;
   return sign*DiscCornerArea( std::fabs( x ), std::fabs( y ), r );
}

// Exact area of the disc (cx, cy, r) inside [x0, x1] x [y0, y1].
UFWBPP_DRIZZLE_INLINE double DiscRectangleArea( double cx, double cy, double r,
                          double x0, double x1, double y0, double y1 )
{
   const double ax = x0 - cx, bx = x1 - cx, ay = y0 - cy, by = y1 - cy;
   const double area = DiscQuadrantArea( bx, by, r ) - DiscQuadrantArea( ax, by, r )
                     - DiscQuadrantArea( bx, ay, r ) + DiscQuadrantArea( ax, ay, r );
   return std::max( area, 0.0 );
}

// Bilinear node grid on pixel coordinates, edge-clamped: the arithmetic of
// calibration._add_offset_grid_rows / evaluate_weight_grid_points.
struct NodeGrid
{
   std::span<const double> values;
   std::span<const double> xNodes;
   std::span<const double> yNodes;

   bool Empty() const { return values.empty(); }

   double At( double x, double y ) const
   {
      const auto locate = [&]( std::span<const double> nodes, double value,
                               std::size_t& lo, std::size_t& hi, double& weight )
      {
         const double clipped = std::min( std::max( value, nodes.front() ), nodes.back() );
         std::size_t upper = static_cast<std::size_t>(
            std::upper_bound( nodes.begin(), nodes.end(), clipped ) - nodes.begin() );
         upper = std::min( std::max<std::size_t>( upper, 1 ), nodes.size() - 1 );
         lo = upper - 1;
         hi = upper;
         weight = (clipped - nodes[lo])/(nodes[hi] - nodes[lo]);
      };
      std::size_t xLo, xHi, yLo, yHi;
      double wx, wy;
      locate( xNodes, x, xLo, xHi, wx );
      locate( yNodes, y, yLo, yHi, wy );
      const std::size_t columns = xNodes.size();
      const double top = values[yLo*columns + xLo]*(1.0 - wx) + values[yLo*columns + xHi]*wx;
      const double bottom = values[yHi*columns + xLo]*(1.0 - wx) + values[yHi*columns + xHi]*wx;
      return top*(1.0 - wy) + bottom*wy;
   }
};

struct Normalization
{
   float scale;
   float offset;
   NodeGrid grid;

   float Apply( float value, double referenceX, double referenceY ) const
   {
      float result = value*scale;
      if ( !grid.Empty() )
         result = result + static_cast<float>( grid.At( referenceX, referenceY ) );
      return result + offset;
   }
};

void ValidateGrid( std::span<const double> values, std::span<const double> xNodes,
                   std::span<const double> yNodes, const char* label )
{
   if ( values.empty() )
      return;
   if ( xNodes.size() < 2 || yNodes.size() < 2 || values.size() != xNodes.size()*yNodes.size() )
      throw std::invalid_argument( std::string( label ) + " grid geometry is invalid" );
   for ( double value : values )
      if ( !std::isfinite( value ) )
         throw std::invalid_argument( std::string( label ) + " grid must be finite" );
}

} // namespace

void DrizzleRequest::Validate() const
{
   if ( sourceWidth == 0 || sourceRows == 0 )
      throw std::invalid_argument( "drizzle source band is empty" );
   if ( source.size() != static_cast<std::size_t>( sourceWidth )*sourceRows )
      throw std::invalid_argument( "drizzle source size does not match its geometry" );
   for ( double value : forward )
      if ( !std::isfinite( value ) )
         throw std::invalid_argument( "drizzle forward map must be finite" );
   if ( scale < 1 || scale > 8 )
      throw std::invalid_argument( "drizzle scale must be in [1, 8]" );
   if ( !std::isfinite( pixfrac ) || pixfrac <= 0.0 || pixfrac > 1.0 )
      throw std::invalid_argument( "drizzle pixfrac must be in (0, 1]" );
   if ( kernel != DrizzleKernel::Square && kernel != DrizzleKernel::Circular
     && kernel != DrizzleKernel::Gaussian && kernel != DrizzleKernel::Point )
      throw std::invalid_argument( "drizzle kernel is unknown" );
   if ( !std::isfinite( normalizationScale ) || !std::isfinite( normalizationOffset ) )
      throw std::invalid_argument( "drizzle normalization must be finite" );
   ValidateGrid( grid, gridXNodes, gridYNodes, "drizzle offset" );
   ValidateGrid( weightGrid, weightGridXNodes, weightGridYNodes, "drizzle weight" );
   if ( !mask.empty() )
   {
      if ( maskWidth == 0 || maskHeight == 0
        || mask.size() != static_cast<std::size_t>( maskWidth )*maskHeight )
         throw std::invalid_argument( "drizzle mask size does not match its geometry" );
   }
   if ( HasLattice() )
   {
      const std::size_t nodes = static_cast<std::size_t>( latticeColumns )*latticeRows;
      if ( latticeColumns < 2 || latticeRows < 2 || latticeU.size() != nodes || latticeV.size() != nodes )
         throw std::invalid_argument( "drizzle lattice geometry is invalid" );
      if ( !std::isfinite( latticeX0 ) || !std::isfinite( latticeY0 )
        || !std::isfinite( latticeSpacing ) || latticeSpacing <= 0.0 )
         throw std::invalid_argument( "drizzle lattice origin and spacing must be finite and positive" );
      for ( std::size_t index = 0; index < nodes; ++index )
         if ( !std::isfinite( latticeU[index] ) || !std::isfinite( latticeV[index] ) )
            throw std::invalid_argument( "drizzle lattice nodes must be finite" );
   }
   else if ( !latticeV.empty() )
      throw std::invalid_argument( "drizzle lattice geometry is invalid" );
   if ( channel != 255 && channel > 2 )
      throw std::invalid_argument( "drizzle channel must be 0, 1, 2 or 255" );
   for ( std::uint8_t value : cfaPattern )
      if ( value > 2 )
         throw std::invalid_argument( "drizzle CFA pattern channels must be 0, 1 or 2" );
   if ( !std::isfinite( frameWeight ) || frameWeight < 0.0F )
      throw std::invalid_argument( "drizzle frame weight must be finite and non-negative" );
   if ( outputWidth == 0 || outputRows == 0 )
      throw std::invalid_argument( "drizzle output band is empty" );
   const std::size_t outputCount = static_cast<std::size_t>( outputWidth )*outputRows;
   if ( outputSum.size() != outputCount || outputWeight.size() != outputCount )
      throw std::invalid_argument( "drizzle accumulators do not match the output band" );
   if ( !outputTouched.empty() && outputTouched.size() != outputCount )
      throw std::invalid_argument( "drizzle touch flags do not match the output band" );
   if ( threads < 1 )
      throw std::invalid_argument( "drizzle threads must be positive" );
}

// The band onto the reference grid: every drop through the registration
// homography alone.  Kept as it was before the lattice band below (its
// pixel drop is the same text): restructuring it into a shared body cost
// it 5-6 % on a 26 MP frame.
void BandOnReference( const DrizzleRequest& request )
{
   Homography forward;
   std::copy( request.forward, request.forward + 9, forward.m );
   const Homography inverse = forward.Inverse();
   const double scale = static_cast<double>( request.scale );
   const double half = 0.5*request.pixfrac;
   // Local linear scale of the map (output pixels per input pixel) at the
   // frame centre sizes the circular and Gaussian drops and the margins.
   const double centreX = 0.5*(request.sourceWidth - 1.0);
   const double centreY = request.sourceRow0 + 0.5*(request.sourceRows - 1.0);
   const Point c0 = forward.Map( centreX, centreY );
   const Point cx = forward.Map( centreX + 1.0, centreY );
   const Point cy = forward.Map( centreX, centreY + 1.0 );
   const double jacobian = std::fabs( (cx.x - c0.x)*(cy.y - c0.y) - (cy.x - c0.x)*(cx.y - c0.y) );
   const double localScale = std::sqrt( std::max( jacobian, 1.0e-12 ) );
   const double dropRadius = request.pixfrac*localScale*0.5; // half width of the mapped drop
   const double gaussianSigma = request.pixfrac*localScale/2.3548200450309493; // FWHM = drop width
   const double gaussianReach = 3.0*gaussianSigma;
   double margin = dropRadius*1.5 + 1.0;
   if ( request.kernel == DrizzleKernel::Gaussian )
      margin = gaussianReach + 1.0;
   const Normalization normalization{
      request.normalizationScale, request.normalizationOffset,
      NodeGrid{ request.grid, request.gridXNodes, request.gridYNodes } };
   const NodeGrid weightGrid{ request.weightGrid, request.weightGridXNodes, request.weightGridYNodes };
   const std::uint32_t sourceWidth = request.sourceWidth;
   const std::uint32_t outputWidth = request.outputWidth;
   const float* source = request.source.data();
   const bool useMask = !request.mask.empty();
   const bool selectChannel = request.channel != 255;

   ParallelRange( request.outputRows, request.threads, DrizzleRowGrain,
      [&]( std::size_t bandBegin, std::size_t bandEnd )
      {
         // Output rows [rowLo, rowHi) of this chunk, expanded by the drop
         // margin, mapped back to the input to bound the source rows.
         const double rowLo = static_cast<double>( request.outputRow0 + bandBegin );
         const double rowHi = static_cast<double>( request.outputRow0 + bandEnd - 1 );
         const double vLo = rowLo - 0.5 - margin;
         const double vHi = rowHi + 0.5 + margin;
         const double uLo = -0.5 - margin;
         const double uHi = outputWidth - 0.5 + margin;
         double inputYMin = std::numeric_limits<double>::infinity();
         double inputYMax = -std::numeric_limits<double>::infinity();
         for ( const Point& corner : { Point{ uLo, vLo }, Point{ uHi, vLo }, Point{ uLo, vHi }, Point{ uHi, vHi } } )
         {
            const Point p = inverse.Map( corner.x, corner.y );
            inputYMin = std::min( inputYMin, p.y );
            inputYMax = std::max( inputYMax, p.y );
         }
         if ( !std::isfinite( inputYMin ) || !std::isfinite( inputYMax ) )
         {
            inputYMin = request.sourceRow0;
            inputYMax = request.sourceRow0 + request.sourceRows - 1.0;
         }
         const std::int64_t firstRow = std::max<std::int64_t>(
            request.sourceRow0, static_cast<std::int64_t>( std::floor( inputYMin ) ) - 1 );
         const std::int64_t lastRow = std::min<std::int64_t>(
            static_cast<std::int64_t>( request.sourceRow0 ) + request.sourceRows - 1,
            static_cast<std::int64_t>( std::ceil( inputYMax ) ) + 1 );
         double* sum = request.outputSum.data();
         double* weight = request.outputWeight.data();
         std::uint8_t* touched = request.outputTouched.empty() ? nullptr : request.outputTouched.data();
         const std::int64_t bandRow0 = static_cast<std::int64_t>( request.outputRow0 + bandBegin );
         const std::int64_t bandRow1 = static_cast<std::int64_t>( request.outputRow0 + bandEnd );

         for ( std::int64_t y = firstRow; y <= lastRow; ++y )
         {
            const double fy = static_cast<double>( y );
            // Columns whose mapped row can lie inside the expanded band: the
            // mapped v is monotone along a row (the denominator keeps its
            // sign inside the frame), so the two boundary values bound the run.
            const Point atFirst = forward.Map( 0.0, fy );
            const Point atLast = forward.Map( sourceWidth - 1.0, fy );
            std::int64_t x0 = 0;
            std::int64_t x1 = static_cast<std::int64_t>( sourceWidth ) - 1;
            const double vFirst = atFirst.y, vLast = atLast.y;
            if ( std::fabs( vLast - vFirst ) > 1.0e-9 )
            {
               // Linear interpolation of the row's v(x) is exact for affine
               // maps and a tight approximation for near-affine projective
               // ones; one pixel of slack on each side covers the difference.
               const double slope = (vLast - vFirst)/(sourceWidth - 1.0);
               const double a = (vLo - vFirst)/slope;
               const double b = (vHi - vFirst)/slope;
               const double lo = std::min( a, b ), hi = std::max( a, b );
               if ( hi < -1.0 || lo > sourceWidth )
                  continue;
               x0 = std::max<std::int64_t>( 0, static_cast<std::int64_t>( std::floor( lo ) ) - 1 );
               x1 = std::min<std::int64_t>( sourceWidth - 1, static_cast<std::int64_t>( std::ceil( hi ) ) + 1 );
            }
            else if ( vFirst < vLo || vFirst > vHi )
               continue;
            const float* row = source + static_cast<std::size_t>( y - request.sourceRow0 )*sourceWidth;
            for ( std::int64_t x = x0; x <= x1; ++x )
            {
               const float value = row[x];
               if ( !std::isfinite( value ) )
                  continue;
               if ( selectChannel )
               {
                  const std::uint8_t pixelChannel = request.cfaPattern[((y & 1) << 1) | (x & 1)];
                  if ( pixelChannel != request.channel )
                     continue;
               }
               const double fx = static_cast<double>( x );
               const Point centre = forward.Map( fx, fy );
               if ( !std::isfinite( centre.x ) || !std::isfinite( centre.y ) )
                  continue;
               if ( centre.y < vLo || centre.y > vHi || centre.x < uLo || centre.x > uHi )
                  continue;
               const double referenceX = centre.x/scale;
               const double referenceY = centre.y/scale;
               double pixelWeight = request.frameWeight;
               if ( useMask )
               {
                  const double mx = std::round( referenceX ), my = std::round( referenceY );
                  if ( mx < 0.0 || my < 0.0 || mx >= request.maskWidth || my >= request.maskHeight )
                     continue;
                  if ( request.mask[static_cast<std::size_t>( my )*request.maskWidth + static_cast<std::size_t>( mx )] == 0 )
                     continue;
               }
               if ( !weightGrid.Empty() )
                  pixelWeight *= weightGrid.At( referenceX, referenceY );
               if ( !(pixelWeight > 0.0) )
                  continue;
               const double normalized = static_cast<double>(
                  normalization.Apply( value, referenceX, referenceY ) );
               if ( !std::isfinite( normalized ) )
                  continue;

               auto accumulate = [&]( std::int64_t ou, std::int64_t ov, double area )
               {
                  if ( area <= 0.0 || ov < bandRow0 || ov >= bandRow1 || ou < 0
                    || ou >= static_cast<std::int64_t>( outputWidth ) )
                     return;
                  const std::size_t index =
                     static_cast<std::size_t>( ov - request.outputRow0 )*outputWidth
                     + static_cast<std::size_t>( ou );
                  const double w = pixelWeight*area;
                  sum[index] += w*normalized;
                  weight[index] += w;
                  if ( touched != nullptr )
                     touched[index] = 1;
               };

               switch ( request.kernel )
               {
               case DrizzleKernel::Point:
                  accumulate( static_cast<std::int64_t>( std::round( centre.x ) ),
                              static_cast<std::int64_t>( std::round( centre.y ) ), 1.0 );
                  break;
               case DrizzleKernel::Square:
               {
                  const Point quad[4] = {
                     forward.Map( fx - half, fy - half ), forward.Map( fx + half, fy - half ),
                     forward.Map( fx + half, fy + half ), forward.Map( fx - half, fy + half ) };
                  double minU = quad[0].x, maxU = quad[0].x, minV = quad[0].y, maxV = quad[0].y;
                  for ( int i = 1; i < 4; ++i )
                  {
                     minU = std::min( minU, quad[i].x ); maxU = std::max( maxU, quad[i].x );
                     minV = std::min( minV, quad[i].y ); maxV = std::max( maxV, quad[i].y );
                  }
                  const std::int64_t ou0 = static_cast<std::int64_t>( std::floor( minU + 0.5 ) );
                  const std::int64_t ou1 = static_cast<std::int64_t>( std::floor( maxU + 0.5 ) );
                  const std::int64_t ov0 = std::max( bandRow0, static_cast<std::int64_t>( std::floor( minV + 0.5 ) ) );
                  const std::int64_t ov1 = std::min( bandRow1 - 1, static_cast<std::int64_t>( std::floor( maxV + 0.5 ) ) );
                  for ( std::int64_t ov = ov0; ov <= ov1; ++ov )
                     for ( std::int64_t ou = std::max<std::int64_t>( 0, ou0 );
                           ou <= std::min<std::int64_t>( outputWidth - 1, ou1 ); ++ou )
                        accumulate( ou, ov, QuadPixelOverlap( quad, static_cast<double>( ou ), static_cast<double>( ov ) ) );
                  break;
               }
               case DrizzleKernel::Circular:
               {
                  const double r = dropRadius;
                  const std::int64_t ou0 = std::max<std::int64_t>( 0, static_cast<std::int64_t>( std::floor( centre.x - r + 0.5 ) ) );
                  const std::int64_t ou1 = std::min<std::int64_t>( outputWidth - 1, static_cast<std::int64_t>( std::floor( centre.x + r + 0.5 ) ) );
                  const std::int64_t ov0 = std::max( bandRow0, static_cast<std::int64_t>( std::floor( centre.y - r + 0.5 ) ) );
                  const std::int64_t ov1 = std::min( bandRow1 - 1, static_cast<std::int64_t>( std::floor( centre.y + r + 0.5 ) ) );
                  for ( std::int64_t ov = ov0; ov <= ov1; ++ov )
                     for ( std::int64_t ou = ou0; ou <= ou1; ++ou )
                        accumulate( ou, ov, DiscRectangleArea( centre.x, centre.y, r,
                                                               static_cast<double>( ou ) - 0.5, static_cast<double>( ou ) + 0.5,
                                                               static_cast<double>( ov ) - 0.5, static_cast<double>( ov ) + 0.5 ) );
                  break;
               }
               case DrizzleKernel::Gaussian:
               {
                  // Weights of every output pixel centre within the reach,
                  // normalized so the drop carries the pixel's full weight.
                  const double reach = gaussianReach;
                  const std::int64_t ou0 = std::max<std::int64_t>( 0, static_cast<std::int64_t>( std::floor( centre.x - reach + 0.5 ) ) );
                  const std::int64_t ou1 = std::min<std::int64_t>( outputWidth - 1, static_cast<std::int64_t>( std::floor( centre.x + reach + 0.5 ) ) );
                  const std::int64_t ovAll0 = static_cast<std::int64_t>( std::floor( centre.y - reach + 0.5 ) );
                  const std::int64_t ovAll1 = static_cast<std::int64_t>( std::floor( centre.y + reach + 0.5 ) );
                  const double denominator = 2.0*gaussianSigma*gaussianSigma;
                  double total = 0.0;
                  for ( std::int64_t ov = ovAll0; ov <= ovAll1; ++ov )
                     for ( std::int64_t ou = ou0; ou <= ou1; ++ou )
                     {
                        const double du = static_cast<double>( ou ) - centre.x;
                        const double dv = static_cast<double>( ov ) - centre.y;
                        total += std::exp( -(du*du + dv*dv)/denominator );
                     }
                  if ( total <= 0.0 )
                     break;
                  const double dropArea = request.pixfrac*request.pixfrac*jacobian;
                  for ( std::int64_t ov = std::max( bandRow0, ovAll0 ); ov <= std::min( bandRow1 - 1, ovAll1 ); ++ov )
                     for ( std::int64_t ou = ou0; ou <= ou1; ++ou )
                     {
                        const double du = static_cast<double>( ou ) - centre.x;
                        const double dv = static_cast<double>( ov ) - centre.y;
                        accumulate( ou, ov, dropArea*std::exp( -(du*du + dv*dv)/denominator )/total );
                     }
                  break;
               }
               }
            }
         }
      } );
}

// The band onto a canvas window: every drop through the registration
// homography and then the window's forward lattice.  The lattice has no
// closed inverse, so the source rows a chunk visits come from every row's
// mapped row sampled each RowSampleStep pixels.  The pixel drop is that of
// BandOnReference, text for text: a change to one is a change to both.
void BandOnLattice( const DrizzleRequest& request )
{
   // A local map: its coefficients stay in registers across the
   // accumulator stores of the pixel loop.
   const LatticeForwardMap forward{
      Homography{ { request.forward[0], request.forward[1], request.forward[2],
                    request.forward[3], request.forward[4], request.forward[5],
                    request.forward[6], request.forward[7], request.forward[8] } },
      LatticeMap{ request.latticeU.data(), request.latticeV.data(), request.latticeX0, request.latticeY0,
                  request.latticeSpacing, request.latticeColumns, request.latticeRows } };
   const double scale = static_cast<double>( request.scale );
   const double half = 0.5*request.pixfrac;
   // Local linear scale of the map (output pixels per input pixel) at the
   // frame centre sizes the circular and Gaussian drops and the margins.
   const double centreX = 0.5*(request.sourceWidth - 1.0);
   const double centreY = request.sourceRow0 + 0.5*(request.sourceRows - 1.0);
   const Point c0 = forward.Map( centreX, centreY );
   const Point cx = forward.Map( centreX + 1.0, centreY );
   const Point cy = forward.Map( centreX, centreY + 1.0 );
   const double jacobian = std::fabs( (cx.x - c0.x)*(cy.y - c0.y) - (cy.x - c0.x)*(cx.y - c0.y) );
   const double localScale = std::sqrt( std::max( jacobian, 1.0e-12 ) );
   const double dropRadius = request.pixfrac*localScale*0.5; // half width of the mapped drop
   const double gaussianSigma = request.pixfrac*localScale/2.3548200450309493; // FWHM = drop width
   const double gaussianReach = 3.0*gaussianSigma;
   double margin = dropRadius*1.5 + 1.0;
   if ( request.kernel == DrizzleKernel::Gaussian )
      margin = gaussianReach + 1.0;
   const Normalization normalization{
      request.normalizationScale, request.normalizationOffset,
      NodeGrid{ request.grid, request.gridXNodes, request.gridYNodes } };
   const NodeGrid weightGrid{ request.weightGrid, request.weightGridXNodes, request.weightGridYNodes };
   const std::uint32_t sourceWidth = request.sourceWidth;
   const std::uint32_t outputWidth = request.outputWidth;
   const float* source = request.source.data();
   const bool useMask = !request.mask.empty();
   const bool selectChannel = request.channel != 255;

   // With a lattice, every source row's mapped row v, sampled each
   // RowSampleStep pixels, and its range (a non-finite sample makes the row
   // unbounded).
   const std::size_t samplesPerRow = (sourceWidth - 1 + RowSampleStep - 1)/RowSampleStep + 1;
   std::vector<double> sampledRows( static_cast<std::size_t>( request.sourceRows )*samplesPerRow );
   std::vector<double> rowLow( request.sourceRows );
   std::vector<double> rowHigh( request.sourceRows );
   ParallelRange( request.sourceRows, request.threads, DrizzleRowGrain,
         [&]( std::size_t begin, std::size_t end )
         {
            for ( std::size_t r = begin; r < end; ++r )
            {
               const double fy = static_cast<double>( request.sourceRow0 + r );
               double low = std::numeric_limits<double>::infinity();
               double high = -std::numeric_limits<double>::infinity();
               for ( std::size_t k = 0; k < samplesPerRow; ++k )
               {
                  const double fx = std::min( static_cast<double>( k*RowSampleStep ), sourceWidth - 1.0 );
                  const double v = forward.Map( fx, fy ).y;
                  sampledRows[r*samplesPerRow + k] = v;
                  if ( !std::isfinite( v ) )
                  {
                     low = -std::numeric_limits<double>::infinity();
                     high = std::numeric_limits<double>::infinity();
                  }
                  else
                  {
                     low = std::min( low, v );
                     high = std::max( high, v );
                  }
               }
               rowLow[r] = low;
               rowHigh[r] = high;
            }
         } );

   ParallelRange( request.outputRows, request.threads, DrizzleRowGrain,
      [&]( std::size_t bandBegin, std::size_t bandEnd )
      {
         // Output rows [rowLo, rowHi) of this chunk, expanded by the drop
         // margin, mapped back to the input to bound the source rows.
         const double rowLo = static_cast<double>( request.outputRow0 + bandBegin );
         const double rowHi = static_cast<double>( request.outputRow0 + bandEnd - 1 );
         const double vLo = rowLo - 0.5 - margin;
         const double vHi = rowHi + 0.5 + margin;
         const double uLo = -0.5 - margin;
         const double uHi = outputWidth - 0.5 + margin;
         double* sum = request.outputSum.data();
         double* weight = request.outputWeight.data();
         std::uint8_t* touched = request.outputTouched.empty() ? nullptr : request.outputTouched.data();
         const std::int64_t bandRow0 = static_cast<std::int64_t>( request.outputRow0 + bandBegin );
         const std::int64_t bandRow1 = static_cast<std::int64_t>( request.outputRow0 + bandEnd );

         // The source rows whose drops can reach this chunk, each with the
         // run of columns to visit, in row order.
         std::vector<SourceRun> runs;
         {
            for ( std::size_t r = 0; r < request.sourceRows; ++r )
            {
               if ( rowHigh[r] + RowSlack < vLo || rowLow[r] - RowSlack > vHi )
                  continue;
               // The first and last sampled runs that can reach the rows.
               std::int64_t x0 = -1, x1 = -1;
               const double* samples = sampledRows.data() + r*samplesPerRow;
               for ( std::size_t k = 0; k < samplesPerRow; ++k )
               {
                  const double a = samples[k];
                  const double b = samples[std::min( k + 1, samplesPerRow - 1 )];
                  const bool reach = !std::isfinite( a ) || !std::isfinite( b )
                     || (std::max( a, b ) + RowSlack >= vLo && std::min( a, b ) - RowSlack <= vHi);
                  if ( !reach )
                     continue;
                  if ( x0 < 0 )
                     x0 = static_cast<std::int64_t>( k*RowSampleStep );
                  x1 = std::min<std::int64_t>( static_cast<std::int64_t>( (k + 1)*RowSampleStep ), sourceWidth - 1 );
               }
               if ( x0 < 0 )
                  continue;
               runs.push_back( SourceRun{ static_cast<std::int64_t>( request.sourceRow0 + r ),
                                          std::max<std::int64_t>( 0, x0 - 1 ),
                                          std::min<std::int64_t>( sourceWidth - 1, x1 + 1 ) } );
            }
         }

         for ( const SourceRun& run : runs )
         {
            // Local bounds: the touch-flag stores below may alias the run.
            const std::int64_t y = run.y, x0 = run.x0, x1 = run.x1;
            const double fy = static_cast<double>( y );
            const float* row = source + static_cast<std::size_t>( y - request.sourceRow0 )*sourceWidth;
            for ( std::int64_t x = x0; x <= x1; ++x )
            {
               const float value = row[x];
               if ( !std::isfinite( value ) )
                  continue;
               if ( selectChannel )
               {
                  const std::uint8_t pixelChannel = request.cfaPattern[((y & 1) << 1) | (x & 1)];
                  if ( pixelChannel != request.channel )
                     continue;
               }
               const double fx = static_cast<double>( x );
               const Point centre = forward.Map( fx, fy );
               if ( !std::isfinite( centre.x ) || !std::isfinite( centre.y ) )
                  continue;
               if ( centre.y < vLo || centre.y > vHi || centre.x < uLo || centre.x > uHi )
                  continue;
               const double referenceX = centre.x/scale;
               const double referenceY = centre.y/scale;
               double pixelWeight = request.frameWeight;
               if ( useMask )
               {
                  const double mx = std::round( referenceX ), my = std::round( referenceY );
                  if ( mx < 0.0 || my < 0.0 || mx >= request.maskWidth || my >= request.maskHeight )
                     continue;
                  if ( request.mask[static_cast<std::size_t>( my )*request.maskWidth + static_cast<std::size_t>( mx )] == 0 )
                     continue;
               }
               if ( !weightGrid.Empty() )
                  pixelWeight *= weightGrid.At( referenceX, referenceY );
               if ( !(pixelWeight > 0.0) )
                  continue;
               const double normalized = static_cast<double>(
                  normalization.Apply( value, referenceX, referenceY ) );
               if ( !std::isfinite( normalized ) )
                  continue;

               auto accumulate = [&]( std::int64_t ou, std::int64_t ov, double area )
               {
                  if ( area <= 0.0 || ov < bandRow0 || ov >= bandRow1 || ou < 0
                    || ou >= static_cast<std::int64_t>( outputWidth ) )
                     return;
                  const std::size_t index =
                     static_cast<std::size_t>( ov - request.outputRow0 )*outputWidth
                     + static_cast<std::size_t>( ou );
                  const double w = pixelWeight*area;
                  sum[index] += w*normalized;
                  weight[index] += w;
                  if ( touched != nullptr )
                     touched[index] = 1;
               };

               switch ( request.kernel )
               {
               case DrizzleKernel::Point:
                  accumulate( static_cast<std::int64_t>( std::round( centre.x ) ),
                              static_cast<std::int64_t>( std::round( centre.y ) ), 1.0 );
                  break;
               case DrizzleKernel::Square:
               {
                  const std::array<Point, 4> quad = forward.MapQuad( fx, fy, half );
                  double minU = quad[0].x, maxU = quad[0].x, minV = quad[0].y, maxV = quad[0].y;
                  for ( int i = 1; i < 4; ++i )
                  {
                     minU = std::min( minU, quad[i].x ); maxU = std::max( maxU, quad[i].x );
                     minV = std::min( minV, quad[i].y ); maxV = std::max( maxV, quad[i].y );
                  }
                  const std::int64_t ou0 = static_cast<std::int64_t>( std::floor( minU + 0.5 ) );
                  const std::int64_t ou1 = static_cast<std::int64_t>( std::floor( maxU + 0.5 ) );
                  const std::int64_t ov0 = std::max( bandRow0, static_cast<std::int64_t>( std::floor( minV + 0.5 ) ) );
                  const std::int64_t ov1 = std::min( bandRow1 - 1, static_cast<std::int64_t>( std::floor( maxV + 0.5 ) ) );
                  for ( std::int64_t ov = ov0; ov <= ov1; ++ov )
                     for ( std::int64_t ou = std::max<std::int64_t>( 0, ou0 );
                           ou <= std::min<std::int64_t>( outputWidth - 1, ou1 ); ++ou )
                        accumulate( ou, ov, QuadPixelOverlap( quad.data(), static_cast<double>( ou ), static_cast<double>( ov ) ) );
                  break;
               }
               case DrizzleKernel::Circular:
               {
                  const double r = dropRadius;
                  const std::int64_t ou0 = std::max<std::int64_t>( 0, static_cast<std::int64_t>( std::floor( centre.x - r + 0.5 ) ) );
                  const std::int64_t ou1 = std::min<std::int64_t>( outputWidth - 1, static_cast<std::int64_t>( std::floor( centre.x + r + 0.5 ) ) );
                  const std::int64_t ov0 = std::max( bandRow0, static_cast<std::int64_t>( std::floor( centre.y - r + 0.5 ) ) );
                  const std::int64_t ov1 = std::min( bandRow1 - 1, static_cast<std::int64_t>( std::floor( centre.y + r + 0.5 ) ) );
                  for ( std::int64_t ov = ov0; ov <= ov1; ++ov )
                     for ( std::int64_t ou = ou0; ou <= ou1; ++ou )
                        accumulate( ou, ov, DiscRectangleArea( centre.x, centre.y, r,
                                                               static_cast<double>( ou ) - 0.5, static_cast<double>( ou ) + 0.5,
                                                               static_cast<double>( ov ) - 0.5, static_cast<double>( ov ) + 0.5 ) );
                  break;
               }
               case DrizzleKernel::Gaussian:
               {
                  // Weights of every output pixel centre within the reach,
                  // normalized so the drop carries the pixel's full weight.
                  const double reach = gaussianReach;
                  const std::int64_t ou0 = std::max<std::int64_t>( 0, static_cast<std::int64_t>( std::floor( centre.x - reach + 0.5 ) ) );
                  const std::int64_t ou1 = std::min<std::int64_t>( outputWidth - 1, static_cast<std::int64_t>( std::floor( centre.x + reach + 0.5 ) ) );
                  const std::int64_t ovAll0 = static_cast<std::int64_t>( std::floor( centre.y - reach + 0.5 ) );
                  const std::int64_t ovAll1 = static_cast<std::int64_t>( std::floor( centre.y + reach + 0.5 ) );
                  const double denominator = 2.0*gaussianSigma*gaussianSigma;
                  double total = 0.0;
                  for ( std::int64_t ov = ovAll0; ov <= ovAll1; ++ov )
                     for ( std::int64_t ou = ou0; ou <= ou1; ++ou )
                     {
                        const double du = static_cast<double>( ou ) - centre.x;
                        const double dv = static_cast<double>( ov ) - centre.y;
                        total += std::exp( -(du*du + dv*dv)/denominator );
                     }
                  if ( total <= 0.0 )
                     break;
                  const double dropArea = request.pixfrac*request.pixfrac*jacobian;
                  for ( std::int64_t ov = std::max( bandRow0, ovAll0 ); ov <= std::min( bandRow1 - 1, ovAll1 ); ++ov )
                     for ( std::int64_t ou = ou0; ou <= ou1; ++ou )
                     {
                        const double du = static_cast<double>( ou ) - centre.x;
                        const double dv = static_cast<double>( ov ) - centre.y;
                        accumulate( ou, ov, dropArea*std::exp( -(du*du + dv*dv)/denominator )/total );
                     }
                  break;
               }
               }
            }
         }
      } );
}

void DrizzleBand( const DrizzleRequest& request )
{
   request.Validate();
   if ( request.HasLattice() )
      BandOnLattice( request );
   else
      BandOnReference( request );
}

} // namespace ufwbpp::native
