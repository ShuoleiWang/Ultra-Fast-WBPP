#include "Lanczos3Table.h"

#include <cmath>
#include <vector>

namespace ufwbpp::native::detail
{

namespace
{

constexpr double Pi = 3.141592653589793;
constexpr int SeriesTerms = 12;
constexpr int TapOffsets[6] = { -2, -1, 0, 1, 2, 3 };

// One raw (unnormalized) Lanczos-3 weight sinc(d) * sinc(d/3) for the
// signed distance d between the sample and the tap.  The table's two nodes
// outside [0, 1] use the smooth analytic continuation rather than the
// |d| >= 3 cut-off (sinc(3) is exactly zero anyway), so the interpolation
// sees a smooth function up to the edges of the fraction range.
double RawWeight( double d )
{
   if ( d == 0.0 )
      return 1.0;
   const double primary = DeterministicSinPi( d )/(Pi*d);
   const double reduced = d/3.0;
   const double secondary = DeterministicSinPi( reduced )/(Pi*reduced);
   return primary*secondary;
}

std::vector<double> BuildTable()
{
   std::vector<double> table( static_cast<std::size_t>( Lanczos3TableNodes )*6 );
   for ( std::uint32_t node = 0; node < Lanczos3TableNodes; ++node )
   {
      const double fraction = (static_cast<double>( node ) - 1.0)/static_cast<double>( Lanczos3TableIntervals );
      double raw[6];
      double total = 0.0;
      for ( int tap = 0; tap < 6; ++tap )
      {
         raw[tap] = RawWeight( fraction - static_cast<double>( TapOffsets[tap] ) );
         total = total + raw[tap];
      }
      for ( int tap = 0; tap < 6; ++tap )
         table[static_cast<std::size_t>( node )*6 + static_cast<std::size_t>( tap )] = raw[tap]/total;
   }
   return table;
}

} // namespace

double DeterministicSinPi( double v )
{
   // Reduce to r in [-0.5, 0.5]: sin(pi v) = (-1)^n sin(pi r), v = n + r.
   const double n = std::floor( v + 0.5 );
   const double r = v - n;
   const double x = Pi*r;
   const double x2 = x*x;
   double term = 1.0;
   for ( int k = SeriesTerms; k >= 1; --k )
   {
      const double denominator = static_cast<double>( (2*k)*(2*k + 1) );
      term = 1.0 - x2*term/denominator;
   }
   const double s = x*term;
   const long long parity = static_cast<long long>( n ) & 1LL;
   return parity != 0 ? -s : s;
}

const double* Lanczos3TableNodeValues()
{
   static const std::vector<double> table = BuildTable();
   return table.data();
}

void Lanczos3TableWeights( double fraction, float weights[6] )
{
   const double* table = Lanczos3TableNodeValues();
   const double u = fraction*static_cast<double>( Lanczos3TableIntervals );
   const double floorU = std::floor( u );
   const std::size_t i = static_cast<std::size_t>( floorU );
   const double t = u - floorU;
   // Cubic Lagrange basis through the four nodes at t = -1, 0, 1, 2
   // (fourth-order accurate; Catmull-Rom's finite-difference slopes would
   // only give third order).
   const double a = t + 1.0;
   const double b = t - 1.0;
   const double c = t - 2.0;
   const double b0 = -((t*b)*c)/6.0;
   const double b1 = ((a*b)*c)/2.0;
   const double b2 = -((a*t)*c)/2.0;
   const double b3 = ((a*t)*b)/6.0;
   // Node i sits in row i+1; the four rows i-1 .. i+2 surround the sample.
   const double* p0 = table + i*6;
   const double* p1 = p0 + 6;
   const double* p2 = p1 + 6;
   const double* p3 = p2 + 6;
   double values[6];
   double total = 0.0;
   for ( int tap = 0; tap < 6; ++tap )
   {
      values[tap] = ((b0*p0[tap] + b1*p1[tap]) + b2*p2[tap]) + b3*p3[tap];
      total = total + values[tap];
   }
   for ( int tap = 0; tap < 6; ++tap )
      weights[tap] = static_cast<float>( values[tap]/total );
}

} // namespace ufwbpp::native::detail
