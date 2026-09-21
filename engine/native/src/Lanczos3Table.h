#ifndef OPENASTROFLOW_NATIVE_LANCZOS3TABLE_H
#define OPENASTROFLOW_NATIVE_LANCZOS3TABLE_H

#include <cstddef>
#include <cstdint>

namespace openastroflow::native::detail
{

// Deterministic Lanczos-3 tap weights (see calibration.py / lanczos_table.py).
//
// The six normalized tap weights of a sub-pixel fraction f in [0, 1) are
// read from a table of Lanczos3TableIntervals + 3 nodes at f = i/N
// (i = -1 .. N+1) with cubic Lagrange interpolation.  The node values are
// computed from their own Taylor series in a fixed Float64 operation order,
// never from the C library, so every platform and library version builds
// the same table and the Python reference (`lanczos_table.py`) reproduces
// it value for value.  Interpolation errors are below 1e-12, four orders of
// magnitude under the Float32 resolution of the stored weights.
constexpr std::uint32_t Lanczos3TableIntervals = 2048;
constexpr std::uint32_t Lanczos3TableNodes = Lanczos3TableIntervals + 3;

// Node values, row i+1 holds the six normalized weights at f = i/N.
const double* Lanczos3TableNodeValues();

// sin(pi v) by the same reduction and series as the Python reference.
double DeterministicSinPi( double v );

// Normalized weights at ``fraction`` (in [0, 1)): interpolation of the
// table, Float32 of the Float64 quotient by the Float64 total.
void Lanczos3TableWeights( double fraction, float weights[6] );

} // namespace openastroflow::native::detail

#endif
