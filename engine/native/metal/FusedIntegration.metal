#include <metal_stdlib>

using namespace metal;

struct FusedParameters
{
   uint width;
   uint referenceHeight;
   uint firstRow;
   uint rowCount;
   uint frameCount;
   uint gridWidth;
   uint gridHeight;
   uint rejectionBits;
   float outputScale;
   float outputOffset;
};

struct RangeParameters
{
   uint sampleCount;
};

struct OutputNormalizationParameters
{
   uint sampleCount;
   float scale;
   float offset;
};

struct RobustParameters
{
   uint width;
   uint referenceHeight;
   uint firstRow;
   uint rowCount;
   uint frameCount;
   uint gridWidth;
   uint gridHeight;
   uint winsorIterations;
   float rangeLow;
   float lowSigma;
   float highSigma;
   float winsorSigma;
   float outputScale;
   float outputOffset;
};

struct LinearFitParameters
{
   uint width;
   uint referenceHeight;
   uint firstRow;
   uint rowCount;
   uint frameCount;
   uint gridWidth;
   uint gridHeight;
   uint fitBisectionIterations;
   uint rejectionIterations;
   float rangeLow;
   float lowTolerance;
   float highTolerance;
   float outputScale;
   float outputOffset;
};

inline int reflected_index( int index, int length )
{
   if ( index < 0 )
      return -index - 1;
   if ( index >= length )
      return 2*length - index - 1;
   return index;
}

inline float positive_cube( float value )
{
   return value > 0.0f ? value*value*value : 0.0f;
}

inline float bspline_weight( float x )
{
   return (positive_cube( x + 2.0f )
         - 4.0f*positive_cube( x + 1.0f )
         + 6.0f*positive_cube( x )
         - 4.0f*positive_cube( x - 1.0f ))/6.0f;
}

inline float bicubic_bspline(
   device const float* grid,
   uint width,
   uint height,
   float x,
   float y )
{
   const int x1 = int( floor( x ) );
   const int y1 = int( floor( y ) );
   const float dx = x - float( x1 );
   const float dy = y - float( y1 );
   float value = 0.0f;
   for ( int row = 0; row < 4; ++row )
   {
      const int gy = reflected_index( y1 + row - 1, int( height ) );
      const float wy = bspline_weight( float( row - 1 ) - dy );
      for ( int column = 0; column < 4; ++column )
      {
         const int gx = reflected_index( x1 + column - 1, int( width ) );
         const float wx = bspline_weight( float( column - 1 ) - dx );
         value = fma( grid[uint( gy )*width + uint( gx )], wx*wy, value );
      }
   }
   return value;
}

kernel void fused_masked_weighted_integration(
   device const float* samples [[buffer(0)]],
   device const uchar* rejectionMask [[buffer(1)]],
   device const float* scaleGrids [[buffer(2)]],
   device const float* zeroOffsetGrids [[buffer(3)]],
   device const float* frameWeights [[buffer(4)]],
   device float* integrated [[buffer(5)]],
   device ushort* acceptedSamples [[buffer(6)]],
   device ushort* rejectedSamples [[buffer(7)]],
   constant FusedParameters& parameters [[buffer(8)]],
   uint index [[thread_position_in_grid]] )
{
   const uint tilePixels = parameters.width*parameters.rowCount;
   if ( index >= tilePixels )
      return;

   const uint x = index % parameters.width;
   const uint y = parameters.firstRow + index/parameters.width;
   const uint gridSamples = parameters.gridWidth*parameters.gridHeight;
   const float gx = float( parameters.gridWidth )
                  / float( parameters.width )*float( x );
   const float gy = float( parameters.gridHeight )
                  / float( parameters.referenceHeight )*float( y );
   float weighted = 0.0f;
   float weightedCompensation = 0.0f;
   float weightSum = 0.0f;
   float weightCompensation = 0.0f;
   ushort accepted = 0;
   ushort rejected = 0;
   for ( uint frame = 0; frame < parameters.frameCount; ++frame )
   {
      const uint sampleIndex = frame*tilePixels + index;
      if ( (uint( rejectionMask[sampleIndex] )
             & parameters.rejectionBits) != 0 )
      {
         ++rejected;
         continue;
      }
      const float sample = samples[sampleIndex];
      const float weight = frameWeights[frame];
      if ( !isfinite( sample ) || weight <= 0.0f )
      {
         ++rejected;
         continue;
      }
      const uint gridOffset = frame*gridSamples;
      const bool identityNormalization = gridSamples == 4
         && scaleGrids[gridOffset] == 1.0f
         && scaleGrids[gridOffset + 1] == 1.0f
         && scaleGrids[gridOffset + 2] == 1.0f
         && scaleGrids[gridOffset + 3] == 1.0f
         && zeroOffsetGrids[gridOffset] == 0.0f
         && zeroOffsetGrids[gridOffset + 1] == 0.0f
         && zeroOffsetGrids[gridOffset + 2] == 0.0f
         && zeroOffsetGrids[gridOffset + 3] == 0.0f;
      const float a = identityNormalization ? 1.0f : bicubic_bspline(
         scaleGrids + gridOffset,
         parameters.gridWidth, parameters.gridHeight, gx, gy );
      const float b = identityNormalization ? 0.0f : bicubic_bspline(
         zeroOffsetGrids + gridOffset,
         parameters.gridWidth, parameters.gridHeight, gx, gy );
      const float normalized = identityNormalization
         ? sample : fma( a, sample, b );
      if ( !isfinite( normalized ) )
      {
         ++rejected;
         continue;
      }
      const float weightedTerm = weight*normalized;
      const float correctedWeighted = weightedTerm - weightedCompensation;
      const float nextWeighted = weighted + correctedWeighted;
      weightedCompensation = (nextWeighted - weighted) - correctedWeighted;
      weighted = nextWeighted;
      const float correctedWeight = weight - weightCompensation;
      const float nextWeight = weightSum + correctedWeight;
      weightCompensation = (nextWeight - weightSum) - correctedWeight;
      weightSum = nextWeight;
      ++accepted;
   }
   integrated[index] = weightSum > 0.0f
      ? fma( parameters.outputScale, weighted/weightSum,
             parameters.outputOffset )
      : 0.0f;
   acceptedSamples[index] = accepted;
   rejectedSamples[index] = rejected;
}

inline uint ordered_float_bits( float value )
{
   const uint bits = as_type<uint>( value );
   return (bits & 0x80000000u) != 0 ? ~bits : bits ^ 0x80000000u;
}

kernel void reduce_finite_output_range(
   device const float* samples [[buffer(0)]],
   device atomic_uint* range [[buffer(1)]],
   constant RangeParameters& parameters [[buffer(2)]],
   uint index [[thread_position_in_grid]],
   uint lane [[thread_index_in_threadgroup]] )
{
   threadgroup float localMinimum[256];
   threadgroup float localMaximum[256];
   threadgroup uint localCount[256];
   const float value = index < parameters.sampleCount
      ? samples[index] : NAN;
   const bool finite = isfinite( value );
   localMinimum[lane] = finite ? value : INFINITY;
   localMaximum[lane] = finite ? value : -INFINITY;
   localCount[lane] = finite ? 1u : 0u;
   threadgroup_barrier( mem_flags::mem_threadgroup );

   for ( uint stride = 128; stride != 0; stride >>= 1 )
   {
      if ( lane < stride )
      {
         localMinimum[lane] = min(
            localMinimum[lane], localMinimum[lane + stride] );
         localMaximum[lane] = max(
            localMaximum[lane], localMaximum[lane + stride] );
         localCount[lane] += localCount[lane + stride];
      }
      threadgroup_barrier( mem_flags::mem_threadgroup );
   }
   if ( lane == 0 && localCount[0] != 0 )
   {
      atomic_fetch_min_explicit(
         range, ordered_float_bits( localMinimum[0] ),
         memory_order_relaxed );
      atomic_fetch_max_explicit(
         range + 1, ordered_float_bits( localMaximum[0] ),
         memory_order_relaxed );
      atomic_fetch_add_explicit(
         range + 2, localCount[0], memory_order_relaxed );
   }
}

kernel void normalize_output_range_in_place(
   device float* samples [[buffer(0)]],
   constant OutputNormalizationParameters& parameters [[buffer(1)]],
   uint index [[thread_position_in_grid]] )
{
   if ( index >= parameters.sampleCount )
      return;
   const float value = samples[index];
   if ( isfinite( value ) )
      samples[index] = fma( parameters.scale, value, parameters.offset );
}

kernel void fused_native_robust_integration(
   device const float* samples [[buffer(0)]],
   device const float* scaleGrids [[buffer(1)]],
   device const float* zeroOffsetGrids [[buffer(2)]],
   device const float* frameWeights [[buffer(3)]],
   device float* integrated [[buffer(4)]],
   device ushort* acceptedSamples [[buffer(5)]],
   device ushort* rejectedSamples [[buffer(6)]],
   constant RobustParameters& parameters [[buffer(7)]],
   uint index [[thread_position_in_grid]] )
{
   const uint tilePixels = parameters.width*parameters.rowCount;
   if ( index >= tilePixels )
      return;
   const uint x = index % parameters.width;
   const uint y = parameters.firstRow + index/parameters.width;
   const uint gridSamples = parameters.gridWidth*parameters.gridHeight;
   const float gx = float( parameters.gridWidth )
                  / float( parameters.width )*float( x );
   const float gy = float( parameters.gridHeight )
                  / float( parameters.referenceHeight )*float( y );
   thread float values[64];
   thread uchar valid[64];
   float sum = 0.0f;
   float squareSum = 0.0f;
   uint validCount = 0;
   for ( uint frame = 0; frame < parameters.frameCount; ++frame )
   {
      const uint gridOffset = frame*gridSamples;
      const float a = bicubic_bspline(
         scaleGrids + gridOffset,
         parameters.gridWidth, parameters.gridHeight, gx, gy );
      const float b = bicubic_bspline(
         zeroOffsetGrids + gridOffset,
         parameters.gridWidth, parameters.gridHeight, gx, gy );
      const float input = samples[frame*tilePixels + index];
      const float value = fma( a, input, b );
      const bool isValid = isfinite( input )
                        && input > parameters.rangeLow
                        && isfinite( value )
                        && frameWeights[frame] > 0.0f;
      values[frame] = value;
      valid[frame] = isValid ? 1 : 0;
      if ( isValid )
      {
         sum += value;
         squareSum = fma( value, value, squareSum );
         ++validCount;
      }
   }
   if ( validCount == 0 )
   {
      integrated[index] = 0.0f;
      acceptedSamples[index] = 0;
      rejectedSamples[index] = ushort( parameters.frameCount );
      return;
   }

   float mean = sum/float( validCount );
   float deviation = sqrt( max(
      0.0f, squareSum/float( validCount ) - mean*mean ) );
   for ( uint iteration = 0;
         iteration < parameters.winsorIterations && deviation > 0.0f;
         ++iteration )
   {
      const float low = mean - parameters.winsorSigma*deviation;
      const float high = mean + parameters.winsorSigma*deviation;
      sum = 0.0f;
      squareSum = 0.0f;
      for ( uint frame = 0; frame < parameters.frameCount; ++frame )
         if ( valid[frame] != 0 )
         {
            const float value = clamp( values[frame], low, high );
            sum += value;
            squareSum = fma( value, value, squareSum );
         }
      mean = sum/float( validCount );
      deviation = sqrt( max(
         0.0f, squareSum/float( validCount ) - mean*mean ) );
   }

   const float low = mean - parameters.lowSigma*deviation;
   const float high = mean + parameters.highSigma*deviation;
   float weighted = 0.0f;
   float weightSum = 0.0f;
   ushort accepted = 0;
   ushort rejected = 0;
   for ( uint frame = 0; frame < parameters.frameCount; ++frame )
   {
      const float value = values[frame];
      if ( valid[frame] == 0 || value < low || value > high )
      {
         ++rejected;
         continue;
      }
      const float weight = frameWeights[frame];
      weighted = fma( weight, value, weighted );
      weightSum += weight;
      ++accepted;
   }
   integrated[index] = weightSum > 0.0f
      ? fma( parameters.outputScale, weighted/weightSum,
             parameters.outputOffset )
      : 0.0f;
   acceptedSamples[index] = accepted;
   rejectedSamples[index] = rejected;
}

inline float median64_in_place( thread float* data, uint count )
{
   const int middle = int( count >> 1 );
   int left = 0;
   int right = int( count ) - 1;
   while ( left < right )
   {
      const float pivot = data[middle];
      int i = left;
      int j = right;
      do
      {
         while ( i <= right && data[i] < pivot )
            ++i;
         while ( j >= left && data[j] > pivot )
            --j;
         if ( i <= j )
         {
            const float temporary = data[i];
            data[i] = data[j];
            data[j] = temporary;
            ++i;
            --j;
         }
      }
      while ( i <= j );
      if ( j < middle )
         left = i;
      if ( middle < i )
         right = j;
   }
   const float upper = data[middle];
   if ( (count & 1u) != 0 )
      return upper;
   float lower = data[0];
   for ( int i = 1; i < middle; ++i )
      lower = max( lower, data[i] );
   return 0.5f*(lower + upper);
}

struct LadLine
{
   float a;
   float b;
   float averageDeviation;
};

inline LadLine fit_sorted_lad_line(
   thread const float* values,
   uint count,
   uint bisectionIterations )
{
   float low = 0.0f;
   float high = max(
      1.0e-12f, 4.0f*(values[count - 1] - values[0])/float( count - 1 ) );
   thread float scratch[64];
   float intercept = 0.0f;
   for ( uint iteration = 0; iteration < bisectionIterations; ++iteration )
   {
      const float slope = 0.5f*(low + high);
      for ( uint i = 0; i < count; ++i )
         scratch[i] = values[i] - slope*float( i );
      intercept = median64_in_place( scratch, count );
      float score = 0.0f;
      for ( uint i = 0; i < count; ++i )
      {
         const float residual = values[i]
                              - (intercept + slope*float( i ));
         score += residual > 0.0f ? float( i )
                : residual < 0.0f ? -float( i ) : 0.0f;
      }
      if ( score > 0.0f )
         low = slope;
      else
         high = slope;
   }
   LadLine line;
   line.b = 0.5f*(low + high);
   for ( uint i = 0; i < count; ++i )
      scratch[i] = values[i] - line.b*float( i );
   line.a = median64_in_place( scratch, count );
   line.averageDeviation = 0.0f;
   for ( uint i = 0; i < count; ++i )
      line.averageDeviation += fabs(
         values[i] - (line.a + line.b*float( i )) );
   line.averageDeviation /= float( count );
   return line;
}

kernel void fused_native_linear_fit_integration(
   device const float* samples [[buffer(0)]],
   device const float* scaleGrids [[buffer(1)]],
   device const float* zeroOffsetGrids [[buffer(2)]],
   device const float* frameWeights [[buffer(3)]],
   device float* integrated [[buffer(4)]],
   device ushort* acceptedSamples [[buffer(5)]],
   device ushort* rejectedSamples [[buffer(6)]],
   constant LinearFitParameters& parameters [[buffer(7)]],
   uint index [[thread_position_in_grid]] )
{
   const uint tilePixels = parameters.width*parameters.rowCount;
   if ( index >= tilePixels )
      return;
   const uint x = index % parameters.width;
   const uint y = parameters.firstRow + index/parameters.width;
   const uint gridSamples = parameters.gridWidth*parameters.gridHeight;
   const float gx = float( parameters.gridWidth )
                  / float( parameters.width )*float( x );
   const float gy = float( parameters.gridHeight )
                  / float( parameters.referenceHeight )*float( y );
   thread float values[64];
   thread ushort sourceFrames[64];
   uint count = 0;
   for ( uint frame = 0; frame < parameters.frameCount; ++frame )
   {
      const uint gridOffset = frame*gridSamples;
      const float a = bicubic_bspline(
         scaleGrids + gridOffset,
         parameters.gridWidth, parameters.gridHeight, gx, gy );
      const float b = bicubic_bspline(
         zeroOffsetGrids + gridOffset,
         parameters.gridWidth, parameters.gridHeight, gx, gy );
      const float input = samples[frame*tilePixels + index];
      const float value = fma( a, input, b );
      if ( !isfinite( input ) || input <= parameters.rangeLow
        || !isfinite( value )
        || frameWeights[frame] <= 0.0f )
         continue;
      uint position = count;
      while ( position > 0 && values[position - 1] > value )
      {
         values[position] = values[position - 1];
         sourceFrames[position] = sourceFrames[position - 1];
         --position;
      }
      values[position] = value;
      sourceFrames[position] = ushort( frame );
      ++count;
   }

   for ( uint round = 0;
         round < parameters.rejectionIterations && count >= 5; ++round )
   {
      const LadLine line = fit_sorted_lad_line(
         values, count, parameters.fitBisectionIterations );
      const float fitScale = 2.0f*line.averageDeviation
                           * sqrt( 1.0f + line.b*line.b );
      if ( !(fitScale > 0.0f) || 1.0f + fitScale == 1.0f )
         break;
      uint kept = 0;
      for ( uint i = 0; i < count; ++i )
      {
         const float distance =
            (values[i] - (line.a + line.b*float( i )))/fitScale;
         if ( distance < -parameters.lowTolerance
           || distance > parameters.highTolerance )
            continue;
         values[kept] = values[i];
         sourceFrames[kept] = sourceFrames[i];
         ++kept;
      }
      if ( kept == count )
         break;
      count = kept;
   }

   float weighted = 0.0f;
   float weightSum = 0.0f;
   for ( uint i = 0; i < count; ++i )
   {
      const float weight = frameWeights[sourceFrames[i]];
      weighted = fma( weight, values[i], weighted );
      weightSum += weight;
   }
   integrated[index] = weightSum > 0.0f
      ? fma( parameters.outputScale, weighted/weightSum,
             parameters.outputOffset )
      : 0.0f;
   acceptedSamples[index] = ushort( count );
   rejectedSamples[index] = ushort( parameters.frameCount - count );
}
