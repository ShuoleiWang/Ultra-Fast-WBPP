#ifndef OPENASTROFLOW_NATIVE_PARALLELRANGE_H
#define OPENASTROFLOW_NATIVE_PARALLELRANGE_H

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <thread>
#include <utility>
#include <vector>

namespace openastroflow::native::detail
{

// Runs function(begin, end) over [0, count) on up to `threads` threads that
// claim contiguous chunks of `grain` items from a shared atomic counter, so a
// slow core (efficiency core, SMT sibling, throttled core) never holds the
// tail of the range while the others idle.  Every item is processed exactly
// once and the per-item results never depend on which thread or chunk ran
// it; the calling thread drains chunks too.  Exceptions stop further claims,
// are collected, and are rethrown after every worker has joined so no thread
// outlives the call.
template <class Function>
void ParallelRange( std::size_t count,
                    std::uint32_t threads,
                    std::size_t grain,
                    Function&& function )
{
   if ( count == 0 )
      return;
   grain = std::max<std::size_t>( 1, grain );
   const std::size_t chunks = (count + grain - 1)/grain;
   const std::size_t workers = std::max<std::size_t>(
      1, std::min<std::size_t>( threads, chunks ) );
   if ( workers == 1 )
   {
      function( std::size_t{ 0 }, count );
      return;
   }
   std::atomic<std::size_t> next{ 0 };
   std::vector<std::exception_ptr> errors( workers );
   auto drain = [&]( std::size_t worker )
   {
      try
      {
         for ( ;; )
         {
            const std::size_t begin =
               next.fetch_add( grain, std::memory_order_relaxed );
            if ( begin >= count )
               return;
            function( begin, std::min( count, begin + grain ) );
         }
      }
      catch ( ... )
      {
         errors[worker] = std::current_exception();
         next.store( count, std::memory_order_relaxed );
      }
   };
   std::vector<std::thread> pool;
   pool.reserve( workers - 1 );
   for ( std::size_t index = 1; index < workers; ++index )
      pool.emplace_back( [&drain, index]() { drain( index ); } );
   drain( 0 );
   for ( std::thread& worker : pool )
      worker.join();
   for ( const std::exception_ptr& error : errors )
      if ( error )
         std::rethrow_exception( error );
}

} // namespace openastroflow::native::detail

#endif // OPENASTROFLOW_NATIVE_PARALLELRANGE_H
