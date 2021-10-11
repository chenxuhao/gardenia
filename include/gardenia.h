#pragma once
#include <cassert>
#include <cstdio>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#define USE_OMP

#ifdef USE_TBB
#include <tbb/task_group.h>
#include <tbb/tbb.h>
#endif

#ifdef USE_OMP
#include <omp.h>
#endif

namespace gardenia {

struct blocked_range {
  typedef size_t const_iterator;
  blocked_range(size_t begin, size_t end) : begin_(begin), end_(end) {}
  blocked_range(int begin, int end) : begin_(begin), end_(end) {}
  const_iterator begin() const { return begin_; }
  const_iterator end() const { return end_; }
 private:
  size_t begin_;
  size_t end_;
};

template <typename Func>
void xparallel_for(size_t begin, size_t end, const Func &f) {
  blocked_range r(begin, end);
  f(r);
}

template <typename Func>
void parallel_for(size_t begin, size_t end,
                  const Func &f, size_t /*grainsize*/) {
  assert(end >= begin);
  #pragma omp parallel for
  for (int i = static_cast<int>(begin); i < static_cast<int>(end); ++i)
    f(blocked_range(i, i + 1));
}

template <typename T, typename U>
bool value_representation(U const &value) {
  return static_cast<U>(static_cast<T>(value)) == value;
}

template <typename T, typename Func>
inline void for_(bool parallelize, size_t begin, T end, Func f, size_t grainsize = 100) {
  static_assert(std::is_integral<T>::value, "end must be integral type");
  parallelize = parallelize && value_representation<size_t>(end);
  parallelize ? parallel_for(begin, end, f, grainsize)
              : xparallel_for(begin, end, f);
}

template <typename Func, typename... Args>
void on_each(Func&& f, const Args&... args) {
  #pragma omp parallel
  {
    f(omp_get_thread_num());
  };
}

template <typename T, typename Func>
inline void for_i(bool parallelize, T size, Func f, size_t grainsize = 100u) {
#ifdef USE_SINGLE_THREAD
  for (size_t i = 0; i < size; ++i) {
    f(i);
  }
#else  // #ifdef USE_SINGLE_THREAD
  for_(parallelize, 0u, size, [&](const blocked_range &r) {
#ifdef USE_OMP
    #pragma omp parallel for
    for (int i = static_cast<int>(r.begin());
         i < static_cast<int>(r.end()); i++) {
      f(i);
    }
#else
    for (size_t i = r.begin(); i < r.end(); i++) {
      f(i);
    }
#endif
  }, grainsize);
#endif  // #ifdef USE_SINGLE_THREAD
}

template <typename T, typename Func>
inline void for_i(T size, Func f, size_t grainsize = 100) {
  for_i(true, size, f, grainsize);
}

}
