#include "immintrin.h"
#include "platform_atomics.h"
#include <boost/align/aligned_allocator.hpp>
template <typename T>
using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, 32>>;

typedef pair<ScoreT, IndexT> WN;
const size_t elements_per_line = 16; // 32*16 bits = 64 Bytes (cache-line size)
const size_t buf_size = 16 * 8; // dump 8 lines each time

template <typename T>
void streaming_store(T *src, T *dst) {
	for (size_t i = 0; i < buf_size; i += elements_per_line) {
		__m256i r0 = _mm256_load_si256((__m256i*) &src[i+0]);
		__m256i r1 = _mm256_load_si256((__m256i*) &src[i+8]);
		_mm256_stream_si256((__m256i*) &dst[i+0], r0);
		_mm256_stream_si256((__m256i*) &dst[i+8], r1);
	}
	return;
}

