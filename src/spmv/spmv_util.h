#pragma once
#include <limits>
#include <cmath>
#include <algorithm>

inline size_t bytes_per_spmv(int m, int nnz) {
    size_t bytes = 0;
    bytes += 2*sizeof(IndexT) * m;    // row pointer
    bytes += 1*sizeof(IndexT) * nnz;  // column index
    bytes += 2*sizeof(ValueT) * nnz;  // A[i,j] and x[j]
    bytes += 2*sizeof(ValueT) * m;    // y[i] = y[i] + ...
    return bytes;
}

template <typename T>
T maximum_relative_error(const T * A, const T * B, const size_t N) {
	T max_error = 0;
	T eps = std::sqrt( std::numeric_limits<T>::epsilon() );
	for(size_t i = 0; i < N; i++) {
		const T a = A[i];
		const T b = B[i];
		const T error = std::abs(a - b);
		if (error != 0) {
			max_error = std::max(max_error, error/(std::abs(a) + std::abs(b) + eps) );
			//if (error > 0.0000001) printf("y_to_test[%ld] (%f) != y[%ld] (%f)\n", i, A[i], i, B[i]);
		}
	}
	return max_error;
}

inline void SpmvSerial(int m, int nnz, const uint64_t *Ap, const IndexT *Aj, const ValueT *Ax, const ValueT *x, ValueT *y) {
	for (int i = 0; i < m; i++){
		auto row_begin = Ap[i];
		auto row_end   = Ap[i+1];
		auto sum = y[i];
		for (auto jj = row_begin; jj < row_end; jj++) {
			auto j = Aj[jj];  //column index
			sum += x[j] * Ax[jj];
		}
		y[i] = sum; 
	}
}

template <typename T>
T l2_error(size_t N, const T * a, const T * b) {
	T numerator   = 0;
	T denominator = 0;
	for (size_t i = 0; i < N; i++) {
		numerator   += (a[i] - b[i]) * (a[i] - b[i]);
		denominator += (b[i] * b[i]);
	}
	return numerator/denominator;
}

