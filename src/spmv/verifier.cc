// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "spmv.h"
#include <limits>
#include <cmath>
#include "timer.h"
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

void SpmvVerifier(int num_rows, int *Ap, int *Aj, ValueType *Ax, ValueType *x, ValueType *test_y, ValueType *y_host) {
	printf("Verifying...\n");
	ValueType *y = (ValueType *)malloc(num_rows * sizeof(ValueType));
	for(int i = 0; i < num_rows; i++)
		y[i] = y_host[i];
	Timer t;
	t.Start();
	for (int i = 0; i < num_rows; i++){
		int row_begin = Ap[i];
		int row_end   = Ap[i+1];
		ValueType sum = y[i];
		for (int jj = row_begin; jj < row_end; jj++) {
			const int j = Aj[jj];  //column index
			sum += x[j] * Ax[jj];
		}
		y[i] = sum; 
	}
	t.Stop();
	printf("\truntime [serial] = %f ms.\n", t.Millisecs());

	ValueType max_error = maximum_relative_error(test_y, y, num_rows);
	printf("\t[max error %9f]\n", max_error);
	//for(int i = 0; i < num_rows; i++) printf("test_y[%d] = %f, y[%d] = %f\n", i, test_y[i], i, y[i]);
	if ( max_error > 5 * std::sqrt( std::numeric_limits<ValueType>::epsilon() ) )
		printf("POSSIBLE FAILURE\n");
	else
		printf("Correct\n");
}
