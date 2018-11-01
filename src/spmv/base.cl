// Copyright 2018, National University of Defense Technology
// Authors: Xuhao Chen <cxh.nudt@gmail.com>

__kernel void spmv_kernel(int m, __global int *Ap, __global int *Aj, __global float *Ax, __global float *x, __global float *y) {
	int id = get_global_id(0);
	if (id < m) {
		int row_begin = Ap[id];
		int row_end = Ap[id+1];
		float sum = y[id];
		for (int jj = row_begin; jj < row_end; jj ++) {
			int j = Aj[jj];
			sum += x[j] * Ax[jj];
		}
		y[id] = sum;
	}
}
