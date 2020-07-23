// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "spmv.h"
#include "timer.h"
#include "immintrin.h"
#define SPMV_VARIANT "omp_simd"
// sum up 8 single-precision numbers
inline float hsum_avx(__m256 in256) {
	float *sum = (float *) malloc(sizeof(float));
	__m256 hsum = _mm256_hadd_ps(in256, in256);
	hsum = _mm256_add_ps(hsum, _mm256_permute2f128_ps(hsum, hsum, 0x1));
	__m128 res = _mm_hadd_ps( _mm256_castps256_ps128(hsum), _mm256_castps256_ps128(hsum) );
	printf("res=%f\n", res);
	//_mm_store_ss(sum, res);
	//_mm_store_ss(&sum, _mm_hadd_ps( _mm256_castps256_ps128(hsum), _mm256_castps256_ps128(hsum) ) );
	return *sum;
}

void SpmvSolver(int m, int nnz, IndexT *ApT, IndexT *AjT, ValueT *AxT, IndexT *h_Ap, IndexT *h_Aj, ValueT *h_Ax, ValueT *h_x, ValueT *y, int *degrees) {
	int num_threads = 1;
	#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	int *Ap = (int *) _mm_malloc((m+1)*sizeof(int), 64);
	int *Aj = (int *) _mm_malloc(nnz*sizeof(int), 64);
	ValueT *Ax = (ValueT *) _mm_malloc(nnz*sizeof(ValueT), 64);
	ValueT *x = (ValueT *) _mm_malloc(m*sizeof(ValueT), 64);
	for (int i = 0; i < m+1; i ++) Ap[i] = h_Ap[i];
	for (int i = 0; i < nnz; i ++) Aj[i] = h_Aj[i];
	for (int i = 0; i < nnz; i ++) Ax[i] = h_Ax[i];
	for (int i = 0; i < m; i ++) x[i] = h_x[i];
	printf("Launching OpenMP SpMV solver (%d threads) ...\n", num_threads);
	
	Timer t;
	t.Start();
	#pragma omp parallel for
	for (int i = 0; i < m; i++){
		int row_begin = Ap[i];
		int row_end   = Ap[i+1];
		__m256 sum = _mm256_setzero_ps();// Returns an vector whose bytes are set to zero
		for (int jj = row_begin; jj < row_end; jj += 8) {
			__m256i begin = _mm256_set1_epi32(row_begin);
			__m256i end = _mm256_set1_epi32(row_end);
			__m256i idx = _mm256_set_epi32(jj+7, jj+6, jj+5, jj+4, jj+3, jj+2, jj+1, jj);
			printf("begin=%o, end=%o, idx=%o\n", begin, end, idx);
			__mmask8 mask, mask_begin, mask_end;
			mask_begin = _mm256_cmp_epi32_mask(begin, idx, _MM_CMPINT_LE);
			printf("mask_begin=%o\n", mask_begin);
/*
			mask = _mm256_mask_cmp_epi32_mask(mask_begin, end, idx, _MM_CMPINT_NLE);
			printf("mask=%o\n", mask);
			__m256i vec_idx = _mm256_setzero_si256();
			vec_idx = _mm256_mask_load_epi32(vec_idx, mask, Aj+jj);
			__m256 mat_val = _mm256_setzero_ps();
			mat_val = _mm256_mask_load_ps(mat_val, mask, Ax+jj);
			__m256 vec_x = _mm256_setzero_ps();
			vec_x = _mm256_mmask_i32gather_ps(vec_x, mask, vec_idx, x, 4);
			sum = _mm256_fmadd_ps(mat_val, vec_x, sum);
//*/
		}
		//y[i] += hsum_avx(sum); 
	}
	t.Stop();
	printf("\truntime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());
	return;
}
