// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "pr.h"
#include <omp.h>
#include "timer.h"
#include "omp_target_config.h"
#define PR_VARIANT "omp_target"

#pragma omp declare target
#include "immintrin.h"
#pragma omp end declare target

void PRSolver(int m, int nnz, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, int *degree, ScoreT *scores) {
	double t1, t2;
	const ScoreT base_score = (1.0f - kDamp) / m;
	int *row_offsets = (int *) _mm_malloc((m+1)*sizeof(int), 64);
	int *column_indices = (int *) _mm_malloc(nnz*sizeof(int), 64);
	ScoreT *outgoing_contrib = (ScoreT *) _mm_malloc(m*sizeof(ScoreT), 64);
	for (int i = 0; i < m+1; i ++) row_offsets[i] = in_row_offsets[i];
	for (int i = 0; i < nnz; i ++) column_indices[i] = in_column_indices[i];
	warm_up();
	int iter;
	Timer t;
	t.Start();
#pragma omp target data device(0) map(to:column_indices[0:nnz]) map(tofrom:scores[0:m]) map(to:row_offsets[0:(m+1)]) map(to:degree[0:m]) map(to:outgoing_contrib[0:m]) map(to:base_score)
{
	t1 = omp_get_wtime();
	for (iter = 0; iter < MAX_ITER; iter ++) {
		double error = 0;
		//printf("iter=%d\n", iter);
		#pragma omp target device(0)
		#pragma omp parallel for simd
		for (int n = 0; n < m; n ++)
			outgoing_contrib[n] = scores[n] / degree[n];
		#pragma omp target device(0)
		#pragma omp parallel for reduction(+ : error) schedule(dynamic, 64)
		for (int src = 0; src < m; src ++) {
			ScoreT incoming_total = 0;
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src + 1];
			/*
			//#pragma omp simd
			for (int offset = row_begin; offset < row_end; offset ++) {
				int dst = column_indices[offset];
				incoming_total += outgoing_contrib[dst];
			}
			//*/
			//*
			//printf("\nsrc=%d, row_begin=%d, row_end=%d\n", src, row_begin, row_end);
			// offset should be times of 16 (64B aligned)
			for (int offset = row_begin - (row_begin & (16 - 1)); offset < row_end; offset += 16) {
				__m512i begin = _mm512_set1_epi32(row_begin);
				__m512i end = _mm512_set1_epi32(row_end);
				__m512i idx, dst;
				__mmask16 mask, mask1, mask2;
				idx = _mm512_set_epi32(offset+15, offset+14, offset+13, offset+12, 
					offset+11, offset+10, offset+9, offset+8,
					offset+7, offset+6, offset+5, offset+4,
					offset+3, offset+2, offset+1, offset);
				mask1 = _mm512_cmp_epi32_mask(end, idx, _MM_CMPINT_NLE);
				mask2 = _mm512_cmp_epi32_mask(begin, idx, _MM_CMPINT_LE);
				mask = _mm512_kand(mask1, mask2);
				//printf("src=%d, mask1=%.4x, mask2=%.4x, mask=%.4x\n", src, mask1, mask2, mask);
				dst = _mm512_setzero_epi32();
				dst = _mm512_mask_load_epi32(dst, mask, column_indices+offset); // load 16 integer
				__m512 contrib = _mm512_setzero_ps();
				contrib = _mm512_mask_i32gather_ps(contrib, mask, dst, outgoing_contrib, 4);
				//printf("dst=%o, contrib=%o\n", dst, contrib);
				incoming_total += _mm512_reduce_add_ps(contrib);
			}
			//*/
			ScoreT old_score = scores[src];
			scores[src] = base_score + kDamp * incoming_total;
			error += fabs(scores[src] - old_score);
		}   
		printf(" %2d    %lf\n", iter+1, error);
		if (error < EPSILON) break;
	}
	t2 = omp_get_wtime();
}

	t.Stop();
	printf("\titerations = %d.\n", iter+1);
	//printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, 1000*(t2-t1));
	return;
}
