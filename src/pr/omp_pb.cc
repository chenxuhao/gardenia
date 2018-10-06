// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "pr.h"
#include <omp.h>
#include <stdlib.h>
#include <vector>
#include "timer.h"
#include "immintrin.h"
#include "platform_atomics.h"
#include <boost/align/aligned_allocator.hpp>
template <typename T>
using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, 32>>;

#define PR_VARIANT "omp_pb" // propagation blocking

typedef pair<ScoreT, IndexT> WN;
const size_t buf_size = 16; // 32*16 bits = 64 Bytes (cache-line size)

template <typename T>
void streaming_store(T *src, T *dst) {
	__m256i r0 = _mm256_load_si256((__m256i*) &src[0]);
	__m256i r1 = _mm256_load_si256((__m256i*) &src[8]);
	_mm256_stream_si256((__m256i*) &dst[0], r0);
	_mm256_stream_si256((__m256i*) &dst[8], r1);
	return;
}

void PRSolver(int m, int nnz, IndexT *row_offsets, IndexT *column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *degree, ScoreT *scores) {
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP PR solver (%d threads) ...\n", num_threads);
	const ScoreT base_score = (1.0f - kDamp) / m;
	//ScoreT *sums = (ScoreT *) malloc(m * sizeof(ScoreT));
	//for (int i = 0; i < m; i ++) { sums[i] = 0; }
	vector<ScoreT> sums(m, 0);
	int binWidth = 128 * 1024; // 512KB = 128K vertices
	int numBins = (m-1) / binWidth + 1; // the number of bins is the number of vertices in the graph divided by the bin width
	printf("number of bins: %d\n", numBins);

	int iter;
	Timer t;
	t.Start();
#pragma omp parallel
{
	aligned_vector<aligned_vector<IndexT> > vertex_bins(numBins);
	aligned_vector<aligned_vector<ScoreT> > contri_bins(numBins);
	aligned_vector<aligned_vector<IndexT> > vertex_bufs(numBins, aligned_vector<IndexT>(buf_size)); // in-cache buffer
	aligned_vector<aligned_vector<ScoreT> > contri_bufs(numBins, aligned_vector<ScoreT>(buf_size));
	vector<size_t> counter(numBins, 0);

	// the first iteration
	#pragma omp for
	for (int u = 0; u < m; u ++) {
		const IndexT row_begin = out_row_offsets[u];
		const IndexT row_end = out_row_offsets[u + 1];
		int degree = row_end - row_begin;
		ScoreT c = scores[u] / (ScoreT)degree; // contribution
		for (IndexT offset = row_begin; offset < row_end; offset ++) {
			IndexT v = out_column_indices[offset];
			int dest_bin = v >> 17; // v / binWidth (2^17)
			if (counter[dest_bin] < buf_size) {
				vertex_bufs[dest_bin].push_back(v);
				contri_bufs[dest_bin].push_back(c);
				counter[dest_bin] ++;
				if (counter[dest_bin] == buf_size) {
					// buffer full, dump the data into memory
					int size = contri_bins[dest_bin].size();
					vertex_bins[dest_bin].resize(size+buf_size);
					contri_bins[dest_bin].resize(size+buf_size);
					streaming_store<IndexT>(vertex_bufs[dest_bin].data(), vertex_bins[dest_bin].data()+size);
					streaming_store<ScoreT>(contri_bufs[dest_bin].data(), contri_bins[dest_bin].data()+size);
					vertex_bufs[dest_bin].resize(0);
					contri_bufs[dest_bin].resize(0);
					counter[dest_bin] = 0;
				}
			}
		}
	}
	// dump the residual data in the buffer
	//#pragma omp for
	for (int bid = 0; bid < numBins; bid ++) {
		if (counter[bid] > 0) {
			// padding
			do {
				vertex_bufs[bid].push_back(0);
				contri_bufs[bid].push_back(0);
				counter[bid] ++;
			} while (counter[bid] != buf_size);

			// dump buffer to memory
			int size = contri_bins[bid].size();
				vertex_bins[bid].resize(size+buf_size);
				contri_bins[bid].resize(size+buf_size);
				streaming_store<IndexT>(vertex_bufs[bid].data(), vertex_bins[bid].data()+size);
				streaming_store<ScoreT>(contri_bufs[bid].data(), contri_bins[bid].data()+size);
				vertex_bufs[bid].resize(0);
				contri_bufs[bid].resize(0);
				counter[bid] = 0;
		}
	}
	//#pragma omp for
	for (int bid = 0; bid < numBins; bid ++) {
		for(size_t k = 0; k < vertex_bins[bid].size(); k++) {
			ScoreT c = contri_bins[bid][k];
			IndexT v = vertex_bins[bid][k];
			sums[v] = sums[v] + c;
		}
	}
	for (int bid = 0; bid < numBins; bid ++) {
		contri_bins[bid].resize(0);
	}
	double error = 0;
	//#pragma omp parallel for reduction(+ : error)
	for (int u = 0; u < m; u ++) {
		ScoreT new_score = base_score + kDamp * sums[u];
		error += fabs(new_score - scores[u]);
		scores[u] = new_score;
		sums[u] = 0;
	}
	printf(" %2d    %lf\n", 1, error);
}
	return;
}
/*
	// the following iterations
	for (iter = 1; iter < MAX_ITER; iter ++) {
		//#pragma omp for //schedule(dynamic, 64)
		for (int u = 0; u < m; u ++) {
			const IndexT row_begin = out_row_offsets[u];
			const IndexT row_end = out_row_offsets[u + 1];
			int degree = row_end - row_begin;
			ScoreT c = scores[u] / (ScoreT)degree; // contribution
			for (IndexT offset = row_begin; offset < row_end; offset ++) {
				IndexT v = out_column_indices[offset];
				int dest_bin = v >> 17; // v / binWidth (2^17)
				//vertex_bins[dest_bin].push_back(v);
				//contri_bins[dest_bin].push_back(c);
				if (counter[dest_bin] < buf_size) {
					contri_bufs[dest_bin].push_back(c);
					counter[dest_bin] ++;
					if (counter[dest_bin] == buf_size) {
						int size = contri_bins[dest_bin].size();
						contri_bins[dest_bin].resize(size+buf_size);
						streaming_store<ScoreT>(contri_bufs[dest_bin].data(), contri_bins[dest_bin].data()+size);
						contri_bufs[dest_bin].resize(0);
						counter[dest_bin] = 0;
					}
				}
			}
		}
		// dump the residual data in the buffer
		#pragma omp for
		for (int bid = 0; bid < numBins; bid ++) {
			if (counter[bid] > 0) {
				// padding
				do {
					contri_bufs[bid].push_back(0);
					counter[bid] ++;
				} while (counter[bid] != buf_size);

				// dump buffer to memory
				int size = contri_bins[bid].size();
				contri_bins[bid].resize(size+buf_size);
				streaming_store<ScoreT>(contri_bufs[bid].data(), contri_bins[bid].data()+size);
				contri_bufs[bid].resize(0);
				counter[bid] = 0;
			}
		}
		#pragma omp for
		for (int bid = 0; bid < numBins; bid ++) {
			for(size_t k = 0; k < vertex_bins[bid].size(); k++) {
				ScoreT c = contri_bins[bid][k];
				IndexT v = vertex_bins[bid][k];
				sums[v] = sums[v] + c;
			}
		}
		for (int bid = 0; bid < numBins; bid ++) {
			contri_bins[bid].resize(0);
		}
		double error = 0;
		#pragma omp parallel for reduction(+ : error)
		for (int u = 0; u < m; u ++) {
			ScoreT new_score = base_score + kDamp * sums[u];
			error += fabs(new_score - scores[u]);
			scores[u] = new_score;
			sums[u] = 0;
		}
		printf(" %2d    %lf\n", iter+1, error);
		if (error < EPSILON) break;
	}
}
	t.Stop();
	printf("\titerations = %d.\n", iter+1);
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
	return;
}
//*/
