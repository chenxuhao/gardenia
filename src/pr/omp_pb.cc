// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "pr.h"
#include <vector>
#include "timer.h"
#include "prop_blocking.h"
#define PR_VARIANT "omp_pb" // propagation blocking

// m: number of vertices, nnz: number of non-zero values
void PRSolver(int m, int nnz, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *degrees, ScoreT *scores) {
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
	int num_bins = (m-1) / BIN_WIDTH + 1; // the number of bins is the number of vertices in the graph divided by the bin width
	printf("number of bins: %d\n", num_bins);

	int iter = 0;
	double error = 0;
	vector<vector<aligned_vector<IndexT> > > local_vertex_bins(num_threads);
	vector<vector<aligned_vector<ScoreT> > > local_contri_bins(num_threads);
	vector<aligned_vector<IndexT> > global_vertex_bins(num_bins);
	vector<aligned_vector<ScoreT> > global_contri_bins(num_bins);
	vector<vector<aligned_vector<IndexT> > > vertex_bufs(num_threads);
	vector<vector<aligned_vector<ScoreT> > > contri_bufs(num_threads);
	vector<vector<size_t> > buf_counter(num_threads);
	for (int tid = 0; tid < num_threads; tid ++) {
		local_vertex_bins[tid].resize(num_bins);
		local_contri_bins[tid].resize(num_bins);
		vertex_bufs[tid].resize(num_bins);
		contri_bufs[tid].resize(num_bins);
		buf_counter[tid].resize(num_bins);
		for (int bid = 0; bid < num_bins; bid ++) {
			vertex_bufs[tid][bid].resize(buf_size);
			contri_bufs[tid][bid].resize(buf_size);
			buf_counter[tid][bid] = 0;
		}
	}

	Timer t;
	t.Start();
	// the first iteration
	//printf("Binning phase\n");
	#pragma omp parallel for //schedule(dynamic, 64)
	for (int u = 0; u < m; u ++) {
		int tid = omp_get_thread_num();
		const IndexT row_begin = out_row_offsets[u];
		const IndexT row_end = out_row_offsets[u+1];
		int degree = row_end - row_begin;
		ScoreT c = scores[u] / (ScoreT)degree; // contribution
		for (IndexT offset = row_begin; offset < row_end; offset ++) {
			IndexT v = out_column_indices[offset];
			int dest_bin = v >> BITS; // v / BIN_WIDTH
			if (buf_counter[tid][dest_bin] < buf_size) {
				vertex_bufs[tid][dest_bin][buf_counter[tid][dest_bin]] = v;
				contri_bufs[tid][dest_bin][buf_counter[tid][dest_bin]] = c;
				buf_counter[tid][dest_bin] ++;
				if (buf_counter[tid][dest_bin] == buf_size) {
					// buffer full, dump the data into memory
					int size = local_contri_bins[tid][dest_bin].size();
					local_vertex_bins[tid][dest_bin].resize(size+buf_size);
					local_contri_bins[tid][dest_bin].resize(size+buf_size);
					streaming_store<IndexT>(vertex_bufs[tid][dest_bin].data(), local_vertex_bins[tid][dest_bin].data()+size);
					streaming_store<ScoreT>(contri_bufs[tid][dest_bin].data(), local_contri_bins[tid][dest_bin].data()+size);
					buf_counter[tid][dest_bin] = 0;
				}
			}
		}
	}
	t.Stop();
	printf("\truntime[binning] = %f ms.\n", t.Millisecs());
	t.Start();
	///*
	//printf("dump the residual data in the buffer\n");
	for (int tid = 0; tid < num_threads; tid ++) {
		for (int bid = 0; bid < num_bins; bid ++) {
			if (buf_counter[tid][bid] > 0) {
				// padding
				do {
					vertex_bufs[tid][bid][buf_counter[tid][bid]] = 0;
					contri_bufs[tid][bid][buf_counter[tid][bid]] = 0;
					buf_counter[tid][bid] ++;
				} while (buf_counter[tid][bid] != buf_size);
				// dump buffer to memory
				int size = local_contri_bins[tid][bid].size();
				local_vertex_bins[tid][bid].resize(size+buf_size);
				local_contri_bins[tid][bid].resize(size+buf_size);
				streaming_store<IndexT>(vertex_bufs[tid][bid].data(), local_vertex_bins[tid][bid].data()+size);
				streaming_store<ScoreT>(contri_bufs[tid][bid].data(), local_contri_bins[tid][bid].data()+size);
				buf_counter[tid][bid] = 0;
			}
		}
	}
	//*/

	t.Stop();
	printf("\truntime[dump] = %f ms.\n", t.Millisecs());
	t.Start();
	//printf("merge the local bins into global bins\n");
	vector<int> global_size(num_bins, 0);
	for (int tid = 0; tid < num_threads; tid ++) {
		#pragma omp parallel for
		for (int bid = 0; bid < num_bins; bid ++) {
			int local_size = local_vertex_bins[tid][bid].size();
			size_t copy_start = global_size[bid];
			global_size[bid] += local_size;
			global_vertex_bins[bid].resize(global_size[bid]);
			global_contri_bins[bid].resize(global_size[bid]);
			IndexT *local_bin = local_vertex_bins[tid][bid].data();
			IndexT *global_bin = global_vertex_bins[bid].data();
			std::copy(local_bin, local_bin+local_size, global_bin+copy_start);
			ScoreT *local_contri_bin = local_contri_bins[tid][bid].data();
			ScoreT *global_contri_bin = global_contri_bins[bid].data();
			std::copy(local_contri_bin, local_contri_bin+local_size, global_contri_bin+copy_start);
		}
	}
	t.Stop();
	printf("\truntime[merge] = %f ms.\n", t.Millisecs());
	t.Start();
	//printf("Accumulate phase\n");
	#pragma omp parallel for
	for (int bid = 0; bid < num_bins; bid ++) {
		for(size_t k = 0; k < global_vertex_bins[bid].size(); k++) {
			ScoreT c = global_contri_bins[bid][k];
			IndexT v = global_vertex_bins[bid][k];
			sums[v] = sums[v] + c;
		}
	}
	t.Stop();
	printf("\truntime[accumulate] = %f ms.\n", t.Millisecs());
	t.Start();
	#pragma omp parallel for reduction(+ : error)
	for (int u = 0; u < m; u ++) {
		ScoreT new_score = base_score + kDamp * sums[u];
		error += fabs(new_score - scores[u]);
		scores[u] = new_score;
		sums[u] = 0;
	}
	t.Stop();
	printf("\truntime[reduction] = %f ms.\n", t.Millisecs());
	t.Start();
	printf(" %2d    %lf\n", 1, error);
//}
	//for(int i = 0; i < 4; i ++) printf("scores[%d]=%f\n", i, scores[i]);
	//return;}
///*
	// the following iterations
	vector<vector<int> > pos(num_threads, vector<int>(num_bins));
	do {
		iter ++;
		for (int tid = 0; tid < num_threads; tid ++) {
			for (int bid = 0; bid < num_bins; bid ++) {
				pos[tid][bid] = 0;
			}
		}
		#pragma omp parallel for //schedule(dynamic, 64)
		for (int u = 0; u < m; u ++) {
			int tid = omp_get_thread_num();
			const IndexT row_begin = out_row_offsets[u];
			const IndexT row_end = out_row_offsets[u+1];
			int degree = row_end - row_begin;
			ScoreT c = scores[u] / (ScoreT)degree; // contribution
			for (IndexT offset = row_begin; offset < row_end; offset ++) {
				IndexT v = out_column_indices[offset];
				int dest_bin = v >> BITS; // v / BIN_WIDTH
				if (buf_counter[tid][dest_bin] < buf_size) {
					contri_bufs[tid][dest_bin][buf_counter[tid][dest_bin]++] = c;
					if (buf_counter[tid][dest_bin] == buf_size) {
						streaming_store<ScoreT>(contri_bufs[tid][dest_bin].data(), local_contri_bins[tid][dest_bin].data()+pos[tid][dest_bin]);
						pos[tid][dest_bin] += buf_size;
						buf_counter[tid][dest_bin] = 0;
					}
				}
			}
		}
		for (int tid = 0; tid < num_threads; tid ++) {
			for (int bid = 0; bid < num_bins; bid ++) {
				if (buf_counter[tid][bid] > 0) {
					// padding
					do {
						contri_bufs[tid][bid][buf_counter[tid][bid]++] = 0;
					} while (buf_counter[tid][bid] != buf_size);
					// dump buffer to memory
					streaming_store<ScoreT>(contri_bufs[tid][bid].data(), local_contri_bins[tid][bid].data()+pos[tid][bid]);
					pos[tid][bid] += buf_size;
					buf_counter[tid][bid] = 0;
				}
			}
		}

		//printf("merge the local bins into global bins\n");
		for (int bid = 0; bid < num_bins; bid ++) {
			int global_pos = 0;
			for (int tid = 0; tid < num_threads; tid ++) {
				int local_size = local_contri_bins[tid][bid].size();
				size_t copy_start = global_pos;
				global_pos += local_size;
				ScoreT *local_contri_bin = local_contri_bins[tid][bid].data();
				ScoreT *global_contri_bin = global_contri_bins[bid].data();
				std::copy(local_contri_bin, local_contri_bin+local_size, global_contri_bin+copy_start);
			}
		}

		#pragma omp parallel for
		for (int bid = 0; bid < num_bins; bid ++) {
			for(size_t k = 0; k < global_vertex_bins[bid].size(); k++) {
				ScoreT c = global_contri_bins[bid][k];
				IndexT v = global_vertex_bins[bid][k];
				sums[v] = sums[v] + c;
			}
		}
		error = 0;
		#pragma omp parallel for reduction(+ : error)
		for (int u = 0; u < m; u ++) {
			ScoreT new_score = base_score + kDamp * sums[u];
			error += fabs(new_score - scores[u]);
			scores[u] = new_score;
			sums[u] = 0;
		}
		printf(" %2d    %lf\n", iter+1, error);
		if (error < EPSILON) break;
	} while(iter < MAX_ITER);
	t.Stop();
	printf("\titerations = %d.\n", iter+1);
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
	return;
}
//*/
