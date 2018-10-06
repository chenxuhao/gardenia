// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "sssp.h"
#include <omp.h>
#include <vector>
#include <stdlib.h>
#include "timer.h"
#include "platform_atomics.h"
#ifdef SIM
#include "sim.h"
#endif

#define SSSP_VARIANT "openmp"
/*
[1] Ulrich Meyer and Peter Sanders. "δ-stepping: a parallelizable shortest path
	algorithm." Journal of Algorithms, 49(1):114–152, 2003.
*/

void SSSPSolver(int m, int nnz, int source, IndexType *row_offsets, IndexType *column_indices, DistT *weight, DistT *dist, int delta) {
	//omp_set_num_threads(8);
	int num_threads = 1;
#ifdef SIM
	omp_set_num_threads(4);
	map_m5_mem();
#endif
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP SSSP solver (%d threads) ...\n", num_threads);
	Timer t;
	dist[source] = 0;
	IndexType *frontier = (IndexType *)malloc(nnz*sizeof(IndexType));
	// two element arrays for double buffering curr=iter&1, next=(iter+1)&1
	size_t shared_indexes[2] = {0, kDistInf};
	size_t frontier_tails[2] = {1, 0}; 
	frontier[0] = source;

	t.Start();
#ifdef SIM
	m5_checkpoint(0,0);
	set_addr_bounds(0,(uint64_t)frontier,(uint64_t)&frontier[nnz],8);
	set_addr_bounds(1,(uint64_t)row_offsets,(uint64_t)&row_offsets[m+1],4);
	set_addr_bounds(2,(uint64_t)column_indices,(uint64_t)&column_indices[nnz],8);
	set_addr_bounds(3,(uint64_t)dist,(uint64_t)&dist[m],8);
	set_addr_bounds(5,(uint64_t)weight,(uint64_t)&weight[nnz],8);
	printf("Begin of ROI\n");
#endif
	#pragma omp parallel
	{
		vector<vector<IndexType> > local_bins(0);
		int iter = 0;
		while (static_cast<DistT>(shared_indexes[iter&1]) != kDistInf) {
			size_t &curr_bin_index = shared_indexes[iter&1];
			size_t &next_bin_index = shared_indexes[(iter+1)&1];
			size_t &curr_frontier_tail = frontier_tails[iter&1];
			size_t &next_frontier_tail = frontier_tails[(iter+1)&1];
			//#pragma omp single
			//printf("\titer = %d, frontier_size = %ld.\n", iter, curr_frontier_tail);
			#pragma omp for nowait schedule(dynamic, 64)
			for (size_t i = 0; i < curr_frontier_tail; i ++) {
				IndexType src = frontier[i];
				if (dist[src] >= delta * static_cast<DistT>(curr_bin_index)) {
					IndexType row_begin = row_offsets[src];
					IndexType row_end = row_offsets[src + 1];
					for (IndexType offset = row_begin; offset < row_end; offset ++) {
						IndexType dst = column_indices[offset];
						DistT old_dist = dist[dst];
						DistT new_dist = dist[src] + weight[offset];
						if (new_dist < old_dist) {
							bool changed_dist = true;
							while (!compare_and_swap(dist[dst], old_dist, new_dist)) {
								old_dist = dist[dst];
								if (old_dist <= new_dist) {
									changed_dist = false;
									break;
								}
							}
							if (changed_dist) {
								size_t dest_bin = new_dist/delta;
								if (dest_bin >= local_bins.size()) {
									local_bins.resize(dest_bin+1);
								}
								local_bins[dest_bin].push_back(dst);
							}
						}
					}
				}
			}
			for (size_t i = curr_bin_index; i < local_bins.size(); i ++) {
				if (!local_bins[i].empty()) {
					#pragma omp critical
					next_bin_index = min(next_bin_index, i);
					break;
				}
			}
			#pragma omp barrier
			#pragma omp single nowait
			{
				curr_bin_index = kDistInf;
				curr_frontier_tail = 0;
			}
			if (next_bin_index < local_bins.size()) {
				size_t copy_start = fetch_and_add(next_frontier_tail,
						local_bins[next_bin_index].size());
				copy(local_bins[next_bin_index].begin(),
						local_bins[next_bin_index].end(), frontier + copy_start);
				local_bins[next_bin_index].resize(0);
			}
			iter++;
			#pragma omp barrier
		}
	}
#ifdef SIM
	printf("End of ROI\n");
	m5_dumpreset_stats(0,0);
#endif
	t.Stop();
	//printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", SSSP_VARIANT, t.Millisecs());
	return;
}
