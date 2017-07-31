// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "sssp.h"
#include <omp.h>
#include "timer.h"
#include "omp_target_config.h"
#define SSSP_VARIANT "omp_target"

#pragma omp declare target
#include <vector>
#include "platform_atomics.h"
#pragma omp end declare target

void SSSPSolver(int m, int nnz, int source, int *row_offsets, int *column_indices, DistT *weight, DistT *dist, int delta) {
	Timer t;
	dist[source] = 0;
	//int *frontier = (int *) malloc(nnz*sizeof(int));
	//frontier[0] = source;
	t.Start();
	warm_up();
	double t1, t2;
	
#pragma omp target data device(0) map(tofrom:dist[0:m]) map(to:row_offsets[0:(m+1)]) map(to:column_indices[0:nnz]) map(to:weight[0:nnz]) map(to:m,nnz,source,delta) // map(to:frontier[0:nnz])
{
	#pragma omp target device(0)
	{
	vector<int> frontier(nnz);
	frontier[0] = source;
	size_t shared_indexes[2] = {0, kDistInf};
	size_t frontier_tails[2] = {1, 0}; 
	t1 = omp_get_wtime();
	#pragma omp parallel
	{
		std::vector<std::vector<int> > local_bins(0);
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
				int src = frontier[i];
				if (dist[src] >= delta * static_cast<DistT>(curr_bin_index)) {
					int row_begin = row_offsets[src];
					int row_end = row_offsets[src + 1];
					for (int offset = row_begin; offset < row_end; offset ++) {
						int dst = column_indices[offset];
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
				size_t copy_start = fetch_and_add(next_frontier_tail, local_bins[next_bin_index].size());
				copy(local_bins[next_bin_index].begin(), local_bins[next_bin_index].end(), frontier.data() + copy_start);
				local_bins[next_bin_index].resize(0);
			}
			iter++;
			#pragma omp barrier
		}
	}
	t2 = omp_get_wtime();
	}
}
	t.Stop();
	//printf("\truntime [%s] = %f ms.\n", SSSP_VARIANT, t.Millisecs());
	printf("\truntime [%s] = %f ms.\n", SSSP_VARIANT, 1000*(t2-t1));
	return;
}
