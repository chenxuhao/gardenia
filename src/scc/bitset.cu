// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "scc.h"
#include "bitset.h"
#include "cutil_subset.h"
#include <set>
#include <cub/cub.cuh>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include "timer.h"
#define debug 0
// find forward reachable vertices
__global__ void fwd_step(int m, int *row_offsets, int *column_indices, unsigned *colors, unsigned char *status, int *scc_root, bool *changed) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src = tid;
	// if src is in the forward frontier (valid, visited but not expanded)
	if(src < m && is_fwd_front(status[src])) {
		set_fwd_expanded(&status[src]);
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1]; 
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			// if dst is valid, not visited and has the same color as src
			if (!is_removed(status[dst]) && !is_fwd_visited(status[dst]) && (colors[dst] == colors[src])) {
				*changed = true;
				set_fwd_visited(&status[dst]);
				scc_root[dst] = scc_root[src];
			}
		}
	}
}

__global__ void fwd_step_lb(int m, int *row_offsets, int *column_indices, unsigned char *status, int *scc_root, bool *changed) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
	const int SCRATCHSIZE = BLOCK_SIZE;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[SCRATCHSIZE];
	__shared__ unsigned srcsrc[BLOCK_SIZE];
	gather_offsets[threadIdx.x] = 0;
	int neighbor_size = 0;
	int neighbor_offset = 0;
	int scratch_offset = 0;
	int total_edges = 0;
	if(tid < m && is_fwd_front(status[tid])) {
		set_fwd_expanded(&status[tid]);
		neighbor_offset = row_offsets[tid];
		neighbor_size = row_offsets[tid+1] - neighbor_offset;
	}
	BlockScan(temp_storage).ExclusiveSum(neighbor_size, scratch_offset, total_edges);
	int done = 0;
	int neighbors_done = 0;
	while(total_edges > 0) {
		__syncthreads();
		int i, index;
		for(i = 0; neighbors_done + i < neighbor_size && (index = scratch_offset + i - done) < SCRATCHSIZE; i++) {
			gather_offsets[index] = neighbor_offset + neighbors_done + i;
			srcsrc[index] = tid;
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		int src, dst = 0;
		int edge = gather_offsets[threadIdx.x];
		if(threadIdx.x < total_edges) {
			dst = column_indices[edge];
			src = srcsrc[threadIdx.x];
			// if dst is valid, not visited
			if (!is_removed(status[dst]) && !is_fwd_visited(status[dst])) {
				*changed = true;
				set_fwd_visited(&status[dst]);
				scc_root[dst] = scc_root[src];
			}
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
}

// find backward reachable vertices
__global__ void bwd_step(int m, int *row_offsets, int *column_indices, unsigned *colors, unsigned char *status, bool *changed) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src = tid;
	// if src is in the forward frontier (valid, visited but not expanded)
	if(src < m && is_bwd_front(status[src])) {
		set_bwd_expanded(&status[src]);
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1]; 
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			// if dst is valid, not visited and has the same color as src
			if (!is_removed(status[dst]) && !is_bwd_visited(status[dst]) && (colors[dst] == colors[src])) {
				*changed = true;
				set_bwd_visited(&status[dst]);
			}
		}
	}
}

__global__ void bwd_step_lb(int m, int *row_offsets, int *column_indices, unsigned char *status, bool *changed) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
	const int SCRATCHSIZE = BLOCK_SIZE;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[SCRATCHSIZE];
	gather_offsets[threadIdx.x] = 0;
	int neighbor_size = 0;
	int neighbor_offset = 0;
	int scratch_offset = 0;
	int total_edges = 0;
	if(tid < m && is_bwd_front(status[tid])) {
		set_bwd_expanded(&status[tid]);
		neighbor_offset = row_offsets[tid];
		neighbor_size = row_offsets[tid+1] - neighbor_offset;
	}
	BlockScan(temp_storage).ExclusiveSum(neighbor_size, scratch_offset, total_edges);
	int done = 0;
	int neighbors_done = 0;
	while(total_edges > 0) {
		__syncthreads();
		int i, index;
		for(i = 0; neighbors_done + i < neighbor_size && (index = scratch_offset + i - done) < SCRATCHSIZE; i++) {
			gather_offsets[index] = neighbor_offset + neighbors_done + i;
		}
		neighbors_done += i;
		scratch_offset += i;
		__syncthreads();
		int dst = 0;
		int edge = gather_offsets[threadIdx.x];
		if(threadIdx.x < total_edges) {
			dst = column_indices[edge];
			// if dst is valid, not visited
			if (!is_removed(status[dst]) && !is_bwd_visited(status[dst])) {
				*changed = true;
				set_bwd_visited(&status[dst]);
			}
		}
		total_edges -= BLOCK_SIZE;
		done += BLOCK_SIZE;
	}
}

// trimming trivial SCCs
// Making sure self loops are removed before calling this routine
__global__ void trim_kernel(int m, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, unsigned *colors, unsigned char *status, int *scc_root, bool *changed) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src = tid;
	if(src < m && !is_removed(status[src])) {
		int in_degree = 0;
		int out_degree = 0;
		// calculate the number of incoming neighbors
		int row_begin = in_row_offsets[src];
		int row_end = in_row_offsets[src + 1]; 
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = in_column_indices[offset];
			if(!is_removed(status[dst]) && colors[dst] == colors[src]) { in_degree ++; break; }
		}
		if (in_degree != 0) {
			// calculate the number of outgoing neighbors
			row_begin = out_row_offsets[src];
			row_end = out_row_offsets[src + 1]; 
			for (int offset = row_begin; offset < row_end; ++ offset) {
				int dst = out_column_indices[offset];
				if(!is_removed(status[dst]) && colors[dst] == colors[src]) { out_degree ++; break; }
			}
		}

		// remove (disable) the trival SCC
		if (in_degree == 0 || out_degree == 0) {
			set_removed(&status[src]);
			set_trimmed(&status[src]);
			scc_root[src] = src;
			if(debug) printf("found vertex %d trimmed\n", src);
			*changed = true;
		}
	}
}

__global__ void trim2_kernel(int m, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, unsigned *colors, unsigned char *status, int *scc_root) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src = tid;
	if (src < m && !is_removed(status[src])) {
		unsigned nbr, num_neighbors = 0;
		bool isActive = false;
		// outgoing edges
		int row_begin = out_row_offsets[src];
		int row_end = out_row_offsets[src + 1];
		unsigned c = colors[src];
		for (int offset = row_begin; offset < row_end; offset ++) {
			int dst = out_column_indices[offset];
			if (src != dst && c == colors[dst] && !is_removed(status[dst])) {
				num_neighbors++;
				if (num_neighbors > 1) break;
				nbr = dst;
			}
		}
		if (num_neighbors == 1) {
			num_neighbors = 0;
			row_begin = out_row_offsets[nbr];
			row_end = out_row_offsets[nbr + 1];
			for (int offset = row_begin; offset < row_end; offset ++) {
				int dst = out_column_indices[offset];
				if (nbr != dst && c == colors[dst] && !is_removed(status[dst])) {
					if (dst != src) {
						isActive = true;
						break;
					}
					num_neighbors++;
				}
			}
			if (!isActive && num_neighbors == 1) {
				if (src < nbr) {
					status[src] = 20;
					status[nbr] = 4;
					scc_root[src] = src;
					scc_root[nbr] = src;
				} else {
					status[src] = 4;
					status[nbr] = 20;
					scc_root[src] = nbr;
					scc_root[nbr] = nbr;
				}
				return;
			}	
		}
		num_neighbors = 0;
		isActive = false;
		// incoming edges
		row_begin = in_row_offsets[src];
		row_end = in_row_offsets[src + 1];
		for (int offset = row_begin; offset < row_end; offset ++) {
			int dst = in_column_indices[offset];
			if (src != dst && c == colors[dst] && !is_removed(status[dst])) {
				num_neighbors++;
				if (num_neighbors > 1) break;
				nbr = dst;
			}
		}
		if (num_neighbors == 1) {
			num_neighbors = 0;
			row_begin = in_row_offsets[nbr];
			row_end = in_row_offsets[nbr + 1];
			for (int offset = row_begin; offset < row_end; offset ++) {
				int dst = in_column_indices[offset];
				if (nbr != dst && c == colors[dst] && !is_removed(status[dst])) {
					if (dst != src) {
						isActive = true;
						break;
					}
					num_neighbors++;
				}
			}
			if (!isActive && num_neighbors == 1) {
				if (src < nbr) {
					status[src] = 20;
					status[nbr] = 4;
					scc_root[src] = src;
					scc_root[nbr] = src;
				} else {
					status[src] = 4;
					status[nbr] = 20;
					scc_root[src] = nbr;
					scc_root[nbr] = nbr;
				}
				return;
			}
		}
	}
}

__global__ void first_trim_kernel(int m, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, unsigned char *status, bool *changed) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src = tid;
	if(src < m && !is_removed(status[src])) {
		int in_degree = 0;
		int out_degree = 0;
		// calculate the number of incoming neighbors
		int row_begin = in_row_offsets[src];
		int row_end = in_row_offsets[src + 1]; 
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = in_column_indices[offset];
			if(!is_removed(status[dst])) { in_degree ++; break; }
		}
		if (in_degree != 0) {
			// calculate the number of outgoing neighbors
			row_begin = out_row_offsets[src];
			row_end = out_row_offsets[src + 1]; 
			for (int offset = row_begin; offset < row_end; ++ offset) {
				int dst = out_column_indices[offset];
				if(!is_removed(status[dst])) { out_degree ++; break; }
			}
		}

		// remove (disable) the trival SCC
		if (in_degree == 0 || out_degree == 0) {
			set_removed(&status[src]);
			set_trimmed(&status[src]);
			if(debug) printf("found vertex %d trimmed\n", src);
			*changed = true;
		}
	}
}

__global__ void update_kernel(int m, unsigned *colors, unsigned char *status, unsigned *locks, int *scc_root, bool *has_pivot) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src = tid;
	if (src < m && !is_removed(status[src])) {
		unsigned new_subgraph = (is_fwd_visited(status[src])?1:0) + (is_bwd_visited(status[src])?2:0); // F intersec B == 3, F/B == 1 B/F == 2 (V/F)/B == 0
		if (new_subgraph == 3) {
			set_removed(&status[src]);
			//if(debug) printf("\tfind %d (color %d) in the SCC\n", src, colors[src]);
			return;
		}
		unsigned par_subgraph = colors[src];
		unsigned new_color = 3 * par_subgraph + new_subgraph;
		colors[src] = new_color;
		status[src] = 0;
		//pivots generation
		if (locks[new_color & PIVOT_HASH_CONST] == 0) {
			if (atomicCAS(&locks[new_color & PIVOT_HASH_CONST], 0, src) == 0) {
				*has_pivot = true;
				status[src] = 19; // set fwd_visited bwd_visited & is_pivot
				scc_root[src] = src;
				//if(debug) printf("\tselect %d (color %d) as a pivot\n", src, colors[src]);
			}
		}
	}
}

__global__ void update_colors_kernel(int m, unsigned *colors, unsigned char *status) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if (src < m && !is_removed(status[src])) {   
		unsigned new_subgraph = (is_fwd_visited(status[src])?1:0) + (is_bwd_visited(status[src])?2:0); // F intersec B == 3, F/B == 1 B/F == 2 (V/F)/B == 0
		if (new_subgraph == 3) {
			set_removed(&status[src]);
			return;
		}
		unsigned par_subgraph = colors[src];
		unsigned new_color = 3 * par_subgraph + new_subgraph;
		colors[src] = new_color;
		status[src] = 0;
	}
}	

__global__ void find_removed_vertices_kernel(int m, unsigned char *status, int *mark) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if (src < m && is_removed(status[src]))
		mark[src] = 1;
}

// find forward reachable set
void fwd_reach(int m, int *out_row_offsets, int *out_column_indices, unsigned *colors, unsigned char *status, int *scc_root) {
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	bool h_changed, *d_changed;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));
	do {
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		fwd_step<<<nblocks, nthreads>>>(m, out_row_offsets, out_column_indices, colors, status, scc_root, d_changed);
		CudaTest("solving kernel fw_step failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(h_changed), cudaMemcpyDeviceToHost));
	} while (h_changed);
	CUDA_SAFE_CALL(cudaFree(d_changed));
}

void fwd_reach_lb(int m, int *out_row_offsets, int *out_column_indices, unsigned char *status, int *scc_root) {
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	bool h_changed, *d_changed;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));
	do {
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		fwd_step_lb<<<nblocks, nthreads>>>(m, out_row_offsets, out_column_indices, status, scc_root, d_changed);
		CudaTest("solving kernel fw_step failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(h_changed), cudaMemcpyDeviceToHost));
	} while (h_changed);
	CUDA_SAFE_CALL(cudaFree(d_changed));
}

// find backward reachable set
void bwd_reach(int m, int *in_row_offsets, int *in_column_indices, unsigned *colors, unsigned char *status) {
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	bool h_changed, *d_changed;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));
	do {
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		bwd_step<<<nblocks, nthreads>>>(m, in_row_offsets, in_column_indices, colors, status, d_changed);
		CudaTest("solving kernel bw_step failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
	} while (h_changed);
	CUDA_SAFE_CALL(cudaFree(d_changed));
}

void bwd_reach_lb(int m, int *in_row_offsets, int *in_column_indices, unsigned char *status) {
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	bool h_changed, *d_changed;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));
	do {
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		bwd_step_lb<<<nblocks, nthreads>>>(m, in_row_offsets, in_column_indices, status, d_changed);
		CudaTest("solving kernel fw_step failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(h_changed), cudaMemcpyDeviceToHost));
	} while (h_changed);
	CUDA_SAFE_CALL(cudaFree(d_changed));
}

void iterative_trim(int m, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, unsigned *colors, unsigned char *status, int *scc_root) {
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	bool h_changed, *d_changed;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));
	int iter = 0;
	do {
		iter ++;
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		trim_kernel<<<nblocks, nthreads>>>(m, in_row_offsets, in_column_indices, out_row_offsets, out_column_indices, colors, status, scc_root, d_changed);
		CudaTest("solving kernel trim failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
	} while (h_changed);
	CUDA_SAFE_CALL(cudaFree(d_changed));
}

void first_trim(int m, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, unsigned char *status) {
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	bool h_changed, *d_changed;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));
	int iter = 0;
	do {
		iter ++;
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		first_trim_kernel<<<nblocks, nthreads>>>(m, in_row_offsets, in_column_indices, out_row_offsets, out_column_indices, status, d_changed);
		CudaTest("solving kernel trim failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
	} while (h_changed);
	CUDA_SAFE_CALL(cudaFree(d_changed));
}

bool update(int m, unsigned *colors, unsigned char *status, unsigned *locks, int *scc_root) {
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	bool h_has_pivot, *d_has_pivot;
	h_has_pivot = false;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_has_pivot, sizeof(bool)));
	CUDA_SAFE_CALL(cudaMemcpy(d_has_pivot, &h_has_pivot, sizeof(h_has_pivot), cudaMemcpyHostToDevice));
	update_kernel<<<nblocks, nthreads>>>(m, colors, status, locks, scc_root, d_has_pivot);
	CudaTest("solving kernel update failed");
	CUDA_SAFE_CALL(cudaMemcpy(&h_has_pivot, d_has_pivot, sizeof(h_has_pivot), cudaMemcpyDeviceToHost));
	return h_has_pivot;
}

void update_colors(int m, unsigned *colors, unsigned char *status) {
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	update_colors_kernel<<<nblocks, nthreads>>>(m, colors, status);
	CudaTest("solving kernel update_colors failed");
}

void trim2(int m, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, unsigned *colors, unsigned char *status, int *scc_root) {
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	trim2_kernel<<<nblocks, nthreads>>>(m, in_row_offsets, in_column_indices, out_row_offsets, out_column_indices, colors, status, scc_root);
	CudaTest("solving kernel trim2 failed");
}

void find_removed_vertices(int m, unsigned char *status, int *mark) {
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	find_removed_vertices_kernel<<<nblocks, nthreads>>>(m, status, mark);
	CudaTest("solving kernel update_colors failed");
}

void print_statistics(int m, int *scc_root, unsigned char *status) {
	int total_num_trimmed = 0;
	int total_num_pivots = 0;
	int num_trivial_scc = 0;
	int num_nontrivial_scc = 0;
	int total_num_scc = 0;
	int biggest_scc_size = 0;
	for (int i = 0; i < m; i ++) {
		if (is_trimmed(status[i])) {
			total_num_trimmed ++;
		}
		else if (is_pivot(status[i])) total_num_pivots ++;
	}
	vector<set<int> > scc_sets(m);
	for (int i = 0; i < m; i ++) {
		scc_sets[scc_root[i]].insert(i);
		if(scc_root[i] == i) total_num_scc ++;
	}
	for (int i = 0; i < m; i ++) {
		if (scc_sets[i].size() == 1) num_trivial_scc ++;
		else if (scc_sets[i].size() > 1) num_nontrivial_scc ++;
		if (scc_sets[i].size() > biggest_scc_size) biggest_scc_size = scc_sets[i].size();
	}
	printf("\tnum_trimmed=%d, num_pivots=%d, total_num_scc=%d\n", total_num_trimmed, total_num_pivots, total_num_trimmed+total_num_pivots);
	printf("\tnum_trivial=%d, num_nontrivial=%d, total_num_scc=%d, biggest_scc_size=%d\n", num_trivial_scc, num_nontrivial_scc, total_num_scc, biggest_scc_size);
}

