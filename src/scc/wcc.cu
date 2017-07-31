#include "wcc.h"
#define debug_wcc 0
static __global__ void wcc_min(int m, int *row_offsets, int *column_indices, unsigned *colors, unsigned char *status, unsigned *wcc, bool *changed) {
	int src = blockDim.x * blockIdx.x + threadIdx.x;
	if (src < m && !is_removed(status[src])) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1];
		unsigned wcc_src = wcc[src];
		for(int offset = row_begin; offset < row_end; offset ++) {
			int dst = column_indices[offset];
			if(!is_removed(status[dst]) && colors[src] == colors[dst]) {
				if (wcc[dst] < wcc_src) {
					wcc_src = wcc[dst];
					*changed = true;
				}
			}
		}
		wcc[src] = wcc_src;
	}
}

static __global__ void wcc_update(int m, unsigned char *status, unsigned *wcc, bool *changed) {
	int src = blockDim.x * blockIdx.x + threadIdx.x;
	if (src < m && !is_removed(status[src])) {
		unsigned wcc_src = wcc[src];
		unsigned wcc_k = wcc[wcc_src];
		if (wcc_src != src && wcc_src != wcc_k) {
			wcc[src] = wcc_k;
			*changed = true;
		}
	}
}

static __global__ void update_pivot_color(int m, unsigned *wcc, unsigned *colors, unsigned char *status, bool *has_pivot, int *scc_root, unsigned *min_color) {
	int src = blockDim.x * blockIdx.x + threadIdx.x;
	if (src < m && !is_removed(status[src])) {
		if (wcc[src] == src) {
			unsigned new_color = atomicAdd(min_color, 1); 
			//printf("wcc: select vertex %d as pivot, old_color=%u, new_color=%u\n", src, colors[src], new_color);
			colors[src] = new_color;
			status[src] = 19; // set as a pivot
			scc_root[src] = src;
			*has_pivot = true;
		}
	}
}

static __global__ void update_colors(int m, unsigned *wcc, unsigned *colors, unsigned char *status) {
	int src = blockIdx.x * blockDim.x + threadIdx.x;
	if (src < m && !is_removed(status[src])) {
		unsigned wcc_src = wcc[src];
		if (wcc_src != src)
			colors[src] = colors[wcc_src];
	}
}

bool find_wcc(int m, int *d_row_offsets, int *d_column_indices, unsigned *d_colors, unsigned char *d_status, int *d_scc_root, unsigned min_color) {
	bool h_changed, *d_changed;
	int iter = 0;
	unsigned *d_wcc, *d_min_color;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_wcc, sizeof(unsigned) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_changed, sizeof(bool)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_min_color, sizeof(unsigned)));
	CUDA_SAFE_CALL(cudaMemcpy(d_min_color, &min_color, sizeof(unsigned), cudaMemcpyHostToDevice));
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	bool has_pivot = false;
	bool *d_has_pivot;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_has_pivot, sizeof(bool)));
	CUDA_SAFE_CALL(cudaMemcpy(d_has_pivot, &has_pivot, sizeof(bool), cudaMemcpyHostToDevice));
	thrust::sequence(thrust::device, d_wcc, d_wcc + m);
	do {
		++ iter;
		h_changed = false;
		CUDA_SAFE_CALL(cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice));
		wcc_min<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_colors, d_status, d_wcc, d_changed);
		CudaTest("solving kernel wcc_min failed");
		wcc_update<<<nblocks, nthreads>>>(m, d_status, d_wcc, d_changed);
		CudaTest("solving kernel wcc_update failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
	} while (h_changed);
	//CUDA_SAFE_CALL(cudaDeviceSynchronize());
	if(debug_wcc) {
		unsigned *h_wcc = (unsigned *)malloc(m*sizeof(unsigned));
		CUDA_SAFE_CALL(cudaMemcpy(h_wcc, d_wcc, m*sizeof(unsigned), cudaMemcpyDeviceToHost));
		FILE *fp=fopen("wcc.txt", "w");
		for(int i = 0; i < m; i ++) fprintf(fp, "wcc[%d]=%u\n", i, h_wcc[i]);
		fclose(fp);
	}
	update_pivot_color<<<nblocks, nthreads>>>(m, d_wcc, d_colors, d_status, d_has_pivot, d_scc_root, d_min_color);
	CudaTest("solving kernel update_pivot_color failed");
	update_colors<<<nblocks, nthreads>>>(m, d_wcc, d_colors, d_status);
	CudaTest("solving kernel update_colors failed");
	CUDA_SAFE_CALL(cudaMemcpy(&has_pivot, d_has_pivot, sizeof(bool), cudaMemcpyDeviceToHost));
	if(debug_wcc) {
		unsigned char *h_status = (unsigned char *)malloc(m*sizeof(unsigned char));
		unsigned *h_colors = (unsigned *)malloc(m*sizeof(unsigned));
		CUDA_SAFE_CALL(cudaMemcpy(h_status, d_status, m*sizeof(unsigned char), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(h_colors, d_colors, m*sizeof(unsigned), cudaMemcpyDeviceToHost));
		FILE *fp1=fopen("pivot.txt", "w");
		for(int i = 0; i < m; i ++)
			if(!is_removed(h_status[i]) && h_status[i]==19)
				fprintf(fp1, "%d\n", i);
		fclose(fp1);
	}
	//printf("wcc_iteration=%d\n", iter);
	CUDA_SAFE_CALL(cudaFree(d_changed));
	CUDA_SAFE_CALL(cudaFree(d_wcc));
	return has_pivot;
}

