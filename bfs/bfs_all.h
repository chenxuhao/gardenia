#define BFS_VARIANT "worklistc"
#include "worklistc.h"
#include "gbar.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include <cub/cub.cuh>
#define BLKSIZE 128
const int IN_CORE = 0;
//texture <int, 1, cudaReadModeElementType> columns_indices;
//texture <int, 1, cudaReadModeElementType> row_offsets;

__global__ void initialize(foru *dist, unsigned int m) {
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		dist[id] = MYINFINITY;
	}
}

typedef cub::BlockScan<int, BLKSIZE> BlockScan;
__device__ void expandByCta(int m, int *csrRowPtr, int *csrColInd, foru *dist, Worklist2 &inwl, Worklist2 &outwl, unsigned iteration) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	int vertex;
	__shared__ int owner;
	__shared__ int sh_vertex;
	//int total_inputs = (*inwl.dindex + gridDim.x * blockDim.x - 1)/(gridDim.x * blockDim.x);
	owner = -1;
	int size = 0;
	if(inwl.pop_id(id, vertex)) {
		//int row_begin = tex1Dfetch(row_offsets, vertex);
		//int row_end = tex1Dfetch(row_offsets, vertex + 1);
		size = csrRowPtr[vertex + 1] - csrRowPtr[vertex];
	}
	while(true) {
		if(size > BLKSIZE)
			owner = threadIdx.x;
		__syncthreads();
		if(owner == -1)
			break;
		__syncthreads();
		if(owner == threadIdx.x) {
			sh_vertex = vertex;
			inwl.dwl[id] = -1;
			owner = -1;
			size = 0;
		}
		__syncthreads();
		//= tex1Dfetch(row_offsets, sh_vertex);
		int row_begin = csrRowPtr[sh_vertex];
		//= tex1Dfetch(row_offsets, sh_vertex + 1) - neighboroffset;
		int row_end = csrRowPtr[sh_vertex + 1];
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + blockDim.x - 1) / blockDim.x) * blockDim.x;
		for(int i = threadIdx.x; i < num; i += blockDim.x) {
			int ncnt = 0;
			int dst = 0;
			int edge = row_begin + i;
			if(i < neighbor_size) {
				dst = csrColInd[edge];
				assert(dst < m);
				if(dist[dst] == MYINFINITY) {
					dist[dst] = iteration;
					ncnt = 1;
				}
			}
			outwl.push_1item<BlockScan>(ncnt, dst, BLKSIZE);
		}
	}
}

__device__ __forceinline__ unsigned LaneId() {
	unsigned ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

#define WARP_SIZE 32
#define LOG_WARP_SIZE 5
#define NUM_WARPS (BLKSIZE / WARP_SIZE)
__device__ __forceinline__ void expandByWarp(int m, int *row_offsets, int *column_indices, foru *dist, Worklist2 &inwl, Worklist2 &outwl, unsigned iteration) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned warp_id = threadIdx.x >> LOG_WARP_SIZE;
	unsigned lane_id = LaneId();
	__shared__ int owner[NUM_WARPS];
	__shared__ int sh_vertex[NUM_WARPS];
	owner[warp_id] = -1;
	int size = 0;
	int vertex;
	if(inwl.pop_id(id, vertex)) {
		if (vertex != -1)
			size = row_offsets[vertex + 1] - row_offsets[vertex];
	}
	while(__any(size) >= WARP_SIZE) {
		if(size >= WARP_SIZE)
			owner[warp_id] = lane_id;
		if(owner[warp_id] == lane_id) {
			sh_vertex[warp_id] = vertex;
			// mark this vertex as processed already
			inwl.dwl[id] = -1;
			owner[warp_id] = -1;
			size = 0;
		}
		int winner = sh_vertex[warp_id];
		//int winner_tid = warp_id * WARP_SIZE + owner[warp_id];
		int row_begin = row_offsets[winner];
		//row_begin = __ldg(row_offsets + winner);
		int row_end = row_offsets[winner + 1];
		//row_end = __ldg(row_offsets + winner + 1);
		int neighbor_size = row_end - row_begin;
		int num = ((neighbor_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
		for(int i = lane_id; i < num; i+= WARP_SIZE) {
			int ncnt = 0;
			int dst = 0;
			int edge = row_begin + i;
			if(i < neighbor_size) {
				dst = column_indices[edge];
				assert(dst < m);
				if(dist[dst] == MYINFINITY) {
					dist[dst] = iteration;
					ncnt = 1;
				}
			}
			outwl.push_1item<BlockScan>(ncnt, dst, BLKSIZE);
		}
	}
}

__device__ unsigned processnode2(int m, int *csrRowPtr, int *csrColInd, foru *dist, Worklist2 &inwl, Worklist2 &outwl, unsigned iteration) {
	expandByCta(m, csrRowPtr, csrColInd, dist, inwl, outwl, iteration);
	//expandByWarp(m, csrRowPtr, csrColInd, dist, inwl, outwl, iteration);
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	int vertex;
	const int SCRATCHSIZE = BLKSIZE;
	__shared__ BlockScan::TempStorage temp_storage;
	__shared__ int gather_offsets[SCRATCHSIZE];
	gather_offsets[threadIdx.x] = 0;
	//int total_inputs = (*inwl.dindex + gridDim.x * blockDim.x - 1)/(gridDim.x * blockDim.x);
	//while(total_inputs-- > 0) { 
		int neighborsize = 0;
		int neighboroffset = 0;
		int scratch_offset = 0;
		int total_edges = 0;
		if(inwl.pop_id(id, vertex)) {	  
			if(vertex != -1) {
				//neighboroffset = tex1Dfetch(row_offsets, vertex);
				neighboroffset = csrRowPtr[vertex];
				//neighborsize = tex1Dfetch(row_offsets, vertex + 1) - neighboroffset;
				neighborsize = csrRowPtr[vertex + 1] - neighboroffset;
			}
		}
		BlockScan(temp_storage).ExclusiveSum(neighborsize, scratch_offset, total_edges);
		int done = 0;
		int neighborsdone = 0;
		while(total_edges > 0) {
			__syncthreads();
			int i;
			for(i = 0; neighborsdone + i < neighborsize && (scratch_offset + i - done) < SCRATCHSIZE; i++) {
				gather_offsets[scratch_offset + i - done] = neighboroffset + neighborsdone + i;
			}
			neighborsdone += i;
			scratch_offset += i;
			__syncthreads();
			int ncnt = 0;
			int dst = 0;
			int edge = gather_offsets[threadIdx.x];
			if(threadIdx.x < total_edges) {
				dst = csrColInd[edge];
				assert(dst < m);
				if(dist[dst] == MYINFINITY) {
					dist[dst] = iteration;
					ncnt = 1;
				}
			}
			outwl.push_1item<BlockScan>(ncnt, dst, BLKSIZE);
			total_edges -= BLKSIZE;
			done += BLKSIZE;
		}
		//id += blockDim.x * gridDim.x;
	//}
	return 0;
}
/*
__device__ unsigned processnode(int m, int *csrRowPtr, int *csrColInd, foru *dist, Worklist2 &inwl, Worklist2 &outwl) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	int neighbours[256];
	int ncnt = 0;
	int vertex;
	if(inwl.pop_id(id, vertex)) {
		unsigned row_begin = csrRowPtr[vertex];
		unsigned neighbor_size = csrRowPtr[vertex + 1] - row_begin;
		if(neighbor_size > 256)
			printf("whoa! out of local space");
		for (unsigned i = 0; i < neighbor_size; ++ i) {
			unsigned dst = m;
			foru olddist = processedge(m, csrRowPtr, csrColInd, dist, vertex, i, dst);
			if (olddist) {
				neighbours[ncnt] = dst;
				ncnt++;
			}
		}
	}
	return outwl.push_nitems<BlockScan>(ncnt, neighbours, BLKSIZE) == 0 && ncnt > 0;
}
*/
__device__ void drelax(int m, int *csrRowPtr, int *csrColInd, foru *dist, unsigned *gerrno, Worklist2 &inwl, Worklist2& outwl, int iteration) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if(iteration == 0) {
		if(id == 0) {
			int item = 0;
			inwl.push(item);
		}
		return;
	}
	else {
		if(processnode2(m, csrRowPtr, csrColInd, dist, inwl, outwl, iteration))
			*gerrno = 1;
	}
}

__global__ void drelax3(int m, int *csrRowPtr, int *csrColInd, foru *dist, unsigned *gerrno, Worklist2 inwl, Worklist2 outwl, int iteration) {
	drelax(m, csrRowPtr, csrColInd, dist, gerrno, inwl, outwl, iteration);
}

__global__ void drelax2(int m, int *csrRowPtr, int *csrColInd, foru *dist, unsigned *gerrno, Worklist2 inwl, Worklist2 outwl, int iteration, GlobalBarrier gb) {
	if(iteration == 0)
		drelax(m, csrRowPtr, csrColInd, dist, gerrno, inwl, outwl, iteration);
	else {
		Worklist2 *in;
		Worklist2 *out;
		Worklist2 *tmp;
		in = &inwl; out = &outwl;
		while(*in->dindex > 0) {
			drelax(m, csrRowPtr, csrColInd, dist, gerrno, *in, *out, iteration);
			gb.Sync();
			tmp = in;
			in = out;
			out = tmp;
			*out->dindex = 0;
			iteration++;
		}
	}
}
/*
__global__ void print_array(int *a, int n) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if(id < n)
		printf("%d %d\n", id, a[id]);
}

__global__ void print_texture(int n) {
	int id = threadIdx.x + blockDim.x * blockIdx.x;
}
*/
void bfs(int m, int nnz, int *d_csrRowPtr, int *d_csrColInd, foru *d_dist, int nSM) {
	foru foruzero = 0;
	int iteration = 0;
	unsigned *nerr;
	double starttime, endtime, runtime;
	const int nthreads = 256;
	int nblocks = (m - 1) / nthreads + 1;
	initialize <<<nblocks, nthreads>>> (d_dist, m);
	CudaTest("initializing failed");
	CUDA_SAFE_CALL(cudaMemcpy(&d_dist[0], &foruzero, sizeof(foruzero), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc((void **)&nerr, sizeof(unsigned)));
	Worklist2 wl1(nnz * 2), wl2(nnz * 2);
	Worklist2 *inwl = &wl1, *outwl = &wl2;
	int nitems = 1;
	//CUDA_SAFE_CALL(cudaBindTexture(0, columns, csrColInd, (nnz + 1) * sizeof(int)));
	//CUDA_SAFE_CALL(cudaBindTexture(0, row_offsets, csrRowPtr, (m + 1) * sizeof(int)));
	const size_t max_blocks = maximum_residency(drelax2, BLKSIZE, 0);
	printf("Solving, nSM=%d, max_blocks=%d, nthreads=%d\n", nSM, max_blocks, nthreads);
	starttime = rtclock();
	if(IN_CORE) {
		GlobalBarrierLifetime gb;
		gb.Setup(nSM * max_blocks);
		//printf("starting kernel...\n");
		drelax2<<<1, BLKSIZE>>>(m, d_csrRowPtr, d_csrColInd, d_dist, nerr, *inwl, *outwl, 0, gb);
		drelax2 <<<nSM * max_blocks, BLKSIZE>>> (m, d_csrRowPtr, d_csrColInd, d_dist, nerr, *inwl, *outwl, 1, gb);
	}
	else {
		drelax3<<<1, BLKSIZE>>>(m, d_csrRowPtr, d_csrColInd, d_dist, nerr, *inwl, *outwl, 0);
		nitems = inwl->nitems();

		while(nitems > 0) {
			++iteration;
			unsigned nblocks = (nitems + BLKSIZE - 1) / BLKSIZE; 
			printf("iteration=%d, nblocks=%d, nthreads=%d, wlsz=%d\n", iteration, nblocks, BLKSIZE, nitems);
			//inwl->display_items();
			drelax3<<<nblocks, BLKSIZE>>>(m, d_csrRowPtr, d_csrColInd, d_dist, nerr, *inwl, *outwl, iteration);
			nitems = outwl->nitems();
			Worklist2 *tmp = inwl;
			inwl = outwl;
			outwl = tmp;
			outwl->reset();
		};
	}
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	endtime = rtclock();
	printf("\titerations = %d.\n", iteration);
	runtime = (1000.0f * (endtime - starttime));
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, runtime);
	CUDA_SAFE_CALL(cudaFree(nerr));
	return;
}
