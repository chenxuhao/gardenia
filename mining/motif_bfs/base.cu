// Copyright (c) 2019, Xuhao Chen
#include "motif.h"
#include "timer.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <cub/cub.cuh>
#define USE_PID
//#define USE_WEDGE
#define USE_SIMPLE
#define VERTEX_INDUCED
#include "gpu_mining/miner.cuh"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#define MOTIF_VARIANT "topo"
typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;

void printout_motifs(int npatterns, AccType *accumulators) {
	std::cout << std::endl;
	if (npatterns == 2) {
		std::cout << "\ttriangles\t" << accumulators[0] << std::endl;
		std::cout << "\t3-chains\t" << accumulators[1] << std::endl;
	} else if (npatterns == 6) {
		std::cout << "\t4-paths --> " << accumulators[0] << std::endl;
		std::cout << "\t3-stars --> " << accumulators[1] << std::endl;
		std::cout << "\t4-cycles --> " << accumulators[2] << std::endl;
		std::cout << "\ttailed-triangles --> " << accumulators[3] << std::endl;
		std::cout << "\tdiamonds --> " << accumulators[4] << std::endl;
		std::cout << "\t4-cliques --> " << accumulators[5] << std::endl;
	} else {
		std::cout << "\ttoo many patterns to show\n";
	}
	std::cout << std::endl;
}

__global__ void extend_alloc(unsigned m, unsigned level, GraphGPU graph, EmbeddingList emb_list, IndexT *num_new_emb) {
	unsigned tid = threadIdx.x;
	unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ IndexT emb[BLOCK_SIZE][MAX_SIZE];
	if(pos < m) {
		IndexT num = 0;
		emb_list.get_embedding(level, pos, emb[tid]);
		for (unsigned i = 0; i < level+1; ++i) {
			IndexT src = emb[tid][i];
			IndexT row_begin = graph.edge_begin(src);
			IndexT row_end = graph.edge_end(src);
			for (IndexT e = row_begin; e < row_end; e++) {
				IndexT dst = graph.getEdgeDst(e);
				if (!is_vertexInduced_automorphism(level+1, emb[tid], i, src, dst, graph))
					num ++;
			}
		}
		num_new_emb[pos] = num;
	}
}

__global__ void extend_insert(unsigned m, unsigned max_size, unsigned level, GraphGPU graph, EmbeddingList emb_list, IndexT *indices) {
	unsigned tid = threadIdx.x;
	unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ IndexT emb[BLOCK_SIZE][MAX_SIZE];
	if(pos < m) {
		emb_list.get_embedding(level, pos, emb[tid]);
		IndexT start = indices[pos];
		for (unsigned i = 0; i < level+1; ++i) {
			IndexT src = emb[tid][i];
			IndexT row_begin = graph.edge_begin(src);
			IndexT row_end = graph.edge_end(src);
			for (IndexT e = row_begin; e < row_end; e++) {
				IndexT dst = graph.getEdgeDst(e);
				if (!is_vertexInduced_automorphism(level+1, emb[tid], i, src, dst, graph)) {
					if (level == 1 && max_size == 4)
						emb_list.set_pid(start, find_3motif_pattern_id(i, dst, emb[tid], graph, start));
					emb_list.set_idx(level+1, start, pos);
					emb_list.set_vid(level+1, start++, dst);
				}
			}
		}
	}
}

__global__ void aggregate(unsigned m, unsigned level, unsigned npatterns, GraphGPU graph, EmbeddingList emb_list, AccType *accumulators) {
	unsigned tid = threadIdx.x;
	unsigned pos = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	__shared__ IndexT emb[BLOCK_SIZE][MAX_SIZE];
	AccType local_num[6];
	for (int i = 0; i < npatterns; i++) local_num[i] = 0;
	if(pos < m) {
		unsigned pattern = 0;
		emb_list.get_embedding(level, pos, emb[tid]);
		//if (pos == 0) printout_embedding(level, emb[tid]);
		unsigned n = level+1;
		assert(n < 4);
		if (n == 3) pattern = emb_list.get_pid(pos);
		for (unsigned i = 0; i < n; ++i) {
			IndexT src = emb[tid][i];
			IndexT row_begin = graph.edge_begin(src);
			IndexT row_end = graph.edge_end(src);
			for (IndexT e = row_begin; e < row_end; e++) {
				IndexT dst = graph.getEdgeDst(e);
				if (!is_vertexInduced_automorphism(n, emb[tid], i, src, dst, graph)) {
					unsigned pid = 1; // 3-chain
					//if (i == 0 && is_connected(emb[tid][1], dst, graph)) pid = 0; // triangle
					if (n == 2) pid = find_3motif_pattern_id(i, dst, emb[tid], graph, pos);
					else pid = find_4motif_pattern_id(n, i, dst, emb[tid], pattern, graph, pos);
					//printf("pid = %u\n", pid);
					local_num[pid] += 1;
				}
			}
		}
	}
	AccType block_num;
	for (int i = 0; i < npatterns; i++) {
		//block_num = BlockReduce(temp_storage).Sum(local_num[i]);
		//if(threadIdx.x == 0) atomicAdd(&accumulators[i], block_num);
		atomicAdd(&accumulators[i], local_num[i]);
	}
}

__global__ void clear(AccType *accumulators) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	accumulators[i] = 0;
}

void parallel_prefix_sum(int n, IndexT *in, IndexT *out) {
	IndexT total = 0;
	for (size_t i = 0; i < n; i++) {
		out[i] = total;
		total += in[i];
	}
	out[n] = total;
}

void MotifSolver(Graph &g, unsigned k, std::vector<AccType> &acc) {
	print_device_info(0);
	size_t npatterns = acc.size();
	AccType *h_accumulators = (AccType *)malloc(sizeof(AccType) * npatterns);
	for (int i = 0; i < npatterns; i++) h_accumulators[i] = 0;
	AccType *d_accumulators;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_accumulators, sizeof(AccType) * npatterns));
	clear<<<1, npatterns>>>(d_accumulators);
	CudaTest("clear accumulator failed");
	int m = g.num_vertices();
	int nnz = g.num_edges();
	int nthreads = BLOCK_SIZE;
	int nblocks = DIVIDE_INTO(m, nthreads);
	printf("Launching CUDA TC solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);
	CUDA_Context_Mining cuda_ctx;
	cuda_ctx.hg = &g;
	cuda_ctx.build_graph_gpu();
	cuda_ctx.emb_list.init(nnz, k, false);
/*
	IndexT *d_num_new_emb, *d_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_num_new_emb, sizeof(IndexT) * (m+1)));
	init_alloc<<<nblocks, nthreads>>>(m, cuda_ctx.gg, cuda_ctx.emb_list, d_num_new_emb);
	CudaTest("Init_alloc failed");
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_indices, sizeof(IndexT) * (m+1)));
	thrust::exclusive_scan(thrust::device, d_num_new_emb, d_num_new_emb+m+1, d_indices);
	CudaTest("Scan failed");
	IndexT num_single_edge_embeddings = 0;
	CUDA_SAFE_CALL(cudaMemcpy(&num_single_edge_embeddings, &d_indices[m], sizeof(IndexT), cudaMemcpyDeviceToHost));
	std::cout << "Number of single-edge embeddings: " << num_single_edge_embeddings << "\n";
	init_insert<<<nblocks, nthreads>>>(m, cuda_ctx.gg, cuda_ctx.emb_list, d_indices);
	CudaTest("Init_insert failed");
	CUDA_SAFE_CALL(cudaFree(d_num_new_emb));
	CUDA_SAFE_CALL(cudaFree(d_indices));
*/
	cuda_ctx.emb_list.init_cpu(cuda_ctx.hg);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	Timer t;
	t.Start();
	unsigned level = 1;
	unsigned num_emb = cuda_ctx.emb_list.size();
	while (level < k-2) {
		//std::cout << "number of embeddings in level " << level << ": " << num_emb << "\n";
		IndexT *num_new_emb, *indices;
		CUDA_SAFE_CALL(cudaMalloc((void **)&num_new_emb, sizeof(IndexT) * (num_emb+1)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&indices, sizeof(IndexT) * (num_emb+1)));
		//CUDA_SAFE_CALL(cudaMemset(num_new_emb, 0, sizeof(IndexT) * (num_emb+1)));
		//nthreads = 1;
		nblocks = (num_emb-1)/nthreads+1;
		extend_alloc<<<nblocks, nthreads>>>(num_emb, level, cuda_ctx.gg, cuda_ctx.emb_list, num_new_emb);
		CudaTest("solving extend_alloc failed");
		//std::cout << "Extend_alloc Done\n";
		thrust::exclusive_scan(thrust::device, num_new_emb, num_new_emb+num_emb+1, indices);
		/*
		IndexT *h_num_new_emb, *h_indices;
		h_num_new_emb = (IndexT *)malloc(sizeof(IndexT) * (num_emb+1));
		h_indices = (IndexT *)malloc(sizeof(IndexT) * (num_emb+1));
		CUDA_SAFE_CALL(cudaMemcpy(h_num_new_emb, num_new_emb, sizeof(IndexT)*(num_emb+1), cudaMemcpyDeviceToHost));
		parallel_prefix_sum(num_emb, h_num_new_emb, h_indices);
		CUDA_SAFE_CALL(cudaMemcpy(indices, h_indices, sizeof(IndexT)*(num_emb+1), cudaMemcpyHostToDevice));
		*/
		CudaTest("Scan failed");
		//std::cout << "PrefixSum Done\n";
		IndexT new_size;
		CUDA_SAFE_CALL(cudaMemcpy(&new_size, &indices[num_emb], sizeof(IndexT), cudaMemcpyDeviceToHost));
		assert(new_size < 4294967296); // TODO: currently do not support vector size larger than 2^32
		//std::cout << "number of new embeddings: " << new_size << "\n";
		cuda_ctx.emb_list.add_level(new_size);
		#ifdef USE_WEDGE
		//if (level == 1 && max_size == 4) {
		//	is_wedge.resize(emb_list.size());
		//	std::fill(is_wedge.begin(), is_wedge.end(), 0);
		//}
		#endif
		extend_insert<<<nblocks, nthreads>>>(num_emb, k, level, cuda_ctx.gg, cuda_ctx.emb_list, indices);
		CudaTest("solving extend_insert failed");
		//CUDA_SAFE_CALL(cudaDeviceSynchronize());
		std::cout << "Extend_insert Done\n";
		num_emb = cuda_ctx.emb_list.size();
		CUDA_SAFE_CALL(cudaFree(num_new_emb));
		CUDA_SAFE_CALL(cudaFree(indices));
		level ++;
	}
	if (k < 5) {
		//nthreads = 2;
		nblocks = (num_emb-1)/nthreads+1;
		//std::cout << "number of agg embeddings: " << num_emb << ", nthreads = " << nthreads << ", nblocks = " << nblocks << "\n";
		aggregate<<<nblocks, nthreads>>>(num_emb, level, npatterns, cuda_ctx.gg, cuda_ctx.emb_list, d_accumulators);
		CudaTest("solving aggregate failed");
		//std::cout << "Aggregate Done\n";
	} else {
		printf("Not supported\n");
	}
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\truntime [%s] = %f ms.\n", MOTIF_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_accumulators, d_accumulators, sizeof(AccType) * npatterns, cudaMemcpyDeviceToHost));
	printout_motifs(npatterns, h_accumulators);
	CUDA_SAFE_CALL(cudaFree(d_accumulators));
}

