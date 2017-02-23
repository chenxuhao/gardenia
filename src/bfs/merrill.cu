// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu> and Pingfan Li <lipingfan@163.com>
#define BFS_VARIANT "merrill"
#include <cub/cub.cuh>
#include "bfs.h"
#include "gbar.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"
#include <b40c_test_util.h>
#include <b40c/graph/builder/dimacs.cuh>
#include <b40c/graph/bfs/csr_problem.cuh>
#include <b40c/graph/csr_graph.cuh>
#include <b40c/graph/bfs/enactor_hybrid.cuh>
#include <b40c/graph/bfs/enactor_two_phase.cuh>
using namespace b40c;
using namespace graph;

void BFSSolver(int m, int nnz, int *in_row_offsets, int *in_column_indices, int *h_row_offsets, int *h_column_indices, int *h_degree, DistT *h_dist) {
	printf("BFS data-driven Merrill's version\n");
	typedef int VertexId;
	typedef unsigned Value;
	typedef int SizeT;
	int *d_row_offsets, *d_column_indices;
	DistT *d_dist;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_dist, m * sizeof(DistT)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_dist, h_dist, m * sizeof(DistT), cudaMemcpyHostToDevice));

	graph::CsrGraph<VertexId, Value, SizeT> csr_graph;
	csr_graph.FromScratch<true>(m, nnz);
	CUDA_SAFE_CALL(cudaMemcpy(csr_graph.row_offsets, d_row_offsets, sizeof(SizeT) * (m + 1), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(csr_graph.column_indices, d_column_indices, sizeof(VertexId) * nnz, cudaMemcpyDeviceToHost));

	typedef bfs::CsrProblem<VertexId, SizeT, false> CsrProblem;
	//bfs::EnactorTwoPhase<false> two_phase(false);
	bfs::EnactorHybrid<false> hybrid(false);
	CsrProblem csr_problem;
	if (csr_problem.FromHostProblem(false, csr_graph.nodes, csr_graph.edges, csr_graph.column_indices, csr_graph.row_offsets, 1)) exit(1);
	cudaError_t	retval = cudaSuccess;
	Timer t;
	t.Start();
	if (retval = csr_problem.Reset(hybrid.GetFrontierType(), 1.3))
	//if (retval = csr_problem.Reset(two_phase.GetFrontierType(), 1.3))
		return;
	if (retval = hybrid.EnactSearch(csr_problem, 0)) {
	//if (retval = two_phase.EnactIterativeSearch(csr_problem, 0)) {
		if (retval && (retval != cudaErrorInvalidDeviceFunction)) {
			exit(1);
		}
	}
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(d_dist, h_dist, m * sizeof(DistT), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
	CUDA_SAFE_CALL(cudaFree(d_dist));
}
