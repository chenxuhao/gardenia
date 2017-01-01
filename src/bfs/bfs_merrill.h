// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu> and Pingfan Li <lipingfan@163.com>
#include <cub/cub.cuh>
#include "gbar.h"
#include "cuda_launch_config.hpp"
#define BFS_VARIANT "merrill"
#include "cutil_subset.h"
#include <b40c_test_util.h>
#include <b40c/graph/builder/dimacs.cuh>
#include <b40c/graph/bfs/csr_problem.cuh>
#include <b40c/graph/csr_graph.cuh>
#include <b40c/graph/bfs/enactor_hybrid.cuh>
#include <b40c/graph/bfs/enactor_two_phase.cuh>
typedef unsigned DistT;
using namespace b40c;
using namespace graph;

void BFSSolver(int m, int nnz, int *csrRowPtr, int *csrColInd, DistT *dist) {
	printf("BFS data-driven load-balance version\n");
	typedef int VertexId;
	typedef unsigned Value;
	typedef int SizeT;
	int *d_csrRowPtr, *d_csrColInd;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_csrRowPtr, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_csrColInd, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_csrRowPtr, csrRowPtr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_csrColInd, csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice));

	graph::CsrGraph<VertexId, Value, SizeT> csr_graph;
	csr_graph.FromScratch<true>(m, nnz);
	CUDA_SAFE_CALL(cudaMemcpy(csr_graph.row_offsets, d_csrRowPtr, sizeof(SizeT) * (m + 1), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(csr_graph.column_indices, d_csrColInd, sizeof(VertexId) * nnz, cudaMemcpyDeviceToHost));

	typedef bfs::CsrProblem<VertexId, SizeT, false> CsrProblem;
	//bfs::EnactorTwoPhase<false> two_phase(false);
	bfs::EnactorHybrid<false> hybrid(false);
	CsrProblem csr_problem;
	if (csr_problem.FromHostProblem(false, csr_graph.nodes, csr_graph.edges, csr_graph.column_indices, csr_graph.row_offsets, 1)) exit(1);
	cudaError_t	retval = cudaSuccess;
	VertexId src = 0;
	double starttime, endtime;
	starttime = rtclock();
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
	endtime = rtclock();
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, 1000 * (endtime - starttime));

	DistT *h_dist;
	h_dist = (DistT *) calloc(m, sizeof(DistT));
	assert(h_dist != NULL);
	if (csr_problem.ExtractResults((int *) h_dist)) exit(1);
	for(int i = 0; i < m; i++)
		if((signed) h_dist[i] == -1)
			h_dist[i] = MYINFINITY;
	CUDA_SAFE_CALL(cudaMemcpy(dist, h_dist, m * sizeof(DistT), cudaMemcpyHostToDevice));
	free(h_dist);
	CUDA_SAFE_CALL(cudaFree(d_csrRowPtr));
	CUDA_SAFE_CALL(cudaFree(d_csrColInd));
}
