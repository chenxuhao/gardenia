// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "pr.h"
#include <stdlib.h>
#include <nvgraph.h>
#include "timer.h"
#define PR_VARIANT "nvgraph"

void check(nvgraphStatus_t status) {
	if (status != NVGRAPH_STATUS_SUCCESS) {
		printf("ERROR : %d\n",status);
		exit(0);
	}
}

void PRSolver(int m, int nnz, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *degrees, ScoreT *scores) {
	size_t vert_sets = 2, edge_sets = 1;
	float alpha1 = kDamp;
	void *alpha1_p = (void *) &alpha1;

	// nvgraph variables
	nvgraphHandle_t handle;
	nvgraphGraphDescr_t graph;
	nvgraphCSCTopology32I_t CSC_input;
	cudaDataType_t edge_dimT = CUDA_R_32F;
	cudaDataType_t* vertex_dimT;

	// Allocate host data
	//float *pr_1 = (float*)malloc(m * sizeof(float));
	void **vertex_dim = (void**)malloc(vert_sets*sizeof(void*));
	vertex_dimT = (cudaDataType_t*)malloc(vert_sets*sizeof(cudaDataType_t));
	CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));

	float *weights_h = (float *)malloc(nnz * sizeof(float));
	for(int i = 0; i < nnz; i++) weights_h[i] = rand() / (RAND_MAX + 1.0);
	float *bookmark_h = (float *)malloc(m * sizeof(float));
	for(int i = 0; i < m; i++) {
		//if(degrees[i] == 0) bookmark_h[i] = 1.0;
		//else bookmark_h[i] = 0.0;
		bookmark_h[i] = 0.0;
	}
	vertex_dim[0] = (void*)bookmark_h;
	vertex_dim[1] = (void*)scores;
	vertex_dimT[0] = CUDA_R_32F;
	vertex_dimT[1] = CUDA_R_32F;
	vertex_dimT[2] = CUDA_R_32F;

	// Starting nvgraph
	check(nvgraphCreate (&handle));
	check(nvgraphCreateGraphDescr (handle, &graph));
	CSC_input->nvertices = m;
	CSC_input->nedges = nnz;
	CSC_input->destination_offsets = in_row_offsets;
	CSC_input->source_indices = in_column_indices;

	// Set graph connectivity and properties (tranfers)
	check(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
	check(nvgraphAllocateVertexData(handle, graph, vert_sets, vertex_dimT));
	check(nvgraphAllocateEdgeData (handle, graph, edge_sets, &edge_dimT));
	for (int i = 0; i < 2; ++i)
		check(nvgraphSetVertexData(handle, graph, vertex_dim[i], i));
	check(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));
	printf("Launching nvGRAPH PR solver ...\n");

	Timer t;
	t.Start();
	check(nvgraphPagerank(handle, graph, 0, alpha1_p, 0, 0, 1, EPSILON, MAX_ITER));
	t.Stop();
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());

	// Get result
	check(nvgraphGetVertexData(handle, graph, vertex_dim[1], 1));
	check(nvgraphDestroyGraphDescr(handle, graph));
	check(nvgraphDestroy(handle));

	//free(pr_1);
	free(vertex_dim);
	free(vertex_dimT);
	free(CSC_input);
	return;
}
