#include <stdio.h>
#include <stdlib.h>
#include <nvgraph.h>

void check(nvgraphStatus_t status) {
	if (status != NVGRAPH_STATUS_SUCCESS) {
		printf("ERROR : %d\n",status);
		exit(0);
	}
}

int main(int argc, char **argv) {
	size_t n = 6, nnz = 10, vert_sets = 2, edge_sets = 1;
	float alpha1 = 0.9f; void *alpha1_p = (void *) &alpha1;
	// nvgraph variables
	nvgraphHandle_t handle; nvgraphGraphDescr_t graph;
	nvgraphCSCTopology32I_t CSC_input;
	cudaDataType_t edge_dimT = CUDA_R_32F;
	cudaDataType_t* vertex_dimT;
	// Allocate host data
	float *pr_1 = (float*)malloc(n*sizeof(float));
	void **vertex_dim = (void**)malloc(vert_sets*sizeof(void*));
	vertex_dimT = (cudaDataType_t*)malloc(vert_sets*sizeof(cudaDataType_t));
	CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct
				nvgraphCSCTopology32I_st));
	// Initialize host data
	float weights_h[] = {0.333333f, 0.5f, 0.333333f, 0.5f, 0.5f, 1.0f,
	0.333333f, 0.5f, 0.5f, 0.5f};
	int destination_offsets_h[] = {0, 1, 3, 4, 6, 8, 10};
	int source_indices_h[] = {2, 0, 2, 0, 4, 5, 2, 3, 3, 4};
	float bookmark_h[] = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	vertex_dim[0] = (void*)bookmark_h; vertex_dim[1]= (void*)pr_1;
	vertex_dimT[0] = CUDA_R_32F; vertex_dimT[1]= CUDA_R_32F, vertex_dimT[2]=
		CUDA_R_32F;
	// Starting nvgraph
	check(nvgraphCreate (&handle));
	check(nvgraphCreateGraphDescr (handle, &graph));
	CSC_input->nvertices = n; CSC_input->nedges = nnz;
	CSC_input->destination_offsets = destination_offsets_h;
	CSC_input->source_indices = source_indices_h;
	// Set graph connectivity and properties (tranfers)
	check(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input,
				NVGRAPH_CSC_32));
	check(nvgraphAllocateVertexData(handle, graph, vert_sets, vertex_dimT));
	check(nvgraphAllocateEdgeData (handle, graph, edge_sets, &edge_dimT));
	for (int i = 0; i < 2; ++i)
		check(nvgraphSetVertexData(handle, graph, vertex_dim[i], i));
	check(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));
	check(nvgraphPagerank(handle, graph, 0, alpha1_p, 0, 0, 1, 0.0f, 0));
	// Get result
	check(nvgraphGetVertexData(handle, graph, vertex_dim[1], 1));
	check(nvgraphDestroyGraphDescr(handle, graph));
	check(nvgraphDestroy(handle));
	free(pr_1); free(vertex_dim); free(vertex_dimT);
	free(CSC_input);
	return 0;
}
