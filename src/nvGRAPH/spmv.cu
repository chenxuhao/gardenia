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
	size_t n = 5, nnz = 10, vertex_numsets = 2, edge_numsets = 1;
	float alpha = 1.0, beta = 0.0;
	void *alpha_p = (void *)&alpha, *beta_p = (void *)&beta;
	void** vertex_dim;
	cudaDataType_t edge_dimT = CUDA_R_32F;
	cudaDataType_t* vertex_dimT;
	// nvgraph variables
	nvgraphStatus_t status; nvgraphHandle_t handle;
	nvgraphGraphDescr_t graph;
	nvgraphCSRTopology32I_t CSR_input;
	// Init host data
	vertex_dim = (void**)malloc(vertex_numsets*sizeof(void*));
	vertex_dimT =
		(cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
	CSR_input = (nvgraphCSRTopology32I_t) malloc(sizeof(struct
				nvgraphCSRTopology32I_st));
	float x_h[] = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
	float y_h[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	vertex_dim[0]= (void*)x_h; vertex_dim[1]= (void*)y_h;
	vertex_dimT[0] = CUDA_R_32F; vertex_dimT[1]= CUDA_R_32F;
	float weights_h[] = {1.0f, 4.0f, 2.0f, 3.0f, 5.0f, 7.0f, 8.0f, 9.0f, 6.0f, 1.5f};
	int source_offsets_h[] = {0, 2, 4, 7, 9, 10};
	int destination_indices_h[] = {0, 1, 1, 2, 0, 3, 4, 2, 4, 2};
	check(nvgraphCreate(&handle));
	check(nvgraphCreateGraphDescr(handle, &graph));
	CSR_input->nvertices = n; CSR_input->nedges = nnz;
	CSR_input->source_offsets = source_offsets_h;
	CSR_input->destination_indices = destination_indices_h;
	// Set graph connectivity and properties (tranfers)
	check(nvgraphSetGraphStructure(handle, graph, (void*)CSR_input,
				NVGRAPH_CSR_32));
	check(nvgraphAllocateVertexData(handle, graph, vertex_numsets,
				vertex_dimT));
	for (int i = 0; i < vertex_numsets; ++i)
		check(nvgraphSetVertexData(handle, graph, vertex_dim[i], i));
	check(nvgraphAllocateEdgeData (handle, graph, edge_numsets, &edge_dimT));
	check(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));
	// Solve
	check(nvgraphSrSpmv(handle, graph, 0, alpha_p, 0, beta_p, 1,
				NVGRAPH_PLUS_TIMES_SR));
	//Get result
	check(nvgraphGetVertexData(handle, graph, (void*)y_h, 1));
	//Clean
	check(nvgraphDestroyGraphDescr(handle, graph));
	check(nvgraphDestroy(handle));
	free(vertex_dim); free(vertex_dimT); free(CSR_input);
	return 0;
}
