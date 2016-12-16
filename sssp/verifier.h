void SSSPVerifier(int m , unsigned *dist, int *row_offsets, int *column_indices, W_TYPE *weight) {
	printf("Verifying...\n");
	unsigned nerr = 0;
	for (int u = 0; u < m; u ++) {
		int row_begin = row_offsets[u];
		int row_end = row_offsets[u + 1]; 
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int v = column_indices[offset];
			W_TYPE wt = weight? weight[offset]:1;
			if (wt > 0 && dist[u] + wt < dist[v]) {
				//printf("%d %d %d %d\n", nn, v, dist[nn], dist[v]);
				++ nerr;
			}
		}
	}   
	printf("\tNumber of errors = %d.\n", nerr);
}

__global__ void dverify(int m, unsigned *dist, int *row_offsets, int *column_indices, W_TYPE *weight, unsigned *nerr) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	if (u < m) {
		int row_begin = row_offsets[u];
		int row_end = row_offsets[u + 1]; 
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int v = column_indices[offset];
			W_TYPE wt = weight[offset];
			if (wt > 0 && dist[u] + wt < dist[v]) {
				//printf("%d %d %d %d\n", u, v, dist[u], dist[v]);
				++*nerr;
			}
		}
	}
}

void write_solution(const char *fname, int m, unsigned *h_dist) {
	//unsigned *h_dist;
	//h_dist = (unsigned *) malloc(m * sizeof(unsigned));
	assert(h_dist != NULL);
	//CUDA_SAFE_CALL(cudaMemcpy(h_dist, dist, m * sizeof(unsigned), cudaMemcpyDeviceToHost));
	printf("Writing solution to %s\n", fname);
	FILE *f = fopen(fname, "w");
	fprintf(f, "Computed solution (source dist): [");
	for(int node = 0; node < m; node++) {
		fprintf(f, "%d:%d\n ", node, h_dist[node]);
	}
	fprintf(f, "]");
}

