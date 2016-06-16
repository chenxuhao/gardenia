// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu> and Pingfan Li <lipingfan@163.com>

// transfer R-MAT generated gr graph to CSR format
void gr2csr(char *gr, int &m, int &nnz, int *&row_offsets, int *&column_indices, foru *&weight) {
	printf("Reading RMAT (.gr) input file %s\n", gr);
	std::ifstream cfile;
	cfile.open(gr);
	std::string str;
	getline(cfile, str);
	char c;
	sscanf(str.c_str(), "%c", &c);
	while (c == 'c') {
		getline(cfile, str);
		sscanf(str.c_str(), "%c", &c);
	}
	char sp[3];
	sscanf(str.c_str(), "%c %s %d %d", &c, sp, &m, &nnz);
	printf("num_vertices %d num_edges %d\n", m, nnz);
	vector<vector<int> > vertices;
	vector<int> neighbors;
	for (int i = 0; i < m; i++)
		vertices.push_back(neighbors);
	int dst, src;
	for (int i = 0; i < nnz; i++) {
		getline(cfile, str);
		sscanf(str.c_str(), "%c %d %d", &c, &src, &dst);
		if (c != 'a')
			printf("line %d\n", __LINE__);
		dst--;
		src--;
		vertices[src].push_back(dst);
		vertices[dst].push_back(src);
	}
	row_offsets = (int *)malloc((m + 1) * sizeof(int));
	int count = 0;
	for (int i = 0; i < m; i++) {
		row_offsets[i] = count;
		count += vertices[i].size();
	}
	row_offsets[m] = count;
	if (count != nnz) {
		printf("This graph is not symmetric\n");
		nnz = count;
	}
	double avgdeg;
	double variance = 0.0;
	int maxdeg = 0;
	int mindeg = m;
	avgdeg = (double)nnz / m;
	for (int i = 0; i < m; i++) {
		int deg_i = row_offsets[i + 1] - row_offsets[i];
		if (deg_i > maxdeg)
			maxdeg = deg_i;
		if (deg_i < mindeg)
			mindeg = deg_i;
		variance += (deg_i - avgdeg) * (deg_i - avgdeg) / m;
	}
	printf("mindeg %d maxdeg %d avgdeg %.2f variance %.2f\n", mindeg, maxdeg, avgdeg, variance);
	column_indices = (int *)malloc(count * sizeof(int));
	vector<int>::iterator neighbor_list;
	for (int i = 0, index = 0; i < m; i++) {
		neighbor_list = vertices[i].begin();
		while (neighbor_list != vertices[i].end()) {
			column_indices[index++] = *neighbor_list;
			neighbor_list++;
		}
	}
}

// transfer *.graph file to CSR format
void graph2csr(char *graph, int &m, int &nnz, int *&row_offsets, int *&column_indices, foru *&weight) {
	printf("Reading .graph input file %s\n", graph);
	std::ifstream cfile;
	cfile.open(graph);
	std::string str;
	getline(cfile, str);
	sscanf(str.c_str(), "%d %d", &m, &nnz);
	printf("num_vertices %d num_edges %d\n", m, nnz);
	vector<vector<int> > vertices;
	vector<int> neighbors;
	for (int i = 0; i < m; i++)
		vertices.push_back(neighbors);
	int dst;
	for (int i = 0; i < m; i++) {
		getline(cfile, str);
		istringstream istr;
		istr.str(str);
		while(istr>>dst) {
			dst --;
			vertices[i].push_back(dst);
			vertices[dst].push_back(i);
		}
		istr.clear();
	}
    cfile.close();
	row_offsets = (int *)malloc((m + 1) * sizeof(int));
	int count = 0;
	for (int i = 0; i < m; i++) {
		row_offsets[i] = count;
		count += vertices[i].size();
	}
	row_offsets[m] = count;
	if (count != nnz) {
		printf("This graph is not symmetric\n");
		nnz = count;
	}
	double avgdeg;
	double variance = 0.0;
	int maxdeg = 0;
	int mindeg = m;
	avgdeg = (double)nnz / m;
	for (int i = 0; i < m; i++) {
		int deg_i = row_offsets[i + 1] - row_offsets[i];
		if (deg_i > maxdeg)
			maxdeg = deg_i;
		if (deg_i < mindeg)
			mindeg = deg_i;
		variance += (deg_i - avgdeg) * (deg_i - avgdeg) / m;
	}
	printf("mindeg %d maxdeg %d avgdeg %.2f variance %.2f\n", mindeg, maxdeg, avgdeg, variance);
	column_indices = (int *)malloc(count * sizeof(int));
	vector<int>::iterator neighbor_list;
	for (int i = 0, index = 0; i < m; i++) {
		neighbor_list = vertices[i].begin();
		while (neighbor_list != vertices[i].end()) {
			column_indices[index++] = *neighbor_list;
			neighbor_list++;
		}
	}
}

struct Edge {
	int dst;
	foru wt;
};

// transfer mtx graph to CSR format
void mtx2csr(char *mtx, int &m, int &nnz, int *&row_offsets, int *&column_indices, foru *&weight) {
	printf("Reading (.mtx) input file %s\n", mtx);
	std::ifstream cfile;
	cfile.open(mtx);
	std::string str;
	getline(cfile, str);
	char c;
	sscanf(str.c_str(), "%c", &c);
	while (c == '%') {
		getline(cfile, str);
		sscanf(str.c_str(), "%c", &c);
	}
	int n;
	sscanf(str.c_str(), "%d %d %d", &m, &n, &nnz);
	if (m != n) {
		printf("error!\n");
		exit(0);
	}
	printf("num_vertices %d num_edges %d\n", m, nnz);
	vector<vector<Edge> > vertices;
	vector<Edge> neighbors;
	for (int i = 0; i < m; i ++)
		vertices.push_back(neighbors);
	int dst, src, wt;
	for (int i = 0; i < nnz; i ++) {
		getline(cfile, str);
		sscanf(str.c_str(), "%d %d %d", &dst, &src, &wt);
		if (wt <= 0) wt = 1;
		dst--;
		src--;
		Edge e1, e2;
		e1.dst = dst; e1.wt = (foru)wt;
		e2.dst = src; e2.wt = (foru)wt;
		vertices[src].push_back(e1);
		vertices[dst].push_back(e2);
	}
	cfile.close();
	row_offsets = (int *)malloc((m + 1) * sizeof(int));
	int count = 0;
	for (int i = 0; i < m; i++) {
		row_offsets[i] = count;
		count += vertices[i].size();
	}
	row_offsets[m] = count;
	if (count != nnz) {
		printf("This graph is not symmetric\n");
		nnz = count;
	}
	double avgdeg;
	double variance = 0.0;
	int maxdeg = 0;
	int mindeg = m;
	avgdeg = (double)nnz / m;
	for (int i = 0; i < m; i++) {
		int deg_i = row_offsets[i + 1] - row_offsets[i];
		if (deg_i > maxdeg)
			maxdeg = deg_i;
		if (deg_i < mindeg)
			mindeg = deg_i;
		variance += (deg_i - avgdeg) * (deg_i - avgdeg) / m;
	}
	printf("mindeg %d maxdeg %d avgdeg %.2f variance %.2f\n", mindeg, maxdeg, avgdeg, variance);
	column_indices = (int *)malloc(count * sizeof(int));
	weight = (foru *)malloc(count * sizeof(foru));
	vector<Edge>::iterator neighbor_list;
	for (int i = 0, index = 0; i < m; i++) {
		neighbor_list = vertices[i].begin();
		while (neighbor_list != vertices[i].end()) {
			column_indices[index] = (*neighbor_list).dst;
			weight[index] = (*neighbor_list).wt;
			index ++;
			neighbor_list ++;
		}
	}
}

void verify(int m , foru *dist, int *row_offsets, int *column_indices, foru *weight, unsigned *nerr) {
	for (int u = 0; u < m; u ++) {
		int row_begin = row_offsets[u];
		int row_end = row_offsets[u + 1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int v = column_indices[offset];
			foru wt = weight[offset];
			if (wt > 0 && dist[u] + wt < dist[v]) {
				//printf("%d %d %d %d\n", nn, v, dist[nn], dist[v]);
				++*nerr;
			}
		}
	}   
}

__global__ void dverify(int m, foru *dist, int *row_offsets, int *column_indices, foru *weight, unsigned *nerr) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	if (u < m) {
		int row_begin = row_offsets[u];
		int row_end = row_offsets[u + 1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int v = column_indices[offset];
			foru wt = weight[offset];
			if (wt > 0 && dist[u] + wt < dist[v]) {
				//printf("%d %d %d %d\n", u, v, dist[u], dist[v]);
				++*nerr;
			}
		}
	}
}

void write_solution(const char *fname, int m, foru *h_dist) {
	//unsigned *h_dist;
	//h_dist = (unsigned *) malloc(m * sizeof(unsigned));
	assert(h_dist != NULL);
	//CUDA_SAFE_CALL(cudaMemcpy(h_dist, dist, m * sizeof(foru), cudaMemcpyDeviceToHost));
	printf("Writing solution to %s\n", fname);
	FILE *f = fopen(fname, "w");
	fprintf(f, "Computed solution (source dist): [");
	for(int node = 0; node < m; node++) {
		fprintf(f, "%d:%d\n ", node, h_dist[node]);
	}   
	fprintf(f, "]");
}

