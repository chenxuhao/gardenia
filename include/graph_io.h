// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu> and Pingfan Li <lipingfan@163.com>

#include <vector>
#include <set>
#include <iostream>
#include <fstream>
#include <sstream>
#define W_TYPE double
struct Edge {
	int dst;
	W_TYPE wt;
};

// transfer R-MAT generated gr graph to CSR format
void gr2csr(char *gr, int &m, int &nnz, int *&row_offsets, int *&column_indices, W_TYPE *&weight) {
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
void graph2csr(char *graph, int &m, int &nnz, int *&row_offsets, int *&column_indices, W_TYPE *&weight) {
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
	weight = (W_TYPE *)malloc(count * sizeof(W_TYPE));
	vector<int>::iterator neighbor_list;
	for (int i = 0, index = 0; i < m; i++) {
		neighbor_list = vertices[i].begin();
		while (neighbor_list != vertices[i].end()) {
			column_indices[index++] = *neighbor_list;
			weight[index] = 1;
			neighbor_list++;
		}
	}
}

// transfer mtx graph to CSR format
void mtx2csr(char *mtx, int &m, int &nnz, int *&row_offsets, int *&column_indices, W_TYPE *&weight) {
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
		e1.dst = dst; e1.wt = (W_TYPE)wt;
		e2.dst = src; e2.wt = (W_TYPE)wt;
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
	weight = (W_TYPE *)malloc(count * sizeof(W_TYPE));
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

void read_graph(int argc, char *argv[], int &m, int &nnz, int *&row_offsets, int *&column_indices, int *&degree, W_TYPE *&weight) {
	if (strstr(argv[1], ".mtx"))
		mtx2csr(argv[1], m, nnz, row_offsets, column_indices, weight);
	else if (strstr(argv[1], ".graph"))
		graph2csr(argv[1], m, nnz, row_offsets, column_indices, weight);
	else if (strstr(argv[1], ".gr"))
		gr2csr(argv[1], m, nnz, row_offsets, column_indices, weight);
	else { printf("Unrecognizable input file format\n"); exit(0); }
	degree = (int *)malloc(m * sizeof(int));
	for (int i = 0; i < m; i++) {
		degree[i] = row_offsets[i + 1] - row_offsets[i];
	}
}

