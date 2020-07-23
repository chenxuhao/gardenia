// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu> and Pingfan Li <lipingfan@163.com>

#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <algorithm>
#include "common.h"
#include "timer.h"

struct WeightedEdge {
	IndexT src;
	IndexT dst;
	WeightT wt;
	int eid;
	//WeightedEdge() : src(0), dst(0), wt(0), eid(0) {}
	//std::string to_string() const;
};

bool compare_id(WeightedEdge a, WeightedEdge b) { return (a.dst < b.dst); }

void fill_data(int m, int &nnz, IndexT *&row_offsets, IndexT *&column_indices, WeightT *&weight, vector<vector<WeightedEdge> > vertices, bool symmetrize, bool sorted, bool remove_selfloops, bool remove_redundents) {
	//sort the neighbor list
	if(sorted) {
		printf("Sorting the neighbor lists...");
		for(int i = 0; i < m; i++) {
			std::sort(vertices[i].begin(), vertices[i].end(), compare_id);
		}
		printf(" Done\n");
	}

	//remove self loops
	int num_selfloops = 0;
	if(remove_selfloops) {
		printf("Removing self loops...");
		for(int i = 0; i < m; i++) {
			for(unsigned j = 0; j < vertices[i].size(); j ++) {
				if(i == vertices[i][j].dst) {
					vertices[i].erase(vertices[i].begin()+j);
					num_selfloops ++;
					j --;
				}
			}
		}
		printf(" %d selfloops are removed\n", num_selfloops);
	}

	// remove redundent
	int num_redundents = 0;
	if(remove_redundents) {
		printf("Removing redundent edges...");
		for (int i = 0; i < m; i++) {
			for (unsigned j = 1; j < vertices[i].size(); j ++) {
				if (vertices[i][j].dst == vertices[i][j-1].dst) {
					vertices[i].erase(vertices[i].begin()+j);
					num_redundents ++;
					j --;
				}
			}
		}
		printf(" %d redundent edges are removed\n", num_redundents);
	}

/*
	// print some neighbor lists
	for (int i = 0; i < 3; i++) {
		cout << "src " << i << ": ";
		for (int j = 0; j < vertices[i].size(); j ++)
			cout << vertices[i][j].dst << "  ";
		cout << endl;
	}
*/
#ifdef SIM
	row_offsets = (IndexT *)aligned_alloc(PAGE_SIZE, (m + 1) * sizeof(IndexT));
#else
	row_offsets = (IndexT *)malloc((m + 1) * sizeof(IndexT));
#endif
	int count = 0;
	for (int i = 0; i < m; i++) {
		row_offsets[i] = count;
		count += vertices[i].size();
	}
	row_offsets[m] = count;
	if (symmetrize) {
		if(count != nnz) {
			nnz = count;
		}
	} else {
		if (count + num_selfloops + num_redundents != nnz)
			printf("Error reading graph, number of edges in edge list %d != %d\n", count, nnz);
		nnz = count;
	}
	printf("num_vertices %d num_edges %d\n", m, nnz);
	/*
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
	printf("min_degree %d max_degree %d avg_degree %.2f variance %.2f\n", mindeg, maxdeg, avgdeg, variance);
	*/
#ifdef SIM
	column_indices = (IndexT *)aligned_alloc(PAGE_SIZE, count * sizeof(IndexT));
	weight = (WeightT *)aligned_alloc(PAGE_SIZE, count * sizeof(WeightT));
#else
	column_indices = (IndexT *)malloc(count * sizeof(IndexT));
	weight = (WeightT *)malloc(count * sizeof(WeightT));
#endif
	vector<WeightedEdge>::iterator neighbor_list;
	for (int i = 0, index = 0; i < m; i++) {
		neighbor_list = vertices[i].begin();
		while (neighbor_list != vertices[i].end()) {
			column_indices[index] = (*neighbor_list).dst;
			weight[index] = (*neighbor_list).wt;
			index ++;
			neighbor_list ++;
		}
	}
	/*
	// print some neighbor lists
	for (int i = 0; i < 6; i++) {
		int row_begin = row_offsets[i];
		int row_end = row_offsets[i + 1];
		cout << "src " << i << ": ";
		for (int j = row_begin; j < row_end; j ++)
			cout << column_indices[j] << "  ";
		cout << endl;
	}
	//*/
	//for (int i = 0; i < 10; i++) cout << weight[i] << ", ";
	//cout << endl;
}

// transfer gr graph to CSR format
void gr2csr(char *gr, int &m, int &nnz, IndexT *&row_offsets, IndexT *&column_indices, WeightT *&weight, bool symmetrize, bool transpose, bool sorted, bool remove_selfloops, bool remove_redundents) {
	printf("Reading 9th DIMACS (.gr) input file %s\n", gr);
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
	printf("Before cleaning, the original num_vertices %d num_edges %d\n", m, nnz);

	getline(cfile, str);
	sscanf(str.c_str(), "%c", &c);
	while (c == 'c') {
		getline(cfile, str);
		sscanf(str.c_str(), "%c", &c);
	}
	vector<vector<WeightedEdge> > vertices;
	vector<WeightedEdge> neighbors;
	for (int i = 0; i < m; i++)
		vertices.push_back(neighbors);
	IndexT src, dst;
	for (int i = 0; i < nnz; i++) {
#ifdef LONG_TYPES
		sscanf(str.c_str(), "%c %ld %ld", &c, &src, &dst);
#else
		sscanf(str.c_str(), "%c %d %d", &c, &src, &dst);
#endif
		if (c != 'a')
			printf("line %d\n", __LINE__);
		src--;
		dst--;
		WeightedEdge e1, e2;
		if(symmetrize) {
			e2.dst = src; e2.wt = 1;
			vertices[dst].push_back(e2);
			transpose = false;
		}
		if(!transpose) {
			e1.dst = dst; e1.wt = 1;
			vertices[src].push_back(e1);
		} else {
			e1.dst = src; e1.wt = 1;
			vertices[dst].push_back(e1);
		}
		if(i != nnz-1) getline(cfile, str);
	}
	fill_data(m, nnz, row_offsets, column_indices, weight, vertices, symmetrize, sorted, remove_selfloops, remove_redundents);
}

// transfer edgelist graph to CSR format
void el2csr(char *el, int &m, int &nnz, IndexT *&row_offsets, IndexT *&column_indices, WeightT *&weight, bool symmetrize, bool transpose, bool sorted, bool remove_selfloops, bool remove_redundents) {
	printf("Reading edgelist (.el) input file %s\n", el);
	std::ifstream cfile;
	cfile.open(el);
	std::string str;
	getline(cfile, str);
	sscanf(str.c_str(), "%d %d", &m, &nnz);
	printf("Before cleaning, the original num_vertices %d num_edges %d\n", m, nnz);
	vector<vector<WeightedEdge> > vertices;
	vector<WeightedEdge> neighbors;
	for (int i = 0; i < m; i++)
		vertices.push_back(neighbors);
	IndexT dst, src;
	WeightT wt = 1;
	for (int i = 0; i < nnz; i ++) {
	//while (!cfile.eof()) {
		getline(cfile, str);
#ifdef LONG_TYPES
		int num = sscanf(str.c_str(), "%ld %ld %ld", &src, &dst, &wt);
#else
		int num = sscanf(str.c_str(), "%d %d %d", &src, &dst, &wt);
#endif
		if (num == 2) wt = 1;
		if (wt < 0) wt = -wt; // non-negtive weight
		src--;
		dst--;
		WeightedEdge e1, e2;
		if(symmetrize && src != dst) {
			e2.dst = src; e2.wt = wt;
			vertices[dst].push_back(e2);
			transpose = false;
		}
		if(!transpose) {
			e1.dst = dst; e1.wt = wt;
			vertices[src].push_back(e1);
		} else {
			e1.dst = src; e1.wt = wt;
			vertices[dst].push_back(e1);
		}
	}
	cfile.close();
	fill_data(m, nnz, row_offsets, column_indices, weight, vertices, symmetrize, sorted, remove_selfloops, remove_redundents);
}

// transfer *.graph file to CSR format
void graph2csr(char *graph, int &m, int &nnz, IndexT *&row_offsets, IndexT *&column_indices, WeightT *&weight, bool symmetrize, bool transpose, bool sorted, bool remove_selfloops, bool remove_redundents) {
	printf("Reading .graph input file %s\n", graph);
	std::ifstream cfile;
	cfile.open(graph);
	std::string str;
	getline(cfile, str);
	sscanf(str.c_str(), "%d %d", &m, &nnz);
	printf("Before cleaning, the original num_vertices %d num_edges %d\n", m, nnz);
	vector<vector<WeightedEdge> > vertices;
	vector<WeightedEdge> neighbors;
	for (int i = 0; i < m; i++)
		vertices.push_back(neighbors);
	IndexT dst;
	for (int src = 0; src < m; src ++) {
		getline(cfile, str);
		istringstream istr;
		istr.str(str);
		while(istr>>dst) {
			dst --;
			WeightedEdge e1;//, e2;
			if(symmetrize && src != dst) {
				// for .graph format, the input file already contains edges in both directions
				//e2.dst = src; e2.wt = 1;
				//vertices[dst].push_back(e2);
				transpose = false;
			}
			if(!transpose) {
				e1.dst = dst; e1.wt = 1;
				vertices[src].push_back(e1);
			} else {
				e1.dst = src; e1.wt = 1;
				vertices[dst].push_back(e1);
			}
		}
		istr.clear();
	}
    cfile.close();
	fill_data(m, nnz, row_offsets, column_indices, weight, vertices, symmetrize, sorted, remove_selfloops, remove_redundents);
}

// transfer mtx graph to CSR format
void mtx2csr(char *mtx, int &m, int &n, int &nnz, IndexT *&row_offsets, IndexT *&column_indices, WeightT *&weight, bool symmetrize, bool transpose, bool sorted, bool remove_selfloops, bool remove_redundents) {
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
	sscanf(str.c_str(), "%d %d %d", &m, &n, &nnz);
	if (m != n) {
		printf("Warning, m(%d) != n(%d)\n", m, n);
	}
	printf("Before cleaning, the original num_vertices %d num_edges %d\n", m, nnz);
	vector<vector<WeightedEdge> > vertices;
	vector<WeightedEdge> neighbors;
	for (int i = 0; i < m; i ++)
		vertices.push_back(neighbors);
	IndexT dst, src;
	WeightT wt = 1;
	for (int i = 0; i < nnz; i ++) {
		getline(cfile, str);
#ifdef LONG_TYPES
		int num = sscanf(str.c_str(), "%ld %ld %ld", &src, &dst, &wt);
#else
		int num = sscanf(str.c_str(), "%d %d %d", &src, &dst, &wt);
#endif
		if (num == 2) wt = 1;
		if (wt < 0) wt = -wt; // non-negtive weight
		src--;
		dst--;
		WeightedEdge e1, e2;
		if(symmetrize && src != dst) {
			e2.dst = src; e2.wt = wt;
			vertices[dst].push_back(e2);
			transpose = false;
		}
		if(!transpose) {
			e1.dst = dst; e1.wt = wt;
			vertices[src].push_back(e1);
		} else {
			e1.dst = src; e1.wt = wt;
			vertices[dst].push_back(e1);
		}
	}
	cfile.close();
	fill_data(m, nnz, row_offsets, column_indices, weight, vertices, symmetrize, sorted, remove_selfloops, remove_redundents);
}
/*
void sort_neighbors(int m, int *row_offsets, int *&column_indices) {
	vector<int> neighbors;
	#pragma omp parallel for
	for(int i = 0; i < m; i++) {
		int row_begin = row_offsets[i];
		int row_end = row_offsets[i + 1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			neighbors.push_back(column_indices[offset]);
		}
		std::sort(neighbors.begin(), neighbors.end());
		int k = 0;
		for (int offset = row_begin; offset < row_end; ++ offset) {
			column_indices[offset] = neighbors[k++];
		}
	}	
}
*/
void read_graph(int argc, char *argv[], int &m, int &n, int &nnz, IndexT *&row_offsets, IndexT *&column_indices, int *&degree, WeightT *&weight, bool is_symmetrize=false, bool is_transpose=false, bool sorted=true, bool remove_selfloops=true, bool remove_redundents=true) {
	Timer t;
	t.Start();
	//if(is_symmetrize) printf("Requiring symmetric graphs for this algorithm\n");
	if (strstr(argv[1], ".mtx"))
		mtx2csr(argv[1], m, n, nnz, row_offsets, column_indices, weight, is_symmetrize, is_transpose, sorted, remove_selfloops, remove_redundents);
	else if (strstr(argv[1], ".graph"))
		graph2csr(argv[1], m, nnz, row_offsets, column_indices, weight, is_symmetrize, is_transpose, sorted, remove_selfloops, remove_redundents);
	else if (strstr(argv[1], ".gr"))
		gr2csr(argv[1], m, nnz, row_offsets, column_indices, weight, is_symmetrize, is_transpose, sorted, remove_selfloops, remove_redundents);
	else { printf("Unrecognizable input file format\n"); exit(0); }
	t.Stop();
	printf("\truntime [%s] = %f ms.\n", "read_graph", t.Millisecs());

	printf("Calculating degree...");
	degree = (int *)malloc(m * sizeof(int));
	for (int i = 0; i < m; i++) {
		degree[i] = row_offsets[i + 1] - row_offsets[i];
	}
	printf(" Done\n");
}

void print_degree(int m, int *in_degree, int *out_degree) {
	if(in_degree != NULL) {
		FILE *fp = fopen("in_degree.txt", "w");
		fprintf(fp,"%d\n", m);
		for(int i = 0; i < m; i ++)
			fprintf(fp,"%d ", in_degree[i]);
		fclose(fp);
	}
	if(out_degree != NULL) {
		FILE *fp = fopen("out_degree.txt", "w");
		fprintf(fp,"%d\n", m);
		for(int i = 0; i < m; i ++)
			fprintf(fp,"%d ", out_degree[i]);
		fclose(fp);
	}
}

