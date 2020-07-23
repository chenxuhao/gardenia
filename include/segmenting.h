// Copyright 2018, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
// This implements the CSR segmenting technique for graph computation

#include "common.h"
#include "timer.h"
#include <vector>
#ifdef GPU_SEGMENTING
#define SUBGRAPH_SIZE (1024*256)
#define RANGE_WIDTH (1024)
#else
#define SUBGRAPH_SIZE (1024*512)
#define RANGE_WIDTH (1024*2)
#endif
#ifdef USE_INT
typedef int ParValueT;
#else
typedef float ParValueT;
#endif

vector<IndexT *> rowptr_blocked;
vector<IndexT *> colidx_blocked;
vector<ValueT *> values_blocked;
vector<IndexT *> idx_map;
vector<IndexT *> range_indices;
//vector<ParValueT *> partial_sums;
vector<int> ms_of_subgraphs;
vector<int> nnzs_of_subgraphs;

// This is for pull model, using incomming edges
void segmenting(int m, IndexT *rowptr, IndexT *colidx, ValueT *values) {
	int num_subgraphs = (m - 1) / SUBGRAPH_SIZE + 1;
	int num_ranges = (m - 1) / RANGE_WIDTH + 1;
	printf("number of subgraphs and ranges: %d, %d\n", num_subgraphs, num_ranges);

	rowptr_blocked.resize(num_subgraphs);
	colidx_blocked.resize(num_subgraphs);
	values_blocked.resize(num_subgraphs);
	ms_of_subgraphs.resize(num_subgraphs);
	nnzs_of_subgraphs.resize(num_subgraphs);
	idx_map.resize(num_subgraphs);
	range_indices.resize(num_subgraphs);
	//partial_sums.resize(num_subgraphs);

	Timer t;
	t.Start();
	std::vector<int> flag(num_subgraphs, false);
	for (int i = 0; i < num_subgraphs; ++i) {
		ms_of_subgraphs[i] = 0;
		nnzs_of_subgraphs[i] = 0;
	}

	printf("calculating number of vertices and edges in each subgraph\n");
	for (IndexT dst = 0; dst < m; ++ dst) {
		int start = rowptr[dst];
		int end = rowptr[dst+1];
		for (IndexT j = start; j < end; ++j) {
			IndexT src = colidx[j];
			int bcol = src / SUBGRAPH_SIZE;
			flag[bcol] = true;
			nnzs_of_subgraphs[bcol]++;
		}
		for (int i = 0; i < num_subgraphs; ++i) {
			if(flag[i]) ms_of_subgraphs[i] ++;
		}
		for (int i = 0; i < num_subgraphs; ++i) flag[i] = false;
	}

	printf("allocating memory for each subgraph\n");
	for (int i = 0; i < num_subgraphs; ++i) {
		rowptr_blocked[i] = (IndexT *) malloc(sizeof(IndexT) * (ms_of_subgraphs[i]+1));
		colidx_blocked[i] = (IndexT *) malloc(sizeof(IndexT) * nnzs_of_subgraphs[i]);
		if(values != NULL) values_blocked[i] = (ValueT *) malloc(sizeof(ValueT) * nnzs_of_subgraphs[i]);
		//partial_sums[i] = (ParValueT *) malloc(sizeof(ParValueT) * ms_of_subgraphs[i]);
		nnzs_of_subgraphs[i] = 0;
		rowptr_blocked[i][0] = 0;
	}

	printf("allocating memory for IdxMap, RangeIndices and IntermBuf\n");
	for (int i = 0; i < num_subgraphs; ++i) {
		//idx_map[i].resize(ms_of_subgraphs[i]);
		//range_indices[i].resize(num_ranges+1);
		idx_map[i] = (IndexT *) malloc(sizeof(IndexT) * ms_of_subgraphs[i]);
		range_indices[i] = (IndexT *) malloc(sizeof(IndexT) * (num_ranges+1));
		range_indices[i][0] = 0;
	}

	printf("constructing the blocked CSR\n");
	std::vector<int> index(num_subgraphs, 0);
	for (IndexT dst = 0; dst < m; ++ dst) {
		for (IndexT j = rowptr[dst]; j < rowptr[dst+1]; ++j) {
			IndexT src = colidx[j];
			int bcol = src / SUBGRAPH_SIZE;
			colidx_blocked[bcol][nnzs_of_subgraphs[bcol]] = src;
			if(values != NULL) values_blocked[bcol][nnzs_of_subgraphs[bcol]] = values[j];
			flag[bcol] = true;
			nnzs_of_subgraphs[bcol]++;
		}
		for (int i = 0; i < num_subgraphs; ++i) {
			if(flag[i]) {
				idx_map[i][index[i]] = dst;
				rowptr_blocked[i][++index[i]] = nnzs_of_subgraphs[i];
			}
		}
		for (int i = 0; i < num_subgraphs; ++i) flag[i] = false;
	}
///*
	printf("printing subgraphs:\n");
	for (int i = 0; i < num_subgraphs; ++i) {
		printf("\tprinting subgraph[%d] (%d vertices, %d edges):\n", i, ms_of_subgraphs[i], nnzs_of_subgraphs[i]);
		/*
		printf("\trow_offsets: ");
		for (int j = 0; j < ms_of_subgraphs[i]+1; ++j)
			printf("%d ", rowptr_blocked[i][j]);
		printf("\n\tcol_indices: ");
		for (int j = 0; j < nnzs_of_subgraphs[i]; ++j)
			printf("%d ", colidx_blocked[i][j]);
		printf("\n");
		//*/
	}
//*/
/*
	int histo[4];
	int total_histo[4];
	for (int j = 0; j < 4; ++j) total_histo[j] = 0;
	for (int i = 0; i < num_subgraphs; ++i) {
		for (int j = 0; j < 4; ++j) histo[j] = 0;
		int nvertices = ms_of_subgraphs[i];
		int *sub_degrees = (int *)malloc(nvertices * sizeof(int));
		for (int j = 0; j < ms_of_subgraphs[i]; ++j) {
			int degree = rowptr_blocked[i][j+1] - rowptr_blocked[i][j];
			if (degree >= 0 && degree < 8) histo[0] ++;
			else if (degree >= 8 && degree < 16) histo[1] ++;
			else if (degree >= 16 && degree < 32) histo[2] ++;
			else if (degree >= 32) histo[3] ++;
			else printf("ERROR\n");
		}
		for (int j = 0; j < 4; ++j)
			total_histo[j] += histo[j];
		for (int j = 0; j < 3; ++j)
			printf("Subgraph[%d] degree %d to %d: %d\n", i, 8*j, 8*(j+1), histo[j]);
		printf("Subgraph[%d] degree 32 and above: %d\n", i, histo[3]);
	}
	for (int j = 0; j < 3; ++j) 
		printf("Total degree %d to %d: %d\n", 8*j, 8*(j+1), total_histo[j]);
	printf("Total degree 32 and above: %d\n", total_histo[3]);
	exit(0);
//*/
	printf("constructing IdxMap and RangeIndices\n");
	for (int i = 0; i < num_subgraphs; ++i) {
		std::vector<int> counts(num_ranges, 0);
		for (int j = 0; j < ms_of_subgraphs[i]; ++ j) {
			int dst = idx_map[i][j];
			counts[dst/RANGE_WIDTH] ++;
		}
		for (int j = 1; j < num_ranges+1; ++j) {
			range_indices[i][j] = range_indices[i][j-1] + counts[j-1];
		}
	}

/*
	printf("printing IdxMaps:\n");
	for (int i = 0; i < num_subgraphs; ++i) {
		printf("\tprinting subgraph[%d] (%d idx_maps, %d ranges):\n", i, ms_of_subgraphs[i], num_ranges);
		printf("\tidx_map: ");
		for (int j = 0; j < ms_of_subgraphs[i]; ++j)
			printf("%d ", idx_map[i][j]);
		printf("\n\tranges: ");
		for (int j = 0; j < num_ranges+1; ++j)
			printf("%d ", range_indices[i][j]);
		printf("\n");
	}
*/
	t.Stop();
	printf("\truntime [preprocessing] = %f ms.\n", t.Millisecs());
}

void free_segments() {
	for (size_t i = 0; i < rowptr_blocked.size(); ++i) {
		free(rowptr_blocked[i]);
		free(colidx_blocked[i]);
		free(values_blocked[i]);
	}
}
