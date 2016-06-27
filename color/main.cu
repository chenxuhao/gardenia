// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu> and Pingfan Li <lipingfan@163.com>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
using namespace std;
#include "common.h"
#include "graph_io.h"
#include "variants.h"

#ifndef	ITERATIONS
#define	ITERATIONS 1
#endif
#ifndef	BLKSIZE
#define	BLKSIZE 128
#endif

// store colour of all vertex
void write_solution(char *fname, int *coloring, int n) {
	int i;
	FILE *fp;
	fp = fopen(fname, "w");
	for (i = 0; i < n; i++) {
		//fprintf(fp, "%d:%d\n", i, coloring[i]);
		fprintf(fp, "%d\n", coloring[i]);
	}
	fclose(fp);
}

// check if correctly coloured
void verify(int m, int nnz, int *csrRowPtr, int *csrColInd, int *coloring, int *correct) {
	int i, offset, neighbor_j;
	for (i = 0; i < m; i++) {
		for (offset = csrRowPtr[i]; offset < csrRowPtr[i + 1]; offset++) {
			neighbor_j = csrColInd[offset];
			if (coloring[i] == coloring[neighbor_j] && neighbor_j != i) {
				*correct = 0;
				//printf("coloring[%d] = coloring[%d] = %d\n", i, neighbor_j, coloring[i]);
				break;
			}
		}	
	}
}

int main(int argc, char *argv[]) {
	if (argc < 2) {
		printf("Usage: %s <graph> [device(0/1)]\n", argv[0]);
		exit(1);
	}
	int m, nnz, *csrRowPtr = NULL, *csrColInd = NULL;
	foru *h_weight = NULL;
	// read graph
	if (strstr(argv[1], ".mtx"))
		mtx2csr(argv[1], m, nnz, csrRowPtr, csrColInd, h_weight);
	else if (strstr(argv[1], ".graph"))
		graph2csr(argv[1], m, nnz, csrRowPtr, csrColInd, h_weight);
	else if (strstr(argv[1], ".gr"))
		gr2csr(argv[1], m, nnz, csrRowPtr, csrColInd, h_weight);
	else { printf("Unrecognizable input file format\n"); exit(0); }

	int device = 0;
	if (argc > 2) device = atoi(argv[2]);
	assert(device == 0 || device == 1);
	int deviceCount = 0;
	CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);
	int nSM = deviceProp.multiProcessorCount;
	fprintf(stdout, "Found %d devices, using device %d (%s), compute capability %d.%d, cores %d*%d.\n", 
			deviceCount, device, deviceProp.name, deviceProp.major, deviceProp.minor, nSM, ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));

	int *coloring = (int *)calloc(m, sizeof(int));
	color(m, nnz, csrRowPtr, csrColInd, coloring);
	write_solution("color.txt", coloring, m);
	int correct = 1;
	verify(m, nnz, csrRowPtr, csrColInd, coloring, &correct);
	if (correct)
		printf("correct.\n");
	else
		printf("incorrect.\n");
	return 0;
}
