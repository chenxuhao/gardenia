#include <vector>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <string>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <fstream>

#pragma once
using namespace std;

struct GPUBucketArrays {
	int* vertex; //vert Ids	
	unsigned int* bucketId;	

	int* farVertex;
	unsigned int* farBuckets;
};

template <bool FarPile>
struct Buckets {
	GPUBucketArrays<unsigned int> gpu;
	unsigned int minBucketId;
	unsigned int maxBucketId; //Mostly just for debugging
	unsigned int num_elements; 
	unsigned int num_buckets;
	unsigned int zero_elements;
	unsigned int max_vertices;
	unsigned int max_edges;
	unsigned int bucket_limit;
	//CPU side for debugging/moving data around
	unsigned int* buckets;
	unsigned int* vertexId;
	//unsigned int* bucketStart;
	//unsigned int* bucketSize;	
	unsigned int farSize;	
	void allocGPUSpace() {
		if(FarPile) {
			cudaMalloc((void**)& gpu.vertex,    sizeof(uint)*max_edges);			
			cudaMalloc((void**)& gpu.bucketId,  sizeof(uint)*max_edges);
			cudaMalloc((void**)& gpu.farVertex, sizeof(uint)*max_edges);
			cudaMalloc((void**)& gpu.farBuckets,sizeof(uint)*max_edges);
		}
		else {
			cudaMalloc((void**)& gpu.vertex,    sizeof(uint)*max_edges);			
		}
	}
	
	Buckets(int sourceId, int closeBucketsLimit, unsigned int delta, int numEdges, int numVertices) {
		num_elements = zero_elements = 1;
		num_buckets = 1;		
		minBucketId = 0;
		farSize = 0;		
		max_vertices = numVertices;
		max_edges = numEdges;
		vertexId = (unsigned int*)malloc(sizeof(unsigned int)*max_edges); //upper bound on number of edges
		vertexId[0] = sourceId;
		//printf("Alloc gpu space\n");
		allocGPUSpace();		
		//printf("Done alloc gpu space %s\n", (FarPile ? "Farpile" : "No FarPile"));
		cudaMemcpy(gpu.vertex, vertexId, sizeof(unsigned int)*num_elements, cudaMemcpyHostToDevice);	
		if(FarPile) {		    
			buckets = (unsigned int*)malloc(sizeof(unsigned int)*max_edges); //upper bound on number of edges
		}
	}
	Buckets<FarPile>(int upperLimit, int numVertices) {
		Buckets<FarPile>(0, 20, 2.0, upperLimit, numVertices);
	}
	~Buckets() {
		freeAllMem();
	}
	
	void freeAllMem() {
		free(vertexId);
		if(FarPile) {
			cudaFree(gpu.bucketId);
			cudaFree(gpu.farVertex);
			cudaFree(gpu.farBuckets);
			free(buckets);
		}
		cudaFree(gpu.vertex);
	}
	void copyBucketInfoToDevice() { }
	void copyBucketInfoToHost(int numElements)
		cudaMemcpy(vertexId, gpu.vertex, sizeof(unsigned int)*numElements, cudaMemcpyDeviceToHost);	
		if(FarPile)
			cudaMemcpy(buckets, gpu.bucketId, sizeof(unsigned int)*numElements, cudaMemcpyDeviceToHost);	
	}
	void printBasicInfo() {
		printf("minBucketId %u\n", minBucketId);
	}
	void printVertexFront(ofstream &f, int numElements) {
		for(size_t i = 0; i < numElements; i++) {			
			f << vertexId[i] << " " << buckets[i] << endl;	
		}
		printf("done\n");
	}
	//(1) Compact New Set (vertex frontier)
	//(2) Sort New Set (bydistance)
	//(3) Remove Duplicate vertices?
	//(4) Merge New Set into Bucket List
};

