#pragma once
#include "graph_gpu.h"
#include "embedding.cuh"

class CUDA_Context_Common {
public:
	int device;
	int id;
	GraphGPU gg;
	Graph *hg;
	void build_graph_gpu() { gg.init(hg); }
};

class CUDA_Context_Mining : public CUDA_Context_Common {
public:
	unsigned max_level;
	EmbeddingList emb_list;
};

