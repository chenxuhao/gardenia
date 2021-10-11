#pragma once
//#include <fstream>
#include "common.h"
//#include "csr_graph.h"
#include "graph.hh"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"

// TODO: make this template data
typedef int edge_data_t;

class CSRGraph {
public:
  CSRGraph() {}
  CSRGraph(Graph &g) {
    n_vertices = g.V();
    n_edges = g.E();
    alloc_device();
    copy_to_device(g);
  }
  ~CSRGraph() { }//dealloc_device(); }
  __device__ __host__ VertexId out_degree(VertexId v) {
    return VertexId(row_start[v+1] - row_start[v]);
  }
  __device__ __host__ VertexId getEdgeDst(uint64_t e) {
    return edge_dst[e];
  }
  __device__ __host__ uint64_t edge_begin(VertexId v) {
    return row_start[v];
  }
  __device__ __host__ uint64_t edge_end(VertexId v) {
    return row_start[v+1];
  }
  __device__ __host__ edge_data_t getWeight(VertexId v, unsigned offset) {
    return edge_data[row_start[v] + offset];
  }
  __device__ __host__ edge_data_t getAbsWeight(uint64_t e) {
    return edge_data[e];
  }
private:
  //bool device_graph;
  VertexId n_vertices;
  uint64_t n_edges;
  uint64_t* row_start;
  VertexId* edge_dst;
  edge_data_t* edge_data;
  //unsigned init();
  void copy_to_device(Graph& g) {
    auto m = g.V();
    auto nnz = g.E();
    auto h_rowptr = g.out_rowptr();
    auto h_colidx = g.out_colidx();
    CUDA_SAFE_CALL(cudaMemcpy(row_start, h_rowptr, (m+1)*sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(edge_dst, h_colidx, nnz*sizeof(VertexId), cudaMemcpyHostToDevice));
  }
  void alloc_device() {
    CUDA_SAFE_CALL(cudaMalloc((void **)&row_start, (n_vertices+1)*sizeof(uint64_t)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&edge_dst, n_edges*sizeof(VertexId)));
  }
  void dealloc_device() {
    CUDA_SAFE_CALL(cudaFree(row_start));
    CUDA_SAFE_CALL(cudaFree(edge_dst));
  }
};

