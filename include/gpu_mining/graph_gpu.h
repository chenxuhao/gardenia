#pragma once
#include <cuda.h>
#include "common.h"
#include "graph.hh"

class GraphGPU {
protected:
  EdgeID *d_rowptr;
  VertexID *d_colidx;
  BYTE *d_labels;
  VertexID num_vertices;
  EdgeID num_edges;
public:
  GraphGPU() {}
  //~GraphGPU() {}
  void clean() {
    CUDA_SAFE_CALL(cudaFree(d_rowptr));
    CUDA_SAFE_CALL(cudaFree(d_colidx));
  }
  void init(Graph *hg) {
    auto m = hg->num_vertices();
    auto nnz = hg->num_edges();
    num_vertices = m;
    num_edges = nnz;
    auto h_rowptr = hg->out_rowptr();
    auto h_colidx = hg->out_colidx();
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_rowptr, (m + 1) * sizeof(EdgeID)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_colidx, nnz * sizeof(VertexID)));
    CUDA_SAFE_CALL(cudaMemcpy(d_rowptr, h_rowptr, (m + 1) * sizeof(EdgeID), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_colidx, h_colidx, nnz * sizeof(VertexID), cudaMemcpyHostToDevice));
#ifdef ENABLE_LABEL
    BYTE *h_labels = (BYTE *)malloc(m * sizeof(BYTE));
    for (int i = 0; i < m; i++) h_labels[i] = hg->getData(i);
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_labels, m * sizeof(BYTE)));
    CUDA_SAFE_CALL(cudaMemcpy(d_labels, h_labels, m * sizeof(BYTE), cudaMemcpyHostToDevice));
#endif
  }
  __device__ __host__ bool valid_vertex(VertexID vertex) { return (vertex < num_vertices); }
  __device__ __host__ bool valid_edge(EdgeID edge) { return (edge < num_edges); }
  __device__ __host__ EdgeID getOutDegree(VertexID src) {
    assert(src < num_vertices);
    return d_rowptr[src+1] - d_rowptr[src];
  };
  __device__ __host__ VertexID getDestination(VertexID src, EdgeID edge) {
    assert(src < num_vertices);
    assert(edge < getOutDegree(src));
    auto abs_edge = d_rowptr[src] + edge;
    assert(abs_edge < num_edges);
    return d_colidx[abs_edge];
  };
  __device__ __host__ VertexID getAbsDestination(EdgeID abs_edge) {
    assert(abs_edge < num_edges);
    return d_colidx[abs_edge];
  };
  inline __device__ __host__ VertexID getEdgeDst(EdgeID edge) {
    assert(edge < num_edges);
    return d_colidx[edge];
  };
  inline __device__ __host__ BYTE getData(VertexID vid) {
    return d_labels[vid];
  }
  inline __device__ __host__ EdgeID edge_begin(VertexID src) {
    assert(src <= num_vertices);
    return d_rowptr[src];
  };
  inline __device__ __host__ EdgeID edge_end(VertexID src) {
    assert(src <= num_vertices);
    return d_rowptr[src+1];
  };
};

