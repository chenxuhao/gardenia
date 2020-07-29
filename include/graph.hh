#pragma once
#include <cassert>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include "VertexSet.h"
#include "scan.h"

constexpr bool map_edges = false; // use mmap() instead of read()
constexpr bool map_vertices = false; // use mmap() instead of read()

class Graph {
private:
  vidType n_vertices, *edges;
  uint64_t n_edges, *vertices;
  vidType max_degree;
  template<typename T>
  static void read_file(std::string fname, T *& pointer, size_t elements) {
    pointer = custom_alloc_global<T>(elements);
    assert(pointer);
    std::ifstream inf(fname.c_str(), std::ios::binary);
    if(!inf.good()) {
      std::cerr << "Failed to open file: " << fname << "\n";
      exit(1);
    }
    inf.read(reinterpret_cast<char*>(pointer), sizeof(T) * elements);
    inf.close();
  }
  template<typename T>
  static void map_file(std::string fname, T *& pointer, size_t elements) {
    int inf = open(fname.c_str(), O_RDONLY, 0);
    if(-1 == inf) {
      std::cerr << "Failed to open file: " << fname << "\n";
      exit(1);
    }
    pointer = (T*)mmap(nullptr, sizeof(T) * elements,
                       PROT_READ, MAP_SHARED, inf, 0);
    assert(pointer != MAP_FAILED);
    close(inf);
  }
  //std::vector<uint64_t> scale_accesses;
public:
  Graph(std::string prefix, bool use_dag = false) {
    VertexSet::release_buffers();
    std::ifstream f_meta((prefix + ".meta.txt").c_str());
    assert(f_meta);
    int vid_size;
    f_meta >> n_vertices >> n_edges >> vid_size >> max_degree;
    assert(sizeof(vidType) == vid_size);
    f_meta.close();
    if(map_vertices) map_file(prefix + ".vertex.bin", vertices, n_vertices+1);
    else read_file(prefix + ".vertex.bin", vertices, n_vertices+1);
    if(map_edges) map_file(prefix + ".edge.bin", edges, n_edges);
    else read_file(prefix + ".edge.bin", edges, n_edges);
    if (max_degree == 0 || max_degree>=n_vertices) exit(1);
    if (use_dag) orientation();
    std::cout << "max_degree: " << max_degree << "\n";
    VertexSet::MAX_DEGREE = std::max(max_degree, VertexSet::MAX_DEGREE);
  }
  ~Graph() {
    if(map_edges) {
      munmap(edges, n_edges*sizeof(vidType));
    } else {
      custom_free(edges, n_edges);
    }
    if(map_vertices) {
      munmap(vertices, (n_vertices+1)*sizeof(uint64_t));
    } else {
      custom_free(vertices, n_vertices+1);
    }
  }
  Graph(const Graph &)=delete;
  Graph& operator=(const Graph &)=delete;
  VertexSet N(vidType vid) {
    assert(vid >= 0);
    assert(vid < n_vertices);
    uint64_t begin = vertices[vid], end = vertices[vid+1];
    assert(begin >= 0);
    if(begin > end) {
      fprintf(stderr, "vertex %u bounds error: [%lu, %lu)\n", vid, begin, end);
      exit(1);
    }
    assert(end <= n_edges);
    return VertexSet(edges + begin, end - begin, vid);
  }
  vidType V() { return n_vertices; }
  size_t E() { return n_edges; }
  size_t size() { return size_t(n_vertices); }
  size_t sizeEdges() { return n_edges; }
  vidType num_vertices() { return n_vertices; }
  size_t num_edges() { return n_edges; }
  uint32_t get_degree(vidType v) { return vertices[v+1] - vertices[v]; }
  uint32_t out_degree(vidType v) { return vertices[v+1] - vertices[v]; }
	uint64_t edge_begin(vidType v) { return vertices[v]; }
	uint64_t edge_end(vidType v) { return vertices[v+1]; }
	uint32_t getEdgeDst(uint64_t e) { return edges[e]; }
	uint32_t get_max_degree() { return max_degree; }
  void orientation() {
    std::cout << "Orientation enabled, using DAG\n";
    std::vector<vidType> degrees(n_vertices, 0);
    #pragma omp parallel for
    for (vidType v = 0; v < n_vertices; v++) {
      degrees[v] = get_degree(v);
    }
    std::vector<vidType> new_degrees(n_vertices, 0);
    #pragma omp parallel for
    for (vidType src = 0; src < n_vertices; src ++) {
      for (auto dst : N(src)) {
        if (degrees[dst] > degrees[src] ||
            (degrees[dst] == degrees[src] && dst > src)) {
          new_degrees[src]++;
        }
      }
    }
    max_degree = *(std::max_element(new_degrees.begin(), new_degrees.end()));
    uint64_t *old_vertices = vertices;
    vidType *old_edges = edges;
    uint64_t *new_vertices = custom_alloc_global<uint64_t>(n_vertices+1);
    //prefix_sum<vidType,uint64_t>(new_degrees, new_vertices);
    parallel_prefix_sum<vidType,uint64_t>(new_degrees, new_vertices);
    auto num_edges = new_vertices[n_vertices];
    vidType *new_edges = custom_alloc_global<vidType>(num_edges);
    #pragma omp parallel for
    for (vidType src = 0; src < n_vertices; src ++) {
      auto begin = new_vertices[src];
      unsigned offset = 0;
      for (auto dst : N(src)) {
        if (degrees[dst] > degrees[src] ||
            (degrees[dst] == degrees[src] && dst > src)) {
          new_edges[begin+offset] = dst;
          offset ++;
        }
      }
    }
    vertices = new_vertices;
    edges = new_edges;
    custom_free<uint64_t>(old_vertices, n_vertices);
    custom_free<vidType>(old_edges, n_edges);
    n_edges = num_edges;
  }
};

