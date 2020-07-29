#pragma once
#include <cassert>
#include <fstream>
#include <sstream>
#include <fcntl.h>
#include <iostream>
#include <unistd.h>
#include <algorithm>
#include <sys/mman.h>
#include "common.h"
#include "custom_alloc.h"

class VertexSet {
private:
  VertexId *ptr;
  VertexId size_;
public:
  VertexSet() : size_(0) {}
  VertexSet(VertexId *p, VertexId s) : 
    ptr(p), size_(s) {}
  VertexId size() { return size_; }
  const VertexId* begin() const { return ptr; }
  const VertexId* end() const { return ptr + size_; }
  VertexId get_intersect_num(const VertexSet &other) const {
    VertexId num = 0;
    VertexId idx_l = 0, idx_r = 0;
    while(idx_l < size_ && idx_r < other.size_) {
      auto left = ptr[idx_l];
      auto right = other.ptr[idx_r];
      if(left <= right) idx_l++;
      if(right <= left) idx_r++;
      if(left == right) num++;
    }
    return num;
  }
};

constexpr bool map_edges = false; // use mmap() instead of read()
constexpr bool map_vertices = false; // use mmap() instead of read()

struct Edge {
  VertexId src;
  VertexId dst;
};

class Graph {
private:
  bool directed;
  bool has_reverse;
  VertexId n_vertices, *edges, *reverse_edges;
  uint64_t n_edges, *vertices, *reverse_vertices;
  VertexId max_degree;
  std::vector<VertexList> adj_lists; // temporary adj list
  template<typename T>
  static void read_bin_file(std::string fname, T *& pointer, size_t elements) {
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
  inline bool next_line(ifstream &infile, string &line, istringstream &iss, VertexId &src, VertexId &dest) {
    do {
      if(!getline(infile, line)) return false;
    } while(line.length() == 0 || line[0] == '#');
    iss.clear();
    iss.str(line);
    return !!(iss >> src >> dest);
  }
  void read_mtx_file(std::string fname, bool symmetrize = false, bool need_reverse = false) {
    std::cout << "Reading (.mtx) input file " << fname << "\n";
    std::ifstream infile(fname.c_str());
    if (!infile) {
      cout << "File not available\n";
      throw 1;
    }
    std::string str;
    getline(infile, str);
    char c;
    sscanf(str.c_str(), "%c", &c);
    // skip header comments
    while (c == '%') {
      getline(infile, str);
      sscanf(str.c_str(), "%c", &c);
    }
    // read m, n, nnz
    int m, n;
    int64_t nnz;
    sscanf(str.c_str(), "%d %d %ld", &m, &n, &nnz);
    if (m != n) {
      printf("Warning, m(%d) != n(%d)\n", m, n);
    }
    //std::cout << "original |V| " << m << " |E| " << nnz << "\n";
    n_vertices = m;
    n_edges = 0;
    string line;
    istringstream iss;
    VertexId edge[2];
    //size_t lineNum = 0;
    adj_lists.resize(m);
    while (next_line(infile, line, iss, edge[0], edge[1])) {
      //if (++lineNum % 1000000 == 0)
      //  printf("%lu edges read\n", lineNum);
      if (edge[0] == edge[1]) continue; // self_loop
      auto src = edge[0] - 1;
      auto dst = edge[1] - 1;
      adj_lists[src].push_back(dst);
      n_edges ++;
      if (symmetrize && src != dst) {
        adj_lists[dst].push_back(src);
        n_edges ++;
      }
    }
    infile.close();
    fill_data(symmetrize, need_reverse, true, true);
	}

  void fill_data(bool symmetrize, bool need_reverse, bool sorted, bool remove_redundents) {
    //sort the neighbor list
    if (sorted) {
      //printf("Sorting the neighbor lists...");
      for(int i = 0; i < n_vertices; i++)
        std::sort(adj_lists[i].begin(), adj_lists[i].end());
      //printf(" Done\n");
    }
    // remove redundent
    int num_redundents = 0;
    if(remove_redundents) {
      printf("Removing redundent edges...");
      for (int i = 0; i < n_vertices; i++) {
        for (unsigned j = 1; j < adj_lists[i].size(); j ++) {
          if (adj_lists[i][j] == adj_lists[i][j-1]) {
            adj_lists[i].erase(adj_lists[i].begin()+j);
            num_redundents ++;
            n_edges --;
            j --;
          }
        }
      }
      printf(" %d redundent edges are removed\n", num_redundents);
    }
    std::cout << "|V| " << n_vertices << " |E| " << n_edges << "\n";
    vertices = custom_alloc_global<uint64_t>(n_vertices+1);
    vertices[0] = 0;
    max_degree = 0;
    for (int i = 1; i < n_vertices+1; i++) {
      auto degree = adj_lists[i-1].size();
      if (VertexId(degree) > max_degree)
        max_degree = VertexId(degree);
      vertices[i] = vertices[i-1] + degree;
    }
    edges = custom_alloc_global<VertexId>(n_edges);
    //#pragma omp parallel for
    for (VertexId i = 0; i < n_vertices; i++) {
      auto begin = vertices[i];
      std::copy(adj_lists[i].begin(), adj_lists[i].end(), &edges[begin]);
    }
    // generate the reverse (transposed) graph for directed graph
    if (!symmetrize && need_reverse) {
      build_reverse_graph();
    }
    for (VertexId i = 0; i < n_vertices; i++)
      adj_lists[i].clear();
    adj_lists.clear();
  }
  void build_reverse_graph() {
    std::vector<VertexList> reverse_adj_lists(n_vertices);
    for (VertexId v = 0; v < n_vertices; v++) {
      //for (auto u : adj_lists[v]) {
      for (auto u : N(v)) {
        reverse_adj_lists[u].push_back(v);
      }
    }
    reverse_vertices = custom_alloc_global<uint64_t>(n_vertices+1);
    reverse_vertices[0] = 0;
    for (VertexId i = 1; i < n_vertices+1; i++) {
      auto degree = reverse_adj_lists[i-1].size();
      reverse_vertices[i] = reverse_vertices[i-1] + degree;
    }
    reverse_edges = custom_alloc_global<VertexId>(n_edges);
    //#pragma omp parallel for
    for (VertexId i = 0; i < n_vertices; i++) {
      auto begin = reverse_vertices[i];
      std::copy(reverse_adj_lists[i].begin(), 
                reverse_adj_lists[i].end(), &reverse_edges[begin]);
    }
    for (VertexId i = 0; i < n_vertices; i++)
      reverse_adj_lists[i].clear();
    reverse_adj_lists.clear();
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
  Graph(std::string prefix, 
        std::string filetype = "bin",
        bool symmetrize = false,
        bool need_reverse = false) {
    if (filetype == "mtx") {
      std::string filename = prefix + ".mtx";
      read_mtx_file(filename, symmetrize, need_reverse);
    } else if (filetype == "bin") {
      std::ifstream f_meta((prefix + ".meta.txt").c_str());
      assert(f_meta);
      max_degree = 0;
      int vid_size;
      f_meta >> n_vertices >> n_edges >> vid_size >> max_degree;
      std::cout << "|V| " << n_vertices << " |E| " << n_edges << "\n";
      assert(sizeof(VertexId) == vid_size);
      f_meta.close();
      if(map_vertices) map_file(prefix + ".vertex.bin", vertices, n_vertices+1);
      else read_bin_file(prefix + ".vertex.bin", vertices, n_vertices+1);
      if(map_edges) map_file(prefix + ".edge.bin", edges, n_edges);
      else read_bin_file(prefix + ".edge.bin", edges, n_edges);
      if (!symmetrize && need_reverse)
        build_reverse_graph();
    }
    directed = false;
    has_reverse = false;
    if (!symmetrize && need_reverse) {
      directed = true;
      printf("This graph maintains both incomming and outgoing edge-list\n"); 
      has_reverse = true;
    }
    if (symmetrize) {
      printf("This graph is symmetrized\n");
      reverse_vertices = vertices;
      reverse_edges = edges;
      has_reverse = true;
    }
    //std::cout << "max_degree: " << max_degree << "\n";
    if (max_degree == 0 || max_degree>=n_vertices) exit(1);
    //if (use_dag) orientation();
  }
  ~Graph() {
    if(map_edges) {
      munmap(edges, n_edges*sizeof(VertexId));
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
  VertexSet N(VertexId vid) const {
    assert(vid >= 0);
    assert(vid < n_vertices);
    uint64_t begin = vertices[vid], end = vertices[vid+1];
    assert(begin >= 0);
    if(begin > end) {
      fprintf(stderr, "vertex %u bounds error: [%lu, %lu)\n", vid, begin, end);
      exit(1);
    }
    assert(end <= n_edges);
    return VertexSet(edges + begin, end - begin);
  }
  VertexSet out_neigh(VertexId vid, VertexId start_offset = 0) const {
    auto begin = vertices[vid];
    auto end = vertices[vid+1];
    auto r = std::min(start_offset, VertexId(end - begin));
		begin += r;
    return VertexSet(edges + begin, end - begin);
  }
  VertexSet in_neigh(VertexId vid) const {
    auto begin = reverse_vertices[vid];
    auto end = reverse_vertices[vid+1];
    return VertexSet(reverse_edges + begin, end - begin);
  }
  VertexId V() { return n_vertices; }
  size_t E() { return n_edges; }
  size_t size() { return size_t(n_vertices); }
  size_t sizeEdges() { return n_edges; }
  VertexId num_vertices() { return n_vertices; }
  size_t num_edges() { return n_edges; }
  VertexId get_degree(VertexId v) { return vertices[v+1] - vertices[v]; }
  VertexId out_degree(VertexId v) { return vertices[v+1] - vertices[v]; }
	uint64_t edge_begin(VertexId v) { return vertices[v]; }
	uint64_t edge_end(VertexId v) { return vertices[v+1]; }
	VertexId getEdgeDst(uint64_t e) { return edges[e]; }
	VertexId get_max_degree() { return max_degree; }
  bool is_directed() { return directed; }
  bool has_reverse_graph() { return has_reverse; }
  uint64_t* out_rowptr() { return vertices; }
  VertexId* out_colidx() { return edges; }
  uint64_t* in_rowptr() { return reverse_vertices; }
  VertexId* in_colidx() { return reverse_edges; }

  void orientation() {
    std::cout << "Orientation enabled, using DAG\n";
    std::vector<VertexId> degrees(n_vertices, 0);
    for (VertexId v = 0; v < n_vertices; v++) {
      degrees[v] = get_degree(v);
    }
    std::vector<VertexId> new_degrees(n_vertices, 0);
    for (VertexId src = 0; src < n_vertices; src ++) {
      for (auto dst : N(src)) {
        if (degrees[dst] > degrees[src] ||
            (degrees[dst] == degrees[src] && dst > src)) {
          new_degrees[src]++;
        }
      }
    }
    uint64_t *old_vertices = vertices;
    VertexId *old_edges = edges;
    uint64_t *new_vertices = custom_alloc_global<uint64_t>(n_vertices+1);
    new_vertices[0] = 0;
    for (VertexId v = 1; v < n_vertices+1; v++) {
      new_vertices[v] = new_vertices[v-1] + new_degrees[v-1];
    }
    //ParallelPrefixSum(new_degrees, new_vertices);
    auto num_edges = new_vertices[n_vertices];
    VertexId *new_edges = custom_alloc_global<VertexId>(num_edges);
    for (VertexId src = 0; src < n_vertices; src ++) {
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
    custom_free<VertexId>(old_edges, n_edges);
    n_edges = num_edges;
    std::cout << "|V| " << n_vertices << " |E| " << n_edges << "\n";
  }
};

inline uint64_t intersection_num(const VertexSet& a, const VertexSet& b) {
  return a.get_intersect_num(b);
}

