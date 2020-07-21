// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef GRAPH_H_
#define GRAPH_H_

#include <vector>
#include <cstddef>
#include <iostream>
#include <algorithm>
#include "misc.h"
#include "VertexSet.h"
//#include "pvector.h"
//#include <cinttypes>
//#include <type_traits>

/*
GAP Benchmark Suite
Class:  CSRGraph
Author: Scott Beamer

Simple container for graph in CSR format
 - Intended to be constructed by a Builder
 - To make weighted, set DestID_ template type to VertexWeight
 - MakeInverse parameter controls whether graph stores its inverse
*/

// Default type signatures for commonly used types
typedef int32_t VertexID;
typedef int32_t WeightT;

// Used to hold node & weight, with another node it makes a weighted edge
template <typename VertexID_, typename WeightT_>
struct VertexWeight {
	VertexID_ v;
	WeightT_ w;
	VertexWeight() {}
	VertexWeight(VertexID_ v) : v(v), w(1) {}
	VertexWeight(VertexID_ v, WeightT_ w) : v(v), w(w) {}
	bool operator< (const VertexWeight& rhs) const {
		return v == rhs.v ? w < rhs.w : v < rhs.v;
	}
	// doesn't check WeightT_s, needed to remove duplicate edges
	bool operator== (const VertexWeight& rhs) const {
		return v == rhs.v;
	}
	// doesn't check WeightT_s, needed to remove self edges
	bool operator== (const VertexID_& rhs) const {
		return v == rhs;
	}
	operator VertexID_() { return v; }
};
typedef VertexWeight<VertexID, WeightT> WVertex;

template <typename VertexID_, typename WeightT_>
std::ostream& operator<<(std::ostream& os, const VertexWeight<VertexID_, WeightT_>& nw) {
	os << nw.v << " " << nw.w;
	return os;
}

template <typename VertexID_, typename WeightT_>
std::istream& operator>>(std::istream& is, VertexWeight<VertexID_, WeightT_>& nw) {
	is >> nw.v >> nw.w;
	return is;
}

// Syntatic sugar for an edge
template <typename SrcT, typename DstT = SrcT>
struct EdgePair {
	SrcT u;
	DstT v;
	EdgePair() {}
	EdgePair(SrcT u, DstT v) : u(u), v(v) {}
};

// SG = serialized graph, these types are for writing graph to file
typedef int SGID;
typedef EdgePair<SGID> SGEdge;
typedef int SGOffset;

template <class VertexID_, class DestID_ = VertexID_, bool MakeInverse = true>
class CSRGraph {
	// Used for *non-negative* offsets within a neighborhood
	//typedef std::make_unsigned<std::ptrdiff_t>::type OffsetT;
	typedef unsigned OffsetT;

	// Used to access neighbors of vertex, basically sugar for iterators
	class Neighborhood {
		VertexID_ n_;
		DestID_** g_index_;
		OffsetT start_offset_;
		public:
		Neighborhood(VertexID_ n, DestID_** g_index, OffsetT start_offset) :
			n_(n), g_index_(g_index), start_offset_(0) {
				OffsetT max_offset = end() - begin();
				start_offset_ = std::min(start_offset, max_offset);
			}
		typedef DestID_* iterator;
		iterator begin() { return g_index_[n_] + start_offset_; }
		iterator end()   { return g_index_[n_+1]; }
	};

	class Neighbors {
		IndexT n_;
		IndexT* g_rowptr_;
		OffsetT start_offset_;
		public:
		Neighbors(IndexT n, IndexT* g_rowptr, OffsetT start_offset) :
			n_(n), g_rowptr_(g_rowptr), start_offset_(0) {
				OffsetT max_offset = end() - begin();
				start_offset_ = std::min(start_offset, max_offset);
			}
		typedef IndexT iterator;
		iterator begin() { return g_rowptr_[n_] + start_offset_; }
		iterator end()   { return g_rowptr_[n_+1]; }
	};

	void ReleaseResources() {
		if (out_index_ != NULL)
			delete[] out_index_;
		if (out_neighbors_ != NULL)
			delete[] out_neighbors_;
		if (directed_) {
			if (in_index_ != NULL)
				delete[] in_index_;
			if (in_neighbors_ != NULL)
				delete[] in_neighbors_;
		}
	}

public:
	CSRGraph() : directed_(false), num_vertices_(-1), num_edges_(-1),
		out_index_(NULL), out_neighbors_(NULL),
		in_index_(NULL), in_neighbors_(NULL) {}
	CSRGraph(int num_vertices, DestID_** index, DestID_* neighs) :
		directed_(false), num_vertices_(num_vertices),
		out_index_(index), out_neighbors_(neighs),
		in_index_(index), in_neighbors_(neighs) {
			num_edges_ = (out_index_[num_vertices_] - out_index_[0]) / 2;
	}
	CSRGraph(int num_vertices, DestID_** out_index, DestID_* out_neighs,
			DestID_** in_index, DestID_* in_neighs) :
		directed_(true), num_vertices_(num_vertices),
		out_index_(out_index), out_neighbors_(out_neighs),
		in_index_(in_index), in_neighbors_(in_neighs) {
			num_edges_ = out_index_[num_vertices_] - out_index_[0];
	}
	CSRGraph(int num_vertices, int* rowptr, DestID_** index, DestID_* neighs) :
		directed_(false), num_vertices_(num_vertices),
		out_rowptr_(rowptr), out_index_(index), out_neighbors_(neighs),
		in_rowptr_(rowptr), in_index_(index), in_neighbors_(neighs) {
			//num_edges_ = (out_rowptr_[num_vertices_] - out_rowptr_[0]) / 2;
			num_edges_ = (out_index_[num_vertices_] - out_index_[0]) / 2;
	}
	CSRGraph(int num_vertices, int* out_rowptr, DestID_** out_index, DestID_* out_neighs, int* in_rowptr, DestID_** in_index, DestID_* in_neighs) :
		directed_(true), num_vertices_(num_vertices),
		out_rowptr_(out_rowptr), out_index_(out_index), out_neighbors_(out_neighs),
		in_rowptr_(in_rowptr), in_index_(in_index), in_neighbors_(in_neighs) {
			num_edges_ = out_rowptr_[num_vertices_] - out_rowptr_[0];
	}
	~CSRGraph() { ReleaseResources(); }
	int * out_rowptr() const { return out_rowptr_; }
	DestID_ * out_colidx() const { return out_neighbors_; }
	int * in_rowptr() const { return in_rowptr_; }
	DestID_ * in_colidx() const { return in_neighbors_; }
	void Setup(int m, int nnz, IndexT* rowptr, IndexT* neighs) {
		directed_ = false;
		num_vertices_ = m;
		num_edges_ = nnz;
		out_rowptr_ = rowptr;
		out_index_ = NULL;
		out_neighbors_ = neighs;
		in_rowptr_ = NULL;
		in_index_ = NULL;
		in_neighbors_ = NULL;
	}
	void Setup(int m, int nnz, IndexT* rowptr, IndexT** index, IndexT* neighs, ValueT *labels) {
		directed_ = false;
		num_vertices_ = m;
		num_edges_ = nnz;
		out_rowptr_ = rowptr;
		out_index_ = index;
		out_neighbors_ = neighs;
		in_rowptr_ = NULL;
		in_index_ = NULL;
		in_neighbors_ = NULL;
		labels_ = labels;
	}
	void Setup(int num_vertices, int* rowptr, DestID_** index, DestID_* neighs) {
		directed_ = false;
		num_vertices_ = num_vertices;
		out_rowptr_ = rowptr;
		out_index_ = index;
		out_neighbors_ = neighs;
		in_rowptr_ = rowptr;
		in_index_ = index;
		in_neighbors_ = neighs;
		num_edges_ = (out_index_[num_vertices_] - out_index_[0]) / 2;
	}
	void Setup(int num_vertices, int* out_rowptr, DestID_** out_index, DestID_* out_neighs, int* in_rowptr, DestID_** in_index, DestID_* in_neighs) {
		directed_ = true;
		num_vertices_ = num_vertices;
		out_rowptr_ = out_rowptr;
		out_index_ = out_index;
		out_neighbors_ = out_neighs;
		in_rowptr_ = in_rowptr;
		in_index_ = in_index;
		in_neighbors_ = in_neighs;
		num_edges_ = out_rowptr_[num_vertices_] - out_rowptr_[0];
	}
	bool directed() const { return directed_; }
  IndexT get_max_degree() { return max_degree; }
	size_t size() const { return num_vertices_; }
	int num_vertices() const { return num_vertices_; }
	size_t sizeEdges() const { return num_edges_; }
	int num_edges() const { return num_edges_; }
	int num_edges_directed() const { return directed_ ? num_edges_ : 2*num_edges_; }
	int out_degree(VertexID_ v) const { return out_index_[v+1] - out_index_[v]; }
	int in_degree(VertexID_ v) const { return in_index_[v+1] - in_index_[v]; }
	IndexT getEdgeDst(DestID_ e) { return out_neighbors_[e]; }
	IndexT edge_begin(int v) { return out_rowptr_[v]; }
	IndexT edge_end(int v) { return out_rowptr_[v+1]; }
	int get_degree(VertexID_ v) const { return out_index_[v+1] - out_index_[v]; }
  void compute_max_degree() {
    max_degree = 0;
    for (int v = 0; v < num_vertices_; v++) {
      if (max_degree < get_degree(v)) 
        max_degree = get_degree(v);
    }
    std::cout << "max_degree: " << max_degree << "\n";
  }

  VertexSet N(VertexID vid) {
    assert(vid >= 0 && vid < num_vertices_);
    auto begin = out_rowptr_[vid], end = out_rowptr_[vid+1];
    assert(begin >= 0);
    if(begin > end) {
      fprintf(stderr, "vertex %d bounds error: [%d, %d)\n", vid, begin, end);
      exit(1);
    }
    assert(end <= num_edges_);
    return VertexSet(&out_neighbors_[begin], end - begin, vid);
  }

	Neighborhood edges(IndexT n, OffsetT start_offset = 0) const {
		return Neighborhood(n, out_index_, start_offset);
	}
	Neighborhood out_neigh(VertexID_ n, OffsetT start_offset = 0) const {
		return Neighborhood(n, out_index_, start_offset);
	}
	Neighborhood in_neigh(VertexID_ n, OffsetT start_offset = 0) const {
		return Neighborhood(n, in_index_, start_offset);
	}
	void PrintStats() const {
		std::cout << "Graph has " << num_vertices_ << " nodes and "
			<< num_edges_ << " ";
		if (!directed_)
			std::cout << "un";
		std::cout << "directed edges for degree: ";
		std::cout << num_edges_/num_vertices_ << std::endl;
	}
	void print_graph() {
		for (int n = 0; n < num_vertices_; n ++) {
			std::cout << "vertex " << n << ": label = " << " " << " edgelist = [ ";
			IndexT row_begin = edge_begin(n);
			IndexT row_end = edge_end(n);
			for (IndexT e = row_begin; e < row_end; e++)
				std::cout << getEdgeDst(e) << " ";
			std::cout << "]" << std::endl;
		}
	}
	void SetupRowptr() {
		VertexID_ length = num_vertices_+1;
		out_rowptr_ = new int[length];
		for (VertexID_ n=0; n < length; n++)
			out_rowptr_[n] = out_index_[n] - out_neighbors_;
		if (!directed_) {
			in_rowptr_ = new int[length];
			for (VertexID_ n=0; n < length; n++)
				in_rowptr_[n] = in_index_[n] - in_neighbors_;
		}
	}
	int getData(IndexT n) { assert(n < num_vertices_); return labels_[n]; }
	std::vector<SGOffset> VertexOffsets(bool in_graph = false) const {
		std::vector<SGOffset> offsets(num_vertices_+1);
		for (VertexID_ n=0; n < num_vertices_+1; n++)
			if (in_graph)
				offsets[n] = in_index_[n] - in_index_[0];
			else
				offsets[n] = out_index_[n] - out_index_[0];
		return offsets;
	}
	Range<VertexID_> vertices() const {
		return Range<VertexID_>(num_vertices());
	}

private:
	bool directed_;
	int num_vertices_;
	int num_edges_;
	IndexT* out_rowptr_;
	IndexT** out_index_;
	IndexT* out_neighbors_;
	IndexT* in_rowptr_;
	IndexT** in_index_;
	IndexT* in_neighbors_;
	ValueT* labels_;
  IndexT max_degree;
};

typedef CSRGraph<VertexID> Graph;
typedef CSRGraph<VertexID, WVertex> WGraph;
#endif  // GRAPH_H_
