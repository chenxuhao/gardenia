// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "cc.h"
#include "timer.h"
#include "platform_atomics.h"
#include <random>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#define CC_VARIANT "omp_afforest"

// Place nodes u and v in same component of lower component ID
void Link(IndexT u, IndexT v, IndexT *comp) {
	IndexT p1 = comp[u];
	IndexT p2 = comp[v];
	while (p1 != p2) {
		IndexT high = p1 > p2 ? p1 : p2;
		IndexT low = p1 + (p2 - high);
		IndexT p_high = comp[high];
		// Was already 'low' or succeeded in writing 'low'
		if ((p_high == low) || (p_high == high && compare_and_swap(comp[high], high, low)))
			break;
		p1 = comp[comp[high]];
		p2 = comp[low];
	}
}

// Reduce depth of tree for each component to 1 by crawling up parents
void Compress(int m, IndexT *comp) {
	#pragma omp parallel for schedule(static, 2048)
	for (IndexT n = 0; n < m; n++) {
		while (comp[n] != comp[comp[n]]) {
			comp[n] = comp[comp[n]];
		}
	}
}
/*
IndexT SampleFrequentElement(int m, IndexT *comp, int64_t num_samples = 1024) {
	std::unordered_map<IndexT, int> sample_counts(32);
	using kvp_type = std::unordered_map<IndexT, int>::value_type;
	// Sample elements from 'comp'
	std::mt19937 gen;
	std::uniform_int_distribution<IndexT> distribution(0, m - 1);
	for (IndexT i = 0; i < num_samples; i++) {
		IndexT n = distribution(gen);
		sample_counts[comp[n]]++;
	}
	// Find most frequent element in samples (estimate of most frequent overall)
	auto most_frequent = std::max_element(
			sample_counts.begin(), sample_counts.end(),
			[](const kvp_type& a, const kvp_type& b) { return a.second < b.second; });
	float frac_of_graph = static_cast<float>(most_frequent->second) / num_samples;
	std::cout
		<< "Skipping largest intermediate component (ID: " << most_frequent->first
		<< ", approx. " << static_cast<int>(frac_of_graph) * 100
		<< "% of the graph)" << std::endl;
	return most_frequent->first;
}
*/
void Afforest(int m, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *row_offsets, IndexT *column_indices, CompT *comp, bool is_directed, int32_t neighbor_rounds = 2) {
	// Process a sparse sampled subgraph first for approximating components.
	// Sample by processing a fixed number of neighbors for each vertex
	for (int r = 0; r < neighbor_rounds; ++r) {
		#pragma omp parallel for
		for (IndexT src = 0; src < m; src ++) {
			//for (IndexT v : g.out_neigh(u, r)) {
			IndexT row_begin = row_offsets[src];
			IndexT row_end = row_offsets[src+1];
			IndexT start_offset = std::min(r, row_end - row_begin);
			row_begin += start_offset;
			for (IndexT offset = row_begin; offset < row_end; offset ++) {
				IndexT dst = column_indices[offset];
				// Link at most one time if neighbor available at offset r
				Link(src, dst, comp);
				break;
			}
		}
		Compress(m, comp);
	}

	// Sample 'comp' to find the most frequent element -- due to prior
	// compression, this value represents the largest intermediate component
	IndexT c = SampleFrequentElement(m, comp);

	// Final 'link' phase over remaining edges (excluding largest component)
	if (!is_directed) {
		#pragma omp parallel for schedule(dynamic, 2048)
		for (IndexT src = 0; src < m; src ++) {
			// Skip processing nodes in the largest component
			if (comp[src] == c) continue;
			// Skip over part of neighborhood (determined by neighbor_rounds)
			//for (IndexT v : g.out_neigh(u, neighbor_rounds)) {
			IndexT row_begin = row_offsets[src];
			IndexT row_end = row_offsets[src+1];
			IndexT start_offset = std::min(neighbor_rounds, row_end - row_begin);
			row_begin += start_offset;
			for (IndexT offset = row_begin; offset < row_end; offset ++) {
				IndexT dst = column_indices[offset];
				Link(src, dst, comp);
			}
		}
	} else {
		#pragma omp parallel for schedule(dynamic, 2048)
		for (IndexT src = 0; src < m; src ++) {
			if (comp[src] == c) continue;
			//for (IndexT v : g.out_neigh(u, neighbor_rounds)) {
			IndexT row_begin = row_offsets[src];
			IndexT row_end = row_offsets[src+1];
			IndexT start_offset = std::min(neighbor_rounds, row_end - row_begin);
			row_begin += start_offset;
			for (IndexT offset = row_begin; offset < row_end; offset ++) {
				IndexT dst = column_indices[offset];
				Link(src, dst, comp);
			}
			// To support directed graphs, process reverse graph completely
			row_begin = in_row_offsets[src];
			row_end = in_row_offsets[src+1];
			for (IndexT offset = row_begin; offset < row_end; offset ++) {
				IndexT dst = in_column_indices[offset];
				Link(src, dst, comp);
			}
		}
	}
	// Finally, 'compress' for final convergence
	Compress(m, comp);
}

void CCSolver(int m, int nnz, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *degrees, CompT *comp, bool is_directed) {
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP CC solver (%d threads) ...\n", num_threads);

	// Initialize each node to a single-node self-pointing tree
	#pragma omp parallel for
	for (int n = 0; n < m; n ++) comp[n] = n;

	Timer t;
	t.Start();
	Afforest(m, in_row_offsets, in_column_indices, out_row_offsets, out_column_indices, comp, is_directed);
	t.Stop();

	//printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", CC_VARIANT, t.Millisecs());
	return;
}
