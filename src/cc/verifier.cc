// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "cc.h"
#include "timer.h"
#include <map>
#include <stack>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>
#include <unordered_map>

IndexT SampleFrequentElement(int m, IndexT *comp, int64_t num_samples) {
	// Sample elements from 'comp'
	std::unordered_map<IndexT, int> sample_counts(32);
	using kvp_type = std::unordered_map<IndexT, int>::value_type;
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

int serial_solver(Graph &g, CompT *components) {
	std::stack<int> DFS;
	int num_comps = 0;
	for (int src = 0; src < g.V(); src ++) {
		if (components[src] == -1) {
			DFS.push(src);
			components[src] = num_comps;
			while (!DFS.empty()) {
				int top = DFS.top();
				DFS.pop();
				for (auto dst : g.N(top)) {
					if (components[dst] == -1) {
						DFS.push(dst);
						components[dst] = num_comps;
					}
				}
			}
			num_comps ++;
		}
	}
	return num_comps;
}

// Verifies CC result by performing a BFS from a vertex in each component
// - Asserts search does not reach a vertex with a different component label
// - If the graph is directed, it performs the search as if it was undirected
// - Asserts every vertex is visited (degree-0 vertex should have own label)
void CCVerifier(Graph &g, CompT *comp_test) {
  auto m = g.V();
	CompT *comp = (CompT *)malloc(m * sizeof(CompT));
	for (int i = 0; i < m; i ++) comp[i] = -1;
	Timer t;
	t.Start();
	serial_solver(g, comp);
	t.Stop();
	
	printf("Verifying...\n");
	map<int, int> label_to_source;
	vector<bool> visited(m);
	vector<int> frontier;
	for (int i=0; i<m; i++) {
		visited[i] = false;
		label_to_source[comp_test[i]] = i;
	}
	frontier.reserve(m);
	map<int, int>::iterator label_source_pair;
	for (label_source_pair = label_to_source.begin(); label_source_pair != label_to_source.end(); label_source_pair ++) {
		int curr_label = label_source_pair->first;
		int source = label_source_pair->second;
		frontier.clear();
		frontier.push_back(source);
		visited[source] = true;
		vector<int>::iterator it;
		for (it = frontier.begin(); it != frontier.end(); it++) {
			int src = *it;
      for (auto dst : g.N(src)) {
				if (comp_test[dst] != curr_label) {
					printf("Wrong\n");
					return;
				}
				if (!visited[dst]) {
					visited[dst] = true;
					frontier.push_back(dst);
				}
			}
			if (g.is_directed()) {
        for (auto dst : g.N(src)) {
					if (comp_test[dst] != curr_label) {
						printf("Wrong\n");
						return;
					}
					if (!visited[dst]) {
						visited[dst] = true;
						frontier.push_back(dst);
					}
				}
			}
		} 
	}
	printf("\truntime [serial] = %f ms.\n", t.Millisecs());

	for (int n = 0; n < m; n ++) {
		if (!visited[n]) {
			printf("Wrong\n");
			return;
		}
	}
	printf("Correct\n");
	return;
}
