// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "cc.h"
#include <unordered_map>
#include <stack>
#include "timer.h"

int serial_solver(int m, int *row_offsets, int *column_indices, CompT *components) {
	std::stack<int> DFS;
	int num_comps = 0;
	for(int src = 0; src < m; src ++) {
		if(components[src] == -1) {
			DFS.push(src);
			components[src] = num_comps;
			while(!DFS.empty()) {
				int top = DFS.top();
				DFS.pop();
				for(int offset = row_offsets[top]; offset < row_offsets[top + 1]; offset ++) {
					int dst = column_indices[offset];
					if(components[dst] == -1) {
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
void CCVerifier(int m, int *row_offsets, int *column_indices, CompT *comp_test) {
	CompT *comp = (CompT *)malloc(m * sizeof(CompT));
	for (int i = 0; i < m; i ++) comp[i] = -1;
	Timer t;
	t.Start();
	serial_solver(m, row_offsets, column_indices, comp);
	t.Stop();
	
	printf("Verifying...\n");
	unordered_map<int, int> label_to_source;
	vector<bool> visited(m);
	vector<int> frontier;
	for (int i=0; i<m; i++) {
		visited[i] = false;
		label_to_source[comp_test[i]] = i;
	}
	frontier.reserve(m);
	for (auto label_source_pair : label_to_source) {
		int curr_label = label_source_pair.first;
		int source = label_source_pair.second;
		frontier.clear();
		frontier.push_back(source);
		visited[source] = true;
		for (auto it = frontier.begin(); it != frontier.end(); it++) {
			int src = *it;
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src + 1]; 
			for (int offset = row_begin; offset < row_end; ++ offset) {
				int dst = column_indices[offset];
				if (comp_test[dst] != curr_label) {
					printf("Wrong\n");
					return;
				}
				if (!visited[dst]) {
					visited[dst] = true;
					frontier.push_back(dst);
				}
			}
			/*
			if (is_directed()) {
				for (unsigned offset = row_begin; offset < row_end; ++ offset) {
					int dst = column_indices[offset];
					if (comp_test[dst] != curr_label)
						return false;
					if (!visited[dst]) {
						visited[dst] = true;
						frontier.push_back(dst);
					}
				}
			}
			*/
		}   
	}
	printf("\truntime[serial] = %f ms.\n", t.Millisecs());

	for (int n = 0; n < m; n ++) {
		if (!visited[n]) {
			printf("Wrong\n");
			return;
		}
	}
	printf("Correct\n");
	return;
}
