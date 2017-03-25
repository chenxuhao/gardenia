#include "scc.h"
#include "timer.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stack>
#include <utility>
#include <string.h>
#include <set>
#include <vector>

#define BIT_SHIFT ((unsigned)1 << 31)

using namespace std;

class TR_vertex {
public:
	unsigned visit;
	unsigned low_link;
	TR_vertex() : visit(0), low_link(0) {}
	inline unsigned getVisited() { return visit; }
	inline void setVisited(unsigned n) { visit = n; }
	inline unsigned getLowLink() { return (low_link & 0x7FFFFFFF); }
	inline void setLowLink(unsigned n) { low_link = (n | (low_link & BIT_SHIFT)); }
	inline bool isInComponent () { return (low_link & BIT_SHIFT); }
	inline void setInComponentBit() { low_link = (low_link | BIT_SHIFT); };
	inline void clearInComponentBit() { low_link = (low_link & ~BIT_SHIFT); };
};

class TR_stack_vertex {
public:
    unsigned id;
    unsigned from;
    TR_stack_vertex() : id(0), from(0) {}
    inline unsigned getId() { return (id & 0x7FFFFFFF); }
    inline void setId(unsigned n) { id = (n | (id & BIT_SHIFT)); }
    inline bool isExpanded () { return (id & BIT_SHIFT); }
    inline void setExpandedBit() { id = (id | BIT_SHIFT); };
    inline void clearExpandedBit() { id = (id & ~BIT_SHIFT); };
    inline unsigned getFrom() { return from; }
    inline void setFrom(unsigned n) { from = n ; }
};

void tarjan_scc(unsigned m, unsigned *row_offsets, unsigned *column_indices, int*scc_dist, int *scc_root, bool *is_trivial, unsigned &num_trivial, unsigned &num_nontrivial, unsigned &total_num_scc, unsigned &biggest_scc_size) {
	bool terminated = false;
	stack<TR_stack_vertex> visit_stack;
	stack<unsigned> scc_stack;
	unsigned scc_top;
	unsigned time = 1; // 0 is null vertex
	TR_stack_vertex stack_vertex;
	TR_vertex *mx = new TR_vertex[ m + 1 ];

	//first initial states
	stack_vertex.setId(1);
	visit_stack.push(stack_vertex);
	unsigned i = 1;
	do {
		while ( !(visit_stack.empty()) ) {
			stack_vertex = visit_stack.top();
			visit_stack.pop();
			if ( ! stack_vertex.isExpanded() ) {
				if (mx[stack_vertex.getId()].getVisited() == 0) {//states hasn't been visited during DFS yet
					mx[stack_vertex.getId()].setVisited(time);
					mx[stack_vertex.getId()].setLowLink(time);
					time++;
					scc_stack.push(stack_vertex.getId());
					stack_vertex.setExpandedBit();
					visit_stack.push(stack_vertex);
					for ( unsigned column = row_offsets[ stack_vertex.getId() ]; column < row_offsets[ stack_vertex.getId() + 1 ]; column++ ) {
						TR_stack_vertex succ_stack_vertex;
						succ_stack_vertex.setId(column_indices[column]);
						succ_stack_vertex.setFrom(stack_vertex.getId());
						visit_stack.push(succ_stack_vertex);
					}
				}
				else {
					if ( ! mx[stack_vertex.getId()].isInComponent() ) {
						if ( mx[stack_vertex.getFrom()].getLowLink() > mx[stack_vertex.getId()].getVisited() ) {
							mx[stack_vertex.getFrom()].setLowLink(mx[stack_vertex.getId()].getVisited());
						}
					}
				}
			}
			else {
				if ( ( mx[stack_vertex.getId()].getVisited() == mx[stack_vertex.getId()].getLowLink() ) &&
					( ! mx[stack_vertex.getId()].isInComponent() ) ) {
					unsigned scc_size = 0;
					do {
						scc_top = scc_stack.top();
						scc_root[scc_top] = stack_vertex.getId();
						scc_stack.pop();
						mx[scc_top].setInComponentBit();
						scc_size ++;
					} while ( scc_top != stack_vertex.getId() );
					total_num_scc ++;
					if (scc_size > biggest_scc_size)
						biggest_scc_size = scc_size;
					if (scc_size == 1) {
						num_trivial ++;
						is_trivial[scc_top] = 1;
					}
					else {
						num_nontrivial ++;
					}
				}// second condition due to initial states
				if ( ( ! mx[stack_vertex.getId()].isInComponent() ) && ( stack_vertex.getFrom() != 0 ) ) {
					if ( mx[stack_vertex.getFrom()].getLowLink() > mx[stack_vertex.getId()].getLowLink() ) {
						mx[stack_vertex.getFrom()].setLowLink( mx[stack_vertex.getId()].getLowLink());
					}
				}
			}
		}
		terminated = true;
		for (; i <= m; i++ ) { //orginally i < m - 1, but it was wrong
			if ( mx[i].getVisited() == 0 ) {
				terminated = false;
				TR_stack_vertex stack_vertex;
				stack_vertex.setId(i);
				visit_stack.push(stack_vertex); 
				break;
			}
		}
	} while ( !terminated );
}

void SCCVerifier(int m, int nnz, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, int *scc_root_test) {
	printf("Verifying...\n");
	unsigned *row_offsets = (unsigned *)malloc((m + 2) * sizeof(unsigned));
	unsigned *column_indices = (unsigned *)malloc(nnz * sizeof(unsigned));
	for (int i = 0; i < m+1; i++) row_offsets[i+1] = out_row_offsets[i];
	for (int i = 0; i < nnz; i++) column_indices[i] = out_column_indices[i] + 1;
	bool *is_trivial = (bool *)malloc((m + 1) * sizeof(int));
	for (int i = 0; i < m+1; i ++) is_trivial[i] = 0;
	int *scc_root = (int *)malloc((m + 1) * sizeof(int));
	int *scc_dist = (int *)malloc((m + 1) * sizeof(int));
	memset(scc_dist, 0, (m + 1) * sizeof(int));
	unsigned num_trivial = 0;
	unsigned num_nontrivial = 0;
	unsigned total_num_scc = 0;
	unsigned biggest_scc_size = 0;

	Timer t;
	t.Start();
	tarjan_scc((unsigned)m, row_offsets, column_indices, scc_dist, scc_root, is_trivial, num_trivial, num_nontrivial, total_num_scc, biggest_scc_size);
	t.Stop();
	printf("\tnum_trivial=%d, num_nontrivial=%d, total_num_scc=%d, biggest_scc_size=%d\n", num_trivial, num_nontrivial, total_num_scc, biggest_scc_size);
	printf("\truntime [verify] = %f ms.\n", t.Millisecs());
	printf("Correct\n");
	return;
}

void write_solution(int m, int *scc_dist, int *scc_root, bool *is_trivial) {
	FILE *fp_trivial;
	fp_trivial = fopen("tarjan-trivial.txt", "w");
	for(int i = 1; i <= m; i ++) {
		fprintf(fp_trivial, "v%d %d\n", i, is_trivial[i]?1:0);
	}   
	fclose(fp_trivial);

	set<int> trivial_set;
	int *indices = (int *)malloc((1 + m) * sizeof(int));
	int count = 0;
	vector<set<int> > vs(1 + m);
	printf("vector init done!\n");
	for (int i = 1; i <= m; i++) {
		vs[scc_root[i]].insert(i);
		if (scc_root[i] == i) {
			indices[++count] = i;
		}
	}
	printf("count=%d\n", count);	
	set<int>::iterator site;
	int sum = 0;

	FILE *fp_scc = fopen("tarjan_scc.txt", "w");
	for (int i = 1; i <= count; i++) {
		site = vs[indices[i]].begin();
		while (site != vs[indices[i]].end()) {
			fprintf(fp_scc, "%d ", *site);
			site++;
		}
		fprintf(fp_scc, "\n");
		sum += vs[indices[i]].size();
		scc_dist[vs[indices[i]].size()]++;
	}
	fclose(fp_scc);

	FILE *fp_dist;
	fp_dist = fopen("tarjan_dist.txt", "w");
	printf("sum=%d\n", sum);
	for (int i = 1; i <= m; i++) {
		if (scc_dist[i] != 0)
			fprintf(fp_dist, "%d %d\n", i, scc_dist[i]);
	}
	fclose(fp_dist);
}
