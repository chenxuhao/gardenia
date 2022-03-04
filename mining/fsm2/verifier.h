
#include <map>
#include <set>
#include <cstdio>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <iterator>
#include <algorithm>
#include <embedding.h>
/*
using types::Edge;
using types::Map2D;
*/
class Verifier {
public:
	Verifier(const Graph &g): counter(0), graph(g) {}
	size_t get_count() { return counter; }
	//void set_graph(Graph &g) { this->graph = g; }
	//virtual void set_min_support(int minsup) { minimal_support = minsup; }
	//void report(Embeddings &, unsigned int);
	void grow(Embeddings_map3 &root, unsigned minsup) {
		int total_single_edge_code = 0;
		for (Embeddings_iterator3 fromlabel = root.begin(); fromlabel != root.end(); ++fromlabel) {
			for (Embeddings_iterator2 elabel = fromlabel->second.begin(); elabel != fromlabel->second.end(); ++elabel) {
				for (Embeddings_iterator1 tolabel = elabel->second.begin(); tolabel != elabel->second.end(); ++tolabel) {
					// Build the initial two-node graph.  It will be grownrecursively within project.
					DFS_CODE.push(0, 1, fromlabel->first, elabel->first, tolabel->first);
					project(tolabel->second, minsup); //Embeddings
					DFS_CODE.pop();
					total_single_edge_code++;
				} // for tolabel
			} // for elabel
		} // for fromlabel
		//std::cout << "Total single edge DFS code " << total_single_edge_code << std::endl;
	}

protected:
	size_t counter;
	Graph graph;
	DFSCode DFS_CODE;
	DFSCode DFS_CODE_IS_MIN;
	Graph GRAPH_IS_MIN;
	unsigned ID;
	bool directed;

	virtual unsigned support_count(Embeddings &embeddings) {
		Map2D node_id_counts;
		for(Embeddings::iterator cur = embeddings.begin(); cur != embeddings.end(); ++cur) {
			Emb *em = &(*cur);
			int dfsindex = DFS_CODE.size() - 1;
			while(em) {
				if(DFS_CODE[dfsindex].to > DFS_CODE[dfsindex].from)    //forward edge
					node_id_counts[DFS_CODE[dfsindex].to][(*em).edge.to]++;
				if(!em->prev)
					node_id_counts[DFS_CODE[dfsindex].from][(*em).edge.from]++;
				em = em->prev;
				dfsindex--;
			}
		}
		unsigned min = 0xffffffff;
		for(Map2D::iterator it = node_id_counts.begin(); it != node_id_counts.end(); it++) {
			if((it->second).size() < min) min = (it->second).size();
		}
		return min;
	}

	void project(Embeddings &embeddings, unsigned minsup) {
		// (1) Support Counting
		// Check if the pattern is frequent enough.
		unsigned sup = support_count(embeddings);
		if(sup < minsup) return;

		// (2) Pruing (Flitering)
		// The minimal DFS code check is more expensive than the support check,
		// hence it is done now, after checking the support.
		// The use of minDFS allows us to prune duplicate/isomorphic patterns.
		if(is_min() == false) { return; }
		// (3) Output the frequent substructure
		//report(embeddings, sup);
		counter ++;

		// (4) Edge Extension 
		// We just outputted a frequent subgraph.  As it is frequent enough, so
		// might be its (n+1)-extension-graphs, hence we enumerate them all.
		const RMPath &rmpath = DFS_CODE.buildRMPath();
		int minlabel = DFS_CODE[0].fromlabel;
		int maxtoc = DFS_CODE[rmpath[0]].to;
		Embeddings_map3 new_fwd_root;
		Embeddings_map2 new_bck_root;
		std::vector<Edge> edges;

		// Enumerate all possible one edge extensions of the current substructure.
		for(unsigned n = 0; n < embeddings.size(); ++n) {
			Emb *cur = &embeddings[n];
			EmbVector history(graph, cur);

			// backward
			for(int i = (int)rmpath.size() - 1; i >= 1; --i) {
				Edge e;
				if(get_backward(graph, history[rmpath[i]], history[rmpath[0]], history, e))
					new_bck_root[DFS_CODE[rmpath[i]].from][e.elabel].push(e, cur);
			}
			// pure forward
			// FIXME: here we pass a too large e->to (== history[rmpath[0]]->to
			// into get_forward_pure, such that the assertion fails.
			// The problem is: history[rmpath[0]]->to > graph.size()
			if(get_forward_pure(graph, history[rmpath[0]], minlabel, history, edges)) {
				for(std::vector<Edge>::iterator it = edges.begin(); it != edges.end(); ++it) {
					new_fwd_root[maxtoc][it->elabel][graph[it->to].label].push(*it, cur);
				}
			}
			// backtracked forward
			for(int i = 0; i < (int)rmpath.size(); ++i) {
				if(get_forward_rmpath(graph, history[rmpath[i]], minlabel, history, edges)) {
					for(std::vector<Edge>::iterator it = edges.begin(); it != edges.end(); ++it) {
						new_fwd_root[DFS_CODE[rmpath[i]].from][it->elabel][graph[it->to].label].push(*it, cur);
					} // for it
				} // if
			} // for i
		} // for n

		// Test all extended substructures.
		// backward
		for(Embeddings_iterator2 to = new_bck_root.begin(); to != new_bck_root.end(); ++to) {
			for(Embeddings_iterator1 elabel = to->second.begin(); elabel != to->second.end(); ++elabel) {
				DFS_CODE.push(maxtoc, to->first, -1, elabel->first, -1);
				project(elabel->second, minsup);
				DFS_CODE.pop();
			}
		}

		// forward
		for(Embeddings_riterator3 from = new_fwd_root.rbegin();
				from != new_fwd_root.rend(); ++from) {
			for(Embeddings_iterator2 elabel = from->second.begin();
					elabel != from->second.end(); ++elabel) {
				for(Embeddings_iterator1 tolabel = elabel->second.begin();
						tolabel != elabel->second.end(); ++tolabel) {
					DFS_CODE.push(from->first, maxtoc + 1, -1, elabel->first, tolabel->first);
					project(tolabel->second, minsup);
					DFS_CODE.pop();
				}
			}
		}
		return;
	}

	bool is_min() {
		if(DFS_CODE.size() == 1) return (true);
		DFS_CODE.toGraph(GRAPH_IS_MIN);
		DFS_CODE_IS_MIN.clear();
		Embeddings_map3 root;
		std::vector<Edge> edges;
		for(unsigned int from = 0; from < GRAPH_IS_MIN.size(); ++from) {
			if(get_forward_root(GRAPH_IS_MIN, GRAPH_IS_MIN[from], edges)) {
				for(std::vector<Edge>::iterator it = edges.begin(); it != edges.end(); ++it) {
					root[GRAPH_IS_MIN[from].label][it->elabel][GRAPH_IS_MIN[it->to].label].push(*it, 0);
				} // for it
			} // if get_forward_root
		} // for from
		Embeddings_iterator3 fromlabel = root.begin();
		Embeddings_iterator2 elabel = fromlabel->second.begin();
		Embeddings_iterator1 tolabel = elabel->second.begin();
		DFS_CODE_IS_MIN.push(0, 1, fromlabel->first, elabel->first, tolabel->first);
		return (project_is_min(tolabel->second));
	}

	bool project_is_min(Embeddings &projected) {
		const RMPath& rmpath = DFS_CODE_IS_MIN.buildRMPath();
		int minlabel = DFS_CODE_IS_MIN[0].fromlabel;
		int maxtoc = DFS_CODE_IS_MIN[rmpath[0]].to;

		// SUBBLOCK 1
		{
			Embeddings_map1 root;
			bool flg = false;
			int newto = 0;
			for(int i = rmpath.size() - 1; !flg  && i >= 1; --i) {
				for(unsigned int n = 0; n < projected.size(); ++n) {
					Emb *cur = &projected[n];
					EmbVector history(GRAPH_IS_MIN, cur);
					Edge e;
					if(get_backward(GRAPH_IS_MIN, history[rmpath[i]], history[rmpath[0]], history, e)) {
						root[e.elabel].push(e, cur);
						newto = DFS_CODE_IS_MIN[rmpath[i]].from;
						flg = true;
					} // if e
				} // for n
			} // for i

			if(flg) {
				Embeddings_iterator1 elabel = root.begin();
				DFS_CODE_IS_MIN.push(maxtoc, newto, -1, elabel->first, -1);
				if(DFS_CODE[DFS_CODE_IS_MIN.size() - 1] != DFS_CODE_IS_MIN [DFS_CODE_IS_MIN.size() - 1]) return false;
				return project_is_min(elabel->second);
			}
		} // SUBBLOCK 1

		// SUBBLOCK 2
		{
			bool flg = false;
			int newfrom = 0;
			Embeddings_map2 root;
			std::vector<Edge> edges;

			for(unsigned n = 0; n < projected.size(); ++n) {
				Emb *cur = &projected[n];
				EmbVector history(GRAPH_IS_MIN, cur);
				if(get_forward_pure(GRAPH_IS_MIN, history[rmpath[0]], minlabel, history, edges)) {
					flg = true;
					newfrom = maxtoc;
					for(std::vector<Edge>::iterator it = edges.begin(); it != edges.end(); ++it)
						root[it->elabel][GRAPH_IS_MIN[it->to].label].push(*it, cur);
				} // if get_forward_pure
			} // for n

			for(int i = 0; !flg && i < (int)rmpath.size(); ++i) {
				for(unsigned int n = 0; n < projected.size(); ++n) {
					Emb *cur = &projected[n];
					EmbVector history(GRAPH_IS_MIN, cur);
					if(get_forward_rmpath(GRAPH_IS_MIN, history[rmpath[i]], minlabel, history, edges)) {
						flg = true;
						newfrom = DFS_CODE_IS_MIN[rmpath[i]].from;
						for(std::vector<Edge>::iterator it = edges.begin(); it != edges.end(); ++it)
							root[it->elabel][GRAPH_IS_MIN[it->to].label].push( *it, cur);
					} // if get_forward_rmpath
				} // for n
			} // for i

			if(flg) {
				Embeddings_iterator2 elabel  = root.begin();
				Embeddings_iterator1 tolabel = elabel->second.begin();
				DFS_CODE_IS_MIN.push(newfrom, maxtoc + 1, -1, elabel->first, tolabel->first);
				if(DFS_CODE[DFS_CODE_IS_MIN.size() - 1] != DFS_CODE_IS_MIN [DFS_CODE_IS_MIN.size() - 1]) return false;
				return project_is_min(tolabel->second);
			} // if(flg)
		} // SUBBLOCK 2
		return true;
	} // project_is_min
};

