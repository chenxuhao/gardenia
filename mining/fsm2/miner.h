
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <algorithm>
#include <graph_types.hpp>

class Miner {
public:
	Miner(const Graph &g, unsigned minsup) : graph(g), counter(0), minimal_support(minsup) {}
	size_t get_count() { return counter; }
	void grow(Projected_map3 root) {
		int total_single_edge_code = 0;
		for(Projected_iterator3 fromlabel = root.begin(); fromlabel != root.end(); ++fromlabel) {
			for(Projected_iterator2 elabel = fromlabel->second.begin(); elabel != fromlabel->second.end(); ++elabel) {
				for(Projected_iterator1 tolabel = elabel->second.begin(); tolabel != elabel->second.end(); ++tolabel) {
					// Build the initial two-node graph. It will be grown recursively within project.
					DFS_CODE.push(0, 1, fromlabel->first, elabel->first, tolabel->first);
					project(tolabel->second);
					DFS_CODE.pop();
					total_single_edge_code++;
				} // for tolabel
			} // for elabel
		} // for fromlabel
		std::cout << "Total single edge DFS code " << total_single_edge_code << std::endl;
	}

protected:
	Graph graph;
	size_t counter;
	int minimal_support;
	Graph GRAPH_IS_MIN;
	DFSCode DFS_CODE;
	DFSCode DFS_CODE_IS_MIN;

	bool is_min() {
		if(DFS_CODE.size() == 1) {
			return (true);
		}
		DFS_CODE.toGraph(GRAPH_IS_MIN);
		DFS_CODE_IS_MIN.clear();
		Projected_map3 root;
		EdgeList edges;

		for(unsigned from = 0; from < GRAPH_IS_MIN.size(); ++from) {
			if(get_forward_root(GRAPH_IS_MIN, GRAPH_IS_MIN[from], edges)) {
				for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
					root[GRAPH_IS_MIN[from].label][(*it)->elabel][GRAPH_IS_MIN[(*it)->to].label].push(0, *it, 0);
				} // for it
			} // if get_forward_root
		} // for from

		Projected_iterator3 fromlabel = root.begin();
		Projected_iterator2 elabel = fromlabel->second.begin();
		Projected_iterator1 tolabel = elabel->second.begin();
		DFS_CODE_IS_MIN.push(0, 1, fromlabel->first, elabel->first, tolabel->first);
		return (project_is_min(tolabel->second));
	}

	bool project_is_min(Projected &projected) {
		const RMPath& rmpath = DFS_CODE_IS_MIN.buildRMPath();
		int minlabel         = DFS_CODE_IS_MIN[0].fromlabel;
		int maxtoc           = DFS_CODE_IS_MIN[rmpath[0]].to;

		// SUBBLOCK 1
		{
			Projected_map1 root;
			bool flg = false;
			int newto = 0;

			for(int i = rmpath.size() - 1; !flg  && i >= 1; --i) {
				for(unsigned n = 0; n < projected.size(); ++n) {
					PDFS *cur = &projected[n];
					History history(GRAPH_IS_MIN, cur);
					Edge *e = get_backward(GRAPH_IS_MIN, history[rmpath[i]], history[rmpath[0]], history);
					if(e) {
						root[e->elabel].push(0, e, cur);
						newto = DFS_CODE_IS_MIN[rmpath[i]].from;
						flg = true;
					} // if e
				} // for n
			} // for i

			if(flg) {
				Projected_iterator1 elabel = root.begin();
				DFS_CODE_IS_MIN.push(maxtoc, newto, -1, elabel->first, -1);
				if(DFS_CODE[DFS_CODE_IS_MIN.size() - 1] != DFS_CODE_IS_MIN [DFS_CODE_IS_MIN.size() - 1]) return false;
				return project_is_min(elabel->second);
			}
		} // SUBBLOCK 1

		// SUBBLOCK 2
		{
			bool flg = false;
			int newfrom = 0;
			Projected_map2 root;
			EdgeList edges;

			for(unsigned n = 0; n < projected.size(); ++n) {
				PDFS *cur = &projected[n];
				History history(GRAPH_IS_MIN, cur);
				if(get_forward_pure(GRAPH_IS_MIN, history[rmpath[0]], minlabel, history, edges)) {
					flg = true;
					newfrom = maxtoc;
					for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it)
						root[(*it)->elabel][GRAPH_IS_MIN[(*it)->to].label].push(0, *it, cur);
				} // if get_forward_pure
			} // for n

			for(int i = 0; !flg && i < (int)rmpath.size(); ++i) {
				for(unsigned n = 0; n < projected.size(); ++n) {
					PDFS *cur = &projected[n];
					History history(GRAPH_IS_MIN, cur);
					if(get_forward_rmpath(GRAPH_IS_MIN, history[rmpath[i]], minlabel, history, edges)) {
						flg = true;
						newfrom = DFS_CODE_IS_MIN[rmpath[i]].from;
						for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it)
							root[(*it)->elabel][GRAPH_IS_MIN[(*it)->to].label].push(0, *it, cur);
					} // if get_forward_rmpath
				} // for n
			} // for i

			if(flg) {
				Projected_iterator2 elabel  = root.begin();
				Projected_iterator1 tolabel = elabel->second.begin();
				DFS_CODE_IS_MIN.push(newfrom, maxtoc + 1, -1, elabel->first, tolabel->first);
				if(DFS_CODE[DFS_CODE_IS_MIN.size() - 1] != DFS_CODE_IS_MIN [DFS_CODE_IS_MIN.size() - 1]) return false;
				return project_is_min(tolabel->second);
			} // if(flg)
		} // SUBBLOCK 2
		return true;
	} // graph_miner::project_is_min

	virtual unsigned support(Projected &projected) {
		Map2D node_id_counts;
		for(Projected::iterator cur = projected.begin(); cur != projected.end(); ++cur) {
			PDFS *em = &(*cur);
			int dfsindex = DFS_CODE.size() - 1;
			while(em) {
				if(DFS_CODE[dfsindex].to > DFS_CODE[dfsindex].from)    //forward edge
					node_id_counts[DFS_CODE[dfsindex].to][em->edge->to]++;
				if(!em->prev)
					node_id_counts[DFS_CODE[dfsindex].from][em->edge->from]++;
				em = em->prev;
				dfsindex--;
			}
		}
		unsigned min = 0xffffffff;
		for(Map2D::iterator it = node_id_counts.begin(); it != node_id_counts.end(); it++) {
			if((it->second).size() < min)
				min = (it->second).size();
		}
		return min;
	}

	void project(Projected &projected) {
		unsigned sup = support(projected);
		if(sup < minimal_support) return;
		if(is_min() == false) return;
		counter ++;
		std::cout << "DFSCode: " << DFS_CODE << ":" << sup << std::endl;
		const RMPath &rmpath = DFS_CODE.buildRMPath();
		int minlabel = DFS_CODE[0].fromlabel;
		int maxtoc = DFS_CODE[rmpath[0]].to;
		Projected_map3 new_fwd_root;
		Projected_map2 new_bck_root;
		EdgeList edges;
		for(unsigned n = 0; n < projected.size(); ++n) {
			unsigned id = projected[n].id;
			PDFS *cur = &projected[n];
			History history(graph, cur);
			// backward
			for(int i = (int)rmpath.size() - 1; i >= 1; --i) {
				Edge *e = get_backward(graph, history[rmpath[i]], history[rmpath[0]], history);
				if(e) new_bck_root[DFS_CODE[rmpath[i]].from][e->elabel].push(id, e, cur);
			}
			// pure forward
			if(get_forward_pure(graph, history[rmpath[0]], minlabel, history, edges)) {
				for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
					new_fwd_root[maxtoc][(*it)->elabel][graph[(*it)->to].label].push(id, *it, cur);
				}
			}
			// backtracked forward
			for(int i = 0; i < (int)rmpath.size(); ++i) {
				if(get_forward_rmpath(graph, history[rmpath[i]], minlabel, history, edges)) {
					for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
						new_fwd_root[DFS_CODE[rmpath[i]].from][(*it)->elabel][graph[(*it)->to].label].push(id, *it, cur);
					} // for it
				} // if
			} // for i
		} // for n
		// Test all extended substructures.
		// backward
		for(Projected_iterator2 to = new_bck_root.begin(); to != new_bck_root.end(); ++to) {
			for(Projected_iterator1 elabel = to->second.begin(); elabel != to->second.end(); ++elabel) {
				DFS_CODE.push(maxtoc, to->first, -1, elabel->first, -1);
				project(elabel->second);
				DFS_CODE.pop();
			}
		}
		// forward
		for(Projected_riterator3 from = new_fwd_root.rbegin(); from != new_fwd_root.rend(); ++from) {
			for(Projected_iterator2 elabel = from->second.begin(); elabel != from->second.end(); ++elabel) {
				for(Projected_iterator1 tolabel = elabel->second.begin(); tolabel != elabel->second.end(); ++tolabel) {
					DFS_CODE.push(from->first, maxtoc + 1, -1, elabel->first, tolabel->first);
					project(tolabel->second);
					DFS_CODE.pop();
				}
			}
		}
		return;
	}
};

