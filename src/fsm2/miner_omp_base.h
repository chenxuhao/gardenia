// Copyright 2019, University of Texas at Austin
// Authors: Xuhao Chen <cxh@utexas.edu>

#include <map>
#include <set>
#include <omp.h>
#include <deque>
#include <vector>
#include <math.h>
#include <cstdio>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <graph_types.hpp>

class Miner_omp {
public:
	Miner_omp(int num_threads, int minsup, unsigned k = 2) {
		if(num_threads > omp_get_max_threads())
			nthreads = omp_get_max_threads();
		else nthreads = num_threads;
		minimal_support = minsup;
		max_level = k;
		for(int i = 0; i < nthreads; i++) {
			frequent_patterns_count.push_back(0);
			current_dfs_level.push_back(0);
			DFSCode dfscode;
			DFS_CODE_V.push_back(dfscode);
			DFS_CODE_IS_MIN_V.push_back(dfscode);
			Graph gr;
			GRAPH_IS_MIN_V.push_back(gr);
			std::vector<std::deque<DFS> > tmp;
			dfs_task_queue.push_back(tmp);
			dfs_task_queue_shared.push_back(tmp);
		}
	}
	virtual ~Miner_omp() {}
	void set_graph(const Graph &g) { this->graph = g; }
	size_t get_count() {
		size_t total = 0;
		for(int i = 0; i < nthreads; i++)
			total += frequent_patterns_count[i];
		return total;
	}
	void set_num_threads(int t) { omp_set_num_threads(t); }

	void project(Projected &projected, int dfs_level, Thread_private_data &gprv) {
		unsigned sup = support(projected, gprv);
		if(sup < minimal_support) return;
		int thread_id = gprv.thread_id; //omp_get_thread_num();
		if(is_min(gprv) == false) return;
		frequent_patterns_count[thread_id]++;
		if (dfs_level == max_level) return;
		//std::cout << "DFSCode: " << gprv.DFS_CODE << ":" << sup << std::endl;
		const RMPath &rmpath = gprv.DFS_CODE.buildRMPath();
		int minlabel = gprv.DFS_CODE[0].fromlabel;
		int maxtoc = gprv.DFS_CODE[rmpath[0]].to;
		Projected_map3 new_fwd_root;
		Projected_map2 new_bck_root;
		EdgeList edges;
		gprv.current_dfs_level = dfs_level;
		for(unsigned n = 0; n < projected.size(); ++n) {
			unsigned id = projected[n].id;
			PDFS *cur = &projected[n];
			History history(graph, cur);
			// backward
			for(int i = (int)rmpath.size() - 1; i >= 1; --i) {
				Edge *e = get_backward(graph, history[rmpath[i]], history[rmpath[0]], history);
				if(e) {
					new_bck_root[gprv.DFS_CODE[rmpath[i]].from][e->elabel].push(id, e, cur);
				}
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
						new_fwd_root[gprv.DFS_CODE[rmpath[i]].from][(*it)->elabel][graph[(*it)->to].label].push(id, *it, cur);
					} // for it
				} // if
			} // for i
		} // for n
		std::deque<DFS> tmp;
		if(gprv.dfs_task_queue.size() <= dfs_level) {
			gprv.dfs_task_queue.push_back(tmp);
		}
		// Test all extended substructures.
		// backward
		for(Projected_iterator2 to = new_bck_root.begin(); to != new_bck_root.end(); ++to) {
			for(Projected_iterator1 elabel = to->second.begin(); elabel != to->second.end(); ++elabel) {
				DFS dfs(maxtoc, to->first, -1, elabel->first, -1);
				gprv.dfs_task_queue[dfs_level].push_back(dfs);
			}
		}
		// forward
		for(Projected_riterator3 from = new_fwd_root.rbegin(); from != new_fwd_root.rend(); ++from) {
			for(Projected_iterator2 elabel = from->second.begin(); elabel != from->second.end(); ++elabel) {
				for(Projected_iterator1 tolabel = elabel->second.begin(); tolabel != elabel->second.end(); ++tolabel) {
					DFS dfs(from->first, maxtoc + 1, -1, elabel->first, tolabel->first);
					gprv.dfs_task_queue[dfs_level].push_back(dfs);
				}
			}
		}
		while(gprv.dfs_task_queue[dfs_level].size() > 0) {
			DFS dfs = gprv.dfs_task_queue[dfs_level].front();
			gprv.dfs_task_queue[dfs_level].pop_front();
			gprv.current_dfs_level = dfs_level;
			gprv.DFS_CODE.push(dfs.from, dfs.to, dfs.fromlabel, dfs.elabel, dfs.tolabel);
			if(dfs.is_backward())
				project(new_bck_root[dfs.to][dfs.elabel], dfs_level + 1, gprv);      //Projected (PDFS vector): each entry contains graph id 0, edge pointer, null PDFS
			else
				project(new_fwd_root[dfs.from][dfs.elabel][dfs.tolabel], dfs_level + 1, gprv);      //Projected (PDFS vector): each entry contains graph id 0, edge pointer, null PDFS
			gprv.DFS_CODE.pop();
		}
		return;
	}

protected:
	int nthreads;
	int minimal_support;
	unsigned max_level;
	long long counter;
	Graph graph;
	unsigned ID;
	bool directed;
	std::vector<int> frequent_patterns_count;
	std::vector<DFSCode> DFS_CODE_V;
	std::vector<DFSCode> DFS_CODE_IS_MIN_V;
	std::vector<Graph> GRAPH_IS_MIN_V;
	std::vector<std::vector<std::deque<DFS> > > dfs_task_queue;       //keep the sibling extensions for each level and for each thread
	std::vector<std::vector<std::deque<DFS> > > dfs_task_queue_shared;       //keep a separate queue for sharing work
	std::vector<std::vector<std::deque<DFS> > > global_task_queue; //shared queue for global work, one for each rank
	std::vector<int> current_dfs_level;                   //keep track of the level of the candidate tree for each thread

	//support function for a single large graph, computes the minimum count of a node in the embeddings
	virtual unsigned support(Projected &projected, Thread_private_data &gprv) {
		std::map<unsigned int, std::map<unsigned int, unsigned int> > node_id_counts;
		int thread_id = gprv.thread_id; //omp_get_thread_num();
		for(Projected::iterator cur = projected.begin(); cur != projected.end(); ++cur) {
			PDFS *em = &(*cur);
			int dfsindex = gprv.DFS_CODE.size() - 1;
			while(em != NULL) {
				if(gprv.DFS_CODE[dfsindex].to > gprv.DFS_CODE[dfsindex].from) {    //forward edge
					node_id_counts[gprv.DFS_CODE[dfsindex].to][em->edge->to]++;
				}
				if(!em->prev) {
					node_id_counts[gprv.DFS_CODE[dfsindex].from][em->edge->from]++;
				}
				em = em->prev;
				dfsindex--;
			}
		}
		unsigned int min = 0xffffffff;
		for(std::map<unsigned int, std::map<unsigned int, unsigned int> >::iterator it = node_id_counts.begin(); it != node_id_counts.end(); it++) {
			if((it->second).size() < min)
				min = (it->second).size();
		}
		if(min == 0xffffffff) min = 0;
		return min;
	}

	bool is_min(Thread_private_data &gprv) {
		if(gprv.DFS_CODE.size() == 1) {
			return (true);
		}
		gprv.DFS_CODE.toGraph(gprv.GRAPH_IS_MIN);
		gprv.DFS_CODE_IS_MIN.clear();
		Projected_map3 root;
		EdgeList edges;
		for(unsigned int from = 0; from < gprv.GRAPH_IS_MIN.size(); ++from) {
			if(get_forward_root(gprv.GRAPH_IS_MIN, gprv.GRAPH_IS_MIN[from], edges)) {
				for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
					root[gprv.GRAPH_IS_MIN[from].label][(*it)->elabel][gprv.GRAPH_IS_MIN[(*it)->to].label].push(0, *it, 0);
				} // for it
			} // if get_forward_root
		} // for from
		Projected_iterator3 fromlabel = root.begin();
		Projected_iterator2 elabel = fromlabel->second.begin();
		Projected_iterator1 tolabel = elabel->second.begin();
		gprv.DFS_CODE_IS_MIN.push(0, 1, fromlabel->first, elabel->first, tolabel->first);
		return (project_is_min(gprv,tolabel->second));
	}

	bool project_is_min(Thread_private_data &gprv, Projected &projected) {
		const RMPath& rmpath = gprv.DFS_CODE_IS_MIN.buildRMPath();
		int minlabel         = gprv.DFS_CODE_IS_MIN[0].fromlabel;
		int maxtoc           = gprv.DFS_CODE_IS_MIN[rmpath[0]].to;

		// SUBBLOCK 1
		{
			Projected_map1 root;
			bool flg = false;
			int newto = 0;

			for(int i = rmpath.size() - 1; !flg  && i >= 1; --i) {
				for(unsigned int n = 0; n < projected.size(); ++n) {
					PDFS *cur = &projected[n];
					History history(gprv.GRAPH_IS_MIN, cur);
					Edge *e = get_backward(gprv.GRAPH_IS_MIN, history[rmpath[i]], history[rmpath[0]], history);
					if(e) {
						root[e->elabel].push(0, e, cur);
						newto = gprv.DFS_CODE_IS_MIN[rmpath[i]].from;
						flg = true;
					} // if e
				} // for n
			} // for i

			if(flg) {
				Projected_iterator1 elabel = root.begin();
				gprv.DFS_CODE_IS_MIN.push(maxtoc, newto, -1, elabel->first, -1);
				if(gprv.DFS_CODE[gprv.DFS_CODE_IS_MIN.size() - 1] != gprv.DFS_CODE_IS_MIN[gprv.DFS_CODE_IS_MIN.size() - 1]) return false;
				return project_is_min(gprv, elabel->second);
			}
		} // SUBBLOCK 1

		// SUBBLOCK 2
		{
			bool flg = false;
			int newfrom = 0;
			Projected_map2 root;
			EdgeList edges;

			for(unsigned int n = 0; n < projected.size(); ++n) {
				PDFS *cur = &projected[n];
				History history(gprv.GRAPH_IS_MIN, cur);
				if(get_forward_pure(gprv.GRAPH_IS_MIN, history[rmpath[0]], minlabel, history, edges)) {
					flg = true;
					newfrom = maxtoc;
					for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it)
						root[(*it)->elabel][gprv.GRAPH_IS_MIN[(*it)->to].label].push(0, *it, cur);
				} // if get_forward_pure
			} // for n
			for(int i = 0; !flg && i < (int)rmpath.size(); ++i) {
				for(unsigned int n = 0; n < projected.size(); ++n) {
					PDFS *cur = &projected[n];
					History history(gprv.GRAPH_IS_MIN, cur);
					if(get_forward_rmpath(gprv.GRAPH_IS_MIN, history[rmpath[i]], minlabel, history, edges)) {
						flg = true;
						newfrom = gprv.DFS_CODE_IS_MIN[rmpath[i]].from;
						for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it)
							root[(*it)->elabel][gprv.GRAPH_IS_MIN[(*it)->to].label].push(0, *it, cur);
					} // if get_forward_rmpath
				} // for n
			} // for i

			if(flg) {
				Projected_iterator2 elabel  = root.begin();
				Projected_iterator1 tolabel = elabel->second.begin();
				gprv.DFS_CODE_IS_MIN.push(newfrom, maxtoc + 1, -1, elabel->first, tolabel->first);
				if(gprv.DFS_CODE[gprv.DFS_CODE_IS_MIN.size() - 1] != gprv.DFS_CODE_IS_MIN[gprv.DFS_CODE_IS_MIN.size() - 1]) return false;
				return project_is_min(gprv, tolabel->second);
			} // if(flg)
		} // SUBBLOCK 2
		return true;
	} // graph_miner::project_is_min
};

