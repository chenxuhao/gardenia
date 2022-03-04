
#include <omp.h>
#include <deque>
#include <lb.hpp>
#include <math.h>
#include <cstdio>
#include <unistd.h>
#include <stdlib.h>
#include <iterator>
#include <iostream>

#define ENABLE_LB

class Miner_omp : protected lb {
//class Miner_omp {
protected:
	Graph graph;
	int minimal_support;
	unsigned max_level;
	int nthreads;
	bool computation_end; // cxh
	int task_split_threshold;
	std::vector<bool> thread_is_working;
	std::vector<int> frequent_patterns_count;
	std::vector<int> embeddings_regeneration_level;       //for each thread
	std::vector<std::vector<std::deque<DFS> > > dfs_task_queue;       //keep the sibling extensions for each level and for each thread
	std::vector<std::vector<std::deque<DFS> > > dfs_task_queue_shared;       //keep a separate queue for sharing work

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

	virtual unsigned support(Projected &projected, Thread_private_data &gprv) {
		Map2D node_id_counts;
		//int thread_id = gprv.thread_id; //omp_get_thread_num();
		//iterated through the all the embeddings
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
		unsigned min = 0xffffffff;
		for(Map2D::iterator it = node_id_counts.begin(); it != node_id_counts.end(); it++) {
			if((it->second).size() < min)
				min = (it->second).size();
		}
		if(min == 0xffffffff) min = 0;
		return min;
	}

	///* dynamic load balancing methods
	//virtual bool working(){ return is_working; }
	virtual void thread_start_working() {
		int thread_id = omp_get_thread_num();
		omp_set_lock(&lock);
		thread_is_working[thread_id] = true;
		omp_unset_lock(&lock);
	}
	virtual bool can_thread_split_work(Thread_private_data &gprv) {
		if(!thread_working(gprv)) return false;
		//int thread_id = gprv.thread_id;
		gprv.task_split_level = 0; // start search from level 0 task queue
		while(gprv.task_split_level < gprv.current_dfs_level && gprv.dfs_task_queue[gprv.task_split_level].size() < task_split_threshold )
			gprv.task_split_level++;
		if( gprv.dfs_task_queue.size() > gprv.task_split_level && gprv.dfs_task_queue[gprv.task_split_level].size() >= task_split_threshold )
			return true;
		return false;
	}
	virtual void thread_split_work(int requesting_thread, int &length, Thread_private_data &gprv) {
		for(int i = 0; i < gprv.task_split_level; i++) {
			if(dfs_task_queue_shared[requesting_thread].size() < (i + 1) ) {
				std::deque<DFS> tmp;
				dfs_task_queue_shared[requesting_thread].push_back(tmp);
			}
			dfs_task_queue_shared[requesting_thread][i].push_back(gprv.DFS_CODE[i]);
		}
		if(dfs_task_queue_shared[requesting_thread].size() < ( gprv.task_split_level + 1) ) {
			std::deque<DFS> tmp;
			dfs_task_queue_shared[requesting_thread].push_back(tmp);
		}
		int num_dfs = gprv.dfs_task_queue[gprv.task_split_level].size() / 2;
		for(int j = 0; j < num_dfs; j++) {
			DFS dfs = gprv.dfs_task_queue[gprv.task_split_level].back();
			dfs_task_queue_shared[requesting_thread][gprv.task_split_level].push_front(dfs);
			gprv.dfs_task_queue[gprv.task_split_level].pop_back();
		}
		embeddings_regeneration_level[requesting_thread] = gprv.task_split_level;
		length = num_dfs;
	}
	virtual void thread_process_received_data(Thread_private_data &gprv) {
		int thread_id = gprv.thread_id;
		gprv.embeddings_regeneration_level = embeddings_regeneration_level[thread_id];
		int num_dfs = dfs_task_queue_shared[thread_id][gprv.embeddings_regeneration_level].size();
		for(int i = 0; i < gprv.embeddings_regeneration_level; i++) {
			if(gprv.dfs_task_queue.size() < (i + 1) ) {
				std::deque<DFS> tmp;
				gprv.dfs_task_queue.push_back(tmp);
			}
			DFS dfs = dfs_task_queue_shared[thread_id][i].back();
			gprv.dfs_task_queue[i].push_back(dfs);
			dfs_task_queue_shared[thread_id][i].pop_back();
		}
		if(gprv.dfs_task_queue.size() < ( gprv.embeddings_regeneration_level + 1) ) {
			std::deque<DFS> tmp;
			gprv.dfs_task_queue.push_back(tmp);
		}
		for(int j = 0; j < num_dfs; j++) {
			DFS dfs = dfs_task_queue_shared[thread_id][gprv.embeddings_regeneration_level].back();
			gprv.dfs_task_queue[gprv.embeddings_regeneration_level].push_front(dfs);
			dfs_task_queue_shared[thread_id][gprv.embeddings_regeneration_level].pop_back();
		}
	}
	//virtual void thread_split_global_work(int requester_rank_id, Thread_private_data &gprv) {}
	//virtual void initiate_global_split_work(int requester_rank_id, Thread_private_data &gprv) {}
	//virtual void complete_global_split_work(int requester_id, Thread_private_data &gprv) {} //process load balancing
	//virtual void process_received_data(int* buffer, int size, Thread_private_data &gprv) {}
	//virtual void set_load_balance_interval(int i) {}
	//virtual void complete_global_work_split_request(int requester_rank_id, Thread_private_data &gprv); //threads load balancing
//*/
public:
	Miner_omp(const Graph &g, int num_threads, int minsup, unsigned k = 2) {
		graph = g;
		minimal_support = minsup;
		max_level = k;
		//numtasks = 1;
		computation_end = false; // cxh
		task_split_threshold = 2;
		//is_working = true;
		//donor_thread = 0;
		nthreads = num_threads;
		std::cout << "num_threads = " << nthreads << std::endl;
		for(int i = 0; i<nthreads; i++) {
			//current_dfs_level.push_back(0);
			//DFSCode dfscode;
			//DFS_CODE_V.push_back(dfscode);
			//DFS_CODE_IS_MIN_V.push_back(dfscode);
			//Graph gr;
			//GRAPH_IS_MIN_V.push_back(gr);
			frequent_patterns_count.push_back(0);
			std::vector<std::deque<DFS> > tmp;
			dfs_task_queue.push_back(tmp);
			dfs_task_queue_shared.push_back(tmp);
			thread_is_working.push_back(false);
			embeddings_regeneration_level.push_back(0);
		}
		init_lb(nthreads);
	}
	virtual ~Miner_omp() {}
	int get_count() {
		int total = 0;
		for(int i = 0; i<nthreads; i++)
			total += frequent_patterns_count[i];
		return total;
	}
	void set_regen_level(int tid, int val) {
		embeddings_regeneration_level[tid] = val;
	}
	void task_schedule(Thread_private_data &gprv) {
		threads_load_balance(gprv);
	}
	void activate_thread(int thread_id) {
		omp_set_lock(&lock);
		thread_is_working[thread_id] = true;
		omp_unset_lock(&lock);
	}
	void deactivate_thread(int thread_id) {
		omp_set_lock(&lock);
		thread_is_working[thread_id] = false;
		omp_unset_lock(&lock);
	}
	virtual bool all_threads_idle() {
		bool all_idle = true;
		omp_set_lock(&lock);
		for(int i = 0; i< num_threads; i++) {
			if(thread_is_working[i] == true) {
				all_idle = false;
				break;
			}
		}
		omp_unset_lock(&lock);
		return all_idle;
	}
	virtual bool thread_working(Thread_private_data &gprv) {
		int thread_id = gprv.thread_id;       //omp_get_thread_num();
		bool th_is_working;
		omp_set_lock(&lock); // cxh?
		th_is_working = thread_is_working[thread_id];
		omp_unset_lock(&lock); // cxh?
		return th_is_working;
	}
	void project(Projected &projected, int dfs_level, Thread_private_data &gprv) {
		unsigned sup = support(projected, gprv);
		if(sup < minimal_support) return;
		if(is_min(gprv) == false) { return; } 
		int thread_id = gprv.thread_id; //omp_get_thread_num();
		frequent_patterns_count[thread_id]++;
		if (dfs_level == max_level) return;
		const RMPath &rmpath = gprv.DFS_CODE.buildRMPath();
		int minlabel = gprv.DFS_CODE[0].fromlabel;
		int maxtoc = gprv.DFS_CODE[rmpath[0]].to;
		Projected_map3 new_fwd_root;
		Projected_map2 new_bck_root;
		EdgeList edges;
		gprv.current_dfs_level = dfs_level;
		// Enumerate all possible one edge extensions of the current substructure.
		for(unsigned int n = 0; n < projected.size(); ++n) {
			unsigned int id = projected[n].id;
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
			#ifdef ENABLE_LB
			if(num_threads > 1) threads_load_balance(gprv);
			#endif
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
	}

	void regenerate_embeddings(Projected &projected, int dfs_level, Thread_private_data &gprv) {
		//int thread_id = gprv.thread_id;
		for(int i = 0; gprv.dfs_task_queue[dfs_level].size() > 0; i++) {
			gprv.current_dfs_level = dfs_level;
			#ifdef ENABLE_LB
			if(num_threads > 1) threads_load_balance(gprv);
			#endif
			DFS dfs = gprv.dfs_task_queue[dfs_level].front();
			gprv.dfs_task_queue[dfs_level].pop_front();
			gprv.DFS_CODE.push(dfs.from, dfs.to, dfs.fromlabel, dfs.elabel, dfs.tolabel);
			Projected new_root;
			for(unsigned n = 0; n < projected.size(); ++n) {
				unsigned id = projected[n].id;
				PDFS *cur = &projected[n];
				History history(graph, cur);
				if(dfs.is_backward() ) {
					Edge *e = get_backward(graph, gprv.DFS_CODE, history);
					if(e)
						new_root.push(id, e, cur);
				} else {
					EdgeList edges;
					if(get_forward(graph, gprv.DFS_CODE, history, edges)) {
						for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
							new_root.push(id, *it, cur);
						}
					}
				}
			}
			if( gprv.embeddings_regeneration_level > dfs_level ) {
				regenerate_embeddings(new_root, dfs_level + 1, gprv);
			} else project(new_root, dfs_level + 1, gprv);
			gprv.DFS_CODE.pop();
		}
	}
};

