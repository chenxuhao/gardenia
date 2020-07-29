#ifndef __GRAPH_TYPES_HPP__
#define __GRAPH_TYPES_HPP__

#include <set>
#include <map>
#include <deque>
#include <string>
#include <cstring>
#include <cassert>
#include <sstream>
#include <iterator>
#include <iostream>
#include <stdexcept>
#include <algorithm>

#include <graph.hpp>
#include <types.hpp>
#include <dfs_code.hpp>

struct Thread_private_data {
	int thread_id;
	int task_split_level, embeddings_regeneration_level, current_dfs_level;
	//int frequent_patterns_count;
	bool is_running;
	DFSCode DFS_CODE;
	DFSCode DFS_CODE_IS_MIN;
	Graph GRAPH_IS_MIN;
	std::vector<std::deque<DFS> >  dfs_task_queue;
	std::deque<DFSCode> dfscodes_to_process;
};

std::ostream &operator<<(std::ostream &out, const DFSCode &code) {
  out << code.to_string();
  return out;
}

struct PDFS {
	unsigned id;      // ID of the original input graph
	Edge *edge;
	PDFS *prev;
	PDFS() : id(0), edge(0), prev(0) {};
	//std::string to_string() const;
	std::string to_string() const {
		std::stringstream ss;
		ss << "[" << id << "," << edge->to_string() << "]";
		return ss.str();
	}
};

// Stores information of edges/nodes that were already visited in the
// current DFS branch of the search.
class History : public std::vector<Edge*> {
private:
	std::set<int> edge;
	std::set<int> vertex;
public:
	History() {}
	History(const Graph &g, PDFS *p) { build(g, p); }
	bool hasEdge(unsigned int id) { return (bool)edge.count(id); }
	bool hasVertex(unsigned int id) { return (bool)vertex.count(id); }
	//void build(const Graph &, PDFS *);
	//std::string to_string() const;
	void build(const Graph &graph, PDFS *e) {
		if(e) {
			push_back(e->edge);
			edge.insert(e->edge->id);
			vertex.insert(e->edge->from);
			vertex.insert(e->edge->to);
			for(PDFS *p = e->prev; p; p = p->prev) {
				push_back(p->edge);       // this line eats 8% of overall instructions(!)
				edge.insert(p->edge->id);
				vertex.insert(p->edge->from);
				vertex.insert(p->edge->to);
			}
			std::reverse(begin(), end());
		}
	}
	std::string to_string() const {
		std::stringstream ss;
		//ostream_iterator<
		for(int i = 0; i < size(); i++) {
			ss << at(i)->to_string() << "; ";
		}
		return ss.str();
	}
};

class Projected : public std::vector<PDFS> {
public:
	void push(int id, Edge *edge, PDFS *prev) {
		PDFS d;
		d.id = id;
		d.edge = edge;
		d.prev = prev;
		push_back(d);
	}
	//std::string to_string() const;
	std::string to_string() const {
		std::stringstream ss;
		for(int i = 0; i < size(); i++) {
			ss << (*this)[i].to_string() << "; ";
		} // for i
		return ss.str();
	} // Projected::to_string
};


typedef std::map<int, std::map <int, std::map <int, Projected> > >           Projected_map3;
typedef std::map<int, std::map <int, Projected> >                            Projected_map2;
typedef std::map<int, Projected>                                             Projected_map1;
typedef std::map<int, std::map <int, std::map <int, Projected> > >::iterator Projected_iterator3;
typedef std::map<int, std::map <int, Projected> >::iterator Projected_iterator2;
typedef std::map<int, Projected>::iterator Projected_iterator1;
typedef std::map<int, std::map <int, std::map <int, Projected> > >::reverse_iterator Projected_riterator3;

/*
bool  get_forward_pure(const Graph &, Edge *,  int, History&, EdgeList &);
bool  get_forward_rmpath(const Graph &, Edge *,  int,  History&, EdgeList &);
bool  get_forward_root(const Graph &, const Vertex &, EdgeList &);
Edge *get_backward(const Graph &, Edge *,  Edge *, History&);
bool  get_forward(const Graph &, const DFSCode &, History &, EdgeList &);
Edge *get_backward(const Graph &graph, const DFSCode &, History &);
*/
/* Original comment:
 * get_forward_pure ()
 *  e1 (from1, elabel1, to1)
 *  from edge e2(from2, elabel2, to2)
 *
 *  minlabel <= elabel2,
 *  (elabel1 < elabel2 ||
 *  (elabel == elabel2 && tolabel1 < tolabel2)
 *  (elabel1, to1)
 *
 * RK comment:
 * ???? gets the edge that starts and extends the right-most path.
 *
 */
bool get_forward_rmpath(const Graph &graph, Edge *e, int minlabel, History& history, EdgeList &result) {
	result.clear();
	assert(e->to >= 0 && e->to < graph.size());
	assert(e->from >= 0 && e->from < graph.size());
	int tolabel = graph[e->to].label;
	for(Vertex::const_edge_iterator it = graph[e->from].edge.begin(); it != graph[e->from].edge.end(); ++it) {
		int tolabel2 = graph[it->to].label;
		if(e->to == it->to || minlabel > tolabel2 || history.hasVertex(it->to))
			continue;
		if(e->elabel < it->elabel || (e->elabel == it->elabel && tolabel <= tolabel2))
			result.push_back(const_cast<Edge*>(&(*it)));
	}
	return (!result.empty());
}

/* Original comment:
 * get_forward_pure ()
 *  e (from, elabel, to)
 * RK comment: this function takes a "pure" forward edge, that is: an
 * edge that extends the last node of the right-most path, i.e., the
 * right-most node.
 *
 */
bool get_forward_pure(const Graph &graph, Edge *e, int minlabel, History& history, EdgeList &result) {
	result.clear();
	assert(e->to >= 0 && e->to < graph.size());
	//if(e->to >= graph.size())
	//cout<< " e->from " << e->from <<  " e->to " << e->to <<endl;
	// Walk all edges leaving from vertex e->to.
	for(Vertex::const_edge_iterator it = graph[e->to].edge.begin();
			it != graph[e->to].edge.end(); ++it) {
		// -e-> [e->to] -it-> [it->to]
		assert(it->to >= 0 && it->to < graph.size());
		//if(it->to >= graph.size())
		//cout<< " it->to " << it->to <<endl;
		if(minlabel > graph[it->to].label || history.hasVertex(it->to))
			continue;
		result.push_back(const_cast<Edge*>(&(*it)));
	}
	return (!result.empty());
}

bool get_forward_root(const Graph &g, const Vertex &v, EdgeList &result) {
	result.clear();
	for(Vertex::const_edge_iterator it = v.edge.begin(); it != v.edge.end(); ++it) {
		assert(it->to >= 0 && it->to < g.size());
		if(v.label <= g[it->to].label) {
			//std::cout << it->to_string() << "\n";
			result.push_back(const_cast<Edge*>(&(*it)));
		}
	}

	return (!result.empty());
}

/* Original comment:
 *  get_backward (graph, e1, e2, history);
 *  e1 (from1, elabel1, to1)
 *  e2 (from2, elabel2, to2)
 *  to2 -> from1
 *
 *  (elabel1 < elabel2 ||
 *  (elabel == elabel2 && tolabel1 < tolabel2) . (elabel1, to1)
 *
 * RK comment: gets backward edge that starts and ends at the right most path
 * e1 is the forward edge and the backward edge goes to e1->from
 */
Edge *get_backward(const Graph &graph, Edge* e1, Edge* e2, History& history) {
	if(e1 == e2)
		return 0;
	assert(e1->from >= 0 && e1->from < graph.size());
	assert(e1->to >= 0 && e1->to < graph.size());
	assert(e2->to >= 0 && e2->to < graph.size());
	for(Vertex::const_edge_iterator it = graph[e2->to].edge.begin();
			it != graph[e2->to].edge.end(); ++it) {
		if(history.hasEdge(it->id))
			continue;
		if((it->to == e1->from) &&
				((e1->elabel < it->elabel) ||
				 (e1->elabel == it->elabel) &&
				 (graph[e1->to].label <= graph[e2->to].label)
				)) {
			return const_cast<Edge*>(&(*it));
		} // if(...)
	} // for(it)
	return 0;
}

bool get_forward(const Graph &graph, const DFSCode &DFS_CODE, History& history, EdgeList &result) {
	result.clear();
	//forward extenstion from dfs_from <=> from
	int dfs_from = DFS_CODE.back().from;
	int from;
	//skip the last one in dfs code
	// get the "from" vertex id from the history
	for(int i = DFS_CODE.size() - 2; i >= 0; i-- ) {
		if( dfs_from == DFS_CODE[i].from) {
			from = history[i]->from;
			break;
		}
		if( dfs_from == DFS_CODE[i].to) {
			from = history[i]->to;
			break;
		}
	}
	DFS dfs = DFS_CODE.back();
	for(Vertex::const_edge_iterator it = graph[from].edge.begin(); it != graph[from].edge.end(); ++it) {
		if( it->elabel == dfs.elabel && graph[it->to].label == dfs.tolabel &&  !history.hasVertex(it->to) )
			result.push_back(const_cast<Edge*>(&(*it)));
	}
	return (!result.empty());
}

Edge *get_backward(const Graph &graph, const DFSCode &DFS_CODE,  History& history) {
	std::map<int, int> vertex_id_map;
	for(int i = 0; i<history.size(); i++) {
		if(vertex_id_map.count(DFS_CODE[i].from) == 0)
			vertex_id_map[DFS_CODE[i].from] = history[i]->from;
		if(vertex_id_map.count(DFS_CODE[i].to) == 0)
			vertex_id_map[DFS_CODE[i].to]   = history[i]->to;
	}
	//now add the backward edge using the last entry of the DFS code
	int from = vertex_id_map[DFS_CODE.back().from];
	int to   = vertex_id_map[DFS_CODE.back().to];
	for(Vertex::const_edge_iterator it = graph[from].edge.begin(); it != graph[from].edge.end(); ++it) {
		if(it->to == to)
			return const_cast<Edge*>(&(*it));
	} // for(it)
	return 0;
}

#endif

