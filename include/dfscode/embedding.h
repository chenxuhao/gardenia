#ifndef __EMBEDDING_HPP__
#define __EMBEDDING_HPP__
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

struct Emb {
	Edge edge;
	Emb  *prev;
	//std::string to_string() const;
	//std::string to_string_global_vid(Graph &g);
	std::string to_string() const {
		std::stringstream ss;
		ss << "[" << edge.to_string() << " " << prev << "]";
		return ss.str();
	}
	std::string to_string_global_vid(Graph &g) {
/*
		Emb *emb = this;
		EmbVector ev = EmbVector(g, emb);
		return ev.to_string_global_vid(g);
*/
		return std::string("");
	}
};

class Embeddings : public std::vector<Emb> {
public:
	void push(Edge edge, Emb *prev) {
		Emb d;
		d.edge = edge;
		d.prev = prev;
		push_back(d);
	}
	//std::string to_string() const;
	//std::string print_global_vid(Graph &g);
	std::string to_string() const {
		std::stringstream ss;
		for(int i = 0; i < size(); i++) {
			ss << (*this)[i].to_string() << "; ";
		} // for i
		return ss.str();
	} // Embeddings::to_string
	std::string print_global_vid(Graph &g) {
		for(std::vector<Emb>::iterator it = this->begin(); it != this->end(); ++it) {
			std::cout << (*it).to_string_global_vid(g) << std::endl;
		}
	}
};

struct Emb2 {
	Edge edge;
	int prev; //index into the previous vector of Emb2 or Embeddings2
	//std::string to_string() const;
	std::string to_string() const {
		std::stringstream ss;
		ss << "[" << edge.to_string() << " " << prev << "]";
		return ss.str();
	}
};

class Embeddings2 : public std::vector<Emb2> {
public:
	void push(Edge edge, int prev) {
		Emb2 d;
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
	} // Embeddings::to_string
};

class EmbVector : public std::vector<Edge> {
private:
	std::set<int> edge;
	std::set<int> vertex;
public:
	EmbVector() { }
	EmbVector(const Graph &g, Emb *p) { build(g, p); }
	bool hasEdge(Edge e) {
		for(std::vector<Edge>::iterator it = this->begin(); it != this->end(); ++it) {
			if(it->from == e.from && it->to == e.to && it->elabel == e.elabel)
				return true;
			else if(it->from == e.to && it->to == e.from && it->elabel == e.elabel)
				return true;
		}
		return false;
		//return (bool)edge.count(id);
	}
	bool hasVertex(unsigned int id) {
		return (bool)vertex.count(id);
	}
	//void build(const Graph &, Emb *);
	//void build(const Graph &, const std::vector<Embeddings2>&, int, int);
	void build(const Graph &graph, Emb *e) {
		// first build history
		if(e) {
			push_back(e->edge);
			edge.insert((*e).edge.id);
			vertex.insert((*e).edge.from);
			vertex.insert((*e).edge.to);
			//cout<<e<<endl;
			for(Emb *p = e->prev; p; p = p->prev) {
				//cout<<p<<endl;
				push_back(p->edge);
				edge.insert((*p).edge.id);
				vertex.insert((*p).edge.from);
				vertex.insert((*p).edge.to);
			}
			std::reverse(begin(), end());
		}
	}
	void build(const Graph &graph, const std::vector<Embeddings2> &emb, int emb_col, int index) {
		//build history of the embedding backwards from index of the last vector
		for(int k = emb_col; k >= 0; k--) {
			Edge e = emb[k][index].edge;
			push_back(e);
			edge.insert(e.id);
			vertex.insert(e.from);
			vertex.insert(e.to);
			index = emb[k][index].prev;
		}
		//cout<<endl;
		std::reverse(begin(), end());
	}
	EmbVector(const Graph &g, const std::vector<Embeddings2>& emb, int emb_col, int emb_index) {
		build(g, emb, emb_col, emb_index);
	}
	//std::string to_string() const;
	//std::string to_string_global_vid(Graph &g) const;
	std::string to_string() const {
		std::stringstream ss;
		//ostream_iterator<
		for(int i = 0; i < size(); i++) {
			ss << at(i).to_string() << "; ";
		}
		return ss.str();
	}
	std::string to_string_global_vid(Graph &g) const {
		std::stringstream ss;
		//ostream_iterator<
		for(int i = 0; i < size(); i++) {
			//ss << "e(" << g[(*this)[i].from].global_vid << "," << g[(*this)[i].to].global_vid << "," << (*this)[i].elabel  << ");";
		}
		return ss.str();
	}
};

typedef std::map<int, std::map <int, std::map <int, Embeddings> > >           Embeddings_map3;
typedef std::map<int, std::map <int, Embeddings> >                            Embeddings_map2;
typedef std::map<int, Embeddings>                                             Embeddings_map1;
typedef std::map<int, std::map <int, std::map <int, Embeddings> > >::iterator Embeddings_iterator3;
typedef std::map<int, std::map <int, Embeddings> >::iterator Embeddings_iterator2;
typedef std::map<int, Embeddings>::iterator Embeddings_iterator1;
typedef std::map<int, std::map <int, std::map <int, Embeddings> > >::reverse_iterator Embeddings_riterator3;
typedef std::map<int, std::map <int, std::map <int, Embeddings2> > >           Embeddings2_map3;
typedef std::map<int, std::map <int, Embeddings2> >                            Embeddings2_map2;
typedef std::map<int, Embeddings2>                                             Embeddings2_map1;
typedef std::map<int, std::map <int, std::map <int, Embeddings2> > >::iterator Embeddings2_iterator3;
typedef std::map<int, std::map <int, Embeddings2> >::iterator Embeddings2_iterator2;
typedef std::map<int, Embeddings2>::iterator Embeddings2_iterator1;
typedef std::map<int, std::map <int, std::map <int, Embeddings2> > >::reverse_iterator Embeddings2_riterator3;
/*
bool  get_forward_root(const Graph &, const Vertex &, std::vector<Edge> &);
bool get_forward_pure(const Graph &, Edge,  int, EmbVector &, std::vector<Edge> &);
bool get_forward_rmpath(const Graph &, Edge,  int,  EmbVector &, std::vector<Edge> &);
bool get_backward(const Graph &, Edge,  Edge, EmbVector &, Edge &);
bool  get_forward(const Graph &, const DFSCode &, EmbVector &, std::vector<Edge> &);
bool  get_backward(const Graph &graph, const DFSCode &, EmbVector &, Edge &);
*/
bool get_forward_rmpath(const Graph &graph, Edge e, int minlabel, EmbVector& history, std::vector<Edge> &result) {
	result.clear();
	assert(e.to >= 0 && e.to < graph.size());
	assert(e.from >= 0 && e.from < graph.size());
	//if(e.to >= graph.size())
	//cout<< " e.from " << e.from <<  " e.to " << e.to <<endl;
	//if(e.from >= graph.size())
	//cout<< " e.from " << e.from <<  " e.to " << e.to <<endl;
	int tolabel = graph[e.to].label;
	for(Vertex::const_edge_iterator it = graph[e.from].edge.begin();
			it != graph[e.from].edge.end(); ++it) {
		int tolabel2 = graph[it->to].label;
		//if(it->to >= graph.size())
		//cout<< " it->to " << it->to <<endl;
		if(e.to == it->to || minlabel > tolabel2 || history.hasVertex(it->to))
			continue;
		if(e.elabel < it->elabel || (e.elabel == it->elabel && tolabel <= tolabel2))
			result.push_back(*it);
	}
	return (!result.empty());
}

bool get_forward_pure(const Graph &graph, Edge e, int minlabel, EmbVector& history, std::vector<Edge> &result) {
	result.clear();
	if(e.to < 0 || e.to >= graph.size())
		std::cout << " e.from " << e.from <<  " e.to " << e.to << std::endl;
	assert(e.to >= 0 && e.to < graph.size());
	// Walk all edges leaving from vertex e->to.
	for(Vertex::const_edge_iterator it = graph[e.to].edge.begin();
			it != graph[e.to].edge.end(); ++it) {
		// -e-> [e->to] -it-> [it->to]
		assert(it->to >= 0 && it->to < graph.size());
		//if(it->to >= graph.size())
		//cout<< " it->to " << it->to <<endl;
		if(minlabel > graph[it->to].label || history.hasVertex(it->to))
			continue;
		result.push_back(*it);
	}
	return (!result.empty());
}

bool get_forward_root(const Graph &g, const Vertex &v, std::vector<Edge> &result) {
	result.clear();
	for(Vertex::const_edge_iterator it = v.edge.begin(); it != v.edge.end(); ++it) {
		assert(it->to >= 0 && it->to < g.size());
		if(v.label <= g[it->to].label)
			result.push_back(*it);
	}
	return (!result.empty());
}

bool get_backward(const Graph &graph, Edge e1, Edge e2, EmbVector& history, Edge& result) {
	if(e1.from == e2.from && e1.to == e2.to && e1.elabel == e2.elabel)
		return false;
	assert(e1.from >= 0 && e1.from < graph.size());
	assert(e1.to >= 0 && e1.to < graph.size());
	assert(e2.to >= 0 && e2.to < graph.size());
	for(Vertex::const_edge_iterator it = graph[e2.to].edge.begin();
			it != graph[e2.to].edge.end(); ++it) {
		if(history.hasEdge(*it))
			continue;
		if((it->to == e1.from) &&
				((e1.elabel < it->elabel) ||
				 (e1.elabel == it->elabel) &&
				 (graph[e1.to].label <= graph[e2.to].label)
				)) {
			result = *it;
			return true;
		} // if(...)
	} // for(it)
	return false;
}

bool get_backward(const Graph &graph, const DFSCode &DFS_CODE, EmbVector& history, Edge& e) {
	std::map<int, int> vertex_id_map;
	for(int i = 0; i<history.size(); i++) {
		if(vertex_id_map.count(DFS_CODE[i].from) == 0)
			vertex_id_map[DFS_CODE[i].from] = history[i].from;
		if(vertex_id_map.count(DFS_CODE[i].to) == 0)
			vertex_id_map[DFS_CODE[i].to]   = history[i].to;
	}
	//now add the backward edge using the last entry of the DFS code
	int from = vertex_id_map[DFS_CODE.back().from];
	int to   = vertex_id_map[DFS_CODE.back().to];
	for(Vertex::const_edge_iterator it = graph[from].edge.begin(); it != graph[from].edge.end(); ++it) {
		if(it->to == to) {
			e = *it;
			return true;
		}
	} // for(it)
	return false;
}

/*
bool get_forward(const Graph &graph, const DFSCode &DFS_CODE, EmbVector& history, std::vector<Edge> &result) {
	result.clear();
	//forward extenstion from dfs_from <=> from
	int dfs_from = DFS_CODE.back().from;
	int from;
	//skip the last one in dfs code
	// get the "from" vertex id from the history
	for(int i = DFS_CODE.size() - 2; i >= 0; i-- ) {
		if( dfs_from == DFS_CODE[i].from) {
			from = history[i].from;
			break;
		}
		if( dfs_from == DFS_CODE[i].to) {
			from = history[i].to;
			break;
		}
	}
	DFS dfs = DFS_CODE.back();
	for(Vertex::const_edge_iterator it = graph[from].edge.begin(); it != graph[from].edge.end(); ++it) {
		if( it->elabel == dfs.elabel && graph[it->to].label == dfs.tolabel &&  !history.hasVertex(it->to) )
			result.push_back(*it);
	}
	return (!result.empty());
}
*/
#endif
