#ifndef GRAPH_H
#define GRAPH_H
#define NLINKS 100000000 //maximum number of edges for memory allocation, will increase if needed
#include <set>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

struct Edge {
	unsigned s;
	unsigned t;
	Edge() : s(0), t(0) {}
	Edge(int from, int to) : s(from), t(to) {}
};

//Compute the maximum of three unsigned integers.
inline unsigned int max3(unsigned int a,unsigned int b,unsigned int c) {
	a=(a>b) ? a : b;
	return (a>c) ? a : c;
}

inline void split(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters = " ") {
	std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
	std::string::size_type pos     = str.find_first_of(delimiters, lastPos);
	while (std::string::npos != pos || std::string::npos != lastPos) {
		tokens.push_back(str.substr(lastPos, pos - lastPos));
		lastPos = str.find_first_not_of(delimiters, pos);
		pos = str.find_first_of(delimiters, lastPos);
	}
}

typedef std::vector<Edge> Edgelist;

class Graph {
public:
	unsigned num_vertices;
	unsigned num_edges;
	bool directed_;
	bool symmetrize_;
	Edgelist edges;//list of edges
	unsigned n;
	unsigned *cd;//cumulative degree: (starts with 0) length=n+1
	unsigned *adj;//truncated list of neighbors
	unsigned core;//core value of the graph
	Graph () {}
	~Graph () {
		free(cd);
		free(adj);
	}
	void read_edgelist(char* input) {
		FILE *file = fopen(input,"r");
		num_vertices = 0;
		unsigned src, dst;
		while (fscanf(file,"%u %u", &src, &dst) == 2) {//Add one edge
			num_vertices = max3(num_vertices, src, dst);
			edges.push_back(Edge(src, dst));
		}
		fclose(file);
		num_vertices ++;
		num_edges = edges.size();
	}
	void read_txt(const char *filename, bool directed = true) {
		num_vertices = 0;
		std::ifstream is;
		is.open(filename, std::ios::in);
		char line[1024];
		std::vector<std::string> result;
		std::set<std::pair<int,int> > edge_set;
		while(true) {
			if(!is.getline(line, 1024)) break;
			result.clear();
			split(line, result);
			if(result.empty()) {
			} else if(result[0] == "t") {
			} else if(result[0] == "v" && result.size() >= 3) {
				num_vertices ++;
			} else if(result[0] == "e" && result.size() >= 4) {
				int src    = atoi(result[1].c_str());
				int dst    = atoi(result[2].c_str());
				//int elabel = atoi(result[3].c_str());
				if (src == dst) continue; // remove self-loop
				if (edge_set.find(std::pair<int, int>(src, dst)) == edge_set.end() && edge_set.find(std::pair<int, int>(dst, src)) == edge_set.end()) {
					edge_set.insert(std::pair<int, int>(src, dst));
					edges.push_back(Edge(src, dst));
					if(directed == false) {
						edge_set.insert(std::pair<int, int>(dst, src));
						edges.push_back(Edge(dst, src));
					}
				}
			}
		}
		is.close();
		num_edges = edges.size();
	}
	void read_mtx(const char *filename, bool symmetrize = false, bool needs_weights = false) {
		std::ifstream in;
		in.open(filename, std::ios::in);
		std::string start, object, format, field, symmetry, line;
		in >> start >> object >> format >> field >> symmetry >> std::ws;
		if (start != "%%MatrixMarket") {
			std::cout << ".mtx file did not start with %%MatrixMarket" << std::endl;
			std::exit(-21);
		}
		if ((object != "matrix") || (format != "coordinate")) {
			std::cout << "only allow matrix coordinate format for .mtx" << std::endl;
			std::exit(-22);
		}
		if (field == "complex") {
			std::cout << "do not support complex weights for .mtx" << std::endl;
			std::exit(-23);
		}
		bool read_weights;
		if (field == "pattern") {
			read_weights = false;
		} else if ((field == "real") || (field == "double") || (field == "integer")) {
			read_weights = true;
		} else {
			std::cout << "unrecognized field type for .mtx" << std::endl;
			std::exit(-24);
		}
		bool undirected;
		if (symmetry == "symmetric") {
			undirected = true;
		} else if ((symmetry == "general") || (symmetry == "skew-symmetric")) {
			undirected = false;
		} else {
			std::cout << "unsupported symmetry type for .mtx" << std::endl;
			std::exit(-25);
		}
		while (true) {
			char c = in.peek();
			if (c == '%') { in.ignore(200, '\n');
			} else { break; }
		}
		int64_t m, n, nonzeros;
		in >> m >> n >> nonzeros >> std::ws;
		if (m != n) {
			std::cout << m << " " << n << " " << nonzeros << std::endl;
			std::cout << "matrix must be square for .mtx" << std::endl;
			std::exit(-26);
		}
		while (std::getline(in, line)) {
			std::istringstream edge_stream(line);
			int u;
			edge_stream >> u;
			if (read_weights) {
				int v;
				edge_stream >> v;
				edges.push_back(Edge(u - 1, v - 1));
				//if (undirected || symmetrize)
				//	edges.push_back(Edge(v - 1, u - 1));
			} else {
				int v;
				edge_stream >> v;
				edges.push_back(Edge(u - 1, v - 1));
				//if (undirected || symmetrize)
				//	edges.push_back(Edge(v - 1, u - 1));
			}
		}
		in.close();
		directed_ = !undirected;
		if (undirected) symmetrize_ = false; // no need to symmetrize undirected graph
		num_vertices = m;
		num_edges = edges.size();
	}
	void relabel(unsigned *rank) {
		for (unsigned i=0; i < num_edges; i ++) {
			unsigned source = rank[edges[i].s];
			unsigned target = rank[edges[i].t];
			if (source < target) {
				unsigned tmp = source;
				source = target;
				target = tmp;
			}
			edges[i].s = source;
			edges[i].t = target;
		}
	}
	//Building the special graph
	void mkgraph() {
		unsigned *d = (unsigned *)calloc(num_vertices, sizeof(unsigned));
		for (unsigned i = 0; i < num_edges; i++) {
			d[edges[i].s]++;
		}
		cd = (unsigned *)malloc((num_vertices + 1) * sizeof(unsigned));
		cd[0] = 0;
		unsigned max = 0;
		for (unsigned i = 1; i < num_vertices+1;i++) {
			cd[i] = cd[i-1] + d[i-1];
			max = (max > d[i-1]) ? max : d[i-1];
			d[i-1]=0;
		}
		printf("core value (max truncated degree) = %u\n",max);
		adj = (unsigned *)malloc(num_edges * sizeof(unsigned));
		for (unsigned i = 0; i < num_edges; i ++) {
			adj[cd[edges[i].s] + d[edges[i].s]++] = edges[i].t;
		}
		free(d);
		core = max;
		n = num_vertices;
	}
};

#endif
