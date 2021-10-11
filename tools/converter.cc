#include <stdio.h>
#include <string.h>
#include <set>
#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <sstream>
#include <cassert>
#include <iostream>
#include <algorithm>
#define USE_ELABEL 0
struct Edge {
	int from;
	int to;
	int elabel;
	Edge() : from(0), to(0), elabel(0) {}
	Edge(int src, int dst, int el) :
		from(src), to(dst), elabel(el) {}
	std::string to_string() const {
		std::stringstream ss;
		ss << "e(" << from << "," << to << "," << elabel << ")";
		return ss.str();
	}
};

typedef std::vector<Edge> EdgeList;

inline void split(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters = " ") {
	std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
	std::string::size_type pos     = str.find_first_of(delimiters, lastPos);
	while (std::string::npos != pos || std::string::npos != lastPos) {
		tokens.push_back(str.substr(lastPos, pos - lastPos));
		lastPos = str.find_first_not_of(delimiters, pos);
		pos = str.find_first_of(delimiters, lastPos);
	}
}

void write_labels_bin(std::vector<int> labels) {
  auto len = labels.size();
	printf("write_labels_bin len: %d\n", len);
  std::ofstream outfile;
  outfile.open("labels.bin", std::ios::binary | std::ios::out);
  outfile.write(reinterpret_cast<char*>(labels.data()), len * sizeof(int));
  outfile.close();
}

void write_labels_txt(std::vector<int> labels) {
  auto len = labels.size();
	printf("write_labels_txt len: %d\n", len);
	std::ofstream outfile;
	outfile.open("labels.txt");
	for (int i = 0; i < len; i ++) {
		std::string outstr = std::to_string(i) + " " + std::to_string(labels[i]) + "\n";
		outfile << outstr;
	}
	outfile.close();
}

std::vector<int> CountDegrees(int m, const EdgeList &el, bool symmetrize = false, bool transpose = false) {
	std::vector<int> degrees(m, 0);
	for (auto it = el.begin(); it < el.end(); it++) {
		Edge e = *it;
		if (symmetrize || (!symmetrize && !transpose))
			degrees[e.from] ++;
		if (symmetrize || (!symmetrize && transpose))
			degrees[e.to] ++;
	}
	return degrees;
}

std::vector<int> PrefixSum(const std::vector<int> &degrees) {
	std::vector<int> sums(degrees.size() + 1);
	int total = 0;
	for (size_t n=0; n < degrees.size(); n++) {
		sums[n] = total;
		total += degrees[n];
	}
	sums[degrees.size()] = total;
	return sums;
}

void MakeCSR(int m, int *&rowptr, int *&colidx, int *&weight, const EdgeList &el, bool symmetrize = false, bool transpose = false) {
	std::vector<int> degrees = CountDegrees(m, el);
	std::vector<int> offsets = PrefixSum(degrees);
	int nnz = offsets[m];
	weight = new int[nnz];
	colidx = new int[nnz];
	rowptr = new int[m+1]; 
	for (int i = 0; i < m+1; i ++) rowptr[i] = offsets[i];
	for (auto it = el.begin(); it < el.end(); it++) {
		Edge e = *it;
		if (symmetrize || (!symmetrize && !transpose)) {
			weight[offsets[e.from]] = e.elabel;
			colidx[offsets[e.from]++] = e.to;
		}
		if (symmetrize || (!symmetrize && transpose)) {
			weight[offsets[e.to]] = e.elabel;
			colidx[offsets[e.to]++] = e.from;
		}
	}
}

void txt2labels(char *infile_name, char *outfile_name = "") {
	std::ifstream infile;
	infile.open(infile_name);
	char line[1024];
	std::vector<std::string> result;
	std::vector<int> labels;
	while(true) {
		unsigned pos = infile.tellg();
		if(!infile.getline(line, 1024)) break;
		result.clear();
		split(line, result);
		if(result.empty()) {
		} else if(result[0] == "t") {
			if (!labels.empty()) {   // use as delimiter
				infile.seekg(pos, std::ios_base::beg);
				break;
			} else { }
		} else if(result[0] == "v" && result.size() >= 3) {
			unsigned id = atoi(result[1].c_str());
			labels.resize(id + 1);
			labels[id] = atoi(result[2].c_str());
		} else if(result[0] == "e") break;
  }
  infile.close();
  //write_labels_bin(labels);
  write_labels_txt(labels);
}

void el2mtx(char *infile_name, char *outfile_name) {
	printf("Reading input file %s\n", infile_name);
	int m, n, nnz;
	std::ifstream infile;
	std::ofstream outfile;
	infile.open(infile_name);
	outfile.open(outfile_name);
	std::string instr;
	getline(infile, instr);
	char c;
	sscanf(instr.c_str(), "%c", &c);
	while (c == '%') {
		getline(infile, instr);
		sscanf(instr.c_str(), "%c", &c);
	}
	sscanf(instr.c_str(), "%d %d %d", &m, &n, &nnz);
	outfile << instr;
	outfile << "\n";
	printf("m %d n %d nnz %d\n", m, n, nnz);
	int dst, src;
	float wt = 1.0f;
	for (int i = 0; i < nnz; i ++) {
		getline(infile, instr);
		//int num = sscanf(instr.c_str(), "%d %d %f", &src, &dst, &wt);
		int num = sscanf(instr.c_str(), "%d %d", &src, &dst);
		src++;
		dst++;
		std::string outstr = std::to_string(src) + " " + std::to_string(dst) + "\n";
		//printf("src %d dst %d\n", src, dst);
		//sprintf(outstr.c_str(), "%d %d\n", src, dst);
		outfile << outstr;
		//std::cout << outstr;
	}
	infile.close();
	outfile.close();
}

void el2sel(char *infile_name, char *outfile_name) {
	printf("Reading input file %s\n", infile_name);
	int m, nnz = 0;
	std::set<int> vertex_set; 
	std::ifstream infile;
	std::ofstream outfile;
	infile.open(infile_name);
	outfile.open(outfile_name);
	std::string instr;
	int dst, src;
	while (getline(infile, instr)) {
		int num = sscanf(instr.c_str(), "%d %d", &src, &dst);
		if (src == dst) continue;
		vertex_set.insert(src);
		vertex_set.insert(dst);
		std::string outstr = std::to_string(src) + " " + std::to_string(dst) + "\n";
		outfile << outstr;
		outstr = std::to_string(dst) + " " + std::to_string(src) + "\n";
		outfile << outstr;
		nnz ++;
	}
	m = vertex_set.size();
	printf("m %d nnz %d\n", m, nnz);
	infile.close();
	outfile.close();
}

void el2cel(char *infile_name, char *outfile_name) {
	printf("[el2cel] Reading input file %s\n", infile_name);
	int m, nnz = 0;
	std::set<int> vertex_set; 
	std::set<std::pair<int, int> > edge_set; 
	std::ifstream infile;
	std::ofstream outfile;
	infile.open(infile_name);
	outfile.open(outfile_name);
	std::string instr, outstr;
	int dst, src;
	while (getline(infile, instr)) {
		int num = sscanf(instr.c_str(), "%d %d", &src, &dst);
		if (src == dst) continue;
		if (edge_set.find(std::pair<int, int>(src, dst)) == edge_set.end() && edge_set.find(std::pair<int, int>(dst, src)) == edge_set.end()) {
			vertex_set.insert(src);
			vertex_set.insert(dst);
			if (src < dst) {
				edge_set.insert(std::pair<int, int>(src, dst)); // avoid redundant
				outstr = std::to_string(src) + " " + std::to_string(dst) + "\n";
			} else {
				edge_set.insert(std::pair<int, int>(dst, src)); // avoid redundant
				outstr = std::to_string(dst) + " " + std::to_string(src) + "\n";
			}
			outfile << outstr;
			nnz ++;
		}
	}
	m = vertex_set.size();
	printf("m %d nnz %d\n", m, nnz);
	infile.close();
	outfile.close();
}

// convert from mtx to edgelist for bipartite graphs (weighted)
void mtx2el_bip(char *infile_name, char *outfile_name) {
	printf("Reading input file %s\n", infile_name);
	int m, n, nnz;
	std::ifstream infile;
	std::ofstream outfile;
	infile.open(infile_name);
	outfile.open(outfile_name);
	std::string instr;
	getline(infile, instr);
	char c;
	sscanf(instr.c_str(), "%c", &c);
	while (c == '%') {
		getline(infile, instr);
		sscanf(instr.c_str(), "%c", &c);
	}
	sscanf(instr.c_str(), "%d %d %d", &m, &n, &nnz);
	outfile << instr;
	outfile << "\n";
	printf("m %d n %d nnz %d\n", m, n, nnz);
	int dst, src;
	int wt = 1;
	for (int i = 0; i < nnz; i ++) {
		getline(infile, instr);
		int num = sscanf(instr.c_str(), "%d %d %d", &src, &dst, &wt);
		src--;
		dst+=m-1;
		std::string outstr = std::to_string(src) + " " + std::to_string(dst) + " " + std::to_string(wt) + "\n";
		//printf("src %d dst %d\n", src, dst);
		//sprintf(outstr.c_str(), "%d %d\n", src, dst);
		outfile << outstr;
		//std::cout << outstr;
	}
	infile.close();
	outfile.close();
}

EdgeList el;
void el2sadj(char *infile_name, char *outfile_name, bool directed = false) {
	std::ifstream infile;
	infile.open(infile_name);
	//std::set<int> vertex_set; 
	std::set<std::pair<int, int> > edge_set;
	std::string instr;
	int from, to;
	int m = 0;
	while (getline(infile, instr)) {
		int num = sscanf(instr.c_str(), "%d %d", &from, &to);
		if (m < from + 1) m = from + 1;
		if (m < to + 1) m = to + 1;
		if (from == to) continue;
		//vertex_set.insert(from);
		//vertex_set.insert(to);
		if (edge_set.find(std::pair<int, int>(from, to)) == edge_set.end()) {
			edge_set.insert(std::pair<int, int>(from, to));
			edge_set.insert(std::pair<int, int>(to, from));
			el.push_back(Edge(from, to, 1));
			el.push_back(Edge(to, from, 1));
		}
	}
	int nnz = el.size();
	printf("m %d nnz %d\n", m, nnz);
	int *rowptr = NULL, *colidx = NULL, *weight = NULL;
	MakeCSR(m, rowptr, colidx, weight, el);
	std::ofstream outfile;
	outfile.open(outfile_name);
	for (int i = 0; i < m; i ++) {
		std::string outstr = std::to_string(i) + " " + "1";//std::to_string(labels[i]);
		for (int offset = rowptr[i]; offset < rowptr[i+1]; offset ++) {
			outstr = outstr + " " + std::to_string(colidx[offset]);
		}
		outstr += "\n";
		outfile << outstr;
	}
	outfile.close();
}

void txt2sadj(char *infile_name, char *outfile_name, bool directed = false) {
	std::ifstream infile;
	infile.open(infile_name);
	char line[1024];
	std::vector<std::string> result;
	std::vector<int> labels;
	std::set<std::pair<int, int> > edge_set;
	while(true) {
		unsigned pos = infile.tellg();
		if(!infile.getline(line, 1024)) break;
		result.clear();
		split(line, result);
		if(result.empty()) {
		} else if(result[0] == "t") {
			if(!labels.empty()) {   // use as delimiter
				infile.seekg(pos, std::ios_base::beg);
				break;
			} else { }
		} else if(result[0] == "v" && result.size() >= 3) {
			unsigned id = atoi(result[1].c_str());
			labels.resize(id + 1);
			labels[id] = atoi(result[2].c_str());
		} else if(result[0] == "e" && result.size() >= 4) {
			int from   = atoi(result[1].c_str());
			int to     = atoi(result[2].c_str());
			int elabel = atoi(result[3].c_str());
			if(labels.size() <= from || labels.size() <= to) {
				std::cerr << "Format Error:  define vertex lists before edges, from: " << from 
					<< "; to: " << to << "; vertex count: " << labels.size() << std::endl;
				exit(1);
			}
			if (edge_set.find(std::pair<int, int>(from, to)) == edge_set.end()) {
				edge_set.insert(std::pair<int, int>(from, to));
				el.push_back(Edge(from, to, elabel));
				if(!directed) {
					edge_set.insert(std::pair<int, int>(to, from));
					el.push_back(Edge(to, from, elabel));
				}
			}
		}
	}
	int m = labels.size();
	int nnz = el.size();
	//if (!directed) symmetrize = false;
	printf("m %d nnz %d\n", m, nnz);
	int *rowptr = NULL, *colidx = NULL, *weight = NULL;
	MakeCSR(m, rowptr, colidx, weight, el);
	std::ofstream outfile;
	outfile.open(outfile_name);
	for (int i = 0; i < m; i ++) {
		std::string outstr = std::to_string(i) + " " + std::to_string(labels[i]);
		for (int offset = rowptr[i]; offset < rowptr[i+1]; offset ++) {
			outstr = outstr + " " + std::to_string(colidx[offset]);
			if(USE_ELABEL) outstr = outstr + " " + std::to_string(weight[offset]);
		}
		outstr += "\n";
		outfile << outstr;
	}
	outfile.close();
}

void txt2ctxt(char *infile_name, char *outfile_name) {
	std::ifstream infile;
	infile.open(infile_name);
	char line[1024];
	std::vector<std::string> result;
	std::vector<int> labels;
	std::set<std::pair<int, int> > edge_set;
	while(true) {
		unsigned pos = infile.tellg();
		if(!infile.getline(line, 1024)) break;
		result.clear();
		split(line, result);
		if(result.empty()) {
		} else if(result[0] == "t") {
			if(!labels.empty()) {   // use as delimiter
				infile.seekg(pos, std::ios_base::beg);
				break;
			} else { }
		} else if(result[0] == "v" && result.size() >= 3) {
			unsigned id = atoi(result[1].c_str());
			labels.resize(id + 1);
			labels[id] = atoi(result[2].c_str());
		} else if(result[0] == "e" && result.size() >= 4) {
			int from   = atoi(result[1].c_str());
			int to     = atoi(result[2].c_str());
			int elabel = atoi(result[3].c_str());
			if(labels.size() <= from || labels.size() <= to) {
				std::cerr << "Format Error:  define vertex lists before edges, from: " << from 
					<< "; to: " << to << "; vertex count: " << labels.size() << std::endl;
				exit(1);
			}
			if (from == to) continue;
			if (edge_set.find(std::pair<int, int>(from, to)) == edge_set.end()) {
				edge_set.insert(std::pair<int, int>(from, to));
				edge_set.insert(std::pair<int, int>(to, from));
				if(from > to)
					el.push_back(Edge(to, from, elabel));
				else 
					el.push_back(Edge(from, to, elabel));
			}
		}
	}
	int m = labels.size();
	int nnz = el.size();
	printf("m %d nnz %d\n", m, nnz);
	std::ofstream outfile;
	outfile.open(outfile_name);
	for (int i = 0; i < m; i ++) {
		std::string outstr = "v " + std::to_string(i) + " " + std::to_string(labels[i]) + "\n";
		outfile << outstr;
	}
	for (int i = 0; i < nnz; i ++) {
		std::string outstr = "e " + std::to_string(el[i].from) + " " + std::to_string(el[i].to) + " 1\n";
		outstr += "\n";
		outfile << outstr;
	}
	outfile.close();
}

void txt2cel(char *infile_name, char *outfile_name, bool directed = true) {
	std::ifstream infile;
	infile.open(infile_name);
	char line[1024];
	std::vector<std::string> result;
	std::vector<int> labels;
	std::set<std::pair<int, int> > edge_set;
	while(true) {
		unsigned pos = infile.tellg();
		if(!infile.getline(line, 1024)) break;
		result.clear();
		split(line, result);
		if(result.empty()) {
		} else if(result[0] == "t") {
			if(!labels.empty()) {   // use as delimiter
				infile.seekg(pos, std::ios_base::beg);
				break;
			} else { }
		} else if(result[0] == "v" && result.size() >= 3) {
			unsigned id = atoi(result[1].c_str());
			labels.resize(id + 1);
			labels[id] = atoi(result[2].c_str());
		} else if(result[0] == "e" && result.size() >= 4) {
			int from   = atoi(result[1].c_str());
			int to     = atoi(result[2].c_str());
			int elabel = atoi(result[3].c_str());
			if (from == to) continue; // remove self-loop
			if(labels.size() <= from || labels.size() <= to) {
				std::cerr << "Format Error:  define vertex lists before edges, from: " << from 
					<< "; to: " << to << "; vertex count: " << labels.size() << std::endl;
				exit(1);
			}
			if (edge_set.find(std::pair<int, int>(from, to)) == edge_set.end()) {
				edge_set.insert(std::pair<int, int>(from, to)); // avoid redundant
				edge_set.insert(std::pair<int, int>(to, from)); // avoid redundant
				int small = from, large = to;
				if (from > to) {
					small = to;
					large = from;
				}
				el.push_back(Edge(small, large, elabel));
				if (!directed) el.push_back(Edge(large, small, elabel));
			}
		}
	}
	int m = labels.size();
	int nnz = el.size();
	printf("m %d nnz %d\n", m, nnz);
	infile.close();
	std::ofstream outfile;
	outfile.open(outfile_name);
	for (int i = 0; i < nnz; i ++) {
		int src = el[i].from;
		int dst = el[i].to;
		std::string outstr = std::to_string(src) + " " + std::to_string(dst) + "\n";
		outfile << outstr;
	}
	outfile.close();
}

void sadj2ctxt(const char *filename, char *outfile_name) {
	FILE* fd = fopen(filename, "r");
	assert (fd != NULL);
	char buf[2048];
	int size = 0, maxsize = 0;
	while (fgets(buf, 2048, fd) != NULL) {
		int len = strlen(buf);
		size += len;
		if (buf[len-1] == '\n') {
			maxsize = std::max(size, maxsize);
			size = 0;
		}
	}
	fclose(fd);

	std::ifstream is;
	is.open(filename, std::ios::in);
	char*line = new char[maxsize+1];
	std::vector<std::string> result;
	std::vector<int> labels;
	while(is.getline(line, maxsize+1)) {
		result.clear();
		split(line, result);
		int src = atoi(result[0].c_str());
		labels.resize(src + 1);
		labels[src] = atoi(result[1].c_str());
		std::set<std::pair<int, int> > neighbors;
		for(size_t i = 2; i < result.size(); i++) {
			int dst = atoi(result[i].c_str());
			if (src == dst) continue; // remove self-loop
			neighbors.insert(std::pair<int, int>(dst, 1)); // remove redundant edge
		}
		for (auto it = neighbors.begin(); it != neighbors.end(); ++it)
			if (src < it->first) el.push_back(Edge(src, it->first, it->second));
	}
	is.close();

	int m = labels.size();
	int nnz = el.size();
	printf("m %d nnz %d\n", m, nnz);
	std::ofstream outfile;
	outfile.open(outfile_name);
	for (int i = 0; i < m; i ++) {
		std::string outstr = "v " + std::to_string(i) + " " + std::to_string(labels[i]) + "\n";
		outfile << outstr;
	}
	for (int i = 0; i < nnz; i ++) {
		std::string outstr = "e " + std::to_string(el[i].from) + " " + std::to_string(el[i].to) + " 1\n";
		outstr += "\n";
		outfile << outstr;
	}
	outfile.close();
}

void sadj2adj(const char *filename, char *outfile_name) {
	FILE* fd = fopen(filename, "r");
	assert (fd != NULL);
	char buf[2048];
	int size = 0, maxsize = 0;
	while (fgets(buf, 2048, fd) != NULL) {
		int len = strlen(buf);
		size += len;
		if (buf[len-1] == '\n') {
			maxsize = std::max(size, maxsize);
			size = 0;
		}
	}
	fclose(fd);

	int m = 0; 
	int nnz = 0;
	std::ifstream infile;
	infile.open(filename, std::ios::in);
	char*line = new char[maxsize+1];
	std::vector<std::string> result;
	std::vector<int> degrees;
	while(infile.getline(line, maxsize+1)) {
		m ++;
		result.clear();
		split(line, result);
		int src = atoi(result[0].c_str());
		degrees.resize(src + 1);
		degrees[src] = result.size() - 2;
		nnz += degrees[src];
	}
	infile.close();

	printf("m %d nnz %d\n", m, nnz);
	std::ofstream outfile;
	infile.open(filename, std::ios::in);
	outfile.open(outfile_name);
	while(infile.getline(line, maxsize+1)) {
		result.clear();
		split(line, result);
		int src = atoi(result[0].c_str());
		std::string outstr = std::to_string(src) + "\t" + std::to_string(degrees[src]);//std::to_string(labels[i]);
		for(size_t i = 2; i < result.size(); i++) {
			int dst = atoi(result[i].c_str());
			outstr = outstr + " " + std::to_string(dst);
		}
		outstr += "\n";
		outfile << outstr;
	}
	infile.close();
	outfile.close();
}

void mtx2cel(const char *filename, char *outfile_name, bool directed = false) {
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
	std::set<std::pair<int, int> > edge_set;
	int m, n, nonzeros;
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
		//if (read_weights) {
		if (0) {
			int v;
			edge_stream >> v;
			int w = 1;
			el.push_back(Edge(u - 1, v - 1, w));
			if (undirected)
				el.push_back(Edge(v - 1, u - 1, w));
		} else {
			int v;
			edge_stream >> v;
			if (u == v) continue;
			int from = u - 1;
			int to = v - 1;
			if (edge_set.find(std::pair<int, int>(from, to)) == edge_set.end()) {
				edge_set.insert(std::pair<int, int>(from, to)); // avoid redundant
				edge_set.insert(std::pair<int, int>(to, from)); // avoid redundant
				//if (undirected) el.push_back(Edge(from, to, 1));
				//else el.push_back(Edge(to, from, 1));
				int small = from, large = to;
				if (from > to) {
					small = to;
					large = from;
				}
				el.push_back(Edge(small, large, 1));
				//if (!directed) el.push_back(Edge(large, small, elabel));
			}
		}
	}
	in.close();

	//if (!directed) symmetrize = false;
	printf("m %d nnz %d\n", m, nonzeros);
	std::ofstream outfile;
	outfile.open(outfile_name);
	for (int i = 0; i < nonzeros; i ++) {
		int src = el[i].from;
		int dst = el[i].to;
		std::string outstr = std::to_string(src) + " " + std::to_string(dst) + "\n";
		outfile << outstr;
	}
	outfile.close();
}

void mtx2sadj(const char *filename, char *outfile_name, bool directed = false) {
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
	int m, n, nonzeros;
	in >> m >> n >> nonzeros >> std::ws;
	if (m != n) {
		std::cout << m << " " << n << " " << nonzeros << std::endl;
		std::cout << "matrix must be square for .mtx" << std::endl;
		std::exit(-26);
	}
	int nnz = 0;
	while (std::getline(in, line)) {
		std::istringstream edge_stream(line);
		int u;
		edge_stream >> u;
		int v;
		edge_stream >> v;
		if (u == v) continue;
		nnz ++;
		if (read_weights) {
			int w = 1;
			el.push_back(Edge(u - 1, v - 1, w));
			if (undirected)
				el.push_back(Edge(v - 1, u - 1, w));
		} else {
			el.push_back(Edge(u - 1, v - 1, 1));
			if (undirected)
				el.push_back(Edge(v - 1, u - 1, 1));
		}
	}
	in.close();

	//if (!directed) symmetrize = false;
	printf("nnz %d\n", nnz);
	printf("m %d nonzeros %d\n", m, nonzeros);
	int *rowptr = NULL, *colidx = NULL, *weight = NULL;
	MakeCSR(m, rowptr, colidx, weight, el);
	std::ofstream outfile;
	outfile.open(outfile_name);
	for (int i = 0; i < m; i ++) {
		std::string outstr = std::to_string(i) + " " + std::to_string(1);
		for (int offset = rowptr[i]; offset < rowptr[i+1]; offset ++) {
			outstr = outstr + " " + std::to_string(colidx[offset]);
			if(USE_ELABEL) outstr = outstr + " " + std::to_string(weight[offset]);
		}
		outstr += "\n";
		outfile << outstr;
	}
	outfile.close();
}

int main(int argc, char *argv[]) {
	//el2mtx(argv[1], argv[2]);
	//el2sel(argv[1], argv[2]);
	//el2cel(argv[1], argv[2]);
	//el2sadj(argv[1], argv[2]);
	//sadj2adj(argv[1], argv[2]);
	//mtx2el_bip(argv[1], argv[2]);
	//txt2sadj(argv[1], argv[2]);
	//txt2labels(argv[1], argv[2]);
	//txt2ctxt(argv[1], argv[2]);
	txt2cel(argv[1], argv[2], false);
	//mtx2cel(argv[1], argv[2], false);
	//sadj2ctxt(argv[1], argv[2]);
	//mtx2sadj(argv[1], argv[2]);
	//txt2el(argv[1], argv[2]);
	return 0;
}
