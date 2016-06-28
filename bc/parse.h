#pragma once

#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <set>
#include <vector>
#include <cstdlib>
#include <boost/algorithm/string.hpp>
#include <boost/bimap.hpp>

class graph 
{
public:	
	graph() : R(NULL), C(NULL), F(NULL), n(-1), m(-1) {}

	void print_adjacency_list();
	void print_BC_scores(const std::vector<float> bc, char *outfile);
	void print_CSR();
	void print_R();
	void print_high_degree_vertices();
	void print_numerical_edge_file(char *outfile);
	void print_number_of_isolated_vertices();
	
	int *R;
	int *C;
	int *F;
	int n; //Number of vertices
	int m; //Number of edges
	boost::bimap<unsigned,std::string> IDs; //Associate vertices with other data. In general the unsigned could be replaced with a struct of attributes. 
};

graph parse(char *file);
graph parse_metis(char *file);
//graph parse_snap(char *file);
graph parse_edgelist(char *file);
