#include <iostream>
#include <iomanip>
#include <cstdlib>

#include "parse.h"
#include "sequential.h"
#include "util.cuh"
#include "kernels.cuh"

int main(int argc, char *argv[])
{
	program_options op = parse_arguments(argc,argv);
	int max_threads_per_block, number_of_SMs;
	choose_device(max_threads_per_block,number_of_SMs,op);
	
	graph g = parse(op.infile);

	std::cout << "Number of nodes: " << g.n << std::endl;
	std::cout << "Number of edges: " << g.m << std::endl;

	//If we're approximating, choose source vertices at random
	std::set<int> source_vertices;
	if(op.approx)
	{
		if(op.k > g.n || op.k < 1)
		{
			op.k = g.n;
		}

		while(source_vertices.size() < op.k)
		{
			int temp_source = rand() % g.n;
			source_vertices.insert(temp_source);
		}
	}

	cudaEvent_t start,end;
	float CPU_time;
	std::vector<float> bc;
	if(op.verify) //Only run CPU code if verifying
	{
		start_clock(start,end);
		bc = bc_cpu(g,source_vertices);
		CPU_time = end_clock(start,end);
	}

	float GPU_time;
	std::vector<float> bc_g;
	start_clock(start,end);
	bc_g = bc_gpu(g,max_threads_per_block,number_of_SMs,op,source_vertices);
	GPU_time = end_clock(start,end);

	if(op.verify)
	{
		verify(g,bc,bc_g);
	}
	if(op.printBCscores)
	{
		g.print_BC_scores(bc_g,op.scorefile);
	}

	std::cout << std::setprecision(9);
	if(op.verify)
	{
		std::cout << "Time for CPU Algorithm: " << CPU_time << " s" << std::endl;
	}
	std::cout << "Time for GPU Algorithm: " << GPU_time << " s" << std::endl;
	
	delete[] g.R;
	delete[] g.C;
	delete[] g.F;

	return 0;
}
