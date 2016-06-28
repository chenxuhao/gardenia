#ifndef BC_UTIL
#define BC_UTIL

#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include <cmath>
#include <getopt.h>

#include "parse.h"

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#ifndef checkCudaErrors
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line)
{   
    if (cudaSuccess != err)
    {   
        std::cerr << "CUDA Error = " << err << ": " << cudaGetErrorString(err) << " from file " << file  << ", line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif

//Command line parsing
class program_options
{
public:
	program_options() : infile(NULL), verify(false), printBCscores(false), scorefile(NULL), device(-1), approx(false), k(256) {}

	char *infile;
	bool verify;
	bool printBCscores;
	char *scorefile;
	int device;
	bool approx;
	int k;
};
program_options parse_arguments(int argc, char *argv[]);

//Timing routines
void start_clock(cudaEvent_t &start, cudaEvent_t &end);
float end_clock(cudaEvent_t &start, cudaEvent_t &end);

//Device routines
void choose_device(int &max_threads_per_block, int &number_of_SMs, program_options op);

//Verification routines
void verify(graph g, const std::vector<float> bc_cpu, const std::vector<float> bc_gpu);

#endif

