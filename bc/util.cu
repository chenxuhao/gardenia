#include "util.cuh"

//Note: Times are returned in seconds
void start_clock(cudaEvent_t &start, cudaEvent_t &end)
{
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&end));
	checkCudaErrors(cudaEventRecord(start,0));
}

float end_clock(cudaEvent_t &start, cudaEvent_t &end)
{
	float time;
	checkCudaErrors(cudaEventRecord(end,0));
	checkCudaErrors(cudaEventSynchronize(end));
	checkCudaErrors(cudaEventElapsedTime(&time,start,end));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(end));

	return time/(float)1000;
}

program_options parse_arguments(int argc, char *argv[])
{
	program_options op;
	int c;

	static struct option long_options[] =
	{
		{"device",required_argument,0,'d'},
		{"help",no_argument,0,'h'},
		{"infile",required_argument,0,'i'},
		{"approx",required_argument,0,'k'},
		{"printscores",optional_argument,0,'p'},
		{"verify",no_argument,0,'v'},
		{0,0,0,0} //Terminate with null
	};

	int option_index = 0;

	while((c = getopt_long(argc,argv,"d:hi:k:p::v",long_options,&option_index)) != -1)
	{
		switch(c)
		{
			case 'd':
				op.device = atoi(optarg);
			break;

			case 'h':
				std::cout << "Usage: " << argv[0] << " -i <input graph file> [-v verify GPU calculation] [-p <output file> print BC scores] [-d <device ID> choose GPU (starting from 0)]" << std::endl;	
			exit(0);

			case 'i':
				op.infile = optarg;
			break;

			case 'k':
				op.approx = true;
				op.k = atoi(optarg);
			break;

			case 'p':
				op.printBCscores = true;
				op.scorefile = optarg;
			break;

			case 'v':
				op.verify = true;
			break;
			
			case '?': //Invalid argument: getopt will print the error msg itself
				
			exit(-1);

			default: //Fatal error
				std::cerr << "Fatal error parsing command line arguments. Terminating." << std::endl;
			exit(-1);

		}
	}

	if(op.infile == NULL)
	{
		std::cerr << "Command line error: Input graph file is required. Use the -i switch." << std::endl;
	}

	return op;
}

void choose_device(int &max_threads_per_block, int &number_of_SMs, program_options op)
{
	int count;
	checkCudaErrors(cudaGetDeviceCount(&count));
	cudaDeviceProp prop;

	if(op.device == -1)
	{
		int maxcc=0, bestdev=0;
		for(int i=0; i<count; i++)
		{
			checkCudaErrors(cudaGetDeviceProperties(&prop,i));
			if((prop.major + 0.1*prop.minor) > maxcc)
			{
				maxcc = prop.major + 0.1*prop.minor;
				bestdev = i;
			}	
		}

		checkCudaErrors(cudaSetDevice(bestdev));
		checkCudaErrors(cudaGetDeviceProperties(&prop,bestdev));
	}
	else if((op.device < -1) || (op.device >= count))
	{
		std::cerr << "Invalid device argument. Valid devices on this machine range from 0 through " << count-1 << "." << std::endl;
		exit(-1);
	}
	else
	{
		checkCudaErrors(cudaSetDevice(op.device));
		checkCudaErrors(cudaGetDeviceProperties(&prop,op.device));
	}

	std::cout << "Chosen Device: " << prop.name << std::endl;
	std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
	std::cout << "Number of Streaming Multiprocessors: " << prop.multiProcessorCount << std::endl;
	std::cout << "Size of Global Memory: " << prop.totalGlobalMem/(float)(1024*1024*1024) << " GB" << std::endl << std::endl;

	max_threads_per_block = prop.maxThreadsPerBlock;
	number_of_SMs = prop.multiProcessorCount;
}

void verify(graph g, const std::vector<float> bc_cpu, const std::vector<float> bc_gpu)
{
	double error = 0;
	double max_error = 0;
	for(int i=0; i<g.n; i++)
	{
		double current_error = abs(bc_cpu[i] - bc_gpu[i]);
		error += current_error*current_error;
		if(current_error > max_error)
		{
			max_error = current_error;
		}
	}
	error = error/(float)g.n;
	error = sqrt(error);
	std::cout << "RMS Error: " << error << std::endl;
	std::cout << "Maximum error: " << max_error << std::endl;
}
