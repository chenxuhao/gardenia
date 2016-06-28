#include "kernels.cuh"

#define DIAMETER_SAMPLES 512

//For portability reasons, we will not use CUDA 6 features here.
std::vector<float> bc_gpu(graph g, int max_threads_per_block, int number_of_SMs, program_options op, const std::set<int> &source_vertices)
{
	//Host result data
	float *bc_gpu = new float[g.n];
	int *next_source = new int;

	//Device pointers
	float *bc_d, *delta_d;
	int *d_d, *R_d, *C_d, *F_d, *Q_d, *Q2_d, *S_d, *endpoints_d, *next_source_d;
	unsigned long long *sigma_d;
	size_t pitch_d, pitch_sigma, pitch_delta, pitch_Q, pitch_Q2, pitch_S, pitch_endpoints;

	int *jia_d, *diameters_d;

	//Grid parameters
	dim3 dimBlock, dimGrid;
	dimGrid.x = number_of_SMs;
	dimGrid.y = 1;
	dimGrid.z = 1;

	dimBlock.x = max_threads_per_block;
	dimBlock.y = 1;
	dimBlock.z = 1;

	next_source[0] = number_of_SMs; 

	//Allocate and transfer data to the GPU
	checkCudaErrors(cudaMalloc((void**)&bc_d,sizeof(float)*g.n));
	checkCudaErrors(cudaMalloc((void**)&R_d,sizeof(int)*(g.n+1)));
	checkCudaErrors(cudaMalloc((void**)&C_d,sizeof(int)*(2*g.m)));
	checkCudaErrors(cudaMalloc((void**)&F_d,sizeof(int)*(2*g.m)));

	checkCudaErrors(cudaMallocPitch((void**)&d_d,&pitch_d,sizeof(int)*g.n,dimGrid.x));
	checkCudaErrors(cudaMallocPitch((void**)&sigma_d,&pitch_sigma,sizeof(unsigned long long)*g.n,dimGrid.x));
	checkCudaErrors(cudaMallocPitch((void**)&delta_d,&pitch_delta,sizeof(float)*g.n,dimGrid.x));
	checkCudaErrors(cudaMallocPitch((void**)&Q_d,&pitch_Q,sizeof(int)*g.n,dimGrid.x)); //Making Queues/Stack of size O(n) since we won't duplicate
	checkCudaErrors(cudaMallocPitch((void**)&Q2_d,&pitch_Q2,sizeof(int)*g.n,dimGrid.x));
	checkCudaErrors(cudaMallocPitch((void**)&S_d,&pitch_S,sizeof(int)*g.n,dimGrid.x));
	checkCudaErrors(cudaMallocPitch((void**)&endpoints_d,&pitch_endpoints,sizeof(int)*(g.n+1),dimGrid.x));

	checkCudaErrors(cudaMalloc((void**)&next_source_d,sizeof(int)));

	thrust::device_vector<int> source_vertices_d(source_vertices.size());
	if(op.approx)
	{
		thrust::copy(source_vertices.begin(),source_vertices.end(),source_vertices_d.begin());
	}

	checkCudaErrors(cudaMalloc((void**)&jia_d,sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&diameters_d,sizeof(int)*DIAMETER_SAMPLES));
	checkCudaErrors(cudaMemset(jia_d,0,sizeof(int)));
	checkCudaErrors(cudaMemset(diameters_d,0,sizeof(int)*DIAMETER_SAMPLES));

	checkCudaErrors(cudaMemcpy(R_d,g.R,sizeof(int)*(g.n+1),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(C_d,g.C,sizeof(int)*(2*g.m),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(F_d,g.F,sizeof(int)*(2*g.m),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(bc_d,0,sizeof(float)*g.n));
	checkCudaErrors(cudaMemcpy(next_source_d,next_source,sizeof(int),cudaMemcpyHostToDevice));

	//Launch kernel
	if(op.approx)
	{

		bc_gpu_opt<<<dimGrid,dimBlock>>>(bc_d,R_d,C_d,F_d,g.n,g.m,d_d,sigma_d,delta_d,Q_d,Q2_d,S_d,endpoints_d,next_source_d,pitch_d,pitch_sigma,pitch_delta,pitch_Q,pitch_Q2,pitch_S,pitch_endpoints,0,op.k,jia_d,diameters_d,thrust::raw_pointer_cast(source_vertices_d.data()),true);
		checkCudaErrors(cudaPeekAtLastError());
	}
	else
	{
		bc_gpu_opt<<<dimGrid,dimBlock>>>(bc_d,R_d,C_d,F_d,g.n,g.m,d_d,sigma_d,delta_d,Q_d,Q2_d,S_d,endpoints_d,next_source_d,pitch_d,pitch_sigma,pitch_delta,pitch_Q,pitch_Q2,pitch_S,pitch_endpoints,0,g.n,jia_d,diameters_d,thrust::raw_pointer_cast(source_vertices_d.data()),false);
		checkCudaErrors(cudaPeekAtLastError());
	}

	//Transfer result to CPU
	checkCudaErrors(cudaMemcpy(bc_gpu,bc_d,sizeof(float)*g.n,cudaMemcpyDeviceToHost));

	//Free memory
	checkCudaErrors(cudaFree(bc_d));
	checkCudaErrors(cudaFree(R_d));
	checkCudaErrors(cudaFree(C_d));
	checkCudaErrors(cudaFree(F_d));
	checkCudaErrors(cudaFree(d_d));
	checkCudaErrors(cudaFree(sigma_d));
	checkCudaErrors(cudaFree(delta_d));
	checkCudaErrors(cudaFree(Q_d));
	checkCudaErrors(cudaFree(Q2_d));
	checkCudaErrors(cudaFree(S_d));
	checkCudaErrors(cudaFree(endpoints_d));
	checkCudaErrors(cudaFree(next_source_d));

	checkCudaErrors(cudaFree(jia_d));
	checkCudaErrors(cudaFree(diameters_d));

	//Copy host result to a vector
	std::vector<float> bc_gpu_v(bc_gpu,bc_gpu+g.n);

	for(int i=0; i<g.n; i++)
	{
		bc_gpu_v[i] /= 2.0f; //Again we don't want to double count the unweighted edges
	}

	delete[] bc_gpu;
	delete next_source;

	return bc_gpu_v;
}

//Note: N must be a power of two
//Simple/Naive bitonic sort. We're only sorting ~512 elements one time, so performance isn't important
__device__ void bitonic_sort(int *values, int N)
{
	unsigned int idx = threadIdx.x;

	for (int k = 2; k <= N; k <<= 1)
	{
		for (int j = k >> 1; j > 0; j = j >> 1)
		{
			while(idx < N) 
			{
				int ixj = idx^j;
				if (ixj > idx) 
				{
					if ((idx&k) == 0 && values[idx] > values[ixj]) 
					{
						//exchange(idx, ixj);
						int tmp = values[idx];
						values[idx] = values[ixj];
						values[ixj] = tmp;
					}
					if ((idx&k) != 0 && values[idx] < values[ixj]) 
					{
						//exchange(idx, ixj);
						int tmp = values[idx];
						values[idx] = values[ixj];
						values[ixj] = tmp;
					}
				}
				idx += blockDim.x;
			}
			__syncthreads();
			idx = threadIdx.x;
		}
	}
}

__global__ void bc_gpu_opt(float *bc, const int *R, const int *C, const int *F, const int n, const int m, int *d, unsigned long long *sigma, float *delta, int *Q, int *Q2, int *S, int *endpoints, int *next_source, size_t pitch_d, size_t pitch_sigma, size_t pitch_delta, size_t pitch_Q, size_t pitch_Q2, size_t pitch_S, size_t pitch_endpoints, int start, int end, int *jia, int *diameters, int *source_vertices, bool approx)
{
	__shared__ int ind;
	__shared__ int i;
	int j = threadIdx.x;
	int *d_row = (int*)((char*)d + blockIdx.x*pitch_d);
	unsigned long long *sigma_row = (unsigned long long*)((char*)sigma + blockIdx.x*pitch_sigma);
	float *delta_row = (float*)((char*)delta + blockIdx.x*pitch_delta);
	__shared__ int *Q_row;
	__shared__ int *Q2_row;
	__shared__ int *S_row;
	__shared__ int *endpoints_row;

	if(j == 0)
	{
		if(approx)
		{
			ind = blockIdx.x + start;
			i = source_vertices[ind];
			Q_row = (int*)((char*)Q + blockIdx.x*pitch_Q);
			Q2_row = (int*)((char*)Q2 + blockIdx.x*pitch_Q2);
			S_row = (int*)((char*)S + blockIdx.x*pitch_S);
			endpoints_row = (int*)((char*)endpoints + blockIdx.x*pitch_endpoints);
			*jia = 0;
		}
		else
		{
			ind = blockIdx.x + start;
			i = ind;
			Q_row = (int*)((char*)Q + blockIdx.x*pitch_Q);
			Q2_row = (int*)((char*)Q2 + blockIdx.x*pitch_Q2);
			S_row = (int*)((char*)S + blockIdx.x*pitch_S);
			endpoints_row = (int*)((char*)endpoints + blockIdx.x*pitch_endpoints);
			*jia = 0;
		}
	}
	__syncthreads();
	if((ind==0) && (j < DIAMETER_SAMPLES))
	{
		diameters[j] = INT_MAX;
	}
	__syncthreads();

	while(ind < end)
	{
		//Initialization
		for(int k=threadIdx.x; k<n; k+=blockDim.x)
		{
			if(k == i) //If k is the source node...
			{
				d_row[k] = 0;
				sigma_row[k] = 1;
			}
			else
			{
				d_row[k] = INT_MAX;
				sigma_row[k] = 0;
			}	
			delta_row[k] = 0;
		}
		__syncthreads();

		//Shortest Path Calculation
		__shared__ int Q_len;
		__shared__ int Q2_len;
	        __shared__ int S_len;
	        __shared__ int current_depth; 
		__shared__ int endpoints_len;
		__shared__ bool sp_calc_done;

		if(j == 0)
		{
			Q_row[0] = i;
			Q_len = 1;
			Q2_len = 0;
			S_row[0] = i;
			S_len = 1;
			endpoints_row[0] = 0;
			endpoints_row[1] = 1;
			endpoints_len = 2;
			current_depth = 0;
			sp_calc_done = false;
		}
		__syncthreads();

		//Do first iteration separately since we already know the edges to traverse
		for(int r=threadIdx.x+R[i]; r<R[i+1]; r+=blockDim.x)
		{
			int w = C[r];
			//No multiple/self edges - each value of w is unique, so no need for atomics
			if(d_row[w] == INT_MAX)
			{
				d_row[w] = 1; 
				int t = atomicAdd(&Q2_len,1);
				Q2_row[t] = w;
			}
			if(d_row[w] == (d_row[i]+1))
			{
				atomicAdd(&sigma_row[w],1); 
			}
		}
		__syncthreads();

		if(Q2_len == 0)
		{
			sp_calc_done = true;
		}
		else
		{
			for(int kk=threadIdx.x; kk<Q2_len; kk+=blockDim.x)
			{
				Q_row[kk] = Q2_row[kk];
				S_row[kk+S_len] = Q2_row[kk];
			}
			__syncthreads();
			if(j == 0)
			{
				endpoints_row[endpoints_len] = endpoints_row[endpoints_len-1] + Q2_len;
				endpoints_len++;
				Q_len = Q2_len;
				S_len += Q2_len;
				Q2_len = 0;
				current_depth++;
			}
		}
		__syncthreads();

		while(!sp_calc_done)
		{
			if((*jia) && (Q_len > 512))
			{
				for(int k=threadIdx.x; k<2*m; k+=blockDim.x)
				{
					int v = F[k];
					if(d_row[v] == current_depth) 
					{
						int w = C[k];
						if(atomicCAS(&d_row[w],INT_MAX,d_row[v]+1) == INT_MAX)
						{
							int t = atomicAdd(&Q2_len,1);
							Q2_row[t] = w;
						}
						if(d_row[w] == (d_row[v]+1))
						{
							atomicAdd(&sigma_row[w],sigma_row[v]);
						}
					}	
				}
			}
			else
			{
				__shared__ int next_index;
				if(j == 0)
				{
					next_index = blockDim.x;
				}
				__syncthreads();
				int k = threadIdx.x; //Initial vertices
				while(k < Q_len)
				{
					int v = Q_row[k];
					for(int r=R[v]; r<R[v+1]; r++)
					{
						int w = C[r];
						//Use atomicCAS to prevent duplicates
						if(atomicCAS(&d_row[w],INT_MAX,d_row[v]+1) == INT_MAX)
						{
							int t = atomicAdd(&Q2_len,1);
							Q2_row[t] = w;
						}
						if(d_row[w] == (d_row[v]+1))
						{
							atomicAdd(&sigma_row[w],sigma_row[v]);
						}
					}
					k = atomicAdd(&next_index,1);
				}
			}
			__syncthreads();

			if(Q2_len == 0) //If there is no additional work found, we're done
			{
				break;
			}
			else //If there is additional work, transfer elements from Q2 to Q, reset lengths, and add vertices to the stack
			{
				for(int kk=threadIdx.x; kk<Q2_len; kk+=blockDim.x)
				{
					Q_row[kk] = Q2_row[kk];
					S_row[kk+S_len] = Q2_row[kk];
				}
				__syncthreads();
				if(j == 0)
				{
					endpoints_row[endpoints_len] = endpoints_row[endpoints_len-1] + Q2_len;
					endpoints_len++;
					Q_len = Q2_len;
					S_len += Q2_len;
					Q2_len = 0;
					current_depth++;
				}
				__syncthreads();
			}
		}

		//The elements at the end of the stack will have the largest distance from the source
		//Using the successor method, we can start from one depth earlier
		if(j == 0)
		{
			current_depth = d_row[S_row[S_len-1]] - 1;
			if(ind<DIAMETER_SAMPLES)
			{
				diameters[ind] = current_depth+1;
			}
		}
		__syncthreads();

		//Dependency Accumulation (Madduri/Ediger successor method)
		while(current_depth > 0)
		{
			int stack_iter_len = endpoints_row[current_depth+1]-endpoints_row[current_depth];
			if((*jia) && (stack_iter_len>512))
			{
				for(int kk=threadIdx.x; kk<2*m; kk+=blockDim.x)
				{
					int w = F[kk];
					if(d_row[w] == current_depth)
					{
						int v = C[kk];
						if(d_row[v] == (d_row[w]+1))
						{
							float change = (sigma_row[w]/(float)sigma_row[v])*(1.0f+delta_row[v]);
							atomicAdd(&delta_row[w],change);
						}		
					}
				}
			}
			else 
			{
				for(int kk=threadIdx.x+endpoints_row[current_depth]; kk<endpoints_row[current_depth+1]; kk+=blockDim.x)
				{
					int w = S_row[kk];
					float dsw = 0;
					float sw = (float)sigma_row[w];
					for(int z=R[w]; z<R[w+1]; z++)
					{
						int v = C[z];
						if(d_row[v] == (d_row[w]+1))
						{
							dsw += (sw/(float)sigma_row[v])*(1.0f+delta_row[v]);
						}
					}
					delta_row[w] = dsw;	
				}
			}
			__syncthreads();
			if(j == 0)
			{
				current_depth--;
			}
			__syncthreads();
		}

		for(int kk=threadIdx.x; kk<n; kk+=blockDim.x)
		{
			atomicAdd(&bc[kk],delta_row[kk]); //Would need to check that kk != i here, but delta_row[kk] is guaranteed to be 0.
		}
		
		if(j == 0)
		{
			ind = atomicAdd(next_source,1);
			if(approx)
			{
				i = source_vertices[ind];
			}
			else
			{
				i = ind;
			}
		}
		__syncthreads();

		if(ind == 2*DIAMETER_SAMPLES) //Might want to play around with this number. Safe to assume that they are done by now? Probably...
		{
			__shared__ int diameter_keys[DIAMETER_SAMPLES];
			for(int kk = threadIdx.x; kk<DIAMETER_SAMPLES; kk+=blockDim.x)
			{
				diameter_keys[kk] = diameters[kk];
			}
			__syncthreads();
			bitonic_sort(diameter_keys,DIAMETER_SAMPLES);
			__syncthreads();
			if(j == 0)
			{
				int log2n = 0;
				int tempn = n;
				while(tempn >>= 1)
				{
					++log2n;
				}
				if(diameter_keys[DIAMETER_SAMPLES/2] < 4*log2n) //Use the median
				{
					*jia = 1;
				}
			}
		}
		__syncthreads();
	}
}



