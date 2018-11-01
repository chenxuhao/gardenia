// Copyright 2018, National University of Defense Technology
// Author: Xuhao Chen <cxh.nudt@gmail.com>
#include "sssp.h"
#include "timer.h"
#include "ocl_util.h"
#include <string.h>
#define SSSP_VARIANT "ocl_base"

void SSSPSolver(int m, int nnz, int src, int *h_row_offsets, int *h_column_indices, DistT *h_weights, DistT *h_dists, int delta) {
	//load OpenCL kernel file
	char *filechar = "base.cl";
	int src_size = 1024*1024;
	char * source = (char *)calloc(src_size, sizeof(char));
	if (!source) { printf("ERROR: calloc(%d) failed\n", src_size); return; }
	FILE * fp = fopen(filechar, "rb");
	if (!fp) { printf("ERROR: unable to open '%s'\n", filechar); return; }
	size_t error = fread(source + strlen(source), src_size, 1, fp);
	if (error) printf("ERROR: file read failed\n");
	fclose(fp);

	cl_platform_id platforms[32];
	cl_uint num_platforms;
	cl_device_id devices[32];
	cl_uint num_devices;
	char deviceName[1024];

	cl_int err = 0;
	err = clGetPlatformIDs(32, platforms, &num_platforms);
	if (err < 0) { fprintf(stderr, "ERROR clGetPlatformIDs failed, err code: %d\n", err); exit(1); }
	//printf("Number of platforms: %u\n", num_platforms);	

	// get number of devices
	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, sizeof(devices), devices, &num_devices);
	if (err < 0) { fprintf(stderr, "ERROR get num of devices failed, err code: %d\n", err); exit(1); }
	//printf("Number of devices: %d\n", num_devices);

	clGetDeviceInfo(devices[0], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
	printf("Device name: %s\n", deviceName);

	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel init_kernel;
	cl_kernel sssp_kernel;

	// create context
	context = clCreateContext(0, 1, &devices[0], NULL, NULL, &err);
	if (err < 0) { fprintf(stderr, "ERROR: create context failed, err code: %d\n", err); exit(1); }

	// create command queue
	queue = clCreateCommandQueue(context, devices[0], 0, &err);
	if (err < 0) { fprintf(stderr, "ERROR: create command queue failed, err code: %d\n", err); exit(1); }

	// create program
	program = clCreateProgramWithSource(context, 1, (const char **) & source, NULL, &err);
	if (err < 0) { fprintf(stderr, "ERROR: create program failed, err code: %d\n", err); exit(1); }

	// build program
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err < 0) { fprintf(stderr, "ERROR: build program failed, err code: %d\n", err); exit(1); }
	if (err < 0) {
		size_t len;
		char buffer[1000];
		printf("ERROR: build program failure\n");
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 1000, buffer, &len);
		printf("error info: %s\n", buffer);
		exit(1);
	}

	// create kernel
	init_kernel = clCreateKernel(program, "init", &err);
	sssp_kernel = clCreateKernel(program, "sssp_step", &err);
	if (err < 0) { fprintf(stderr, "ERROR: create kernel failed, err code: %d\n", err); exit(1); }

	cl_mem d_row_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * (m+1), NULL, NULL);
	cl_mem d_column_indices = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * nnz, NULL, NULL);
	cl_mem d_weights = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(DistT) * nnz, NULL, NULL);
	cl_mem d_dists = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(DistT) * m, NULL, NULL);
	cl_mem d_in_frontier = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m, NULL, NULL);
	cl_mem d_out_frontier = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m, NULL, NULL);
	cl_mem d_in_nitems = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
	cl_mem d_out_nitems = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);

	int out_nitems = 0, in_nitems = 0;
	err  = clEnqueueWriteBuffer(queue, d_row_offsets, CL_TRUE, 0, sizeof(int) * (m+1), h_row_offsets, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_column_indices, CL_TRUE, 0, sizeof(int) * nnz, h_column_indices, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_weights, CL_TRUE, 0, sizeof(DistT) * nnz, h_weights, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_dists, CL_TRUE, 0, sizeof(DistT) * m, h_dists, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_in_nitems, CL_TRUE, 0, sizeof(int), &in_nitems, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_out_nitems, CL_TRUE, 0, sizeof(int), &out_nitems, 0, NULL, NULL);
	if (err < 0) { fprintf(stderr, "ERROR write buffer, err code: %d\n", err); exit(1); }

	err  = clSetKernelArg(init_kernel, 0, sizeof(int), &src);
	err |= clSetKernelArg(init_kernel, 1, sizeof(cl_mem), &d_in_nitems);
	err |= clSetKernelArg(init_kernel, 2, sizeof(cl_mem), &d_in_frontier);
	err |= clSetKernelArg(init_kernel, 3, sizeof(cl_mem), &d_dists);
	if (err < 0) { fprintf(stderr, "ERROR set init kernel arg, err code: %d\n", err); exit(1); }

	err  = clSetKernelArg(sssp_kernel, 0, sizeof(int), &m);
	err |= clSetKernelArg(sssp_kernel, 1, sizeof(cl_mem), &d_row_offsets);
	err |= clSetKernelArg(sssp_kernel, 2, sizeof(cl_mem), &d_column_indices);
	err |= clSetKernelArg(sssp_kernel, 3, sizeof(cl_mem), &d_weights);
	err |= clSetKernelArg(sssp_kernel, 4, sizeof(cl_mem), &d_dists);
	err |= clSetKernelArg(sssp_kernel, 5, sizeof(cl_mem), &d_in_frontier);
	err |= clSetKernelArg(sssp_kernel, 6, sizeof(cl_mem), &d_out_frontier);
	err |= clSetKernelArg(sssp_kernel, 7, sizeof(cl_mem), &d_in_nitems);
	err |= clSetKernelArg(sssp_kernel, 8, sizeof(cl_mem), &d_out_nitems);
	if (err < 0) { fprintf(stderr, "ERROR set sssp kernel arg, err code: %d\n", err); exit(1); }

	int iter = 0;
	size_t globalSize, localSize;
	localSize = BLOCK_SIZE;
	globalSize = ceil(m/(float)localSize)*localSize;
	printf("Launching OpenCL SSSP solver (%ld threads/CTA) ...\n", localSize);

	Timer t;
	t.Start();
	// push the source vertex into the frontier
	err = clEnqueueNDRangeKernel(queue, init_kernel, 1, NULL, &localSize, &localSize, 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR enqueue init kernel, err code: %d\n", err); exit(1);}
	err = clEnqueueReadBuffer(queue, d_in_nitems, CL_TRUE, 0, sizeof(int), &in_nitems, 0, NULL, NULL);
	do {
		++ iter;
		//printf("iteration %d: frontier_size = %d\n", iter, in_nitems);
		//printf(" %2d    %d\n", iter, in_nitems);
		globalSize = ceil(in_nitems/(float)localSize)*localSize;
		err = clEnqueueNDRangeKernel(queue, sssp_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if (err < 0) { fprintf(stderr, "ERROR enqueue sssp kernel, err code: %d\n", err); exit(1); }

		// move the items from out_queue to in_queue
		err = clEnqueueReadBuffer(queue, d_out_nitems, CL_TRUE, 0, sizeof(int), &out_nitems, 0, NULL, NULL);
		err = clEnqueueCopyBuffer(queue, d_out_frontier, d_in_frontier, 0, 0, out_nitems * sizeof(int), 0, NULL, NULL);
		in_nitems = out_nitems;
		err = clEnqueueWriteBuffer(queue, d_in_nitems, CL_TRUE, 0, sizeof(int), &in_nitems, 0, NULL, NULL);
		out_nitems = 0;
		err = clEnqueueWriteBuffer(queue, d_out_nitems, CL_TRUE, 0, sizeof(int), &out_nitems, 0, NULL, NULL);
	} while(in_nitems > 0);
	clFinish(queue);
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", SSSP_VARIANT, t.Millisecs());
	err = clEnqueueReadBuffer(queue, d_dists, CL_TRUE, 0, sizeof(DistT) * m, h_dists, 0, NULL, NULL);
	if (err < 0) { fprintf(stderr, "ERROR enqueue read buffer, err code: %d\n", err); exit(1); }

	clReleaseMemObject(d_row_offsets);
	clReleaseMemObject(d_column_indices);
	clReleaseMemObject(d_weights);
	clReleaseMemObject(d_dists);
	clReleaseMemObject(d_in_frontier);
	clReleaseMemObject(d_out_frontier);

	clReleaseContext(context);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseKernel(init_kernel);
	clReleaseKernel(sssp_kernel);
	return;
}
