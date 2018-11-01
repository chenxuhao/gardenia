// Copyright 2016, National University of Defense Technology
// Author: Xuhao Chen <cxh@illinois.edu>
#include "pr.h"
#include "timer.h"
#include "ocl_util.h"
#include <string.h>
#define PR_VARIANT "ocl_base"

void PRSolver(int m, int nnz, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *out_row_offsets, IndexT *out_column_indices, int *degrees, ScoreT *scores) {
	//load OpenCL kernel file
	char *filechar = "base.cl";
	int sourcesize = 1024*1024;
	char * source = (char *)calloc(sourcesize, sizeof(char));
	if (!source) { printf("ERROR: calloc(%d) failed\n", sourcesize); return; }
	FILE * fp = fopen(filechar, "rb");
	if (!fp) { printf("ERROR: unable to open '%s'\n", filechar); return; }
	size_t error = fread(source + strlen(source), sourcesize, 1, fp);
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
	cl_kernel contrib_kernel;
	cl_kernel pull_kernel;
	cl_kernel l1norm_kernel;

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
	//if (err < 0) { fprintf(stderr, "ERROR: build program failed, err code: %d\n", err); exit(1); }
	if (err < 0) {
		size_t len;
		char buffer[1000];
		printf("ERROR: build program failure\n");
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 1000, buffer, &len);
		printf("error info: %s\n", buffer);
		exit(1);
	}

	// create kernel
	contrib_kernel = clCreateKernel(program, "contrib", &err);
	pull_kernel = clCreateKernel(program, "pull", &err);
	l1norm_kernel = clCreateKernel(program, "l1norm", &err);
	if (err < 0) { fprintf(stderr, "ERROR: create kernel failed, err code: %d\n", err); exit(1); }

	const ScoreT base_score = (1.0f - kDamp) / m;
	cl_mem d_row_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * (m+1), NULL, NULL);
	cl_mem d_column_indices = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * nnz, NULL, NULL);
	cl_mem d_outgoing_contrib = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * m, NULL, NULL);
	cl_mem d_sums = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * m, NULL, NULL);
	cl_mem d_degrees = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m, NULL, NULL);
	cl_mem d_scores = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * nnz, NULL, NULL);
	cl_mem d_diff = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);

	err  = clEnqueueWriteBuffer(queue, d_row_offsets, CL_TRUE, 0, sizeof(int) * (m+1), in_row_offsets, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_column_indices, CL_TRUE, 0, sizeof(int) * nnz, in_column_indices, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_scores, CL_TRUE, 0, sizeof(ValueT) * nnz, scores, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_degrees, CL_TRUE, 0, sizeof(int) * m, degrees, 0, NULL, NULL);
	if (err < 0) { fprintf(stderr, "ERROR: write buffer failed, err code: %d\n", err); exit(1); }

	err  = clSetKernelArg(contrib_kernel, 0, sizeof(int), &m);
	err |= clSetKernelArg(contrib_kernel, 1, sizeof(cl_mem), &d_outgoing_contrib);
	err |= clSetKernelArg(contrib_kernel, 2, sizeof(cl_mem), &d_degrees);
	err |= clSetKernelArg(contrib_kernel, 3, sizeof(cl_mem), &d_scores);
	if (err < 0) { fprintf(stderr, "ERROR: set contrib_kernel arg failed, err code: %d\n", err); exit(1); }

	err  = clSetKernelArg(pull_kernel, 0, sizeof(int), &m);
	err |= clSetKernelArg(pull_kernel, 1, sizeof(cl_mem), &d_row_offsets);
	err |= clSetKernelArg(pull_kernel, 2, sizeof(cl_mem), &d_column_indices);
	err |= clSetKernelArg(pull_kernel, 3, sizeof(cl_mem), &d_sums);
	err |= clSetKernelArg(pull_kernel, 4, sizeof(cl_mem), &d_outgoing_contrib);
	if (err < 0) { fprintf(stderr, "ERROR: set pull_kernel arg failed, err code: %d\n", err); exit(1); }

	err  = clSetKernelArg(l1norm_kernel, 0, sizeof(int), &m);
	err |= clSetKernelArg(l1norm_kernel, 1, sizeof(cl_mem), &d_scores);
	err |= clSetKernelArg(l1norm_kernel, 2, sizeof(cl_mem), &d_sums);
	err |= clSetKernelArg(l1norm_kernel, 3, sizeof(cl_mem), &d_diff);
	err |= clSetKernelArg(l1norm_kernel, 4, sizeof(float), &base_score);
	err |= clSetKernelArg(l1norm_kernel, 5, sizeof(float), &kDamp);
	if (err < 0) { fprintf(stderr, "ERROR: set l1norm_kernel arg failed, err code: %d\n", err); exit(1); }

	size_t globalSize, localSize;
	localSize = BLOCK_SIZE;
	globalSize = ceil(m/(float)localSize)*localSize;
	printf("Launching OpenCL PR solver ...\n");

	Timer t;
	t.Start();
	int iter = 0;
	float h_diff;
	do {
		++ iter;
		h_diff = 0;
		err = clEnqueueWriteBuffer(queue, d_diff, CL_TRUE, 0, sizeof(float), &h_diff, 0, NULL, NULL);
		if (err < 0) { fprintf(stderr, "ERROR write buffer, err code: %d\n", err); exit(1); }
		err = clEnqueueNDRangeKernel(queue, contrib_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if (err < 0) { fprintf(stderr, "ERROR enqueue contrib kernel, err code: %d\n", err); exit(1); }
		err = clEnqueueNDRangeKernel(queue, pull_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if (err < 0) { fprintf(stderr, "ERROR enqueue pull kernel, err code: %d\n", err); exit(1); }
		err = clEnqueueNDRangeKernel(queue, l1norm_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if (err < 0) { fprintf(stderr, "ERROR enqueue l1norm kernel, err code: %d\n", err); exit(1); }
		err = clEnqueueReadBuffer(queue, d_diff, CL_TRUE, 0, sizeof(float), &h_diff, 0, NULL, NULL);
		if (err < 0) { fprintf(stderr, "ERROR enqueue read buffer, err code: %d\n", err); exit(1); }
		printf(" %2d    %f\n", iter, h_diff);
	} while (h_diff > EPSILON && iter < MAX_ITER);
	clFinish(queue);
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", PR_VARIANT, t.Millisecs());
	err = clEnqueueReadBuffer(queue, d_scores, CL_TRUE, 0, sizeof(CompT) * m, scores, 0, NULL, NULL);
	if (err < 0) { fprintf(stderr, "ERROR enqueue read buffer, err code: %d\n", err); exit(1); }

	clReleaseMemObject(d_row_offsets);
	clReleaseMemObject(d_column_indices);
	clReleaseMemObject(d_scores);
	clReleaseMemObject(d_diff);
	clReleaseProgram(program);
	clReleaseKernel(contrib_kernel);
	clReleaseKernel(pull_kernel);
	clReleaseKernel(l1norm_kernel);
	clReleaseCommandQueue(queue);
	return;
}
