// Copyright 2016, National University of Defense Technology
// Author: Xuhao Chen <cxh@illinois.edu>
#include "bc.h"
#include "timer.h"
#include "ocl_util.h"
#include <string.h>
#include <vector>
#define BC_VARIANT "ocl_base"

void BCSolver(int m, int nnz, int src, int *h_row_offsets, int *h_column_indices, ScoreT *h_scores) {
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
	cl_kernel insert_kernel;
	cl_kernel bc_forward;
	cl_kernel bc_reverse;
	cl_kernel push_frontier;
	cl_kernel bc_normalize;
	cl_kernel max_element;

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
	init_kernel = clCreateKernel(program, "init", &err);
	insert_kernel = clCreateKernel(program, "insert", &err);
	push_frontier = clCreateKernel(program, "push_frontier", &err);
	bc_forward = clCreateKernel(program, "bc_forward", &err);
	bc_reverse = clCreateKernel(program, "bc_reverse", &err);
	bc_normalize = clCreateKernel(program, "bc_normalize", &err);
	max_element = clCreateKernel(program, "max_element", &err);
	if (err < 0) { fprintf(stderr, "ERROR: create kernel failed, err code: %d\n", err); exit(1); }

	cl_mem d_row_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * (m+1), NULL, NULL);
	cl_mem d_column_indices = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * nnz, NULL, NULL);
	cl_mem d_scores = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(ScoreT) * m, NULL, NULL);
	cl_mem d_deltas = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(ScoreT) * m, NULL, NULL);
	cl_mem d_depths = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m, NULL, NULL);
	cl_mem d_path_counts = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m, NULL, NULL);
	cl_mem d_in_queue = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m, NULL, NULL);
	cl_mem d_out_queue = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m, NULL, NULL);
	cl_mem d_frontiers = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * (m + 1), NULL, NULL);
	cl_mem d_in_nitems = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
	cl_mem d_out_nitems = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
	cl_mem d_max_score = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(ScoreT), NULL, NULL);

	int out_nitems = 0, in_nitems = 0;
	err  = clEnqueueWriteBuffer(queue, d_row_offsets, CL_TRUE, 0, sizeof(int) * (m+1), h_row_offsets, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_column_indices, CL_TRUE, 0, sizeof(int) * nnz, h_column_indices, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_scores, CL_TRUE, 0, sizeof(int) * m, h_scores, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_in_nitems, CL_TRUE, 0, sizeof(int), &in_nitems, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_out_nitems, CL_TRUE, 0, sizeof(int), &out_nitems, 0, NULL, NULL);
	if (err < 0) { fprintf(stderr, "ERROR write buffer, err code: %d\n", err); exit(1); }

	err  = clSetKernelArg(init_kernel, 0, sizeof(int), &m);
	err |= clSetKernelArg(init_kernel, 1, sizeof(cl_mem), &d_depths);
	err |= clSetKernelArg(init_kernel, 2, sizeof(cl_mem), &d_path_counts);
	err |= clSetKernelArg(init_kernel, 3, sizeof(cl_mem), &d_deltas);
	if (err < 0) { fprintf(stderr, "ERROR set init kernel arg, err code: %d\n", err); exit(1); }

	err  = clSetKernelArg(insert_kernel, 0, sizeof(int), &src);
	err |= clSetKernelArg(insert_kernel, 1, sizeof(cl_mem), &d_in_nitems);
	err |= clSetKernelArg(insert_kernel, 2, sizeof(cl_mem), &d_in_queue);
	err |= clSetKernelArg(insert_kernel, 3, sizeof(cl_mem), &d_path_counts);
	err |= clSetKernelArg(insert_kernel, 4, sizeof(cl_mem), &d_depths);
	if (err < 0) { fprintf(stderr, "ERROR set insert kernel arg, err code: %d\n", err); exit(1); }

	err  = clSetKernelArg(bc_forward, 0, sizeof(cl_mem), &d_row_offsets);
	err |= clSetKernelArg(bc_forward, 1, sizeof(cl_mem), &d_column_indices);
	err |= clSetKernelArg(bc_forward, 2, sizeof(cl_mem), &d_path_counts);
	err |= clSetKernelArg(bc_forward, 3, sizeof(cl_mem), &d_depths);
	//err |= clSetKernelArg(bc_forward, 4, sizeof(int), &depth);
	err |= clSetKernelArg(bc_forward, 5, sizeof(cl_mem), &d_in_queue);
	err |= clSetKernelArg(bc_forward, 6, sizeof(cl_mem), &d_out_queue);
	err |= clSetKernelArg(bc_forward, 7, sizeof(cl_mem), &d_in_nitems);
	err |= clSetKernelArg(bc_forward, 8, sizeof(cl_mem), &d_out_nitems);
	if (err < 0) { fprintf(stderr, "ERROR set bc_forward kernel arg, err code: %d\n", err); exit(1); }

	//err  = clSetKernelArg(bc_reverse, 0, sizeof(int), &nitems);
	err |= clSetKernelArg(bc_reverse, 1, sizeof(cl_mem), &d_row_offsets);
	err |= clSetKernelArg(bc_reverse, 2, sizeof(cl_mem), &d_column_indices);
	//err |= clSetKernelArg(bc_reverse, 3, sizeof(int), &depth_index[d]);
	err |= clSetKernelArg(bc_reverse, 4, sizeof(cl_mem), &d_frontiers);
	err |= clSetKernelArg(bc_reverse, 5, sizeof(cl_mem), &d_scores);
	err |= clSetKernelArg(bc_reverse, 6, sizeof(cl_mem), &d_path_counts);
	err |= clSetKernelArg(bc_reverse, 7, sizeof(cl_mem), &d_depths);
	//err |= clSetKernelArg(bc_reverse, 8, sizeof(int), &depth);
	err |= clSetKernelArg(bc_reverse, 9, sizeof(cl_mem), &d_deltas);
	if (err < 0) { fprintf(stderr, "ERROR set bc_reverse kernel arg, err code: %d\n", err); exit(1); }

	err  = clSetKernelArg(bc_normalize, 0, sizeof(int), &m);
	err |= clSetKernelArg(bc_normalize, 1, sizeof(cl_mem), &d_scores);
	//err |= clSetKernelArg(bc_normalize, 2, sizeof(ScoreT), &max_score);
	if (err < 0) { fprintf(stderr, "ERROR set bc_normalize kernel arg, err code: %d\n", err); exit(1); }

	//err  = clSetKernelArg(push_frontier, 0, sizeof(int), &nitems);
	err |= clSetKernelArg(push_frontier, 1, sizeof(cl_mem), &d_in_queue);
	err |= clSetKernelArg(push_frontier, 2, sizeof(cl_mem), &d_frontiers);
	//err |= clSetKernelArg(push_frontier, 3, sizeof(int), &frontiers_len);
	if (err < 0) { fprintf(stderr, "ERROR set push_frontier kernel arg, err code: %d\n", err); exit(1); }

	err  = clSetKernelArg(max_element, 0, sizeof(int), &m);
	err |= clSetKernelArg(max_element, 1, sizeof(cl_mem), &d_scores);
	err |= clSetKernelArg(max_element, 2, sizeof(cl_mem), &d_max_score);
	if (err < 0) { fprintf(stderr, "ERROR set max_element kernel arg, err code: %d\n", err); exit(1); }

	int depth = 0;
	int frontiers_len = 0;
	vector<int> depth_index;
	depth_index.push_back(0);
	size_t globalSize, localSize;
	localSize = BLOCK_SIZE;
	globalSize = ceil(m/(float)localSize)*localSize;
	printf("Launching OpenCL BC solver (%ld threads/CTA) ...\n", localSize);

	// push the source vertex into the frontier
	err = clEnqueueNDRangeKernel(queue, init_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	if (err < 0) { fprintf(stderr, "ERROR enqueue init kernel, err code: %d\n", err); exit(1); }

	Timer t;
	t.Start();
	err = clEnqueueNDRangeKernel(queue, insert_kernel, 1, NULL, &localSize, &localSize, 0, NULL, NULL);
	if (err < 0) { fprintf(stderr, "ERROR enqueue insert kernel, err code: %d\n", err); exit(1); }
	do {
		err = clEnqueueReadBuffer(queue, d_in_nitems, CL_TRUE, 0, sizeof(int), &in_nitems, 0, NULL, NULL);
		if (err < 0) { fprintf(stderr, "ERROR enqueue read buffer 1, err code: %d\n", err); exit(1); }
		//printf(" %2d    %d\n", depth, in_nitems);
		globalSize = ceil(in_nitems/(float)localSize)*localSize;
		err = clSetKernelArg(push_frontier, 0, sizeof(int), &in_nitems);
		err = clSetKernelArg(push_frontier, 3, sizeof(int), &frontiers_len);
		err = clEnqueueNDRangeKernel(queue, push_frontier, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if (err < 0) { fprintf(stderr, "ERROR enqueue push_frontier kernel, err code: %d\n", err); exit(1);}
		frontiers_len += in_nitems;
		depth_index.push_back(frontiers_len);
		depth ++;
		err |= clSetKernelArg(bc_forward, 4, sizeof(int), &depth);
		err = clEnqueueNDRangeKernel(queue, bc_forward, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if (err < 0) { fprintf(stderr, "ERROR enqueue bc_forward kernel, err code: %d\n", err); exit(1); }

		// move the items from out_queue to in_queue
		err = clEnqueueReadBuffer(queue, d_out_nitems, CL_TRUE, 0, sizeof(int), &out_nitems, 0, NULL, NULL);
		err = clEnqueueCopyBuffer(queue, d_out_queue, d_in_queue, 0, 0, out_nitems * sizeof(int), 0, NULL, NULL);
		in_nitems = out_nitems;
		err = clEnqueueWriteBuffer(queue, d_in_nitems, CL_TRUE, 0, sizeof(int), &in_nitems, 0, NULL, NULL);
		out_nitems = 0;
		err = clEnqueueWriteBuffer(queue, d_out_nitems, CL_TRUE, 0, sizeof(int), &out_nitems, 0, NULL, NULL);
	} while(in_nitems > 0);

	int nitems = 0;
	for (int d = depth_index.size() - 2; d >= 0; d--) {
		nitems = depth_index[d+1] - depth_index[d];
		globalSize = ceil(nitems/(float)localSize)*localSize;
		//printf("Reverse: depth=%d, frontier_size=%d\n", d, nitems);
		err  = clSetKernelArg(bc_reverse, 0, sizeof(int), &nitems);
		err |= clSetKernelArg(bc_reverse, 3, sizeof(int), &depth_index[d]);
		err |= clSetKernelArg(bc_reverse, 8, sizeof(int), &d);
		err = clEnqueueNDRangeKernel(queue, bc_reverse, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if (err < 0) { fprintf(stderr, "ERROR enqueue bc_reverse kernel, err code: %d\n", err); exit(1); }
	}
	globalSize = ceil(m/(float)localSize)*localSize;
	err = clEnqueueNDRangeKernel(queue, max_element, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	if (err < 0) { fprintf(stderr, "ERROR enqueue max_element, err code: %d\n", err); exit(1); }

	ScoreT max_score;
	err = clEnqueueReadBuffer(queue, d_max_score, CL_TRUE, 0, sizeof(ScoreT), &max_score, 0, NULL, NULL);
	if (err < 0) { fprintf(stderr, "ERROR read buffer, err code: %d\n", err); exit(1); }
	//printf("max_score %f\n", max_score);

	err = clSetKernelArg(bc_normalize, 2, sizeof(ScoreT), &max_score);
	err = clEnqueueNDRangeKernel(queue, bc_normalize, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	if (err < 0) { fprintf(stderr, "ERROR enqueue bc_normalize kernel, err code: %d\n", err); exit(1); }
	clFinish(queue);
	t.Stop();

	printf("\titerations = %d.\n", depth);
	printf("\truntime [%s] = %f ms.\n", BC_VARIANT, t.Millisecs());
	err = clEnqueueReadBuffer(queue, d_scores, CL_TRUE, 0, sizeof(DistT) * m, h_scores, 0, NULL, NULL);
	if (err < 0) { fprintf(stderr, "ERROR enqueue read buffer, err code: %d\n", err); exit(1); }

	clReleaseMemObject(d_row_offsets);
	clReleaseMemObject(d_column_indices);
	clReleaseMemObject(d_depths);
	clReleaseMemObject(d_in_queue);
	clReleaseMemObject(d_out_queue);
	clReleaseMemObject(d_frontiers);

	clReleaseContext(context);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseKernel(init_kernel);
	clReleaseKernel(insert_kernel);
	clReleaseKernel(bc_forward);
	clReleaseKernel(bc_reverse);
	clReleaseKernel(bc_normalize);
	return;
}
