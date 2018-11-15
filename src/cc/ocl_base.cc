// Copyright 2016, National University of Defense Technology
// Author: Xuhao Chen <cxh@illinois.edu>
#include "cc.h"
#include "timer.h"
#include "ocl_util.h"
#include <string.h>
#define CC_VARIANT "ocl_base"

void CCSolver(int m, int nnz, IndexT *in_row_offsets, IndexT *in_column_indices, IndexT *row_offsets, IndexT *column_indices, int *degrees, CompT *comp, bool is_directed) {
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
	cl_kernel hook_kernel;
	cl_kernel shortcut_kernel;

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
	hook_kernel = clCreateKernel(program, "hook", &err);
	shortcut_kernel = clCreateKernel(program, "shortcut", &err);
	if (err < 0) { fprintf(stderr, "ERROR: create kernel failed, err code: %d\n", err); exit(1); }

	cl_mem d_row_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * (m+1), NULL, NULL);
	cl_mem d_column_indices = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * nnz, NULL, NULL);
	cl_mem d_comp = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(ValueT) * nnz, NULL, NULL);
	cl_mem d_changed = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(bool), NULL, NULL);

	err  = clEnqueueWriteBuffer(queue, d_row_offsets, CL_TRUE, 0, sizeof(int) * (m+1), row_offsets, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_column_indices, CL_TRUE, 0, sizeof(int) * nnz, column_indices, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_comp, CL_TRUE, 0, sizeof(ValueT) * nnz, comp, 0, NULL, NULL);
	if (err < 0) { fprintf(stderr, "ERROR: write buffer failed, err code: %d\n", err); exit(1); }

	err  = clSetKernelArg(hook_kernel, 0, sizeof(int), &m);
	err |= clSetKernelArg(hook_kernel, 1, sizeof(cl_mem), &d_row_offsets);
	err |= clSetKernelArg(hook_kernel, 2, sizeof(cl_mem), &d_column_indices);
	err |= clSetKernelArg(hook_kernel, 3, sizeof(cl_mem), &d_comp);
	err |= clSetKernelArg(hook_kernel, 4, sizeof(cl_mem), &d_changed);
	if (err < 0) { fprintf(stderr, "ERROR: set hook_kernel arg failed, err code: %d\n", err); exit(1); }

	err  = clSetKernelArg(shortcut_kernel, 0, sizeof(int), &m);
	err |= clSetKernelArg(shortcut_kernel, 1, sizeof(cl_mem), &d_comp);
	if (err < 0) { fprintf(stderr, "ERROR: set shortcut_kernel arg failed, err code: %d\n", err); exit(1); }

	size_t globalSize, localSize;
	localSize = BLOCK_SIZE;
	globalSize = ceil(m/(float)localSize)*localSize;
	printf("Launching OpenCL CC solver ...\n");

	Timer t;
	t.Start();
	int iter = 0;
	bool changed;
	do {
		++ iter;
		changed = false;
		err = clEnqueueWriteBuffer(queue, d_changed, CL_TRUE, 0, sizeof(changed), &changed, 0, NULL, NULL);
		if (err < 0) { fprintf(stderr, "ERROR write buffer, err code: %d\n", err); exit(1); }
		err = clEnqueueNDRangeKernel(queue, hook_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if (err < 0) { fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1); }
		err = clEnqueueNDRangeKernel(queue, shortcut_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if (err < 0) { fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1); }
		err = clEnqueueReadBuffer(queue, d_changed, CL_TRUE, 0, sizeof(changed), &changed, 0, NULL, NULL);
		if (err < 0) { fprintf(stderr, "ERROR enqueue read buffer, err code: %d\n", err); exit(1); }
	} while (changed);
	clFinish(queue);
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", CC_VARIANT, t.Millisecs());
	err = clEnqueueReadBuffer(queue, d_comp, CL_TRUE, 0, sizeof(CompT) * m, comp, 0, NULL, NULL);
	if (err < 0) { fprintf(stderr, "ERROR enqueue read buffer, err code: %d\n", err); exit(1); }

	clReleaseMemObject(d_row_offsets);
	clReleaseMemObject(d_column_indices);
	clReleaseMemObject(d_comp);
	clReleaseMemObject(d_changed);
	clReleaseProgram(program);
	clReleaseKernel(hook_kernel);
	clReleaseKernel(shortcut_kernel);
	clReleaseCommandQueue(queue);
	return;
}
