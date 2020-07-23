// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "spmv.h"
#include "timer.h"
#include "ocl_util.h"
#include <string.h>
#define SPMV_VARIANT "ocl_base"

void SpmvSolver(int m, int nnz, IndexT *ApT, IndexT *AjT, ValueT *AxT, int *Ap, int *Aj, ValueT *Ax, ValueT *x, ValueT *y, int *degrees) {
	//print_device_info();

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
	cl_kernel kernel;

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
	kernel = clCreateKernel(program, "spmv_kernel", &err);
	if (err < 0) { fprintf(stderr, "ERROR: create kernel failed, err code: %d\n", err); exit(1); }

	cl_mem d_Ap = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * (m+1), NULL, NULL);
	cl_mem d_Aj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * nnz, NULL, NULL);
	cl_mem d_Ax = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(ValueT) * nnz, NULL, NULL);
	cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(ValueT) * nnz, NULL, NULL);
	cl_mem d_y = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(ValueT) * m, NULL, NULL);

	err = clEnqueueWriteBuffer(queue, d_Ap, CL_TRUE, 0, sizeof(int) * (m+1), Ap, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_Aj, CL_TRUE, 0, sizeof(int) * nnz, Aj, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_Ax, CL_TRUE, 0, sizeof(ValueT) * nnz, Ax, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_x, CL_TRUE, 0, sizeof(ValueT) * nnz, x, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_y, CL_TRUE, 0, sizeof(ValueT) * m, y, 0, NULL, NULL);
	if (err < 0) { fprintf(stderr, "ERROR: write buffer failed, err code: %d\n", err); exit(1); }

	err = clSetKernelArg(kernel, 0, sizeof(int), &m);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_Ap);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_Aj);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_Ax);
	err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_x);
	err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_y);
	if (err < 0) { fprintf(stderr, "ERROR: set kernel arg failed, err code: %d\n", err); exit(1); }

	size_t globalSize, localSize;
	localSize = BLOCK_SIZE;
	globalSize = ceil(m/(float)localSize)*localSize;
	printf("Launching OpenCL SpMV solver ...\n");

	Timer t;
	t.Start();
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	if (err < 0) { fprintf(stderr, "ERROR: enqueue nd range failed, err code: %d\n", err); exit(1); }
	clFinish(queue);
	t.Stop();

	printf("\truntime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());
	err = clEnqueueReadBuffer(queue, d_y, CL_TRUE, 0, sizeof(ValueT) * m, y, 0, NULL, NULL);
	if (err < 0) { fprintf(stderr, "ERROR enqueue read buffer, err code: %d\n", err); exit(1); }

	clReleaseMemObject(d_Ap);
	clReleaseMemObject(d_Aj);
	clReleaseMemObject(d_Ax);
	clReleaseMemObject(d_x);
	clReleaseMemObject(d_y);

	clReleaseContext(context);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	return;
}
