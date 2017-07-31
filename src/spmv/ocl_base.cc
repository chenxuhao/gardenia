// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "spmv.h"
#include <CL/cl.h>
#include <string.h>
#include "timer.h"
#define SPMV_VARIANT "ocl_base"

// local variables
static cl_context	    context;
static cl_command_queue cmd_queue;
static cl_device_type   device_type;
static cl_device_id   * device_list;
static cl_int           num_devices;

int initialize(int use_gpu) {
    cl_int result;
    size_t size;
    // create OpenCL context
    cl_platform_id platform_id;
    if (clGetPlatformIDs(1, &platform_id, NULL) != CL_SUCCESS) { printf("ERROR: clGetPlatformIDs(1,*,0) failed\n"); return -1; }
    cl_context_properties ctxprop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0};
    device_type = use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
    context = clCreateContextFromType( ctxprop, device_type, NULL, NULL, NULL );
    if( !context ) { fprintf(stderr, "ERROR: clCreateContextFromType(%s) failed\n", use_gpu ? "GPU" : "CPU"); return -1; }
    // get the list of GPUs
    result = clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &size );
    num_devices = (int) (size / sizeof(cl_device_id));
    printf("num_devices = %d\n", num_devices);
    if( result != CL_SUCCESS || num_devices < 1 ) { fprintf(stderr, "ERROR: clGetContextInfo() failed\n"); return -1; }
    device_list = new cl_device_id[num_devices];
    if( !device_list ) { fprintf(stderr, "ERROR: new cl_device_id[] failed\n"); return -1; }
    result = clGetContextInfo( context, CL_CONTEXT_DEVICES, size, device_list, NULL );
    if( result != CL_SUCCESS ) { fprintf(stderr, "ERROR: clGetContextInfo() failed\n"); return -1; }
    // create command queue for the first device
    cmd_queue = clCreateCommandQueue( context, device_list[0], 0, NULL );
    if( !cmd_queue ) { fprintf(stderr, "ERROR: clCreateCommandQueue() failed\n"); return -1; }
    return 0;
}

void SpmvSolver(int num_rows, int nnz, int *Ap, int *Aj, ValueType *Ax, ValueType *x, ValueType *y) {
	printf("Launching OpenCL SpMV solver ...\n");

	//load OpenCL kernel file
	cl_int err = 0;
	char *filechar = "base.cl";
	int sourcesize = 1024*1024;
	char * source = (char *)calloc(sourcesize, sizeof(char));
	if(!source) { printf("ERROR: calloc(%d) failed\n", sourcesize); return; }
	FILE * fp = fopen(filechar, "rb");
	if(!fp) { printf("ERROR: unable to open '%s'\n", filechar); return; }
	fread(source + strlen(source), sourcesize, 1, fp);
	fclose(fp);

	// OpenCL initialization
	if(initialize(1)) return;
	const char * slist[2] = { source, 0 };
	cl_program prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
	if(err != CL_SUCCESS) { printf("ERROR: clCreateProgramWithSource() => %d\n", err); return; }
	err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
	if(err != CL_SUCCESS) { printf("ERROR: clBuildProgram() => %d\n", err); return; }

	Timer t;
	t.Start();

	t.Stop();
	printf("\truntime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());
	return;
}
