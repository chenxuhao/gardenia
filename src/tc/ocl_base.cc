// Author: Xuhao Chen <cxh@mit.edu>
#include "timer.h"
#include "graph.hh"
#include "ocl_util.h"

void TCSolver(Graph &g, uint64_t &total) {
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

  eidType* row_offsets = g.out_rowptr();
  vidType* column_indices = g.out_colidx();
  auto m = g.V();
  auto nnz = g.E();

  cl_platform_id platforms[32];
  cl_uint num_platforms;
  cl_device_id devices[32];
  cl_uint num_devices;
  char deviceName[1024];

  cl_int err = 0;
  err = clGetPlatformIDs(32, platforms, &num_platforms);
  if (err < 0) { fprintf(stderr, "ERROR clGetPlatformIDs failed, err code: %d\n", err); exit(1); }

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
  kernel = clCreateKernel(program, "ordered_count", &err);
  if (err < 0) { fprintf(stderr, "ERROR: create kernel failed, err code: %d\n", err); exit(1); }

  cl_mem d_row_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * (m+1), NULL, NULL);
  cl_mem d_column_indices = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * nnz, NULL, NULL);
  cl_mem d_total = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
  int h_total = 0;

  err  = clEnqueueWriteBuffer(queue, d_row_offsets, CL_TRUE, 0, sizeof(int) * (m+1), row_offsets, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(queue, d_column_indices, CL_TRUE, 0, sizeof(int) * nnz, column_indices, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(queue, d_total, CL_TRUE, 0, sizeof(int), &h_total, 0, NULL, NULL);
  if (err < 0) { fprintf(stderr, "ERROR write buffer, err code: %d\n", err); exit(1); }

  err  = clSetKernelArg(kernel, 0, sizeof(int), &m);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_row_offsets);
  err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_column_indices);
  err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_total);
  if (err < 0) { fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1); }

  size_t globalSize, localSize;
  localSize = BLOCK_SIZE;
  globalSize = ceil(m/(float)localSize)*localSize;
  std::cout << "Launching OpenCL TC solver...\n";

  Timer t;
  t.Start();
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
  if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}
  clFinish(queue);
  t.Stop();

  std::cout << "runtime [ocl_base] = " << t.Seconds() << " sec\n";
  err = clEnqueueReadBuffer(queue, d_total, CL_TRUE, 0, sizeof(int), &h_total, 0, NULL, NULL);
  if(err < 0){fprintf(stderr, "ERROR enqueue read buffer, err code: %d\n", err); exit(1);}
  total = (size_t)h_total;

  clReleaseMemObject(d_row_offsets);
  clReleaseMemObject(d_column_indices);
  clReleaseMemObject(d_total);

  clReleaseContext(context);
  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  return;
}
