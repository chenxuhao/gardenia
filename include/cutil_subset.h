#ifndef CUTIL_SUBSET_H
#define CUTIL_SUBSET_H

#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                      \
    cudaError err = call;                                                      \
    if( cudaSuccess != err) {                                                  \
        fprintf(stderr, "error %d: Cuda error in file '%s' in line %i : %s.\n",\
                err, __FILE__, __LINE__, cudaGetErrorString( err) );           \
        exit(EXIT_FAILURE);                                                    \
    } }

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);              \

#  define CUDA_SAFE_THREAD_SYNC( ) {                                           \
    cudaError err = CUT_DEVICE_SYNCHRONIZE();                                  \
    if ( cudaSuccess != err) {                                                 \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",          \
                __FILE__, __LINE__, cudaGetErrorString( err) );                \
    } }

// from http://forums.nvidia.com/index.php?showtopic=186669
static __device__ unsigned get_smid(void) {
     unsigned ret;
     asm("mov.u32 %0, %smid;" : "=r"(ret) );
     return ret;
}

static unsigned CudaTest(const char *msg) {
	cudaError_t e;
	//cudaThreadSynchronize();
	if (cudaSuccess != (e = cudaGetLastError())) {
		fprintf(stderr, "%s: %d\n", msg, e); 
		fprintf(stderr, "%s\n", cudaGetErrorString(e));
		exit(-1);
		//return 1;
	}
	return 0;
}

inline int ConvertSMVer2Cores(int major, int minor) {
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct {
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = { 
		{ 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
		{ 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
		{ 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
		{ 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
		{ 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192}, // Fermi Generation (SM 3.0) GK10x class
		{ 0x35, 192}, // Kepler Generation (SM 3.5) GK110 class
		{ 0x37, 192}, // Kepler Generation (SM 3.7) GK210 class
		{   -1, -1 }
	};

	int index = 0;
	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
			return nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}
	printf("MapSMtoCores SM %d.%d is undefined (please update to the latest SDK)!\n", major, minor);
	return -1;
}

static void print_device_info(int device) {
	int deviceCount = 0;
	CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));
	CUDA_SAFE_CALL(cudaSetDevice(device));
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device));
	int nSM = deviceProp.multiProcessorCount;
	fprintf(stdout, "Found %d devices, using device %d (%s), compute capability %d.%d, cores %d*%d.\n", 
			deviceCount, device, deviceProp.name, deviceProp.major, deviceProp.minor, nSM, ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
}

#include <cusparse_v2.h>
/*
static const char * cusparseGetErrorString(cusparseStatus_t error) {
	// Read more at: http://docs.nvidia.com/cuda/cusparse/index.html#ixzz3f79JxRar
	switch (error) {
		case CUSPARSE_STATUS_SUCCESS:
			return "The operation completed successfully.";
		case CUSPARSE_STATUS_NOT_INITIALIZED:
			return "The cuSPARSE library was not initialized. This is usually caused by the lack of a prior call, an error in the CUDA Runtime API called by the cuSPARSE routine, or an error in the hardware setup.\n" \
				"To correct: call cusparseCreate() prior to the function call; and check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed.";

		case CUSPARSE_STATUS_ALLOC_FAILED:
			return "Resource allocation failed inside the cuSPARSE library. This is usually caused by a cudaMalloc() failure.\n"\
				"To correct: prior to the function call, deallocate previously allocated memory as much as possible.";

		case CUSPARSE_STATUS_INVALID_VALUE:
			return "An unsupported value or parameter was passed to the function (a negative vector size, for example).\n"\
				"To correct: ensure that all the parameters being passed have valid values.";

		case CUSPARSE_STATUS_ARCH_MISMATCH:
			return "The function requires a feature absent from the device architecture; usually caused by the lack of support for atomic operations or double precision.\n"\
				"To correct: compile and run the application on a device with appropriate compute capability, which is 1.1 for 32-bit atomic operations and 1.3 for double precision.";

		case CUSPARSE_STATUS_MAPPING_ERROR:
			return "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.\n"\
				"To correct: prior to the function call, unbind any previously bound textures.";

		case CUSPARSE_STATUS_EXECUTION_FAILED:
			return "The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.\n"\
				"To correct: check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed.";

		case CUSPARSE_STATUS_INTERNAL_ERROR:
			return "An internal cuSPARSE operation failed. This error is usually caused by a cudaMemcpyAsync() failure.\n"\
				"To correct: check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed. Also, check that the memory passed as a parameter to the routine is not being deallocated prior to the routine’s completion.";

		case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
			return "The matrix type is not supported by this function. This is usually caused by passing an invalid matrix descriptor to the function.\n"\
				"To correct: check that the fields in cusparseMatDescr_t descrA were set correctly.";
	}

	return "<unknown>";
}
//*/
static void CudaSparseCheckCore(cusparseStatus_t code, const char *file, int line) {
	if (code != CUSPARSE_STATUS_SUCCESS) {
		fprintf(stderr,"Cuda Error %d : %s %s %d\n", code, cusparseGetErrorString(code), file, line);
		exit(code);
	}
}

#define CudaSparseCheck( test ) { CudaSparseCheckCore((test), __FILE__, __LINE__); }

#endif
