#include <sys/time.h>
#include <cassert>

typedef unsigned foru;
#define MYINFINITY	1000000000
#define MAXNBLOCKS  (4*NBLOCKS)
#define BLOCKSIZE   256
#define MAXBLOCKSIZE    1024
#define MAXSHARED   (48*1024)
#define MAXSHAREDUINT   (MAXSHARED / 4)
#define SHAREDPERTHREAD (MAXSHAREDUINT / MAXBLOCKSIZE)

double rtclock()
{
	struct timezone Tzp;
	struct timeval Tp; 
	int stat;
	stat = gettimeofday (&Tp, &Tzp);
	if (stat != 0) printf("Error return from gettimeofday: %d",stat);
	return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

static unsigned CudaTest(char *msg)
{
	cudaError_t e;

	cudaThreadSynchronize();
	if (cudaSuccess != (e = cudaGetLastError())) {
		fprintf(stderr, "%s: %d\n", msg, e); 
		fprintf(stderr, "%s\n", cudaGetErrorString(e));
		exit(-1);
		//return 1;
	}
	return 0;
}

inline int ConvertSMVer2Cores(int major, int minor)
{
        // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
        typedef struct {
                int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
                int Cores;
        } sSMtoCores;

        sSMtoCores nGpuArchCoresPerSM[] =
        { { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
          { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
          { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
          { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
          { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
          { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
          { 0x30, 192}, // Fermi Generation (SM 3.0) GK10x class
	  { 0x35, 192}, // Kepler Generation (SM 3.5) GK110 class
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
