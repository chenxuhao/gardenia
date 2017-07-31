#ifndef COMMON_H_
#define COMMON_H_

#include <stdio.h>
#include <cassert>
//#include <vector>
//#include <algorithm>
//#include <iomanip>
#include <limits>
#include <climits>
#include <math.h>
using namespace std;
#define LONG_TYPES
#ifndef LONG_TYPES
typedef float ScoreT;
typedef float WeightT;
typedef float ValueType;
typedef int CompT;
typedef unsigned DistT;
typedef int IndexType;
#else
typedef double ScoreT;
typedef double WeightT;
typedef double ValueType;
typedef long int CompT;
typedef long unsigned int DistT;
typedef long unsigned int IndexType;
#endif
const float kDamp = 0.85;

#define	MAXCOLOR 128 // assume graph can be colored with less than 128 colors
#define MYINFINITY	1000000000
//#define BLOCK_SIZE  256
#define BLOCK_SIZE  128
#define WARP_SIZE   32
#define MAXBLOCKSIZE    1024
#define MAXSHARED   (48*1024)
#define MAXSHAREDUINT   (MAXSHARED / 4)
#define SHAREDPERTHREAD (MAXSHAREDUINT / MAXBLOCKSIZE)
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)
#define MAX_THREADS (30 * 1024)
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define MAX_BLOCKS (MAX_THREADS / BLOCK_SIZE)
#define LOG_WARP_SIZE 5
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

#endif
