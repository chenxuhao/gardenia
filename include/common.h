#pragma once

#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <limits>
#include <cassert>
#include <climits>
#include <cstdint>
//#include <algorithm>

using namespace std;
//#define LONG_TYPES
typedef uint8_t BYTE;
#ifndef LONG_TYPES
typedef float ScoreT;
typedef float ValueT;
typedef float LatentT;
typedef uint32_t DistT;
typedef int CompT;
typedef int IndexT;
typedef int WeightT;
#else
typedef double ScoreT;
typedef double ValueT;
typedef double LatentT;
typedef uint64_t DistT;
typedef int64_t CompT;
typedef int64_t IndexT;
typedef int64_t WeightT;
#endif
extern double hub_factor;

typedef int VertexId;
typedef std::vector<VertexId> VertexList;
typedef std::vector<BYTE> ByteList;

#define PAGE_SIZE 4096
#define	MAXCOLOR 128 // assume graph can be colored with less than 128 colors
#define MYINFINITY	1000000000
#define BLOCK_SIZE  256
//#define BLOCK_SIZE  128
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

