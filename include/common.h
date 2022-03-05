#pragma once

#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

#include <set>
#include <map>
#include <vector>
#include <limits>
#include <cassert>
#include <cstring>
#include <climits>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <unordered_map>

using namespace std;

typedef uint8_t BYTE;
typedef uint8_t mask_t;
typedef uint8_t label_t;
typedef uint8_t vlabel_t;
typedef uint8_t elabel_t;
typedef uint8_t cmap_vt; // cmap value type
typedef int32_t vidType;
typedef int64_t eidType;
typedef unsigned long long AccType;

//#define LONG_TYPES
#ifndef LONG_TYPES
typedef float ScoreT;
typedef float ValueT;
typedef float LatentT;
typedef int DistT;
typedef int CompT;
typedef int IndexT;
typedef int WeightT;
#else
typedef double ScoreT;
typedef double ValueT;
typedef double LatentT;
typedef int64_t DistT;
typedef int64_t CompT;
typedef int64_t IndexT;
typedef int64_t WeightT;
#endif

typedef int32_t VertexId;
typedef int32_t VertexID;
typedef int64_t EdgeID;
typedef std::vector<VertexId> VertexList;
typedef std::vector<BYTE> ByteList;

#define PAGE_SIZE 4096
#define	MAXCOLOR 128 // assume graph can be colored with less than 128 colors
#define MYINFINITY	1000000000
#define BLOCK_SIZE  256
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
#define FULL_MASK 0xffffffff
#define ADJ_SIZE_THREASHOLD 1024
#define NUM_BUCKETS 128
#define BUCKET_SIZE 1024

