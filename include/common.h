#ifndef COMMON_H_
#define COMMON_H_

#include <sys/time.h>
#include <stdio.h>
#include <cassert>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <math.h>
#define BLKSIZE 256
#define	MAXCOLOR 128 // assume graph can be colored with less than 128 colors
using namespace std;
typedef float ScoreT;
typedef unsigned DistT;
typedef int CompT;
const float kDamp = 0.85;

#define MYINFINITY	1000000000
#define MAXNBLOCKS  (4*NBLOCKS)
#define BLOCKSIZE   256
#define MAXBLOCKSIZE    1024
#define MAXSHARED   (48*1024)
#define MAXSHAREDUINT   (MAXSHARED / 4)
#define SHAREDPERTHREAD (MAXSHAREDUINT / MAXBLOCKSIZE)

#endif
