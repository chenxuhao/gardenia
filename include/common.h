#ifndef COMMON_H_
#define COMMON_H_

#include <sys/time.h>
#include <stdio.h>
#include <cassert>
#include <vector>
#include <algorithm>
typedef int NodeID;
typedef float ScoreT;
using namespace std;

#define MYINFINITY	1000000000
#define MAXNBLOCKS  (4*NBLOCKS)
#define BLOCKSIZE   256
#define MAXBLOCKSIZE    1024
#define MAXSHARED   (48*1024)
#define MAXSHAREDUINT   (MAXSHARED / 4)
#define SHAREDPERTHREAD (MAXSHAREDUINT / MAXBLOCKSIZE)


#endif
