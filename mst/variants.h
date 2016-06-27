#pragma once

#define BFS_WLW 1
#define BFS_WLC 2
#define BFS_FUSION 3
#define BFS_LDB 4

#ifndef VARIANT
#error "VARIANT not defined."
#endif

#if VARIANT==BFS_LDB
#include "bfs_ldb.h"
#elif VARIANT==BFS_WLC
#include "bfs_wlc.h"
#elif VARIANT==BFS_WLW
#include "bfs_wlw.h"
#elif VARIANT==BFS_FUSION
#include "bfs_fusion.h"
#else 
#error "Unknown variant"
#endif
