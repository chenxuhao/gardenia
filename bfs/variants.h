#pragma once

#define BFS_WLW 1
#define BFS_WLC 2
#define BFS_TOPO 3
#define BFS_MERRILL 4

#ifndef VARIANT
#error "VARIANT not defined."
#endif

#if VARIANT==BFS_MERRILL
#include "bfs_merrill.h"
#elif VARIANT==BFS_WLC
#include "bfs_wlc.h"
#elif VARIANT==BFS_WLW
#include "bfs_wlw.h"
#elif VARIANT==BFS_TOPO
#include "bfs_topo.h"
#else 
#error "Unknown variant"
#endif
