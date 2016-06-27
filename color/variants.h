#pragma once

#define DATA_BASE 0 // the baseline data-driven version
#define DATA_LDG 1 // using __ldg() intrinsic
#define DATA_BITSET 2 // bitset for forbiddenColors
#define DATA_COARSE 3 // thread coarsening
#define DATA_FUSION 4 // kernel fusion
#define DATA_WLC 5  // worklistc from lonestargpu
#define DATA_LDB 6  // load balancing using merrill's scheme
#define DATA_PQ 7
#define DATA_BEST 8
#define DATA_COMB1 9

#ifndef VARIANT
#error "VARIANT not defined."
#endif

#if VARIANT==DATA_BASE
#include "kernel_base.h"
#elif VARIANT==DATA_LDG
#include "kernel_ldg.h"
#elif VARIANT==DATA_BITSET
#include "kernel_bitset.h"
#elif VARIANT==DATA_COARSE
#include "kernel_tc.h"
#elif VARIANT==DATA_FUSION
#include "kernel_fusion.h"
#elif VARIANT==DATA_WLC
#include "kernel_wlc.h"
#elif VARIANT==DATA_LDB
#include "kernel_ldb.h"
#elif VARIANT==DATA_PQ
#include "kernel_pq.h"
#elif VARIANT==DATA_BEST
#include "kernel_best.h"
#elif VARIANT==DATA_COMB1
#include "kernel_comb.h"
#else 
#error "Unknown variant"
#endif
