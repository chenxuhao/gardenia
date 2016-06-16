#pragma once

#define SSSP_WLW 1
#define SSSP_WLC 2
#define SSSP_FUSION 3
#define SSSP_LDB 4

#ifndef VARIANT
#error "VARIANT not defined."
#endif

#if VARIANT==SSSP_LDB
#include "sssp_ldb.h"
#elif VARIANT==SSSP_WLC
#include "sssp_wlc.h"
#elif VARIANT==SSSP_WLW
#include "sssp_wlw.h"
#elif VARIANT==SSSP_FUSION
#include "sssp_fusion.h"
#else 
#error "Unknown variant"
#endif
