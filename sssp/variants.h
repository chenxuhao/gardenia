#pragma once

#define SSSP_WLW 1
#define SSSP_WLC 2
#define SSSP_TOPO 3

#ifndef VARIANT
#error "VARIANT not defined."
#endif

#if VARIANT==SSSP_TOPO
#include "sssp_topo.h"
#elif VARIANT==SSSP_WLC
#include "sssp_wlc.h"
#elif VARIANT==SSSP_WLW
#include "sssp_wlw.h"
#else 
#error "Unknown variant"
#endif
