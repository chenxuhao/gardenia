#pragma once

#define TC_TOPO 0
#define TC_WLW 1
#define TC_WLC 2

#ifndef VARIANT
#error "VARIANT not defined."
#endif

#if VARIANT==TC_TOPO
#include "tc.h"
#elif VARIANT==TC_WLW
#include "tc_wlw.h"
#elif VARIANT==TC_WLC
#include "tc_wlc.h"
#else 
#error "Unknown variant"
#endif
