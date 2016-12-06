#pragma once

#define CC_TOPO 0
#define CC_WLW 1
#define CC_WLC 2

#ifndef VARIANT
#error "VARIANT not defined."
#endif

#if VARIANT==CC_TOPO
#include "cc.h"
#elif VARIANT==CC_WLW
#include "cc_wlw.h"
#elif VARIANT==CC_WLC
#include "cc_wlc.h"
#else 
#error "Unknown variant"
#endif
