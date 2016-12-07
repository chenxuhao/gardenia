#pragma once

#define SGD_TOPO 0
#define SGD_WLW 1
#define SGD_WLC 2

#ifndef VARIANT
#error "VARIANT not defined."
#endif

#if VARIANT==SGD_TOPO
#include "sgd.h"
#elif VARIANT==SGD_WLW
#include "sgd_wlw.h"
#elif VARIANT==SGD_WLC
#include "sgd_wlc.h"
#else 
#error "Unknown variant"
#endif
