#pragma once

#define PR_TOPO 0
#define PR_WLW 1
#define PR_WLC 2

#ifndef VARIANT
#error "VARIANT not defined."
#endif

#if VARIANT==PR_TOPO
#include "pr_topo.h"
#elif VARIANT==PR_WLW
#include "pr_wlw.h"
#elif VARIANT==PR_WLC
#include "pr_wlc.h"
#else 
#error "Unknown variant"
#endif
