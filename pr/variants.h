#pragma once

#define PR_WLW 1
#define PR_WLC 2

#ifndef VARIANT
#error "VARIANT not defined."
#endif

#if VARIANT==PR_WLW
#include "pr_wlw.h"
#elif VARIANT==PR_WLC
#include "pr_wlc.h"
#else 
#error "Unknown variant"
#endif
