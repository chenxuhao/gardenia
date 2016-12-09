#pragma once

#define PR_SCATTER 0
#define PR_GATHER 1
#define PR_LB 2

#ifndef VARIANT
#error "VARIANT not defined."
#endif

#if VARIANT==PR_SCATTER
#include "pr_scatter.h"
#elif VARIANT==PR_GATHER
#include "pr_gather.h"
#elif VARIANT==PR_LB
#include "pr_lb.h"
#else 
#error "Unknown variant"
#endif
