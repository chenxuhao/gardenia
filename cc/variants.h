#pragma once

#define CC 0
#define CC_LB 1

#ifndef VARIANT
#error "VARIANT not defined."
#endif

#if VARIANT==CC
#include "cc.h"
#elif VARIANT==CC_LB
#include "cc_lb.h"
#else 
#error "Unknown variant"
#endif
