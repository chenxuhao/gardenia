#pragma once

#define TC 0
#define TC_LB 1

#ifndef VARIANT
#error "VARIANT not defined."
#endif

#if VARIANT==TC
#include "tc.h"
#elif VARIANT==TC_LB
#include "tc_lb.h"
#else 
#error "Unknown variant"
#endif
