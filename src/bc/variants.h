#pragma once

#define BC 0
#define BC_LB 1

#ifndef VARIANT
#error "VARIANT not defined."
#endif

#if VARIANT==BC
#include "bc.h"
#elif VARIANT==BC_LB
#include "bc_lb.h"
#else 
#error "Unknown variant"
#endif
