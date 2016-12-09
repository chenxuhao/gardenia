#pragma once

#define SGD 0
#define SGD_LB 1

#ifndef VARIANT
#error "VARIANT not defined."
#endif

#if VARIANT==SGD
#include "sgd.h"
#elif VARIANT==SGD_LB
#include "sgd_lb.h"
#else 
#error "Unknown variant"
#endif
