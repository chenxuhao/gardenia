#pragma once

#define CU_BASE 0
#define CU_LB 1
#define OMP_BASE 2

#ifndef VARIANT
#error "VARIANT not defined."
#endif

#if VARIANT==CU_BASE
#include "cu_base.h"
#elif VARIANT==CU_LB
#include "cu_lb.h"
#elif VARIANT==OMP_BASE
#include "omp_base.h"
#else 
#error "Unknown variant"
#endif
