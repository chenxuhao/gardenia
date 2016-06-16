#pragma once

#define PR 1

#ifndef VARIANT
#error "VARIANT not defined."
#endif

#if VARIANT==PR
#include "pr.h"
#else 
#error "Unknown variant"
#endif
