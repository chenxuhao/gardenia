// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>
#pragma once
#include "common.h"
#include "graph.hh"
#include "pattern.h"
/*
GARDENIA Benchmark Suite
Kernel: Subgraph Counting (SC)
Author: Xuhao Chen

Will count the occurrances of a given arbitrary pattern
*/

void SglSolver(Graph &g, Pattern &p, uint64_t &total);
void SglVerifier(const Graph &g, uint64_t test_total);
