#ifndef SEQUENTIAL
#define SEQUENTIAL

#include <stack>
#include <queue>
#include <vector>
#include <set>

#include "parse.h"

std::vector<float> bc_cpu(graph g, const std::set<int> &source_vertices);

#endif
