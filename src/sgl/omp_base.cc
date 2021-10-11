#include "sgl.h"
#include "sim.h"
#include "intersect.h"
#define NUM_SAMPLES 1000000

void SglSolver(Graph &g, Pattern &p, uint64_t &total) {
  int num_threads = 1;
  //#pragma omp parallel
  {
  //  num_threads = omp_get_num_threads();
  }
  printf("Launching OpenMP Sgl solver (%d threads) ...\n", num_threads);
  uint64_t counter = 0;
  std::vector<std::vector<uint8_t>> cmaps(num_threads);
  for (int tid = 0; tid < num_threads; tid ++) {
    auto &cmap = cmaps[tid];
    cmap.resize(g.size()); // the connectivity code
    std::fill(cmap.begin(), cmap.end(), 0);
  }
#ifdef PROFILE_LATENCY
  std::vector<uint64_t> nticks(num_threads, 0);
  std::vector<uint64_t> nqueries(num_threads, 0);
#endif

  Timer t;
  t.Start();
  roi_begin();
  // PLAN_HERE
#ifdef USE_CMAP
  if (p.is_house()) {
    #include "house_cmap.h"
  } else if (p.is_pentagon()) {
    #include "pentagon_cmap.h"
  } else if (p.is_rectangle()) {
    #include "rectangle_cmap.h"
  } else {
    #include "diamond_cmap.h"
  }
#else
  if (p.is_house()) {
    #include "house.h"
  } else if (p.is_pentagon()) {
    #include "pentagon.h"
  } else if (p.is_rectangle()) {
    #include "rectangle.h"
  } else {
    #include "diamond.h"
  }
#endif
  total = counter;
  roi_end();
  t.Stop();
#ifdef PROFILE_LATENCY
  uint64_t total_query_latency = 0;
  uint64_t total_num_queries = 0;
  for (int tid = 0; tid < num_threads; tid ++) {
    total_query_latency += nticks[tid];
    total_num_queries += nqueries[tid];
  }
  auto avg_query_latency = total_query_latency / total_num_queries;
  std::cout << "average c-map query latency: " <<  avg_query_latency << " cycles\n";
#endif
  std::cout << "runtime = " <<  t.Seconds() << " seconds\n";
  return;
}

