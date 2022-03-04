
// ad-hoc 4-clique
void automine_4clique(Graph &g, std::vector<uint64_t> &counters) {
   tbb::parallel_for(tbb::blocked_range<vidType>(0, g.size()),
              [&g, &counters](tbb::blocked_range<vidType> &r) {
    auto tid = tbb::task_arena::current_thread_index();
    auto &counter = counters.at(tid);
    for (vidType v1 = r.begin(); v1 != r.end(); v1++) {
      auto y1 = g.N(v1);
      for (auto v2 : y1) {
        auto y1y2 = intersection_set(y1, g.N(v2));
        for (auto v3 : y1y2) {
          counter += intersection_num(y1y2, g.N(v3));
        }
      }
    }
  });
}

// ad-hoc 5-clique
void automine_5clique(Graph &g, std::vector<uint64_t> &counters) {
  tbb::parallel_for(tbb::blocked_range<vidType>(0, g.size()),
              [&g, &counters](tbb::blocked_range<vidType> &r) {
    auto tid = tbb::task_arena::current_thread_index();
    auto &counter = counters.at(tid);
    for (vidType v1 = r.begin(); v1 != r.end(); v1++) {
      auto y1 = g.N(v1);
      for (auto v2 : y1) {
        auto y1y2 = intersection_set(y1, g.N(v2));
        for (auto v3 : y1y2) {
          auto y1y2y3 = intersection_set(y1y2, g.N(v3));
          for (auto v4 : y1y2y3) {
            counter += intersection_num(y1y2y3, g.N(v4));
          }
        }
      }
    }
  });
}

void automine_kclique(Graph &g, unsigned k, std::vector<uint64_t> &counters) {
  if (k == 4) {
    automine_4clique(g, counters);
  } else if (k == 5) {
    automine_5clique(g, counters);
  } else {
    std::cout << "Not implemented yet\n";
    exit(0);
  }
}

