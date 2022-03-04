// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>

//#define USE_CMAP
#include <omp.h>
#include "motif.h"
#include "sim.h"
#include "emb_list.h"
#include "automine_base.h"
#include "ccode_base.h"

void extend_motif(unsigned level, unsigned k, Graph &g, EmbList &emb_list, 
                  std::vector<uint8_t> &ccodes, std::vector<uint64_t> &counter) {
  if (level == k - 2) {
    for (int pcode = 0; pcode < num_possible_patterns[level+1]; pcode++) {
      for (VertexID emb_id = 0; emb_id < emb_list.size(level, pcode); emb_id++) {
        auto v = emb_list.get_vertex(level, emb_id, pcode);
        //std::cout << "\t\t v3 = " << v << "\n";
        //std::cout << "\t v2 = " << v << "\n";
        emb_list.push_history(v);
        update_ccodes(level, g, v, ccodes);
        uint8_t src_idx = 0;
        if (k > 3) src_idx = emb_list.get_src(level, emb_id, pcode);
        for (unsigned id = 0; id < level+1; id++) {
          auto src   = emb_list.get_history(id);
          auto begin = g.edge_begin(src);
          auto end = g.edge_end(src);
          for (auto e = begin; e < end; e++) {
            auto dst = g.getEdgeDst(e);
            auto emb_ptr = emb_list.get_history_ptr();
            if (do_early_break(k, level, dst, id, emb_ptr)) break;
            auto ccode = ccodes[dst];
            //std::cout << "\t\t\t v4 = " << dst << " ccode = " << unsigned(ccode)
            //          << " pos = " << id << " emb = " << emb_list.to_string() << "\n";
            //std::cout << "\t\t v3 = " << dst << " ccode = " << unsigned(ccode) << "\n";
            if (is_canonical(k, level, dst, id, ccode, emb_ptr)) {
              auto pid = get_pattern_id(level, ccode, pcode, src_idx);
              //std::cout << "\t\t\t found pid = " << pid << "\n";
              //std::cout << "\t\t\t\t found pid = " << pid << "\n";
              counter[pid] ++;
            }
          }
        }
        resume_ccodes(level, g, v, ccodes);
        emb_list.pop_history();
      }
    }
    return;
  }
  for (int pcode = 0; pcode < num_possible_patterns[level+1]; pcode++) {
    for (VertexID emb_id = 0; emb_id < emb_list.size(level, pcode); emb_id++) {
      auto v = emb_list.get_vertex(level, emb_id, pcode);
      //std::cout << "\t v2 = " << v << "\n";
      emb_list.push_history(v);
      update_ccodes(level, g, v, ccodes);
      emb_list.clear_size(level+1);
      for (unsigned idx = 0; idx < level+1; idx++) {
        auto src   = emb_list.get_history(idx);
        auto begin = g.edge_begin(src);
        auto end = g.edge_end(src);
        for (auto e = begin; e < end; e++) {
          auto dst = g.getEdgeDst(e);
          auto emb_ptr = emb_list.get_history_ptr();
          if (do_early_break(k, level, dst, idx, emb_ptr)) break;
          uint8_t pcode = 0;
          auto ccode = ccodes[dst];
          if (is_canonical(k, level, dst, idx, ccode, emb_ptr)) {
            auto pid = get_pattern_id(level, ccode, pcode, idx);
            emb_list.add_emb(level+1, dst, pid, idx);
          }
        }
      }
      extend_motif(level+1, k, g, emb_list, ccodes, counter);
      resume_ccodes(level, g, v, ccodes);
      emb_list.pop_history();
    }
  }
}

void kmotif(Graph &g, unsigned k, std::vector<std::vector<uint64_t>> &counters,
                  std::vector<EmbList> &emb_lists, 
                  std::vector<std::vector<uint8_t>> &ccodes) {
  #pragma omp parallel
  {
    auto tid = omp_get_thread_num();
    auto &counter = counters.at(tid);
    auto &local_ccodes = ccodes[tid];
    auto &emb_list = emb_lists[tid];
    #pragma omp for schedule(dynamic, 1) nowait
    for (VertexID v = 0; v < g.size(); v ++) {
      //std::cout << "v1 = " << v << "\n";
      emb_list.clear_size(1);
      auto begin = g.edge_begin(v);
      auto end = g.edge_end(v);
      for (auto e = begin; e < end; e++) {
        auto dst = g.getEdgeDst(e);
        local_ccodes[dst] = 1;
        if (dst < v) emb_list.add_emb(1, dst);
      }
      emb_list.push_history(v);
      extend_motif(1, k, g, emb_list, local_ccodes, counter);
      for (auto e = begin; e < end; e++) {
        auto dst = g.getEdgeDst(e);
        local_ccodes[dst] = 0;
      }
      emb_list.pop_history();
    }
  }
}

void MotifSolver(Graph &g, unsigned k, std::vector<uint64_t> &total) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("Launching OpenMP Motif solver (%d threads) ...\n", num_threads);

  int num_patterns = num_possible_patterns[k];
  std::vector<std::vector<uint64_t>> global_counters(num_threads);
  std::vector<EmbList> emb_lists(num_threads);
  std::vector<std::vector<uint8_t>> ccodes(num_threads);
  auto max_degree = g.get_max_degree();
  for (int tid = 0; tid < num_threads; tid ++) {
    auto &local_counters = global_counters[tid];
    local_counters.resize(num_patterns);
    std::fill(local_counters.begin(), local_counters.end(), 0);
    auto &local_ccodes = ccodes[tid];
    local_ccodes.resize(g.size()); // the connectivity code
    std::fill(local_ccodes.begin(), local_ccodes.end(), 0);
    auto &emb_list = emb_lists[tid];
    emb_list.init(k, max_degree, num_patterns);
  }

  Timer t;
  t.Start();
  roi_begin();
#ifdef USE_CMAP
  //kmotif(g, k, global_counters, emb_lists, ccodes);
  ccode_kmotif(g, k, global_counters, ccodes);
#else
  automine_kmotif(g, k, global_counters);
#endif
  for (int tid = 0; tid < num_threads; tid++)
    for (int pid = 0; pid < num_patterns; pid++)
      total[pid] += global_counters[tid][pid];
  roi_end();
  t.Stop();
  for (int i = 0; i < num_patterns; i++) { 
    std::cout << "pattern " << i << ": " << total[i] << "\n";
  }
  printf("runtime [%s] = %f s\n", "omp_base", t.Seconds());
}

