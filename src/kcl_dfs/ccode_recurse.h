
static inline void reset_ccodes(unsigned level, EmbList &emb_list, std::vector<uint8_t> &ccodes) {
  for (size_t id = 0; id < emb_list.size(level+1); id++) {
    auto v = emb_list.get_vertex(level+1, id);
    ccodes[v] = level;
  }
}

void extend_clique(unsigned level, unsigned k, Graph &g, EmbList &emb_list, std::vector<uint8_t> &ccodes, uint64_t &counter) {
  if (level == k - 2) {
    for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id++) {
      auto v     = emb_list.get_vertex(level, emb_id);
      auto begin = g.edge_begin(v);
      auto end = g.edge_end(v);
      for (auto e = begin; e < end; e++) {
        auto dst = g.getEdgeDst(e);
        auto ccode = ccodes[dst];
        if (ccode == level) counter ++;
      }
    }
    return;
  }
  for (size_t emb_id = 0; emb_id < emb_list.size(level); emb_id++) {
    auto v     = emb_list.get_vertex(level, emb_id);
    emb_list.set_size(level+1, 0);
    auto begin = g.edge_begin(v);
    auto end = g.edge_end(v);
    for (auto e = begin; e < end; e++) {
      auto dst = g.getEdgeDst(e);
      auto ccode = ccodes[dst];
      if (ccode == level) {
        emb_list.add_emb(level+1, dst);
        ccodes[dst] = level+1;
      }
    }
    extend_clique(level+1, k, g, emb_list, ccodes, counter);
    reset_ccodes(level, emb_list, ccodes);
  }
}

