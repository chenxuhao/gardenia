#pragma once

// ccode map data structure
#ifndef USE_HASHMAP
class cmap_t {
private:
  std::vector<uint8_t> cmap; // ccode map
public:
  cmap_t() {}
  ~cmap_t() {}
  void init(size_t n) {
    cmap.resize(n);
    std::fill(cmap.begin(), cmap.end(), 0);
  }
  uint8_t get(uint32_t k) { return cmap[k]; }
  void set(uint32_t k, uint8_t v) { cmap[k] = v; }
};
#else
#include <boost/unordered_map.hpp>
/*
class HashEntry {
public:
  uint32_t key;
  uint8_t value;
  HashEntry(uint32_t k, uint8_t v) : 
    key(k), value(v) {};
};

class HashMap {
private:
  std::vector<HashEntry> buckets;
  int hashFunc(uint32_t key) {
    return key % SIZE;
  }
  std::vector<HashEntry>& getBucket(uint32_t key);
  std::vector<HashEntry>& HashMap::getBucket(uint32_t key) {
    return buckets[hashFunc(key)];
  }
public:
  void init(size_t n) { buckets.resize(n); }
  //bool keyExists(int k);
  HashEntry get(int k) {
    std::vector<HashEntry>& bucket = getBucket(k);
    for (HashEntry entry : bucket) {
      if (entry.key == k) {
        return entry;
      }
    }
  }
  bool set(int k, int v) {
    std::vector<HashEntry>& bucket = getBucket(k);
    if (keyExists(k)) return false;
    bucket.push_back(HashEntry(k, v));
    return true;
  }
  //bool remove(int k);
  void clear() { buckets.clear(); }
};
//*/
class cmap_t {
private:
  //galois::gstl::UnorderedMap<uint32_t, uint8_t> cmap; // ccode map
  //boost::unordered_map<uint32_t, uint8_t> cmap; // ccode map
  std::unordered_map<uint32_t, uint8_t> cmap; // ccode map
public:
  cmap_t() {}
  ~cmap_t() {}
  void init(size_t n) {
    cmap.reserve(n);
  }
  uint8_t get(uint32_t k) const {
    auto it = cmap.find(k);
    if (it == cmap.end()) return 0; 
    return it->second;
  }
  void set(uint32_t k, uint8_t v) {
    cmap[k] = v;
    //auto it = cmap.find(k);
    //if (it != cmap.end()) it->second = v;
    //else cmap[k] = v;
  }
  void clear() { cmap.clear(); }
};
#endif

static inline bool is_clique(uint32_t level, vidType v, cmap_t &cmap) {
  return cmap.get(v) == uint8_t(level);
}

