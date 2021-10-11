#pragma once

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

