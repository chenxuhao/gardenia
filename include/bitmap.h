// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef BITMAP_H_
#define BITMAP_H_

//#include <algorithm>
//#include <cinttypes>
#include "platform_atomics.h"
typedef unsigned long uint64_t;
/*
GAP Benchmark Suite
Class:  Bitmap
Author: Scott Beamer

Parallel bitmap that is thread-safe
 - Can set bits in parallel (set_bit_atomic) unlike std::vector<bool>
*/


class Bitmap {
 public:
  explicit Bitmap(size_t size) {
    uint64_t num_words = (size + kBitsPerWord - 1) / kBitsPerWord;
#ifdef SIM
    start_ = (uint64_t *)aligned_alloc(PAGE_SIZE, num_words*sizeof(uint64_t));
#else
    start_ = new uint64_t[num_words];
#endif
    end_ = start_ + num_words;
  }

  ~Bitmap() {
    delete[] start_;
  }

  void reset() {
    //std::fill(start_, end_, 0);
	uint64_t *ptr = start_;
	for(; ptr != end_; ++ ptr) *ptr = 0;
  }

  void set_bit(size_t pos) {
    start_[word_offset(pos)] |= ((uint64_t) 1l << bit_offset(pos));
  }

  void set_bit_atomic(size_t pos) {
    uint64_t old_val, new_val;
    do {
      old_val = start_[word_offset(pos)];
      new_val = old_val | ((uint64_t) 1l << bit_offset(pos));
    } while (!compare_and_swap(start_[word_offset(pos)], old_val, new_val));
  }

  bool get_bit(size_t pos) const {
    return (start_[word_offset(pos)] >> bit_offset(pos)) & 1l;
  }

  void swap(Bitmap &other) {
    //std::swap(start_, other.start_);
    //std::swap(end_, other.end_);
	uint64_t *temp = start_;
	start_ = other.start_;
	other.start_ = temp;
	temp = end_;
	end_ = other.end_;
	other.end_ = temp;
  }

  uint64_t *start_;
  uint64_t *end_;

 private:
  static const uint64_t kBitsPerWord = 64;
  static uint64_t word_offset(size_t n) { return n / kBitsPerWord; }
  static uint64_t bit_offset(size_t n) { return n & (kBitsPerWord - 1); }
};

#endif  // BITMAP_H_
