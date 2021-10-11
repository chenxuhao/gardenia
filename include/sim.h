#pragma once

#include <string>
#include <iostream>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "timer.h"
#include "common.h"
#include "gem5/m5ops.h"

/*
void *m5_mem = NULL;
inline void map_m5_mem() {
#ifdef M5OP_ADDR
	int fd;
	fd = open("/dev/mem", O_RDWR | O_SYNC);
	if (fd == -1) {
		perror("Can't open /dev/mem");
		exit(1);
	}
	m5_mem = mmap(NULL, 0x10000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, M5OP_ADDR);
	if (!m5_mem) {
		perror("Can't mmap /dev/mem");
		exit(1);
	}
#endif
}
*/
inline void roi_begin() {
#ifdef SIM
  //omp_set_num_threads(1);
  //map_m5_mem();
  printf("Begin of ROI\n");
  m5_checkpoint(0,0);
  m5_reset_stats(0,0);
#endif
}

inline void roi_end() {
#ifdef SIM
  //m5_dump_reset_stats(0,0);
  m5_dump_stats(0,0);
  printf("End of ROI\n");
  fflush(NULL);
#endif
}

#ifdef SETHUB 
#include "bitmap.h"
float hub_factor = 0.1;
void set_addr_bounds(int, uint64_t, uint64_t, int) {}

inline int set_hub(Graph &g, Bitmap &hub) {
  auto m = g.V();
  auto nnz = g.E();
  int num_hubs = 0;
  uint32_t threshold = hub_factor * nnz / m;
  printf("hub_factor = %f, threshold_degree = %d\n", hub_factor, threshold);
  for (int i = 0; i < m; i ++) {
    if(g.get_degree(i) > threshold) {
      hub.set_bit(i);
      num_hubs ++;
    }
  }
  return num_hubs;
}

inline void roi_begin(Graph &g, int *labels = NULL, VertexId *frontier = NULL, int *weights = NULL) {
#ifdef SIM
  auto m = g.V();
  auto nnz = g.E();
  auto rowptr = g.out_rowptr();
  auto colidx = g.out_colidx();
  Bitmap hub(m);
  hub.reset();
  int num_hubs = set_hub(g, hub);
  m5_checkpoint(0,0);
  if (frontier) set_addr_bounds(0,(uint64_t)frontier,(uint64_t)&frontier[nnz],8);
  set_addr_bounds(1,(uint64_t)rowptr,(uint64_t)&rowptr[m+1],4);
  set_addr_bounds(2,(uint64_t)colidx,(uint64_t)&colidx[nnz],8);
  if (labels) set_addr_bounds(3,(uint64_t)labels,(uint64_t)&labels[m],8);
  if (weights) set_addr_bounds(5,(uint64_t)weights,(uint64_t)&weights[nnz],8);
  set_addr_bounds(6,(uint64_t)hub.start_,(uint64_t)hub.end_,8);
  printf("Begin of ROI\n");
  printf("This graph has %d hub vertices\n", num_hubs);
#endif
}
#endif
