// Copyright 2020
// Authors: Xuhao Chen <cxh@mit.edu>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include "/home/cxh/gem5/include/gem5/m5ops.h"

int main(int argc, char *argv[]) {
  int n = 1024 * 1024;
  if(argc == 2) n = atoi(argv[1]);
  printf("Vector Addition num = %d\n", n);
  std::vector<float> a(n, 1), b(n, 1), c(n, 0);
  printf("Begin of ROI\n");
  m5_checkpoint(0,0);
  m5_reset_stats(0,0);
  for(int i = 0; i < n; i ++)
    c[i] = a[i] + b[i];
  m5_dump_stats(0,0);
  printf("End of ROI\n");
  fflush(NULL);
  return 0;
}
