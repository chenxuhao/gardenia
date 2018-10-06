#ifndef SIM_H_
#define SIM_H_

#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "common.h"
#include "bitmap.h"
#include "/home/cxh/gem5-ics/util/m5/m5op.h"

void *m5_mem = NULL;
static void map_m5_mem() {
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

int set_hub(int m, int nnz, int *degree, Bitmap &hub) {
	int num_hubs = 0;
	int threshold = hub_factor * nnz / m;
	printf("Labelling hub vertices, hub_factor = %f, threshold_degree = %d\n", hub_factor, threshold);
	for (int i = 0; i < m; i ++) {
		if(degree[i] > threshold) {
			hub.set_bit(i);
			num_hubs ++;
		}
	}
	return num_hubs;
}
#endif
