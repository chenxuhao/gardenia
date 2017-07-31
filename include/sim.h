#ifndef SIM_H_
#define SIM_H_

#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
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

#endif
