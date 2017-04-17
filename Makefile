# See LICENSE.txt for license details.
include src/common.mk
KERNELS = bc bfs cc mst pr sgd sssp tc vc
SUITE = $(KERNELS) nvGRAPH

.PHONY: all
all:
	mkdir -p bin
	cd src/bc; make
	cd src/bfs; make
	cd src/cc; make
	cd src/pr; make
	cd src/sgd; make
	cd src/spmv; make
	cd src/sssp; make
	cd src/symgs; make
	cd src/tc; make
	cd src/vc; make

.PHONY: clean
clean:
	rm src/*/*.o
