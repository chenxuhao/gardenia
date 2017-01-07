# See LICENSE.txt for license details.
include src/common.mk
KERNELS = bc bfs cc color mst pr sgd sssp tc
SUITE = $(KERNELS) nvGRAPH

.PHONY: all
all:
	mkdir -p bin
	cd src/bc; make
	cd src/bfs; make
	cd src/cc; make
	cd src/color; make
	cd src/mst; make
	cd src/pr; make
	cd src/sgd; make
	cd src/sssp; make
	cd src/tc; make

.PHONY: clean
clean:
	rm src/*/*.o
