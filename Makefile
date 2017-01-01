# See LICENSE.txt for license details.
include src/common.mk
KERNELS = bc bfs cc color mst pr sgd sssp tc
SUITE = $(KERNELS) nvGRAPH

.PHONY: all
all:
	mkdir bin
	cd $(GARDINIA_ROOT)/src/bc; make
	cd $(GARDINIA_ROOT)/src/bfs; make
	cd $(GARDINIA_ROOT)/src/cc; make
	cd $(GARDINIA_ROOT)/src/color; make
	cd $(GARDINIA_ROOT)/src/mst; make
	cd $(GARDINIA_ROOT)/src/pr; make
	cd $(GARDINIA_ROOT)/src/sgd; make
	cd $(GARDINIA_ROOT)/src/sssp; make
	cd $(GARDINIA_ROOT)/src/tc; make

.PHONY: clean
clean:
	rm -rf bin/
