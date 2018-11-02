# See LICENSE.txt for license details.
include src/common.mk
KERNELS = bc bfs cc pr sgd spmv sssp vc symgs tc
SUITE = $(KERNELS)

.PHONY: all
all: bin_dir $(SUITE)

bin_dir:
	mkdir -p bin

% : src/%/Makefile
	cd src/$@; make; cd ../..

# Testing
include test/test.mk

.PHONY: clean
clean:
	rm src/*/*.o test/out/*
