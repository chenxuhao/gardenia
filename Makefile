# See LICENSE.txt for license details.
ANALYTICS = bc bfs cc pr sgd spmv sssp vc symgs
MINING = tc kcl_dfs sgl motif_dfs
KERNELS = $(ANALYTICS)
KERNELS = $(MINING)
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
