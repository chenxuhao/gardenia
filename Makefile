# See LICENSE.txt for license details.

include src/common.mk
CXX_FLAGS += -std=c++11 -O3 -Wall
PAR_FLAG = -fopenmp

ifneq (,$(findstring icpc,$(CXX)))
	PAR_FLAG = -openmp
endif

ifneq (,$(findstring sunCC,$(CXX)))
	CXX_FLAGS = -std=c++11 -xO3 -m64 -xtarget=native
	PAR_FLAG = -xopenmp
endif

ifneq ($(SERIAL), 1)
	CXX_FLAGS += $(PAR_FLAG)
endif

KERNELS = bc bfs cc color mst pr sgd sssp tc
SUITE = $(KERNELS)

.PHONY: all
all: $(SUITE)

% : src/%/*.cu src/%/*.h
	cd src/$@
	make
	#$(CXX) $(CXX_FLAGS) $< -o $@

.PHONY: clean
clean:
	rm -f $(SUITE) test/out/*
