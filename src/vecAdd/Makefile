include ../common.mk
#INCLUDES+=-I./cub-1.1.1
LIBS=-L$(CUDA_HOME)/lib64
EXTRA=-cudart shared
SRC=main.cc

all: vecadd_simd

vecadd_gpu:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -o vecAdd $(LIBS)
	mv vecAdd $(BIN)

vecadd_simd:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o vecAdd -lgomp -fopenmp

clean:
	rm vecAdd

