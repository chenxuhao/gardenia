include ../common.mk
INCLUDES+=-I $(CUB_DIR)
SRC=main.cc

all: tc_base tc_omp

omp_base.o: omp_base.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.cc -fopenmp -c -o $@

cu_base.o: cu_base.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) cu_base.cu -c -o $@

tc_base: cu_base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) cu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_lb:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -o $@ $(LIBS)
	mv $@ $(BIN)

tc_omp: omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
