include ../common.mk
SRC=main.cc
all: bc_base bc_omp

omp_base.o: omp_base.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.cc -fopenmp -c -o $@

base.o: base.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) base.cu -c -o $@

bc_base: base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=0 base.o -o $@ $(LIBS)
	mv $@  $(BIN)

bc_lb:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=1 -o $@ $(LIBS)
	mv $@ $(BIN)

bc_omp: omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -DVARIANT=2 omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
