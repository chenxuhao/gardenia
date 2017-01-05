include ../common.mk
SRC=main.cc
all: bc_base bc_omp

$(BIN)/omp_base.o: omp_base.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.cc -fopenmp -c -o $@

$(BIN)/cu_base.o: cu_base.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) cu_base.cu -c -o $@

bc_base: $(BIN)/cu_base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=0 $(BIN)/cu_base.o -o $@ $(LIBS)
	mv bc_base  $(BIN)

bc_lb:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=1 -o $@ $(LIBS)
	mv bc_lb $(BIN)

bc_omp: $(BIN)/omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -DVARIANT=2 $(BIN)/omp_base.o -o $@ -lgomp
	mv bc_omp $(BIN)

clean:
	rm $(BIN)/*.o $(BIN)/bc_base $(BIN)/bc_omp
