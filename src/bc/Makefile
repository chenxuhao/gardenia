include ../common.mk
INCLUDES+=-I $(CUB_DIR)
CXXFLAGS+=-std=c++11
SRC=main.cc
OBJS=omp_base.o linear_base.o linear_lb.o topo.o main.o verifier.o
all: $(OBJS) bc_linear_base bc_linear_lb bc_omp bc_topo

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(PARFLAG) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

bc_omp: omp_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_base.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

bc_linear_base: linear_base.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o linear_base.o verifier.o -o $@ $(LIBS)
	mv $@  $(BIN)

bc_linear_lb: linear_lb.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o linear_lb.o verifier.o -o $@ $(LIBS)
	mv $@  $(BIN)

bc_topo: topo.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o topo.o verifier.o -o $@ $(LIBS)
	mv $@  $(BIN)

bc_lb:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
