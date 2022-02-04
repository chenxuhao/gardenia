include ../common.mk
#INCLUDES+=-I $(CUB_DIR)
ICPCFLAGS+=-std=c++98
CXXFLAGS+=-std=c++11
SIMFLAGS+=-std=c++11
SRC=main.cc
#OBJS=omp_base.o linear_base.o linear_lb.o topo_base.o topo_lb.o main.o verifier.o
OBJS=omp_base.o linear_lb.o main.o verifier.o

#all: $(OBJS) bc_linear_base bc_linear_lb bc_topo_base bc_topo_lb bc_omp_base
all: $(OBJS) bc_omp_base bc_linear_base bc_topo_base bc_linear_lb bc_topo_lb 

%.o: %.cpp
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) -c $<

ifeq ($(HOST),X86)
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<
else
%.o: %.cc
	$(CXX) $(SIMFLAGS) $(INCLUDES) -c $<
endif

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

bc_ocl_base: ocl_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o ocl_base.o verifier.o -o $@ -lOpenCL
	mv $@ $(BIN)

bc_omp_base: omp_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_base.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

bc_omp_target: main.o omp_target.o verifier.o
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) $(EXTRA) main.o omp_target.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

bc_omp_base_sim: omp_base.o main.o verifier.o
	$(CXX) $(SIMFLAGS) $(INCLUDES) main.o omp_base.o verifier.o $(M5OP) -o $@ -lgomp
	mv $@ $(BIN)

bc_linear_base: linear_base.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o linear_base.o verifier.o -o $@ $(LIBS)
	mv $@  $(BIN)

bc_linear_lb: linear_lb.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o linear_lb.o verifier.o -o $@ $(LIBS)
	mv $@  $(BIN)

bc_linear_pb: linear_pb.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o linear_pb.o verifier.o -o $@ $(LIBS)
	mv $@  $(BIN)

bc_topo_base: topo_base.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o topo_base.o verifier.o -o $@ $(LIBS)
	mv $@  $(BIN)

bc_topo_lb: topo_lb.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o topo_lb.o verifier.o -o $@ $(LIBS)
	mv $@  $(BIN)

bc_topo_pb: topo_pb.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o topo_pb.o verifier.o -o $@ $(LIBS)
	mv $@  $(BIN)

bc_hybrid_base: hybrid_base.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o hybrid_base.o verifier.o -o $@ $(LIBS)
	mv $@  $(BIN)

bc_hybrid_lb: hybrid_lb.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o hybrid_lb.o verifier.o -o $@ $(LIBS)
	mv $@  $(BIN)

bc_hybrid_tile: hybrid_tile.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o hybrid_tile.o verifier.o -o $@ $(LIBS)
	mv $@  $(BIN)

clean:
	rm *.o
