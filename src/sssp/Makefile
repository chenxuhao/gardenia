include ../common.mk
INCLUDES+=-I $(CUB_DIR)
#ICPCFLAGS+=-std=c++98
EXE=sssp
SRC=main.cc
#OBJS= topo_base.o topo_lb.o linear_base.o linear_lb.o main.o verifier.o omp_base.o
OBJS= linear_lb.o main.o verifier.o omp_base.o

#all: $(OBJS) $(EXE)_linear_base $(EXE)_linear_lb $(EXE)_topo_base $(EXE)_topo_lb $(EXE)_omp_base
all: $(OBJS) $(EXE)_linear_lb $(EXE)_omp_base

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

sssp_ocl_base: ocl_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o ocl_base.o verifier.o -o $@ -lOpenCL
	mv $@ $(BIN)

$(EXE)_omp_base: omp_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_base.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

$(EXE)_omp_base_sim: omp_base.o main.o verifier.o
	$(CXX) $(SIMFLAGS) $(INCLUDES) main.o omp_base.o verifier.o $(M5OP) -o $@ -lgomp
	mv $@ $(BIN)

$(EXE)_omp_target: main.o omp_target.o verifier.o
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) $(EXTRA) main.o omp_target.o verifier.o -o $@
	mv $@ $(BIN)

$(EXE)_linear_base: main.o linear_base.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o linear_base.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

$(EXE)_linear_lb: main.o linear_lb.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o linear_lb.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

$(EXE)_dstep: main.o dstep.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o dstep.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

$(EXE)_topo_base: main.o topo_base.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o topo_base.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

$(EXE)_topo_lb: main.o topo_lb.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o topo_lb.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o

