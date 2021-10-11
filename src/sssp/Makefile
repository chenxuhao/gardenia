include ../common.mk
INCLUDES+=-I $(CUB_DIR)
EXE=sssp
OBJS=main.o verifier.o
all: $(EXE)_linear_base $(EXE)_linear_lb $(EXE)_omp_base

%.o: %.cpp
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) -c $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

sssp_ocl_base: ocl_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ocl_base.o -o $@ -lOpenCL
	mv $@ $(BIN)

$(EXE)_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) omp_base.o -o $@ -lgomp
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

