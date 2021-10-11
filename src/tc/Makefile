include ../common.mk
INCLUDES+=-I $(CUB_DIR)
OBJS=main.o verifier.o VertexSet.o
all: tc_omp_base

%.o: %.cpp
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) -c $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

tc_base: base.o $(OBJS) 
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) base.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_warp: warp.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) warp.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_bin: bin.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) bin.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

tc_ocl_base: ocl_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ocl_base.o $@ -lOpenCL
	mv $@ $(BIN)

tc_omp_target: omp_target.o $(OBJS)
	$(ICPC) $(CXXFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) omp_target.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
