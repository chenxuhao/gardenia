include ../common.mk
CXXFLAGS+=-std=c++17
#CXXFLAGS+=-DUSE_NUMA -lnuma -ltbb
OBJS=main.o
all: $(OBJS) motif_omp_base motif_omp_formula

%.o: %.cpp
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) -c $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

motif_base: base.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) base.o -o $@ $(LIBS)
	mv $@ $(BIN)

motif_warp: warp.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) warp.o -o $@ $(LIBS)
	mv $@ $(BIN)

motif_omp_base: omp_base.o VertexSet.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) omp_base.o VertexSet.o -o $@ -lgomp
	mv $@ $(BIN)

motif_omp_formula: omp_formula.o VertexSet.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) omp_formula.o VertexSet.o -o $@ -lgomp
	mv $@ $(BIN)

motif_ocl_base: ocl_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) ocl_base.o -o $@ -lOpenCL
	mv $@ $(BIN)

motif_omp_target: omp_target.o $(OBJS)
	$(ICPC) $(CXXFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) omp_target.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
