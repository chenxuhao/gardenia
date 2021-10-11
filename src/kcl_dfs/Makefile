include ../common.mk
OBJS=main.o verifier.o
#CXXFLAGS+=-std=c++17 -Werror=return-type -Wshadow -Wno-reorder -DNDEBUG
#CXXFLAGS+=-DUSE_NUMA -lnuma -ltbb
CXXFLAGS+=-std=c++17
all: kcl_omp_base

%.o: %.cpp
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) -c $<

ifeq ($(HOST),X86)
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) -c $<
else
%.o: %.cc
	$(CXX) $(SIMFLAGS) $(INCLUDES) -c $<
endif

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

kcl_base: base.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(LIBS) $(EXTRA) base.o $(OBJS) -o $@
	mv $@ $(BIN)

kcl_warp: warp.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(LIBS) $(EXTRA) warp.o $(OBJS) -o $@
	mv $@ $(BIN)

kcl_tbb_base: tbb_base.o VertexSet.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) $(EXTRA) tbb_base.o VertexSet.o $(OBJS) -o $@
	mv $@ $(BIN)

kcl_omp_base: omp_base.o VertexSet.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) $(EXTRA) omp_base.o VertexSet.o $(OBJS) -o $@
	mv $@ $(BIN)

kcl_ocl_base: ocl_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) $(EXTRA) ocl_base.o $(OBJS) -o $@ -lOpenCL
	mv $@ $(BIN)

kcl_omp_target: omp_target.o $(OBJS)
	$(ICPC) $(CXXFLAGS) $(INCLUDES) $(EXTRA) omp_target.o $(OBJS) -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm -f *.o

.PHONY: clean

