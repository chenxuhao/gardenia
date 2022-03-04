include ../common.mk
all: clustering_omp_base

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

clustering_gpu_base: gpu_base.o $(OBJS) 
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

clustering_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

clustering_ocl_base: ocl_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ocl_base.o $@ -lOpenCL
	mv $@ $(BIN)

clustering_omp_target: omp_target.o $(OBJS)
	$(ICPC) $(CXXFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) omp_target.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
