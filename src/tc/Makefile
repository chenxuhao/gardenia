include ../common.mk
all: tc_omp_base tc_gpu_base tc_omp_target tc_ocl_base

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

tc_gpu_base: gpu_base.o $(OBJS) 
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

tc_ocl_base: ocl_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -L$(OCL_DIR) $(OBJS) ocl_base.o -o $@ -lOpenCL
	mv $@ $(BIN)

tc_omp_target: omp_target.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) omp_target.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
